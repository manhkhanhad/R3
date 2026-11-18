"""Effect-defined positives: which other users' latents are good for an anchor, and what they share.

For `n_anchors` train rows i and `n_random` random other-user rows j each, swap r_j into i's slot and
score i's own target with the decoder:  u_ij = NLL(y_i | x_i, r_i) - NLL(y_i | x_i, r_j).
A donor is "good" when u_ij > -tau.  Reports the share of anchors with >= 1 good donor among the
random ones, how many, and what good donors share with the anchor compared with bad ones:
same target, same target brand (first title word), donor target title similarity to the anchor's
target (Qwen3 title embeddings), history-item Jaccard, latent cosine r_i.r_j, and donor target in
the anchor's history.  Also good rate by title-similarity bin.

Usage: bank_random_donors.py --checkpoint DIR --category Toys_and_Games --tag NAME
         [--limit 10000] [--n_anchors 400] [--n_random 64] [--tau 0.1] [--batch_size 8]
Output: analysis/results/bank_quality/<tag>_random_donors.json
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--category", default="Toys_and_Games")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--n_anchors", type=int, default=400)
    ap.add_argument("--n_random", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X.configure_category(args.category)
    X.seed_all(args.seed)
    model, tokenizer, device = X.load_model(Path(args.checkpoint))
    ds = X.PromptDataset("train", tokenizer, args.limit)
    frame = ds.frame
    users = frame["user_id"].astype(str).values
    targets = frame["item_id"].astype(int).values
    hist_ids = [set(ast.literal_eval(h)) if isinstance(h, str) else set(h) for h in frame["history_item_id"]]
    brand = frame["item_title"].astype(str).str.strip(' "').str.split(" ").str[0].str.lower().values
    n = len(ds)
    pad = tokenizer.pad_token_id
    item_emb = torch.load(X.DATA_ROOT / "info" / f"{args.category}_qwen3emb_1536.pt", map_location="cpu",
                          weights_only=False)["embeddings"].float()
    item_emb = F.normalize(item_emb, dim=-1)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: X.collate(b, pad))
    bank = torch.zeros(n, model.config.hidden_size)
    k = 0
    for batch in loader:
        ids, mask, pos = (batch[x].to(device) for x in ("input_ids", "attention_mask", "thought_pos"))
        thought, _ = X.thought_embeds(model, ids, mask, pos)
        bank[k:k + ids.size(0)] = thought.cpu()
        k += ids.size(0)
    zb = F.normalize(bank, dim=-1)
    print(f"bank {n} rows", flush=True)

    rng = np.random.default_rng(args.seed)
    anchors = np.sort(rng.choice(n, size=min(args.n_anchors, n), replace=False))
    rows = []  # one record per (anchor, donor)
    per_anchor = []
    for s0 in range(0, len(anchors), args.batch_size):
        idx = anchors[s0:s0 + args.batch_size]
        batch = X.collate([ds[int(a)] for a in idx], pad)
        ids, mask, labels, pos = (batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos"))
        own_vec, embeds = X.thought_embeds(model, ids, mask, pos)
        own = X.per_example_nll(model, embeds, mask, labels, ids).cpu()
        donors = []
        for a in idx:
            cand = np.nonzero(users != users[a])[0]
            donors.append(rng.choice(cand, size=args.n_random, replace=False))
        donors = np.stack(donors)  # (b, n_random)
        u = torch.zeros(len(idx), args.n_random)
        for kk in range(args.n_random):
            vec = bank[torch.tensor(donors[:, kk])].to(device)
            nll = X.per_example_nll(model, X.substitute(embeds, pos, vec), mask, labels, ids).cpu()
            u[:, kk] = own - nll
        for b, a in enumerate(idx):
            good_count = int((u[b] > -args.tau).sum())
            per_anchor.append({"anchor": int(a), "own_nll": float(own[b]), "good_count": good_count,
                               "best_u": float(u[b].max()), "mean_u": float(u[b].mean())})
            for kk in range(args.n_random):
                j = int(donors[b, kk])
                rows.append({
                    "u": float(u[b, kk]), "good": bool(u[b, kk] > -args.tau),
                    "same_target": bool(targets[j] == targets[a]),
                    "same_brand": bool(brand[j] == brand[a]),
                    "title_cos": float(item_emb[targets[j]] @ item_emb[targets[a]]),
                    "hist_jaccard": len(hist_ids[a] & hist_ids[j]) / max(1, len(hist_ids[a] | hist_ids[j])),
                    "donor_target_in_anchor_history": bool(targets[j] in hist_ids[a]),
                    "anchor_target_in_donor_history": bool(targets[a] in hist_ids[j]),
                    "r_cos": float(zb[a] @ zb[j]),
                })
        if s0 % (args.batch_size * 10) == 0:
            print(f"  anchors {s0 + len(idx)}/{len(anchors)}", flush=True)

    df = pd.DataFrame(rows)
    pa = pd.DataFrame(per_anchor)
    good, bad = df[df.good], df[~df.good]
    feats = ["same_target", "same_brand", "title_cos", "hist_jaccard", "donor_target_in_anchor_history",
             "anchor_target_in_donor_history", "r_cos"]
    summary = {
        "n_anchors": int(len(pa)), "n_random": args.n_random, "tau": args.tau,
        "share_anchors_with_>=1_good_random_donor": float((pa.good_count >= 1).mean()),
        "share_anchors_with_>=3_good": float((pa.good_count >= 3).mean()),
        "mean_good_count_of_64": float(pa.good_count.mean()),
        "share_anchors_best_u>0": float((pa.best_u > 0).mean()),
        "donor_good_rate": float(df.good.mean()),
        "good_vs_bad_donor_features": {f: {"good": float(good[f].astype(float).mean()), "bad": float(bad[f].astype(float).mean())} for f in feats},
        "good_rate_by_title_cos_bin": {
            f"{lo:.1f}-{hi:.1f}": {"n": int(m.sum()), "good_rate": float(df.good[m].mean()) if m.sum() else None}
            for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))
            for m in [(df.title_cos >= lo) & (df.title_cos < hi)]
        },
        "good_rate_same_target": float(df.good[df.same_target].mean()) if df.same_target.any() else None,
        "good_rate_same_brand": float(df.good[df.same_brand].mean()) if df.same_brand.any() else None,
        "good_rate_hist_overlap>0": float(df.good[df.hist_jaccard > 0].mean()) if (df.hist_jaccard > 0).any() else None,
        "good_rate_r_cos_top_decile": float(df.good[df.r_cos >= df.r_cos.quantile(0.9)].mean()),
        "good_rate_r_cos_bottom_decile": float(df.good[df.r_cos <= df.r_cos.quantile(0.1)].mean()),
        "corr_u_title_cos": float(df.u.corr(df.title_cos)), "corr_u_r_cos": float(df.u.corr(df.r_cos)),
        "corr_u_hist_jaccard": float(df.u.corr(df.hist_jaccard)),
    }
    print(json.dumps(summary, indent=1), flush=True)
    out = X.ANALYSIS / "results" / "bank_quality" / f"{args.tag}_random_donors.json"
    X.atomic_json(out, {"checkpoint": args.checkpoint, "summary": summary, "per_anchor": per_anchor})
    df.to_csv(out.with_suffix(".pairs.csv"), index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
