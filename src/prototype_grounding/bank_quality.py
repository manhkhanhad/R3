"""Quality of a latent bank for collaborative InfoNCE: coverage, geometry and *usefulness*.

Bank = the first `limit` train rows' latents r (the same rows `experiment.py extract --split train
--limit` uses).  For a sample of anchor rows i we swap other rows' r_j into i's thought slot and
score i's own target with the decoder (teacher forcing, same path as the interventions):

    u_ij = NLL(y_i | x_i, r_i) - NLL(y_i | x_i, r_j)        (> 0: r_j is at least as good as own r)

Candidates j per anchor: same target / other user (the InfoNCE positives), same user / other
target (B's hard negatives), random other user (plain negatives), and the bank mean.  Reported
per relation: mean u, share of pairs with u > -tau ("good"), and best-of-k over the anchor's
same-target candidates (does *some* good positive exist?).

Usage:
  bank_quality.py --checkpoint DIR --category Toys_and_Games --tag NAME [--limit 10000]
                  [--n_anchors 800] [--k_pos 4] [--tau 0.1] [--batch_size 8]
Output: analysis/results/bank_quality/<tag>.json
"""

import argparse
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


def centred_cos(z, a, b):
    return float((z[a] * z[b]).sum(-1).mean()) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--category", default="Toys_and_Games")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--n_anchors", type=int, default=800)
    ap.add_argument("--k_pos", type=int, default=4)
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
    n = len(ds)
    pad = tokenizer.pad_token_id
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: X.collate(b, pad))

    # ---- bank latents (own r for every row)
    bank = torch.zeros(n, model.config.hidden_size)
    k = 0
    for batch in loader:
        ids, mask, pos = (batch[x].to(device) for x in ("input_ids", "attention_mask", "thought_pos"))
        thought, _ = X.thought_embeds(model, ids, mask, pos)
        bank[k:k + ids.size(0)] = thought.cpu()
        k += ids.size(0)
    print(f"bank: {n} rows, |r| {bank.norm(dim=-1).mean():.1f}", flush=True)

    # ---- coverage inside the bank
    t_series = pd.Series(targets)
    u_series = pd.Series(users)
    other_user_same_target = np.array([
        ((targets == t) & (users != u)).sum() for t, u in zip(targets, users)
    ])
    same_user_other_target = np.array([
        ((users == u) & (targets != t)).sum() for t, u in zip(targets, users)
    ])
    coverage = {
        "rows": int(n),
        "distinct_users": int(u_series.nunique()), "distinct_targets": int(t_series.nunique()),
        "anchors_with_same_target_other_user": float((other_user_same_target >= 1).mean()),
        "anchors_with_>=3_same_target_other_user": float((other_user_same_target >= 3).mean()),
        "anchors_with_same_user_other_target": float((same_user_other_target >= 1).mean()),
    }

    # ---- geometry (centred cosines)
    z = F.normalize(bank - bank.mean(0), dim=-1)
    zr = F.normalize(bank, dim=-1)
    g = torch.Generator().manual_seed(args.seed)
    i = torch.randint(n, (20000,), generator=g); j = torch.randint(n, (20000,), generator=g)
    rnd = (users[i.numpy()] != users[j.numpy()]) & (targets[i.numpy()] != targets[j.numpy()])
    pairs_st, pairs_su = [], []
    for a in range(n):
        st = np.nonzero((targets == targets[a]) & (users != users[a]))[0]
        su = np.nonzero((users == users[a]) & (targets != targets[a]))[0]
        if len(st): pairs_st.append((a, int(st[0])))
        if len(su): pairs_su.append((a, int(su[0])))
    st_a = torch.tensor([p[0] for p in pairs_st]); st_b = torch.tensor([p[1] for p in pairs_st])
    su_a = torch.tensor([p[0] for p in pairs_su]); su_b = torch.tensor([p[1] for p in pairs_su])
    s = torch.linalg.svdvals(zr); p = s ** 2 / (s ** 2).sum()
    geometry = {
        "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())),
        "random_pair_cos_centred": centred_cos(z, i[torch.tensor(rnd)], j[torch.tensor(rnd)]),
        "same_target_other_user_cos_centred": centred_cos(z, st_a, st_b),
        "same_user_other_target_cos_centred": centred_cos(z, su_a, su_b),
        "random_pair_cos_raw": float((zr[i] * zr[j]).sum(-1)[torch.tensor(rnd)].mean()),
        "same_target_other_user_cos_raw": float((zr[st_a] * zr[st_b]).sum(-1).mean()) if len(st_a) else float("nan"),
    }
    print("coverage", coverage, "\ngeometry", {k_: round(v, 4) for k_, v in geometry.items()}, flush=True)

    # ---- usefulness: swap-in utility on a sample of anchors that have same-target positives
    rng = np.random.default_rng(args.seed)
    eligible = np.nonzero(other_user_same_target >= 1)[0]
    anchors = np.sort(rng.choice(eligible, size=min(args.n_anchors, len(eligible)), replace=False))
    mean_r = bank.mean(0)
    rec = {"own": [], "pos": [], "pos_best": [], "hard_same_user": [], "random": [], "mean": [], "anchor": []}
    for s0 in range(0, len(anchors), args.batch_size):
        idx = anchors[s0:s0 + args.batch_size]
        batch = X.collate([ds[int(a)] for a in idx], pad)
        ids, mask, labels, pos = (batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos"))
        own_vec, embeds = X.thought_embeds(model, ids, mask, pos)
        own = X.per_example_nll(model, embeds, mask, labels, ids).cpu()
        # candidate sets
        pos_sets, hard_j, rand_j = [], [], []
        for a in idx:
            st = np.nonzero((targets == targets[a]) & (users != users[a]))[0]
            pos_sets.append(rng.choice(st, size=min(args.k_pos, len(st)), replace=False))
            su = np.nonzero((users == users[a]) & (targets != targets[a]))[0]
            hard_j.append(int(rng.choice(su)) if len(su) else -1)
            cand = np.nonzero((users != users[a]) & (targets != targets[a]))[0]
            rand_j.append(int(rng.choice(cand)))
        kmax = max(len(ps) for ps in pos_sets)
        pos_nll = torch.full((len(idx), kmax), float("nan"))
        for kk in range(kmax):
            vec = torch.stack([bank[ps[kk]] if kk < len(ps) else own_vec[b].cpu() for b, ps in enumerate(pos_sets)]).to(device)
            nll = X.per_example_nll(model, X.substitute(embeds, pos, vec), mask, labels, ids).cpu()
            for b, ps in enumerate(pos_sets):
                if kk < len(ps): pos_nll[b, kk] = nll[b]
        hv = torch.stack([bank[j_] if j_ >= 0 else own_vec[b].cpu() for b, j_ in enumerate(hard_j)]).to(device)
        hard = X.per_example_nll(model, X.substitute(embeds, pos, hv), mask, labels, ids).cpu()
        rv = torch.stack([bank[j_] for j_ in rand_j]).to(device)
        rand = X.per_example_nll(model, X.substitute(embeds, pos, rv), mask, labels, ids).cpu()
        mv = mean_r.expand(len(idx), -1).to(device)
        meannll = X.per_example_nll(model, X.substitute(embeds, pos, mv), mask, labels, ids).cpu()
        for b, a in enumerate(idx):
            u_pos = (own[b] - pos_nll[b]).numpy()
            u_pos = u_pos[~np.isnan(u_pos)]
            rec["anchor"].append(int(a)); rec["own"].append(float(own[b]))
            rec["pos"].append(float(u_pos.mean())); rec["pos_best"].append(float(u_pos.max()))
            rec["hard_same_user"].append(float(own[b] - hard[b]) if hard_j[b] >= 0 else float("nan"))
            rec["random"].append(float(own[b] - rand[b])); rec["mean"].append(float(own[b] - meannll[b]))
        if s0 % (args.batch_size * 20) == 0:
            print(f"  anchors {s0 + len(idx)}/{len(anchors)}", flush=True)

    def summ(v):
        v = np.array(v, float); v = v[~np.isnan(v)]
        return {"n": int(len(v)), "mean_u": float(v.mean()), "median_u": float(np.median(v)),
                "share_good(u>-tau)": float((v > -args.tau).mean()), "share_better(u>0)": float((v > 0).mean())}
    usefulness = {
        "own_nll_mean": float(np.mean(rec["own"])),
        "same_target_other_user (positives, mean over k)": summ(rec["pos"]),
        "same_target_other_user (best of k)": summ(rec["pos_best"]),
        "same_user_other_target (B hard negatives)": summ(rec["hard_same_user"]),
        "random_other_user": summ(rec["random"]),
        "bank_mean": summ(rec["mean"]),
        "k_pos": args.k_pos, "tau": args.tau,
    }
    print("usefulness", json.dumps(usefulness, indent=1), flush=True)
    out = X.ANALYSIS / "results" / "bank_quality" / f"{args.tag}.json"
    X.atomic_json(out, {"checkpoint": args.checkpoint, "category": args.category, "limit": args.limit,
                        "coverage": coverage, "geometry": geometry, "usefulness": usefulness,
                        "per_anchor": {k_: [None if (isinstance(x, float) and np.isnan(x)) else x for x in v] for k_, v in rec.items()}})
    print("wrote", out)


if __name__ == "__main__":
    main()
