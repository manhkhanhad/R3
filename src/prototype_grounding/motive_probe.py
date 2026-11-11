"""Is the latent r "collaborative" in the user's sense?  (MWC-M property check, any R3 checkpoint)

On the first `limit` train rows (users are contiguous in the CSV, so within-user pairs exist):
  cross-user: for anchors with a same-item row of another user in the subset,
      mean cos(r_i, r_j) over same-item other-user rows  vs  over random other-user rows,
      and retrieval: among the top-k nearest other-user latents, share with the same next item
      (vs the chance share of same-item rows among other users).
  within-user: mean cos over a user's pairs with similar next items (title cos >= t or identical)
      vs pairs with dissimilar next items.
Cosines on raw r (the slot vector the decoder reads).
Usage: motive_probe.py --checkpoint DIR --tag NAME [--category Toys_and_Games] [--limit 20000] [--t 0.8]
Output: analysis/results/motive_probe/<tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--category", default="Toys_and_Games")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    X.configure_category(args.category)
    X.seed_all(42)
    model, tokenizer, device = X.load_model(Path(args.checkpoint))
    ds = X.PromptDataset("train", tokenizer, args.limit)
    frame = ds.frame
    users = frame["user_id"].astype(str).values
    items = frame["item_id"].astype(int).values
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: X.collate(b, tokenizer.pad_token_id))
    vecs = []
    with torch.no_grad():
        for batch in loader:
            ids, mask, pos = (batch[x].to(device) for x in ("input_ids", "attention_mask", "thought_pos"))
            vecs.append(X.thought_embeds(model, ids, mask, pos)[0].float().cpu())
    z = F.normalize(torch.cat(vecs), dim=-1)
    n = len(z)
    emb = F.normalize(torch.load(X.DATA_ROOT / "info" / f"{args.category}_qwen3emb_1536.pt", map_location="cpu",
                                 weights_only=False)["embeddings"].float(), dim=-1)
    uid = {u: i for i, u in enumerate(dict.fromkeys(users))}
    u = torch.tensor([uid[x] for x in users])
    it = torch.tensor(items)
    rng = np.random.default_rng(0)

    same_c, rand_c, prec, chance = [], [], [], []
    within_sim, within_dis = [], []
    for s in range(0, n, 2048):
        a = torch.arange(s, min(n, s + 2048))
        C = z[a] @ z.T
        other = u[a][:, None] != u[None, :]
        same = other & (it[a][:, None] == it[None, :])
        for r_ in range(len(a)):
            if same[r_].any():
                same_c.append(float(C[r_][same[r_]].mean()))
                o = torch.nonzero(other[r_]).flatten()
                pick = o[torch.from_numpy(rng.choice(len(o), min(256, len(o)), replace=False))]
                rand_c.append(float(C[r_][pick].mean()))
                cs = C[r_].masked_fill(~other[r_], -2)
                top = cs.topk(args.k).indices
                prec.append(float(same[r_][top].float().mean()))
                chance.append(float(same[r_].sum()) / float(other[r_].sum()))
            mine = torch.nonzero((u == u[a[r_]]) & (torch.arange(n) != a[r_])).flatten()
            if len(mine):
                sim = emb[it[mine]] @ emb[it[a[r_]]]
                similar = (sim >= args.t) | (it[mine] == it[a[r_]])
                if similar.any():
                    within_sim.append(float(C[r_][mine[similar]].mean()))
                if (~similar).any():
                    within_dis.append(float(C[r_][mine[~similar]].mean()))
    m = lambda v: float(np.mean(v)) if v else None
    out = {
        "checkpoint": args.checkpoint, "rows": n, "threshold": args.t, "k": args.k,
        "cross_user": {"anchors": len(same_c), "cos_same_item": m(same_c), "cos_random_other": m(rand_c),
                       "gap": m(np.array(same_c) - np.array(rand_c)) if same_c else None,
                       f"precision@{args.k}_same_item": m(prec), "chance_same_item": m(chance)},
        "within_user": {"anchors_similar": len(within_sim), "anchors_dissimilar": len(within_dis),
                        "cos_similar_items": m(within_sim), "cos_dissimilar_items": m(within_dis),
                        "gap": (m(within_sim) - m(within_dis)) if within_sim and within_dis else None},
        "global_mean_pairwise_cos": float((z[:2000] @ z[:2000].T).mean()),
    }
    print(json.dumps(out, indent=1), flush=True)
    X.atomic_json(X.ANALYSIS / "results" / "motive_probe" / f"{args.tag}.json", out)


if __name__ == "__main__":
    main()
