"""Does a RETRIEVED reasoning add to the history? (no leakage: neighbours are other users' TRAIN rows)

Query = normalize(history) + w * normalize(mean of top-k history-neighbours' reasoning), ranked against
GEARec's mxbai catalog item embeddings (history items masked). Baseline = history alone (w = 0).
All test rows are used (rows without own reasoning are fine: only neighbours' reasoning is used).
"""

import ast

import fire
import numpy as np
import pandas as pd
import torch

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
ITEM_EMB = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/outputs/item_emb_mxbai.pt"


def main(emb=f"{R3}/output_dir/reasoning_geometry/mxbai", chunk=512):
    tr, te = torch.load(f"{emb}/train.pt"), torch.load(f"{emb}/test.pt")
    dtr, dte = pd.read_csv(f"{DATA}/train/{STEM}.csv"), pd.read_csv(f"{DATA}/test/{STEM}.csv")
    ok = np.where(tr["has_reasoning"].numpy())[0]
    Rb, Hb = tr["reasoning"].float()[ok], tr["history"].float()[ok]
    Hte = te["history"].float()
    uid = {u: k for k, u in enumerate(pd.unique(np.concatenate([dtr["user_id"], dte["user_id"]])))}
    ub = torch.tensor([uid[u] for u in dtr["user_id"].to_numpy()[ok]])
    uq = np.array([uid[u] for u in dte["user_id"]])
    items = torch.load(ITEM_EMB).float()
    items = items / items.norm(dim=-1, keepdim=True)
    hist_ids = [ast.literal_eval(h) for h in dte["history_item_id"]]
    target = dte["item_id"].to_numpy()
    norm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    configs = [("history alone (baseline)", 0, 0)] + [(f"history + {w} x mean top-{k} neighbours' reasoning", k, w)
                                                     for k in (1, 5, 20) for w in (0.5, 1.0)]
    configs += [("neighbours' reasoning alone, top-5", 5, None)]
    ranks = {c[0]: [] for c in configs}
    for s in range(0, len(dte), chunk):
        rows = np.arange(s, min(s + chunk, len(dte)))
        H = Hte[rows]
        hs = H @ Hb.T
        hs[torch.from_numpy(uq[rows])[:, None] == ub[None, :]] = -2
        top = hs.topk(20, dim=1).indices
        for name, k, w in configs:
            if k == 0:
                q = norm(H)
            elif w is None:
                q = norm(Rb[top[:, :k]].mean(1))
            else:
                q = norm(norm(H) + w * norm(Rb[top[:, :k]].mean(1)))
            S = q @ items.T
            for r, i in enumerate(rows):
                excl = [h for h in hist_ids[i] if h != target[i]]
                if excl:
                    S[r, excl] = float("-inf")
                ranks[name].append(int((S[r] > S[r, target[i]]).sum()) + 1)
    base = np.array(ranks["history alone (baseline)"])
    print(f"Test rows {len(dte)}; rank of the true next item among {items.shape[0]} items (mxbai space)")
    print(f"{'query':52s} {'median':>7s} {'HR@10':>7s} {'HR@20':>7s} {'HR@100':>7s} {'better/worse rank vs baseline':>30s}")
    for name, v in ranks.items():
        v = np.array(v)
        print(f"{name:52s} {np.median(v):7.0f} {np.mean(v <= 10):7.1%} {np.mean(v <= 20):7.1%} {np.mean(v <= 100):7.1%} "
              f"{np.mean(v < base):14.1%} / {np.mean(v > base):.1%}")


if __name__ == "__main__":
    fire.Fire(main)
