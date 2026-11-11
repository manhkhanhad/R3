"""Simulated imperfect retriever: give each test row the k-th MOST SIMILAR training reasoning (other users).

Similarity = mxbai cosine to the test row's OWN (target-aware) reasoning. The own reasoning is only used to
choose; the model receives the neighbour's text. k = 1 is the best any retriever could do with this bank;
larger k simulates a worse retriever. Rows without own reasoning get none.

Writes <out>/test_nn<k>.json (format of rationale_text.build) and prints, per k, the mean cosine to the own
reasoning and how often the neighbour's next item equals the test target.
"""

import json
import os

import fire
import numpy as np
import pandas as pd
import torch

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
TRAIN_TEXT = R3 + "/output_dir/rationale_sft_runs/rationale_sft/gearec_v2/best/train.json"


def main(out, ks=(1, 10, 100, 1000), emb=f"{R3}/output_dir/reasoning_geometry/mxbai", chunk=512):
    ks = [int(k) for k in (ks if isinstance(ks, (list, tuple)) else [ks])]
    tr, te = torch.load(f"{emb}/train.pt"), torch.load(f"{emb}/test.pt")
    dtr, dte = pd.read_csv(f"{DATA}/train/{STEM}.csv"), pd.read_csv(f"{DATA}/test/{STEM}.csv")
    texts = json.load(open(TRAIN_TEXT))["rationales"]
    ok = np.where(tr["has_reasoning"].numpy())[0]
    assert all(texts[i] for i in ok)
    Rb = tr["reasoning"].float()[ok]
    uid = {u: k for k, u in enumerate(pd.unique(np.concatenate([dtr["user_id"], dte["user_id"]])))}
    ub = torch.tensor([uid[u] for u in dtr["user_id"].to_numpy()[ok]])
    uq = np.array([uid[u] for u in dte["user_id"]])
    tr_item = dtr["item_id"].to_numpy()[ok]
    te_item = dte["item_id"].to_numpy()
    has = te["has_reasoning"].numpy()
    q = np.where(has)[0]
    n = len(dte)
    chosen = {k: [""] * n for k in ks}
    cos = {k: [] for k in ks}
    same_target = {k: 0 for k in ks}
    kmax = max(ks)
    for s in range(0, len(q), chunk):
        rows = q[s:s + chunk]
        sim = te["reasoning"].float()[rows] @ Rb.T
        sim[torch.from_numpy(uq[rows])[:, None] == ub[None, :]] = -2
        vals, idx = sim.topk(kmax, dim=1)
        for r, i in enumerate(rows):
            for k in ks:
                j = int(idx[r, k - 1])
                chosen[k][i] = texts[ok[j]]
                cos[k].append(float(vals[r, k - 1]))
                same_target[k] += int(tr_item[j] == te_item[i])
    os.makedirs(out, exist_ok=True)
    for k in ks:
        path = f"{out}/test_nn{k}.json"
        json.dump({"split": "test", "kind": f"nearest-neighbour #{k} training reasoning (other users) by mxbai cosine "
                   "to the row's own GEARec v2 reasoning", "rationales": chosen[k],
                   "counts": {"rows": n, "with_reasoning": int(has.sum())}}, open(path, "w"))
        c = np.array(cos[k])
        print(f"k={k:5d}: mean cosine to own {c.mean():.3f} (p10 {np.percentile(c, 10):.3f}, p90 {np.percentile(c, 90):.3f}); "
              f"neighbour's next item == test target {same_target[k] / len(q):.1%} -> {path}")


if __name__ == "__main__":
    fire.Fire(main)
