"""Geometry of GEARec v2 reasoning (mxbai space): could a SIMILAR row's reasoning stand in for a row's own?

Inputs: <emb>/<split>.pt from embed.py; GEARec catalog item embeddings (outputs/item_emb_mxbai.pt, by item_id).
Queries: test rows with reasoning (their own, target-aware reasoning = the oracle we want to approximate).
Bank: TRAIN rows of OTHER users (train precedes test in time).

Reports
  A. reference cosines: random pairs; same user consecutive rows; same next item (different users);
     same target category (different item/user); different category
  B. for each test row, cosine between its own reasoning and: random bank reasoning; the reasoning of its
     k nearest bank rows by HISTORY similarity; the single closest bank reasoning (oracle ceiling)
  C. GEARec-style target rank (catalog items, history masked) when the query text is: own reasoning,
     history-retrieved reasoning (top-1, mean of top-5/20), oracle-closest bank reasoning, random reasoning,
     the history text itself
"""

import ast
import json

import fire
import numpy as np
import pandas as pd
import torch

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
ITEM_EMB = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/outputs/item_emb_mxbai.pt"


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return f"mean {x.mean():.3f}  sd {x.std():.3f}  p10 {np.percentile(x, 10):.3f}  p50 {np.median(x):.3f}  p90 {np.percentile(x, 90):.3f}  (n={len(x)})"


def main(emb=f"{R3}/output_dir/reasoning_geometry/mxbai", seed=0, chunk=512):
    rng = np.random.default_rng(seed)
    tr = torch.load(f"{emb}/train.pt"); te = torch.load(f"{emb}/test.pt")
    dtr = pd.read_csv(f"{DATA}/train/{STEM}.csv"); dte = pd.read_csv(f"{DATA}/test/{STEM}.csv")
    Rtr, Htr, has_tr = tr["reasoning"].float(), tr["history"].float(), tr["has_reasoning"].numpy()
    Rte, Hte, has_te = te["reasoning"].float(), te["history"].float(), te["has_reasoning"].numpy()

    # categories of target items
    need = set(dtr["item_asin"]) | set(dte["item_asin"])
    cat = {}
    for line in open(f"{DATA}/meta_Toys_and_Games.json"):
        m = json.loads(line)
        if m.get("asin") in need and m["asin"] not in cat:
            cat[m["asin"]] = " > ".join((m.get("category") or [])[1:]) or "unknown"
    tr_item = dtr["item_id"].to_numpy(); tr_user = dtr["user_id"].to_numpy()
    tr_cat = np.array([cat.get(a, "unknown") for a in dtr["item_asin"]])
    ok = np.where(has_tr)[0]

    # ---------------- A. reference cosines on TRAIN reasoning
    print("A. Cosine between two rows' reasoning (train):")
    a, b = rng.choice(ok, 200000), rng.choice(ok, 200000)
    keep = (tr_user[a] != tr_user[b]) & (tr_item[a] != tr_item[b])
    cos = lambda X, i, j: (X[i] * X[j]).sum(-1).numpy()
    rand = cos(Rtr, a[keep], b[keep])
    print("   random pair (different user, different next item) ", stats(rand))
    print("   ... their HISTORY texts                            ", stats(cos(Htr, a[keep], b[keep])))
    same_cat = keep & (tr_cat[a] == tr_cat[b]) & (tr_cat[a] != "unknown")
    diff_cat = keep & (tr_cat[a] != tr_cat[b])
    print("   same next-item CATEGORY, different item & user     ", stats(cos(Rtr, a[same_cat], b[same_cat])))
    print("   different next-item category                       ", stats(cos(Rtr, a[diff_cat], b[diff_cat])))
    # same user, consecutive training rows (both with reasoning)
    i = np.arange(len(dtr) - 1)
    m = (tr_user[i] == tr_user[i + 1]) & has_tr[i] & has_tr[i + 1]
    print("   same user, consecutive rows (overlapping history)  ", stats(cos(Rtr, i[m], i[m] + 1)))
    # same next item, different users
    by_item = pd.Series(ok).groupby(tr_item[ok]).apply(list)
    pi, pj = [], []
    for rows in by_item:
        if len(rows) >= 2:
            for _ in range(3):
                x, y = rng.choice(rows, 2, replace=False)
                if tr_user[x] != tr_user[y]:
                    pi.append(x); pj.append(y)
    print("   SAME next item, different users                    ", stats(cos(Rtr, np.array(pi), np.array(pj))))
    # does history similarity predict reasoning similarity?
    hs, rs = cos(Htr, a[keep], b[keep]), rand
    print(f"   corr(history cosine, reasoning cosine) over random pairs: Pearson {np.corrcoef(hs, rs)[0,1]:.3f}")

    # ---------------- B/C. test queries vs train bank (other users)
    q = np.where(has_te)[0]
    te_user = dte["user_id"].to_numpy()
    items = torch.load(ITEM_EMB).float()
    items = items / items.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    hist_ids = [ast.literal_eval(h) for h in dte["history_item_id"]]
    target = dte["item_id"].to_numpy()
    bank = torch.from_numpy(ok)
    Rb, Hb = Rtr[bank], Htr[bank]
    uid = {u: k for k, u in enumerate(pd.unique(np.concatenate([tr_user, te_user])))}
    ub = torch.tensor([uid[u] for u in tr_user[ok]])
    uq = np.array([uid[u] for u in te_user])

    def rank(vecs, rows):
        v = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = v @ items.T
        out = []
        for r, i in enumerate(rows):
            excl = [h for h in hist_ids[i] if h != target[i]]
            if excl:
                S[r, excl] = float("-inf")
            out.append(int((S[r] > S[r, target[i]]).sum()) + 1)
        return out

    res = {k: [] for k in ("own", "hist_top1", "hist_mean5", "hist_mean20", "oracle_bank", "random_bank", "history_text")}
    cosB = {k: [] for k in ("random", "hist_top1", "hist_best_of5", "hist_best_of20", "oracle_bank")}
    for s in range(0, len(q), chunk):
        rows = q[s:s + chunk]
        R, H = Rte[rows], Hte[rows]
        same = torch.from_numpy(uq[rows])[:, None] == ub[None, :]
        hsim = H @ Hb.T; hsim[same] = -2
        rsim = R @ Rb.T; rsim[same] = -2
        top = hsim.topk(20, dim=1).indices
        own_cos_top = torch.gather(rsim, 1, top)
        cosB["hist_top1"] += own_cos_top[:, 0].tolist()
        cosB["hist_best_of5"] += own_cos_top[:, :5].max(1).values.tolist()
        cosB["hist_best_of20"] += own_cos_top.max(1).values.tolist()
        best = rsim.max(1)
        cosB["oracle_bank"] += best.values.tolist()
        rnd = torch.from_numpy(rng.integers(0, len(ok), len(rows)))
        cosB["random"] += rsim[torch.arange(len(rows)), rnd].tolist()
        res["own"] += rank(R.clone(), rows)
        res["hist_top1"] += rank(Rb[top[:, 0]].clone(), rows)
        res["hist_mean5"] += rank(Rb[top[:, :5]].mean(1), rows)
        res["hist_mean20"] += rank(Rb[top].mean(1), rows)
        res["oracle_bank"] += rank(Rb[best.indices].clone(), rows)
        res["random_bank"] += rank(Rb[rnd].clone(), rows)
        res["history_text"] += rank(H.clone(), rows)
        print(f"   ... {min(s + chunk, len(q))}/{len(q)} test queries", flush=True)

    print(f"\nB. Cosine between a TEST row's own reasoning and bank reasoning (train, other users; n={len(q)}):")
    for k, v in cosB.items():
        print(f"   {k:16s} {stats(v)}")
    print(f"\nC. Rank of the true next item among {items.shape[0]} catalog items when the query is ... (history masked)")
    print(f"   {'query':16s} {'median':>7s} {'HR@10':>7s} {'HR@20':>7s} {'HR@100':>7s}")
    for k, v in res.items():
        v = np.array(v)
        print(f"   {k:16s} {np.median(v):7.0f} {np.mean(v <= 10):7.1%} {np.mean(v <= 20):7.1%} {np.mean(v <= 100):7.1%}")


if __name__ == "__main__":
    fire.Fire(main)
