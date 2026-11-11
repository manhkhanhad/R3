"""Gate A at fine granularity: ~5-10 training paragraphs per cluster, prototype = cluster medoid paragraph.

Variants (space used for clustering):
  full    : mxbai embedding of the full paragraph
  anchor  : normalize([target-part embedding ; next-item embedding])  (members point to similar next items)
ORACLE assignment for test rows: the cluster nearest to the row's own vector in the same space (uses the
target-aware paragraph / true next item, so this is a CEILING, not a method). The model gets the medoid text.

Writes <out>/test_proto_<variant>_K<k>.json (rationale_text format) + prints cluster-size stats and, per
variant, how often the medoid's next item equals / shares a category with the test target.
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
G = R3 + "/output_dir/reasoning_geometry"
TRAIN_TEXT = R3 + "/output_dir/rationale_sft_runs/rationale_sft/gearec_v2/best/train.json"
ITEM_EMB = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/outputs/item_emb_mxbai.pt"


def kmeans(X, k, iters=25, seed=0, chunk=8192):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for it in range(iters):
        assign = torch.cat([(X[s:s + chunk] @ C.T).argmax(1) for s in range(0, len(X), chunk)])
        newC = torch.zeros_like(C).index_add_(0, assign, X)
        cnt = torch.bincount(assign, minlength=k).float()
        empty = cnt == 0
        newC[~empty] /= cnt[~empty, None]
        # re-seed empty clusters with random points
        if empty.any():
            newC[empty] = X[torch.randint(len(X), (int(empty.sum()),), generator=g)]
        C = newC / newC.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    assign = torch.cat([(X[s:s + chunk] @ C.T).argmax(1) for s in range(0, len(X), chunk)])
    return C, assign


def main(out=f"{R3}/output_dir/rationale_sft_runs/rationale_sft/gearec_v2/proto_fine", configs=("full:7500", "anchor:5300", "anchor:10600")):
    configs = [c.split(":") for c in (configs if isinstance(configs, (list, tuple)) else [configs])]
    tr_full = torch.load(f"{G}/mxbai/train.pt"); te_full = torch.load(f"{G}/mxbai/test.pt")
    tr_p = torch.load(f"{G}/parts/train_parts.pt"); te_p = torch.load(f"{G}/parts/test_parts.pt")
    dtr, dte = pd.read_csv(f"{DATA}/train/{STEM}.csv"), pd.read_csv(f"{DATA}/test/{STEM}.csv")
    texts = json.load(open(TRAIN_TEXT))["rationales"]
    items = torch.load(ITEM_EMB).float(); items = items / items.norm(dim=-1, keepdim=True)
    need = set(dtr["item_asin"]) | set(dte["item_asin"])
    cat = {}
    for line in open(f"{DATA}/meta_Toys_and_Games.json"):
        m = json.loads(line)
        if m.get("asin") in need and m["asin"] not in cat:
            cat[m["asin"]] = " > ".join((m.get("category") or [])[1:]) or "unknown"
    ok = np.where(tr_full["has_reasoning"].numpy())[0]
    q = np.where(te_full["has_reasoning"].numpy())[0]
    tr_item, te_item = dtr["item_id"].to_numpy(), dte["item_id"].to_numpy()
    tr_cat = np.array([cat.get(a, "unknown") for a in dtr["item_asin"]])
    te_cat = np.array([cat.get(a, "unknown") for a in dte["item_asin"]])
    norm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def space(variant, part, full, item_ids, rows):
        if variant == "full":
            return norm(full["reasoning"].float()[rows])
        tp = part["target_part"].float()[rows]
        return norm(torch.cat([norm(tp), items[torch.as_tensor(item_ids[rows])]], -1))

    os.makedirs(out, exist_ok=True)
    for variant, k in configs:
        k = int(k)
        Xtr = space(variant, tr_p, tr_full, tr_item, ok)
        C, assign = kmeans(Xtr, k)
        sizes = torch.bincount(assign, minlength=k).numpy()
        # medoid = member closest to its centroid
        sim = (Xtr * C[assign]).sum(-1)
        medoid = np.full(k, -1)
        best = np.full(k, -2.0)
        for j, (a, s) in enumerate(zip(assign.numpy(), sim.numpy())):
            if s > best[a]:
                best[a], medoid[a] = s, ok[j]
        Xte = space(variant, te_p, te_full, te_item, q)
        te_assign = (Xte @ C.T).argmax(1).numpy()
        rationales = [""] * len(dte)
        same_item = same_cat = 0
        used = []
        for i, a in zip(q, te_assign):
            if medoid[a] < 0:
                continue
            m = medoid[a]
            rationales[i] = texts[m]
            same_item += tr_item[m] == te_item[i]
            same_cat += tr_cat[m] == te_cat[i] and te_cat[i] != "unknown"
            used.append(sizes[a])
        path = f"{out}/test_proto_{variant}_K{k}.json"
        json.dump({"split": "test", "kind": f"ORACLE fine-cluster medoid ({variant}, K={k})", "rationales": rationales,
                   "counts": {"rows": len(dte), "with_reasoning": int(sum(bool(r) for r in rationales))}}, open(path, "w"))
        nz = sizes[sizes > 0]
        print(f"{variant:6s} K={k:6d}: train cluster size median {int(np.median(nz))} (p10 {int(np.percentile(nz, 10))}, "
              f"p90 {int(np.percentile(nz, 90))}, empty {int((sizes == 0).sum())}); test: medoid next item == target "
              f"{same_item / len(q):.1%}, same category {same_cat / len(q):.1%}, assigned cluster size median {int(np.median(used))} -> {path}",
              flush=True)


if __name__ == "__main__":
    fire.Fire(main)
