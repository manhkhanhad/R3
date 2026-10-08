#!/usr/bin/env python3
"""Aggregate parallel cluster-hypothesis paths (cluster_paths_decode.py) by marginalization.

    score(item) = (1 - lam) * sum_{c in top-K} w_c * p(item | x, c)  +  lam * p(item | x, R3)
    w_c ∝ p(c | x) ** alpha,   p(item | x, path) = softmax(beam scores / T) over the path's 10 beams

Hyper-parameters (predictor, K, alpha, lam, T) are chosen on the first ``--tune_rows``
test rows by HR@10 (then NDCG@10) and reported on the remaining rows, which were not
used for anything else.  Baselines on the same rows: the R3 path alone, the top-1
cluster path (plug-in), and two upper bounds: the best single cluster path chosen
with knowledge of the target, and whether any decoded path contains the target.

Usage: python cluster_paths_aggregate.py --paths <dir>/paths.json [--tune_rows 1000]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def softmax(x, t):
    x = np.asarray(x, float) / t
    e = np.exp(x - x.max())
    return e / e.sum()


def ranked_clusters(rec, predictor):
    pr = rec["probs"][predictor]
    return sorted(pr, key=lambda c: -pr[c])


def aggregate(rec, predictor, k, alpha, lam, t):
    """Top-10 titles of the marginalized distribution for one record."""
    scores = {}
    clusters = [c for c in ranked_clusters(rec, predictor)[:k] if c in rec["paths"]]
    if clusters and lam < 1:
        w = np.array([rec["probs"][predictor][c] for c in clusters]) ** alpha
        w = (1 - lam) * w / max(w.sum(), 1e-12)
        for wc, c in zip(w, clusters):
            path = rec["paths"][c]
            for title, pt in zip(path["titles"], softmax(path["scores"], t)):
                scores[title] = scores.get(title, 0.0) + wc * pt
    if lam > 0 or not clusters:
        path = rec["paths"]["r3"]
        for title, pt in zip(path["titles"], softmax(path["scores"], t)):
            scores[title] = scores.get(title, 0.0) + (lam if clusters else 1.0) * pt
    return sorted(scores, key=lambda x: -scores[x])[:10]


def metrics(lists, targets):
    ranks = np.array([l.index(t) + 1 if t in l else 10 ** 9 for l, t in zip(lists, targets)])
    return {"HR@10": float((ranks <= 10).mean()),
            "NDCG@10": float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()), "n": len(ranks)}


def evaluate(records, cfg):
    return metrics([aggregate(r, *cfg) for r in records], [r["target"] for r in records])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--tune_rows", type=int, default=1000)
    a = p.parse_args()
    records = json.load(a.paths.open())
    tune, test = records[:a.tune_rows], records[a.tune_rows:]
    targets = [r["target"] for r in test]
    predictors = list(records[0]["probs"])

    grid = list(itertools.product(predictors, (1, 2, 3, 5), (0.5, 1.0, 2.0), (0.0, 0.25, 0.5, 0.75), (0.5, 1.0, 2.0)))
    scored = [(evaluate(tune, cfg), cfg) for cfg in grid]
    best_res, best = max(scored, key=lambda x: (x[0]["HR@10"], x[0]["NDCG@10"]))
    best_per_pred = {pr: max([s for s in scored if s[1][0] == pr], key=lambda x: (x[0]["HR@10"], x[0]["NDCG@10"]))[1]
                     for pr in predictors}
    best_no_r3 = max([s for s in scored if s[1][3] == 0.0], key=lambda x: (x[0]["HR@10"], x[0]["NDCG@10"]))[1]

    r3_lists = [r["paths"]["r3"]["titles"] for r in test]
    r3_hit = np.array([t in l for t, l in zip(targets, r3_lists)])
    out = {"tune_rows": len(tune), "test_rows": len(test), "selected": dict(zip(("predictor", "K", "alpha", "lam", "T"), best)),
           "tune_score_of_selected": best_res, "results": {}}

    def add(name, lists):
        hit = np.array([t in l[:10] for t, l in zip(targets, lists)])
        res = metrics(lists, targets)
        res.update({"HR@10_on_r3_miss": float(hit[~r3_hit].mean()), "HR@10_on_r3_hit": float(hit[r3_hit].mean()),
                    "gained_vs_r3": int((hit & ~r3_hit).sum()), "lost_vs_r3": int((~hit & r3_hit).sum())})
        out["results"][name] = res

    add("r3_path", r3_lists)
    for pr in predictors:
        add(f"plugin_top1[{pr}]", [aggregate(r, pr, 1, 1.0, 0.0, 1.0) for r in test])
        add(f"marginal_best[{pr}] {best_per_pred[pr]}", [aggregate(r, *best_per_pred[pr]) for r in test])
    add(f"marginal_no_r3_path {best_no_r3}", [aggregate(r, *best_no_r3) for r in test])
    add(f"SELECTED {best}", [aggregate(r, *best) for r in test])
    # upper bounds: best single cluster path chosen with the target known; any decoded path contains it
    best_path = []
    for r, t in zip(test, targets):
        cand = [r["paths"][c]["titles"] for c in r["paths"] if c != "r3"]
        best_path.append(next((l for l in cand if t in l), cand[0] if cand else r["paths"]["r3"]["titles"]))
    add("upper_bound_best_cluster_path", best_path)
    any_hit = np.array([any(t in r["paths"][c]["titles"] for c in r["paths"]) for r, t in zip(test, targets)])
    out["any_path_contains_target"] = float(any_hit.mean())
    out["true_cluster_decoded"] = float(np.mean([str(r["target_cluster"]) in r["paths"] for r in test]))
    out["true_cluster_decoded_on_r3_miss"] = float(np.mean([str(r["target_cluster"]) in r["paths"]
                                                            for r, h in zip(test, r3_hit) if not h]))
    out["paths_per_row"] = float(np.mean([len(r["paths"]) - 1 for r in records]))

    lines = [f"Tuned on {len(tune)} test rows, reported on {len(test)} disjoint test rows. "
             f"{out['paths_per_row']:.2f} cluster paths per row; true cluster decoded on {out['true_cluster_decoded']:.3f} "
             f"of rows ({out['true_cluster_decoded_on_r3_miss']:.3f} of R3 misses); any path contains the target on "
             f"{out['any_path_contains_target']:.3f}.", "",
             "| method | HR@10 | NDCG@10 | HR@10 on R3 misses | HR@10 on R3 hits | gained / lost vs R3 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in out["results"].items():
        lines.append(f"| {name} | {r['HR@10']:.4f} | {r['NDCG@10']:.4f} | {r['HR@10_on_r3_miss']:.4f} | "
                     f"{r['HR@10_on_r3_hit']:.4f} | {r['gained_vs_r3']} / {r['lost_vs_r3']} |")
    text = "\n".join(lines)
    print(text, flush=True)
    (a.paths.parent / "aggregate.md").write_text(text + "\n")
    with (a.paths.parent / "aggregate.json").open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
