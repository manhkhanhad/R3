"""Paired bootstrap of HR@10 / NDCG@10 between eval JSONs on the same test rows.

Usage: paired_bootstrap.py <run_dir> <reference_mode> <mode> [<mode> ...]
Modes resolve to <run_dir>/eval_<mode>.json, else to an experiment.py intervention
file <run_dir>/generation_*_<mode>_t*.json (rows matched by sample_id).
Matching mirrors src/utils/ranking_metrics.py (strip ' "\\n', exact match, last index).
"""
import glob
import json
import math
import os
import sys

import numpy as np


def resolve(run_dir, mode):
    """eval_<mode>.json (latent_attention_eval) or an experiment.py intervention
    file generation_<split>_..._<mode>_t<topk>_....json in the same directory."""
    path = f"{run_dir}/eval_{mode}.json"
    if os.path.exists(path):
        return path
    matches = sorted(glob.glob(f"{run_dir}/generation_*_{mode}_t*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{mode}: expected one eval/generation JSON in {run_dir}, found {matches}")
    return matches[0]


def per_row(path):
    rows = json.load(open(path))
    if isinstance(rows, dict) and "records" in rows:  # experiment.py generate output
        records = sorted(rows["records"], key=lambda r: r["sample_id"])
        ranks = np.array([r["rank"] for r in records], dtype=float)
        hit = ranks <= 10
        ndcg = np.where(hit, 1.0 / np.log2(np.minimum(ranks, 10) + 1), 0.0)
        return hit.astype(float), ndcg, [str(r["target"]) for r in records]
    hr, ndcg = [], []
    for row in rows:
        target = str(row["output"]).strip(' "\n')
        rank = None
        for i, p in enumerate(row["predict"]):
            if str(p).strip(' "\n') == target:
                rank = i
        hit = rank is not None and rank < 10
        hr.append(1.0 if hit else 0.0)
        ndcg.append(math.log(2) / math.log(rank + 2) if hit else 0.0)
    return np.array(hr), np.array(ndcg), [str(r["output"]) for r in rows]


run_dir, ref = sys.argv[1], sys.argv[2]
ref_hr, ref_ndcg, ref_targets = per_row(resolve(run_dir, ref))
rng = np.random.default_rng(42)
idx = rng.integers(0, len(ref_hr), size=(2000, len(ref_hr)))
print(f"reference {ref}: HR@10 {ref_hr.mean():.4f} NDCG@10 {ref_ndcg.mean():.4f} (n={len(ref_hr)})")
for mode in sys.argv[3:]:
    hr, ndcg, targets = per_row(resolve(run_dir, mode))
    assert targets == ref_targets, f"{mode}: rows not aligned"
    for name, a, b in (("HR@10", ref_hr, hr), ("NDCG@10", ref_ndcg, ndcg)):
        d = a - b
        boot = d[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = "yes" if lo > 0 or hi < 0 else "no"
        print(f"  {ref} - {mode:16s} {name:8s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  significant: {sig}")
