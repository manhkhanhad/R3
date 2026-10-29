"""Report for scripts/history_subset_probe.py: HR/NDCG @1/3/5/10 per variant, on
all test rows and on the rows that lost items (history longer than k), plus
paired bootstraps (2,000 resamples, seed 42) of each variant against the full
history and of oracle / antioracle / random against recent at the same k.

Matching mirrors scripts/paired_bootstrap_eval.py (strip ' "\\n', exact match,
last index).  Usage: history_subset_probe_report.py <probe_dir> [<out.md>]
"""
import ast
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CUTOFFS = (1, 3, 5, 10)
VARIANTS = ["full"] + [f"{r}_k{k}" for k in (1, 3) for r in ("recent", "random", "oracle", "antioracle")]


def ranks(path, targets=None):
    rows = json.load(open(path))
    if targets is not None and [r["output"] for r in rows] != targets:
        raise ValueError(f"{path}: rows not aligned with the full-history eval")
    out = []
    for row in rows:
        target = str(row["output"]).strip(' "\n')
        rank = None
        for i, p in enumerate(row["predict"]):
            if str(p).strip(' "\n') == target:
                rank = i
        out.append(rank if rank is not None and rank < 10 else 99)
    return np.array(out)


def metrics(rank):
    m = {}
    for c in CUTOFFS:
        hit = rank < c
        m[f"HR@{c}"] = hit.mean()
        if c > 1:
            m[f"NDCG@{c}"] = np.where(hit, math.log(2) / np.log(rank + 2), 0.0).mean()
    return m


def boot(a, b, rng_idx):
    d = a - b
    lo, hi = np.percentile(d[rng_idx].mean(1), [2.5, 97.5])
    return f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    probe_dir = sys.argv[1]
    full_csv = os.path.join(probe_dir, "full_inputs.csv")
    lengths = None
    if os.path.exists(full_csv):
        lengths = pd.read_csv(full_csv, index_col=0)["history_item_id"].map(lambda x: len(ast.literal_eval(x))).to_numpy()
    targets = [r["output"] for r in json.load(open(os.path.join(probe_dir, "eval_full.json")))]
    R = {v: ranks(os.path.join(probe_dir, f"eval_{v}.json"), targets) for v in VARIANTS
         if os.path.exists(os.path.join(probe_dir, f"eval_{v}.json"))}
    n = len(R["full"])
    lines = [f"# History-subset probe: {probe_dir}", "", f"n = {n}; NDCG@1 = HR@1; @20 = @10 with 10 beams.", ""]

    def table(title, mask):
        lines.append(f"## {title} (n = {int(mask.sum())})")
        lines.append("")
        cols = [f"HR@{c}" for c in CUTOFFS] + [f"NDCG@{c}" for c in CUTOFFS if c > 1]
        lines.append("| variant | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---:|" * len(cols))
        for v, r in R.items():
            m = metrics(r[mask])
            lines.append(f"| {v} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
        lines.append("")

    def comparisons(title, mask):
        idx = np.random.default_rng(42).integers(0, int(mask.sum()), size=(2000, int(mask.sum())))
        lines.append(f"### Paired bootstrap, {title}: Δ HR@10 / Δ NDCG@10 [95% CI] (* significant)")
        lines.append("")
        hit = {v: (r[mask] < 10).astype(float) for v, r in R.items()}
        nd = {v: np.where(r[mask] < 10, math.log(2) / np.log(r[mask] + 2), 0.0) for v, r in R.items()}
        pairs = [(v, "full") for v in R if v != "full"]
        pairs += [(f"{x}_k{k}", f"recent_k{k}") for k in (1, 3) for x in ("oracle", "antioracle", "random")]
        for a, b in pairs:
            if a in R and b in R:
                lines.append(f"- {a} − {b}: {boot(hit[a], hit[b], idx)} / {boot(nd[a], nd[b], idx)}")
        lines.append("")

    table("All test rows", np.ones(n, bool))
    comparisons("all rows", np.ones(n, bool))
    if lengths is not None:
        for k in (1, 3):
            mask = lengths > k
            table(f"Rows with more than {k} history items (the only rows k = {k} changes)", mask)
            comparisons(f"history > {k}", mask)
    text = "\n".join(lines)
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text + "\n")


if __name__ == "__main__":
    main()
