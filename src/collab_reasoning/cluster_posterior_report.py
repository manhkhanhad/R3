#!/usr/bin/env python3
"""E2-diag report: is the decoder posterior q(c|x,y) a better selector target than the cluster label?

Reads cluster_posterior.py output for the test split and the decoded cluster paths
(cluster_paths_decode.py, same first rows) and reports, on all rows and on R3 v1 misses:

1. Selection by q among the decoded paths ∪ ∅ (R3's own list), the ceiling for a selector
   trained on q, versus selection by the label (true cluster's path when decoded, else R3),
   the best-path oracle, and R3 alone.
2. Mass on ∅ on R3 hits vs misses (is ∅ a usable learned defer target?).
3. How often argmax q is the label cluster, and how often its path converts (hits).
4. Learnability proxy: rank of argmax q vs of the label cluster under the mix predictor p(c|x).

Usage: python cluster_posterior_report.py --posterior .../test.pt --paths .../paths.json --out .../report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    post = torch.load(a.posterior, weights_only=False)
    recs = json.load(a.paths.open())
    n = min(len(recs), len(post["candidates"]))
    mix = post["predictors"]["mix"]

    def hit(rec, key):
        return rec["target"].strip(' \n"') in rec["paths"][key]["titles"]

    rows = []
    for i in range(n):
        rec = recs[i]
        cands, lp = post["candidates"][i], post["logp"][i]
        score = dict(zip(cands, lp))
        decoded = [int(c) for c in rec["paths"] if c != "r3"]
        r3_hit = hit(rec, "r3")
        # q over all candidates + ∅ (uniform prior)
        q_all = softmax(list(lp) + [post["null_logp"][i]])
        q_null = float(q_all[-1])
        top = int(cands[int(np.argmax(lp))])
        # q restricted to decoded paths ∪ ∅
        keys = [c for c in decoded if c in score]
        q_dec = softmax([score[c] for c in keys] + [post["null_logp"][i]])
        pick = "r3" if int(np.argmax(q_dec)) == len(keys) else str(keys[int(np.argmax(q_dec))])
        tc = int(post["target_cluster"][i])
        label_pick = str(tc) if str(tc) in rec["paths"] else "r3"
        order = np.argsort(-mix[i])
        rank_of = lambda c: int(np.where(order == c)[0][0]) + 1 if c >= 0 else 151
        rows.append({
            "r3": r3_hit, "q_sel": hit(rec, pick), "label_sel": hit(rec, label_pick),
            "oracle": r3_hit or any(hit(rec, str(c)) for c in decoded),
            "q_null": q_null, "q_picks_null": pick == "r3",
            "top_is_label": top == tc, "top_converts": hit(rec, str(top)) if str(top) in rec["paths"] else None,
            "rank_top": rank_of(top), "rank_label": rank_of(tc),
        })

    def summarize(sel):
        R = [r for r, s in zip(rows, sel) if s]
        m = lambda k: float(np.mean([r[k] for r in R])) if R else float("nan")
        conv = [r["top_converts"] for r in R if r["top_converts"] is not None]
        return {
            "rows": len(R), "R3 HR@10": m("r3"), "select by q (decoded ∪ ∅)": m("q_sel"),
            "select by label": m("label_sel"), "best-path oracle": m("oracle"),
            "mean q(∅)": m("q_null"), "q picks ∅": m("q_picks_null"),
            "argmax q = label cluster": m("top_is_label"),
            "argmax q path converts (when decoded)": float(np.mean(conv)) if conv else float("nan"),
            "argmax q decoded": len(conv) / max(len(R), 1),
            "median mix rank of argmax q": float(np.median([r["rank_top"] for r in R])) if R else float("nan"),
            "median mix rank of label": float(np.median([r["rank_label"] for r in R])) if R else float("nan"),
        }

    groups = {
        "all rows": [True] * n,
        "R3 hits": [r["r3"] for r in rows],
        "R3 misses": [not r["r3"] for r in rows],
        "report rows 1,000+ (misses)": [i >= 1000 and not rows[i]["r3"] for i in range(n)],
    }
    res = {g: summarize(s) for g, s in groups.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, a.out.with_suffix(".json").open("w"), indent=2)
    keys = list(next(iter(res.values())).keys())
    lines = ["# E2-diag: decoder posterior over cluster evidence (test rows)", "",
             "| metric | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{res[g][k]:.4f}" if isinstance(res[g][k], float) else str(res[g][k])
                                             for g in groups) + " |")
    lines += ["", "Gate: on R3 misses, *argmax q path converts* (and *select by q*) must clearly exceed the "
              "22.7% true-path selection of the best predictor; *median mix rank of argmax q* well below that "
              "of the label means q is easier to learn from x than the label."]
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
