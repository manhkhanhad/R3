"""Does a CF model (SASRec) carry information R3 misses?  Offline, no training.

Rows are split into *copy rows* (the target title's first word, ~brand, matches
the first word of a history title) and *other rows*.  For each LLM item model
and SASRec: HR/NDCG per subset, paired-bootstrap differences, rows only one
model hits, and the oracle per-row best-of-both HR@10 (upper bound for routing).

SASRec recommendations come from the Verifiable-Rec Cornac export (top-20
training-catalog items, history excluded), one row per test CSV row.

Usage: cf_complementarity.py --sasrec <test_recommendations.csv>
           --item_eval R3=<json> --item_eval T2=<json> [--out report.md]
"""

import argparse
import ast
import json
import math

import numpy as np
import pandas as pd

KS = (1, 3, 5, 10)


def first_word(title):
    words = str(title).lower().split()
    return words[0] if words else ""


def rank_of(target, ranked):
    """1-based rank of target in ranked[:10], or 0."""
    for i, x in enumerate(ranked[:10]):
        if x == target:
            return i + 1
    return 0


def metrics(ranks):
    ranks = np.asarray(ranks)
    out = {}
    for k in KS:
        hit = (ranks > 0) & (ranks <= k)
        out[f"hr@{k}"] = hit.mean()
        out[f"ndcg@{k}"] = np.where(hit, 1 / np.log2(ranks + 1.0, where=ranks > 0, out=np.ones(len(ranks))), 0).mean()
    return out


def boot(a, b, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), (n, len(d)))
    s = d[idx].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default="data/Toys_and_Games/test/Toys_and_Games_5_2017-1-2018-11.csv")
    p.add_argument("--sasrec", required=True)
    p.add_argument("--item_eval", action="append", required=True, help="NAME=path to per-row eval JSON")
    p.add_argument("--out")
    args = p.parse_args()

    test = pd.read_csv(args.test_csv)
    sas = pd.read_csv(args.sasrec)
    assert len(sas) == len(test) and (sas["target_asin"].values == test["item_asin"].values).all(), "row misalignment"
    n = len(test)
    hist_titles = [ast.literal_eval(h) for h in test["history_item_title"]]
    copy = np.array([first_word(t) in {first_word(h) for h in hs} for t, hs in zip(test["item_title"], hist_titles)])
    target_in_hist = np.array([a in ast.literal_eval(h) for a, h in zip(test["item_asin"], test["item_asins"])])

    ranks = {"SASRec": np.array([rank_of(a, json.loads(r)) for a, r in zip(test["item_asin"], sas["recommended_asins"])])}
    for spec in args.item_eval:
        name, path = spec.split("=", 1)
        rows = json.load(open(path))
        assert len(rows) == n
        ranks[name] = np.array([rank_of(r["output"].strip().strip('"'), [x.strip().strip('"') for x in r["predict"]])
                                for r in rows])

    lines = [f"# CF complementarity (n = {n}; @20 = @10 for the 10-beam LLMs)", "",
             f"Copy rows (target first word in a history title's first words): {copy.sum()}; other rows: {(~copy).sum()}. "
             f"Target itself in history: {target_in_hist.sum()} rows (SASRec export excludes history items).", ""]
    subsets = [("all", np.ones(n, bool)), ("copy rows", copy), ("other rows", ~copy)]
    for label, m in subsets:
        lines += [f"## {label} (n = {m.sum()})", "",
                  "| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |", "|---|" + "---:|" * 7]
        for name, r in ranks.items():
            x = metrics(r[m])
            lines.append(f"| {name} | " + " | ".join(f"{x[k]:.4f}" for k in
                         ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@3", "ndcg@5", "ndcg@10")) + " |")
        lines.append("")
        hit = {k: (v[m] > 0) for k, v in ranks.items()}
        for name in ranks:
            if name == "SASRec":
                continue
            d, lo, hi = boot(hit["SASRec"], hit[name])
            only_s = (hit["SASRec"] & ~hit[name]).sum()
            only_l = (hit[name] & ~hit["SASRec"]).sum()
            both = (hit["SASRec"] & hit[name]).sum()
            oracle = (hit["SASRec"] | hit[name])
            od, olo, ohi = boot(oracle, hit[name])
            lines += [f"- SASRec − {name} HR@10: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]; both hit {both}, SASRec only {only_s}, {name} only {only_l}",
                      f"- oracle best-of-both HR@10 {oracle.mean():.4f}; − {name}: {od:+.4f} [{olo:+.4f}, {ohi:+.4f}]"]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
