"""Where does the target land in SASRec's full dot-product ranking?

SASRec scores every catalog item by h_u · e_i (no normalisation, no bias in our
Cornac checkpoint).  Per test row: the target's 1-based rank among all SASRec
catalog items, history items excluded as in the top-k export.  Reports the rank
distribution (bins and quantiles) on all rows, copy rows (target first word in a
history title) and other rows, and separately on rows where the target is
outside the top-10 (how near are the misses).

Usage: sasrec_target_rank.py --sasrec_model <.pkl> [--sasrec_csv <export>] [--out report.md]
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"
BINS = [(1, 1), (2, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 1000), (1001, 10 ** 9)]


def first_word(title):
    words = str(title).lower().split()
    return words[0] if words else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--sasrec_model", required=True)
    p.add_argument("--sasrec_csv", help="top-k export, to check the ranks agree with it")
    p.add_argument("--out")
    args = p.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "latent"))
    from sasrec_pair_features import encode_rows, load_sasrec

    rec = load_sasrec(args.sasrec_model)
    test = pd.read_csv(os.path.join(args.data_dir, "test", SPLIT_FILE))
    seq, target = encode_rows(rec, test)  # normalised h_u: per-row order unchanged
    items = rec.model.item_emb.weight.detach().float()[: rec.num_items].numpy()
    scores = seq @ items.T
    for r, value in enumerate(test["item_asins"]):  # exclude history items, as the export does
        hist = [rec.iid_map[a] for a in ast.literal_eval(value) if rec.iid_map.get(a, rec.num_items) < rec.num_items]
        scores[r, hist] = -np.inf
    known = target >= 0
    rank = np.full(len(test), -1)
    t = target[known]
    rank[known] = (scores[known] > scores[known, t][:, None]).sum(1) + 1
    in_hist = known & np.isneginf(scores[np.arange(len(test)), np.maximum(target, 0)])
    rank[in_hist] = -1  # target in history: not rankable under the export's rule
    copy = np.array([first_word(x) in {first_word(h) for h in ast.literal_eval(hs)}
                     for x, hs in zip(test["item_title"], test["history_item_title"])])

    lines = [f"# Target rank under SASRec's full dot-product ranking (Toys test, n = {len(test)})", "",
             f"Catalog {rec.num_items} items; target outside SASRec's catalog: {(~known).sum()} rows; "
             f"target in history (excluded): {in_hist.sum()} rows.", ""]
    if args.sasrec_csv:
        export = pd.read_csv(args.sasrec_csv)
        hit_export = np.array([a in json.loads(r)[:10] for a, r in zip(test["item_asin"], export["recommended_asins"])])
        lines += [f"Check: rows with rank ≤ 10 here {((rank >= 1) & (rank <= 10)).sum()}, "
                  f"hits@10 in the export {hit_export.sum()}.", ""]
    head = "| rows | n | " + " | ".join(f"{a}" if a == b else (f"{a}–{b}" if b < 10 ** 9 else f">{a - 1}") for a, b in BINS) + \
           " | median | p25 | p75 |"
    lines += ["Share of rows per target-rank bin.", "", head, "|---|---:|" + "---:|" * (len(BINS) + 3)]
    for label, m in (("all", np.ones(len(test), bool)), ("copy", copy), ("other", ~copy),
                     ("miss@10, all", rank > 10), ("miss@10, copy", (rank > 10) & copy),
                     ("miss@10, other", (rank > 10) & ~copy)):
        r = rank[m & (rank > 0)]
        cells = [f"{np.mean((r >= a) & (r <= b)):.3f}" for a, b in BINS]
        lines.append(f"| {label} | {len(r)} | " + " | ".join(cells) +
                     f" | {np.median(r):.0f} | {np.percentile(r, 25):.0f} | {np.percentile(r, 75):.0f} |")
    rand = rec.num_items / 2
    lines += ["", f"Random ranking: median ≈ {rand:.0f}; P(rank ≤ 50) = {50 / rec.num_items:.3f}."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
