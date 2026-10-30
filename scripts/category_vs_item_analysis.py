"""Category prediction vs item prediction on the same test rows.

For every item-ranking eval (per-row JSONs, same row order as the test CSV) and
the category model's eval (scripts/category_prediction.sbatch), on test rows
whose target has a category:

  * category-model HR@1/3/5/10 and item-model HR@1/3/5/10;
  * 2x2 tables of category-model top-1 hit vs item-model hit@1 and hit@10;
  * the item model's *implied* category (category of its top-1 item), the
    baseline a dedicated category model has to beat, and the rows where the
    category model is right while the item model misses and its top-1 item's
    category is also wrong (information the item ranking does not reflect).

Proportions get 95% bootstrap CIs (2,000 resamples, seed 42).
Usage: category_vs_item_analysis.py --category_eval <json> --item_eval NAME=<json> ... --out <md>
"""

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

import category_prediction_data as cpd

CUTOFFS = (1, 3, 5, 10)


def ranks(rows):
    out = []
    for row in rows:
        target = str(row["output"]).strip(' "\n')
        preds = [str(p).strip(' "\n') for p in row["predict"]]
        out.append(next((i for i, p in enumerate(preds) if p == target), 99))
    return np.array(out)


def ci(x, rng_idx):
    lo, hi = np.percentile(x[rng_idx].mean(1), [2.5, 97.5])
    return f"{x.mean():.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_eval", required=True)
    parser.add_argument("--item_eval", action="append", required=True, help="NAME=path")
    parser.add_argument("--data_dir", default="data/Toys_and_Games")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.data_dir))
    paths = cpd.category_map(os.path.join(args.data_dir, f"meta_{name}.json"), args.level)
    test = pd.read_csv(os.path.join(args.data_dir, "test", cpd.SPLIT_FILE))
    title_cat = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(os.path.join(args.data_dir, split, cpd.SPLIT_FILE))
        for asin, title in zip(frame["item_asin"], frame["item_title"]):
            title_cat.setdefault(str(title).strip(' "\n'), paths.get(asin))
        for asins, titles in zip(frame["item_asins"], frame["history_item_title"]):
            for asin, title in zip(ast.literal_eval(asins), ast.literal_eval(titles)):
                title_cat.setdefault(str(title).strip(' "\n'), paths.get(asin))

    cat_rows = json.load(open(args.category_eval))
    if len(cat_rows) != len(test):
        raise ValueError(f"category eval has {len(cat_rows)} rows, test has {len(test)}")
    target_cat = test["item_asin"].map(paths)
    keep = target_cat.notna().to_numpy()
    for row, cat in zip(cat_rows, target_cat):
        if pd.notna(cat) and str(row["output"]).strip(' "\n') != cat:
            raise ValueError("category eval rows are not aligned with the test CSV")
    cat_rank = ranks(cat_rows)[keep]
    n = int(keep.sum())
    idx = np.random.default_rng(42).integers(0, n, size=(2000, n))
    lines = [f"# Category vs item prediction ({name}, level-{args.level} categories)", "",
             f"Test rows with a target category: n = {n} of {len(test)}. @20 = @10 with 10 beams.", "",
             "## Category model", "",
             "| HR@1 | HR@3 | HR@5 | HR@10 |", "|---:|---:|---:|---:|",
             "| " + " | ".join(f"{(cat_rank < c).mean():.4f}" for c in CUTOFFS) + " |", ""]
    cat1 = (cat_rank < 1).astype(float)
    tc = target_cat.to_numpy()[keep]
    last = np.array([paths.get(ast.literal_eval(a)[-1]) for a in test["item_asins"]], dtype=object)[keep]
    in_hist = np.array([t in {paths.get(x) for x in ast.literal_eval(a)}
                        for a, t in zip(test["item_asins"][keep], tc)], dtype=float)
    lines += [f"- trivial baseline, last history item's category: top-1 {ci((last == tc).astype(float), idx)}",
              f"- target category appears in the history: {ci(in_hist, idx)}",
              f"- category model − last-item baseline (top-1): {ci(cat1 - (last == tc), idx)}", ""]

    for spec in args.item_eval:
        label, path = spec.split("=", 1)
        rows = json.load(open(path))
        if len(rows) != len(test) or any(str(r["output"]).strip(' "\n') != str(t).strip(' "\n')
                                         for r, t in zip(rows, test["item_title"])):
            raise ValueError(f"{label}: rows not aligned with the test CSV")
        item_rank = ranks(rows)[keep]
        top1_cat = np.array([title_cat.get(str(r["predict"][0]).strip(' "\n')) if r["predict"] else None
                             for r in rows], dtype=object)[keep]
        implied = (top1_cat == target_cat.to_numpy()[keep]).astype(float)
        top10_has = np.array([target_cat.iloc[i] in {title_cat.get(str(p).strip(' "\n')) for p in rows[i]["predict"]}
                              for i in np.flatnonzero(keep)], dtype=float)
        lines += [f"## Item model: {label}", "",
                  "| HR@1 | HR@3 | HR@5 | HR@10 |", "|---:|---:|---:|---:|",
                  "| " + " | ".join(f"{(item_rank < c).mean():.4f}" for c in CUTOFFS) + " |", "",
                  f"- implied category (top-1 item's category) accuracy: {ci(implied, idx)}",
                  f"- target category among the top-10 items' categories: {ci(top10_has, idx)}",
                  f"- category model top-1 accuracy (same rows): {ci(cat1, idx)}",
                  f"- category model − implied: {ci(cat1 - implied, idx)}", ""]
        for c in (1, 10):
            hit = item_rank < c
            cat = cat1.astype(bool)
            lines += [f"### Category top-1 vs item hit@{c}", "",
                      "| | item hit | item miss |", "|---|---:|---:|",
                      f"| category right | {int((cat & hit).sum())} | **{int((cat & ~hit).sum())}** |",
                      f"| category wrong | {int((~cat & hit).sum())} | {int((~cat & ~hit).sum())} |", "",
                      f"- P(category right | item hit@{c}) = {cat[hit].mean():.4f}; "
                      f"P(category right | item miss@{c}) = {cat[~hit].mean():.4f}",
                      f"- P(item hit@{c} | category right) = {hit[cat].mean():.4f}; "
                      f"P(item hit@{c} | category wrong) = {hit[~cat].mean():.4f}", ""]
        miss = item_rank >= 10
        novel = cat1.astype(bool) & miss & (implied == 0)
        novel10 = novel & (top10_has == 0)
        lines += ["### Rows the item ranking does not reflect", "",
                  f"- category model right, item miss@10, top-1 item's category wrong: {int(novel.sum())} "
                  f"({novel.mean():.4f} of rows, {novel[miss].mean():.4f} of item misses)",
                  f"- ... and no top-10 item has the target category: {int(novel10.sum())} ({novel10.mean():.4f} of rows)",
                  f"- converse (implied category right, category model wrong): {int(((implied == 1) & (cat1 == 0)).sum())}", ""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
