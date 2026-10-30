"""Next-item *category* prediction data: the split CSVs with ``item_title``
replaced by the target item's Amazon category path cut to ``--level`` levels
(history titles unchanged), plus a catalog file of the distinct paths.

Train / valid rows whose target has no category are dropped; test rows are all
kept (row order = the item-ranking evals) and get ``--missing`` as target, which
is not in the catalog, so the analysis can exclude them.

Usage: category_prediction_data.py --data_dir data/Toys_and_Games --output_dir <dir>
"""

import argparse
import ast
import html
import json
import os

import pandas as pd

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


def category_map(meta_file, level):
    paths = {}
    with open(meta_file) as handle:
        for line in handle:
            record = json.loads(line)
            parts = [html.unescape(str(p)).strip() for p in (record.get("category") or [])][:level]
            if parts:
                paths[record["asin"]] = " > ".join(parts)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Toys_and_Games")
    parser.add_argument("--meta_file", default=None)
    parser.add_argument("--split_file", default=SPLIT_FILE)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--missing", default="Unknown")
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.data_dir))
    paths = category_map(args.meta_file or os.path.join(args.data_dir, f"meta_{name}.json"), args.level)
    catalog, asins = set(), set()
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(os.path.join(args.data_dir, split, args.split_file))
        asins.update(frame["item_asin"])
        for history in frame["item_asins"]:
            asins.update(ast.literal_eval(history))
        frame["item_title"] = frame["item_asin"].map(paths)
        missing = frame["item_title"].isna()
        if split == "test":
            frame["item_title"] = frame["item_title"].fillna(args.missing)
        else:
            frame = frame[~missing].reset_index(drop=True)
            catalog.update(frame["item_title"])
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
        frame.to_csv(os.path.join(args.output_dir, split, args.split_file), index=False)
        print(f"{split}: {len(frame)} rows, {int(missing.sum())} targets without a category"
              f"{' (kept)' if split == 'test' else ' (dropped)'}")
    catalog.update(paths[a] for a in asins if a in paths)  # every item in the dataset
    catalog.discard(args.missing)
    with open(os.path.join(args.output_dir, "categories.txt"), "w") as handle:
        for i, path in enumerate(sorted(catalog)):
            handle.write(f"{path}\t{i}\n")
    print(f"catalog: {len(catalog)} level-{args.level} paths")


if __name__ == "__main__":
    main()
