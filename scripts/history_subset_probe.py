"""Step 0 for B1 (r-steered history reading): is there headroom in choosing
which history items the decoder reads?

Writes test CSVs in which every row keeps only ``k`` of its history items
(chronological order preserved), chosen by

  recent       the k most recent items
  random       k items uniformly at random (seed 42)
  oracle       the k items whose Qwen3 title embedding is most similar to the target's
  antioracle   the k least similar items

Rows with at most k items are left unchanged.  Deleting items from the prompt
hides them from every position, so the existing ranking eval runs unchanged.
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch

LIST_COLUMNS = ["item_asins", "history_item_id", "history_item_title", "history_rating", "history_timestamp"]


def keep_indices(history_ids, target_id, k, rule, emb, rng):
    n = len(history_ids)
    if n <= k:
        return list(range(n))
    if rule == "recent":
        return list(range(n - k, n))
    if rule == "random":
        return sorted(rng.choice(n, size=k, replace=False).tolist())
    sim = emb[history_ids] @ emb[target_id]
    order = np.argsort(-sim, kind="stable") if rule == "oracle" else np.argsort(sim, kind="stable")
    return sorted(order[:k].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ks", default="1,3")
    parser.add_argument("--rules", default="recent,random,oracle,antioracle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=False)["embeddings"].numpy()
    frame = pd.read_csv(args.test_file, index_col=0)
    os.makedirs(args.output_dir, exist_ok=True)
    for k in [int(x) for x in args.ks.split(",")]:
        for rule in args.rules.split(","):
            rng = np.random.default_rng(args.seed)
            out = frame.copy()
            for col in LIST_COLUMNS:
                out[col] = out[col].astype(object)
            for i, row in frame.iterrows():
                ids = ast.literal_eval(row["history_item_id"])
                keep = keep_indices(ids, int(row["item_id"]), k, rule, emb, rng)
                for col in LIST_COLUMNS:
                    values = ast.literal_eval(row[col])
                    if len(values) != len(ids):
                        raise ValueError(f"row {i}: {col} has {len(values)} entries, history has {len(ids)}")
                    out.at[i, col] = str([values[j] for j in keep])
            path = os.path.join(args.output_dir, f"{rule}_k{k}.csv")
            out.to_csv(path)
            print(f"{path}: rows changed {(frame['history_item_id'] != out['history_item_id']).sum()} / {len(out)}")


if __name__ == "__main__":
    main()
