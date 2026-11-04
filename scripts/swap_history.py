"""Give every test row the history of a random row from a different user (intervention).

The target columns (item_*, rating, timestamp, user_id) stay with the row, so
category latents built from the row's target are unchanged; only the history
columns (history_*, item_asins) are swapped.  Row order and the index column
(split.py output) are preserved.

Usage: swap_history.py --input <task_dir>/0.csv --output <task_dir>/0_swaphist.csv [--seed 42]
"""

import argparse

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ["history_item_title", "history_item_id", "item_asins", "history_rating", "history_timestamp"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    frame = pd.read_csv(args.input, index_col=0)
    users = frame["user_id"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    source = []
    for i in range(len(frame)):
        j = int(rng.integers(len(frame)))
        while users[j] == users[i]:
            j = int(rng.integers(len(frame)))
        source.append(j)
    swapped = frame.copy()
    for c in HISTORY_COLUMNS:
        swapped[c] = frame[c].to_numpy()[source]
    swapped.to_csv(args.output, index=True)
    print(f"{len(frame)} rows; history taken from another user's row (seed {args.seed})")


if __name__ == "__main__":
    main()
