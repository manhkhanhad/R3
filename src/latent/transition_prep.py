"""Item-transition neighbour sets for the transition-reading latent (transition_latent.py).

Transition graph from the training rows: a -> b whenever a user bought b right
after a (consecutive items of history + target; one count per user and
position, self-transitions dropped).  The graph is time-causal: a row only
sees transitions whose successor purchase happened before the row's own target
time.  Toys is split by time, so this rule is the same for training and test
rows and no row sees its own answer.

For each row: the top-M successors (by count) of each of its last L history
items, excluding items already in the history; one entry per distinct item
(kept: largest count, nearest source).  Padded to L*M with item -1.

Output (.pt): {split: {"items" [N, L*M] long (-1 = pad), "src" [N, L*M] long
(source position from the end, 0 = last item), "count" [N, L*M] float,
"covered" [N] bool (target in the set)}}, plus "L", "M".  Rows follow the CSV
row order of each split.

Usage (from the R3 main checkout):
  transition_prep.py --data_dir data/Toys_and_Games --output output_dir/transition_latent/prep/Toys_transitions.pt
"""

import argparse
import ast
import bisect
import collections
import glob
import os

import numpy as np
import pandas as pd
import torch


def load(data_dir, split):
    path = glob.glob(os.path.join(data_dir, split, "*_5_*.csv"))
    assert len(path) == 1, path
    return pd.read_csv(path[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--output", required=True)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--M", type=int, default=10)
    args = p.parse_args()
    frames = {s: load(args.data_dir, s) for s in ("train", "valid", "test")}

    # successor lists per source item, sorted by time: (time, successor)
    succ = collections.defaultdict(list)
    seen = set()
    tr = frames["train"]
    for u, h, t, ht, tt in zip(tr.user_id, tr.history_item_id, tr.item_id, tr.history_timestamp, tr.timestamp):
        s = ast.literal_eval(h) + [int(t)]
        ts = ast.literal_eval(ht) + [int(tt)]
        for i in range(len(s) - 1):
            key = (u, i, s[i], s[i + 1])
            if key in seen or s[i] == s[i + 1]:
                continue
            seen.add(key)
            succ[s[i]].append((ts[i + 1], s[i + 1]))
    for a in succ:
        succ[a].sort()
    times = {a: [x[0] for x in v] for a, v in succ.items()}

    width = args.L * args.M
    out = {"L": args.L, "M": args.M}
    for split, df in frames.items():
        n = len(df)
        items = torch.full((n, width), -1, dtype=torch.long)
        src = torch.zeros((n, width), dtype=torch.long)
        count = torch.zeros((n, width))
        covered = torch.zeros(n, dtype=torch.bool)
        sizes = []
        for r, (h, t, tt) in enumerate(zip(df.history_item_id, df.item_id, df.timestamp)):
            h = ast.literal_eval(h)
            hist = set(h)
            best = {}  # item -> (count, src)
            for pos, a in enumerate(reversed(h[-args.L:])):
                if a not in succ:
                    continue
                k = bisect.bisect_left(times[a], int(tt))  # transitions strictly before the target time
                c = collections.Counter(b for _, b in succ[a][:k])
                for b, cnt in c.most_common(args.M):
                    if b in hist:
                        continue
                    if b not in best or cnt > best[b][0]:
                        best[b] = (cnt, pos)
            ranked = sorted(best.items(), key=lambda x: (-x[1][0], x[1][1]))[:width]
            for j, (b, (cnt, pos)) in enumerate(ranked):
                items[r, j], src[r, j], count[r, j] = b, pos, cnt
            covered[r] = int(t) in best
            sizes.append(len(ranked))
        out[split] = {"items": items, "src": src, "count": count, "covered": covered}
        print(f"{split}: {n} rows, set size {np.mean(sizes):.1f} (max {max(sizes)}), "
              f"target covered {covered.float().mean():.4f}, empty {np.mean(np.array(sizes) == 0):.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)


if __name__ == "__main__":
    main()
