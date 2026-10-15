"""Why does routing fail? Diagnose what the K intents encode, from saved per-intent lists.

1. Do intents partition the item catalogue? Mutual information between intent id and
   the item cluster of what it recommends.
2. For each test row, which intents put the target in their top-10, and how often
   exactly one does (routing must then be exact).
3. Is the winning intent predictable from the target's item cluster (label leakage)
   and is the target's cluster predictable from the model's own predictions (a ceiling
   for history-only routing)?
"""
import collections
import json
import math
import sys

import torch

run_dir, info_path, cluster_path = sys.argv[1:4]
title_to_id = {}
for line in open(info_path):
    parts = line.rstrip("\n").split("\t")
    title_to_id[parts[0].strip(' "\n')] = int(parts[-1])
clusters = torch.load(cluster_path, map_location="cpu", weights_only=False)
cluster_of = clusters["cluster_id"].tolist()
n_clusters = int(clusters["k"])
rows = json.load(open(f"{run_dir}/eval_mixture.json"))
K = len(rows[0]["intent_predict"])
clean = lambda t: str(t).strip(' "\n')


def cluster(title):
    item = title_to_id.get(clean(title))
    return None if item is None else cluster_of[item]


# 1. intent -> cluster distribution of recommended items
joint = collections.Counter()
for row in rows:
    for k, items in row["intent_predict"].items():
        for t in items:
            c = cluster(t)
            if c is not None:
                joint[(int(k), c)] += 1
total = sum(joint.values())
pk = collections.Counter()
pc = collections.Counter()
for (k, c), n in joint.items():
    pk[k] += n
    pc[c] += n
mi = sum(n / total * math.log((n / total) / ((pk[k] / total) * (pc[c] / total))) for (k, c), n in joint.items())
hk = -sum(n / total * math.log(n / total) for n in pk.values())
hc = -sum(n / total * math.log(n / total) for n in pc.values())
print(f"[1] intent vs recommended-item cluster ({n_clusters} SASRec clusters): MI {mi:.3f} nats, "
      f"NMI {mi / math.sqrt(hk * hc):.3f}  (0 = intents ignore item region, 1 = intents = regions)")
for k in range(K):
    top = sorted(((n, c) for (kk, c), n in joint.items() if kk == k), reverse=True)[:3]
    share = sum(n for n, _ in top) / pk[k]
    print(f"    intent {k}: top-3 clusters {[c for _, c in top]} cover {share:.2f} of its recommendations")
overlap = []
for row in rows:
    lists = [set(map(clean, row["intent_predict"][str(k)])) for k in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            overlap.append(len(lists[i] & lists[j]) / 10)
print(f"    mean top-10 overlap between intents: {sum(overlap) / len(overlap):.3f}")

# 2. which intents hit the target
hits_dist = collections.Counter()
single_winner = collections.Counter()
for row in rows:
    target = clean(row["output"])
    hitters = [k for k in range(K) if target in map(clean, row["intent_predict"][str(k)])]
    hits_dist[len(hitters)] += 1
    if len(hitters) == 1:
        single_winner[hitters[0]] += 1
n = len(rows)
print(f"[2] rows by number of intents whose top-10 contains the target: "
      + ", ".join(f"{h}: {hits_dist[h] / n:.4f}" for h in range(K + 1)))
union = 1 - hits_dist[0] / n
print(f"    union-of-intents HR@10 (any intent hits): {union:.4f}; "
      f"rows where exactly one intent hits: {hits_dist[1] / n:.4f}")
print(f"    when exactly one hits, which one: " + ", ".join(f"{k}: {single_winner[k] / max(1, hits_dist[1]):.2f}" for k in range(K)))

# 3. is the winning intent determined by the target's cluster?
win_by_cluster = collections.defaultdict(collections.Counter)
for row in rows:
    target = clean(row["output"])
    c = cluster(row["output"])
    hitters = [k for k in range(K) if target in map(clean, row["intent_predict"][str(k)])]
    if c is not None and len(hitters) == 1:
        win_by_cluster[c][hitters[0]] += 1
correct = sum(max(cnt.values()) for cnt in win_by_cluster.values())
count = sum(sum(cnt.values()) for cnt in win_by_cluster.values())
print(f"[3] predicting the single winning intent from the TARGET's cluster (majority per cluster): "
      f"accuracy {correct / max(1, count):.3f} (chance ~{1 / K:.2f}) over {count} rows")
# Ceiling for history-only routing: does the model's own top prediction land in the target's cluster?
same = 0
seen = 0
for row in rows:
    c = cluster(row["output"])
    preds = row["predict"]
    if c is None or not preds:
        continue
    pc_top = cluster(preds[0])
    if pc_top is None:
        continue
    seen += 1
    same += int(pc_top == c)
print(f"    model's top-1 mixture prediction lies in the target's cluster: {same / max(1, seen):.3f} "
      f"(chance ~{1 / n_clusters:.3f}); this is how well history alone locates the target's region")
