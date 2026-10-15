"""Offline re-ranking of the pooled per-intent top-10 lists (no new generation).

Scores each candidate from the saved per-intent lists only:
  borda   : sum over intents of (10 - rank) for intents that list it
  count   : number of intents listing it, ties broken by best rank
  rrf     : reciprocal-rank fusion, sum 1/(60 + rank)
Compares with the saved max-merge `mixture` ranking.
"""
import json
import math
import sys

run_dir = sys.argv[1]
rows = json.load(open(f"{run_dir}/eval_mixture.json"))
clean = lambda t: str(t).strip(' "\n')
K = len(rows[0]["intent_predict"])


def metrics(ranked_lists):
    hr = {k: 0.0 for k in (1, 3, 5, 10)}
    nd = {k: 0.0 for k in (1, 3, 5, 10)}
    for row, ranked in zip(rows, ranked_lists):
        target = clean(row["output"])
        rank = next((i for i, t in enumerate(ranked) if t == target), None)
        for k in hr:
            if rank is not None and rank < k:
                hr[k] += 1
                nd[k] += math.log(2) / math.log(rank + 2)
    n = len(rows)
    return " ".join(f"HR@{k} {hr[k] / n:.4f}" for k in hr) + " | " + " ".join(f"NDCG@{k} {nd[k] / n:.4f}" for k in nd)


def fuse(score_fn):
    out = []
    for row in rows:
        scores = {}
        best = {}
        for k in range(K):
            for rank, t in enumerate(map(clean, row["intent_predict"][str(k)])):
                scores[t] = scores.get(t, 0.0) + score_fn(rank)
                best[t] = min(best.get(t, 99), rank)
        out.append(sorted(scores, key=lambda t: (-scores[t], best[t]))[:10])
    return out


print("mixture (saved max-merge):", metrics([[clean(t) for t in r["predict"]] for r in rows]))
print("borda                    :", metrics(fuse(lambda r: 10 - r)))
print("count + best rank        :", metrics(fuse(lambda r: 1.0)))
print("reciprocal-rank fusion   :", metrics(fuse(lambda r: 1.0 / (60 + r))))
any_hit = sum(clean(r["output"]) in {clean(t) for k in range(K) for t in r["intent_predict"][str(k)]} for r in rows)
print(f"pool recall (any intent lists target): {any_hit / len(rows):.4f}")
