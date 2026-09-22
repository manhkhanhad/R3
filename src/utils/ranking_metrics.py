"""Ranking metrics shared by inference and final-result evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TOP_K = (1, 3, 5, 10, 20)


def infer_dataset_name(item_path: str) -> str:
    path = Path(item_path)
    if path.parent.name == "info" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem.split("_5_")[0]


def calculate_ranking_metrics(
    test_data: Iterable[Mapping[str, Any]],
    item_path: str,
    top_k: Sequence[int] = DEFAULT_TOP_K,
) -> dict[str, float | int]:
    item_file = Path(item_path)
    if item_file.suffix != ".txt":
        item_file = item_file.with_suffix(".txt")

    with item_file.open("r") as handle:
        known_items = {
            line.split("\t")[0].strip(' "\n')
            for line in handle
        }

    rows = list(test_data)
    if not rows:
        raise ValueError("Cannot calculate ranking metrics for an empty result set")

    hits = [0 for _ in top_k]
    ndcgs = [0.0 for _ in top_k]
    unknown_predictions = 0

    for row in rows:
        predictions = [str(value).strip(' "\n') for value in row["predict"]]
        target_value = row["output"]
        if isinstance(target_value, list):
            target_value = target_value[0]
        target = str(target_value).strip(' "\n')

        rank = None
        for index, prediction in enumerate(predictions):
            if prediction not in known_items:
                unknown_predictions += 1
            if rank is None and prediction == target:
                rank = index

        if rank is None:
            continue
        for index, cutoff in enumerate(top_k):
            if rank < cutoff:
                hits[index] += 1
                ndcgs[index] += math.log(2) / math.log(rank + 2)

    metrics: dict[str, float | int] = {
        "samples": len(rows),
        "unknown_predictions": unknown_predictions,
    }
    for index, cutoff in enumerate(top_k):
        metrics[f"ndcg@{cutoff}"] = ndcgs[index] / len(rows)
        metrics[f"hr@{cutoff}"] = hits[index] / len(rows)
    return metrics
