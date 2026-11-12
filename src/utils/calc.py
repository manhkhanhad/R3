import json
import os
import sys

import fire
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from experiment_tracking import init_evaluation_run
from ranking_metrics import DEFAULT_TOP_K, calculate_ranking_metrics, infer_dataset_name


def gao(path, item_path, dataset=None, wandb_run_name=None):
    """Calculate final ranking metrics and log them to the experiment run."""
    print(path)
    print(item_path)
    paths = path if isinstance(path, list) else [path]
    dataset = dataset or infer_dataset_name(item_path)
    wandb_run = init_evaluation_run(
        dataset=dataset,
        run_name=wandb_run_name,
        config={"result_path": paths, "item_path": item_path, "dataset": dataset},
    )

    all_metrics = {}
    for result_path in paths:
        with open(result_path, "r") as handle:
            test_data = json.load(handle)

        metrics = calculate_ranking_metrics(test_data, item_path)
        all_metrics[result_path] = metrics
        ndcg = np.array([metrics[f"ndcg@{cutoff}"] for cutoff in DEFAULT_TOP_K])
        hit_rate = np.array([metrics[f"hr@{cutoff}"] for cutoff in DEFAULT_TOP_K])
        print(ndcg)
        print(hit_rate)
        print(metrics["unknown_predictions"])

        if wandb_run is not None:
            wandb_run.log({f"evaluation/{key}": value for key, value in metrics.items()})

    if wandb_run is not None:
        wandb_run.finish()
    return all_metrics


if __name__ == "__main__":
    fire.Fire(gao)
