#!/usr/bin/env python3
"""Plot E1 ranking metrics by intervention, grouped across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = (
    ("raw", "Raw"),
    ("mask", "Mask"),
    ("mean", "Mean"),
    ("shuffle_user", "Shuffle\nuser"),
    ("shuffle_target", "Shuffle\ntarget"),
    ("noise_0.25", "Noise\n0.25x"),
    ("noise_0.5", "Noise\n0.50x"),
    ("noise_1.0", "Noise\n1.00x"),
)
METRICS = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="run version, e.g. v4 or v5")
    parser.add_argument("--steps", nargs="+", type=int, default=(384, 768, 1152))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_metrics(version: str, step: int, mode: str) -> dict[str, float]:
    path = (
        ROOT
        / "analysis/results/intervention"
        / f"latent_{version}_checkpoint_{step}"
        / f"generation_test_all_k32_s42_{mode}_t1_temp0.1_a1.0.json"
    )
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    if metrics.get("n") != 6157:
        raise ValueError(f"incomplete result in {path}: n={metrics.get('n')}")
    if metrics.get("catalog_violations") != 0:
        raise ValueError(f"catalog violations in {path}: {metrics['catalog_violations']}")
    return metrics


def main() -> None:
    args = parse_args()
    values = {
        step: {mode: load_metrics(args.version, step, mode) for mode, _ in MODES}
        for step in args.steps
    }

    x = np.arange(len(MODES))
    width = 0.8 / len(args.steps)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)

    for ax, metric in zip(axes.flat, METRICS):
        for index, step in enumerate(args.steps):
            offset = (index - (len(args.steps) - 1) / 2) * width
            heights = [values[step][mode][metric] for mode, _ in MODES]
            ax.bar(
                x + offset,
                heights,
                width,
                label=f"Checkpoint {step}",
                color=colors[index],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(metric, fontsize=13, weight="bold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[-1]:
        ax.set_xticks(x, [label for _, label in MODES])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Intervention")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(args.steps),
        frameon=False,
    )
    fig.suptitle(
        f"CDs_and_Vinyl latent_{args.version}: causal intervention ranking",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output = args.output or (
        ROOT / "analysis/figures" / f"latent_{args.version}_e1_ranking_grouped_bars.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
