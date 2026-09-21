#!/usr/bin/env python3
"""Aggregate and plot latent-distribution diagnostics across SFT epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
CHECKPOINTS = ((1, 384), (2, 768), (3, 1152))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", help="training-run version, e.g. v3 or v4")
    parser.add_argument("--dataset", default="CDs_and_Vinyl", help="category name, e.g. Musical_Instruments")
    parser.add_argument("--steps", default="", help="comma-separated checkpoint steps, e.g. 519,1038,1557 (default: 384,768,1152)")
    parser.add_argument("--tag-prefix", default="", help="tag prefix before _checkpoint_<step> (default: latent_<version>)")
    args = parser.parse_args()
    run = f"latent_{args.version}"
    tag_prefix = args.tag_prefix or run
    checkpoints = (
        tuple(enumerate((int(s) for s in args.steps.split(",")), start=1))
        if args.steps else CHECKPOINTS
    )

    rows = []
    for epoch, step in checkpoints:
        tag = f"{tag_prefix}_checkpoint_{step}"
        distribution_path = (
            ANALYSIS / "results/latent_distribution" / tag / "train_latent_distribution.json"
        )
        extraction_path = (
            ANALYSIS / f"evidence/{args.dataset}" / tag / "train/summary.json"
        )
        distribution = json.loads(distribution_path.read_text())
        extraction = json.loads(extraction_path.read_text())
        templates = distribution["exact_templates_at_stored_precision"]
        pca = distribution["pca"]
        rows.append(
            {
                "epoch": epoch,
                "step": step,
                "tag": tag,
                "checkpoint": distribution["checkpoint"],
                "checkpoint_sha256": distribution["checkpoint_sha256"],
                "n": distribution["n"],
                "raw_train_nll": extraction["raw_nll"],
                "zero_mask_train_nll": extraction["masked_nll"],
                "zero_mask_penalty": extraction["mean_utility"],
                "mean_norm_ratio": distribution["mean_norm_over_mean_sample_norm"],
                "random_pair_cosine": distribution["pairwise_cosine"]["random"]["mean"],
                "unique_vectors": templates["unique_vectors"],
                "unique_fraction": templates["unique_fraction"],
                "top_10_template_fraction": templates["top_10_sample_fraction"],
                "pc1_variance_fraction": pca["top_eigenvalue_fraction"],
                "top_2_variance_fraction": pca["top_2_fraction"],
                "entropy_effective_rank": pca["entropy_effective_rank"],
                "participation_ratio": pca["participation_ratio"],
                "components_for_90pct": pca["components_for_90pct"],
            }
        )

    result_dir = ANALYSIS / f"results/latent_distribution/{run}_epoch_comparison"
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "summary.json"
    output.write_text(json.dumps({"checkpoints": rows}, indent=2, sort_keys=True) + "\n")

    epochs = [row["epoch"] for row in rows]
    panels = (
        ("random_pair_cosine", "Random-pair cosine", None),
        ("mean_norm_ratio", "Global-mean norm ratio", None),
        ("unique_fraction", "Unique vectors", "percent"),
        ("top_10_template_fraction", "Top-10 template coverage", "percent"),
        ("entropy_effective_rank", "Entropy effective rank", "log"),
        ("zero_mask_penalty", "Zero-mask train NLL penalty", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title, scale) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        if scale == "percent":
            values = [100 * value for value in values]
        ax.plot(epochs, values, marker="o", linewidth=2.2, markersize=7)
        ax.set(title=title, xlabel="SFT epoch", xticks=epochs)
        ax.grid(alpha=0.25)
        if scale == "percent":
            ax.set_ylabel("Percent")
        elif scale == "log":
            ax.set_yscale("log")
        for x, value in zip(epochs, values):
            ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle(f"{args.dataset} {args.version} latent geometry across SFT epochs", fontsize=15)
    fig.tight_layout()
    figure = ANALYSIS / f"figures/{run}_epoch_collapse_trajectory.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": str(output), "figure": str(figure), "checkpoints": rows}, indent=2))


if __name__ == "__main__":
    main()
