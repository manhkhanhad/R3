#!/usr/bin/env python3
"""Fit t-SNE directly to raw latent vectors and create a Plotly scatter plot.

There is intentionally no PCA pre-reduction, centering, normalization,
subsampling, clustering, or template mapping. Every row in ``thoughts.pt`` is
passed directly to t-SNE and becomes one point in the interactive figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def color_values(metadata: pd.DataFrame, vectors: np.ndarray,
                 color: str) -> tuple[np.ndarray, str, str]:
    if color == "utility":
        return metadata.utility.to_numpy(), "Target-NLL gain versus zero", "Viridis"
    if color == "raw_nll":
        return metadata.raw_nll.to_numpy(), "Raw target NLL", "Plasma"
    if color == "history_len":
        return metadata.history_len.to_numpy(), "History length", "Turbo"
    if color == "norm":
        return np.linalg.norm(vectors, axis=1), "Raw latent L2 norm", "Cividis"
    raise ValueError(color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="Directory containing thoughts.pt and metadata.csv")
    parser.add_argument("--tag", required=True,
                        help="Short output name, for example rl or sft_only")
    parser.add_argument("--title", default="Raw latent-reasoning t-SNE")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--early-exaggeration", type=float, default=12.0)
    parser.add_argument("--learning-rate", default="auto",
                        help="Use 'auto' or a positive numeric value")
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--angle", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color", choices=("utility", "raw_nll", "history_len", "norm"),
                        default="history_len")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--plotlyjs", choices=("inline", "cdn"), default="inline")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if not args.tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--tag must contain only letters, numbers, underscores, or hyphens")
    thought_path = args.evidence_dir / "thoughts.pt"
    metadata_path = args.evidence_dir / "metadata.csv"
    stored = torch.load(thought_path, map_location="cpu", weights_only=True)
    vectors = stored.float().numpy()
    metadata = pd.read_csv(metadata_path)
    if len(vectors) != len(metadata):
        raise ValueError(f"vector/metadata mismatch: {len(vectors)} != {len(metadata)}")
    if args.perplexity >= len(vectors):
        raise ValueError("perplexity must be smaller than the number of vectors")

    try:
        learning_rate: str | float = float(args.learning_rate)
    except ValueError:
        if args.learning_rate != "auto":
            raise ValueError("--learning-rate must be 'auto' or numeric") from None
        learning_rate = "auto"

    cache_dir = ANALYSIS / "cache/tsne"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = (f"{args.tag}_raw_rows_perp{args.perplexity:g}_ee{args.early_exaggeration:g}_"
                 f"lr{args.learning_rate}_iter{args.max_iter}_angle{args.angle:g}_seed{args.seed}")
    cache_path = cache_dir / f"{cache_key}.npz"
    if args.reuse_cache and cache_path.exists():
        cached = np.load(cache_path)
        coordinates = cached["coordinates"]
        kl_divergence = float(cached["kl_divergence"])
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            early_exaggeration=args.early_exaggeration,
            learning_rate=learning_rate,
            max_iter=args.max_iter,
            init="random",
            method="barnes_hut",
            angle=args.angle,
            metric="euclidean",
            random_state=args.seed,
            n_jobs=-1,
            verbose=args.verbose,
        )
        coordinates = reducer.fit_transform(vectors).astype(np.float32)
        kl_divergence = float(reducer.kl_divergence_)
        np.savez_compressed(cache_path, coordinates=coordinates,
                            kl_divergence=np.float64(kl_divergence))

    colors, color_title, colorscale = color_values(metadata, vectors, args.color)
    customdata = np.column_stack((
        metadata.sample_id.astype(str),
        metadata.user_id.astype(str),
        metadata.target_item.astype(str),
        metadata.history_len.astype(str),
        metadata.utility.astype(str),
        metadata.raw_nll.astype(str),
    ))
    figure = go.Figure(go.Scattergl(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        mode="markers",
        marker={
            "size": 4,
            "opacity": 0.45,
            "color": colors,
            "colorscale": colorscale,
            "colorbar": {"title": color_title},
            "line": {"width": 0},
        },
        customdata=customdata,
        hovertemplate=(
            "sample=%{customdata[0]}<br>user=%{customdata[1]}"
            "<br>target item=%{customdata[2]}<br>history length=%{customdata[3]}"
            "<br>target-NLL gain vs zero=%{customdata[4]}"
            "<br>raw target NLL=%{customdata[5]}"
            "<br>t-SNE 1=%{x:.3f}<br>t-SNE 2=%{y:.3f}<extra></extra>"),
    ))
    figure.update_layout(
        title={"text": f"{args.title}<br><sup>All {len(vectors):,} raw {vectors.shape[1]}D vectors; no pre-reduction or preprocessing</sup>",
               "x": 0.5},
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        height=850,
        width=1200,
        hovermode="closest",
    )

    figure_dir = ANALYSIS / "figures"
    result_dir = ANALYSIS / "results/latent_distribution" / args.tag
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    html_path = figure_dir / f"{args.tag}_raw_latent_tsne_plotly.html"
    coordinate_path = result_dir / "train_raw_latent_tsne_coordinates.npz"
    figure.write_html(
        html_path,
        include_plotlyjs=True if args.plotlyjs == "inline" else "cdn",
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    np.savez_compressed(coordinate_path, coordinates=coordinates)
    summary = {
        "evidence_dir": str(args.evidence_dir),
        "thought_sha256": sha256(thought_path),
        "metadata_sha256": sha256(metadata_path),
        "rows": len(vectors),
        "input_dimension": vectors.shape[1],
        "input_dtype": str(stored.dtype),
        "input": "raw vectors; no PCA, centering, normalization, subsampling, or template mapping",
        "method": "sklearn Barnes-Hut t-SNE",
        "metric": "euclidean",
        "init": "random",
        "perplexity": args.perplexity,
        "early_exaggeration": args.early_exaggeration,
        "learning_rate": learning_rate,
        "max_iter": args.max_iter,
        "angle": args.angle,
        "seed": args.seed,
        "kl_divergence": kl_divergence,
        "color": args.color,
        "html": str(html_path),
        "coordinates": str(coordinate_path),
        "cache": str(cache_path),
    }
    atomic_json(result_dir / "train_raw_latent_tsne_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
