#!/usr/bin/env python3
"""Visualize all raw APD latent vectors with an interactive t-SNE plot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--tag", default="latent_dev_apd")
    parser.add_argument("--splits", default="train,valid,test",
                        help="Comma-separated split directories to include")
    parser.add_argument("--color", choices=("split", "recommendation_loss", "norm"),
                        default="split")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    requested_splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    if not requested_splits:
        raise ValueError("--splits must name at least one split")
    scope = "all" if requested_splits == ("train", "valid", "test") else "_".join(requested_splits)
    vectors, records, source_hashes = [], [], {}
    for split in requested_splits:
        latent_path = args.evidence_root / split / "latents.pt"
        metadata_path = args.evidence_root / split / "metadata.jsonl"
        stored = torch.load(latent_path, map_location="cpu", weights_only=True)
        if stored.ndim != 2:
            raise ValueError(f"expected a 2-D tensor in {latent_path}, got {stored.shape}")
        split_records = [json.loads(line) for line in metadata_path.open() if line.strip()]
        if len(stored) != len(split_records):
            raise ValueError(f"latent/metadata mismatch for {split}")
        if not torch.isfinite(stored).all():
            raise ValueError(f"non-finite latent values in {split}")
        vectors.append(stored.float().numpy())
        records.extend({**record, "split": split} for record in split_records)
        source_hashes[split] = {
            "latents_sha256": sha256(latent_path),
            "metadata_sha256": sha256(metadata_path),
            "rows": len(stored),
        }

    matrix = np.concatenate(vectors, axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    cache_dir = ANALYSIS / "cache/tsne"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{args.tag}_{scope}_raw_perp{args.perplexity:g}_iter{args.max_iter}_seed{args.seed}.npz")
    if args.reuse_cache and cache_path.exists():
        cached = np.load(cache_path)
        coordinates = cached["coordinates"]
        kl_divergence = float(cached["kl_divergence"])
        if coordinates.shape != (len(matrix), 2):
            raise ValueError("cached coordinate shape does not match current evidence")
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            early_exaggeration=12.0,
            learning_rate="auto",
            max_iter=args.max_iter,
            init="random",
            method="barnes_hut",
            angle=0.5,
            metric="euclidean",
            random_state=args.seed,
            n_jobs=-1,
            verbose=1,
        )
        coordinates = reducer.fit_transform(matrix).astype(np.float32)
        kl_divergence = float(reducer.kl_divergence_)
        np.savez_compressed(cache_path, coordinates=coordinates,
                            kl_divergence=np.float64(kl_divergence))

    split_values = np.asarray([record["split"] for record in records])
    colors = {"train": "#4C78A8", "valid": "#F58518", "test": "#E45756"}
    figure = go.Figure()
    for split in requested_splits:
        indices = np.flatnonzero(split_values == split)
        customdata = np.asarray([
            [records[i]["sample_id"], records[i]["row_index"],
             records[i]["target_item_id"], f"{norms[i]:.5f}",
             records[i].get("is_success"), records[i].get("top5_rank"),
             records[i].get("recommendation_loss")]
            for i in indices
        ], dtype=object)
        if args.color == "recommendation_loss":
            marker_color = np.asarray([
                np.nan if records[i].get("recommendation_loss") is None
                else float(records[i]["recommendation_loss"]) for i in indices])
            marker = {"size": 4, "opacity": 0.45, "color": marker_color,
                      "colorscale": "Viridis",
                      "colorbar": {"title": "Recommendation loss"},
                      "line": {"width": 0}}
        elif args.color == "norm":
            marker = {"size": 4, "opacity": 0.45, "color": norms[indices],
                      "colorscale": "Cividis", "colorbar": {"title": "Latent L2 norm"},
                      "line": {"width": 0}}
        else:
            marker = {"size": 4, "opacity": 0.45,
                      "color": colors.get(split, "#4C78A8"), "line": {"width": 0}}
        figure.add_trace(go.Scattergl(
            x=coordinates[indices, 0], y=coordinates[indices, 1],
            mode="markers", name=f"{split} ({len(indices):,})",
            marker=marker,
            customdata=customdata,
            hovertemplate=(
                f"split={split}<br>sample=%{{customdata[0]}}"
                "<br>row=%{customdata[1]}<br>target item=%{customdata[2]}"
                "<br>latent L2 norm=%{customdata[3]}"
                "<br>generation success=%{customdata[4]}"
                "<br>top-5 rank=%{customdata[5]}"
                "<br>recommendation loss=%{customdata[6]}"
                "<br>t-SNE 1=%{x:.3f}<br>t-SNE 2=%{y:.3f}<extra></extra>"),
        ))
    figure.update_layout(
        title={
            "text": (f"{args.tag} raw latent-reasoning t-SNE<br>"
                     f"<sup>All {len(matrix):,} raw {matrix.shape[1]}D vectors; "
                     "no preprocessing or subsampling</sup>"),
            "x": 0.5,
        },
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
        template="plotly_white", height=850, width=1200,
        hovermode="closest", legend_title="Dataset split",
    )

    figure_dir = ANALYSIS / "figures"
    result_dir = ANALYSIS / "results/latent_distribution" / args.tag
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    html_path = figure_dir / f"{args.tag}_{scope}_raw_latent_tsne_plotly.html"
    coordinate_path = result_dir / f"{scope}_raw_latent_tsne_coordinates.npz"
    figure.write_html(html_path, include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    np.savez_compressed(
        coordinate_path,
        coordinates=coordinates,
        split=split_values,
        row_index=np.asarray([record["row_index"] for record in records]),
        latent_norm=norms.astype(np.float32),
    )
    summary = {
        "evidence_root": str(args.evidence_root.resolve()),
        "rows": len(matrix), "input_dimension": matrix.shape[1],
        "input": "raw vectors; no PCA, centering, normalization, clustering, or subsampling",
        "split_counts": {split: int((split_values == split).sum())
                         for split in requested_splits},
        "method": "sklearn Barnes-Hut t-SNE", "metric": "euclidean",
        "perplexity": args.perplexity, "max_iter": args.max_iter,
        "seed": args.seed, "kl_divergence": kl_divergence,
        "color": args.color,
        "sources": source_hashes,
        "html": str(html_path.resolve()),
        "coordinates": str(coordinate_path.resolve()),
        "cache": str(cache_path.resolve()),
    }
    atomic_json(result_dir / f"{scope}_raw_latent_tsne_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
