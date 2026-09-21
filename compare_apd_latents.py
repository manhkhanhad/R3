#!/usr/bin/env python3
"""Numerically compare two aligned APD latent tensors and jointly visualize them."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    result = torch.quantile(values, torch.tensor(levels))
    return {f"q{level:g}": float(value) for level, value in zip(levels, result)}


def load_source(directory: Path) -> tuple[torch.Tensor, list[dict], Path, Path]:
    latent_path = directory / "latents.pt"
    if not latent_path.exists():
        latent_path = directory / "thoughts.pt"
    metadata_path = directory / "metadata.jsonl"
    if metadata_path.exists():
        records = [json.loads(line) for line in metadata_path.open() if line.strip()]
    else:
        metadata_path = directory / "metadata.csv"
        records = pd.read_csv(metadata_path).to_dict(orient="records")
    tensor = torch.load(latent_path, map_location="cpu", weights_only=True).float()
    return tensor, records, latent_path, metadata_path


def row_index(record: dict) -> int:
    return int(record["row_index"] if "row_index" in record else record["sample_id"])


def target_item_id(record: dict) -> int:
    return int(record["target_item_id"] if "target_item_id" in record else record["target_item"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-extract", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--tag", default="latent_dev_apd_comparison")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    sources = (("pre_extract", args.pre_extract), ("local_apd", args.local))
    tensors, metadata, source_paths = {}, {}, {}
    for label, directory in sources:
        tensors[label], metadata[label], latent_path, metadata_path = load_source(directory)
        source_paths[label] = (latent_path, metadata_path)
        if tensors[label].ndim != 2 or len(tensors[label]) != len(metadata[label]):
            raise ValueError(f"invalid or misaligned source: {label}")
        if not torch.isfinite(tensors[label]).all():
            raise ValueError(f"non-finite vectors in {label}")
    a, b = tensors["pre_extract"], tensors["local_apd"]
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
    if any(row_index(x) != row_index(y) or target_item_id(x) != target_item_id(y)
           for x, y in zip(metadata["pre_extract"], metadata["local_apd"])):
        raise ValueError("metadata rows are not aligned")

    cosine = F.cosine_similarity(a, b, dim=1)
    l2 = (a - b).norm(dim=1)
    norm_a, norm_b = a.norm(dim=1), b.norm(dim=1)
    relative_l2 = l2 / ((norm_a + norm_b) / 2).clamp_min(1e-12)
    metrics = {
        "rows": len(a), "dimension": a.shape[1], "metadata_aligned": True,
        "exact_row_fraction": float((a == b).all(dim=1).float().mean()),
        "exact_element_fraction": float((a == b).float().mean()),
        "paired_cosine": {"mean": float(cosine.mean()), "std": float(cosine.std()),
                          "quantiles": quantiles(cosine)},
        "paired_l2": {"mean": float(l2.mean()), "std": float(l2.std()),
                      "quantiles": quantiles(l2)},
        "relative_l2": {"mean": float(relative_l2.mean()),
                        "std": float(relative_l2.std()),
                        "quantiles": quantiles(relative_l2)},
        "latent_norm": {
            "pre_extract_mean": float(norm_a.mean()), "pre_extract_std": float(norm_a.std()),
            "local_apd_mean": float(norm_b.mean()), "local_apd_std": float(norm_b.std()),
            "pearson": float(torch.corrcoef(torch.stack((norm_a, norm_b)))[0, 1]),
        },
        "source_sha256": {
            label: {"latents": sha256(source_paths[label][0]),
                    "metadata": sha256(source_paths[label][1])}
            for label, _ in sources
        },
        "source_summary": {
            label: json.load((path / "summary.json").open()) for label, path in sources
        },
    }

    matrix = torch.cat((a, b)).numpy()
    cache_dir = ROOT / "analysis/cache/tsne"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.tag}_joint_perp{args.perplexity:g}_iter{args.max_iter}_seed{args.seed}.npz"
    if args.reuse_cache and cache_path.exists():
        cached = np.load(cache_path)
        coordinates = cached["coordinates"]
        kl = float(cached["kl_divergence"])
    else:
        reducer = TSNE(n_components=2, perplexity=args.perplexity,
                       early_exaggeration=12.0, learning_rate="auto",
                       max_iter=args.max_iter, init="random", method="barnes_hut",
                       angle=0.5, metric="euclidean", random_state=args.seed,
                       n_jobs=-1, verbose=1)
        coordinates = reducer.fit_transform(matrix).astype(np.float32)
        kl = float(reducer.kl_divergence_)
        np.savez_compressed(cache_path, coordinates=coordinates,
                            kl_divergence=np.float64(kl))

    figure = go.Figure()
    palette = {"pre_extract": "#4C78A8", "local_apd": "#E45756"}
    for source_index, (label, _) in enumerate(sources):
        start, stop = source_index * len(a), (source_index + 1) * len(a)
        rows = metadata[label]
        custom = np.asarray([[row_index(r), target_item_id(r),
                              f"{float(tensors[label][i].norm()):.4f}",
                              f"{float(cosine[i]):.5f}"]
                             for i, r in enumerate(rows)], dtype=object)
        figure.add_trace(go.Scattergl(
            x=coordinates[start:stop, 0], y=coordinates[start:stop, 1],
            mode="markers", name=f"{label} ({len(a):,})",
            marker={"size": 3.5, "opacity": 0.38, "color": palette[label],
                    "line": {"width": 0}}, customdata=custom,
            hovertemplate=(f"source={label}<br>row=%{{customdata[0]}}"
                           "<br>target=%{customdata[1]}<br>norm=%{customdata[2]}"
                           "<br>paired cosine=%{customdata[3]}"
                           "<br>t-SNE 1=%{x:.3f}<br>t-SNE 2=%{y:.3f}<extra></extra>")))
    figure.update_layout(
        title={"text": (f"Pre-extracted vs local APD latent embeddings<br>"
                        f"<sup>Joint t-SNE of all {len(matrix):,} raw {a.shape[1]}D vectors; "
                        "no preprocessing or subsampling</sup>"), "x": 0.5},
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", template="plotly_white",
        width=1200, height=850, hovermode="closest", legend_title="Source")

    figure_dir = ROOT / "analysis/figures"
    result_dir = ROOT / "analysis/results/latent_distribution" / args.tag
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    html_path = figure_dir / f"{args.tag}_joint_tsne.html"
    coordinate_path = result_dir / "joint_tsne_coordinates.npz"
    figure.write_html(html_path, include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    np.savez_compressed(coordinate_path, coordinates=coordinates,
                        source=np.repeat(["pre_extract", "local_apd"], len(a)),
                        row_index=np.tile(np.arange(len(a)), 2))
    metrics["joint_tsne"] = {
        "method": "sklearn Barnes-Hut t-SNE", "rows": len(matrix),
        "perplexity": args.perplexity, "max_iter": args.max_iter,
        "seed": args.seed, "kl_divergence": kl,
        "input": "raw vectors; no PCA, centering, normalization, or subsampling",
        "html": str(html_path.resolve()), "coordinates": str(coordinate_path.resolve()),
        "cache": str(cache_path.resolve()),
    }
    temporary = result_dir / "comparison.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, result_dir / "comparison.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
