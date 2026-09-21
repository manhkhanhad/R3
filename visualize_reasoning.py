from pathlib import Path

import fire
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


def _load_evidence(
    latent_path: str,
    metadata_path: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load an extracted thoughts tensor and its aligned metadata CSV."""
    latent_file = Path(latent_path)
    metadata_file = Path(metadata_path)
    if not latent_file.is_file():
        raise FileNotFoundError(f"Latent tensor not found: {latent_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_file}")

    vectors = torch.load(latent_file, map_location="cpu", weights_only=True)
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor in {latent_file}, got "
            f"{type(vectors).__name__} with shape {getattr(vectors, 'shape', None)}"
        )
    vectors = vectors.float().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError(f"Latent tensor contains NaN or infinite values: {latent_file}")

    metadata = pd.read_csv(metadata_file)
    if len(metadata) != len(vectors):
        raise ValueError(
            "Latent/metadata row-count mismatch: "
            f"{len(vectors):,} vectors versus {len(metadata):,} metadata rows"
        )
    if "user_id" not in metadata.columns:
        raise ValueError(f"Metadata must contain a 'user_id' column: {metadata_file}")
    return metadata, vectors


def plot_latent_tsne(
    vectors: np.ndarray,
    user_ids: list[str] | None = None,
    output_path: str = "latent_tsne.html",
    method: str = "tsne",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    color_by_user: bool = False,
) -> None:
    if user_ids is not None and len(user_ids) != len(vectors):
        raise ValueError("user_ids must align one-to-one with latent vectors")
    method = method.lower()
    if method == "tsne":
        perplexity = min(perplexity, max(5, vectors.shape[0] // 3))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,
            init="random",
            random_state=random_state,
        )
        embedded = reducer.fit_transform(vectors)
        dim1_label, dim2_label, title = "t-SNE dim 1", "t-SNE dim 2", "t-SNE of Latent Evidence Vectors"
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        embedded = reducer.fit_transform(vectors)
        var = reducer.explained_variance_ratio_
        dim1_label = f"PC1 ({100*var[0]:.1f}% var)"
        dim2_label = f"PC2 ({100*var[1]:.1f}% var)"
        title = "PCA of Latent Evidence Vectors"
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'tsne' or 'pca'")

    frame = pd.DataFrame({dim1_label: embedded[:, 0], dim2_label: embedded[:, 1]})
    if user_ids is not None:
        frame["user_id"] = user_ids
    fig = px.scatter(
        frame,
        x=dim1_label,
        y=dim2_label,
        color="user_id" if color_by_user and user_ids is not None else None,
        hover_data={"user_id": True} if user_ids is not None else None,
        opacity=0.6,
        title=title,
    )
    fig.update_traces(marker=dict(size=6))
    fig.write_html(output_path)


def main(
    # latent_path: str = "analysis/evidence/CDs_and_Vinyl/train/thoughts.pt",
    latent_path: str = "analysis/evidence/CDs_and_Vinyl/sft_only/train/thoughts.pt",
    metadata_path: str = "analysis/evidence/CDs_and_Vinyl/train/metadata.csv",
    output_path: str = "analysis/figures/latent_tsne.html",
    method: str = "tsne",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    color_by_user: bool = False,
) -> None:
    """Create an interactive 2D projection plot with user IDs available on hover.

    `method` selects the dimensionality-reduction technique: "tsne" (default)
    or "pca". Color-by-user is intended for a small selected sample; for the
    full training split it creates too many categories, so hover-only is the
    default.
    """
    metadata, vectors = _load_evidence(latent_path, metadata_path)
    user_ids = metadata["user_id"].astype(str).tolist()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_latent_tsne(vectors, user_ids=user_ids, output_path=output_path,
                     method=method, perplexity=perplexity, n_iter=n_iter,
                     random_state=random_state, color_by_user=color_by_user)


if __name__ == "__main__":
    fire.Fire(main)
