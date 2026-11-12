"""Build per-item label tables for the multi-aspect R3 model.

Aspects (each maps an item id to a class index, -1 = unknown):

* ``item``     : the item itself (identity, n_items classes)
* ``category`` : leaf category path (from the gte category file)
* ``sid_l1``   : level-1 semantic id, k-means over title embeddings
* ``sid_l2``   : full 2-level semantic id, residual k-means inside level 1
                 (class = l1 * k2 + l2)

Output: a .pt with ``names``, ``codes`` [K, n_items] and ``num_classes``.
"""

import fire
import numpy as np
import torch


def kmeans(x, k, seed, iters=50):
    """Cosine-free Euclidean k-means (Lloyd), k-means++ init."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.stack(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.stack(centers)
    for _ in range(iters):
        assign = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([x[assign == j].mean(0) if (assign == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return assign, c


def main(
    title_embeddings: str = "data/Musical_Instruments/info/Musical_Instruments_qwen3emb_1536.pt",
    category_file: str = "data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt",
    output: str = "data/Musical_Instruments/info/Musical_Instruments_aspects.pt",
    k1: int = 32,
    k2: int = 16,
    pca_dim: int = 64,
    seed: int = 0,
):
    emb = torch.load(title_embeddings, map_location="cpu", weights_only=False)["embeddings"].float()
    emb = torch.nn.functional.normalize(emb, dim=-1).numpy()
    n = emb.shape[0]
    # PCA to keep k-means fast and less noisy
    x = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    x = x @ vt[:pca_dim].T

    l1, c1 = kmeans(x, k1, seed)
    residual = x - c1[l1]
    l2 = np.zeros(n, dtype=np.int64)
    for j in range(k1):
        idx = np.nonzero(l1 == j)[0]
        if len(idx) >= k2:
            l2[idx], _ = kmeans(residual[idx], k2, seed + 1 + j)
        else:
            l2[idx] = np.arange(len(idx))

    cat = torch.load(category_file, map_location="cpu", weights_only=True)
    paths = list(cat["paths"])
    valid = cat["valid"].bool().tolist()
    uniq = sorted({p for p, v in zip(paths, valid) if v})
    cidx = {p: i for i, p in enumerate(uniq)}
    category = torch.tensor([cidx[p] if v else -1 for p, v in zip(paths, valid)])
    assert len(paths) == n, (len(paths), n)

    codes = torch.stack([
        torch.arange(n),
        category,
        torch.from_numpy(l1).long(),
        torch.from_numpy(l1 * k2 + l2).long(),
    ])
    out = {"names": ["item", "category", "sid_l1", "sid_l2"],
           "codes": codes,
           "num_classes": [n, len(uniq), k1, k1 * k2]}
    torch.save(out, output)
    occupied = len(np.unique(l1 * k2 + l2))
    print({"n_items": n, "categories": len(uniq), "sid_l1": k1, "sid_l2_classes": k1 * k2,
           "sid_l2_occupied": occupied, "largest_l1_cluster": int(np.bincount(l1).max()), "output": output})


if __name__ == "__main__":
    fire.Fire(main)
