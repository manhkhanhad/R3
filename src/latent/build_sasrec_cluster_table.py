"""SASRec cluster labels and centroids for the staged-refinement R3 (staged_refine_attention.py).

Spherical k-means over the normalized SASRec item embeddings
(``*_sasrec_item.pt`` from build_sasrec_oracle_tables.py, indexed by item_id).
Output: ``cluster_id`` [n_items] (-1 = item not in SASRec's catalog),
``centroids`` [k, d] (unit norm), ``k``.
"""

import fire
import torch
import torch.nn.functional as F


def spherical_kmeans(x, k, iters=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (x @ centers.T).argmax(1)
        new = torch.zeros_like(centers).index_add_(0, assign, x)
        empty = new.norm(dim=1) == 0
        new[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=g)]
        new = F.normalize(new, dim=1)
        if torch.allclose(new, centers):
            break
        centers = new
    return F.normalize(centers, dim=1), (x @ centers.T).argmax(1)


def main(
    item_table: str = "data/Musical_Instruments/info/Musical_Instruments_sasrec_item.pt",
    output: str = "data/Musical_Instruments/info/Musical_Instruments_sasrec_k60_refine.pt",
    k: int = 60,
    seed: int = 0,
):
    t = torch.load(item_table, map_location="cpu", weights_only=False)
    emb, valid = F.normalize(t["embeddings"].float(), dim=-1), t["valid"].bool()
    centers, assign = spherical_kmeans(emb[valid], k, seed=seed)
    cluster_id = torch.full((len(emb),), -1, dtype=torch.long)
    cluster_id[valid] = assign
    sizes = torch.bincount(assign, minlength=k)
    torch.save({"cluster_id": cluster_id, "centroids": centers, "k": k, "source": item_table, "seed": seed}, output)
    print({"items": len(emb), "valid": int(valid.sum()), "k": k, "nonempty": int((sizes > 0).sum()),
           "size_min/median/max": (int(sizes.min()), int(sizes.median()), int(sizes.max())), "output": output})


if __name__ == "__main__":
    fire.Fire(main)
