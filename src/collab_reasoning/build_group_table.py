"""Coarse CF groups for the VRec-style verifier (k ~ 20 classes per dimension, as in VRec).

Spherical k-means over the normalized SASRec item embeddings of
``Musical_Instruments_sasrec_item.pt``, with the same procedure and output format as
the main repo's ``src/latent/build_sasrec_oracle_tables.py`` (cluster150 / cluster412):
``{"embeddings": centroid per item, "valid", "cluster_id", "k", "source", "kind"}``.

Usage: python src/collab_reasoning/build_group_table.py --k 20
"""

import argparse

import torch
import torch.nn.functional as F

import data as D


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    items = torch.load(D.DATA / "info" / "Musical_Instruments_sasrec_item.pt", weights_only=False)
    table, valid = items["embeddings"].float(), items["valid"].bool()
    centers, assign = spherical_kmeans(table[valid], a.k, seed=a.seed)
    cluster_table = torch.zeros_like(table)
    cluster_table[valid] = centers[assign]
    cluster_id = torch.full((len(table),), -1, dtype=torch.long)
    cluster_id[valid] = assign
    sizes = torch.bincount(assign, minlength=a.k)
    out = D.DATA / "info" / f"Musical_Instruments_sasrec_cluster{a.k}.pt"
    torch.save({"embeddings": cluster_table, "valid": valid, "cluster_id": cluster_id, "k": a.k,
                "source": items["source"], "kind": "sasrec_cluster"}, out)
    test = D.load_rows("test")
    tc = cluster_id[torch.as_tensor(test.target)]
    share = torch.bincount(tc[tc >= 0], minlength=a.k).float() / (tc >= 0).sum()
    entropy = float(-(share[share > 0] * share[share > 0].log2()).sum())
    print({"out": str(out), "k": a.k, "valid": int(valid.sum()), "nonempty": int((sizes > 0).sum()),
           "size_min/median/max": (int(sizes.min()), int(sizes.median()), int(sizes.max())),
           "test_target_group_entropy_bits": round(entropy, 2)})


if __name__ == "__main__":
    main()
