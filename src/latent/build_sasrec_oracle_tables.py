"""Build per-item SASRec lookup tables for the CF oracle tests (Experiment 1 extension).

Writes two tables indexed by the dataset's item_id, in the same format as the
category table ({"embeddings": [n_items, d], "valid": [n_items], ...}) so they
can be used with ``--side_info_kind oracle_category --side_info_file <table>``:

* ``*_sasrec_item.pt``: each item's own normalized SASRec item embedding
  (item-level CF oracle, the CF ceiling);
* ``*_sasrec_cluster412.pt``: the normalized centroid of the item's spherical
  k-means cluster over SASRec item embeddings, k = 412 = number of category
  paths (CF oracle at the same granularity as the category oracle).

Items outside the SASRec training catalog are marked invalid (zero side vector).
"""

import ast
import os

import fire
import pandas as pd
import torch
import torch.nn.functional as F

from sasrec_pair_features import item_embeddings, load_sasrec


def item_asins(data_dir, prefix):
    mapping = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(os.path.join(data_dir, split, f"{prefix}.csv"))
        for i, a in zip(frame.item_id, frame.item_asin):
            mapping.setdefault(int(i), a)
        for ids, asins in zip(frame.history_item_id, frame.item_asins):
            for i, a in zip(ast.literal_eval(ids), ast.literal_eval(asins)):
                mapping.setdefault(int(i), a)
    return mapping


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
    checkpoint="/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/Musical_Instruments_5_2015-1-2018-11/models/SASRec/2026-09-26_00-07-12-933295.pkl",
    data_dir="data/Musical_Instruments",
    prefix="Musical_Instruments_5_2015-1-2018-11",
    k=412,
    seed=0,
):
    rec = load_sasrec(checkpoint)
    emb = torch.from_numpy(item_embeddings(rec))  # normalized, SASRec index order
    mapping = item_asins(data_dir, prefix)
    n = max(mapping) + 1
    table = torch.zeros(n, emb.shape[1])
    valid = torch.zeros(n, dtype=torch.bool)
    for item_id, asin in mapping.items():
        idx = rec.iid_map.get(asin)
        if idx is not None and idx < rec.num_items:
            table[item_id] = emb[idx]
            valid[item_id] = True
    out = os.path.join(data_dir, "info")
    torch.save({"embeddings": table, "valid": valid, "source": checkpoint, "kind": "sasrec_item"},
               os.path.join(out, "Musical_Instruments_sasrec_item.pt"))

    centers, assign = spherical_kmeans(table[valid], k, seed=seed)
    cluster_table = torch.zeros_like(table)
    cluster_table[valid] = centers[assign]
    cluster_id = torch.full((n,), -1, dtype=torch.long)
    cluster_id[valid] = assign
    sizes = torch.bincount(assign, minlength=k)
    torch.save({"embeddings": cluster_table, "valid": valid, "cluster_id": cluster_id, "k": k,
                "source": checkpoint, "kind": "sasrec_cluster"},
               os.path.join(out, f"Musical_Instruments_sasrec_cluster{k}.pt"))
    print({"items": n, "valid": int(valid.sum()), "k": k, "nonempty_clusters": int((sizes > 0).sum()),
           "cluster_size_min/median/max": (int(sizes.min()), int(sizes.median()), int(sizes.max()))})


if __name__ == "__main__":
    fire.Fire(main)
