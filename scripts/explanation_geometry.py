"""Geometry and cluster structure of teacher-explanation embeddings.

Usage: python scripts/explanation_geometry.py <embeddings.pt> <train.csv> <out.json>
Reports anisotropy, same-user/same-target vs random cosine, PCA spectrum, a spherical
k-means sweep (on centred vectors, PCA-128) with silhouette / Davies-Bouldin /
Calinski-Harabasz and NMI to target/user against a chance baseline, and HDBSCAN.
"""
import json
import sys

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score as nmi, silhouette_score)

KS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def pair_cos(X, groups, rng):
    pairs = []
    for members in pd.Series(np.arange(len(X))).groupby(groups).apply(list):
        if len(members) > 1:
            members = rng.permutation(members)
            pairs += [(members[i], members[i + 1]) for i in range(0, len(members) - 1, 2)]
    pairs = np.array(pairs)
    return float((X[pairs[:, 0]] * X[pairs[:, 1]]).sum(1).mean())


def analyse(X, users, items, rng):
    r = {}
    mu = X.mean(0)
    r["cos_to_mean"] = float((X @ mu / np.linalg.norm(mu)).mean())
    a, b = rng.integers(0, len(X), (2, 20000))
    r["random_pair_cos"] = float((X[a] * X[b]).sum(1).mean())
    r["same_user_cos"] = pair_cos(X, users, rng)
    r["same_target_cos"] = pair_cos(X, items, rng)
    Xc = X - mu
    sv = np.linalg.svd(Xc[rng.permutation(len(X))[:20000]], compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(ev)
    r.update(pc1_frac=float(ev[0]), pcs_50=int(np.searchsorted(cum, .5) + 1),
             pcs_90=int(np.searchsorted(cum, .9) + 1),
             eff_rank=float(np.exp(-(ev * np.log(ev + 1e-12)).sum())), participation=float(1 / (ev ** 2).sum()))
    Xn = (Xc / np.linalg.norm(Xc, axis=1, keepdims=True)).astype("float32")
    pca = faiss.PCAMatrix(Xn.shape[1], 128)
    pca.train(Xn)
    Z = pca.apply(Xn)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    sub = rng.permutation(len(Z))[:12000]
    r["kmeans"] = []
    for k in KS:
        km = faiss.Kmeans(128, k, niter=40, nredo=2, seed=1, spherical=True, verbose=False)
        km.train(Z)
        lab = km.index.search(Z, 1)[1].ravel()
        chance = rng.permutation(lab)
        r["kmeans"].append({
            "k": k, "silhouette": float(silhouette_score(Z[sub], lab[sub], metric="cosine")),
            "davies_bouldin": float(davies_bouldin_score(Z[sub], lab[sub])),
            "calinski": float(calinski_harabasz_score(Z[sub], lab[sub])),
            "nmi_target": float(nmi(items, lab)), "nmi_target_chance": float(nmi(items, chance)),
            "nmi_user": float(nmi(users, lab)), "nmi_user_chance": float(nmi(users, chance)),
            "min_size": int(np.bincount(lab, minlength=k).min())})
    labels = HDBSCAN(min_cluster_size=50, min_samples=10).fit(Z[sub[:10000], :32]).labels_
    r["hdbscan"] = {"n_clusters": int(labels.max() + 1), "noise_frac": float((labels < 0).mean())}
    return r


def main():
    path, train_csv, out = sys.argv[1:4]
    d = torch.load(path, weights_only=False)
    train = pd.read_csv(train_csv)
    rows = np.nonzero(d["mask"].numpy())[0]
    users, items = train.user_id.to_numpy()[rows], train.item_id.to_numpy()[rows]
    result = {"source": path, "dim": int(d["feature"].size(1)), "n": int(len(rows))}
    for view in d["views"]:
        X = d[view][d["mask"]].float().numpy()
        result[view] = analyse(X, users, items, np.random.default_rng(0))
        print(view, json.dumps({k: v for k, v in result[view].items() if k != "kmeans"}), flush=True)
        for row in result[view]["kmeans"]:
            print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
