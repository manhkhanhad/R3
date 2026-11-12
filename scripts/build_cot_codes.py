"""Quantize teacher-CoT embeddings into discrete reasoning codes (k-means / residual k-means).

Usage:
  python scripts/build_cot_codes.py --emb <explanation_emb.pt> --train_csv <train.csv>
      --item_emb <items_qwen3emb.pt> --out_codes <codes.pt> --out_json <diag.json>

For each CoT view (full / post_evidence / feature) the embeddings are mean-centred
(removes the shared template direction), PCA-reduced, re-normalised and clustered with
spherical k-means.  Schemes: flat K in --ks, and a 2-level residual code (K1 x K2).

Diagnostics per (view, scheme):
  * usage: cluster sizes, code perplexity / K
  * NMI(code, target item / user / target genre-proxy / repeat flag) vs permutation chance
    (genre proxy = k-means-64 of item-title embeddings, since CDs has no category field)
  * code-only ranking: rows split 80/20; held-out target ranked by the per-code target
    popularity of the 80% part (backed off to global popularity).  HR/NDCG@1,3,5,10,20,
    vs global popularity.  This measures how much target information the TRUE code carries
    on its own (oracle code, no history), i.e. an information upper bound, not a model.
"""
import argparse
import ast
import json

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score as nmi

CUTS = (1, 3, 5, 10, 20)


def spherical_kmeans(Z, k, seed):
    km = faiss.Kmeans(Z.shape[1], k, niter=50, nredo=3, seed=seed, spherical=True, verbose=False)
    km.train(Z)
    return km.centroids.copy(), km.index.search(Z, 1)[1].ravel()


def project(X, dim):
    mu = X.mean(0)
    Xc = X - mu
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    pca = faiss.PCAMatrix(Xc.shape[1], dim)
    pca.train(Xc.astype("float32"))
    Z = pca.apply(Xc.astype("float32"))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    A = faiss.vector_to_array(pca.A).reshape(dim, Xc.shape[1])
    b = faiss.vector_to_array(pca.b)
    return Z, {"mean": mu.astype("float32"), "pca_A": A.astype("float32"), "pca_b": b.astype("float32")}


def residual_codes(Z, k1, k2, seed):
    c1, l1 = spherical_kmeans(Z, k1, seed)
    R = Z - c1[l1]
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    c2, l2 = spherical_kmeans(R.astype("float32"), k2, seed + 1)
    return (c1, c2), (l1, l2)


def nmi_vs_chance(code, y, rng):
    return {"nmi": float(nmi(y, code)), "chance": float(nmi(y, rng.permutation(code)))}


def code_only_ranking(code, target, n_items, rng, alpha=1.0):
    idx = rng.permutation(len(code))
    cut = int(0.8 * len(code))
    tr, te = idx[:cut], idx[cut:]
    glob = np.bincount(target[tr], minlength=n_items).astype(np.float64)
    glob_p = glob / glob.sum()
    tables = {}
    for c in np.unique(code[tr]):
        m = tr[code[tr] == c]
        counts = np.bincount(target[m], minlength=n_items).astype(np.float64)
        tables[c] = (counts + alpha * n_items * glob_p) / (counts.sum() + alpha * n_items)

    def metrics(score_fn):
        hr = {k: 0.0 for k in CUTS}
        nd = {k: 0.0 for k in CUTS}
        for i in te:
            s = score_fn(i)
            rank = int((s > s[target[i]]).sum())  # 0-based, ties favour target
            for k in CUTS:
                if rank < k:
                    hr[k] += 1
                    nd[k] += 1 / np.log2(rank + 2)
        n = len(te)
        return {f"HR@{k}": hr[k] / n for k in CUTS} | {f"NDCG@{k}": nd[k] / n for k in CUTS}

    return {"code": metrics(lambda i: tables.get(code[i], glob_p)), "popularity": metrics(lambda i: glob_p),
            "n_test": int(len(te))}


def usage(code, k):
    sizes = np.bincount(code, minlength=k)
    p = sizes[sizes > 0] / sizes.sum()
    return {"k": int(k), "used": int((sizes > 0).sum()), "min": int(sizes.min()), "max": int(sizes.max()),
            "perplexity_frac": float(np.exp(-(p * np.log(p)).sum()) / k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--item_emb", required=True)
    ap.add_argument("--out_codes", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--views", default="full,post_evidence,feature")
    ap.add_argument("--ks", default="64,256")
    ap.add_argument("--rq", default="256,256")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    faiss.omp_set_num_threads(16)

    d = torch.load(args.emb, weights_only=False)
    mask = d["mask"].numpy().astype(bool)
    df = pd.read_csv(args.train_csv)
    assert len(df) == len(mask), (len(df), len(mask))
    rows = np.flatnonzero(mask)
    target = df["item_id"].to_numpy()[rows]
    users = pd.factorize(df["user_id"])[0][rows]
    hist = df["history_item_id"].map(ast.literal_eval)
    repeat = np.array([t in h for t, h in zip(df["item_id"], hist)])[rows].astype(int)

    items = torch.load(args.item_emb, weights_only=False)
    items = (items if torch.is_tensor(items) else items.get("embeddings", next(
        v for v in items.values() if torch.is_tensor(v) and v.dim() == 2))).float().numpy()
    n_items = max(items.shape[0], int(df["item_id"].max()) + 1)
    it = items / np.linalg.norm(items, axis=1, keepdims=True)
    _, genre = spherical_kmeans(it.astype("float32"), 64, args.seed)
    genre_t = genre[target]

    out_codes = {"rows": torch.from_numpy(rows), "n_rows": len(df), "emb": args.emb, "views": {}}
    report = {"emb": args.emb, "n_explained": int(len(rows)), "repeat_rate": float(repeat.mean()),
              "n_items": int(n_items), "results": []}
    print(f"rows {len(rows)}/{len(df)}  items {n_items}  repeat-target rate {repeat.mean():.3f}", flush=True)

    schemes = [("flat", int(k)) for k in args.ks.split(",")] + [("rq", tuple(int(x) for x in args.rq.split(",")))]
    for view in args.views.split(","):
        X = d[view].float().numpy()[rows]
        Z, proj = project(X, args.pca_dim)
        out_codes["views"][view] = {"proj": {k: torch.from_numpy(v) for k, v in proj.items()}, "schemes": {}}
        for kind, k in schemes:
            if kind == "flat":
                cents, lab = spherical_kmeans(Z, k, args.seed)
                code, name, K = lab, f"flat{k}", k
                saved = {"labels": torch.from_numpy(lab), "centroids": torch.from_numpy(cents)}
            else:
                (c1, c2), (l1, l2) = residual_codes(Z, k[0], k[1], args.seed)
                code, name, K = l1 * k[1] + l2, f"rq{k[0]}x{k[1]}", k[0] * k[1]
                saved = {"labels": torch.from_numpy(np.stack([l1, l2], 1)),
                         "centroids": [torch.from_numpy(c1), torch.from_numpy(c2)]}
            out_codes["views"][view]["schemes"][name] = saved
            res = {"view": view, "scheme": name, "usage": usage(code, K),
                   "nmi_target": nmi_vs_chance(code, target, rng), "nmi_user": nmi_vs_chance(code, users, rng),
                   "nmi_genre": nmi_vs_chance(code, genre_t, rng), "nmi_repeat": nmi_vs_chance(code, repeat, rng),
                   "rank": code_only_ranking(code, target, n_items, np.random.default_rng(args.seed))}
            report["results"].append(res)
            r = res["rank"]
            print(f"{view:14s} {name:10s} used {res['usage']['used']:6d} perp {res['usage']['perplexity_frac']:.2f} "
                  f"NMI tgt {res['nmi_target']['nmi']:.3f}/{res['nmi_target']['chance']:.3f} "
                  f"genre {res['nmi_genre']['nmi']:.3f}/{res['nmi_genre']['chance']:.3f} "
                  f"user {res['nmi_user']['nmi']:.3f}/{res['nmi_user']['chance']:.3f} | "
                  + " ".join(f"HR@{c} {r['code'][f'HR@{c}']:.4f}" for c in CUTS)
                  + " | pop HR@10 " + f"{r['popularity']['HR@10']:.4f}", flush=True)

    torch.save(out_codes, args.out_codes)
    json.dump(report, open(args.out_json, "w"), indent=1)
    print("saved", args.out_codes, args.out_json)


if __name__ == "__main__":
    main()
