#!/usr/bin/env python3
"""Complementarity screen, fused/residual CF scorers, group predictor and candidate tables.

Steps (document/collaborative_reasoning_experiments.md):

1. Run every source in cf_sources.py on train/valid/test (top-100 per row, cached).
2. **Screen (idea 1)**: on the test split, for every source report HR@10/NDCG@10,
   hits on the rows R3 v1 misses, the R3 union upper bound and the overlap with
   R3 hits.  Train-row hit rates (leave-user-out) are reported next to valid/test
   to expose the in-period advantage of training-row candidates.
3. **Fused scorer**: a listwise MLP over per-source rank features, trained on the
   validation split (out-of-period for every statistic, like test).
4. **Residual scorer (idea 4)**: the same model trained only on validation rows
   R3 v1 misses, with R3's (wrong) top-10 added to the pool as negatives.  It
   cannot see R3 at inference; it learns which source features point to targets
   R3 does not find (titlesim is the proxy for R3-like picks).
5. **Group predictor (idea 5)**: p(CF cluster k=150) = mass of the fused item
   distribution per cluster.
6. Candidate tables (top-10, history excluded) and group-probability tables for
   every split.  Training-row tables are *calibrated*: with probability q a row's
   list (vector) is replaced by another random training row's, with q chosen so
   the training hit rate matches the held-out validation hit rate.  This stops
   the LLM from learning to trust candidates more than they deserve at test time.

Usage: python build_candidates.py --fold_manifest <dir>/manifest.json --valid_preds <dir> [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import cf_sources as S
import data as D

POOL_PER_SOURCE = 50
FEATS_PER_SOURCE = 3
TOP_N = 100
CAND_K = 10
CLUSTER_FILE = D.DATA / "info" / "Musical_Instruments_sasrec_cluster150.pt"
TABLE_DIR = D.DATA / "info" / "collab_reasoning"
CACHE = Path("/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/collab_reasoning/sources")


# --------------------------------------------------------------------------- sources
def run_all_sources(ctx, splits, limit, tag):
    out = {}
    for split in splits:
        for src in S.SOURCES:
            path = CACHE / tag / f"{split}_{src}.npz"
            if path.exists():
                z = np.load(path)
                out[(split, src)] = (z["ids"], z["scores"], z["rank"])
                continue
            print(f"source {src} on {split}", flush=True)
            lim = limit if split != "train" else (limit * 4 if limit > 0 else -1)
            ids, sc, rank = S.run_source(ctx, src, split, top_n=TOP_N, limit=lim)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, ids=ids, scores=sc, rank=rank)
            out[(split, src)] = (ids, sc, rank)
    return out


# --------------------------------------------------------------------------- features
def row_features(src_out, split, rows_idx, extra_items=None):
    """Pool (union of per-source top-50) and per-source rank features for each row.

    Returns items [B, P] (-1 padded) and features [B, P, F].
    """
    pools = []
    for b, i in enumerate(rows_idx):
        pool = []
        seen = set()
        for src in S.SOURCES:
            for it in src_out[(split, src)][0][i, :POOL_PER_SOURCE]:
                if it not in seen:
                    seen.add(int(it)); pool.append(int(it))
        if extra_items is not None:
            for it in extra_items[b]:
                if it >= 0 and it not in seen:
                    seen.add(int(it)); pool.append(int(it))
        pools.append(pool)
    P = max(map(len, pools))
    items = np.full((len(rows_idx), P), -1, np.int64)
    for b, pool in enumerate(pools):
        items[b, :len(pool)] = pool
    B = len(rows_idx)
    F = np.zeros((B, P, FEATS_PER_SOURCE * len(S.SOURCES)), np.float32)
    n_items = int(max(items.max(), max(src_out[(split, s)][0].max() for s in S.SOURCES))) + 2
    safe = np.where(items >= 0, items, n_items - 1)  # padding reads a never-ranked column
    brow = np.arange(B)[:, None]
    for k, src in enumerate(S.SOURCES):
        ids, sc = src_out[(split, src)][0][rows_idx].astype(np.int64), src_out[(split, src)][1][rows_idx]
        z = np.clip((sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-6), -3, 6)
        pos = np.full((B, n_items), -1, np.int32)
        pos[brow, ids] = np.arange(ids.shape[1])[None, :]
        zz = np.full((B, n_items), -3.0, np.float32)
        zz[brow, ids] = z
        p = pos[brow, safe]
        ranked = p >= 0
        F[:, :, 3 * k] = np.where(ranked, 1.0 / np.maximum(p + 1.0, 1.0), 0.0)
        F[:, :, 3 * k + 1] = (ranked & (p < 10)).astype(np.float32)
        F[:, :, 3 * k + 2] = zz[brow, safe]
    F[items < 0] = 0.0
    return items, F


class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_scorer(items, F, target, seed=0, epochs=60):
    """Listwise softmax over the pool; 20% of rows held out for early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    has = (items == target[:, None]).any(1)
    idx = np.nonzero(has)[0]
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, ho = idx[:cut], idx[cut:]
    X = torch.from_numpy(F)
    M = torch.from_numpy(items >= 0)
    Y = torch.from_numpy((items == target[:, None]).argmax(1))
    model = Scorer(F.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(tr))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            logits = model(X[b]).masked_fill(~M[b], -1e9)
            loss = nn.functional.cross_entropy(logits, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(X[ho]).masked_fill(~M[ho], -1e9)
            rank = (lg > lg.gather(1, Y[ho][:, None])).sum(1)
            hr = float((rank < 10).float().mean())
        if hr > best:
            best, best_state, wait = hr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6:
                break
    model.load_state_dict(best_state)
    return model, {"heldout_hr10_within_pool": best, "train_rows": int(len(tr)), "heldout_rows": int(len(ho)),
                   "pool_recall": float(has.mean()), "heldout_idx": ho.tolist()}


def apply_scorer(model, src_out, split, n, batch=1024):
    """Top-10 candidates and the softmax item distribution (sparse, over the pool) per row."""
    cands = np.full((n, CAND_K), -1, np.int64)
    dists = []
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        items, F = row_features(src_out, split, idx)
        with torch.no_grad():
            lg = model(torch.from_numpy(F)).masked_fill(torch.from_numpy(items < 0), -1e9)
        order = torch.argsort(lg, 1, descending=True)[:, :CAND_K].numpy()
        cands[idx] = np.take_along_axis(items, order, 1)
        p = torch.softmax(lg, 1).numpy()
        dists += [(items[b], p[b]) for b in range(len(idx))]
    return cands, dists


# --------------------------------------------------------------------------- metrics
def canonical(items, titles, title_id):
    """Map item ids to the first id with the same title (duplicate catalog titles)."""
    canon = np.array([title_id.get(t, i) for i, t in enumerate(titles)] + [-1], dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    return canon[np.where(items >= 0, items, len(titles))]


def screen_table(name, topk, target, r3_hit, rank=None):
    hit10 = D.hits_at(topk, target, 10)
    res = {
        "hr10": float(hit10.mean()), "ndcg10": float(D.ndcg_at(topk, target, 10).mean()),
        "hr10_on_r3_miss": float(hit10[~r3_hit].mean()),
        "union_hr10": float((hit10 | r3_hit).mean()),
        "overlap_r3_hit_given_hit": float(r3_hit[hit10].mean()) if hit10.any() else 0.0,
    }
    if topk.shape[1] >= 50:
        res["hr20_on_r3_miss"] = float(D.hits_at(topk, target, 20)[~r3_hit].mean())
        res["hr50_on_r3_miss"] = float(D.hits_at(topk, target, 50)[~r3_hit].mean())
    return res


def calibrate(train_table, train_hit, target_rate, seed=0):
    """Replace a fraction q of training rows by another row's entry so the hit rate matches."""
    rate = float(train_hit.mean())
    q = float(np.clip(1.0 - target_rate / max(rate, 1e-9), 0.0, 1.0))
    rng = np.random.default_rng(seed)
    out = train_table.copy()
    swap = rng.random(len(out)) < q
    donors = rng.integers(0, len(out), size=len(out))
    out[swap] = train_table[donors[swap]]
    return out, {"train_rate_before": rate, "target_rate": float(target_rate), "q": q,
                 "rows_replaced": int(swap.sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold_manifest", type=Path, required=True)
    p.add_argument("--valid_preds", type=Path, default=None, help="R3 v1 predictions on the valid split")
    p.add_argument("--limit", type=int, default=-1, help="smoke mode: rows per eval split")
    p.add_argument("--tag", default="full")
    a = p.parse_args()
    out_dir = D.RESULTS / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = D.load_all()
    manifest = json.load(a.fold_manifest.open())
    fold_models = {int(k): v["path"] for k, v in manifest["models"].items()}
    ctx = S.Context(rows, fold_models=fold_models, folds=manifest["folds"])
    src_out = run_all_sources(ctx, D.SPLITS, a.limit, a.tag)
    n = {s: len(src_out[(s, S.SOURCES[0])][0]) for s in D.SPLITS}

    titles = D.item_titles() + [""] * (ctx.n - len(D.item_titles()))
    title_id = {}
    for i, t in enumerate(titles):
        title_id.setdefault(t, i)
    tgt = {s: canonical(rows[s].target[:n[s]], titles, title_id) for s in D.SPLITS}
    test_r3 = canonical(np.array(D.load_r3_predictions(D.V1_PREDICTIONS, rows["test"]))[:n["test"]], titles, title_id)
    r3_hit = {"test": D.hits_at(test_r3, tgt["test"], 10)}
    valid_r3 = None
    if a.valid_preds is not None and (a.valid_preds / "predictions.json").exists():
        valid_r3 = canonical(np.array(D.load_r3_predictions(a.valid_preds, rows["valid"]))[:n["valid"]], titles, title_id)
        r3_hit["valid"] = D.hits_at(valid_r3, tgt["valid"], 10)

    report = {"n": n, "r3_v1": {"test_hr10": float(r3_hit["test"].mean())}, "sources": {}}
    if "valid" in r3_hit:
        report["r3_v1"]["valid_hr10"] = float(r3_hit["valid"].mean())
    for src in S.SOURCES:
        ids = {s: canonical(src_out[(s, src)][0].astype(np.int64), titles, title_id) for s in D.SPLITS}
        entry = {"test": screen_table(src, ids["test"], tgt["test"], r3_hit["test"])}
        entry["train_hr10_leave_user_out"] = float(D.hits_at(ids["train"], tgt["train"], 10).mean())
        entry["valid_hr10"] = float(D.hits_at(ids["valid"], tgt["valid"], 10).mean())
        report["sources"][src] = entry
        print(src, json.dumps(entry["test"]), "train", round(entry["train_hr10_leave_user_out"], 4),
              "valid", round(entry["valid_hr10"], 4), flush=True)
    # in-sample SASRec on training rows: how much the baseline model has already fit them
    if not a.limit > 0:
        ctx_full = S.Context(rows, fold_models={f: str(D.SASREC_FULL) for f in fold_models}, folds=manifest["folds"])
        ids, _, _ = S.run_source(ctx_full, "sasrec", "train", top_n=10, limit=20000)
        report["sources"]["sasrec"]["train_hr10_in_sample_full_model"] = float(
            D.hits_at(ids.astype(np.int64), rows["train"].target[:20000], 10).mean())

    # ---- fused and residual scorers, trained on validation rows
    vidx = np.arange(n["valid"])
    items_v, F_v = row_features(src_out, "valid", vidx)
    fused, info_f = train_scorer(items_v, F_v, rows["valid"].target[:n["valid"]])
    scorers = {"fused": (fused, info_f)}
    if valid_r3 is not None:
        miss = ~r3_hit["valid"]
        midx = vidx[miss]
        items_m, F_m = row_features(src_out, "valid", midx, extra_items=valid_r3[miss])
        residual, info_r = train_scorer(items_m, F_m, rows["valid"].target[:n["valid"]][miss], seed=1)
        scorers["residual"] = (residual, info_r)
    else:
        print("no valid R3 predictions: residual scorer skipped", flush=True)

    clusters = torch.load(CLUSTER_FILE, weights_only=False)
    cid = clusters["cluster_id"].numpy()
    k_clusters = int(clusters["k"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables_meta = {}
    for name, (model, info) in scorers.items():
        heldout = np.array(info.pop("heldout_idx"), dtype=np.int64)
        cand, dist = {}, {}
        for s in D.SPLITS:
            cand[s], dist[s] = apply_scorer(model, src_out, s, n[s])
        # target hit rate for calibration: the scorer's own held-out validation rows
        ho = heldout if name == "fused" else vidx[~r3_hit["valid"]][heldout]
        valid_rate = float(D.hits_at(cand["valid"][ho], rows["valid"].target[ho], 10).mean())
        train_hit = D.hits_at(cand["train"], rows["train"].target[:n["train"]], 10)
        cal, cal_info = calibrate(cand["train"], train_hit, valid_rate)
        cal_info["valid_heldout_hr10"] = valid_rate
        cand_c = {s: canonical(cand[s], titles, title_id) for s in D.SPLITS}
        report["sources"][name] = {
            "test": screen_table(name, cand_c["test"], tgt["test"], r3_hit["test"]),
            "train_hr10_leave_user_out": float(train_hit.mean()), "calibration": cal_info, "scorer": info,
        }
        print(name, json.dumps(report["sources"][name]), flush=True)
        path = TABLE_DIR / f"candidates_{name}_{a.tag}.pt"
        torch.save({"train": torch.from_numpy(cal), "train_uncalibrated": torch.from_numpy(cand["train"]),
                    "valid": torch.from_numpy(cand["valid"]), "test": torch.from_numpy(cand["test"]),
                    "k": CAND_K, "scorer": name, "calibration": cal_info}, path)
        tables_meta[f"candidates_{name}"] = str(path)

        if name == "fused":
            # group predictor: cluster mass of the fused item distribution
            group, top1 = {}, {}
            for s in D.SPLITS:
                g = np.zeros((n[s], k_clusters), np.float32)
                for b, (its, pr) in enumerate(dist[s]):
                    ok = its >= 0
                    c = cid[its[ok]]
                    v = c >= 0
                    np.add.at(g[b], c[v], pr[ok][v])
                g /= np.maximum(g.sum(1, keepdims=True), 1e-9)
                group[s] = g
                top1[s] = g.argmax(1)
            tc = {s: cid[rows[s].target[:n[s]]] for s in D.SPLITS}
            last_c = {s: np.array([cid[h[-1]] for h in rows[s].history[:n[s]]]) for s in D.SPLITS}
            mode_c = {s: np.array([np.bincount(cid[h][cid[h] >= 0], minlength=k_clusters).argmax()
                                   if (cid[h] >= 0).any() else -1 for h in rows[s].history[:n[s]]]) for s in D.SPLITS}
            top5 = np.argsort(-group["test"], 1)[:, :5]
            ok_t = tc["test"] >= 0
            gt = {
                "test_top1_acc": float((top1["test"] == tc["test"])[ok_t].mean()),
                "test_top5_acc": float((top5 == tc["test"][:, None]).any(1)[ok_t].mean()),
                "test_last_item_cluster_acc": float((last_c["test"] == tc["test"])[ok_t].mean()),
                "test_history_mode_cluster_acc": float((mode_c["test"] == tc["test"])[ok_t].mean()),
                "test_majority_cluster_acc": float((tc["test"][ok_t] == np.bincount(tc["train"][tc["train"] >= 0]).argmax()).mean()),
                "valid_heldout_top1_acc": float((top1["valid"][heldout] == tc["valid"][heldout]).mean()),
                "train_top1_acc_before_calibration": float((top1["train"] == tc["train"]).mean()),
            }
            g_train, g_cal = calibrate(group["train"], top1["train"] == tc["train"], gt["valid_heldout_top1_acc"], seed=2)
            gt["calibration"] = g_cal
            report["group_predictor"] = gt
            print("group", json.dumps(gt), flush=True)
            rep_item = np.full(k_clusters, -1, np.int64)
            for it in range(len(cid)):
                if cid[it] >= 0 and rep_item[cid[it]] < 0:
                    rep_item[cid[it]] = it
            path = TABLE_DIR / f"group_pred150_{a.tag}.pt"
            torch.save({"train": torch.from_numpy(g_train).half(), "valid": torch.from_numpy(group["valid"]).half(),
                        "test": torch.from_numpy(group["test"]).half(), "k": k_clusters,
                        "test_top1_rep_item": torch.from_numpy(rep_item[top1["test"]]),
                        "calibration": g_cal}, path)
            tables_meta["group_pred150"] = str(path)

    report["tables"] = tables_meta
    with (out_dir / "screen.json").open("w") as f:
        json.dump(report, f, indent=2)
    lines = ["| source | test HR@10 | NDCG@10 | HR@10 on R3 misses | HR@50 on R3 misses | union with R3 HR@10 | P(R3 hit given source hit) | train HR@10 (leave-user-out) | valid HR@10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in report["sources"].items():
        t = e["test"]
        lines.append(f"| {name} | {t['hr10']:.4f} | {t['ndcg10']:.4f} | {t['hr10_on_r3_miss']:.4f} | "
                     f"{t.get('hr50_on_r3_miss', float('nan')):.4f} | {t['union_hr10']:.4f} | "
                     f"{t['overlap_r3_hit_given_hit']:.3f} | {e['train_hr10_leave_user_out']:.4f} | {e.get('valid_hr10', float('nan')):.4f} |")
    (out_dir / "screen.md").write_text(f"R3 v1 test HR@10 = {report['r3_v1']['test_hr10']:.4f}\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
