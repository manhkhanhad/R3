#!/usr/bin/env python3
"""Cluster-sequence predictor: p(target CF cluster | history clusters).

The input is only the CF-cluster id (k = 150, spherical k-means over SASRec item
embeddings) of each history item.  The question is whether reading the history at
cluster granularity predicts the target's cluster on the rows R3 misses better
than the item-level fused predictor (build_candidates.py) does: pooling sparse
item transitions into 150 groups gives denser statistics, which may help rows
with rare items.

Models (trained on the training split; selected on validation):
  markov      cluster-level first-order transitions from the last 3 history clusters
  gru10       GRU over the clusters of the 10-item prompt history
  grufull     GRU over the clusters of the user's whole past (beyond the prompt window)
  mix         average of grufull and the fused predictor's cluster distribution

Accuracy is reported on the test rows R3 v1 hits and misses.  Writes per-split
probability tables (same format as group_pred150_*.pt) for grufull and mix.

Usage: python cluster_seq_predictor.py [--tag full] [--epochs 8]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import data as D

CLUSTER_FILE = D.DATA / "info" / "Musical_Instruments_sasrec_cluster150.pt"
TABLE_DIR = D.DATA / "info" / "collab_reasoning"
MAX_FULL = 50


def cluster_sequences(rows, cid, k, full):
    seqs = []
    for i in range(len(rows)):
        items = rows.full_past[i][-MAX_FULL:] if full else rows.history[i]
        c = [int(cid[x]) for x in items]
        seqs.append([x if x >= 0 else k for x in c])  # k = unknown-cluster token
    return seqs


def pad(seqs, k, length):
    out = np.full((len(seqs), length), k + 1, np.int64)  # k + 1 = padding
    lens = np.zeros(len(seqs), np.int64)
    for i, s in enumerate(seqs):
        s = s[-length:]
        out[i, length - len(s):] = s
        lens[i] = len(s)
    return torch.from_numpy(out), torch.from_numpy(lens)


class ClusterGRU(nn.Module):
    def __init__(self, k, dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(k + 2, dim, padding_idx=k + 1)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, k)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h[:, -1])  # sequences are left-padded: the last step is the newest item


def train_gru(xtr, ytr, xva, yva, k, epochs, seed=0):
    torch.manual_seed(seed)
    model = ClusterGRU(k)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    keep_tr, keep_va = ytr >= 0, yva >= 0
    xtr, ytr, xva, yva = xtr[keep_tr], ytr[keep_tr], xva[keep_va], yva[keep_va]
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for s in range(0, len(perm), 512):
            b = perm[s:s + 512]
            loss = nn.functional.cross_entropy(model(xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc5 = float((model(xva).topk(5, 1).indices == yva[:, None]).any(1).float().mean())
        print(f"  epoch {ep}: valid top-5 {acc5:.4f}", flush=True)
        if acc5 > best:
            best, best_state = acc5, {k_: v.clone() for k_, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def predict(model, x):
    with torch.no_grad():
        return torch.softmax(torch.cat([model(x[s:s + 4096]) for s in range(0, len(x), 4096)]), 1).numpy()


def markov_probs(train_seqs, seqs, k):
    T = np.ones((k + 1, k), np.float64)  # add-one smoothing; row k = unknown cluster
    for s in train_seqs:
        for a, b in zip(s[:-1], s[1:]):
            if b < k:
                T[a, b] += 1
    T /= T.sum(1, keepdims=True)
    out = np.zeros((len(seqs), k))
    for i, s in enumerate(seqs):
        for w, c in zip((1.0, 0.5, 0.25), reversed(s[-3:])):
            out[i] += w * T[c]
    return out / out.sum(1, keepdims=True)


def accuracy(probs, target, mask):
    ok = mask & (target >= 0)
    top = np.argsort(-probs[ok], 1)[:, :5]
    t = target[ok]
    return {"top1": float((top[:, 0] == t).mean()), "top5": float((top == t[:, None]).any(1).mean()), "n": int(ok.sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="full")
    p.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    rows = D.load_all()
    clusters = torch.load(CLUSTER_FILE, weights_only=False)
    cid, k = clusters["cluster_id"].numpy(), int(clusters["k"])
    target = {s: cid[rows[s].target] for s in D.SPLITS}

    titles = D.item_titles()
    v1 = np.array(D.load_r3_predictions(D.V1_PREDICTIONS, rows["test"]))
    r3_hit = np.array([titles[t] in {titles[x] for x in p_ if x >= 0} for t, p_ in zip(rows["test"].target, v1)])
    everyone = np.ones(len(r3_hit), bool)

    probs, report = {}, {}
    # training timelines at cluster level, for the Markov model
    train_seq_items = {}
    for u, h, t in zip(rows["train"].user, rows["train"].history, rows["train"].target):
        train_seq_items.setdefault(u, list(h)).append(int(t))
    train_cluster_seqs = [[int(cid[x]) if cid[x] >= 0 else k for x in s] for s in train_seq_items.values()]
    probs["markov"] = {s: markov_probs(train_cluster_seqs, cluster_sequences(rows[s], cid, k, False), k) for s in D.SPLITS}

    for name, full, length in (("gru10", False, 10), ("grufull", True, MAX_FULL)):
        x = {s: pad(cluster_sequences(rows[s], cid, k, full), k, length)[0] for s in D.SPLITS}
        y = {s: torch.from_numpy(target[s]) for s in D.SPLITS}
        print(f"training {name}", flush=True)
        model, best = train_gru(x["train"], y["train"], x["valid"], y["valid"], k, a.epochs)
        probs[name] = {s: predict(model, x[s]) for s in D.SPLITS}
        report[f"{name}_valid_top5"] = best

    fused = torch.load(TABLE_DIR / f"group_pred150_{a.tag}.pt", weights_only=False)
    probs["fused"] = {s: fused[s].float().numpy() for s in D.SPLITS}
    probs["mix"] = {s: 0.5 * probs["grufull"][s] + 0.5 * probs["fused"][s] for s in D.SPLITS}

    # R3's own implied cluster: the cluster of its top-1 item (one-hot)
    r3c = np.zeros((len(v1), k))
    top1 = cid[np.maximum(v1[:, 0], 0)]
    r3c[np.arange(len(v1))[top1 >= 0], top1[top1 >= 0]] = 1.0
    probs["r3_top1_item"] = {"test": r3c}

    lines = ["| predictor | test top-1 | test top-5 | top-1 on R3 hits | top-1 on R3 misses | top-5 on R3 misses |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, pr in probs.items():
        t = pr["test"]
        res = {"all": accuracy(t, target["test"], everyone), "r3_hit": accuracy(t, target["test"], r3_hit),
               "r3_miss": accuracy(t, target["test"], ~r3_hit)}
        report[name] = res
        lines.append(f"| {name} | {res['all']['top1']:.4f} | {res['all']['top5']:.4f} | {res['r3_hit']['top1']:.4f} | "
                     f"{res['r3_miss']['top1']:.4f} | {res['r3_miss']['top5']:.4f} |")
    print("\n".join(lines), flush=True)

    for name in ("grufull", "mix"):
        torch.save({s: torch.from_numpy(probs[name][s]).half() for s in D.SPLITS} | {"k": k, "predictor": name},
                   TABLE_DIR / f"cluster_{name}_{a.tag}.pt")
    out = D.RESULTS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "cluster_predictors.md").write_text("\n".join(lines) + "\n")
    with (out / "cluster_predictors.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}/cluster_predictors.md", flush=True)


if __name__ == "__main__":
    main()
