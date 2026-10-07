"""Collaborative-filtering candidate sources (idea 1 screen, idea 6 signals).

Every source scores the whole catalog for a row and returns its top-N items,
excluding items already in the row's 10-item history (they are in the prompt).

Leakage rules, applied uniformly:

* Statistics are fit on the *training* split only (item co-occurrence,
  transitions, popularity, SASRec), except the time-filtered sources below.
* For a **training** row, the row's own user is removed from every statistic
  (leave-user-out) and SASRec scores come from a fold model that never saw the
  user (train_sasrec_folds.py).  Otherwise training-row candidates would
  already contain the row's own target and the LLM would learn to over-trust
  them.  Validation/test rows use the full statistics, which only contain the past.
* ``trend`` and ``donor`` read other users' interactions from all splits, but
  only events strictly earlier than the row's latest history timestamp, and
  never the row's own user.

Sources:
  sasrec   SASRec next-item scores over the 10-item history
  itemknn  item-item cosine co-occurrence (users x items), recency-weighted over the history
  longknn  the same over the user's whole past timeline (beyond the prompt window)
  markov   first-order transitions from the last three history items
  pop      training popularity
  trend    popularity among other users in the 60 days before the row's history cutoff
  donor    outcomes of other users' rows whose history overlaps (session-kNN donors)
  titlesim TF-IDF title similarity to the history (a proxy for what a title reader picks)
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

import data as D

SOURCES = ("sasrec", "itemknn", "longknn", "markov", "pop", "trend", "donor", "titlesim")
TREND_WINDOW = 60 * 86400


def recency_weights(n: int, decay: float = 0.8) -> np.ndarray:
    return decay ** np.arange(n - 1, -1, -1, dtype=np.float32)


class Context:
    """Training-split statistics shared by the sources."""

    def __init__(self, rows: dict[str, D.Rows], fold_models: dict[int, str] | None = None, folds: int = 4):
        self.rows = rows
        self.n = D.n_items()
        train = rows["train"]
        # each training user's training timeline: first row's history + all train targets
        self.train_seq: dict[str, list[int]] = {}
        for u, h, t in zip(train.user, train.history, train.target):
            seq = self.train_seq.setdefault(u, list(h))
            seq.append(int(t))
        users = sorted(self.train_seq)
        self.user_index = {u: i for i, u in enumerate(users)}
        ri, ci = [], []
        for u, seq in self.train_seq.items():
            for it in set(seq):
                ri.append(self.user_index[u]); ci.append(it)
        self.X = sp.csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(len(users), self.n))
        C = (self.X.T @ self.X).toarray().astype(np.float32)
        self.count = np.diag(C).copy()
        np.fill_diagonal(C, 0.0)
        self.C0 = C
        # first-order transitions within each training timeline
        T = np.zeros((self.n, self.n), np.float32)
        self.own_pairs: dict[str, list[tuple[int, int]]] = {}
        for u, seq in self.train_seq.items():
            pairs = list(zip(seq[:-1], seq[1:]))
            self.own_pairs[u] = pairs
            for a, b in pairs:
                T[a, b] += 1
        self.T = T
        self.T_rowsum = T.sum(1)
        # all observed interaction events (user, item, ts) across splits, for trend/donor
        ev = {}
        for s in D.SPLITS:
            r = rows[s]
            for u, h, hts, t, tts in zip(r.user, r.history, r.history_ts, r.target, r.target_ts):
                for it, ts in zip(h, hts):
                    ev[(u, it, ts)] = 1
                ev[(u, int(t), int(tts))] = 1
        keys = sorted(ev, key=lambda x: x[2])
        self.ev_user = np.array([k[0] for k in keys])
        self.ev_item = np.array([k[1] for k in keys], dtype=np.int64)
        self.ev_ts = np.array([k[2] for k in keys], dtype=np.int64)
        self.fold_models = fold_models or {}
        self.folds = folds
        self._sasrec_cache = {}
        self._title_matrix = None
        self._donor = None

    # ---------------------------------------------------------------- helpers
    def own_vector(self, user: str, split: str) -> np.ndarray:
        """Binary training interactions of ``user`` when leave-user-out applies (training rows)."""
        v = np.zeros(self.n, np.float32)
        if split == "train" and user in self.train_seq:
            v[list(set(self.train_seq[user]))] = 1.0
        return v

    def sasrec(self, path: str):
        if path not in self._sasrec_cache:
            from sasrec_pair_features import load_sasrec
            import torch
            rec = load_sasrec(path)
            asin = D.item_asin_map()
            col = np.full(self.n, -1, np.int64)  # dataset item id -> SASRec index
            for item, a in asin.items():
                idx = rec.iid_map.get(a)
                if idx is not None and idx < rec.num_items:
                    col[item] = idx
            self._sasrec_cache[path] = (rec, col, torch)
        return self._sasrec_cache[path]


def _knn_scores(ctx: Context, histories, weights_list, users, split) -> np.ndarray:
    B = len(histories)
    A = np.zeros((B, ctx.n), np.float32)
    Xu = np.zeros((B, ctx.n), np.float32)
    for b, (h, w, u) in enumerate(zip(histories, weights_list, users)):
        xu = ctx.own_vector(u, split)
        Xu[b] = xu
        nh = ctx.count[h] - xu[h]
        a = np.where(nh > 0, w / np.sqrt(np.maximum(nh, 1e-9)), 0.0)
        np.add.at(A[b], h, a)
    num = A @ ctx.C0 - (A * Xu).sum(1, keepdims=True) * Xu + A * Xu
    denom = np.sqrt(np.maximum(ctx.count[None, :] - Xu, 0.0))
    return np.where(denom > 0, num / np.maximum(denom, 1e-9), 0.0)


def score_batch(ctx: Context, source: str, split: str, idx: np.ndarray) -> np.ndarray:
    r = ctx.rows[split]
    users = r.user[idx]
    if source == "itemknn":
        hs = [r.history[i] for i in idx]
        return _knn_scores(ctx, hs, [recency_weights(len(h)) for h in hs], users, split)
    if source == "longknn":
        hs, ws = [], []
        for i in idx:
            past = r.full_past[i]
            # most recent occurrence of each item, weighted by recency in the whole timeline
            last = {it: p for p, it in enumerate(past)}
            items = np.array(list(last), dtype=np.int64)
            pos = np.array([last[it] for it in items])
            hs.append(items)
            ws.append((0.9 ** (len(past) - 1 - pos)).astype(np.float32))
        return _knn_scores(ctx, hs, ws, users, split)
    if source == "markov":
        out = np.zeros((len(idx), ctx.n), np.float32)
        for b, i in enumerate(idx):
            u, h = users[b], r.history[i]
            own = defaultdict(float)
            own_rows = defaultdict(float)
            if split == "train":
                for x, y in ctx.own_pairs.get(u, []):
                    own[(x, y)] += 1
                    own_rows[x] += 1
            for w, src in zip((1.0, 0.5, 0.25), reversed(h[-3:])):
                rs = ctx.T_rowsum[src] - own_rows[src]
                if rs <= 0:
                    continue
                row = ctx.T[src].copy()
                for (x, y), c in own.items():
                    if x == src:
                        row[y] -= c
                out[b] += w * row / rs
        return out
    if source == "pop":
        out = np.tile(ctx.count, (len(idx), 1))
        if split == "train":
            for b, u in enumerate(users):
                out[b] -= ctx.own_vector(u, split)
        return out
    if source == "trend":
        out = np.zeros((len(idx), ctx.n), np.float32)
        for b, i in enumerate(idx):
            hi = np.searchsorted(ctx.ev_ts, r.cutoff[i], side="left")  # strictly earlier events
            lo = np.searchsorted(ctx.ev_ts, r.cutoff[i] - TREND_WINDOW, side="left")
            sel = ctx.ev_user[lo:hi] != users[b]
            out[b] = np.bincount(ctx.ev_item[lo:hi][sel], minlength=ctx.n)
        return out + 1e-3 * ctx.count[None, :]  # popularity tie-break
    if source == "donor":
        return _donor_scores(ctx, split, idx)
    if source == "titlesim":
        M = ctx._title_matrix
        if M is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            titles = D.item_titles()
            titles = titles + [""] * (ctx.n - len(titles))
            M = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)).fit_transform(titles)
            ctx._title_matrix = M = M.tocsr().astype(np.float32)
        hs = [r.history[i] for i in idx]
        W = sp.lil_matrix((len(idx), ctx.n), dtype=np.float32)
        for b, h in enumerate(hs):
            for it, w in zip(h, recency_weights(len(h))):
                W[b, it] += w
        return np.asarray(((W.tocsr() @ M) @ M.T).todense(), dtype=np.float32)
    if source == "sasrec":
        return _sasrec_scores(ctx, split, idx)
    raise ValueError(source)


def _sasrec_scores(ctx: Context, split: str, idx: np.ndarray) -> np.ndarray:
    from train_sasrec_folds import user_fold
    r = ctx.rows[split]
    out = np.full((len(idx), ctx.n), -1e9, np.float32)
    if split == "train":
        if not ctx.fold_models:
            raise ValueError("training-row SASRec scores need fold models (train_sasrec_folds.py)")
        groups = defaultdict(list)
        for b, i in enumerate(idx):
            groups[ctx.fold_models[user_fold(str(r.user[i]), ctx.folds)]].append(b)
    else:
        groups = {str(D.SASREC_FULL): list(range(len(idx)))}
    for path, members in groups.items():
        rec, col, torch = ctx.sasrec(path)
        valid_items = np.nonzero(col >= 0)[0]
        seqs = []
        for b in members:
            h = [col[it] for it in r.history[idx[b]] if col[it] >= 0][-rec.max_len:]
            seqs.append([rec.pad_idx] * (rec.max_len - len(h)) + h)
        with torch.no_grad():
            seq_t = torch.as_tensor(seqs, dtype=torch.long)
            hidden = rec.model._encode(seq_t)
            items = torch.as_tensor(col[valid_items])
            emb = rec.model.item_emb(items)
            bias = rec.model.item_biases(items) if hasattr(rec.model, "item_biases") else None
            s = rec.model._score_items(hidden, emb, bias).float().numpy()
        out[np.array(members)[:, None], valid_items[None, :]] = s
    return out


def _donor_scores(ctx: Context, split: str, idx: np.ndarray, top_donors: int = 200) -> np.ndarray:
    if ctx._donor is None:
        bank_rows = []
        for s in ("train", "valid", "test"):
            r = ctx.rows[s]
            for i in range(len(r)):
                bank_rows.append((r.user[i], r.history[i], int(r.target[i]), int(r.target_ts[i])))
        df = np.zeros(ctx.n, np.float32)
        for _, h, _, _ in bank_rows:
            df[list(set(h))] += 1
        idf = np.log((1 + len(bank_rows)) / (1 + df)).astype(np.float32)
        ri, ci, vals = [], [], []
        for k, (_, h, _, _) in enumerate(bank_rows):
            for it in set(h):
                ri.append(k); ci.append(it); vals.append(idf[it])
        B = sp.csr_matrix((vals, (ri, ci)), shape=(len(bank_rows), ctx.n), dtype=np.float32)
        norms = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
        B = sp.diags(1.0 / np.maximum(norms, 1e-9)) @ B
        ctx._donor = {
            "B": B.tocsr(), "idf": idf,
            "user": np.array([x[0] for x in bank_rows]),
            "target": np.array([x[2] for x in bank_rows], dtype=np.int64),
            "ts": np.array([x[3] for x in bank_rows], dtype=np.int64),
        }
    d = ctx._donor
    r = ctx.rows[split]
    ri, ci, vals = [], [], []
    for b, i in enumerate(idx):
        for it in set(r.history[i]):
            ri.append(b); ci.append(it); vals.append(d["idf"][it])
    Q = sp.csr_matrix((vals, (ri, ci)), shape=(len(idx), ctx.n), dtype=np.float32)
    sims = np.asarray((Q @ d["B"].T).todense(), dtype=np.float32)
    # eligibility: another user, outcome strictly before the row's recommendation time
    ok = (d["ts"][None, :] < r.cutoff[idx][:, None]) & (d["user"][None, :] != r.user[idx][:, None])
    sims = np.where(ok, sims, 0.0)
    k = min(top_donors, sims.shape[1])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = np.zeros((len(idx), ctx.n), np.float32)
    rows_b = np.repeat(np.arange(len(idx)), k)
    np.add.at(out, (rows_b, d["target"][top].ravel()), sims[np.arange(len(idx))[:, None], top].ravel())
    return out


def run_source(ctx: Context, source: str, split: str, top_n: int = 100, batch: int = 512, limit: int = -1):
    """Top-``top_n`` item ids/scores per row (history excluded) and the target's full rank."""
    r = ctx.rows[split]
    n = len(r) if limit <= 0 else min(limit, len(r))
    ids = np.zeros((n, top_n), np.int32)
    scores = np.zeros((n, top_n), np.float32)
    rank = np.zeros(n, np.int64)
    for start in range(0, n, batch):
        idx = np.arange(start, min(n, start + batch))
        s = score_batch(ctx, source, split, idx).astype(np.float32)
        for b, i in enumerate(idx):
            s[b, r.history[i]] = -np.inf
        # small deterministic jitter breaks ties without favouring low item ids
        s = s + 1e-7 * ((np.arange(ctx.n)[None, :] * 2654435761 % 1000) / 1000.0)
        top = np.argpartition(-s, top_n - 1, axis=1)[:, :top_n]
        order = np.argsort(-np.take_along_axis(s, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        ids[idx] = top
        scores[idx] = np.take_along_axis(s, top, 1)
        tgt = s[np.arange(len(idx)), r.target[idx]]
        rank[idx] = np.where(np.isfinite(tgt), (s > tgt[:, None]).sum(1) + 1, ctx.n + 1)
        if os.environ.get("COLLAB_VERBOSE") and start % (batch * 20) == 0:
            print(f"  {source}/{split} {idx[-1] + 1}/{n}", flush=True)
    return ids, scores, rank
