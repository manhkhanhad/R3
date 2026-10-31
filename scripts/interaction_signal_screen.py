"""Screen interaction-only signals for information R3 lacks (Toys test).

Every signal uses user-item interactions only (ids + timestamps).  The split is
global-temporal (train < 2018-02-08 <= valid < 2018-04-15 <= test), so "past" =
all train + valid interactions predates every test target.  Signals:

* pop_past: item popularity over the past interactions (static).
* pop_recent_{7,30}d: popularity among all interactions observed strictly before
  the row's timestamp within the window (past + earlier test-row histories:
  what a deployed system has seen; no test target is used unless it already
  appears in a later row's history with an earlier timestamp).
* itemknn: sequential item-to-item: score(j) = sum_i w_i C(i->j) / sqrt(N_i N_j),
  C = within-user co-occurrence with j after i (gap <= 3) in past sequences,
  w_i = 0.8^(distance of history item i from the end).
* userknn: users with cosine-similar past item sets (top 50); score(j) = sum of
  similarities of neighbours who interacted with j.
* exports: SASRec / LightGCN / BPR Cornac top-20 exports (top-10 used; ranks
  beyond 20 unknown).
History items are excluded from every ranking (as in the exports).  Reported on
all / copy (target first word in a history title) / other rows: HR@10, HR@50,
median target rank (score-based signals), hits only this signal has vs R3, and
best-of-both HR@10.

Usage: interaction_signal_screen.py --r3_eval <json> --export NAME=<csv> ... [--out md]
"""

import argparse
import ast
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"
DAY = 86400


def first_word(title):
    words = str(title).lower().split()
    return words[0] if words else ""


def boot(d, n=2000, seed=42):
    d = np.asarray(d, float)
    s = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def interactions(frame):
    """Unique (user, item, ts) from row histories and targets."""
    out = set()
    for u, ids, tss, i, t in zip(frame["user_id"], frame["history_item_id"], frame["history_timestamp"],
                                 frame["item_id"], frame["timestamp"]):
        for a, b in zip(ast.literal_eval(ids), ast.literal_eval(tss)):
            out.add((u, int(a), int(b)))
        out.add((u, int(i), int(t)))
    return out


def ranks_from_scores(scores, target, exclude):
    """1-based rank of the target (ties count half), history items excluded."""
    s = scores.astype(np.float64).copy()
    s[exclude] = -np.inf
    t = s[target]
    return int((s > t).sum() + 1 + 0.5 * ((s == t).sum() - 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--r3_eval", required=True)
    p.add_argument("--export", action="append", default=[], help="NAME=Cornac test_recommendations.csv")
    p.add_argument("--out")
    args = p.parse_args()

    read = lambda s: pd.read_csv(os.path.join(args.data_dir, s, SPLIT_FILE))
    train, valid, test = read("train"), read("valid"), read("test")
    info = os.path.join(args.data_dir, "info", SPLIT_FILE.replace(".csv", ".txt"))
    title_id = {}
    for line in open(info):
        title, i = line.rstrip("\n").rsplit("\t", 1)
        title_id.setdefault(title.strip(), int(i))
    n_items = sum(1 for _ in open(info))
    asin_id = {}
    for f in (train, valid, test):
        asin_id.update(zip(f["item_asin"], f["item_id"].astype(int)))
        for a_s, i_s in zip(f["item_asins"], f["history_item_id"]):
            asin_id.update(zip(ast.literal_eval(a_s), ast.literal_eval(i_s)))

    past = interactions(train) | interactions(valid)
    t0 = test["timestamp"].min()
    assert max(t for _, _, t in past) <= t0, "past interactions must predate the test split"
    observed = sorted(past | interactions(test.assign(item_id=-1, timestamp=-1)) - {(u, -1, -1) for u in test["user_id"]},
                      key=lambda x: x[2])
    obs_ts = np.array([x[2] for x in observed])
    obs_item = np.array([x[1] for x in observed])

    target = test["item_id"].astype(int).to_numpy()
    hist = [ast.literal_eval(h) for h in test["history_item_id"]]
    ts = test["timestamp"].to_numpy()
    copy = np.array([first_word(x) in {first_word(h) for h in ast.literal_eval(hs)}
                     for x, hs in zip(test["item_title"], test["history_item_title"])])

    # past sequences per user
    seqs = defaultdict(list)
    for u, i, t in sorted(past, key=lambda x: (x[0], x[2])):
        seqs[u].append(i)
    count = np.bincount([i for _, i, _ in past], minlength=n_items).astype(float)
    rows, cols = [], []
    for s in seqs.values():
        for a in range(len(s)):
            for b in range(a + 1, min(len(s), a + 4)):
                if s[a] != s[b]:
                    rows.append(s[a]); cols.append(s[b])
    trans = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items, n_items)).tocsr()
    norm = np.sqrt(np.maximum(count, 1))
    trans = sp.diags(1 / norm) @ trans @ sp.diags(1 / norm)
    users = list(seqs)
    ui = sp.csr_matrix((np.ones(sum(len(set(seqs[u])) for u in users)),
                        ([k for k, u in enumerate(users) for _ in set(seqs[u])],
                         [i for u in users for i in set(seqs[u])])), shape=(len(users), n_items))
    ui_n = sp.diags(1 / np.sqrt(np.maximum(np.asarray(ui.sum(1)).ravel(), 1))) @ ui

    signals = {"pop_past": [], "pop_recent_7d": [], "pop_recent_30d": [], "itemknn": [], "userknn": []}
    tops = {k: [] for k in signals}
    for r in range(len(test)):
        excl = np.array(hist[r], dtype=int)
        before = obs_ts < ts[r]
        sc = {"pop_past": count}
        for w in (7, 30):
            m = before & (obs_ts >= ts[r] - w * DAY)
            sc[f"pop_recent_{w}d"] = np.bincount(obs_item[m], minlength=n_items).astype(float)
        wts = 0.8 ** np.arange(len(hist[r]))[::-1]
        sc["itemknn"] = np.asarray(trans[hist[r]].T @ wts).ravel()
        q = np.zeros(n_items); q[list(set(hist[r]))] = 1
        sim = ui_n @ (q / np.sqrt(q.sum()))
        nb = np.argsort(-sim)[:50]
        sc["userknn"] = np.asarray(ui[nb].T @ sim[nb]).ravel()
        for k, s in sc.items():
            signals[k].append(ranks_from_scores(s, target[r], excl))
            s2 = s.astype(float).copy(); s2[excl] = -np.inf
            tops[k].append(set(np.argsort(-s2, kind="stable")[:10].tolist()))
    ranks = {k: np.array(v) for k, v in signals.items()}

    hit10 = {k: np.array([target[r] in tops[k][r] for r in range(len(test))]) for k in tops}
    r3 = json.load(open(args.r3_eval))
    hit10["R3"] = np.array([target[r] in {title_id.get(x.strip().strip('"')) for x in row["predict"][:10]}
                            for r, row in enumerate(r3)])
    for spec in args.export:
        name, path = spec.split("=", 1)
        e = pd.read_csv(path)
        hit10[name] = np.array([a in json.loads(x)[:10] for a, x in zip(test["item_asin"], e["recommended_asins"])])

    lines = [f"# Interaction-only signal screen (Toys test, n = {len(test)}; copy {copy.sum()}, other {(~copy).sum()})", "",
             "History items excluded. Median rank over all items (ties half); exports have top-20 only.", "",
             "| signal | HR@10 all | HR@10 copy | HR@10 other | HR@50 other | median rank other | ranks 11–100 other | only-this hits vs R3 (other) | best-of-both HR@10 other |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    oth = ~copy
    for k in list(ranks) + [x for x in hit10 if x not in ranks]:
        h = hit10[k]
        only = int((h & ~hit10["R3"] & oth).sum()) if k != "R3" else 0
        best = (h | hit10["R3"])[oth].mean()
        if k in ranks:
            rk = ranks[k]
            extra = f"{np.mean(rk[oth] <= 50):.4f} | {np.median(rk[oth]):.0f} | {np.mean((rk[oth] > 10) & (rk[oth] <= 100)):.3f}"
        else:
            extra = "– | – | –"
        lines.append(f"| {k} | {h.mean():.4f} | {h[copy].mean():.4f} | {h[oth].mean():.4f} | {extra} | "
                     f"{only if k != 'R3' else '–'} | {best:.4f} |")
    lines += ["", "Random: median rank ≈ " + f"{n_items // 2}" + f", P(rank ≤ 50) ≈ {50 / n_items:.4f}.", ""]
    for k in hit10:
        if k == "R3":
            continue
        d, lo, hi = boot(hit10[k][oth].astype(float) - hit10["R3"][oth])
        lines.append(f"- other rows, {k} − R3 HR@10: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
