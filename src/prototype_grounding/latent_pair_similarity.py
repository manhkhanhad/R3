#!/usr/bin/env python3
"""How similar are latents of samples with the same user / the same target item?

Two subcommands:

  extract  One first pass per row: the thought vector r (the attention-pooled
           latent spliced into the decoder) and two history-only baselines from
           the same pass, h_last (last hidden state before <|Thought|>, which is
           also the pooling query) and h_mean (mean last hidden state over the
           user-input tokens). Saved as fp16 under output_dir/.

  report   Pair similarities for random / same-user / same-target pairs, binned
           by history overlap (Jaccard of history item ids), group variance
           fractions (eta^2, permutation-adjusted) and nearest-neighbour
           retrieval. Every number is computed for r and for both baselines,
           so "r groups by target" can be told apart from "the history already
           groups by target".

Confounds this design controls:
  * Same-user rows are sliding-window prefixes of one sequence, so they share
    most of their history. Same-user pairs are compared with cross-user pairs
    in the same history-overlap bin.
  * r has a large shared component, which makes every raw cosine high. The
    main metric is the cosine after subtracting the split mean ("centered");
    raw cosine is reported alongside.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as E

KINDS = ("r", "h_last", "h_mean")
BINS = (("0", 0.0, 0.0), ("(0,0.5]", 1e-9, 0.5), ("(0.5,1)", 0.5 + 1e-9, 1 - 1e-9), ("1", 1.0, 1.0))
MAX_GROUP = 60  # members sampled per user/target group when enumerating pairs
BOOT = 2000


def out_dir(tag: str, split: str) -> Path:
    return E.ROOT / "output_dir/latent_pair_similarity" / tag / split


def extract(split: str, batch_size: int, limit: int, checkpoint: Path, tag: str) -> None:
    model, tokenizer, device = E.load_model(checkpoint)
    ds = E.PromptDataset(split, tokenizer, limit)
    # The instruction and the "### User Input" header are identical in every
    # row; h_mean averages only the tokens after this shared prefix.
    a, b = ds.rows[0]["input_ids"], ds.rows[1]["input_ids"]
    prefix = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: E.collate(x, tokenizer.pad_token_id))
    vecs = {k: [] for k in KINDS}
    meta = []
    start = time.time()
    for step, batch in enumerate(loader):
        ids, mask, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        with torch.no_grad():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            r = model.attention(hidden, mask, pos)
        rows = torch.arange(ids.size(0), device=device)
        span = torch.arange(ids.size(1), device=device)[None]
        user_part = ((span >= prefix) & (span < pos[:, None])).to(hidden.dtype)
        h_mean = (hidden * user_part[..., None]).sum(1) / user_part.sum(1, keepdim=True).clamp_min(1)
        for k, v in (("r", r), ("h_last", hidden[rows, pos - 1]), ("h_mean", h_mean)):
            vecs[k].append(v.float().cpu().half())
        meta.extend(batch["meta"])
        if step % 200 == 0:
            print(f"extract {split}: {min((step + 1) * batch_size, len(ds))}/{len(ds)}", flush=True)
    out = out_dir(tag, split)
    out.mkdir(parents=True, exist_ok=True)
    for k in KINDS:
        torch.save(torch.cat(vecs[k]), out / f"{k}.pt")
    frame = pd.DataFrame(meta)
    frame["history_item_ids"] = frame.history_item_ids.map(json.dumps)
    frame.to_csv(out / "metadata.csv", index=False)
    E.atomic_json(out / "summary.json", {
        "split": split, "n": len(ds), "shared_prefix_tokens": prefix, "checkpoint": str(checkpoint),
        "checkpoint_sha256": E.sha256(checkpoint / "model.safetensors"),
        "split_sha256": E.sha256(E.SPLIT_FILES[split]), "limit": limit,
        "elapsed_seconds": time.time() - start,
    })
    print(f"saved {out}", flush=True)


# ---------------------------------------------------------------- pairs

def jaccard_and_nested(hist: list[set], seq: list[list], i: np.ndarray, j: np.ndarray):
    jac, nested = np.empty(len(i)), np.zeros(len(i), dtype=bool)
    for t, (a, b) in enumerate(zip(i, j)):
        A, B = hist[a], hist[b]
        jac[t] = len(A & B) / max(1, len(A | B))
        s, u = seq[a], seq[b]
        if len(s) > len(u):
            s, u = u, s
        nested[t] = u[:len(s)] == s
    return jac, nested


def group_pairs(codes: np.ndarray, rng, users: np.ndarray | None = None):
    """All within-group pairs (members capped at MAX_GROUP); optionally cross-user only."""
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    left, right, group = [], [], []
    for g, members in enumerate(np.split(order, bounds)):
        if len(members) < 2:
            continue
        if len(members) > MAX_GROUP:
            members = rng.choice(members, MAX_GROUP, replace=False)
        a, b = np.triu_indices(len(members), 1)
        a, b = members[a], members[b]
        if users is not None:
            keep = users[a] != users[b]
            a, b = a[keep], b[keep]
        left.append(a); right.append(b); group.append(np.full(len(a), codes[members[0]]))
    if not left:
        return (np.empty(0, int),) * 3
    return np.concatenate(left), np.concatenate(right), np.concatenate(group)


def overlap_pairs(hist_lists: list[list], users: np.ndarray, rng, per_item: int = 40):
    """Cross-user pairs that share at least one history item (matched control for same-user)."""
    index: dict[int, list[int]] = {}
    for row, items in enumerate(hist_lists):
        for it in set(items):
            index.setdefault(it, []).append(row)
    left, right = [], []
    for rows in index.values():
        if len(rows) < 2:
            continue
        rows = np.asarray(rows)
        a = rng.choice(rows, per_item); b = rng.choice(rows, per_item)
        keep = users[a] != users[b]
        left.append(a[keep]); right.append(b[keep])
    a, b = np.concatenate(left), np.concatenate(right)
    pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def boot_mean(values: np.ndarray, clusters: np.ndarray, rng) -> np.ndarray:
    """Cluster-bootstrap replicates of the pair mean (clusters resampled with replacement)."""
    _, c = np.unique(clusters, return_inverse=True)
    sums = np.bincount(c, weights=values); counts = np.bincount(c).astype(float)
    draws = rng.integers(0, len(sums), size=(BOOT, len(sums)))
    return sums[draws].sum(1) / counts[draws].sum(1)


def summarize(values: np.ndarray, clusters: np.ndarray, rng) -> dict:
    if len(values) == 0:
        return {"pairs": 0}
    reps = boot_mean(values, clusters, rng)
    return {"pairs": int(len(values)), "clusters": int(len(np.unique(clusters))),
            "mean": float(values.mean()), "ci95": [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))],
            "_reps": reps}


def delta(a: dict, b: dict) -> dict:
    if not a.get("pairs") or not b.get("pairs"):
        return {}
    d = a["_reps"] - b["_reps"]
    return {"delta": a["mean"] - b["mean"], "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}


# ---------------------------------------------------------------- report

def eta_squared(x: torch.Tensor, codes: np.ndarray, seed: int) -> dict:
    """Fraction of centered sum of squares explained by group means, minus a permutation null."""
    total = float(x.square().sum())
    codes_t = torch.from_numpy(codes).long().to(x.device)
    groups = int(codes.max()) + 1

    def eta(c):
        sums = torch.zeros(groups, x.size(1), device=x.device).index_add_(0, c, x)
        counts = torch.bincount(c, minlength=groups).float().clamp_min(1)
        return float((sums.square().sum(1) / counts).sum() / total)

    observed = eta(codes_t)
    g = torch.Generator(device="cpu").manual_seed(seed)
    null = [eta(codes_t[torch.randperm(len(codes), generator=g).to(x.device)]) for _ in range(10)]
    return {"groups": groups, "observed": observed, "permutation_mean": float(np.mean(null)),
            "adjusted": observed - float(np.mean(null))}


def neighbour_retrieval(q: torch.Tensor, bank: torch.Tensor, q_meta: pd.DataFrame, b_meta: pd.DataFrame,
                        same_split: bool, k: int = 10) -> dict:
    """Among each query's k nearest bank rows from *other users*, the fraction with the query's target."""
    q_user, b_user = q_meta.user_id.to_numpy(), b_meta.user_id.to_numpy()
    q_tgt = torch.as_tensor(q_meta.target_item.to_numpy(), device=q.device)
    b_tgt = torch.as_tensor(b_meta.target_item.to_numpy(), device=q.device)
    _, b_user_codes = np.unique(np.concatenate([q_user, b_user]), return_inverse=True)
    qu = torch.as_tensor(b_user_codes[:len(q_user)], device=q.device)
    bu = torch.as_tensor(b_user_codes[len(q_user):], device=q.device)
    hit1, prec, chance, eligible = [], [], [], []
    for s in range(0, len(q), 1024):
        sim = q[s:s + 1024] @ bank.T
        other = bu[None] != qu[s:s + 1024, None]
        sim = sim.masked_fill(~other, -float("inf"))
        top = sim.topk(k, dim=1).indices
        match = b_tgt[top] == q_tgt[s:s + 1024, None]
        positives = ((b_tgt[None] == q_tgt[s:s + 1024, None]) & other).sum(1).float()
        hit1.append(match[:, 0].float()); prec.append(match.float().mean(1))
        chance.append(positives / other.sum(1).float()); eligible.append(positives > 0)
    hit1, prec, chance, eligible = [torch.cat(x).cpu().numpy() for x in (hit1, prec, chance, eligible)]
    e = eligible
    return {"queries_with_a_positive": int(e.sum()), "top1_same_target": float(hit1[e].mean()),
            f"precision_at_{k}": float(prec[e].mean()), "chance": float(chance[e].mean()),
            f"lift_precision_at_{k}": float(prec[e].mean() / max(chance[e].mean(), 1e-12))}


def same_user_retrieval(x: torch.Tensor, meta: pd.DataFrame) -> dict:
    """Fraction of rows whose nearest other row is from the same user, vs. chance."""
    _, u = np.unique(meta.user_id.to_numpy(), return_inverse=True)
    u = torch.as_tensor(u, device=x.device)
    counts = torch.bincount(u).float()
    hits = []
    for s in range(0, len(x), 1024):
        sim = x[s:s + 1024] @ x.T
        sim[torch.arange(sim.size(0)), torch.arange(s, s + sim.size(0))] = -float("inf")
        hits.append((u[sim.argmax(1)] == u[s:s + 1024]).float())
    hits = torch.cat(hits)
    eligible = counts[u] > 1
    chance = ((counts[u] - 1) / (len(x) - 1))[eligible]
    return {"rows_with_another_row": int(eligible.sum()), "top1_same_user": float(hits[eligible].mean()),
            "chance": float(chance.mean())}


def report(tag: str, splits: list[str], seed: int, random_pairs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {"tag": tag, "seed": seed, "max_group": MAX_GROUP, "bootstrap": BOOT, "splits": {}}
    loaded = {}
    for split in splits:
        base = out_dir(tag, split)
        meta = pd.read_csv(base / "metadata.csv")
        hist_lists = [json.loads(h) for h in meta.history_item_ids]
        hist = [set(h) for h in hist_lists]
        users = meta.user_id.to_numpy()
        user_codes = pd.factorize(meta.user_id)[0]
        target_codes = pd.factorize(meta.target_item)[0]
        rng = np.random.default_rng(seed)
        n = len(meta)

        pair_sets = {}
        a, b = rng.integers(0, n, random_pairs), rng.integers(0, n, random_pairs)
        keep = users[a] != users[b]
        pair_sets["random_cross_user"] = (a[keep], b[keep], np.arange(keep.sum()))
        pair_sets["same_user"] = group_pairs(user_codes, rng)
        pair_sets["same_target_cross_user"] = group_pairs(target_codes, rng, users)
        a, b = overlap_pairs(hist_lists, users, rng)
        pair_sets["cross_user_overlapping_history"] = (a, b, np.arange(len(a)))
        for name, (a, b, g) in list(pair_sets.items()):
            jac, nested = jaccard_and_nested(hist, hist_lists, a, b)
            same_target = meta.target_item.to_numpy()[a] == meta.target_item.to_numpy()[b]
            pair_sets[name] = (a, b, g, jac, nested, same_target)

        split_res = {"n": n, "pair_counts": {k: int(len(v[0])) for k, v in pair_sets.items()}, "kinds": {}}
        for kind in KINDS:
            x = torch.load(base / f"{kind}.pt", map_location="cpu", weights_only=True).float()
            loaded[(split, kind)] = (x, meta)
            centered = F.normalize(x - x.mean(0), dim=1)
            raw = F.normalize(x, dim=1)
            kind_res = {"norm_mean": float(x.norm(dim=1).mean()),
                        "cosine_to_mean": float((raw @ F.normalize(x.mean(0), dim=0)).mean()),
                        "pairs": {}, "deltas": {}}
            stats = {}
            for name, (a, b, g, jac, nested, same_target) in pair_sets.items():
                cc = (centered[a] * centered[b]).sum(1).numpy()
                cr = (raw[a] * raw[b]).sum(1).numpy()
                for bin_name, lo, hi in (("all", -1, 2),) + BINS:
                    sel = (jac >= lo) & (jac <= hi)
                    sel_sets = {"": sel}
                    if name == "same_user" and bin_name == "all":
                        sel_sets["|nested_prefix"] = nested
                        sel_sets["|not_nested"] = ~nested
                    for suffix, s in sel_sets.items():
                        key = f"{name}|J={bin_name}{suffix}"
                        stats[key] = summarize(cc[s], g[s], rng)
                        stats[key]["raw_cosine_mean"] = float(cr[s].mean()) if s.any() else None
            # Same-target vs random at equal (zero) history overlap is the key test;
            # same-user vs cross-user pairs at matched overlap controls for shared history.
            d = kind_res["deltas"]
            d["same_target_vs_random|J=0"] = delta(stats["same_target_cross_user|J=0"], stats["random_cross_user|J=0"])
            d["same_target_vs_random|all"] = delta(stats["same_target_cross_user|J=all"], stats["random_cross_user|J=all"])
            for bin_name, *_ in BINS:
                d[f"same_user_vs_cross_user_overlap|J={bin_name}"] = delta(
                    stats[f"same_user|J={bin_name}"],
                    stats[f"cross_user_overlapping_history|J={bin_name}"] if bin_name != "0"
                    else stats["random_cross_user|J=0"])
            kind_res["pairs"] = {k: {kk: vv for kk, vv in v.items() if kk != "_reps"} for k, v in stats.items()}
            xc = (x - x.mean(0)).to(device)
            kind_res["eta_squared"] = {"user": eta_squared(xc, user_codes, seed),
                                       "target": eta_squared(xc, target_codes, seed)}
            xn = centered.to(device)
            kind_res["retrieval_within_split"] = {
                "same_target_other_user": neighbour_retrieval(xn, xn, meta, meta, True),
                "same_user": same_user_retrieval(xn, meta)}
            split_res["kinds"][kind] = kind_res
        results["splits"][split] = split_res

    # Test queries against the train bank: do test latents land near train
    # latents (of other users) that had the same target?
    if "train" in splits and "test" in splits:
        results["test_vs_train_bank"] = {}
        for kind in KINDS:
            (xt, mt), (xb, mb) = loaded[("test", kind)], loaded[("train", kind)]
            mu = xb.mean(0)
            q = F.normalize(xt - mu, dim=1).to(device)
            bank = F.normalize(xb - mu, dim=1).to(device)
            results["test_vs_train_bank"][kind] = neighbour_retrieval(q, bank, mt, mb, False)

    out = E.ANALYSIS / "results/latent_pair_similarity" / tag / "report.json"
    E.atomic_json(out, results)
    print_tables(results)
    print(f"saved {out}", flush=True)


def print_tables(res: dict) -> None:
    def fmt(s):
        if not s or not s.get("pairs"):
            return "—"
        return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] (n={s['pairs']})"
    for split, sr in res["splits"].items():
        print(f"\n### {split} (n={sr['n']}) centered cosine")
        rows = [k for k in sr["kinds"]["r"]["pairs"]]
        print("| pairs | " + " | ".join(KINDS) + " |")
        print("|---|" + "---|" * len(KINDS))
        for k in rows:
            print(f"| {k} | " + " | ".join(fmt(sr["kinds"][kind]["pairs"][k]) for kind in KINDS) + " |")
        print("\n| delta | " + " | ".join(KINDS) + " |")
        print("|---|" + "---|" * len(KINDS))
        for k in sr["kinds"]["r"]["deltas"]:
            cells = []
            for kind in KINDS:
                d = sr["kinds"][kind]["deltas"][k]
                cells.append(f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "—")
            print(f"| {k} | " + " | ".join(cells) + " |")
        for kind in KINDS:
            kr = sr["kinds"][kind]
            print(f"{kind}: norm {kr['norm_mean']:.1f}, cos-to-mean {kr['cosine_to_mean']:.3f}, "
                  f"eta2 user {kr['eta_squared']['user']['adjusted']:.3f}, target {kr['eta_squared']['target']['adjusted']:.3f}; "
                  f"retrieval {json.dumps(kr['retrieval_within_split'])}")
    for kind, r in res.get("test_vs_train_bank", {}).items():
        print(f"test->train bank {kind}: {json.dumps(r)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("--checkpoint", type=Path, required=True)
    x.add_argument("--tag", required=True)
    x.add_argument("--category", default="Toys_and_Games")
    x.add_argument("--split", choices=("train", "valid", "test"), required=True)
    x.add_argument("--limit", type=int, default=-1)
    x.add_argument("--batch-size", type=int, default=8)
    x = sub.add_parser("report")
    x.add_argument("--tag", required=True)
    x.add_argument("--category", default="Toys_and_Games")
    x.add_argument("--splits", nargs="+", default=["train", "test"])
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--random-pairs", type=int, default=200_000)
    a = p.parse_args()
    E.configure_category(a.category)
    E.seed_all(42)
    if a.cmd == "extract":
        extract(a.split, a.batch_size, a.limit, a.checkpoint, a.tag)
    else:
        report(a.tag, a.splits, a.seed, a.random_pairs)


if __name__ == "__main__":
    main()
