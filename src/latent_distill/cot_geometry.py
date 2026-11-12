"""Embedding geometry of the CoT teacher targets (document/latent_distillation.md).

For a teacher-state file (teacher_states.py or step_embeddings.py), per step:
* spread: norms, pairwise cosine of random row pairs, raw and standardised
  (standardised = per step and dimension over the valid rows, as larec_latent.py
  does at load);
* spectrum of the standardised states: variance share of the top principal
  components and the participation ratio (effective dimension);
* what the geometry encodes, in standardised space: mean cosine of row pairs that
  share the user, the target item or the last history item (different user), against
  random pairs; the share of variance between users (eta^2 of user identity) and
  between target items; 10-NN purity for the same three keys;
* cosine between the steps of the same row against different rows.

Usage: python src/latent_distill/cot_geometry.py --states <file>.pt --csv <train csv>
"""

import ast
import json

import fire
import numpy as np
import pandas as pd
import torch


def pair_cos(x, a, b):
    return (x[a] * x[b]).sum(1)


def eta2(x, labels):
    """Share of total variance explained by group means (one-way ANOVA), chance-adjusted.

    With G groups over N rows, random labels already give eta^2 ~ (G-1)/(N-1) (0.23 for
    ~9.7k users over ~42k rows), so the adjusted value rescales that to 0.
    """
    x = x - x.mean(0)
    total = (x ** 2).sum()
    df = pd.DataFrame(x)
    means = df.groupby(labels).transform("mean").to_numpy()
    raw = float((means ** 2).sum() / total)
    chance = (len(np.unique(labels)) - 1) / (len(labels) - 1)
    return {"raw": raw, "chance": chance, "adjusted": (raw - chance) / (1 - chance)}


def knn_purity(x, keys, k=10, chunk=2048):
    """For each row, the share of its k nearest neighbours (cosine) sharing each key."""
    t = torch.from_numpy(x).cuda() if torch.cuda.is_available() else torch.from_numpy(x)
    hits = {name: [] for name in keys}
    for s in range(0, len(x), chunk):
        sim = t[s:s + chunk] @ t.T
        idx = torch.arange(s, min(s + chunk, len(x)), device=sim.device)
        sim[torch.arange(len(idx)), idx] = -2
        nn = sim.topk(k, dim=1).indices.cpu().numpy()
        for name, key in keys.items():
            hits[name].append((key[nn] == key[s:s + chunk, None]).mean(1))
    return {name: float(np.concatenate(v).mean()) for name, v in hits.items()}


def sample_pairs(groups, rng, n, exclude_users=None):
    """n random pairs of rows within the same group (optionally from different users)."""
    a, b = [], []
    members = [m for m in groups if len(m) > 1]
    sizes = np.array([len(m) for m in members], dtype=float)
    picks = rng.choice(len(members), size=n * 3, p=sizes / sizes.sum())
    for g in picks:
        i, j = rng.choice(members[g], 2, replace=False)
        if exclude_users is not None and exclude_users[i] == exclude_users[j]:
            continue
        a.append(i); b.append(j)
        if len(a) == n:
            break
    return np.array(a), np.array(b)


def main(states: str, csv: str, pairs: int = 50000, seed: int = 0, out: str = None):
    rng = np.random.default_rng(seed)
    spec = torch.load(states)
    valid = spec["valid"].bool().numpy()
    x_all = spec["states"].float().numpy()[valid]
    frame = pd.read_csv(csv).iloc[np.nonzero(valid)[0]].reset_index(drop=True)
    user = frame.user_id.astype("category").cat.codes.to_numpy()
    target = frame.item_id.to_numpy()
    last = frame.history_item_id.map(lambda h: ast.literal_eval(h)[-1]).to_numpy()
    n, k, d = x_all.shape
    names = ["Evidence", "Interpretation", "Recommendation"] if k == 3 else [f"step{i + 1}" for i in range(k)]
    report = {"states": states, "rows": n, "dim": d, "steps": {}}

    ra, rb = rng.integers(n, size=pairs), rng.integers(n, size=pairs)
    keep = ra != rb
    ra, rb = ra[keep], rb[keep]
    by = lambda key: pd.Series(np.arange(n)).groupby(key).apply(list).tolist()
    su = sample_pairs(by(user), rng, pairs)
    st = sample_pairs(by(target), rng, pairs, exclude_users=user)
    sl = sample_pairs(by(last), rng, pairs, exclude_users=user)

    std_steps = []
    for j in range(k):
        x = x_all[:, j]
        raw = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        z = (x - x.mean(0)) / x.std(0).clip(1e-3)
        std_steps.append(z)
        zn = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-8)
        ev = np.linalg.svd(z[rng.permutation(n)[:20000]] / np.sqrt(20000), compute_uv=False) ** 2
        ev = ev / ev.sum()
        cos_rand = pair_cos(zn, ra, rb)
        row = {
            "norm_mean": float(np.linalg.norm(x, axis=1).mean()),
            "raw_cos_random": float(pair_cos(raw, ra, rb).mean()),
            "std_cos_random": float(cos_rand.mean()),
            "std_cos_random_p95": float(np.percentile(cos_rand, 95)),
            "pc_share_top1": float(ev[:1].sum()), "pc_share_top10": float(ev[:10].sum()),
            "pc_share_top100": float(ev[:100].sum()),
            "participation_ratio": float(1 / (ev ** 2).sum()),
            "std_cos_same_user": float(pair_cos(zn, *su).mean()),
            "std_cos_same_target_diff_user": float(pair_cos(zn, *st).mean()),
            "std_cos_same_last_item_diff_user": float(pair_cos(zn, *sl).mean()),
            "eta2_user": eta2(z, user), "eta2_target": eta2(z, target), "eta2_last_item": eta2(z, last),
            "knn10": knn_purity(zn.astype(np.float32), {"user": user, "target": target, "last_item": last}),
        }
        report["steps"][names[j]] = row
    # chance levels for the eta^2 and kNN keys
    report["groups"] = {"users": int(user.max() + 1), "targets": int(len(np.unique(target))),
                        "last_items": int(len(np.unique(last)))}
    report["knn10_chance"] = {
        "user": float((np.bincount(user) ** 2).sum() / n ** 2),
        "target": float((np.unique(target, return_counts=True)[1] ** 2).sum() / n ** 2),
        "last_item": float((np.unique(last, return_counts=True)[1] ** 2).sum() / n ** 2)}
    cross = {}
    for a in range(k):
        for b in range(a + 1, k):
            za = std_steps[a] / np.linalg.norm(std_steps[a], axis=1, keepdims=True)
            zb = std_steps[b] / np.linalg.norm(std_steps[b], axis=1, keepdims=True)
            cross[f"{names[a]}-{names[b]}"] = {"same_row": float((za * zb).sum(1).mean()),
                                               "random_row": float((za[ra] * zb[rb]).sum(1).mean())}
    report["cross_step"] = cross
    text = json.dumps(report, indent=1)
    print(text)
    if out:
        with open(out, "w") as handle:
            handle.write(text)


if __name__ == "__main__":
    fire.Fire(main)
