#!/usr/bin/env python3
"""Exact decoder posterior over CF-cluster evidence (E2-diag; refine-logs/EXPERIMENT_PLAN.md).

For each row and each candidate cluster c, the teacher-forced log-likelihood of the
target title under the ``oracle_cfcluster150`` decoder with c in the thought slot:

    q(c | x, y) ∝ p_θ(y | x, c)          (uniform prior; the selector's prior is applied later)

plus the null option ∅ = log p(y | x) under plain R3 v1 (its own thought, no evidence).
Candidates: true cluster ∪ top-15 of the mix and grufull predictors ∪ 8 random clusters.

Writes ``<out>/<split>.pt`` with, per row: candidate clusters, their summed target
log-probs, the null log-prob, the target cluster, the target token count, R3 v1's hit
flag, and both models' base thoughts (features for the E3 selector).

Usage: python cluster_posterior.py --cluster_model ... --r3_model ... --split test --rows 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import experiment as E
from side_info_eval import VariantPromptDataset, load, lookup_component

sys.path.insert(0, str(E.ROOT / "src" / "collab_reasoning"))
import data as D  # noqa: E402

PREDICTORS = {"mix": "cluster_mix_{tag}.pt", "grufull": "cluster_grufull_{tag}.pt"}


def target_logp(model, embeds, mask, labels) -> torch.Tensor:
    """Summed log-probability of the supervised target tokens, per example."""
    with torch.no_grad():
        hidden = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, :-1]).float()
    tok = labels[:, 1:]
    valid = tok.ne(-100)
    lp = F.log_softmax(logits, dim=-1).gather(-1, tok.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (lp * valid).sum(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_model", type=Path, required=True)
    p.add_argument("--r3_model", type=Path, required=True)
    p.add_argument("--split", choices=["valid", "test"], required=True)
    p.add_argument("--rows", type=int, default=-1)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--n_random", type=int, default=8)
    p.add_argument("--table_tag", default="full")
    p.add_argument("--row_batch", type=int, default=4)
    p.add_argument("--fwd_batch", type=int, default=32)
    p.add_argument("--r3_hits", type=Path, required=True,
                   help="test: cluster_paths/full/paths.json; valid: eval/v1_valid_predictions/predictions.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    a.out.mkdir(parents=True, exist_ok=True)

    cmodel, tokenizer, device, spec = load(a.cluster_model)
    if spec is None or spec["kind"] != "oracle_category":
        raise ValueError("cluster_model must be an oracle_category side-information checkpoint")
    cid = torch.load(spec["source"], map_location="cpu", weights_only=False)["cluster_id"].numpy()
    k = int(cid.max()) + 1
    rep = np.full(k, -1, np.int64)  # any member item: every member's side vector is the cluster centroid
    for it in range(len(cid)):
        if cid[it] >= 0 and rep[cid[it]] < 0:
            rep[cid[it]] = it
    r3, _, _ = E.load_model(a.r3_model)

    ds = VariantPromptDataset(a.split, tokenizer, a.rows)
    n = len(ds)
    table_dir = D.DATA / "info" / "collab_reasoning"
    probs = {name: torch.load(table_dir / f.format(tag=a.table_tag), weights_only=False)[a.split][:n].float().numpy()
             for name, f in PREDICTORS.items()}

    hits = json.load(a.r3_hits.open())
    if a.split == "test":
        r3_top = [h["paths"]["r3"]["titles"] for h in hits]
    else:
        r3_top = [[t.strip(' \n"') for t in h["top10"]] for h in hits]
    if len(r3_top) < n:
        raise ValueError(f"{a.r3_hits} has {len(r3_top)} rows, need {n}")
    r3_hit = np.array([ds.rows[i]["target_title"].strip(' \n"') in r3_top[i] for i in range(n)])

    rng = np.random.default_rng(0)
    target_cluster = np.array([int(cid[r["target_item"]]) for r in ds.rows])
    cands = []
    for i in range(n):
        s = {int(c) for pr in probs.values() for c in np.argsort(-pr[i])[:a.topk]}
        if target_cluster[i] >= 0:
            s.add(int(target_cluster[i]))
        s |= {int(c) for c in rng.choice(k, size=a.n_random, replace=False)}
        cands.append(sorted(c for c in s if rep[c] >= 0))

    pad = tokenizer.pad_token_id
    logp = [None] * n
    null_logp = np.zeros(n, np.float32)
    n_tok = np.zeros(n, np.int64)
    r_cluster, r_v1 = [], []
    for s in range(0, n, a.row_batch):
        rows = ds.rows[s:s + a.row_batch]
        b = E.collate(rows, pad)
        ids, mask, labels, pos = (b[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos"))
        n_tok[s:s + len(rows)] = labels[:, 1:].ne(-100).sum(1).cpu().numpy()
        rv, emb_v = E.thought_embeds(r3, ids, mask, pos)
        null_logp[s:s + len(rows)] = target_logp(r3, emb_v, mask, labels).cpu().numpy()
        rb, emb = E.thought_embeds(cmodel, ids, mask, pos)
        r_cluster.append(rb.half().cpu())
        r_v1.append(rv.half().cpu())
        jobs = [(j, c) for j in range(len(rows)) for c in cands[s + j]]
        out = []
        for t in range(0, len(jobs), a.fwd_batch):
            chunk = jobs[t:t + a.fwd_batch]
            jj = torch.tensor([j for j, _ in chunk], device=device)
            idx = torch.tensor([rep[c] for _, c in chunk])
            vec = cmodel.combine(rb[jj], lookup_component(cmodel, idx)).float()
            out.append(target_logp(cmodel, E.substitute(emb[jj], pos[jj], vec), mask[jj], labels[jj]).cpu())
        out = torch.cat(out).numpy()
        o = 0
        for j in range(len(rows)):
            m = len(cands[s + j])
            logp[s + j] = out[o:o + m].astype(np.float32)
            o += m
        if s % (a.row_batch * 100) == 0:
            print(f"{a.split} {s + len(rows)}/{n}", flush=True)

    torch.save({
        "split": a.split, "candidates": cands, "logp": logp, "null_logp": null_logp, "n_tokens": n_tok,
        "target_cluster": target_cluster, "r3_hit": r3_hit,
        "r_cluster": torch.cat(r_cluster), "r_v1": torch.cat(r_v1),
        "predictors": {name: pr for name, pr in probs.items()},
    }, a.out / f"{a.split}.pt")
    print(f"wrote {a.out / (a.split + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
