#!/usr/bin/env python3
"""Geometric collapse of the thought vector r for several checkpoints.

On the first ``--limit`` test rows, per checkpoint:
- mean ||r||, mean cosine to the mean r, mean ||r - mean_r|| / ||r|| (sample-specific share);
- mean pairwise cosine between different rows;
- effective rank of the centered r matrix: exp(entropy of normalized singular values),
  and the participation ratio (sum s^2)^2 / sum s^4;
- variance share of the top-1 / top-10 principal components;
- near-duplicate share: rows whose nearest other r has cosine > 0.999.

Usage: python latent_geometry.py --checkpoints name=path ... --out geometry.json
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import experiment as E
from mux_eval import load
from side_info_eval import VariantPromptDataset


def geometry(r: torch.Tensor) -> dict:
    r = r.double()
    mean = r.mean(0)
    unit = F.normalize(r, dim=-1)
    cos_mean = (unit @ F.normalize(mean, dim=0)).mean()
    spec = ((r - mean).norm(dim=-1) / r.norm(dim=-1)).mean()
    sims = unit @ unit.T
    n = len(r)
    off = sims[~torch.eye(n, dtype=torch.bool)]
    nn_sim = sims.masked_fill(torch.eye(n, dtype=torch.bool), -2).max(1).values
    s = torch.linalg.svdvals(r - mean)
    p = s / s.sum()
    var = s ** 2 / (s ** 2).sum()
    return {
        "n": n, "norm_mean": float(r.norm(dim=-1).mean()), "cos_to_mean": float(cos_mean),
        "sample_specific_share": float(spec), "pairwise_cos_mean": float(off.mean()),
        "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())),
        "participation_ratio": float((s ** 2).sum() ** 2 / (s ** 4).sum()),
        "top1_var_share": float(var[0]), "top10_var_share": float(var[:10].sum()),
        "near_duplicate_share": float((nn_sim > 0.999).double().mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    out = {}
    for spec in a.checkpoints:
        name, path = spec.split("=", 1)
        model, tokenizer, device = load(path)
        ds = VariantPromptDataset("test", tokenizer, a.limit)
        rs = []
        with torch.no_grad():
            for s in range(0, len(ds), a.batch_size):
                b = E.collate(ds.rows[s:s + a.batch_size], tokenizer.pad_token_id)
                r, _ = E.thought_embeds(model, b["input_ids"].to(device), b["attention_mask"].to(device), b["thought_pos"].to(device))
                rs.append(r.float().cpu())
        out[name] = geometry(torch.cat(rs))
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
        E.atomic_json(E.Path(a.out), out)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
