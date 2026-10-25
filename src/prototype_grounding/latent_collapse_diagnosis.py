#!/usr/bin/env python3
"""Why is r near-constant? Decompose R3's attention pooling on a checkpoint.

r = softmax(q_{t-1} K^T / sqrt(d)) V over positions < <|Thought|>, with
V = W_v h + b_v on the LLM's last hidden states h. The instruction prefix is
identical in every row, so (causal LM) its hidden states are identical too:
any attention mass on the prefix contributes the same vector to every r.

Per row this measures:
  * hidden-state norms at position 0, the rest of the shared prefix and the
    user-input tokens;
  * pooling attention mass on each of these regions, entropy, argmax position;
  * r split into prefix and user-token contributions, and how much of r's
    sample-specific variance each carries;
  * r recomputed in fp32 (bf16 hidden states upcast, fp32 weights): if the
    collapse is bf16 rounding, fp32 r should have many more distinct values;
  * counterfactual pooling restricted to user-input tokens (fp32), with the
    trained module and with a freshly initialised one (collapse at init?).
Nothing is trained or modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as E
from model import apply_rotary_pos_emb  # src/grpo_attention_tuning, put on sys.path by experiment


def pool(module, hidden, valid, query_pos, return_weights=False):
    """Re-implementation of SelfAttentionLayer.forward with an arbitrary key mask."""
    cos, sin = module.rope(hidden, position_ids=torch.arange(hidden.size(1), device=hidden.device)[None])
    q, k, v = module.query(hidden), module.key(hidden), module.value(hidden)
    q, k = apply_rotary_pos_emb(q, k, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
    rows = torch.arange(hidden.size(0), device=hidden.device)
    scores = (q[rows, query_pos][:, None] @ k.transpose(-2, -1)).squeeze(1) * module.scale_score
    w = scores.float().masked_fill(~valid, -float("inf")).softmax(-1).to(v.dtype)
    out = (w[:, None] @ v).squeeze(1)
    return (out, w, v) if return_weights else out


def spread(x: torch.Tensor) -> dict:
    x = x.float()
    c = x - x.mean(0)
    return {"norm": float(x.norm(dim=1).mean()), "residual_norm": float(c.norm(dim=1).mean()),
            "residual_over_norm": float((c.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)).mean()),
            "distinct_vectors": int(torch.unique(x, dim=0).size(0)),
            "per_dim_residual_std": float(c.std())}


def run(checkpoint: Path, split: str, limit: int, batch_size: int, tag: str) -> None:
    model, tokenizer, device = E.load_model(checkpoint)
    ds = E.PromptDataset(split, tokenizer, limit)
    a, b = ds.rows[0]["input_ids"], ds.rows[1]["input_ids"]
    prefix = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: E.collate(x, tokenizer.pad_token_id))
    trained = model.attention
    trained32 = copy.deepcopy(trained).float()
    torch.manual_seed(0)
    fresh32 = type(trained)(trained.hidden_size).to(device).float()

    acc = {k: [] for k in ("norm_pos0", "norm_prefix", "norm_user", "mass_pos0", "mass_prefix", "mass_user",
                           "entropy", "argmax_pos", "user_tokens")}
    vecs = {k: [] for k in ("r_bf16", "r_fp32", "r_prefix_part", "r_user_part",
                            "r_user_only_trained", "r_user_only_fresh", "r_full_fresh")}
    for step, batch in enumerate(loader):
        ids, mask, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        with torch.no_grad():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            r_ref = trained(hidden, mask, pos)
            span = torch.arange(ids.size(1), device=device)[None]
            before = span < pos[:, None]
            user = before & (span >= prefix)
            shared = before & (span < prefix)
            qpos = pos - 1
            r_bf16, w, v = pool(trained, hidden, before, qpos, return_weights=True)
            assert torch.allclose(r_bf16.float(), r_ref.float(), atol=1e-2, rtol=1e-2), "re-implementation mismatch"
            h32 = hidden.float()
            r32, w32, v32 = pool(trained32, h32, before, qpos, return_weights=True)
            wp = (w32 * shared)
            wu = (w32 * user)
            vecs["r_prefix_part"].append((wp[:, None] @ v32).squeeze(1).cpu())
            vecs["r_user_part"].append((wu[:, None] @ v32).squeeze(1).cpu())
            vecs["r_bf16"].append(r_ref.float().cpu())
            vecs["r_fp32"].append(r32.cpu())
            vecs["r_user_only_trained"].append(pool(trained32, h32, user, qpos).cpu())
            vecs["r_user_only_fresh"].append(pool(fresh32, h32, user, qpos).cpu())
            vecs["r_full_fresh"].append(pool(fresh32, h32, before, qpos).cpu())
            n = hidden.float().norm(dim=-1)
            acc["norm_pos0"].append(n[:, 0].cpu())
            acc["norm_prefix"].append(((n * shared).sum(1) / shared.sum(1)).cpu())
            acc["norm_user"].append(((n * user).sum(1) / user.sum(1)).cpu())
            acc["mass_pos0"].append(w32[:, 0].cpu())
            acc["mass_prefix"].append(wp.sum(1).cpu())
            acc["mass_user"].append(wu.sum(1).cpu())
            acc["entropy"].append(-(w32.clamp_min(1e-30).log() * w32).sum(1).cpu())
            acc["argmax_pos"].append(w32.argmax(1).float().cpu())
            acc["user_tokens"].append(user.sum(1).float().cpu())
        if step % 50 == 0:
            print(f"{split}: {min((step + 1) * batch_size, len(ds))}/{len(ds)}", flush=True)

    acc = {k: torch.cat(v) for k, v in acc.items()}
    vecs = {k: torch.cat(v) for k, v in vecs.items()}
    prefix_part, user_part = vecs["r_prefix_part"], vecs["r_user_part"]
    total_var = float((vecs["r_fp32"] - vecs["r_fp32"].mean(0)).square().sum())
    argmax = acc["argmax_pos"].long()
    report = {
        "checkpoint": str(checkpoint), "split": split, "n": len(ds), "shared_prefix_tokens": prefix,
        "hidden_norm": {k: float(acc[f"norm_{k}"].mean()) for k in ("pos0", "prefix", "user")},
        "attention_mass": {k: float(acc[f"mass_{k}"].mean()) for k in ("pos0", "prefix", "user")},
        "attention_entropy_nats": float(acc["entropy"].mean()),
        "uniform_entropy_nats": float(acc["user_tokens"].add(prefix).log().mean()),
        "argmax_position_counts": {str(int(p)): int((argmax == p).sum()) for p in torch.unique(argmax)[:20]},
        "value_bias_norm": float(trained32.value.bias.norm()),
        "contribution_norm": {"prefix": float(prefix_part.norm(dim=1).mean()),
                              "user": float(user_part.norm(dim=1).mean())},
        # Share of r's sample-specific sum of squares carried by each part
        # (cross term reported separately; the three add to 1).
        "residual_variance_share": {
            "prefix": float((prefix_part - prefix_part.mean(0)).square().sum()) / total_var,
            "user": float((user_part - user_part.mean(0)).square().sum()) / total_var,
            "cross": 2 * float(((prefix_part - prefix_part.mean(0)) * (user_part - user_part.mean(0))).sum()) / total_var,
        },
        "spread": {k: spread(v) for k, v in vecs.items()},
        "fp32_vs_bf16_max_abs_diff": float((vecs["r_fp32"] - vecs["r_bf16"]).abs().max()),
    }
    def finite(o, path=""):
        if isinstance(o, dict):
            return {k: finite(v, f"{path}.{k}") for k, v in o.items()}
        if isinstance(o, float) and not np.isfinite(o):
            print(f"non-finite value at {path}", flush=True)
            return None
        return o

    report = finite(report)
    out = E.ANALYSIS / "results/latent_collapse_diagnosis" / tag / f"{split}.json"
    E.atomic_json(out, report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Toys_and_Games")
    p.add_argument("--split", choices=("train", "valid", "test"), default="test")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    E.configure_category(a.category)
    E.seed_all(42)
    run(a.checkpoint, a.split, a.limit, a.batch_size, a.tag)


if __name__ == "__main__":
    main()
