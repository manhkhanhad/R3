#!/usr/bin/env python3
"""Trace where the thought vector's large norm comes from.

r = W_V · (Σ_i α_i h_i) + b_V, where h_i are final-layer hidden states taken
after the backbone's final RMSNorm. This reports the norm at each stage, the
final-norm gain, the value projection's gain, and input-embedding norms, for
trained checkpoints and for the base model with a freshly initialized pooling
layer (the state R3 training starts from).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment as E


def trace(model, tokenizer, device, limit: int) -> dict:
    ds = E.PromptDataset("test", tokenizer, limit)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=lambda x: E.collate(x, tokenizer.pad_token_id))
    att = model.attention
    h_norm, pooled_norm, r_norm, pooled_cos, max_rel_err = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state.float()
            for b in range(len(ids)):
                t = int(pos[b])
                h_norm.append(h[b, :t].norm(dim=-1).mean().item())
                hb = torch.nn.functional.normalize(h[b, :t], dim=-1)
                pooled_cos.append(((hb @ hb.T).sum() - t) .item() / max(t * (t - 1), 1))
            # reproduce SelfAttentionLayer: pooled = Σ α V(h) = W_V (Σ α h) + b_V (weights sum to 1)
            wmask = att.mask_to_weights(mask.clone(), pos)
            cos, sin = att.rope(h, position_ids=torch.arange(h.shape[1], device=device).unsqueeze(0))
            Q = att.query(h.to(att.query.weight.dtype)).float()
            K = att.key(h.to(att.key.weight.dtype)).float()
            Q, K = E_apply_rope(Q, K, cos, sin)
            q = Q[torch.arange(len(ids)), pos - 1].unsqueeze(1)
            w = torch.softmax((q @ K.transpose(-2, -1)) * att.scale_score + wmask.unsqueeze(1), dim=-1)
            pooled = (w @ h).squeeze(1)
            r = att.value(pooled.to(att.value.weight.dtype)).float()
            pooled_norm += pooled.norm(dim=-1).tolist()
            r_norm += r.norm(dim=-1).tolist()
            r_ref = att(h.to(att.value.weight.dtype), mask, pos).float()
            max_rel_err.append(((r - r_ref).norm(dim=-1) / r_ref.norm(dim=-1)).max().item())
    emb = model.model.embed_tokens.weight.float()
    g = model.model.norm.weight.float()
    W = att.value.weight.float()
    mean = lambda x: float(sum(x) / len(x))
    return {
        "input_embedding_norm_mean": float(emb.norm(dim=-1).mean()),
        "final_norm_gain_mean": float(g.mean()), "final_norm_gain_abs_max": float(g.abs().max()),
        "sqrt_hidden_size": float(emb.shape[1] ** 0.5),
        "final_hidden_norm_mean": mean(h_norm),
        "final_hidden_mean_pairwise_cosine": mean(pooled_cos),
        "pooled_norm_mean": mean(pooled_norm),
        "r_norm_mean": mean(r_norm),
        "value_gain_r_over_pooled": mean(r_norm) / mean(pooled_norm),
        "value_weight_frobenius": float(W.norm()), "value_weight_spectral": float(torch.linalg.matrix_norm(W, ord=2)),
        "value_bias_norm": float(att.value.bias.float().norm()),
        "reconstruction_max_rel_err": max(max_rel_err),
    }


def E_apply_rope(Q, K, cos, sin):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0).float(), sin.squeeze(0).float(), unsqueeze_dim=0)
    return Q, K


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="tag=path, or tag=base for fresh init")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    results = {}
    for spec in a.checkpoints:
        tag, path = spec.split("=", 1)
        if path == "base":
            # Base backbone + freshly initialized pooling layer, as at the start of R3 SFT.
            ref = Path(a.checkpoints[0].split("=", 1)[1])  # tokenizer with the <|Thought|> token
            model, tokenizer, device = E.load_model(ref)
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16).to(device)
            base.resize_token_embeddings(model.model.embed_tokens.num_embeddings)
            model.model.load_state_dict(base.model.state_dict())
            torch.manual_seed(0)
            for lin in (model.attention.query, model.attention.key, model.attention.value):
                lin.reset_parameters()
            del base
        else:
            model, tokenizer, device = E.load_model(Path(path))
        results[tag] = trace(model, tokenizer, device, a.limit)
        print(tag, json.dumps(results[tag], indent=1), flush=True)
        del model
        torch.cuda.empty_cache()
    E.atomic_json(a.out, results)


if __name__ == "__main__":
    main()
