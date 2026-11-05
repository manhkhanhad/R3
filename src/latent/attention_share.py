"""Guaranteed attention share for the latent thought r (plain R3, prompt unchanged).

In every decoder layer and head, each query position t after the thought
position p gives r an explicit share rho of its attention; the other positions
share the rest in proportion to their usual softmax weights:

    w'_tj = (1 - rho_t) * a_tj / (1 - a_tp)   (j != p),     w'_tp = rho_t

with a = the ordinary causal softmax (including p).  Positions up to p are
unchanged.  rho is not a trainable parameter (a learnable share would shrink
back to ~0 because r is redundant with the history):

* ``fixed``:  rho_t = rho_bar for every layer, head and query position.
* ``budget``: the layer's budget rho_bar is distributed over its heads in
  proportion to their own softmax attention to r,
      rho_t^(h) = min(1, rho_bar * a_tp^(h) / mean_h' a_tp^(h')),
  so every query position gives r rho_bar on average over the heads (before
  clipping) while heads that naturally read r more get more of it.  Defined per
  query position, so training (teacher forcing) and KV-cache decoding agree.

Implemented as an attention function registered under the name
``latent_share`` (eager attention + the reweighting); the thought position per
row is set by ``LatentModel.forward`` at prefill (``set_share_positions``).
"""

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

NAME = "latent_share"


class ShareState:
    def __init__(self, mode, rho):
        if mode not in ("fixed", "budget"):
            raise ValueError(f"unknown attention share mode {mode}")
        self.mode, self.rho = mode, float(rho)
        self.positions = None      # [B] key index of the thought token (-1: none)
        self.last_share = None     # diagnostics: mean realised share per layer


def reweight(weights, positions, q_len, state):
    """weights [B, H, T, S] (fp32 softmax) -> reweighted copy."""
    b, h, t, s = weights.shape
    if positions is None:
        return weights
    pos = positions.to(weights.device).long()
    if pos.numel() != b:  # beam search expanded the batch after prefill
        pos = pos.repeat_interleave(b // pos.numel())
    qpos = torch.arange(s - q_len, s, device=weights.device)                  # absolute query positions
    active = (qpos[None, :] > pos[:, None]) & (pos[:, None] >= 0)            # [B, T]
    if not bool(active.any()):
        return weights
    idx = pos.clamp_min(0).view(b, 1, 1, 1).expand(b, h, t, 1)
    a_p = weights.gather(-1, idx).squeeze(-1)                                 # [B, H, T]
    if state.mode == "fixed":
        rho = torch.full_like(a_p, state.rho)
    else:
        rho = (state.rho * a_p / a_p.mean(1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    scale = (1 - rho) / (1 - a_p).clamp_min(1e-6)
    new = weights * scale.unsqueeze(-1)
    new = new.scatter(-1, idx, rho.unsqueeze(-1))
    mask = active[:, None, :, None]
    return torch.where(mask, new, weights)


def share_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    state = module._latent_share
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    weights = reweight(weights, state.positions, query.shape[2], state)
    weights = weights.to(query.dtype)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return out, weights


def enable_attention_share(model):
    """Switch every decoder layer to the share attention (config.latent_attention_share)."""
    spec = model.config.latent_attention_share
    ALL_ATTENTION_FUNCTIONS[NAME] = share_attention_forward
    state = ShareState(spec["mode"], spec["rho"])
    model._latent_share = state
    model.config._attn_implementation = NAME
    for layer in model.model.layers:
        layer.self_attn._latent_share = state
    return state


def set_share_positions(model, input_ids, thought_id):
    """Record the (first) thought position of every row; -1 where there is none."""
    hit = input_ids == thought_id
    pos = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(hit[:, 0], -1, dtype=torch.long))
    model._latent_share.positions = pos.long()
