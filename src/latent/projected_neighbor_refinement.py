"""Attempt-10 projection-guided, bounded inference-time thought correction.

The retrieval corpus is fixed train-split evidence.  Targets in the corpus are
used only as values after retrieval; neither query retrieval nor the gate sees
the held-out target.  The same layer runs during training and generation.
"""

from __future__ import annotations

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborRefinementLayer(nn.Module):
    """Checkpoint-compatible base for the projected correction path."""

    def __init__(self, base_attention: nn.Module, hidden_size: int, spec: dict) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.mode = spec.get("mode", "knn")
        self.topn = int(spec.get("topn", 32))
        key_dim = int(spec.get("key_dim", 256))
        value_hidden = int(spec.get("value_hidden", 512))
        bank_size = int(spec["bank_size"])
        n_items = int(spec["n_items"])
        item_dim = int(spec["item_dim"])
        self.register_buffer("bank_key", torch.zeros(bank_size, hidden_size, dtype=torch.float16))
        self.register_buffer("bank_mean", torch.zeros(hidden_size))
        self.register_buffer("bank_item", torch.zeros(bank_size, dtype=torch.int64))
        self.register_buffer("bank_time", torch.zeros(bank_size, dtype=torch.int64))
        self.register_buffer("item_emb", torch.zeros(n_items, item_dim, dtype=torch.float16))
        self.query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, key_dim))
        self.key = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, key_dim))
        self.value = nn.Sequential(
            nn.LayerNorm(hidden_size + item_dim),
            nn.Linear(hidden_size + item_dim, value_hidden),
            nn.GELU(), nn.Linear(value_hidden, hidden_size),
        )
        self.gate = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.scale = 1.0 / math.sqrt(key_dim)
        self.query_time = None
        self.last_attention_entropy = None
        self.last_refinement_ratio = None
        self.last_valid_fraction = None

    def reset_new_parameters(self) -> None:
        for module in (self.query, self.key, self.value, self.gate):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    @torch.no_grad()
    def _select(self, r: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "knn":
            q = F.normalize(r.float() - self.bank_mean, dim=-1).to(self.bank_key.dtype)
            score = (q @ self.bank_key.T).float()
        else:
            score = torch.rand(r.size(0), self.bank_key.size(0), device=r.device)
        if self.query_time is not None:
            eligible = self.bank_time[None, :] < self.query_time.to(self.bank_time.device)[:, None]
            score.masked_fill_(~eligible, float("-inf"))
        top, idx = score.topk(min(self.topn, score.size(1)), dim=1)
        return idx, torch.isfinite(top)


class ProjectedNeighborRefinementLayer(NeighborRefinementLayer):
    def __init__(self, base_attention: nn.Module, hidden_size: int, spec: dict) -> None:
        super().__init__(base_attention, hidden_size, spec)
        dim = int(spec.get("projection_dim", 256))
        self.user_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim, bias=False))
        self.target_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim, bias=False))
        self.user_weight = float(spec.get("user_weight", 0.5))
        self.max_delta_ratio = float(spec.get("max_delta_ratio", 0.1))
        if not 0 <= self.user_weight <= 1 or self.max_delta_ratio <= 0:
            raise ValueError("user_weight must be in [0,1] and max_delta_ratio must be positive")
        bank_size = int(spec["bank_size"])
        self.register_buffer("bank_raw", torch.zeros(bank_size, hidden_size, dtype=torch.float16))
        self.register_buffer("bank_user_key", torch.zeros(bank_size, dim, dtype=torch.float16))
        self.register_buffer("bank_target_key", torch.zeros(bank_size, dim, dtype=torch.float16))

    @torch.no_grad()
    def initialize_from_attempt10(self, user_projection: nn.Module, target_projection: nn.Module,
                                  raw: torch.Tensor) -> None:
        """Copy the trained heads and build persistent train-only retrieval keys."""
        if raw.shape != self.bank_raw.shape:
            raise ValueError(f"bank shape {tuple(raw.shape)} != {tuple(self.bank_raw.shape)}")
        self.user_projection.load_state_dict(user_projection.state_dict())
        self.target_projection.load_state_dict(target_projection.state_dict())
        self.bank_raw.copy_(raw.to(self.bank_raw))
        for start in range(0, len(raw), 1024):
            block = raw[start:start + 1024].to(self.bank_raw.device, dtype=next(self.user_projection.parameters()).dtype)
            self.bank_user_key[start:start + len(block)].copy_(
                F.normalize(self.user_projection(block).float(), dim=-1).to(self.bank_user_key.dtype)
            )
            self.bank_target_key[start:start + len(block)].copy_(
                F.normalize(self.target_projection(block).float(), dim=-1).to(self.bank_target_key.dtype)
            )
        for parameter in self.user_projection.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_projection.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _select(self, r: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode != "knn":
            return super()._select(r, mode)
        dtype = next(self.user_projection.parameters()).dtype
        query = r.to(dtype)
        user = F.normalize(self.user_projection(query).float(), dim=-1)
        target = F.normalize(self.target_projection(query).float(), dim=-1)
        score = self.user_weight * (user @ self.bank_user_key.float().T)
        score += (1 - self.user_weight) * (target @ self.bank_target_key.float().T)
        if self.query_time is not None:
            eligible = self.bank_time[None, :] < self.query_time.to(self.bank_time.device)[:, None]
            score.masked_fill_(~eligible, float("-inf"))
        top, index = score.topk(min(self.topn, score.size(1)), dim=1)
        return index, torch.isfinite(top)

    def refine(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the same correction to generated and counterfactual latents."""
        mode = os.environ.get("NEIGHBOR_REFINEMENT_MODE", self.mode)
        if mode == "off":
            return raw
        index, valid = self._select(raw, mode)
        dtype = self.query[1].weight.dtype
        neighbour = self.bank_raw[index].to(dtype)
        items = self.item_emb[self.bank_item[index]].to(dtype)
        query = raw.to(dtype)
        logits = torch.einsum("bk,bnk->bn", self.query(query), self.key(neighbour)).float() * self.scale
        logits = logits.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(logits, dim=-1).nan_to_num(0.0)
        values = self.value(torch.cat([neighbour, items], dim=-1))
        evidence = torch.einsum("bn,bnh->bh", alpha.to(values.dtype), values)
        delta = (torch.sigmoid(self.gate(query)) * evidence).float()
        max_norm = self.max_delta_ratio * raw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (max_norm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        corrected = raw.float() + delta * scale
        with torch.no_grad():
            self.last_attention_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1).mean()
            self.last_valid_fraction = valid.float().mean()
            self.last_refinement_ratio = (
                (corrected - raw.float()).norm(dim=-1) / raw.float().norm(dim=-1).clamp_min(1e-6)
            ).mean()
        return corrected.to(raw.dtype)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        raw = self.base_attention(hidden_states=hidden_states, attention_mask=attention_mask,
                                  thought_id_idx=thought_id_idx)
        return self.refine(raw)


def wrap_with_projected_neighbor_refinement(base_attention: nn.Module, hidden_size: int,
                                            spec: dict) -> ProjectedNeighborRefinementLayer:
    return ProjectedNeighborRefinementLayer(base_attention, hidden_size, spec)
