"""History-keyed, outcome-grounded training memories for R3.

Retrieval reads donor history keys only. Values encode donor history latents
and observed next-item embeddings. Every donor must be from another user and
have an outcome strictly earlier than the query's latest observed history event.
Missing eligibility metadata raises instead of silently opening the whole bank.
"""

import ast
import hashlib
import os

import torch
import torch.nn.functional as F

from projected_neighbor_refinement import ProjectedNeighborRefinementLayer
from reasoning_dataset import LatentRDataset


def user_key(user_id):
    """Stable signed-int64-compatible identity, shared by bank and all splits."""
    return int.from_bytes(hashlib.sha256(str(user_id).encode()).digest()[:8], "big") & ((1 << 63) - 1)


def history_cutoff(row):
    """Recommendation time from observed history, never the future target time."""
    times = ast.literal_eval(row["history_timestamp"])
    if not isinstance(times, (list, tuple)) or not times:
        raise ValueError("outcome memory requires nonempty history_timestamp")
    return max(int(t) for t in times)


class OutcomeMemoryDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        row = self.data.iloc[idx]
        example["query_time"] = history_cutoff(row)
        example["query_user"] = user_key(row["user_id"])
        return example


class OutcomeMemoryLayer(ProjectedNeighborRefinementLayer):
    def __init__(self, base_attention, hidden_size, spec):
        super().__init__(base_attention, hidden_size, spec)
        if self.topn <= 0 or self.bank_raw.shape[0] == 0:
            raise ValueError("outcome memory requires positive topn and a nonempty bank")
        self.register_buffer("bank_user", torch.zeros(len(self.bank_raw), dtype=torch.long))
        self.query_user = None
        self.retrieval_space = spec.get("retrieval_space", "projected")
        if self.retrieval_space not in {"projected", "history"}:
            raise ValueError("retrieval_space must be projected or history")
        self.last_gate_mean = None
        self.last_evidence_norm = None
        self.last_raw_norm = None
        self.last_delivered_refinement_ratio = None
        self.memory_value_mode = spec.get("memory_value_mode", "outcome")
        if self.memory_value_mode not in {"outcome", "latent_only", "shuffle_outcome"}:
            raise ValueError("unknown memory_value_mode")

    @torch.no_grad()
    def initialize_history_bank(self, raw):
        """Margin checkpoints have no collaborative heads: use history keys."""
        if raw.shape != self.bank_raw.shape:
            raise ValueError("history bank shape mismatch")
        self.bank_raw.copy_(raw.to(self.bank_raw))
        self.bank_mean.copy_(raw.float().mean(0).to(self.bank_mean))
        centered = raw.to(self.bank_key.device).float() - self.bank_mean
        self.bank_key.copy_(F.normalize(centered, dim=-1).to(self.bank_key))
        for module in (self.user_projection, self.target_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _expand(value, batch_size, device):
        value = torch.as_tensor(value, device=device).long().reshape(-1)
        if not len(value) or batch_size % len(value):
            raise ValueError("memory metadata must align with the prompt or expanded beams")
        return value.repeat_interleave(batch_size // len(value))

    def eligible(self, batch_size):
        if self.query_time is None or self.query_user is None:
            raise ValueError("outcome memory requires query_time and query_user for retrieval")
        times = self._expand(self.query_time, batch_size, self.bank_time.device)
        users = self._expand(self.query_user, batch_size, self.bank_user.device)
        return ((self.bank_time[None, :] < times[:, None]) &
                (self.bank_user[None, :] != users[:, None]))

    @torch.no_grad()
    def _select(self, raw, mode):
        eligible = self.eligible(len(raw))
        if mode == "knn" and self.retrieval_space == "history":
            query = F.normalize(raw.float() - self.bank_mean, dim=-1)
            score = query @ self.bank_key.float().T
        elif mode == "knn":
            dtype = next(self.user_projection.parameters()).dtype
            query = raw.to(dtype)
            user = F.normalize(self.user_projection(query).float(), dim=-1)
            target = F.normalize(self.target_projection(query).float(), dim=-1)
            # Both projections consume history-only raw latents; donor outcomes
            # enter neither retrieval scores nor the query.
            score = self.user_weight * (user @ self.bank_user_key.float().T)
            score += (1 - self.user_weight) * (target @ self.bank_target_key.float().T)
        elif mode == "random":
            score = torch.rand(eligible.shape, device=raw.device)
        else:
            raise ValueError(f"unknown outcome-memory retrieval mode: {mode}")
        score.masked_fill_(~eligible, float("-inf"))
        top, index = score.topk(min(self.topn, score.size(1)), dim=1)
        return index, torch.isfinite(top)

    def refine(self, raw):
        mode = os.environ.get("NEIGHBOR_REFINEMENT_MODE", self.mode)
        if mode == "off":
            return raw
        index, valid = self._select(raw, mode)
        dtype = self.query[1].weight.dtype
        neighbour = self.bank_raw[index].to(dtype)
        items = self.item_emb[self.bank_item[index]].to(dtype)
        value_mode = os.environ.get("OUTCOME_MEMORY_VALUE_MODE", self.memory_value_mode)
        if value_mode == "latent_only":
            items = torch.zeros_like(items)
        elif value_mode == "shuffle_outcome":
            # Permute outcomes only among this query's eligible selected donors.
            # No future or same-user outcome may enter even a diagnostic control.
            items = items.clone()
            for b in range(len(raw)):
                slots = torch.nonzero(valid[b]).flatten()
                items[b, slots] = items[b, slots.roll(1)]
        elif value_mode != "outcome":
            raise ValueError(f"unknown outcome-memory value mode: {value_mode}")
        query = raw.to(dtype)
        logits = torch.einsum("bk,bnk->bn", self.query(query), self.key(neighbour)).float() * self.scale
        # A fully ineligible row must have finite gradients and be an identity.
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1) * valid.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        # R3 history latents and title embeddings have very different norms.
        # Balance the two evidence sources before learning their relationship.
        memory_input = torch.cat([F.normalize(neighbour.float(), dim=-1),
                                  F.normalize(items.float(), dim=-1)], dim=-1).to(dtype)
        values = self.value(memory_input)
        evidence = torch.einsum("bn,bnh->bh", alpha.to(values.dtype), values)
        gate = torch.sigmoid(self.gate(query))
        delta = (gate * evidence).float()
        max_norm = self.max_delta_ratio * raw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (max_norm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        corrected = raw.float() + delta * scale
        delivered = corrected.to(raw.dtype)
        with torch.no_grad():
            self.last_gate_mean = gate.float().mean()
            self.last_evidence_norm = evidence.float().norm(dim=-1).mean()
            self.last_raw_norm = raw.float().norm(dim=-1).mean()
            self.last_delivered_refinement_ratio = ((delivered.float() - raw.float()).norm(dim=-1) /
                                                     raw.float().norm(dim=-1).clamp_min(1e-6)).mean()
            self.last_attention_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1).mean()
            self.last_valid_fraction = valid.float().mean()
            self.last_refinement_ratio = ((corrected - raw.float()).norm(dim=-1) /
                                          raw.float().norm(dim=-1).clamp_min(1e-6)).mean()
        return delivered
