"""Behaviour-grounded two-step latent reasoning for plain R3.

Step 1 (own intuition):   r0 = R3 pooling over the user's own history.
Step 2 (grounding):       r1 = r0 + g * ||r0|| * dir( sum_j a_j * V(r0_j, e(y_j)) )

Donors j are *other users* whose observed next item y_j happened strictly
before the query's latest history event.  A donor contributes its own reasoning
r0_j together with the embedding of what that user actually did next, so the
evidence is behavioural, not another summary of similar text.

Two training signals beyond recommendation CE:

* Item grounding (step 2 supervision): r1 must score the true next item over the
  whole catalogue, ``CE(normalize(P(r1)) @ E^T / tau_item, y)``.
* Behaviour-supervised retrieval: the usefulness of donor j for query i is how
  much grounding on j alone raises the log-probability of i's *observed* next
  item, ``u_j = log p(y_i | r1^(j)) - log p(y_i | r0)``.  The retriever's
  distribution over candidates is distilled toward ``softmax(u / tau_util)``.

Donors come from a run-local FIFO of training rows while training (every row is
accepted: observed outcomes need no correctness gate) and from a persistent
bank extracted over the full train split after training for inference.
"""

import ast
import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel
from reasoning_dataset import LatentRDataset

GROUNDING_MODES = {"knn", "off", "random", "latent_only", "shuffle_outcome"}


def user_key(user_id):
    """Stable non-negative int64 identity shared by the bank and every split."""
    return int.from_bytes(hashlib.sha256(str(user_id).encode()).digest()[:8], "big") & ((1 << 63) - 1)


def history_cutoff(row):
    """Recommendation time = latest observed history event, never the target time."""
    times = ast.literal_eval(row["history_timestamp"]) if isinstance(row["history_timestamp"], str) \
        else row["history_timestamp"]
    if not times:
        raise ValueError("grounded reasoning requires a nonempty history_timestamp")
    return max(int(t) for t in times)


def row_metadata(row):
    return {
        "query_time": history_cutoff(row),
        "query_user": user_key(row["user_id"]),
        "target_item": int(row["item_id"]),
        "outcome_time": int(row["timestamp"]),
    }


class GroundedReasoningDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        if not self.test:
            example.update(row_metadata(self.data.iloc[idx]))
        return example


class GroundedReasoningLayer(nn.Module):
    def __init__(self, base_attention, hidden_size, spec):
        super().__init__()
        self.base_attention = base_attention
        self.topn = int(spec.get("topn", 32))
        self.explore = int(spec.get("explore", 8))
        self.tau_item = float(spec.get("tau_item", 0.07))
        self.tau_util = float(spec.get("tau_util", 0.5))
        self.mode = spec.get("mode", "knn")
        key_dim = int(spec.get("key_dim", 256))
        value_hidden = int(spec.get("value_hidden", 1024))
        n_items, item_dim = int(spec["n_items"]), int(spec["item_dim"])
        bank_size, queue_size = int(spec["bank_size"]), int(spec.get("queue_size", 20_000))

        self.query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, key_dim))
        self.key = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, key_dim))
        self.value = nn.Sequential(
            nn.LayerNorm(hidden_size + item_dim), nn.Linear(hidden_size + item_dim, value_hidden),
            nn.GELU(), nn.Linear(value_hidden, hidden_size),
        )
        self.item_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, item_dim))
        # One global, unbounded evidence scale; no per-sample gate that can close.
        self.evidence_scale = nn.Parameter(torch.tensor(float(spec.get("init_scale", 0.5))))
        self.key_scale = 1.0 / math.sqrt(key_dim)

        self.register_buffer("item_emb", torch.zeros(n_items, item_dim, dtype=torch.float16))
        # Persistent inference bank, filled from the final model after training.
        self.register_buffer("bank_r", torch.zeros(bank_size, hidden_size, dtype=torch.float16))
        self.register_buffer("bank_item", torch.zeros(bank_size, dtype=torch.int64))
        self.register_buffer("bank_user", torch.zeros(bank_size, dtype=torch.int64))
        self.register_buffer("bank_time", torch.zeros(bank_size, dtype=torch.int64))
        self.register_buffer("bank_ready", torch.zeros((), dtype=torch.bool))
        # Run-local training FIFO; never saved.
        for name, shape, dtype in (
            ("queue_r", (queue_size, hidden_size), torch.float16),
            ("queue_item", (queue_size,), torch.int64),
            ("queue_user", (queue_size,), torch.int64),
            ("queue_time", (queue_size,), torch.int64),
        ):
            self.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.int64), persistent=False)

        # Per-call query metadata, set by the model (training) or eval script.
        self.query_time = None
        self.query_user = None
        self.target_item = None
        self.last_raw = None
        self.aux_loss = None
        self.stats = {}

    @torch.no_grad()
    def reset_new_parameters(self, init_scale=0.5):
        """New modules are absent from base checkpoints; initialise them explicitly."""
        for module in (self.query, self.key, self.value, self.item_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.evidence_scale.fill_(init_scale)

    # ----------------------------------------------------------------- donors
    def _donors(self):
        if bool(self.bank_ready):
            return self.bank_r, self.bank_item, self.bank_user, self.bank_time
        n = int(self.queue_count)
        return self.queue_r[:n], self.queue_item[:n], self.queue_user[:n], self.queue_time[:n]

    @torch.no_grad()
    def push(self, r, item, user, time):
        n, size = r.size(0), self.queue_r.size(0)
        pointer = int(self.queue_pointer)
        index = (torch.arange(n, device=r.device) + pointer) % size
        self.queue_r[index] = r.detach().to(self.queue_r.dtype)
        self.queue_item[index] = item.to(self.queue_item)
        self.queue_user[index] = user.to(self.queue_user)
        self.queue_time[index] = time.to(self.queue_time)
        self.queue_pointer.fill_((pointer + n) % size)
        self.queue_count.fill_(min(size, int(self.queue_count) + n))

    @staticmethod
    def _expand(value, batch_size, device):
        value = torch.as_tensor(value, device=device).long().reshape(-1)
        if len(value) == 0 or batch_size % len(value):
            raise ValueError("grounding metadata must align with the prompts or expanded beams")
        return value.repeat_interleave(batch_size // len(value))

    def _eligible(self, batch_size, donor_user, donor_time):
        if self.query_time is None or self.query_user is None:
            raise ValueError("grounded reasoning needs query_time and query_user")
        time = self._expand(self.query_time, batch_size, donor_time.device)
        user = self._expand(self.query_user, batch_size, donor_user.device)
        # Only other users' outcomes that were already observed at query time.
        return (donor_time[None, :] < time[:, None]) & (donor_user[None, :] != user[:, None])

    # -------------------------------------------------------------- reasoning
    def item_log_probs(self, r):
        projected = F.normalize(self.item_head(r.to(self.item_head[1].weight.dtype)).float(), dim=-1)
        logits = projected @ F.normalize(self.item_emb.float(), dim=-1).T / self.tau_item
        return logits.log_softmax(-1)

    def _ground(self, r0, values, alpha):
        evidence = torch.einsum("bn,bnh->bh", alpha.to(values.dtype), values).float()
        direction = F.normalize(evidence, dim=-1)
        return r0.float() + self.evidence_scale * r0.float().norm(dim=-1, keepdim=True) * direction

    def refine(self, r0):
        mode = os.environ.get("GROUNDING_MODE", self.mode)
        if mode not in GROUNDING_MODES:
            raise ValueError(f"unknown GROUNDING_MODE {mode}")
        self.aux_loss = None
        donor_r, donor_item, donor_user, donor_time = self._donors()
        if mode == "off" or donor_r.size(0) == 0:
            return r0
        batch = r0.size(0)
        eligible = self._eligible(batch, donor_user, donor_time)
        dtype = self.query[1].weight.dtype
        q = self.query(r0.to(dtype)).float()

        with torch.no_grad():
            if mode == "random":
                score = torch.rand(eligible.shape, device=r0.device)
            else:
                score = q @ self.key(donor_r.to(dtype)).float().T * self.key_scale
            score = score.masked_fill(~eligible, float("-inf"))
            top, index = score.topk(min(self.topn, score.size(1)), dim=1)
            valid = torch.isfinite(top)
            if self.training and self.explore:
                noise = torch.rand(eligible.shape, device=r0.device).masked_fill(~eligible, -1.0)
                noise.scatter_(1, index, -1.0)
                extra_top, extra = noise.topk(min(self.explore, noise.size(1)), dim=1)
                index = torch.cat([index, extra], dim=1)
                valid = torch.cat([valid, extra_top >= 0], dim=1)

        if not bool(valid.any()):
            return r0
        neighbour = donor_r[index].to(dtype)
        items = self.item_emb[donor_item[index]].to(dtype)
        if mode == "latent_only":
            items = torch.zeros_like(items)
        elif mode == "shuffle_outcome":
            items = items.clone()
            for b in range(batch):
                slots = torch.nonzero(valid[b]).flatten()
                items[b, slots] = items[b, slots.roll(1)]
        # Recomputed with gradients so the retriever learns through attention too.
        logits = torch.einsum("bk,bnk->bn", q, self.key(neighbour).float()) * self.key_scale
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1) * valid.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        values = self.value(torch.cat(
            [F.normalize(neighbour.float(), dim=-1), F.normalize(items.float(), dim=-1)], dim=-1
        ).to(dtype))
        has_donor = valid.any(-1, keepdim=True)
        r1 = torch.where(has_donor, self._ground(r0, values, alpha), r0.float())

        rows = has_donor.squeeze(-1)
        if self.training and self.target_item is not None and bool(rows.any()):
            target = self._expand(self.target_item, batch, r0.device)
            item_loss = -self.item_log_probs(r1)[torch.arange(batch), target][rows].mean()
            # Behaviour-supervised retrieval: usefulness of each donor alone.
            with torch.no_grad():
                base = self.item_log_probs(r0)[torch.arange(batch), target]
                single = r0.float()[:, None] + self.evidence_scale * r0.float().norm(dim=-1)[:, None, None] \
                    * F.normalize(values.float(), dim=-1)
                gained = self.item_log_probs(single.flatten(0, 1)).view(batch, -1, self.item_emb.size(0))
                utility = gained.gather(2, target[:, None, None].expand(-1, values.size(1), 1)).squeeze(2) \
                    - base[:, None]
                teacher = (utility / self.tau_util).masked_fill(~valid, float("-inf")).softmax(-1)
                teacher = teacher.nan_to_num(0.0)
            log_student = logits.log_softmax(-1)
            kl = (teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1)[rows].mean()
            self.aux_loss = {"item": item_loss, "retrieval": kl}
            with torch.no_grad():
                hit = lambda lp: (lp.topk(min(10, lp.size(-1)), dim=-1).indices == target[:, None]).any(-1).float()[rows].mean()
                self.stats.update({
                    "item_nll_r0": -base[rows].mean(), "item_nll_r1": item_loss.detach(),
                    "item_hit10_r0": hit(self.item_log_probs(r0)), "item_hit10_r1": hit(self.item_log_probs(r1)),
                    "utility_std": utility.masked_fill(~valid, 0).std(),
                    "utility_top_gain": utility.masked_fill(~valid, float("-inf")).max(-1).values[rows].mean(),
                })
        with torch.no_grad():
            self.stats.update({
                "evidence_scale": self.evidence_scale.detach(),
                "attention_entropy": -(alpha * alpha.clamp_min(1e-12).log()).sum(-1).mean(),
                "valid_fraction": valid.float().mean(),
                "donor_rate": has_donor.float().mean(),
                "donor_count": torch.tensor(float(donor_r.size(0))),
            })
        return r1.to(r0.dtype)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        r0 = self.base_attention(hidden_states=hidden_states, attention_mask=attention_mask,
                                 thought_id_idx=thought_id_idx)
        self.last_raw = r0
        r1 = self.refine(r0)
        override = getattr(self, "slot_override", None)   # eval-only slot intervention (SLOT_MODE)
        if override is not None:
            r1 = override.repeat_interleave(r1.size(0) // override.size(0), 0).to(r1)
        return r1


class GroundedReasoningLatentModel(LatentModel):
    """Training wrapper: sets query metadata, adds grounding losses, fills the FIFO."""

    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_grounded_reasoning
        self.item_weight = float(spec.get("item_weight", 0.2))
        self.retrieval_weight = float(spec.get("retrieval_weight", 0.1))
        self.grad_accum_steps = 1
        self.last_losses = {}

    @staticmethod
    def _ce(logits, labels):
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                               ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None, query_time=None,
                query_user=None, target_item=None, outcome_time=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        layer = self.attention
        layer.query_time, layer.query_user = query_time, query_user
        layer.target_item = target_item if self.training else None
        try:
            if labels is None or not self.training:
                return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                       labels=labels, **kwargs)
            out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            ce = self._ce(out.logits, labels)
            loss = ce
            aux = layer.aux_loss or {}
            if aux:
                loss = loss + self.item_weight * aux["item"] + self.retrieval_weight * aux["retrieval"]
            layer.push(layer.last_raw, target_item, query_user, outcome_time)
            self.last_losses = {"ce_loss": ce.detach(), **{f"{k}_loss": v.detach() for k, v in aux.items()},
                                **layer.stats}
            return CausalLMOutputWithPast(loss=loss / self.grad_accum_steps, logits=out.logits)
        finally:
            layer.query_time = layer.query_user = layer.target_item = None

    @torch.no_grad()
    def build_inference_bank(self, loader, device):
        """Fill the persistent bank with every train row's own-history latent r0."""
        layer = self.attention
        previous = os.environ.get("GROUNDING_MODE")
        os.environ["GROUNDING_MODE"] = "off"
        self.eval()
        offset = 0
        try:
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                self.generate_embs(ids, mask)
                n = ids.size(0)
                layer.bank_r[offset:offset + n] = layer.last_raw.to(layer.bank_r.dtype)
                layer.bank_item[offset:offset + n] = batch["target_item"].to(device)
                layer.bank_user[offset:offset + n] = batch["query_user"].to(device)
                layer.bank_time[offset:offset + n] = batch["outcome_time"].to(device)
                offset += n
        finally:
            if previous is None:
                os.environ.pop("GROUNDING_MODE", None)
            else:
                os.environ["GROUNDING_MODE"] = previous
        if offset != layer.bank_r.size(0):
            raise RuntimeError(f"bank filled {offset} rows, expected {layer.bank_r.size(0)}")
        layer.bank_ready.fill_(True)
        return offset
