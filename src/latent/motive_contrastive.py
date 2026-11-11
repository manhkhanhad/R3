"""MWC-M: margin + WIDER, then "same-motive" multi-positive InfoNCE on the R3 latent.

The user's definition of collaborative reasoning in the latent space:
  * cross-user: two users who bought the same next item should have similar latents;
  * within-user: two rows of one user should have similar latents only when their next
    items are similar (Qwen3 title cosine >= ``within_threshold``) or identical.
Same-user rows with dissimilar next items are *neutral*: neither positive nor negative.
Negatives are other users' rows with a different next item.

Training (3 epochs, as 356322):
  epoch 1  CE + margin (other in-batch row's r, detached) + WIDER on r.  Every row's r is
           written into a bank indexed by train-row id (all 53.9k rows), so every row has
           its positives available when phase 2 starts.
  epochs 2+  + motive_weight * InfoNCE on a projection of r (LayerNorm + Linear, 256-d,
           temperature 0.1), positives sampled from the bank (<= 16 cross-user, <= 16
           within-user), 256 bank negatives.
All terms are summed and divided once by grad_accum_steps.  The bank is a persistent buffer
so a resumed (preempted) run keeps it.  Inference is the plain LatentModel path.
"""

import ast
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from grounded_reasoning import user_key
from layers import LatentModel
from margin_loss_attention import MarginLatentModel
from reasoning_dataset import LatentRDataset


def build_positive_index(train_file, item_emb, threshold=0.8, cap=64, seed=0):
    """CSR positive lists per train row: (cross_ptr, cross_idx, within_ptr, within_idx).

    cross: other users' rows with the same next item; within: the same user's other rows whose
    next item is identical or has title cosine >= threshold.  Lists are capped at ``cap``
    (random subset) to bound memory; sampling at train time draws from the filled ones.
    """
    train = pd.read_csv(train_file)
    items = train["item_id"].astype(np.int64).values
    users = train["user_id"].astype(str).values
    emb = F.normalize(item_emb.float(), dim=-1)
    rng = np.random.default_rng(seed)
    by_item, by_user = defaultdict(list), defaultdict(list)
    for i, (it, u) in enumerate(zip(items, users)):
        by_item[it].append(i)
        by_user[u].append(i)
    cross, within = [], []
    for i in range(len(train)):
        c = [j for j in by_item[items[i]] if users[j] != users[i]]
        rows = [j for j in by_user[users[i]] if j != i]
        if rows:
            sim = (emb[items[rows]] @ emb[items[i]]).numpy()
            w = [j for j, s in zip(rows, sim) if items[j] == items[i] or s >= threshold]
        else:
            w = []
        cross.append(c if len(c) <= cap else list(rng.choice(c, cap, replace=False)))
        within.append(w if len(w) <= cap else list(rng.choice(w, cap, replace=False)))

    def csr(lists):
        ptr = np.zeros(len(lists) + 1, dtype=np.int64)
        ptr[1:] = np.cumsum([len(x) for x in lists])
        idx = np.fromiter((j for x in lists for j in x), dtype=np.int64, count=int(ptr[-1]))
        return torch.from_numpy(ptr), torch.from_numpy(idx)

    stats = {"rows": len(train), "cross_rate": float(np.mean([len(x) > 0 for x in cross])),
             "within_rate": float(np.mean([len(x) > 0 for x in within])),
             "either_rate": float(np.mean([len(a) > 0 or len(b) > 0 for a, b in zip(cross, within)]))}
    return (*csr(cross), *csr(within)), items, np.array([user_key(u) for u in users], dtype=np.int64), stats


class MotiveDataset(LatentRDataset):
    """Plain R3 rows plus the train-row id (CSV row position) for the bank."""

    def pre(self, idx):
        example = super().pre(idx)
        if not self.test:
            example["row_id"] = int(self.data.index[idx])
        return example


class MotiveMarginLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_motive
        hidden, n = config.hidden_size, int(spec["bank_size"])
        self.projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(spec.get("projection_dim", 256)), bias=False))
        self.register_buffer("bank_r", torch.zeros(n, hidden, dtype=torch.float16))
        self.register_buffer("bank_filled", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("bank_item", torch.zeros(n, dtype=torch.int64))
        self.register_buffer("bank_user", torch.zeros(n, dtype=torch.int64))
        for name in ("cross_ptr", "cross_idx", "within_ptr", "within_idx"):
            self.register_buffer(name, torch.zeros(1, dtype=torch.int64), persistent=False)
        self.temperature = float(spec.get("temperature", 0.1))
        self.motive_weight = float(spec.get("weight", 0.1))
        self.max_positives = int(spec.get("max_positives", 16))
        self.num_negatives = int(spec.get("num_negatives", 256))
        self.margin, self.margin_weight = 0.2, 1.0
        self.contrastive_enabled = False
        self.grad_accum_steps = 1
        self.last_losses = {}

    _per_example_nll = staticmethod(MarginLatentModel._per_example_nll)
    _thought_positions = MarginLatentModel._thought_positions

    def set_positive_index(self, index, items, users):
        for name, value in zip(("cross_ptr", "cross_idx", "within_ptr", "within_idx"), index):
            setattr(self, name, value.to(self.bank_r.device))
        self.bank_item.copy_(torch.as_tensor(items))
        self.bank_user.copy_(torch.as_tensor(users))

    def _decode(self, embeds, attention_mask):
        return Qwen2ForCausalLM.forward(self, inputs_embeds=embeds, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True)

    def _sample_filled(self, ptr, idx, row):
        cand = idx[ptr[row]:ptr[row + 1]]
        cand = cand[self.bank_filled[cand]]
        if cand.numel() > self.max_positives:
            cand = cand[torch.randperm(cand.numel(), device=cand.device)[: self.max_positives]]
        return cand

    def _motive_loss(self, r, row_id):
        device = r.device
        proj = lambda x: F.normalize(self.projection(x.float()), dim=-1)
        anchors = proj(r)
        filled = torch.nonzero(self.bank_filled).flatten()
        losses, stats = [], defaultdict(list)
        for b in range(r.size(0)):
            i = int(row_id[b])
            cross = self._sample_filled(self.cross_ptr, self.cross_idx, i)
            within = self._sample_filled(self.within_ptr, self.within_idx, i)
            pos = torch.cat([cross, within])
            stats["has_cross"].append(float(cross.numel() > 0))
            stats["has_within"].append(float(within.numel() > 0))
            if pos.numel() == 0 or filled.numel() == 0:
                continue
            neg = filled[torch.randint(filled.numel(), (self.num_negatives * 2,), device=device)]
            neg = neg[(self.bank_user[neg] != self.bank_user[i]) & (self.bank_item[neg] != self.bank_item[i])]
            neg = neg[: self.num_negatives]
            if neg.numel() == 0:
                continue
            p, q = proj(self.bank_r[pos]), proj(self.bank_r[neg])
            pos_s, neg_s = anchors[b] @ p.T / self.temperature, anchors[b] @ q.T / self.temperature
            losses.append(torch.logsumexp(torch.cat([pos_s, neg_s]), 0) - torch.logsumexp(pos_s, 0))
            with torch.no_grad():
                if cross.numel():
                    stats["cos_cross"].append(float((anchors[b] @ p[: cross.numel()].T).mean()))
                if within.numel():
                    stats["cos_within"].append(float((anchors[b] @ p[cross.numel():].T).mean()))
                stats["cos_neg"].append(float((anchors[b] @ q.T).mean()))
        loss = torch.stack(losses).mean() if losses else anchors.sum() * 0.0
        return loss, {k: torch.tensor(float(np.mean(v))) for k, v in stats.items() if v}

    def forward(self, input_ids=None, attention_mask=None, labels=None, row_id=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        training_prefill = (self.training and labels is not None and input_ids is not None
                            and attention_mask is not None and kwargs.get("inputs_embeds") is None)
        if not training_prefill:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        embeds = self.generate_embs(input_ids, attention_mask)      # WIDER state stores r here
        thought_idx = self._thought_positions(input_ids)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        r = embeds[rows, thought_idx]
        pos_out = self._decode(embeds, attention_mask)
        pos_nll = self._per_example_nll(pos_out.logits, labels)
        ce = pos_nll.mean()
        margin_loss = ce.new_zeros(())
        if input_ids.size(0) > 1 and self.margin_weight > 0:
            perm = torch.roll(rows, 1)
            neg = embeds.clone()
            neg[rows, thought_idx] = embeds[perm, thought_idx[perm]].detach()
            neg_nll = self._per_example_nll(self._decode(neg, attention_mask).logits, labels)
            margin_loss = F.relu(self.margin + pos_nll - neg_nll).mean()
        loss = ce + self.margin_weight * margin_loss
        wider = None
        if getattr(self.config, "latent_wider", None):
            from wider_reg import wider_state
            state = wider_state(self)
            wider = state.penalty() if state is not None else None
            if wider is not None:
                loss = loss + state.weight * wider
        motive, mstats = ce.new_zeros(()), {}
        if self.contrastive_enabled and row_id is not None:
            motive, mstats = self._motive_loss(r, row_id)
            loss = loss + self.motive_weight * motive
        if row_id is not None:
            with torch.no_grad():
                self.bank_r[row_id] = r.detach().to(self.bank_r.dtype)
                self.bank_filled[row_id] = True
        self.last_losses = {
            "ce_loss": ce.detach(), "margin_loss": margin_loss.detach(), "motive_loss": motive.detach(),
            "contrastive_enabled": torch.tensor(float(self.contrastive_enabled)),
            "bank_filled": self.bank_filled.float().mean(),
            **({"wider_loss": wider.detach()} if wider is not None else {}), **mstats,
        }
        return CausalLMOutputWithPast(loss=loss / self.grad_accum_steps, logits=pos_out.logits)
