"""G3: margin + WIDER(r0) + CF-prior donor evidence trained with an evidence-swap margin.

r0  = R3 pooling over the user's own history (unchanged prompt).
r1  = r0 + s * ||r0|| * dir(e - mu),   e = sum_j a_j V(x_j),   x_j = [e(y_j), log1p(shared_j), log1p(gap_j)/5]

Donors j are *fixed by a collaborative-filtering prior*, not by a learned retriever: the
`topn` training rows of other users whose history shares >= 1 item with the query history
and whose observed next item y_j happened strictly before the query's latest history event,
ranked by number of shared items, then recency (CFDonorIndex).  The attention a_j (query
from r0, key from x_j) only re-weights this set.  Evidence is the donors' *outcomes*:
the random-donor diagnostic (bank_random_donors.py) showed other users' r carries no
collaborative signal, while the target is among the CF donors' outcomes for ~10% of
non-copy test rows (1% for random causal donors).  mu is an EMA of e (centring against
the constant-direction collapse seen in MWC-G).

Losses (summed, then divided once by grad_accum_steps, see feedback-hf-grad-accum-aux-loss):
  CE(y | x, r1)
  + margin_weight   * [m + NLL(y|x,r1) - NLL(y|x, r0_other (+) dir_own)]_+        (history content required; neg detached)
  + evidence_weight * [m_e + NLL(y|x,r1) - NLL(y|x, r0 (+) dir_other_donors)]_+   (evidence content required;
                                                                                    grads through the evidence path)
  + item_weight * item-grounding CE on r1 + retrieval_weight * KL(single-donor utility teacher || a)
  + WIDER on r0 (cross-user spectral penalty, wider_reg.py)
Eval donor controls (latent_attention_eval.py, GROUNDING_MODE): knn = CF donors, off, random
(random causal other-user train rows), swap (CF donors of another test row).
"""

import ast
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from grounded_reasoning import GroundedReasoningDataset, history_cutoff, user_key
from layers import LatentModel
from margin_loss_attention import MarginLatentModel

N_FEAT = 2
CF_MODES = {"knn", "off", "random", "swap"}


def _as_list(value):
    return list(ast.literal_eval(value)) if isinstance(value, str) else list(value)


class CFDonorIndex:
    """Time-causal, other-user donors from the training split, selected by history overlap."""

    def __init__(self, train_file, topn=32):
        train = pd.read_csv(train_file)
        self.topn = int(topn)
        self.user = np.array([user_key(u) for u in train["user_id"]], dtype=np.int64)
        self.time = train["timestamp"].astype(np.int64).values
        self.item = train["item_id"].astype(np.int64).values
        inverted = defaultdict(list)
        for j, hist in enumerate(train["history_item_id"]):
            for item in set(_as_list(hist)):
                inverted[int(item)].append(j)
        self.inverted = {k: np.asarray(v, dtype=np.int64) for k, v in inverted.items()}

    def __len__(self):
        return len(self.item)

    def _pack(self, rows, shared, query_time):
        idx = np.full(self.topn, -1, dtype=np.int64)
        feat = np.zeros((self.topn, N_FEAT), dtype=np.float32)
        n = len(rows)
        if n:
            gap_days = np.maximum(query_time - self.time[rows], 0) / 86400.0
            idx[:n] = rows
            feat[:n, 0] = np.log1p(shared)
            feat[:n, 1] = np.log1p(gap_days) / 5.0
        return idx, feat

    def cf(self, history, user, query_time):
        counts = defaultdict(int)
        for item in set(int(i) for i in history):
            for j in self.inverted.get(item, ()):
                counts[j] += 1
        if not counts:
            return self._pack(np.zeros(0, np.int64), np.zeros(0), query_time)
        rows = np.fromiter(counts.keys(), np.int64, len(counts))
        shared = np.fromiter(counts.values(), np.float64, len(counts))
        keep = (self.time[rows] < query_time) & (self.user[rows] != user)
        rows, shared = rows[keep], shared[keep]
        order = np.lexsort((-self.time[rows], -shared))[: self.topn]
        return self._pack(rows[order], shared[order], query_time)

    def random(self, user, query_time, rng):
        cand = np.nonzero((self.time < query_time) & (self.user != user))[0]
        rows = rng.choice(cand, size=min(self.topn, len(cand)), replace=False) if len(cand) else cand
        return self._pack(rows, np.zeros(len(rows)), query_time)

    def for_row(self, row):
        return self.cf(_as_list(row["history_item_id"]), user_key(row["user_id"]), history_cutoff(row))


class CFGroundedDataset(GroundedReasoningDataset):
    """Grounded-reasoning rows plus their fixed CF donor set (donor_idx, donor_feat flattened)."""

    donor_index = None

    def pre(self, idx):
        example = super().pre(idx)
        if not self.test:
            donor_idx, donor_feat = self.donor_index.for_row(self.data.iloc[idx])
            example["donor_idx"] = donor_idx.tolist()
            example["donor_feat"] = donor_feat.reshape(-1).tolist()
        return example


class CFGroundedLayer(nn.Module):
    def __init__(self, base_attention, hidden_size, spec):
        super().__init__()
        self.base_attention = base_attention
        self.tau_item = float(spec.get("tau_item", 0.07))
        self.tau_util = float(spec.get("tau_util", 0.1))
        self.centre = bool(spec.get("centre", True))
        self.mean_momentum = float(spec.get("mean_momentum", 0.99))
        key_dim = int(spec.get("key_dim", 256))
        value_hidden = int(spec.get("value_hidden", 1024))
        n_items, item_dim, n_bank = int(spec["n_items"]), int(spec["item_dim"]), int(spec["bank_size"])
        donor_dim = item_dim + N_FEAT

        self.query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, key_dim))
        self.key = nn.Sequential(nn.LayerNorm(donor_dim), nn.Linear(donor_dim, key_dim))
        self.value = nn.Sequential(nn.LayerNorm(donor_dim), nn.Linear(donor_dim, value_hidden), nn.GELU(),
                                   nn.Linear(value_hidden, hidden_size))
        self.item_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, item_dim))
        self.evidence_scale = nn.Parameter(torch.tensor(float(spec.get("init_scale", 0.5))))
        self.key_scale = 1.0 / math.sqrt(key_dim)

        self.register_buffer("item_emb", torch.zeros(n_items, item_dim, dtype=torch.float16))
        self.register_buffer("bank_item", torch.zeros(n_bank, dtype=torch.int64))
        self.register_buffer("evidence_mean", torch.zeros(hidden_size, dtype=torch.float32))

        # Per-call inputs set by the model (training) or the eval script.
        self.donor_idx = None        # (B, N) int64, -1 = no donor
        self.donor_feat = None       # (B, N, N_FEAT) or (B, N * N_FEAT)
        self.target_item = None
        self.slot_override = None
        self.last = {}
        self.last_raw = None
        self.aux_loss = None
        self.stats = {}

    @torch.no_grad()
    def reset_new_parameters(self, init_scale=0.5):
        for module in (self.query, self.key, self.value, self.item_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.evidence_scale.fill_(init_scale)
        self.evidence_mean.zero_()

    @staticmethod
    def _expand(value, batch_size):
        if value.size(0) == batch_size:
            return value
        if batch_size % value.size(0):
            raise ValueError("donor metadata must align with the prompts or expanded beams")
        return value.repeat_interleave(batch_size // value.size(0), 0)

    def _donor_inputs(self, idx, feat, batch_size):
        idx = self._expand(idx.long(), batch_size)
        feat = self._expand(feat.float().reshape(feat.size(0), -1, N_FEAT), batch_size)
        valid = idx >= 0
        items = self.item_emb[self.bank_item[idx.clamp_min(0)]].float()
        x = torch.cat([F.normalize(items, dim=-1), feat], dim=-1)
        return x, valid

    # ---------------------------------------------------------------- evidence
    def item_log_probs(self, r):
        projected = F.normalize(self.item_head(r.float()), dim=-1)
        logits = projected @ F.normalize(self.item_emb.float(), dim=-1).T / self.tau_item
        return logits.log_softmax(-1)

    def evidence(self, r0, idx, feat):
        """Attention-weighted donor evidence for each row of r0 given its donor set."""
        x, valid = self._donor_inputs(idx, feat, r0.size(0))
        q = self.query(r0.float())
        logits = torch.einsum("bk,bnk->bn", q, self.key(x)) * self.key_scale
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1) * valid.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        values = self.value(x)
        e = torch.einsum("bn,bnh->bh", alpha, values)
        return {"e": e, "alpha": alpha, "logits": logits, "values": values, "valid": valid,
                "has": valid.any(-1)}

    def direction(self, e):
        if self.centre:
            e = e - self.evidence_mean
        return F.normalize(e, dim=-1)

    def compose(self, r0, direction, has):
        r1 = r0.float() + self.evidence_scale * r0.float().norm(dim=-1, keepdim=True) * direction
        return torch.where(has[:, None], r1, r0.float())

    @torch.no_grad()
    def update_mean(self, e, has):
        if bool(has.any()):
            batch_mean = e[has].float().mean(0)
            if float(self.evidence_mean.norm()) == 0.0:      # seed with the first batch
                self.evidence_mean.copy_(batch_mean)
                return
            self.evidence_mean.mul_(self.mean_momentum).add_((1 - self.mean_momentum) * batch_mean)

    # ----------------------------------------------------------------- forward
    def forward(self, hidden_states, attention_mask, thought_id_idx):
        r0 = self.base_attention(hidden_states=hidden_states, attention_mask=attention_mask,
                                 thought_id_idx=thought_id_idx)
        self.last_raw = r0
        self.aux_loss = None
        self.last = {}
        mode = os.environ.get("GROUNDING_MODE", "knn")
        if mode not in CF_MODES:
            raise ValueError(f"unknown GROUNDING_MODE {mode}")
        r1 = r0
        if mode != "off" and self.donor_idx is not None:
            ev = self.evidence(r0, self.donor_idx, self.donor_feat)
            direction = self.direction(ev["e"])
            r1 = self.compose(r0, direction, ev["has"])
            self.last = {**ev, "direction": direction}
            if self.training and self.target_item is not None and bool(ev["has"].any()):
                self._aux(r0, r1, ev)
            with torch.no_grad():
                self._stats(ev, direction)
            r1 = r1.to(r0.dtype)
        if self.slot_override is not None:   # eval-only slot intervention (SLOT_MODE)
            r1 = self._expand(self.slot_override, r1.size(0)).to(r1)
        return r1

    def _aux(self, r0, r1, ev):
        batch = r0.size(0)
        rows = ev["has"]
        target = self._expand(torch.as_tensor(self.target_item, device=r0.device).long().reshape(-1), batch)
        ar = torch.arange(batch, device=r0.device)
        item_loss = -self.item_log_probs(r1)[ar, target][rows].mean()
        with torch.no_grad():
            base = self.item_log_probs(r0)[ar, target]
            single_dir = self.direction(ev["values"])            # each donor alone, centred
            single = r0.float()[:, None] + self.evidence_scale * r0.float().norm(dim=-1)[:, None, None] * single_dir
            gained = self.item_log_probs(single.flatten(0, 1)).view(batch, -1, self.item_emb.size(0))
            utility = gained.gather(2, target[:, None, None].expand(-1, single.size(1), 1)).squeeze(2) - base[:, None]
            teacher = (utility / self.tau_util).masked_fill(~ev["valid"], float("-inf")).softmax(-1).nan_to_num(0.0)
        log_student = ev["logits"].log_softmax(-1)
        kl = (teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1)[rows].mean()
        self.aux_loss = {"item": item_loss, "retrieval": kl}
        with torch.no_grad():
            hit = lambda lp: (lp.topk(10, dim=-1).indices == target[:, None]).any(-1).float()[rows].mean()
            self.stats.update({
                "item_nll_r0": -base[rows].mean(), "item_nll_r1": item_loss.detach(),
                "item_hit10_r0": hit(self.item_log_probs(r0)), "item_hit10_r1": hit(self.item_log_probs(r1)),
                "utility_top_gain": utility.masked_fill(~ev["valid"], float("-inf")).max(-1).values[rows].mean(),
            })

    @staticmethod
    def _offdiag_cos(v):
        if v.size(0) < 2:
            return torch.tensor(float("nan"), device=v.device)
        z = F.normalize(v.float(), dim=-1)
        c = z @ z.T
        n = v.size(0)
        return (c.sum() - c.diagonal().sum()) / (n * (n - 1))

    def _stats(self, ev, direction):
        has = ev["has"]
        alpha = ev["alpha"]
        self.stats.update({
            "evidence_scale": self.evidence_scale.detach(),
            "attention_entropy": -(alpha * alpha.clamp_min(1e-12).log()).sum(-1)[has].mean() if bool(has.any()) else torch.tensor(0.0),
            "donor_rate": has.float().mean(),
            "donor_count": ev["valid"].float().sum(-1).mean(),
            "evidence_cos_raw": self._offdiag_cos(ev["e"][has]),
            "evidence_cos_centred": self._offdiag_cos(direction[has]),
            "evidence_mean_norm": self.evidence_mean.norm(),
        })


class CFGroundedMarginLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_cf_grounded
        self.item_weight = float(spec.get("item_weight", 0.2))
        self.retrieval_weight = float(spec.get("retrieval_weight", 0.1))
        self.evidence_margin = float(spec.get("evidence_margin", 0.2))
        self.evidence_weight = float(spec.get("evidence_weight", 1.0))
        self.margin = 0.2
        self.margin_weight = 1.0
        self.grad_accum_steps = 1
        self.last_losses = {}

    _per_example_nll = staticmethod(MarginLatentModel._per_example_nll)
    _thought_positions = MarginLatentModel._thought_positions

    def keep_new_modules_fp32(self):
        layer = self.attention
        for module in (layer.query, layer.key, layer.value, layer.item_head):
            module.float()
        layer.evidence_scale.data = layer.evidence_scale.data.float()

    def _decode(self, embeds, attention_mask):
        return Qwen2ForCausalLM.forward(self, inputs_embeds=embeds, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True)

    def _with_slot(self, embeds, rows, thought_idx, slot):
        out = embeds.clone()
        out[rows, thought_idx] = slot.to(out.dtype)
        return out

    def forward(self, input_ids=None, attention_mask=None, labels=None, donor_idx=None, donor_feat=None,
                target_item=None, **kwargs):
        for key in ("query_time", "query_user", "outcome_time", "num_items_in_batch"):
            kwargs.pop(key, None)
        layer = self.attention
        if donor_idx is not None:
            layer.donor_idx, layer.donor_feat = donor_idx, donor_feat
        layer.target_item = target_item if self.training else None
        training_prefill = (self.training and labels is not None and input_ids is not None
                            and attention_mask is not None and kwargs.get("inputs_embeds") is None)
        try:
            if not training_prefill:
                return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            return self._train_step(input_ids, attention_mask, labels, target_item)
        finally:
            if donor_idx is not None:
                layer.donor_idx = layer.donor_feat = None
            layer.target_item = None

    def _train_step(self, input_ids, attention_mask, labels, target_item):
        layer = self.attention
        embeds = self.generate_embs(input_ids, attention_mask)       # slot = r1
        r0 = layer.last_raw
        wider = None
        if getattr(self.config, "latent_wider", None):
            from wider_reg import wider_state
            state = wider_state(self)
            if state is not None:
                state.latent = r0                                     # WIDER on r0, not r1
                wider = state.penalty()
        thought_idx = self._thought_positions(input_ids)
        batch = input_ids.size(0)
        rows = torch.arange(batch, device=input_ids.device)

        pos_out = self._decode(embeds, attention_mask)
        pos_nll = self._per_example_nll(pos_out.logits, labels)
        ce = pos_nll.mean()
        zero = ce.new_zeros(())
        margin_loss = evidence_loss = zero
        ev = layer.last
        if batch > 1 and ev:
            has, direction = ev["has"], ev["direction"]
            # (1) history swap: another row's r0 with this row's evidence (detached, as Attempt 2).
            if self.margin_weight > 0:
                perm = torch.roll(rows, 1)
                slot = layer.compose(r0[perm], direction, has).detach()
                neg_nll = self._per_example_nll(
                    self._decode(self._with_slot(embeds, rows, thought_idx, slot), attention_mask).logits, labels)
                margin_loss = F.relu(self.margin + pos_nll - neg_nll).mean()
            # (2) evidence swap: this row's r0 with another row's donor set; grads through the evidence path.
            if self.evidence_weight > 0:
                perm2 = torch.roll(rows, min(2, batch - 1))
                other = layer.evidence(r0.detach(), layer.donor_idx[perm2], layer.donor_feat[perm2])
                both = has & other["has"]
                if bool(both.any()):
                    slot = layer.compose(r0.detach(), layer.direction(other["e"]), both)
                    neg_nll = self._per_example_nll(
                        self._decode(self._with_slot(embeds, rows, thought_idx, slot), attention_mask).logits, labels)
                    evidence_loss = F.relu(self.evidence_margin + pos_nll - neg_nll)[both].mean()
                    self.last_losses["evidence_gap"] = (neg_nll - pos_nll)[both].mean().detach()

        loss = ce + self.margin_weight * margin_loss + self.evidence_weight * evidence_loss
        aux = layer.aux_loss or {}
        if aux:
            loss = loss + self.item_weight * aux["item"] + self.retrieval_weight * aux["retrieval"]
        if wider is not None:
            loss = loss + self.config.latent_wider["weight"] * wider
        if ev:
            layer.update_mean(ev["e"].detach(), ev["has"])
        self.last_losses.update({
            "ce_loss": ce.detach(), "margin_loss": margin_loss.detach(), "evidence_loss": evidence_loss.detach(),
            **({"wider_loss": wider.detach()} if wider is not None else {}),
            **{f"{k}_loss": v.detach() for k, v in aux.items()}, **layer.stats,
        })
        return CausalLMOutputWithPast(loss=loss / self.grad_accum_steps, logits=pos_out.logits)

    @torch.no_grad()
    def build_slot_bank(self, loader, device, limit):
        """Slot vectors r1 (CF donors) and users of the first `limit` rows, for mean / shuffle_user."""
        layer = self.attention
        self.eval()
        vectors, users = [], []
        for batch in loader:
            layer.donor_idx = batch["donor_idx"].to(device)
            layer.donor_feat = batch["donor_feat"].to(device)
            ids = batch["input_ids"].to(device)
            embeds = self.generate_embs(ids, batch["attention_mask"].to(device))
            vectors.append(embeds[torch.arange(ids.size(0), device=device), self._thought_positions(ids)].float().cpu())
            users.append(batch["query_user"].cpu())
            if sum(len(v) for v in vectors) >= limit:
                break
        layer.donor_idx = layer.donor_feat = None
        return torch.cat(vectors)[:limit], torch.cat(users)[:limit]


def eval_donors(model, frame, mode, seed=42):
    """Donor tensors (idx (n, N), feat (n, N*N_FEAT)) for every eval row under a GROUNDING_MODE control."""
    spec = model.config.latent_cf_grounded
    index = CFDonorIndex(spec["train_file"], spec.get("topn", 32))
    rows = [frame.iloc[i] for i in range(len(frame))]
    cf = [index.for_row(r) for r in rows]
    users = [user_key(r["user_id"]) for r in rows]
    if mode == "random":
        rng = np.random.default_rng(seed)
        cf = [index.random(u, history_cutoff(r), rng) for u, r in zip(users, rows)]
    elif mode == "swap":
        # Another eval row's CF donor set (next row of a different user).
        n, swapped = len(rows), []
        for i in range(n):
            k = 1
            while users[(i + k) % n] == users[i] and k < n:
                k += 1
            swapped.append(cf[(i + k) % n])
        cf = swapped
    idx = torch.tensor(np.stack([c[0] for c in cf]))
    feat = torch.tensor(np.stack([c[1].reshape(-1) for c in cf]))
    print(f"CF donors ({mode}): rows {len(rows)}, with donor {(idx >= 0).any(1).float().mean():.3f}, "
          f"mean donors {(idx >= 0).sum(1).float().mean():.1f}", flush=True)
    return idx, feat


def restore_fp32(model, checkpoint):
    """Eval loads in bf16; put the G3 modules back in fp32 with their saved fp32 weights and buffers."""
    from safetensors import safe_open

    layer = model.attention
    for module in (layer.query, layer.key, layer.value, layer.item_head):
        module.float()
    layer.evidence_scale.data = layer.evidence_scale.data.float()
    layer.evidence_mean.data = layer.evidence_mean.data.float()
    names = ("attention.query.", "attention.key.", "attention.value.", "attention.item_head.",
             "attention.evidence_scale", "attention.evidence_mean")
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    loaded = 0
    for name in [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")]:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith(names) and key in tensors:
                    with torch.no_grad():
                        tensors[key].copy_(f.get_tensor(key).to(tensors[key].dtype))
                    loaded += 1
    print(f"G3: restored {loaded} fp32 tensors, evidence_scale={float(layer.evidence_scale):.4f}, "
          f"|evidence_mean|={float(layer.evidence_mean.norm()):.4f}", flush=True)
    return loaded
