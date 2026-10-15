"""Disentangled-interest latent reasoning for R3.

One history supports several interests that lead to different items, and only
one of them explains the item the user actually chose.  A single deterministic
``r = f(x)`` can only average those interests.  Here the latent is a choice
among K reasoning paths:

    r0        = R3 pooling over the history
    r_k       = r0 + g * ||r0|| * normalize(U_k trunk(r0) + b_k)  k = 1..K
    p(k | x)  = softmax(prior(r0))                                (inference)
    q(k | x,y) ∝ p_item(y | r_k)  [optionally × p(k | x)]           (training only)

``p_item`` is a cheap catalogue scorer.  Training is winner-takes-all: the
decoder sees only the path the posterior credits with the observed item, the
prior learns to predict that path from history alone, and a mixture item loss
lets paths specialise.  At inference the eval script runs one beam search per
intent and merges by ``log p(k|x) + sequence score``.

Interest-collapse fixes from the multi-interest literature (all training-only):

* MIMA (arXiv 2609.12842): single-positive training lets one interest absorb every
  target. ``assignment="hungarian"`` treats the target plus the user's next items as
  positives and assigns each to a distinct path (Hungarian matching on -log p_item).
* Re4 (WWW 2022, arXiv 2208.08011) Re-contrast: each path attends over the history
  items; it is pulled to the items it attends to above 1/n and pushed from the other
  history items, the other paths and random items (InfoNCE).
* REMI (RecSys 2023, arXiv 2302.14532) routing regularisation: penalises the
  per-interest variance of the routing weights so a path cannot collapse onto a
  single history item.  (REMI's hard-negative mining targets sampled softmax; the
  item scorer here already uses a full softmax over the catalogue.)

Phase 2 (DivReason, SIGIR 2026, "Reasoning Path Aggregation"): a router scores each
path, redundancy among path offsets is suppressed, and a sharp softmax (pruned at
inference) weights the paths into one latent.  The router is trained through the
decoder's recommendation loss with everything else frozen.  Modes ``aggregate``,
``aggregate_uniform`` (equal weights) and ``prior_blend`` (prior weights, no router).

``INTENT_MODE`` (eval): ``map`` (default) | ``r0`` | ``random`` | ``oracle``;
a forced intent (set by the eval script) overrides it.
"""

import ast
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel
from reasoning_dataset import LatentRDataset

INTENT_MODES = {"map", "r0", "random", "oracle", "aggregate", "aggregate_uniform", "prior_blend"}


MAX_HISTORY = 10


def later_items_by_row(csv_path, n_later):
    """For each CSV row, the user's next distinct items after this row (training labels only)."""
    df = pd.read_csv(csv_path, usecols=["user_id", "item_id", "timestamp"])
    df["_row"] = range(len(df))
    df = df.sort_values(["user_id", "timestamp", "_row"])
    later = {}
    for _, group in df.groupby("user_id", sort=False):
        rows, items = group["_row"].tolist(), group["item_id"].tolist()
        for i, row in enumerate(rows):
            chosen = []
            for item in items[i + 1:]:
                if item != items[i] and item not in chosen:
                    chosen.append(int(item))
                    if len(chosen) == n_later:
                        break
            later[row] = chosen
    return later


class IntentDataset(LatentRDataset):
    """Adds the target item, padded history item ids (-1) and, optionally, the user's
    next items (MIMA-style extra positives; -1 padded)."""

    def __init__(self, *args, n_later=0, **kwargs):
        self.n_later = int(n_later)
        self.later = later_items_by_row(kwargs["train_file"], self.n_later) if self.n_later else {}
        super().__init__(*args, **kwargs)

    def pre(self, idx):
        example = super().pre(idx)
        if not self.test:
            row = self.data.iloc[idx]
            example["target_item"] = int(row["item_id"])
            history = row["history_item_id"]
            history = ast.literal_eval(history) if isinstance(history, str) else list(history)
            history = [int(h) for h in history][-MAX_HISTORY:]
            example["history_items"] = history + [-1] * (MAX_HISTORY - len(history))
            if self.n_later:
                later = self.later.get(int(self.data.index[idx]), [])
                example["later_items"] = later + [-1] * (self.n_later - len(later))
        return example


class IntentReasoningLayer(nn.Module):
    def __init__(self, base_attention, hidden_size, spec):
        super().__init__()
        self.base_attention = base_attention
        self.n_intents = int(spec.get("n_intents", 4))
        path_hidden = int(spec.get("path_hidden", 1024))
        n_items, item_dim = int(spec["n_items"]), int(spec["item_dim"])
        self.tau_item = float(spec.get("tau_item", 0.07))
        self.balance_momentum = float(spec.get("balance_momentum", 0.99))
        # Credit the path that explains the observed item; including the prior lets an
        # early, confident prior confirm its own choice regardless of the item.
        self.posterior_uses_prior = bool(spec.get("posterior_uses_prior", False))
        # Literature fixes for interest collapse (see module docstring).
        self.assignment = spec.get("assignment", "argmax")          # MIMA: "hungarian"
        self.route_tau = float(spec.get("route_tau", 0.1))
        self.recontrast_tau = float(spec.get("recontrast_tau", 0.1))
        self.recontrast_negatives = int(spec.get("recontrast_negatives", 64))
        if self.assignment not in {"argmax", "hungarian"}:
            raise ValueError("assignment must be argmax or hungarian")

        # Shared trunk over the history latent, one output head per intent: independent
        # heads start near-orthogonal, so paths are distinct before training specialises
        # them (a single MLP over [r0; c_k] is dominated by r0 and collapses the paths).
        self.path = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, path_hidden), nn.GELU())
        self.intent_heads = nn.Parameter(torch.zeros(self.n_intents, path_hidden, hidden_size))
        self.intent_bias = nn.Parameter(torch.zeros(self.n_intents, hidden_size))
        self.prior = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 512), nn.GELU(),
            nn.Linear(512, self.n_intents),
        )
        self.item_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, item_dim))
        self.path_scale = nn.Parameter(torch.tensor(float(spec.get("init_scale", 0.5))))
        self.register_buffer("item_emb", torch.zeros(n_items, item_dim, dtype=torch.float16))
        self.register_buffer("usage_ema", torch.full((self.n_intents,), 1.0 / self.n_intents),
                             persistent=False)
        # DivReason-style path aggregation (phase 2): router score per path, redundancy
        # suppression on path offsets, sharp softmax, pruning at inference.
        aggregator = spec.get("aggregator") or {}
        self.aggregator_enabled = bool(aggregator)
        if self.aggregator_enabled:
            self.router = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 256), nn.GELU(),
                                        nn.Linear(256, 1))
            self.agg_log_beta = nn.Parameter(torch.tensor(0.0))
            self.agg_tau = float(aggregator.get("tau", 0.5))
            self.agg_prune = float(aggregator.get("prune", 0.05))
            self.last_aggregation = None

        self.forced_intent = None   # int or per-prompt tensor (eval)
        self.target_item = None     # per-row tensor (training, oracle eval)
        self.history_items = None   # (B, MAX_HISTORY) item ids, -1 padded (training)
        self.later_items = None     # (B, n_later) item ids, -1 padded (training)
        self.explore = False        # sample k* from q instead of argmax
        self.aux_loss = None
        self.stats = {}
        self.last_paths = None
        self.last_selected = None
        self.last_prior_logp = None

    @torch.no_grad()
    def prepare_aggregator(self, beta=0.5):
        """Initialise the phase-2 router (absent from phase-1 checkpoints) in fp32, so its
        updates are not rounded away under bf16 training."""
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.router.float()
        self.agg_log_beta.data = torch.tensor(float(torch.log(torch.tensor(beta))), dtype=torch.float32,
                                              device=self.agg_log_beta.device)

    def aggregate(self, r0, paths, prior_logp, mode="aggregate"):
        """Weights over paths -> one latent. Redundancy uses the paths' offsets from r0
        (all paths share r0, so raw cosine would be ~1 for every pair)."""
        batch, k = paths.shape[:2]
        if mode == "aggregate_uniform":
            weights = paths.new_full((batch, k), 1.0 / k)
        elif mode == "prior_blend":
            weights = prior_logp.exp()
        else:
            scores = self.router(paths.to(self.router[1].weight.dtype)).float().squeeze(-1)
            unit = F.normalize(paths - r0.float()[:, None], dim=-1)
            redundancy = (unit @ unit.transpose(1, 2)).sum(-1) - 1.0
            scores = scores - self.agg_log_beta.float().exp() * redundancy
            weights = (scores / self.agg_tau).softmax(-1)
            if not self.training and self.agg_prune > 0:
                keep = (weights >= self.agg_prune) | (weights == weights.max(-1, keepdim=True).values)
                weights = weights * keep
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            self.last_aggregation = {
                "agg_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean(),
                "agg_max_weight": weights.max(-1).values.mean(),
                "agg_prior_agree": (weights.argmax(-1) == prior_logp.argmax(-1)).float().mean(),
                "agg_beta": self.agg_log_beta.float().exp() if self.aggregator_enabled else weights.new_zeros(()),
            }
        self.last_selected = weights.argmax(-1)
        return torch.einsum("bk,bkh->bh", weights, paths)

    @torch.no_grad()
    def reset_new_parameters(self, init_scale=0.5):
        for module in (self.path, self.prior, self.item_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.normal_(self.intent_heads, std=self.intent_heads.size(1) ** -0.5)
        nn.init.zeros_(self.intent_bias)
        self.path_scale.fill_(init_scale)

    def paths(self, r0):
        trunk = self.path(r0.to(self.path[1].weight.dtype))
        delta = torch.einsum("bp,kph->bkh", trunk, self.intent_heads.to(trunk.dtype)) + self.intent_bias.to(trunk.dtype)
        delta = F.normalize(delta.float(), dim=-1)
        return r0.float()[:, None] + self.path_scale * r0.float().norm(dim=-1)[:, None, None] * delta

    def item_log_probs(self, r):
        projected = F.normalize(self.item_head(r.to(self.item_head[1].weight.dtype)).float(), dim=-1)
        logits = projected @ F.normalize(self.item_emb.float(), dim=-1).T / self.tau_item
        return logits.log_softmax(-1)

    @staticmethod
    def _expand(value, batch, device):
        value = torch.as_tensor(value, device=device).long().reshape(-1)
        if len(value) == 1:
            return value.expand(batch)
        if batch % len(value):
            raise ValueError("intent metadata must align with prompts or expanded beams")
        return value.repeat_interleave(batch // len(value))

    def _item_queries(self, paths):
        return F.normalize(self.item_head(paths.to(self.item_head[1].weight.dtype)).float(), dim=-1)

    def _hungarian(self, lp_all, target, later):
        """MIMA: give each positive (target first, then later items) its own path."""
        batch = lp_all.size(0)
        positives = torch.cat([target[:, None], later], dim=1) if later is not None else target[:, None]
        valid = positives >= 0
        cost = -lp_all.gather(2, positives.clamp_min(0)[:, None, :].expand(-1, self.n_intents, -1))
        selected = torch.empty(batch, dtype=torch.long, device=lp_all.device)
        matched = []
        cost_cpu, valid_cpu = cost.detach().float().cpu(), valid.cpu()
        for b in range(batch):
            columns = torch.nonzero(valid_cpu[b]).flatten()[: self.n_intents]
            rows_k, cols = linear_sum_assignment(cost_cpu[b][:, columns].numpy())
            for k, c in zip(rows_k, cols):
                matched.append(cost[b, int(k), int(columns[c])])
                if int(columns[c]) == 0:
                    selected[b] = int(k)
        return selected, torch.stack(matched).mean(), valid.sum(1).float().mean()

    def _routing(self, queries, history):
        """Each path's attention over the user's history items (in item space)."""
        mask = history >= 0
        items = F.normalize(self.item_emb[history.clamp_min(0)].float(), dim=-1)
        logits = torch.einsum("bkd,bnd->bkn", queries, items) / self.route_tau
        logits = logits.masked_fill(~mask[:, None], float("-inf"))
        return logits.softmax(-1).nan_to_num(0.0), items, mask

    def _routing_regularization(self, routing, mask):
        """REMI: per-interest variance of routing weights over history items."""
        n = mask.sum(1, keepdim=True).float()
        rows = n.squeeze(1) >= 2
        if not bool(rows.any()):
            return routing.sum() * 0.0
        mean = routing.sum(-1, keepdim=True) / n[:, None]
        centered = (routing - mean) * mask[:, None]
        variance = (centered ** 2).sum(-1)               # diag(C) per interest
        return (variance ** 2).sum(-1)[rows].mean()

    def _recontrast(self, queries, routing, items, mask):
        """Re4: each path is pulled to the history items it attends to (> 1/n) and
        pushed from the remaining history items, the other paths, and random items."""
        batch, k = queries.shape[:2]
        n = mask.sum(1).clamp_min(1).float()
        positive = (routing > (1.0 / n)[:, None, None]) & mask[:, None]
        negative_hist = (~positive) & mask[:, None]
        random_items = F.normalize(self.item_emb[torch.randint(
            self.item_emb.size(0), (batch, self.recontrast_negatives), device=queries.device)].float(), dim=-1)
        tau = self.recontrast_tau
        hist_scores = torch.einsum("bkd,bnd->bkn", queries, items) / tau
        path_scores = torch.einsum("bkd,bjd->bkj", queries, queries) / tau
        eye = torch.eye(k, dtype=torch.bool, device=queries.device)[None]
        rand_scores = torch.einsum("bkd,bnd->bkn", queries, random_items) / tau
        negatives = torch.cat([
            hist_scores.masked_fill(~negative_hist, float("-inf")),
            path_scores.masked_fill(eye, float("-inf")),
            rand_scores,
        ], dim=-1)
        neg_lse = torch.logsumexp(negatives, dim=-1, keepdim=True)
        per_positive = -(hist_scores - torch.logaddexp(hist_scores, neg_lse))
        rows = (mask.sum(1) >= 2)[:, None, None] & positive
        if not bool(rows.any()):
            return queries.sum() * 0.0
        return per_positive[rows].mean()

    def _target_log_probs(self, paths, target):
        batch = paths.size(0)
        lp = self.item_log_probs(paths.flatten(0, 1)).view(batch, self.n_intents, -1)
        return lp.gather(2, target[:, None, None].expand(-1, self.n_intents, 1)).squeeze(2)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        r0 = self.base_attention(hidden_states=hidden_states, attention_mask=attention_mask,
                                 thought_id_idx=thought_id_idx)
        batch, rows = r0.size(0), torch.arange(r0.size(0), device=r0.device)
        paths = self.paths(r0)
        prior_logp = self.prior(r0.to(self.prior[1].weight.dtype)).float().log_softmax(-1)
        self.last_paths, self.last_prior_logp, self.aux_loss = paths, prior_logp, None

        if self.training and self.target_item is not None:
            extra = {}
            target = self._expand(self.target_item, batch, r0.device)
            lp_all = self.item_log_probs(paths.flatten(0, 1)).view(batch, self.n_intents, -1)
            lp_y = lp_all.gather(2, target[:, None, None].expand(-1, self.n_intents, 1)).squeeze(2)
            joint = prior_logp + lp_y
            with torch.no_grad():
                posterior = (joint if self.posterior_uses_prior else lp_y).softmax(-1)
            aux = {}
            if self.assignment == "hungarian":
                later = None if self.later_items is None else self.later_items.to(r0.device).long()
                selected, aux["assign"], extra["positives_per_row"] = self._hungarian(lp_all, target, later)
            else:
                with torch.no_grad():
                    selected = (torch.multinomial(posterior, 1).squeeze(1) if self.explore
                                else posterior.argmax(-1))
                aux["mixture"] = -torch.logsumexp(joint, dim=-1).mean()
            if self.history_items is not None:
                queries = self._item_queries(paths)
                routing, items, mask = self._routing(queries, self.history_items.to(r0.device).long())
                aux["routing_reg"] = self._routing_regularization(routing, mask)
                aux["recontrast"] = self._recontrast(queries, routing, items, mask)
                with torch.no_grad():
                    has = mask.sum(1) >= 2
                    if bool(has.any()):
                        extra["routing_max"] = routing.max(-1).values[has].mean()
            prior_loss = F.nll_loss(prior_logp, selected)
            batch_usage = prior_logp.exp().mean(0)
            usage = self.balance_momentum * self.usage_ema + (1 - self.balance_momentum) * batch_usage
            balance_loss = (usage * (usage.clamp_min(1e-8) * self.n_intents).log()).sum()
            with torch.no_grad():
                self.usage_ema.copy_(usage.detach())
                unit = F.normalize(paths - r0.float()[:, None], dim=-1)
                pair_cos = (unit @ unit.transpose(1, 2)).sum((1, 2)) - self.n_intents
                hit = lambda lp: (lp.topk(min(10, lp.size(-1)), -1).indices == target[:, None]).any(-1).float().mean()
                selected_lp = self.item_log_probs(paths[rows, selected])
                self.stats = {
                    **extra,
                    "posterior_entropy": -(posterior * posterior.clamp_min(1e-12).log()).sum(-1).mean(),
                    "prior_entropy": -(prior_logp.exp() * prior_logp).sum(-1).mean(),
                    "prior_posterior_agree": (prior_logp.argmax(-1) == selected).float().mean(),
                    "item_hit10_r0": hit(self.item_log_probs(r0)),
                    "item_hit10_selected": hit(selected_lp),
                    "item_hit10_prior_map": hit(self.item_log_probs(paths[rows, prior_logp.argmax(-1)])),
                    "path_pair_cos": pair_cos.mean() / max(1, self.n_intents * (self.n_intents - 1)),
                    "path_scale": self.path_scale.detach(),
                    **{f"usage_{k}": self.usage_ema[k] for k in range(self.n_intents)},
                }
            self.aux_loss = {**aux, "prior": prior_loss, "balance": balance_loss}
        elif self.forced_intent is not None:
            selected = self._expand(self.forced_intent, batch, r0.device)
        else:
            default = "aggregate" if self.aggregator_enabled else "map"
            mode = os.environ.get("INTENT_MODE", default)
            if mode not in INTENT_MODES:
                raise ValueError(f"unknown INTENT_MODE {mode}")
            if mode == "aggregate" and not self.aggregator_enabled:
                raise ValueError("INTENT_MODE=aggregate needs a checkpoint with a trained aggregator")
            if mode in {"aggregate", "aggregate_uniform", "prior_blend"}:
                return self.aggregate(r0, paths, prior_logp, mode).to(r0.dtype)
            if mode == "r0":
                self.last_selected = None
                return r0
            if mode == "random":
                selected = torch.randint(self.n_intents, (batch,), device=r0.device)
            elif mode == "oracle":
                if self.target_item is None:
                    raise ValueError("oracle intent needs target_item")
                target = self._expand(self.target_item, batch, r0.device)
                credit = self._target_log_probs(paths, target)
                selected = (credit + prior_logp if self.posterior_uses_prior else credit).argmax(-1)
            else:
                selected = prior_logp.argmax(-1)
        self.last_selected = selected
        return paths[rows, selected].to(r0.dtype)


class IntentLatentModel(LatentModel):
    """Training wrapper: winner-takes-all decoder CE plus the intent losses."""

    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_intent_reasoning
        self.mixture_weight = float(spec.get("mixture_weight", 0.2))
        self.assign_weight = float(spec.get("assign_weight", 0.2))
        self.recontrast_weight = float(spec.get("recontrast_weight", 0.0))
        self.routing_reg_weight = float(spec.get("routing_reg_weight", 0.0))
        self.prior_weight = float(spec.get("prior_weight", 0.1))
        self.balance_weight = float(spec.get("balance_weight", 0.01))
        self.margin = float(spec.get("margin", 0.2))
        self.margin_weight = float(spec.get("margin_weight", 0.0))
        self.grad_accum_steps = 1
        self.last_losses = {}

    @staticmethod
    def _example_nll(logits, labels):
        shift_logits, shift_labels = logits[:, :-1].float(), labels[:, 1:]
        nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                              ignore_index=-100, reduction="none").view_as(shift_labels)
        mask = shift_labels != -100
        return (nll * mask).sum(1), mask.sum(1)

    def _decode(self, embeds, attention_mask):
        return Qwen2ForCausalLM.forward(self, inputs_embeds=embeds, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, target_item=None,
                history_items=None, later_items=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        layer = self.attention
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   **kwargs)
        layer.target_item = target_item
        layer.history_items = history_items if (self.recontrast_weight or self.routing_reg_weight) else None
        layer.later_items = later_items
        try:
            embeds = self.generate_embs(input_ids, attention_mask)
        finally:
            layer.target_item = layer.history_items = layer.later_items = None
        out = self._decode(embeds, attention_mask)
        nll_sum, counts = self._example_nll(out.logits, labels)
        ce = nll_sum.sum() / counts.sum().clamp_min(1)
        aux = layer.aux_loss or {}
        weights = {"mixture": self.mixture_weight, "assign": self.assign_weight,
                   "prior": self.prior_weight, "balance": self.balance_weight,
                   "recontrast": self.recontrast_weight, "routing_reg": self.routing_reg_weight}
        loss = ce + sum(weights[name] * value for name, value in aux.items())
        margin_loss = None
        if self.margin_weight > 0 and layer.n_intents > 1:
            # Decoder must prefer the credited path over another intent of the same history.
            rows = torch.arange(input_ids.size(0), device=input_ids.device)
            other = (layer.last_selected + torch.randint(1, layer.n_intents, rows.shape,
                                                         device=rows.device)) % layer.n_intents
            positions = (input_ids == self.model.embed_tokens.num_embeddings - 1).float().argmax(1)
            wrong = embeds.clone()
            wrong[rows, positions] = layer.last_paths[rows, other].to(wrong.dtype)
            wrong_nll, _ = self._example_nll(self._decode(wrong, attention_mask).logits, labels)
            per_token = lambda s: s / counts.clamp_min(1)
            margin_loss = F.relu(self.margin + per_token(nll_sum) - per_token(wrong_nll)).mean()
            loss = loss + self.margin_weight * margin_loss
        self.last_losses = {"ce_loss": ce.detach(), **{f"{k}_loss": v.detach() for k, v in aux.items()},
                            **layer.stats}
        if margin_loss is not None:
            self.last_losses["margin_loss"] = margin_loss.detach()
        return CausalLMOutputWithPast(loss=loss / self.grad_accum_steps, logits=out.logits)
