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

``INTENT_MODE`` (eval): ``map`` (default) | ``r0`` | ``random`` | ``oracle``;
a forced intent (set by the eval script) overrides it.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel
from reasoning_dataset import LatentRDataset

INTENT_MODES = {"map", "r0", "random", "oracle"}


class IntentDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        if not self.test:
            example["target_item"] = int(self.data.iloc[idx]["item_id"])
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

        self.forced_intent = None   # int or per-prompt tensor (eval)
        self.target_item = None     # per-row tensor (training, oracle eval)
        self.explore = False        # sample k* from q instead of argmax
        self.aux_loss = None
        self.stats = {}
        self.last_paths = None
        self.last_selected = None
        self.last_prior_logp = None

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
            target = self._expand(self.target_item, batch, r0.device)
            lp_y = self._target_log_probs(paths, target)
            joint = prior_logp + lp_y
            mixture_loss = -torch.logsumexp(joint, dim=-1).mean()
            with torch.no_grad():
                posterior = (joint if self.posterior_uses_prior else lp_y).softmax(-1)
                selected = (torch.multinomial(posterior, 1).squeeze(1) if self.explore
                            else posterior.argmax(-1))
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
            self.aux_loss = {"mixture": mixture_loss, "prior": prior_loss, "balance": balance_loss}
        elif self.forced_intent is not None:
            selected = self._expand(self.forced_intent, batch, r0.device)
        else:
            mode = os.environ.get("INTENT_MODE", "map")
            if mode not in INTENT_MODES:
                raise ValueError(f"unknown INTENT_MODE {mode}")
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, target_item=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        layer = self.attention
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   **kwargs)
        layer.target_item = target_item
        try:
            embeds = self.generate_embs(input_ids, attention_mask)
        finally:
            layer.target_item = None
        out = self._decode(embeds, attention_mask)
        nll_sum, counts = self._example_nll(out.logits, labels)
        ce = nll_sum.sum() / counts.sum().clamp_min(1)
        aux = layer.aux_loss or {}
        loss = ce + self.mixture_weight * aux["mixture"] + self.prior_weight * aux["prior"] \
            + self.balance_weight * aux["balance"]
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
