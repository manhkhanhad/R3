"""Collaborative multi-positive InfoNCE on the latent vector of plain R3.

This is deliberately additive: the R3 encoder, the full latent ``r``, and the
recommendation decoder path are unchanged.  During training only, recent
high-quality latent vectors are kept in a non-persistent FIFO corpus.  Two
projection heads organize that shared latent space independently by user and
by target item.  The heads and corpus do not participate in inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel


class CollaborativeContrastiveLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        settings = getattr(config, "collaborative_contrastive", {}) or {}
        hidden = config.hidden_size
        projection_dim = int(settings.get("projection_dim", 256))
        self.bank_size = int(settings.get("bank_size", 10_000))

        self.user_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False)
        )
        self.target_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False)
        )

        # The queue is run-local state: stale vectors must not enter a resumed
        # model merely because they happened to be present when it was saved.
        self.register_buffer(
            "bank_r", torch.zeros(self.bank_size, hidden, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "bank_user", torch.zeros(self.bank_size, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "bank_target", torch.zeros(self.bank_size, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "bank_sample", torch.zeros(self.bank_size, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "bank_nll", torch.zeros(self.bank_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("bank_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("bank_count", torch.zeros((), dtype=torch.int64), persistent=False)

        self.temperature = 0.1
        self.user_contrastive_weight = 0.1
        self.target_contrastive_weight = 0.1
        self.max_user_positives = 16
        self.max_target_positives = 16
        self.num_negatives = 256
        self.positive_min_token_accuracy = 1.0
        self.positive_max_nll = float("inf")
        self.grad_accum_steps = 1

        self.last_ce_loss = None
        self.last_user_contrastive_loss = None
        self.last_target_contrastive_loss = None
        self.last_user_positive_rate = None
        self.last_target_positive_rate = None
        self.last_queue_accept_rate = None
        self.last_queue_count = None

    def _thought_positions(self, input_ids):
        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        positions = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        positions[where[:, 0]] = where[:, 1]
        return positions

    @staticmethod
    def _loss_and_quality(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        token_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        mask = shift_labels != -100
        counts = mask.sum(dim=1).clamp(min=1)
        example_nll = (token_nll * mask).sum(dim=1) / counts
        token_accuracy = ((shift_logits.argmax(dim=-1) == shift_labels) & mask).sum(dim=1) / counts
        ce_loss = (token_nll * mask).sum() / mask.sum().clamp(min=1)
        return ce_loss, example_nll, token_accuracy

    def _run_second_pass(self, embeds, attention_mask, position_ids, cache_position, num_logits_to_keep):
        return Qwen2ForCausalLM.forward(
            self,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

    def _bank_view(self):
        """Return valid queue entries in oldest-to-newest FIFO order."""
        count = int(self.bank_count.item())
        if count == 0:
            return (
                self.bank_r[:0], self.bank_user[:0], self.bank_target[:0],
                self.bank_sample[:0], self.bank_nll[:0],
            )
        if count < self.bank_size:
            sl = slice(0, count)
            return (
                self.bank_r[sl], self.bank_user[sl], self.bank_target[sl],
                self.bank_sample[sl], self.bank_nll[sl],
            )
        pointer = int(self.bank_pointer.item())
        order = torch.cat(
            [torch.arange(pointer, self.bank_size, device=self.bank_r.device),
             torch.arange(0, pointer, device=self.bank_r.device)]
        )
        return (
            self.bank_r[order], self.bank_user[order], self.bank_target[order],
            self.bank_sample[order], self.bank_nll[order],
        )

    @torch.no_grad()
    def _push_bank(self, r, user_hash, target_hash, sample_hash, nll, quality):
        keep = torch.nonzero(quality, as_tuple=False).flatten()
        if keep.numel() == 0:
            return
        r = r.detach()[keep].to(device=self.bank_r.device, dtype=self.bank_r.dtype)
        user_hash = user_hash.detach()[keep].to(self.bank_user.device)
        target_hash = target_hash.detach()[keep].to(self.bank_target.device)
        sample_hash = sample_hash.detach()[keep].to(self.bank_sample.device)
        nll = nll.detach()[keep].to(self.bank_nll.device, dtype=self.bank_nll.dtype)

        if r.size(0) > self.bank_size:
            r, user_hash, target_hash, sample_hash, nll = (
                value[-self.bank_size:]
                for value in (r, user_hash, target_hash, sample_hash, nll)
            )
        pointer = int(self.bank_pointer.item())
        first = min(r.size(0), self.bank_size - pointer)
        second = r.size(0) - first

        def copy_ring(buffer, values):
            buffer[pointer:pointer + first].copy_(values[:first])
            if second:
                buffer[:second].copy_(values[first:])

        for buffer, values in (
            (self.bank_r, r), (self.bank_user, user_hash),
            (self.bank_target, target_hash), (self.bank_sample, sample_hash),
            (self.bank_nll, nll),
        ):
            copy_ring(buffer, values)
        self.bank_pointer.fill_((pointer + r.size(0)) % self.bank_size)
        self.bank_count.fill_(min(self.bank_size, int(self.bank_count.item()) + r.size(0)))

    @staticmethod
    def _sample(indices, limit):
        if limit <= 0 or indices.numel() <= limit:
            return indices
        return indices[torch.randperm(indices.numel(), device=indices.device)[:limit]]

    def _relation_loss(
        self, anchor_r, candidate_r, anchor_user, anchor_target,
        anchor_sample, candidate_user, candidate_target, candidate_sample,
        projection, relation, max_positives,
    ):
        zero = anchor_r.sum() * 0.0
        if candidate_r.size(0) == 0:
            return zero, zero.detach()

        projection_dtype = next(projection.parameters()).dtype
        anchors = F.normalize(projection(anchor_r.to(projection_dtype)).float(), dim=-1)
        candidates = F.normalize(projection(candidate_r.to(projection_dtype)).float(), dim=-1)
        not_self = anchor_sample[:, None] != candidate_sample[None, :]
        if relation == "user":
            positives = not_self & (anchor_user[:, None] == candidate_user[None, :])
        else:
            positives = not_self & (anchor_target[:, None] == candidate_target[None, :])
        negatives = (
            not_self
            & (anchor_user[:, None] != candidate_user[None, :])
            & (anchor_target[:, None] != candidate_target[None, :])
        )

        losses = []
        has_positive = positives.any(dim=1)
        for row in range(anchor_r.size(0)):
            pos_idx = self._sample(torch.nonzero(positives[row], as_tuple=False).flatten(), max_positives)
            neg_idx = self._sample(torch.nonzero(negatives[row], as_tuple=False).flatten(), self.num_negatives)
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            pos_scores = anchors[row] @ candidates[pos_idx].T / self.temperature
            neg_scores = anchors[row] @ candidates[neg_idx].T / self.temperature
            # Multi-positive InfoNCE: maximize the probability mass assigned
            # to any eligible positive rather than trusting one random match.
            losses.append(
                torch.logsumexp(torch.cat([pos_scores, neg_scores]), dim=0)
                - torch.logsumexp(pos_scores, dim=0)
            )
        loss = torch.stack(losses).mean() if losses else zero
        return loss, has_positive.float().mean().detach()

    def _collaborative_losses(self, r, user_hash, target_hash, sample_hash, quality):
        bank_r, bank_user, bank_target, bank_sample, _ = self._bank_view()
        current = torch.nonzero(quality, as_tuple=False).flatten()
        candidate_r = torch.cat([bank_r.to(r.device), r.detach()[current]], dim=0)
        candidate_user = torch.cat([bank_user.to(r.device), user_hash[current]], dim=0)
        candidate_target = torch.cat([bank_target.to(r.device), target_hash[current]], dim=0)
        candidate_sample = torch.cat([bank_sample.to(r.device), sample_hash[current]], dim=0)

        common = (
            r, candidate_r, user_hash, target_hash, sample_hash,
            candidate_user, candidate_target, candidate_sample,
        )
        user_loss, user_rate = self._relation_loss(
            *common, self.user_projection, "user", self.max_user_positives
        )
        target_loss, target_rate = self._relation_loss(
            *common, self.target_projection, "target", self.max_target_positives
        )
        return user_loss, target_loss, user_rate, target_rate

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        user_hash=None,
        target_hash=None,
        sample_hash=None,
        **kwargs,
    ):
        is_training_prefill = (
            self.training and labels is not None and input_ids is not None
            and inputs_embeds is None and attention_mask is not None
            and input_ids.size() == attention_mask.size()
            and user_hash is not None and target_hash is not None and sample_hash is not None
        )
        if not is_training_prefill:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep, **kwargs,
            )

        embeds = self.generate_embs(input_ids, attention_mask)
        thought_idx = self._thought_positions(input_ids)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        r = embeds[rows, thought_idx].float()
        out = self._run_second_pass(
            embeds, attention_mask, position_ids, cache_position, num_logits_to_keep
        )
        ce_loss, example_nll, token_accuracy = self._loss_and_quality(out.logits, labels)
        quality = (
            (token_accuracy >= self.positive_min_token_accuracy)
            & (example_nll <= self.positive_max_nll)
        )

        user_loss, target_loss, user_rate, target_rate = self._collaborative_losses(
            r, user_hash, target_hash, sample_hash, quality
        )
        self._push_bank(r, user_hash, target_hash, sample_hash, example_nll, quality)

        self.last_ce_loss = ce_loss.detach()
        self.last_user_contrastive_loss = user_loss.detach()
        self.last_target_contrastive_loss = target_loss.detach()
        self.last_user_positive_rate = user_rate
        self.last_target_positive_rate = target_rate
        self.last_queue_accept_rate = quality.float().mean().detach()
        self.last_queue_count = self.bank_count.detach().clone()

        total_loss = (
            ce_loss
            + self.user_contrastive_weight * user_loss
            + self.target_contrastive_weight * target_loss
        ) / self.grad_accum_steps
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
