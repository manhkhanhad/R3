"""Multi-negative InfoNCE + temperature curriculum + VICReg-style geometry
regularizer for LatentModel. Does not modify layers.py, reasoning_dataset.py,
latent_attention_train.py, or the margin-loss attempt. Additive only: delete
this file plus latent_attention_train_contrastive.py to remove the approach.

Motivation (see analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md,
Attempt 2, and the follow-up literature review): the in-distribution
counterfactual margin/hinge loss (margin_loss_attention.py) fixed collapse at
margin=0.2 but catastrophically collapsed geometry at margin=0.5 -- the hinge
term was pinned at the ceiling (never satisfied) in both runs, and a hinge that
can't be satisfied applies constant, non-saturating gradient pressure, which
triplet-loss literature documents as a known cause of degenerate "scrambled"
solutions. A single random in-batch negative is also weak/high-variance.

This module replaces the single-negative hinge with three changes:

1. Multi-negative InfoNCE: for each anchor, ALL other examples in the same
   micro-batch (not just one) serve as negatives. Built cheaply via the same
   roll-and-splice trick as the margin loss, generalized to every nonzero
   shift (1..batch_size-1), giving batch_size total forward passes per
   micro-batch (1 positive + batch_size-1 negatives) instead of building a
   batch_size^2 comparison matrix. Negatives are detached, as in the margin
   loss, so gradient only shapes the decoder and each example's own (positive)
   thought, never another example's thought via its use as someone else's
   negative.
2. Temperature curriculum instead of a fixed hinge margin: temperature anneals
   from `temp_start` (soft, easy) to `temp_end` (sharp, hard) linearly over
   training. Unlike a hard margin, a softmax-based loss is always
   differentiable and saturates gracefully as separation grows, so there is no
   "unreachable target" failure mode -- this directly targets the confirmed
   mechanism behind the margin=0.5 regression.
3. A VICReg-style variance + covariance penalty directly on the (positive)
   thought vectors in each micro-batch, decoupled from the contrastive loss:
   variance term keeps each dimension from collapsing to a constant, the
   covariance term decorrelates dimensions. This directly optimizes the same
   effective-rank / PC1-concentration diagnostics used to evaluate collapse,
   independent of whatever the contrastive loss is doing. Caveat: with a small
   micro-batch (4 by default) the covariance estimate is noisy (only 3 degrees
   of freedom for a 1536-dim vector); it is kept at a low weight by default
   for this reason, acting as a soft, cumulative-over-many-steps nudge rather
   than a precise per-step estimate.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel


class ContrastiveGeometryLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = 1.0
        self.contrastive_weight = 1.0
        self.geom_var_weight = 1.0
        self.geom_cov_weight = 0.04
        self.geom_var_target = 1.0
        # See margin_loss_attention.py for why this is necessary: Trainer skips
        # its own loss/gradient_accumulation_steps division whenever forward()
        # has a **kwargs catch-all, assuming num_items_in_batch-based
        # normalization already happened. We compute our own scalar loss and
        # never see that value, so we must divide ourselves.
        self.grad_accum_steps = 1
        self.last_ce_loss = None
        self.last_contrastive_loss = None
        self.last_var_loss = None
        self.last_cov_loss = None

    def _thought_positions(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        idx = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        idx[where[:, 0]] = where[:, 1]
        return idx

    @staticmethod
    def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())
        mask = (shift_labels != -100).float()
        return (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        is_prefill = (
            input_ids is not None
            and inputs_embeds is None
            and attention_mask is not None
            and input_ids.size() == attention_mask.size()
        )

        if not (is_prefill and self.training and labels is not None):
            # eval / generation / no-label calls: identical to vanilla LatentModel
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )

        thought_idx = self._thought_positions(input_ids)
        pos_embeds = self.generate_embs(input_ids, attention_mask)
        batch_size = pos_embeds.size(0)
        batch_idx = torch.arange(batch_size, device=pos_embeds.device)

        pos_out = self._run_second_pass(pos_embeds, attention_mask, position_ids, cache_position, num_logits_to_keep)
        pos_nll = self._per_example_nll(pos_out.logits, labels)
        ce_loss = pos_nll.mean()

        r_vec = pos_embeds[batch_idx, thought_idx]  # [B, H], this example's own thought

        if batch_size > 1:
            nll_columns = [pos_nll]  # column 0 = positive (shift 0)
            for shift in range(1, batch_size):
                perm = torch.roll(batch_idx, shifts=shift)
                # another example's thought, detached: gradient shapes the
                # decoder and this row's own (positive) thought, never the
                # borrowed negative's generation.
                neg_r = pos_embeds[perm, thought_idx[perm]].detach()
                neg_embeds = pos_embeds.clone()
                neg_embeds[batch_idx, thought_idx] = neg_r
                neg_out = self._run_second_pass(neg_embeds, attention_mask, position_ids, cache_position, num_logits_to_keep)
                nll_columns.append(self._per_example_nll(neg_out.logits, labels))
            nll_matrix = torch.stack(nll_columns, dim=1)  # [B, B]: col 0 positive, cols 1..B-1 negatives
            logits = -nll_matrix / self.temperature
            targets = torch.zeros(batch_size, dtype=torch.long, device=pos_embeds.device)
            contrastive_loss = F.cross_entropy(logits, targets)

            r_centered = r_vec.float() - r_vec.float().mean(dim=0, keepdim=True)
            std = torch.sqrt(r_centered.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(self.geom_var_target - std).mean()
            cov = (r_centered.T @ r_centered) / (batch_size - 1)
            hidden = cov.size(0)
            off_diag_mask = ~torch.eye(hidden, dtype=torch.bool, device=cov.device)
            cov_loss = cov[off_diag_mask].pow(2).sum() / hidden
        else:
            contrastive_loss = pos_nll.new_zeros(())
            var_loss = pos_nll.new_zeros(())
            cov_loss = pos_nll.new_zeros(())

        self.last_ce_loss = ce_loss.detach()
        self.last_contrastive_loss = contrastive_loss.detach()
        self.last_var_loss = var_loss.detach()
        self.last_cov_loss = cov_loss.detach()

        total_loss = (
            ce_loss
            + self.contrastive_weight * contrastive_loss
            + self.geom_var_weight * var_loss
            + self.geom_cov_weight * cov_loss
        ) / self.grad_accum_steps

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=pos_out.logits,
            past_key_values=pos_out.past_key_values,
            hidden_states=pos_out.hidden_states,
            attentions=pos_out.attentions,
        )
