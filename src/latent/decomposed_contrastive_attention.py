"""Attempt 6 training model: explicit r = mu + z decomposition (see
decomposed_attention.py) + alpha-capped attention schedule + task-coupled
contrastive/geometry pressure applied to z only.

Three ingredients, each closing one hole (Attempt 6 in
analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md):

  1. centering  -> the shared prior can only live in `mu`        (decomposed_attention.py)
  2. fixed scale -> z cannot shrink relative to mu                (decomposed_attention.py)
  3. this file  -> the decoder must actually *read* z:
       - roll-and-splice InfoNCE (Attempt 3): the decoder must score its own
         mu + z_i above mu + z_j for every other j in the micro-batch. Because
         mu is identical across rows it cancels out of every comparison, so
         the pressure lands on z alone -- unlike Attempt 3's V0, which pushed
         all of r apart and thereby destroyed the prior (performance halved).
       - VICReg variance + covariance on the unit residual z_hat (scaled by
         sqrt(H) so an isotropic spread has per-dim std ~1): forbids the 1-D
         sample-identity code V0 found. Note the covariance term alone would
         NOT prevent that (a single active dim has zero covariance with the
         rest); the per-dim variance floor is what does.

The attention schedule (scheduled_attention.py, alpha capped below 1.0) stays
on to close the *other* shortcut: the decoder reading raw history instead of r.
Contrastive/geometry terms are skipped on `history_only` batches, where r is
discarded at the thought position and the comparison is vacuous.

Single-GPU only, like the alpha=0.75 run: the schedule draws one mode per
batch and the EMA centering buffer is not synchronized across DDP ranks.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from scheduled_attention import ScheduledLatentModel


class DecomposedContrastiveLatentModel(ScheduledLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = 0.1
        self.contrastive_weight = 1.0
        self.geom_var_weight = 1.0
        self.geom_cov_weight = 0.04
        self.geom_var_target = 1.0
        # Trainer skips its loss / gradient_accumulation_steps division for any
        # forward() with **kwargs (see margin_loss_attention.py); divide ourselves.
        self.grad_accum_steps = 1
        self.last_ce_loss = None
        self.last_contrastive_loss = None
        self.last_var_loss = None
        self.last_cov_loss = None

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
            # eval / generation: ScheduledLatentModel already falls back to vanilla "both"
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
        z_unit = self.attention.last_z_unit  # [B, H] float32, set by DecomposedSelfAttentionLayer
        batch_size = pos_embeds.size(0)
        batch_idx = torch.arange(batch_size, device=pos_embeds.device)

        mode = self._draw_mode()
        self.last_schedule_mode = mode
        if mode == "history_only":
            thought_token_id = self.model.embed_tokens.num_embeddings - 1
            plain_embed = self.model.embed_tokens.weight[thought_token_id].to(pos_embeds.dtype)
            pos_embeds = pos_embeds.clone()
            pos_embeds[batch_idx, thought_idx] = plain_embed
        if mode == "both":
            forward_attention_mask = attention_mask
        else:
            forward_attention_mask = self._scheduled_attention_bias(attention_mask, thought_idx, pos_embeds.dtype, mode)

        pos_out = self._run_second_pass(pos_embeds, forward_attention_mask, position_ids, cache_position, num_logits_to_keep)
        pos_nll = self._per_example_nll(pos_out.logits, labels)
        ce_loss = pos_nll.mean()

        if batch_size > 1 and mode != "history_only":
            nll_columns = [pos_nll]
            for shift in range(1, batch_size):
                perm = torch.roll(batch_idx, shifts=shift)
                # mu + z_j, detached: mu cancels against the positive's mu + z_i,
                # so the gradient only shapes the decoder and this row's own z_i.
                neg_r = pos_embeds[perm, thought_idx[perm]].detach()
                neg_embeds = pos_embeds.clone()
                neg_embeds[batch_idx, thought_idx] = neg_r
                neg_out = self._run_second_pass(neg_embeds, forward_attention_mask, position_ids, cache_position, num_logits_to_keep)
                nll_columns.append(self._per_example_nll(neg_out.logits, labels))
            nll_matrix = torch.stack(nll_columns, dim=1)  # [B, B]: col 0 positive
            logits = -nll_matrix / self.temperature
            targets = torch.zeros(batch_size, dtype=torch.long, device=pos_embeds.device)
            contrastive_loss = F.cross_entropy(logits, targets)

            z_scaled = z_unit * (z_unit.size(-1) ** 0.5)
            z_centered = z_scaled - z_scaled.mean(dim=0, keepdim=True)
            std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(self.geom_var_target - std).mean()
            cov = (z_centered.T @ z_centered) / (batch_size - 1)
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
