"""Two-phase plain R3: counterfactual margin first, then collaborative InfoNCE.

Combines Attempt 2 (``margin_loss_attention.py``) with Attempt 10
(``collaborative_contrastive_attention.py``):

* Phase 1 (``contrastive_enabled=False``): CE + margin hinge, where the
  negative branch replaces each example's ``r`` with another in-batch example's
  detached ``r``.  Correct latents are still pushed to the FIFO corpus, so the
  queue is already full when phase 2 starts.
* Phase 2 (``contrastive_enabled=True``): the user/target multi-positive
  InfoNCE terms are added on top.  The margin term stays active unless
  ``margin_weight`` is set to 0 for that phase.

The training script toggles the phase from the Trainer epoch.  Inference is
the ordinary LatentModel path.
"""

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel


class CollaborativeMarginLatentModel(CollaborativeContrastiveLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.margin = 0.2
        self.margin_weight = 1.0
        self.contrastive_enabled = False
        self.last_margin_loss = None
        self.last_contrastive_enabled = None
        self.last_wider_loss = None

    def _margin_loss(self, embeds, thought_idx, pos_nll, labels, attention_mask,
                     position_ids, cache_position, num_logits_to_keep):
        batch_size = embeds.size(0)
        if batch_size < 2 or self.margin_weight == 0:
            return pos_nll.new_zeros(())
        rows = torch.arange(batch_size, device=embeds.device)
        perm = torch.roll(rows, shifts=1)
        # Another example's r, detached: the hinge shapes the decoder and this
        # example's own r, not how the donor's r is generated.
        neg_embeds = embeds.clone()
        neg_embeds[rows, thought_idx] = embeds[perm, thought_idx[perm]].detach()
        neg_out = self._run_second_pass(
            neg_embeds, attention_mask, position_ids, cache_position, num_logits_to_keep
        )
        _, neg_nll, _ = self._loss_and_quality(neg_out.logits, labels)
        return F.relu(self.margin + pos_nll - neg_nll).mean()

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
        target_title_features=None,
        target_item_index=None,
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
                num_logits_to_keep=num_logits_to_keep,
                user_hash=user_hash, target_hash=target_hash, sample_hash=sample_hash,
                target_title_features=target_title_features,
                target_item_index=target_item_index, **kwargs,
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
        margin_loss = self._margin_loss(
            embeds, thought_idx, example_nll, labels, attention_mask,
            position_ids, cache_position, num_logits_to_keep,
        )

        if self.contrastive_enabled:
            user_loss, target_loss, cf_loss, user_rate, target_rate, cf_rate = (
                self._collaborative_losses(
                    r, user_hash, target_hash, sample_hash, quality,
                    target_title_features, target_item_index,
                )
            )
        else:
            user_loss = target_loss = cf_loss = r.sum() * 0.0
            user_rate = target_rate = cf_rate = user_loss.detach()
        # The queue fills in both phases so phase 2 starts with a full corpus.
        self._push_bank(
            r, user_hash, target_hash, sample_hash, example_nll, quality,
            target_title_features, target_item_index,
        )

        self.last_ce_loss = ce_loss.detach()
        self.last_margin_loss = margin_loss.detach()
        self.last_user_contrastive_loss = user_loss.detach()
        self.last_target_contrastive_loss = target_loss.detach()
        self.last_cf_contrastive_loss = cf_loss.detach()
        self.last_user_positive_rate = user_rate
        self.last_target_positive_rate = target_rate
        self.last_cf_positive_rate = cf_rate
        self.last_queue_accept_rate = quality.float().mean().detach()
        self.last_queue_count = self.bank_count.detach().clone()
        self.last_contrastive_enabled = torch.tensor(float(self.contrastive_enabled))

        total_loss = (
            ce_loss
            + self.margin_weight * margin_loss
            + self.user_contrastive_weight * user_loss
            + self.target_contrastive_weight * target_loss
            + self.cf_contrastive_weight * cf_loss
        )
        # Optional WIDER regulariser on the pooled r (wider_reg.py; generate_embs stored r above).
        # Added before the /grad_accum_steps below, as in MarginLatentModel (no loss_scale needed).
        if getattr(self.config, "latent_wider", None):
            from wider_reg import wider_state
            state = wider_state(self)
            penalty = state.penalty() if state is not None else None
            if penalty is not None:
                self.last_wider_loss = penalty.detach()
                total_loss = total_loss + state.weight * penalty
        total_loss = total_loss / self.grad_accum_steps
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
