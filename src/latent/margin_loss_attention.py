"""In-distribution counterfactual margin loss for LatentModel.

Does not modify layers.py, reasoning_dataset.py, or latent_attention_train.py.
Additive only: delete this file plus latent_attention_train_margin.py to fully
remove the approach.

Motivation (see analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md
and the E1 causal-intervention results on the attention-schedule checkpoint):
vanilla R3 lets the decoder see the raw history directly, in parallel with the
latent thought `r`, under unrestricted ("both") attention -- the actual deployed
configuration. Nothing in plain next-token likelihood forces the decoder to
depend on which valid `r` it receives. Restricting attention during training
(the attention-schedule approach) did not fix this: mean/shuffle-thought
interventions on that checkpoint still showed ~0 ranking delta versus raw,
while the zero-mask penalty shrank relative to vanilla checkpoints -- consistent
with the model learning a *mode-conditional* policy (rely on `r` only when
history is structurally blocked) rather than genuine content-dependence under
the deployed "both" attention.

This module instead adds a loss term computed directly under unrestricted
("both") attention -- the actual deployed configuration -- requiring the
correct thought to beat a wrong one by a margin:

    -logP(y|x,r_correct) + margin <= -logP(y|x,r_wrong)

The wrong thought `r_wrong` is another example's `r` from the same training
batch (batch-shuffled by one position), matching the same "shuffle" style
intervention used in the E1 evaluation -- so this loss directly targets the
exact number that failed. `r_wrong` is detached before use, so gradient from
the hinge term does not update how that other example's own `r` is generated;
it only shapes the decoder (and, through the positive branch, this example's
own `r`) to make the correct thought distinguishable from a wrong one.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel


class MarginLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.margin = 0.2
        self.margin_weight = 1.0
        # Trainer decides whether to divide the returned loss by
        # gradient_accumulation_steps based on whether forward() has a **kwargs
        # catch-all (see transformers.Trainer.__init__, model_accepts_loss_kwargs).
        # It does here, so Trainer assumes we already normalized via
        # num_items_in_batch -- but we compute our own scalar loss and never see
        # that value. Without this, the accumulated gradient is
        # gradient_accumulation_steps times too large. The training script sets
        # this attribute to the real value; 1 is a safe no-op default (used by
        # the unit tests, and by eval/generation which don't hit this branch).
        self.grad_accum_steps = 1
        self.last_ce_loss = None
        self.last_margin_loss = None

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

        pos_out = Qwen2ForCausalLM.forward(
            self,
            inputs_embeds=pos_embeds,
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
        pos_nll = self._per_example_nll(pos_out.logits, labels)
        ce_loss = pos_nll.mean()

        batch_size = pos_embeds.size(0)
        if batch_size > 1:
            perm = torch.roll(torch.arange(batch_size, device=pos_embeds.device), shifts=1)
            batch_idx = torch.arange(batch_size, device=pos_embeds.device)
            # another example's r, detached: only this example's decoder-facing
            # comparison is supervised, not that example's own r generation.
            neg_r = pos_embeds[perm, thought_idx[perm]].detach()

            neg_embeds = pos_embeds.clone()
            neg_embeds[batch_idx, thought_idx] = neg_r

            neg_out = Qwen2ForCausalLM.forward(
                self,
                inputs_embeds=neg_embeds,
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
            neg_nll = self._per_example_nll(neg_out.logits, labels)
            hinge = F.relu(self.margin + pos_nll - neg_nll)
            margin_loss = hinge.mean()
        else:
            margin_loss = pos_nll.new_zeros(())

        self.last_ce_loss = ce_loss.detach()
        self.last_margin_loss = margin_loss.detach()

        total_loss = ce_loss + self.margin_weight * margin_loss
        # Optional WIDER regulariser on the pooled r (wider_reg.py; generate_embs stored r above).
        # This training branch bypasses LatentModel.forward, so add it here; the division by
        # grad_accum_steps below already gives the per-step scale (no loss_scale needed).
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
            logits=pos_out.logits,
            past_key_values=pos_out.past_key_values,
            hidden_states=pos_out.hidden_states,
            attentions=pos_out.attentions,
        )
