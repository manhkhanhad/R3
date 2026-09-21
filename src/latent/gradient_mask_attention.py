"""Gradient-Mask training for LatentModel, adapted from "Faithfulness as
Information Flow: Evaluating and Training Faithful Chain-of-Thought
Reasoning" (arXiv:2605.24286), Eq. 8. Does not modify layers.py,
reasoning_dataset.py, or latent_attention_train.py. Additive only: delete
this file plus latent_attention_train_gradmask.py to remove the approach.

Motivation (see avoid_reasoning_collapse_and_shortcut_learning.md, Attempt 3):
every approach so far that forced genuine content-dependence on `r` did so by
also reducing what the decoder can compute -- attention-schedule and CASE's
L_Zdrop both structurally hide history from the decoder some or all of the
time (real information loss, real forward-pass capacity cost), and Attempt 3's
V0 achieved strong content-dependence only by collapsing `r` into a damaged,
near-1-dimensional code, roughly halving task performance. Gradient Mask
targets the actual trade-off directly: the *forward* pass is completely
unchanged (the decoder still sees history and `r` together, full capacity,
identical to vanilla R3) -- only the *backward* pass is altered, so the
optimizer cannot get gradient credit for routing information through the
direct history-to-target attention edges specifically. All other edges
(target-to-r, target-to-target, and everything in the first pass that
computes `r`) are completely unaffected.

Formally, for attention score entry z_ij (query i, key j), CASE's Eq. 8 is:

    z~_ij = sg(G_ij * z_ij) + (1 - G_ij) * z_ij,   G_ij = 1[i is a target
                                                            position AND
                                                            j is a history
                                                            position]

where sg is stop-gradient. Since sg(x) has the same *value* as x but zero
gradient, z~_ij == z_ij numerically for every entry (forward pass identical
to vanilla), while entries with G_ij=1 contribute no gradient to whatever
produced them (Q/K projections), and entries with G_ij=0 behave normally.
This is implemented here as `torch.where(block, attn_scores.detach(),
attn_scores)`, which is exactly this formula (block=True picks the detached
branch, block=False picks the attached one) applied to the pre-softmax
attention scores, before the causal/padding bias is folded in via softmax.

Unlike attention-mask-based approaches (the schedule, CASE's L_Zdrop), this
cannot be expressed as a bias added to the standard `attention_mask` argument
passed into `Qwen2ForCausalLM.forward()`, because a bias affects the forward
and backward passes identically -- we specifically need them to differ. It
requires intercepting the raw attention-score computation inside every
transformer layer, which the public `attention_mask` API does not expose.
This is done here by monkey-patching the module-level
`eager_attention_forward` function that `Qwen2Attention.forward()` looks up
by name at every call (see modeling_qwen2.py) -- requires `attn_implementation
="eager"`. The patched function is a strict no-op wrapper around the
original when inactive (checked via a thread-local flag, off by default), so
importing this module is harmless for any other model/script in the same
process; it only alters behavior inside the `with _GradientMaskScope(...)`
block in `GradientMaskLatentModel.forward()`'s training branch.

Because this only changes gradient flow (not the loss computation itself,
and not the input embeddings), the standard `Qwen2ForCausalLM.forward(...,
labels=...)` built-in loss path is used unchanged -- no custom NLL, no extra
forward passes, and critically no grad_accum_steps bug (unlike the margin/
contrastive/causal attempts): Trainer's own num_items_in_batch-aware
normalization applies exactly as it does for vanilla LatentModel, since we
never bypass it.
"""

import threading

import torch
import torch.nn.functional as F
import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
from transformers import Qwen2ForCausalLM

from layers import LatentModel

_state = threading.local()
_original_eager_attention_forward = qwen2_modeling.eager_attention_forward


def _gradient_masked_eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout: float = 0.0, **kwargs
):
    if not getattr(_state, "active", False):
        return _original_eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )

    thought_idx = _state.thought_idx
    key_states = qwen2_modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_modeling.repeat_kv(value, module.num_key_value_groups)

    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_scores = attn_scores + causal_mask

    seq_len_q, seq_len_k = attn_scores.size(-2), attn_scores.size(-1)
    device = attn_scores.device
    q_pos = torch.arange(seq_len_q, device=device).view(1, 1, -1, 1)
    k_pos = torch.arange(seq_len_k, device=device).view(1, 1, 1, -1)
    thought_pos = thought_idx.view(-1, 1, 1, 1)
    is_target_query = q_pos >= thought_pos  # includes the thought position's own query
    is_history_key = k_pos < thought_pos
    block = is_target_query & is_history_key  # [batch, 1, seq_len_q, seq_len_k]

    # Eq. 8: same forward value everywhere, zero gradient into Q/K for blocked entries.
    attn_scores = torch.where(block, attn_scores.detach(), attn_scores)

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_modeling.eager_attention_forward = _gradient_masked_eager_attention_forward


class _GradientMaskScope:
    def __init__(self, thought_idx: torch.LongTensor):
        self.thought_idx = thought_idx

    def __enter__(self):
        _state.active = True
        _state.thought_idx = self.thought_idx
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state.active = False
        _state.thought_idx = None
        return False


class GradientMaskLatentModel(LatentModel):
    def _thought_positions(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        idx = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        idx[where[:, 0]] = where[:, 1]
        return idx

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
            # eval / generation / no-label calls: identical to vanilla LatentModel,
            # gradient mask never active (no backward pass happens here anyway)
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
        inputs_embeds = self.generate_embs(input_ids, attention_mask)

        with _GradientMaskScope(thought_idx):
            return Qwen2ForCausalLM.forward(
                self,
                input_ids=None,
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
