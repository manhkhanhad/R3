"""Annealed attention-pattern curriculum for LatentModel.

Does not modify layers.py, reasoning_dataset.py, or latent_attention_train.py.
Everything here is additive: delete this file (and latent_attention_train_scheduled.py)
to fully remove the approach.

See analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md for the
motivation: standard R3 lets the recommendation decoder see the raw history
tokens directly, in parallel with the latent thought `r`, so nothing forces
sample-specific dependence on `r`. This schedules, per training batch, which
path(s) the decoder is allowed to use:

  alpha >= 1        -> unrestricted attention (vanilla R3, "both")
  alpha in [0, 1)   -> one random draw u ~ U(0,1) per batch:
        u <= alpha  -> "reasoning_only": target tokens attend to the thought
                       token and each other, history tokens are hidden
        u >  alpha  -> "history_only":   target tokens attend to history and
                       each other, the thought token is hidden

Note on "history_only": attention masking alone is not enough to make target
predictions independent of `r`. Transformer residual streams carry the
embedding at a position forward regardless of attention edges, so the very
first predicted token (whose query sits at the thought position itself) would
otherwise still see `r` "for free" through the residual path even with all
attention to it blocked. To close that leak, "history_only" additionally
replaces the spliced-in `r` embedding with the plain (pre-splice) special-token
embedding at that position, so no per-example r-derived signal reaches the
loss in that mode at all -- confirmed empirically, see the smoke test notes.

Only the second (recommendation) forward pass is affected. The first pass,
which computes `r` from the history via `LatentModel.generate_embs`, is
unchanged. The schedule is only active during training (`self.training`);
eval and generation always use vanilla ("both") attention so validation loss
stays comparable to the deployed configuration throughout training.
"""

import random

import torch
from transformers import Qwen2ForCausalLM

from layers import LatentModel


class ScheduledLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = 0.0
        self.last_schedule_mode = "both"

    def _thought_positions(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        idx = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        idx[where[:, 0]] = where[:, 1]
        return idx

    def _draw_mode(self) -> str:
        if self.alpha >= 1.0:
            return "both"
        u = random.random()
        return "reasoning_only" if u <= self.alpha else "history_only"

    def _scheduled_attention_bias(
        self,
        attention_mask: torch.Tensor,
        thought_idx: torch.LongTensor,
        dtype: torch.dtype,
        mode: str,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        key_valid = attention_mask.bool()  # [batch, seq_len], padding mask over keys
        allowed = causal.unsqueeze(0) & key_valid.unsqueeze(1)  # [batch, q, k]

        if mode != "both":
            pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
            thought_pos = thought_idx.unsqueeze(1)  # [batch, 1]
            is_history_key = pos < thought_pos  # [batch, seq_len]
            is_thought_key = pos == thought_pos  # [batch, seq_len]
            is_target_query = pos >= thought_pos  # [batch, seq_len]; includes the thought position's own query

            if mode == "history_only":
                block_key = is_thought_key
            elif mode == "reasoning_only":
                block_key = is_history_key
            else:
                raise ValueError(f"unknown schedule mode {mode!r}")

            block = is_target_query.unsqueeze(2) & block_key.unsqueeze(1)  # [batch, q, k]
            allowed = allowed & (~block)

        bias = torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)
        bias = bias.masked_fill(~allowed, torch.finfo(dtype).min)
        return bias.unsqueeze(1)  # [batch, 1, q, k]; Qwen2 passes a 4D mask through unchanged

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
        forward_attention_mask = attention_mask
        is_prefill = (
            input_ids is not None
            and inputs_embeds is None
            and attention_mask is not None
            and input_ids.size() == attention_mask.size()
        )

        if is_prefill:
            thought_idx = self._thought_positions(input_ids)
            inputs_embeds = self.generate_embs(input_ids, attention_mask)
            input_ids = None

            if self.training:
                mode = self._draw_mode()
            else:
                # eval / generation prefill: always vanilla R3 ("both") behavior
                mode = "both"
            self.last_schedule_mode = mode

            if mode == "history_only":
                # see module docstring: close the residual-stream leak by discarding `r`
                # entirely at the thought position, not just masking attention to it.
                thought_token_id = self.model.embed_tokens.num_embeddings - 1
                plain_embed = self.model.embed_tokens.weight[thought_token_id].to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.clone()
                batch_idx = torch.arange(inputs_embeds.size(0), device=inputs_embeds.device)
                inputs_embeds[batch_idx, thought_idx] = plain_embed

            if mode == "both":
                forward_attention_mask = attention_mask
            else:
                forward_attention_mask = self._scheduled_attention_bias(
                    attention_mask, thought_idx, inputs_embeds.dtype, mode
                )

        return Qwen2ForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=forward_attention_mask,
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
