"""Attempt 7 training model: VAE-shaped encoder (mu_i, z_i = attention(x)),
r_i = mu_i + z_i, no sampling, prior N(0, I) imposed on z by an aggregate KL.

  L = CE + beta * KL( N(m, diag(s^2)) || N(0, I) )          (7a)
        + gamma * mean_i ||mu_i - mu_ema||^2 / H              (7b, gamma > 0)

  KL per dim = 0.5 * (s_d^2 + m_d^2 - 1 - log s_d^2), averaged over dims.
  m, s^2 are the mean / per-dim variance of z over [current micro-batch ; queue
  of recent detached z].  A micro-batch of 4 alone gives a 3-dof variance
  estimate, far too noisy for the log barrier; the queue gives a stable
  estimate while gradient still flows through the current rows.

What the KL does: m -> 0 pushes shared content out of z into mu; s^2 -> 1 with
a log barrier makes it impossible to shrink any dimension of z (and forbids a
1-D code), and is two-sided so z cannot explode either.  ||mu_i|| is fixed by
the layer (see decomposed_attention.VAEDecomposedSelfAttentionLayer) so the
mu:z ratio cannot be gamed through the mean.

The alpha-capped attention schedule (scheduled_attention.py) is kept, as in
Attempt 6, to hold the decoder-side x-shortcut guard constant across attempts;
the only difference from Attempt 6 is the z mechanism.  No contrastive term.
KL / consistency terms are skipped on `history_only` batches, where r is
discarded and z has no path to the loss.  Single-GPU only.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from scheduled_attention import ScheduledLatentModel


class VAEDecomposedLatentModel(ScheduledLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.kl_weight = 1.0
        self.mu_consistency_weight = 0.0
        self.z_queue_size = 256
        self.grad_accum_steps = 1
        self.register_buffer("z_queue", torch.zeros(0, config.hidden_size, dtype=torch.float32), persistent=False)
        self.last_ce_loss = None
        self.last_kl_loss = None
        self.last_mu_consistency_loss = None
        self.last_z_mean_sq = None
        self.last_z_var = None

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

    def _aggregate_kl(self, z: torch.Tensor) -> torch.Tensor:
        pool = torch.cat([z, self.z_queue.to(z.device)], dim=0) if self.z_queue.numel() else z
        m = pool.mean(dim=0)
        var = pool.var(dim=0, unbiased=False) + 1e-6
        kl = 0.5 * (var + m.pow(2) - 1.0 - torch.log(var))
        self.last_z_mean_sq = m.pow(2).mean().detach()
        self.last_z_var = var.mean().detach()
        return kl.mean()

    def _push_queue(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            self.z_queue = torch.cat([z.detach().float(), self.z_queue.to(z.device)], dim=0)[: self.z_queue_size]

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
        embeds = self.generate_embs(input_ids, attention_mask)
        mu, z = self.attention.last_mu, self.attention.last_z  # [B, H] float32
        batch_idx = torch.arange(embeds.size(0), device=embeds.device)

        mode = self._draw_mode()
        self.last_schedule_mode = mode
        if mode == "history_only":
            thought_token_id = self.model.embed_tokens.num_embeddings - 1
            plain_embed = self.model.embed_tokens.weight[thought_token_id].to(embeds.dtype)
            embeds = embeds.clone()
            embeds[batch_idx, thought_idx] = plain_embed
        if mode == "both":
            forward_attention_mask = attention_mask
        else:
            forward_attention_mask = self._scheduled_attention_bias(attention_mask, thought_idx, embeds.dtype, mode)

        out = self._run_second_pass(embeds, forward_attention_mask, position_ids, cache_position, num_logits_to_keep)
        shift_logits = out.logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        if mode != "history_only":
            kl_loss = self._aggregate_kl(z)
            self._push_queue(z)
            if self.mu_consistency_weight > 0 and bool(self.attention.mu_ema_initialized):
                mu_consistency = (mu - self.attention.mu_ema.unsqueeze(0)).pow(2).mean()
            else:
                mu_consistency = ce_loss.new_zeros(())
        else:
            kl_loss = ce_loss.new_zeros(())
            mu_consistency = ce_loss.new_zeros(())

        self.last_ce_loss = ce_loss.detach()
        self.last_kl_loss = kl_loss.detach()
        self.last_mu_consistency_loss = mu_consistency.detach()

        total_loss = (
            ce_loss + self.kl_weight * kl_loss + self.mu_consistency_weight * mu_consistency
        ) / self.grad_accum_steps

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
