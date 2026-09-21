"""Attempt 8: memory-bank semantic alignment for z.

Builds on the mu/z decomposition from Attempt 7
(decomposed_attention.VAEDecomposedSelfAttentionLayer) but replaces the
aggregate-KL prior as the *primary* z mechanism with a decoder-coupled
alignment loss, motivated by: "the same user or same target item should have
similar reasoning."

Why this is routed through the decoder rather than being a raw z-space
pull/push (cosine or L2 distance between z vectors): a distance-only
constraint is a marginal/pairwise property of z's own distribution,
independent of whether the decoder ever needs that structure -- exactly the
gap that broke Attempt 7a (good geometry, decoder routed around it via the
unconstrained mu channel). Instead:

  - POSITIVE (same user_id or same target_item, found via a memory-bank queue
    of recent (z, user_hash, target_hash, own_nll) tuples from past steps):
    splice the candidate's z onto the ANCHOR's own mu -- r = mu_i + z_j -- and
    minimize the resulting NLL against the ANCHOR's own true target. This
    asks "does this other identity's reasoning still work in my context?",
    weighted by stopgrad(exp(-own_nll_j)): a candidate whose own reasoning
    didn't lead to a correct prediction is down-weighted as a positive, so a
    good anchor isn't dragged toward a low-quality reference.
  - NEGATIVE (different user_id AND different target_item): same splice,
    hinge-pushed up to a margin (reusing margin_loss_attention.py's
    mechanism) -- an unrelated identity's reasoning should predict badly.

mu cancels out of neither term (unlike Attempt 6's negative construction,
where mu was a single shared parameter) -- here mu is the anchor's own,
per-example, history-conditioned mean, kept fixed while z is swapped, so the
comparison is specifically about z's portability across identities.

The alpha-capped attention schedule (scheduled_attention.py) and the
mu-consistency identifiability anchor (7b) are both kept. Single-GPU only:
the memory bank and mu EMA are not DDP-synchronized.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from vae_decomposed_attention import VAEDecomposedLatentModel


class SemanticAlignmentLatentModel(VAEDecomposedLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_weight = 1.0
        self.neg_weight = 1.0
        self.margin = 2.0
        self.bank_size = 4096
        H = config.hidden_size
        self.register_buffer("bank_z", torch.zeros(0, H, dtype=torch.float32), persistent=False)
        self.register_buffer("bank_user", torch.zeros(0, dtype=torch.int64), persistent=False)
        self.register_buffer("bank_target", torch.zeros(0, dtype=torch.int64), persistent=False)
        self.register_buffer("bank_nll", torch.zeros(0, dtype=torch.float32), persistent=False)
        self.last_pos_loss = None
        self.last_neg_loss = None
        self.last_pos_hit_rate = None
        self.last_neg_hit_rate = None
        self.last_pos_weight_mean = None

    def _push_bank(self, z, user_hash, target_hash, nll):
        with torch.no_grad():
            self.bank_z = torch.cat([z.detach().float(), self.bank_z.to(z.device)], dim=0)[: self.bank_size]
            self.bank_user = torch.cat([user_hash.detach(), self.bank_user.to(z.device)], dim=0)[: self.bank_size]
            self.bank_target = torch.cat([target_hash.detach(), self.bank_target.to(z.device)], dim=0)[: self.bank_size]
            self.bank_nll = torch.cat([nll.detach().float(), self.bank_nll.to(z.device)], dim=0)[: self.bank_size]

    def _sample_matches(self, user_hash, target_hash):
        """For each anchor, randomly pick one bank index sharing user_id or
        target_item (positive) and one sharing neither (negative). Returns
        (pos_idx, has_pos, neg_idx, has_neg), all shape [B]."""
        device = user_hash.device
        Q = self.bank_user.size(0)
        if Q == 0:
            empty = torch.zeros(user_hash.size(0), dtype=torch.long, device=device)
            none = torch.zeros(user_hash.size(0), dtype=torch.bool, device=device)
            return empty, none, empty, none
        same_user = user_hash.unsqueeze(1) == self.bank_user.to(device).unsqueeze(0)      # [B, Q]
        same_target = target_hash.unsqueeze(1) == self.bank_target.to(device).unsqueeze(0)  # [B, Q]
        is_pos = same_user | same_target
        is_neg = (~same_user) & (~same_target)

        rand = torch.rand(is_pos.shape, device=device)
        pos_idx = torch.where(is_pos, rand, torch.full_like(rand, -1.0)).argmax(dim=1)
        neg_idx = torch.where(is_neg, rand, torch.full_like(rand, -1.0)).argmax(dim=1)
        return pos_idx, is_pos.any(dim=1), neg_idx, is_neg.any(dim=1)

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

    @staticmethod
    def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        token_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, reduction="none",
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
        user_hash=None,
        target_hash=None,
        **kwargs,
    ):
        is_prefill = (
            input_ids is not None
            and inputs_embeds is None
            and attention_mask is not None
            and input_ids.size() == attention_mask.size()
        )
        if not (is_prefill and self.training and labels is not None and user_hash is not None):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
                cache_position=cache_position, num_logits_to_keep=num_logits_to_keep, **kwargs,
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
        own_nll = self._per_example_nll(out.logits, labels)
        ce_loss = own_nll.mean()

        if mode != "history_only":
            pos_idx, has_pos, neg_idx, has_neg = self._sample_matches(user_hash, target_hash)

            def spliced_nll(z_candidate):
                r = mu + z_candidate
                spliced = embeds.clone()
                spliced[batch_idx, thought_idx] = r.to(spliced.dtype)
                out_s = self._run_second_pass(spliced, forward_attention_mask, position_ids, cache_position, num_logits_to_keep)
                return self._per_example_nll(out_s.logits, labels)

            if has_pos.any():
                z_pos = self.bank_z.to(z.device)[pos_idx]
                nll_pos = spliced_nll(z_pos)
                weight_pos = torch.exp(-torch.nan_to_num(self.bank_nll.to(z.device)[pos_idx], nan=20.0, posinf=20.0).clamp(max=20.0))
                pos_terms = weight_pos * nll_pos
                pos_loss = (pos_terms * has_pos.float()).sum() / has_pos.float().sum().clamp(min=1.0)
                self.last_pos_weight_mean = (weight_pos * has_pos.float()).sum().detach() / has_pos.float().sum().clamp(min=1.0)
            else:
                pos_loss = ce_loss.new_zeros(())
                self.last_pos_weight_mean = ce_loss.new_zeros(())

            if has_neg.any():
                z_neg = self.bank_z.to(z.device)[neg_idx]
                nll_neg = spliced_nll(z_neg)
                neg_terms = F.relu(self.margin - nll_neg)
                neg_loss = (neg_terms * has_neg.float()).sum() / has_neg.float().sum().clamp(min=1.0)
            else:
                neg_loss = ce_loss.new_zeros(())

            self._push_bank(z, user_hash, target_hash, own_nll)
            self.last_pos_hit_rate = has_pos.float().mean().detach()
            self.last_neg_hit_rate = has_neg.float().mean().detach()
        else:
            pos_loss = ce_loss.new_zeros(())
            neg_loss = ce_loss.new_zeros(())
            self.last_pos_hit_rate = ce_loss.new_zeros(())
            self.last_neg_hit_rate = ce_loss.new_zeros(())

        self.last_ce_loss = ce_loss.detach()
        self.last_pos_loss = pos_loss.detach()
        self.last_neg_loss = neg_loss.detach()

        total_loss = (
            ce_loss + self.pos_weight * pos_loss + self.neg_weight * neg_loss
        ) / self.grad_accum_steps

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
