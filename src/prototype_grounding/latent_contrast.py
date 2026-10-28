"""CFG-style latent contrast for constrained beam generation (P1, latent-utilization line).

Guided scores per step:  s = s_null + w * (s_r - s_null)
where s_r are the conditional log-probs (sample's latent in the thought slot, the
branch HF generate runs) and s_null the log-probs of the SAME sequence with the
train-mean latent in the slot.  Writing s_r = h + delta, s_null = h gives
h + w*delta: the shared term is retained and delta is amplified w-fold — an
amplified-incentive construction, not structural cancellation (v2 review).

The null branch keeps its own KV cache, synchronized with beam search by
matching each step's sequence prefixes against the previous step's sequences
inside every beam group (beams never cross batch rows).  Identical prefixes
have identical caches, so ties are safe.  The latent bottleneck (L7) mask is
honored: the null prefill passes ``latent_bottleneck_ids`` so the model
recomputes the (identical) block, and decode steps reuse the cached block —
the same one the conditional branch uses, valid because both branches share
the prompt ids.
"""

import torch
from transformers import LogitsProcessor


def _reorder_past(past, beam_idx):
    if isinstance(past, torch.Tensor):
        return past.index_select(0, beam_idx)
    if hasattr(past, "reorder_cache"):
        past.reorder_cache(beam_idx)
        return past
    if hasattr(past, "batch_select_indices"):
        past.batch_select_indices(beam_idx)
        return past
    if hasattr(past, "key_cache"):  # DynamicCache without the helpers
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i].index_select(0, beam_idx)
            past.value_cache[i] = past.value_cache[i].index_select(0, beam_idx)
        return past
    return tuple(tuple(t.index_select(0, beam_idx) for t in layer) for layer in past)


class ContrastGuidanceProcessor(LogitsProcessor):
    """Must run after any temperature warper and BEFORE the catalog-constraint
    processor, so the contrast acts on unconstrained distributions."""

    def __init__(self, model, null_embs, prompt_mask, prompt_ids, num_beams: int, w: float):
        self.model = model
        self.null_embs = null_embs          # [B, L, D] prompt embeddings, mean latent in the slot
        self.prompt_mask = prompt_mask      # [B, L]
        self.prompt_ids = prompt_ids        # [B, L] for the bottleneck block at prefill
        self.nb = num_beams
        self.w = float(w)
        self.past = None
        self.prev_ids = None
        self.mask = None
        self.stats = {"steps": 0, "sum_mean_abs_delta": 0.0, "sum_kl_r_null": 0.0}

    @torch.no_grad()
    def _prefill(self, input_ids):
        if input_ids.size(0) % self.null_embs.size(0):
            raise RuntimeError(
                f"contrast: rows {input_ids.size(0)} not a multiple of batch {self.null_embs.size(0)}")
        reps = input_ids.size(0) // self.null_embs.size(0)
        embs = self.null_embs.repeat_interleave(reps, 0)
        self.mask = self.prompt_mask.repeat_interleave(reps, 0)
        ids = self.prompt_ids.repeat_interleave(reps, 0)
        if not torch.equal(ids, input_ids):
            raise RuntimeError("contrast: prefill sequences do not match the prompt")
        out = self.model(inputs_embeds=embs, attention_mask=self.mask, use_cache=True,
                         latent_bottleneck_ids=ids, num_logits_to_keep=1)
        self.past = out.past_key_values
        return out.logits[:, -1, :].float()

    def _beam_reorder_index(self, input_ids):
        n = input_ids.size(0)
        beam_idx = torch.empty(n, dtype=torch.long, device=input_ids.device)
        for g in range(n // self.nb):
            sl = slice(g * self.nb, (g + 1) * self.nb)
            cur = input_ids[sl, :-1]
            prev = self.prev_ids[sl]
            eq = (cur.unsqueeze(1) == prev.unsqueeze(0)).all(-1)  # [nb, nb]
            if not bool(eq.any(dim=1).all()):
                raise RuntimeError("contrast: beam prefix not found in previous step")
            beam_idx[sl] = g * self.nb + eq.float().argmax(dim=1)
        return beam_idx

    @torch.no_grad()
    def _decode(self, input_ids):
        self.past = _reorder_past(self.past, self._beam_reorder_index(input_ids))
        ones = torch.ones(self.mask.size(0), 1, dtype=self.mask.dtype, device=self.mask.device)
        self.mask = torch.cat([self.mask, ones], dim=1)
        out = self.model(input_ids=input_ids[:, -1:], attention_mask=self.mask,
                         past_key_values=self.past, use_cache=True, num_logits_to_keep=1)
        self.past = out.past_key_values
        return out.logits[:, -1, :].float()

    def __call__(self, input_ids, scores):
        null_logits = self._prefill(input_ids) if self.past is None else self._decode(input_ids)
        self.prev_ids = input_ids
        s_r = torch.nn.functional.log_softmax(scores.float(), dim=-1)
        s_n = torch.nn.functional.log_softmax(null_logits, dim=-1)
        delta = s_r - s_n
        self.stats["steps"] += 1
        self.stats["sum_mean_abs_delta"] += float(delta.abs().mean())
        self.stats["sum_kl_r_null"] += float((s_r.exp() * delta).sum(-1).mean())
        return (s_n + self.w * delta).to(scores.dtype)

    def summary(self):
        steps = max(self.stats["steps"], 1)
        return {"steps": self.stats["steps"],
                "mean_abs_delta": self.stats["sum_mean_abs_delta"] / steps,
                "mean_kl_r_null": self.stats["sum_kl_r_null"] / steps}
