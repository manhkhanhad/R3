"""PrototypeGuidedLatentModel: Phase 3 wiring (design doc, Mechanism + Training curriculum).

Subclasses `src/latent/layers.py`'s `LatentModel` (the class `latent_attention_train.py`
actually trains) rather than editing it in place -- keeps this feature additive and
easy to A/B against the unmodified baseline, and matches how the existing
`latent_decomposition` (mu/z) variant is already built as an alternate `self.attention`
rather than a fork of the whole class.

What changes vs. the base class: `generate_embs` still computes `r` exactly as before
(unchanged, inherited via `super().generate_embs`) and splices it into the
`<|Thought|>` position -- but if a prototype bank has been attached
(`set_prototype_bank`), `forward()` then retrieves `k = n_heads * k_per_head`
prototypes per instance, projects them, and inserts them as `k` new positions right
after the thought token (`splice.splice_prototypes`), growing the sequence length by
`k` for that forward pass. `attention_mask` and `labels` are extended consistently.

Training-time forward (`input_ids.size() == attention_mask.size()`, a full pass, no KV
cache yet) is handled directly in `forward()`, above. Generation-time -- `.generate()`,
incremental decoding with a KV cache -- is handled separately via
`prepare_inputs_for_generation` (below): splicing happens once, on the first (prefill)
call, and everything after self-corrects because HF's own bookkeeping extends whatever
attention_mask that first call returns rather than recomputing it from the original
prompt length. See that method's docstring for why this is NOT done by switching
generate() to inputs_embeds= (it breaks this dataset's constrained decoding, which
needs real token IDs for the prompt's last few tokens).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent"))
from layers import LatentModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import PrototypeBank  # noqa: E402
from projection import PrototypePairProjector  # noqa: E402
from retrieval import (  # noqa: E402
    PrototypeRetriever, RetrievalResult, soft_bank_attention, straight_through_select,
)
from splice import splice_prototypes  # noqa: E402


def extract_r(model: LatentModel, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """r_i for each row, from any LatentModel-family instance (plain or prototype-guided).

    Used by Stage 2's commitment-loss warm-start (Training curriculum) to get real r's
    from a frozen Stage-1 checkpoint, without needing PrototypeGuidedLatentModel at all
    -- this only needs generate_embs and the thought-token convention, both inherited
    from the base LatentModel.
    """
    with torch.no_grad():
        inputs_embeds = model.generate_embs(input_ids, attention_mask)
        thought_token_id = model.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        batch_size = input_ids.shape[0]
        thought_idx = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        thought_idx[where[:, 0]] = where[:, 1]
        return inputs_embeds[torch.arange(batch_size, device=inputs_embeds.device), thought_idx]


class PrototypeGuidedLatentModel(LatentModel):

    def __init__(self, config):
        super().__init__(config)
        pgr_cfg = getattr(config, "prototype_guided_reasoning", None) or {}
        self.pgr_enabled = bool(pgr_cfg)
        self.prototype_bank: Optional[PrototypeBank] = None
        self._last_retrieval: Optional[list[RetrievalResult]] = None

        if self.pgr_enabled:
            bank_dim = pgr_cfg["bank_dim"]
            self.retriever = PrototypeRetriever(
                hidden_size=config.hidden_size,
                bank_dim=bank_dim,
                n_heads=pgr_cfg.get("n_heads", 3),
                k_per_head=pgr_cfg.get("k_per_head", 3),
                lam=pgr_cfg.get("lam", 0.5),
            )
            self.proto_projector = PrototypePairProjector(
                bank_dim=bank_dim, hidden_size=config.hidden_size,
            )
            # Straight-Through Estimator (retrieval.py::straight_through_select):
            # query_proj otherwise never receives task-relevant gradient at all (MMR's
            # hard argmax blocks CE entirely; commitment/diversity/balance losses are
            # all proxies about how retrieval behaves, never about whether it helps
            # predict the right answer -- confirmed insufficient on their own, see the
            # design doc's Head collapse risk). use_ste=False is kept as an ablation
            # lever, not because it's expected to be a good setting.
            self.use_ste = pgr_cfg.get("use_ste", True)
            self.ste_temperature = pgr_cfg.get("ste_temperature", 0.1)

    def set_prototype_bank(self, bank: PrototypeBank) -> None:
        """Attach the (fixed, non-trainable) prototype bank. Not part of the model's
        own state_dict -- re-attach after every from_pretrained() load."""
        if not self.pgr_enabled:
            raise RuntimeError(
                "this checkpoint's config has no prototype_guided_reasoning block; "
                "construct the model with that config set before attaching a bank"
            )
        self.prototype_bank = bank

    def _generate_embs_with_prototypes(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = super().generate_embs(input_ids, attention_mask)  # r already spliced in-place

        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        batch_size = input_ids.shape[0]
        thought_idx = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        thought_idx[where[:, 0]] = where[:, 1]

        r_batch = inputs_embeds[torch.arange(batch_size, device=inputs_embeds.device), thought_idx]

        retrieval_results = self.retriever.retrieve(r_batch, self.prototype_bank)
        self._last_retrieval = retrieval_results

        # bank.py always stores float32; proto_projector may be bf16/fp16 (e.g.
        # from_pretrained(torch_dtype=bf16) constructs it that way directly) -- match
        # both device and dtype for its input, not just device.
        proj_dtype = next(self.proto_projector.parameters()).dtype
        full_bank = self.prototype_bank.embeddings.to(device=inputs_embeds.device, dtype=proj_dtype)
        full_quality = self.prototype_bank.quality_score.to(device=inputs_embeds.device, dtype=proj_dtype)
        proto_embeds_per_row = []
        for result in retrieval_results:
            per_head_embeds = []
            per_head_quality = []
            for a in range(result.indices.shape[0]):
                head_idx = result.indices[a]
                head_idx = head_idx[head_idx >= 0]  # drop padding sentinel (only fires if bank smaller than k)
                e_hard = full_bank[head_idx]  # [k_a, D], real MMR selection, forward value always this
                if self.use_ste:
                    query_a = result.queries[a].to(device=inputs_embeds.device, dtype=proj_dtype)
                    e_soft = soft_bank_attention(query_a, full_bank, temperature=self.ste_temperature)  # [D]
                    e_hard = straight_through_select(e_hard, e_soft.unsqueeze(0).expand_as(e_hard))
                per_head_embeds.append(e_hard)
                per_head_quality.append(full_quality[head_idx])
            bank_embeddings = torch.cat(per_head_embeds, dim=0)  # [k_total, D]
            bank_quality = torch.cat(per_head_quality, dim=0)  # [k_total]
            proto_embeds_per_row.append(
                self.proto_projector.project(bank_embeddings, bank_quality).to(inputs_embeds.dtype)
            )
        prototype_embeds = torch.stack(proto_embeds_per_row, dim=0)  # [B, k_total, H]

        return splice_prototypes(inputs_embeds, attention_mask, thought_idx, prototype_embeds, labels)

    def prepare_spliced_prompt(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public entry point for prototype-guided generation.

        forward()'s own splicing only fires on the training-time "full pass"
        (input_ids.size() == attention_mask.size()) -- true for .generate()'s first
        (prefill) call, false for every incremental decoding step after. Splicing
        *inside* forward() would grow the KV cache by k positions on that first call
        without GenerationMixin's external attention_mask/position bookkeeping knowing,
        breaking every step after it (cache length and external bookkeeping would
        silently disagree).

        The fix: splice once, here, *before* generate() is ever called, and hand the
        result to generate() as an ordinary (longer) prompt via inputs_embeds= --
        exactly like prefix-tuning / soft-prompt integration. From that point on
        generate()'s incremental decoding needs no special-casing at all, since nothing
        about the sequence length changes unexpectedly mid-generation.

        Returns (inputs_embeds, attention_mask) sized for the spliced prompt -- pass
        both directly to generate(inputs_embeds=..., attention_mask=...). Note
        generate() then returns only the newly generated tokens in `sequences` (no
        input_ids were given, so there's nothing for it to echo back) -- callers must
        not slice a "prompt length" off the output the way input_ids-based generation
        conventionally requires.
        """
        if self.prototype_bank is None:
            raise RuntimeError("attach a bank via set_prototype_bank() before generating")
        inputs_embeds, attention_mask, _ = self._generate_embs_with_prototypes(
            input_ids, attention_mask, labels=None,
        )
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        full_pass = (
            input_ids is not None
            and inputs_embeds is None
            and attention_mask is not None
            and input_ids.size() == attention_mask.size()
        )
        if full_pass and self.prototype_bank is not None:
            inputs_embeds, attention_mask, labels = self._generate_embs_with_prototypes(
                input_ids, attention_mask, labels,
            )
            input_ids = None
        # Anything else (no bank attached, or a KV-cache continuation call during
        # generation) falls through to the base class unmodified -- including the
        # ordinary full_pass-without-a-bank case, which reduces to exactly the
        # original LatentModel behavior since inputs_embeds is still None there.
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
        cache_position=None, **kwargs,
    ):
        """Makes .generate() prototype-aware, called by GenerationMixin every step.

        Deliberately does NOT switch generate() itself to inputs_embeds= (see
        prepare_spliced_prompt's docstring for why that path breaks constrained
        decoding: CFEnhancedLogitsProcessor's hash-based lookup needs real token IDs
        for the prompt's last few tokens, which inputs_embeds-based generation can't
        provide). Instead: call generate(input_ids=..., attention_mask=...) completely
        normally -- GenerationMixin's own input_ids/logits-processor bookkeeping then
        sees real, unmodified token IDs throughout, exactly as it does today -- and
        intercept only the *first* (prefill, past_key_values empty) call here to swap
        in the spliced, longer inputs_embeds/attention_mask for that one forward()
        computation.

        CORRECTED after a real-scale failure (beam search, num_beams=10): the first
        version of this method assumed returning a longer attention_mask on step 1
        would make GenerationMixin's own bookkeeping "self-correct" on every later
        step. It doesn't -- prepare_inputs_for_generation's return value is used only
        for that one forward() call; _update_model_kwargs_for_generation extends its
        own separately-persisted model_kwargs["attention_mask"], which never saw the
        splice, from the *original* unspliced length. The actual cache ends up k
        positions longer than that persisted mask believes, and step 2 crashes with a
        shape mismatch inside scaled_dot_product_attention (confirmed: target/existing
        sizes differed by exactly k_total). Fixed by explicitly re-inserting k ones
        into whatever attention_mask this method receives on every step *after* the
        first, at the same fixed offset (orig_len, recorded on step 1) the splice
        itself used.

        That fixed offset relies on this eval pipeline's own convention
        (tokenizer.padding_side = "left" for generation, set in prototype_eval.py) --
        left-padding means <|Thought|> sits at the same absolute last position for
        every row in the batch, so splicing always lands at the same offset across the
        whole batch. This would need per-row offsets instead if padding_side were ever
        "right" for generation.

        cache_position is omitted from the first-call return (not passed through):
        GenerationMixin computes it before calling this method as
        torch.arange(0, input_ids.shape[1]) -- sized for the original, unspliced
        prompt length, wrong for the spliced sequence this call actually runs on.
        Leaving it out lets Qwen2's own forward() derive it fresh from the real
        inputs_embeds shape (past_seen_tokens=0 on this first call), correct by
        construction.
        """
        no_cache_yet = past_key_values is None or (
            hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0
        )
        if no_cache_yet:
            # Reset unconditionally at the start of every fresh generate() call, not
            # just when a bank is attached -- otherwise stale state from a *previous*
            # generate() call (e.g. one that did splice) would wrongly pad the mask on
            # a later call that has no bank attached at all.
            self._gen_splice_k = None
        if no_cache_yet and self.prototype_bank is not None and inputs_embeds is None:
            self._gen_splice_orig_len = attention_mask.shape[1]
            spliced_embeds, spliced_mask = self.prepare_spliced_prompt(input_ids, attention_mask)
            self._gen_splice_k = spliced_mask.shape[1] - self._gen_splice_orig_len
            return {
                "inputs_embeds": spliced_embeds,
                "attention_mask": spliced_mask,
                "use_cache": kwargs.get("use_cache", True),
            }

        splice_k = getattr(self, "_gen_splice_k", None)
        if splice_k and attention_mask is not None:
            orig_len = self._gen_splice_orig_len
            pad_ones = attention_mask.new_ones((attention_mask.shape[0], splice_k))
            attention_mask = torch.cat(
                [attention_mask[:, :orig_len], pad_ones, attention_mask[:, orig_len:]], dim=1,
            )
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs,
        )
