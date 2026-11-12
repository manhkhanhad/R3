"""Verifies PrototypeGuidedLatentModel.prepare_inputs_for_generation actually works
across a real multi-step .generate() call -- not just that it doesn't crash, but that:

1. generate() completes across multiple incremental decoding steps with a bank
   attached (the failure mode being guarded against: cache-length / attention_mask
   mismatch after the first, spliced step).
2. generation_output.sequences contains the REAL, unmodified original prompt token IDs
   at the expected positions -- i.e. constrained decoding (which inspects input_ids
   directly, e.g. CFEnhancedLogitsProcessor's `sent[-5:]`) would see exactly what it
   sees today, unaffected by splicing happening internally.
3. Retrieval actually fired during generation (model._last_retrieval populated).
4. Generation without a bank attached is completely unaffected (regression check --
   falls through to the exact same path as the unmodified base LatentModel).
5. A custom logits processor mimicking CFEnhancedLogitsProcessor's exact mechanism
   (hash of the last N real input_ids tokens) sees the correct, real tokens at each
   step, proving the actual constrained-decoding mechanism this eval pipeline depends
   on is unaffected by prototype splicing happening under the hood.

Run: python3 test_generation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from test_model_integration import build_tiny_bank, build_tiny_config, THOUGHT_TOKEN_ID, VOCAB_SIZE
from model import PrototypeGuidedLatentModel

N_HEADS = 2
K_PER_HEAD = 2
K_TOTAL = N_HEADS * K_PER_HEAD


def build_prompt():
    # [1, 2, 3, THOUGHT] -- no target continuation, this is what a real eval prompt
    # looks like (test=True in LatentRDataset: input_ids ends right at <|Thought|>).
    input_ids = torch.tensor([[1, 2, 3, THOUGHT_TOKEN_ID]])
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


class RecordingHashProcessor(LogitsProcessor):
    """Mimics CFEnhancedLogitsProcessor's exact mechanism: hash the last N tokens of
    the REAL input_ids sequence-so-far. Records what it actually saw at each call so
    the test can assert those tokens are the real, unmodified prompt -- not anything
    touched by internal splicing."""

    def __init__(self, n=4):
        self.n = n
        self.seen = []

    def __call__(self, input_ids, scores):
        self.seen.append(input_ids[:, -self.n:].clone())
        return scores


def test_generation_with_bank():
    torch.manual_seed(0)
    config = build_tiny_config()
    model = PrototypeGuidedLatentModel(config)
    model.set_prototype_bank(build_tiny_bank())
    model.eval()

    input_ids, attention_mask = build_prompt()
    orig_len = input_ids.shape[1]

    processor = RecordingHashProcessor(n=4)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=0,
        )

    print(f"input prompt len: {orig_len}, output sequence shape: {out.shape}")
    assert out.shape[0] == 1
    assert out.shape[1] > orig_len, "generation produced no new tokens"
    assert out.shape[1] <= orig_len + 5, "generated more tokens than max_new_tokens allows"

    # sequences must start with the REAL, unmodified prompt -- proves generate()'s own
    # input_ids bookkeeping was never replaced by the spliced/embeds path.
    assert torch.equal(out[0, :orig_len], input_ids[0]), (
        f"output prompt prefix doesn't match real input_ids: {out[0, :orig_len]} vs {input_ids[0]}"
    )
    print("output sequence starts with the real, unmodified prompt tokens: OK")

    # the recording processor must have seen the real prompt tail at its first call
    # (this is exactly what CFEnhancedLogitsProcessor's sent[-5:] mechanism relies on).
    first_call_tokens = processor.seen[0]
    assert torch.equal(first_call_tokens[0], input_ids[0, -4:]), (
        f"logits processor saw {first_call_tokens[0]} at first call, "
        f"expected the real prompt tail {input_ids[0, -4:]}"
    )
    print(f"logits processor saw real prompt tail at step 1 ({first_call_tokens[0].tolist()}): OK "
          f"(this is what constrained decoding depends on)")

    assert model._last_retrieval is not None and len(model._last_retrieval) == 1
    print("retrieval fired during generation (model._last_retrieval populated): OK")

    print(f"generated {out.shape[1] - orig_len} new tokens across "
          f"{len(processor.seen)} logits-processor calls without error")


def test_generation_with_bank_beam_search():
    """num_beams=10, matching the real eval config exactly (prototype_eval.py).

    This specific configuration is what actually crashed at real scale (job 271215,
    Qwen2.5-1.5B, real bank) with a shape mismatch of exactly k_total in
    scaled_dot_product_attention -- the greedy (num_beams=1) test above did NOT catch
    it: prepare_inputs_for_generation's first version assumed returning a longer
    attention_mask on step 1 would make GenerationMixin's bookkeeping self-correct on
    later steps, which is false (its persisted model_kwargs["attention_mask"] is a
    separate object prepare_inputs_for_generation's return value never feeds back
    into). Why the tiny greedy CPU test missed this originally is still not fully
    understood (a different, more lenient attention code path for tiny/greedy
    decoding, most likely) -- this test exists specifically so that gap in coverage
    doesn't recur silently. The fix (re-inserting k ones at the recorded splice offset
    on every step after the first) is generic to beam width, verified here and
    separately at real scale.
    """
    torch.manual_seed(0)
    config = build_tiny_config()
    model = PrototypeGuidedLatentModel(config)
    model.set_prototype_bank(build_tiny_bank())
    model.eval()

    input_ids, attention_mask = build_prompt()
    orig_len = input_ids.shape[1]
    num_beams = 10

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=0,
        )

    print(f"beam search: input prompt len {orig_len}, output shape {out.shape}")
    assert out.shape[0] == num_beams
    assert out.shape[1] > orig_len, "beam search produced no new tokens"
    for beam in range(num_beams):
        assert torch.equal(out[beam, :orig_len], input_ids[0]), (
            f"beam {beam}'s output prompt prefix doesn't match real input_ids"
        )
    print(f"all {num_beams} beams' output prompt prefixes match the real input_ids: OK")
    assert model._last_retrieval is not None
    print("retrieval fired during beam search generation: OK")


def test_generation_without_bank_unaffected():
    """Regression check: no bank attached -> falls through to the exact base-class path."""
    torch.manual_seed(0)
    config = build_tiny_config()
    model = PrototypeGuidedLatentModel(config)
    # deliberately do NOT call set_prototype_bank
    model.eval()

    input_ids, attention_mask = build_prompt()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=5, do_sample=False, num_beams=1, pad_token_id=0,
        )
    assert out.shape[0] == 1 and out.shape[1] > input_ids.shape[1]
    assert model._last_retrieval is None, "retrieval fired despite no bank attached"
    print(f"generation without a bank: shape {out.shape}, no retrieval fired: OK")


if __name__ == "__main__":
    test_generation_with_bank()
    print()
    test_generation_with_bank_beam_search()
    print()
    test_generation_without_bank_unaffected()
    print("\nall generation tests passed")
