"""End-to-end integration check for PrototypeGuidedLatentModel, using a tiny
from-scratch Qwen2 config (not the real 1.5B checkpoint) so this runs on CPU in
seconds. Verifies: model builds, forward() runs with a bank attached, splicing
produces the right shapes, loss is finite, and backward() reaches proj_up's
parameters (i.e. the retrieval -> projection -> splice -> CE gradient path is
actually connected, not just shape-compatible) AND retriever.query_proj's (via
the Straight-Through Estimator, retrieval.py::straight_through_select) -- this
is the key regression check for the STE fix: before it existed, this same
assertion was inverted (query_proj provably got NO gradient from CE alone, since
MMR's hard argmax blocks it -- confirmed and documented as the reason
commitment/diversity/balance losses were needed as a workaround). If this
assertion ever fails again, the STE wiring broke, not just an unrelated shape bug.

Run: python3 test_model_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import Qwen2Config

from bank import PrototypeBank
from model import PrototypeGuidedLatentModel

VOCAB_SIZE = 50
HIDDEN_SIZE = 16
BANK_DIM = 8
N_HEADS = 2
K_PER_HEAD = 2
THOUGHT_TOKEN_ID = VOCAB_SIZE - 1  # matches the real model's convention exactly


def build_tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        prototype_guided_reasoning={
            "bank_dim": BANK_DIM,
            "n_heads": N_HEADS,
            "k_per_head": K_PER_HEAD,
            "lam": 0.5,
        },
    )


def build_tiny_bank(n: int = 10) -> PrototypeBank:
    torch.manual_seed(0)
    return PrototypeBank(
        ids=[f"ev_{i}" for i in range(n)],
        facets=["kg_path"] * n,
        anchors=[{}] * n,
        rationales=[f"rationale {i}" for i in range(n)],
        tiers=["verified"] * n,
        support=torch.ones(n, dtype=torch.long),
        quality_score=torch.rand(n),
        embeddings=torch.randn(n, BANK_DIM),
    )


def build_batch():
    # Row 0: prompt [1,2,3], thought, target [10,11], eos(=2) -- 7 tokens
    # Row 1: shorter prompt [1], thought, target [12], eos -- 4 tokens, right-padded to 7
    pad_id = 0
    row0 = [1, 2, 3, THOUGHT_TOKEN_ID, 10, 11, 2]
    row0_labels = [-100, -100, -100, -100, 10, 11, 2]
    row1 = [1, THOUGHT_TOKEN_ID, 12, 2, pad_id, pad_id, pad_id]
    row1_labels = [-100, -100, 12, 2, -100, -100, -100]
    row1_mask = [1, 1, 1, 1, 0, 0, 0]

    input_ids = torch.tensor([row0, row1], dtype=torch.long)
    attention_mask = torch.tensor([[1] * 7, row1_mask], dtype=torch.long)
    labels = torch.tensor([row0_labels, row1_labels], dtype=torch.long)
    return input_ids, attention_mask, labels


def main() -> None:
    config = build_tiny_config()
    model = PrototypeGuidedLatentModel(config)
    model.set_prototype_bank(build_tiny_bank())
    model.train()

    input_ids, attention_mask, labels = build_batch()
    orig_seq_len = input_ids.shape[1]
    k_total = N_HEADS * K_PER_HEAD

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.isfinite(outputs.loss), f"loss is not finite: {outputs.loss}"
    print(f"loss = {outputs.loss.item():.4f} (finite: OK)")

    expected_seq_len = orig_seq_len + k_total
    assert outputs.logits.shape[1] == expected_seq_len, (
        f"expected spliced seq_len={expected_seq_len} (orig {orig_seq_len} + k_total {k_total}), "
        f"got {outputs.logits.shape[1]}"
    )
    print(f"logits seq_len = {outputs.logits.shape[1]} "
          f"(orig {orig_seq_len} + k_total {k_total}: OK)")

    assert model._last_retrieval is not None and len(model._last_retrieval) == 2
    for b, result in enumerate(model._last_retrieval):
        assert result.indices.shape == (N_HEADS, K_PER_HEAD)
        assert (result.indices >= 0).all() and (result.indices < 10).all()
    print("retrieval indices shape/range: OK")

    outputs.loss.backward()
    proj_up_grad = next(model.proto_projector.proj_up.parameters()).grad
    assert proj_up_grad is not None and proj_up_grad.abs().sum() > 0, (
        "proto_projector.proj_up got no gradient -- the retrieval->projection->splice->CE "
        "path is not actually connected"
    )
    print("proto_projector.proj_up.grad nonzero: OK (CE gradient reaches the projection)")

    query_proj_grad = next(model.retriever.query_proj.parameters()).grad
    assert query_proj_grad is not None and query_proj_grad.abs().sum() > 0, (
        "retriever.query_proj got NO gradient from CE alone -- the Straight-Through "
        "Estimator (retrieval.py::straight_through_select) is supposed to connect CE "
        "loss directly to query_proj for the first time; MMR's hard argmax still blocks "
        "gradient on its own, so if this fails the STE wiring in "
        "_generate_embs_with_prototypes is broken (e.g. use_ste=False, or e_soft's graph "
        "got detached somewhere)"
    )
    print("retriever.query_proj.grad nonzero from CE alone: OK (STE connects CE to query_proj -- "
          "this used to be impossible; see the module docstring)")

    print("\nall integration checks passed")


if __name__ == "__main__":
    main()
