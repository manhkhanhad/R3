"""Standalone correctness check for splice_prototypes -- no model/checkpoint needed.

Run: python3 test_splice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from splice import splice_prototypes


def test_basic_insertion_and_shift():
    # Row 0: thought at idx 2, real content [0..2]=prompt+thought, [3,4]=target, [5]=pad
    # Row 1: thought at idx 1 (shorter prompt), [0,1]=prompt+thought, [2]=target, [3,4,5]=pad
    B, L, H, k = 2, 6, 3, 2
    inputs_embeds = torch.arange(B * L * H, dtype=torch.float32).reshape(B, L, H)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0],
    ])
    labels = torch.tensor([
        [-100, -100, -100, 10, 11, -100],
        [-100, -100, 20, -100, -100, -100],
    ])
    thought_idx = torch.tensor([2, 1])
    prototype_embeds = torch.full((B, k, H), -1.0)  # distinctive marker value

    new_embeds, new_mask, new_labels = splice_prototypes(
        inputs_embeds, attention_mask, thought_idx, prototype_embeds, labels,
    )

    assert new_embeds.shape == (B, L + k, H)
    assert new_mask.shape == (B, L + k)
    assert new_labels.shape == (B, L + k)

    # Row 0: prompt+thought [0:3] unchanged, prototypes at [3:5], target shifted to [5:7]
    assert torch.equal(new_embeds[0, :3], inputs_embeds[0, :3])
    assert torch.equal(new_embeds[0, 3:5], prototype_embeds[0])
    assert torch.equal(new_embeds[0, 5:], inputs_embeds[0, 3:])
    assert new_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    assert new_labels[0].tolist() == [-100, -100, -100, -100, -100, 10, 11, -100]

    # Row 1: prompt+thought [0:2] unchanged, prototypes at [2:4], rest (target+pad) shifted to [4:8]
    assert torch.equal(new_embeds[1, :2], inputs_embeds[1, :2])
    assert torch.equal(new_embeds[1, 2:4], prototype_embeds[1])
    assert torch.equal(new_embeds[1, 4:], inputs_embeds[1, 2:])
    assert new_mask[1].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert new_labels[1].tolist() == [-100, -100, -100, -100, 20, -100, -100, -100]

    print("test_basic_insertion_and_shift: OK")


def test_no_labels():
    B, L, H, k = 1, 3, 2, 1
    inputs_embeds = torch.zeros(B, L, H)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    thought_idx = torch.tensor([1])
    prototype_embeds = torch.ones(B, k, H)

    new_embeds, new_mask, new_labels = splice_prototypes(
        inputs_embeds, attention_mask, thought_idx, prototype_embeds, labels=None,
    )
    assert new_labels is None
    assert new_embeds.shape == (B, L + k, H)
    print("test_no_labels: OK")


def test_k_zero_is_noop():
    B, L, H = 1, 3, 2
    inputs_embeds = torch.randn(B, L, H)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    thought_idx = torch.tensor([1])
    prototype_embeds = torch.zeros(B, 0, H)

    new_embeds, new_mask, new_labels = splice_prototypes(
        inputs_embeds, attention_mask, thought_idx, prototype_embeds,
    )
    assert new_embeds is inputs_embeds
    assert new_mask is attention_mask
    print("test_k_zero_is_noop: OK")


if __name__ == "__main__":
    test_basic_insertion_and_shift()
    test_no_labels()
    test_k_zero_is_noop()
    print("all splice tests passed")
