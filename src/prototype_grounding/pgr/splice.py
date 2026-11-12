"""Insert k prototype positions into a batch's embedding sequence (Mechanism step 4,
Training curriculum Phase 3).

The rest of this pipeline (LatentRDataset, DataCollatorForSeq2Seq) produces sequences
shaped [...prompt tokens..., <|Thought|>, target tokens..., eos] with right-padding:
each row's real content occupies [0, real_len_i), padding (attention_mask=0) fills the
rest up to the batch's max length. `labels` is -100 for everything through the thought
token (only the target continuation contributes to CE).

Splicing inserts the k projected prototype vectors immediately after each row's own
thought-token position -- `Concat(x_i, r_i, p_1, ..., p_k)` from the design doc, with
target tokens (and whatever padding already followed them) shifted right by k. This is
a plain positional insert, content-agnostic: it doesn't matter whether what's being
shifted is real target tokens or padding, the row is just cut at `thought_idx[i] + 1`
and k new positions spliced in.

Left alone (position_ids=None on the actual forward call): HF's default position-id
derivation for a right-padded batch gives the newly-inserted positions the correct
next sequential index, same as it would for any other extended sequence -- no special
RoPE handling needed here since this all happens before Qwen2's own transformer stack
runs, not inside the custom SelfAttentionLayer/RoPE machinery (that only ever sees the
original, unspliced sequence, since it computes r from the first pass before retrieval
happens at all).
"""
from __future__ import annotations

import torch


def splice_prototypes(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    thought_idx: torch.Tensor,
    prototype_embeds: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    inputs_embeds: [B, L, H]
    attention_mask: [B, L]
    thought_idx: [B] long -- index of the <|Thought|> token in each row (pre-splice)
    prototype_embeds: [B, k, H] -- already projected into H-dim, per-row ordered
    labels: [B, L] or None

    Returns (inputs_embeds', attention_mask', labels') with L' = L + k.
    """
    B, L, H = inputs_embeds.shape
    k = prototype_embeds.shape[1]
    if k == 0:
        return inputs_embeds, attention_mask, labels

    new_L = L + k
    new_embeds = inputs_embeds.new_zeros(B, new_L, H)
    new_mask = attention_mask.new_zeros(B, new_L)
    new_labels = None if labels is None else labels.new_full((B, new_L), -100)

    for i in range(B):
        insert_at = int(thought_idx[i].item()) + 1
        if insert_at > L:
            raise ValueError(f"thought_idx[{i}]={thought_idx[i].item()} out of range for seq_len={L}")

        new_embeds[i, :insert_at] = inputs_embeds[i, :insert_at]
        new_mask[i, :insert_at] = attention_mask[i, :insert_at]

        new_embeds[i, insert_at:insert_at + k] = prototype_embeds[i]
        new_mask[i, insert_at:insert_at + k] = 1
        # new_labels at these k positions stays -100 (prototype positions are never
        # a prediction target), already set by new_full above.

        new_embeds[i, insert_at + k:] = inputs_embeds[i, insert_at:]
        new_mask[i, insert_at + k:] = attention_mask[i, insert_at:]
        if labels is not None:
            new_labels[i, :insert_at] = labels[i, :insert_at]
            new_labels[i, insert_at + k:] = labels[i, insert_at:]

    return new_embeds, new_mask, new_labels
