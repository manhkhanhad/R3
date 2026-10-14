"""Structural latent bottleneck for LatentModel (research log L7).

Plain R3 splices ``r = f(history)`` into the prompt and lets the target tokens
attend to both the history and ``r``; the decoder then ignores ``r``'s content
(analysis/document/decoder_insensitivity_to_latent.md).  Here, as in RGLT, the
prediction can only go through the latent: in the *second* (recommendation)
forward pass every query at or after the thought token is blocked from the
user-specific prompt span.  It may attend to

  * the fixed instruction prefix (``visible_prefix_ids`` in the config; identical
    for every row, so it carries no user information and keeps the attention sink),
  * the thought token (whose input embedding is ``r``),
  * earlier target tokens.

Everything between the end of the prefix and the thought token (the history
titles and the fixed "### Response:" scaffold) is hidden.  Position ids are
recomputed over the visible tokens only, so the target positions do not leak the
history length either.  The first pass that computes ``r`` is unchanged.

The same mask is applied in training, teacher-forced scoring, and KV-cache
generation / constrained beam search, so there is no train/test mode switch
(Attempt 1's failure mode).
"""

from typing import Optional, Sequence

import torch


def thought_positions(input_ids: torch.Tensor, thought_token_id: int) -> torch.Tensor:
    where = torch.nonzero(input_ids == thought_token_id)
    if where.size(0) != input_ids.size(0) or not torch.equal(
        where[:, 0], torch.arange(input_ids.size(0), device=input_ids.device)
    ):
        raise ValueError("latent bottleneck expects exactly one thought token per row")
    return where[:, 1]


def prompt_block(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    thought_token_id: int,
    visible_prefix_ids: Sequence[int],
):
    """Return ``(block, thought_pos)``.

    ``block[b, k]`` is True for real (non-pad) prompt keys that target-side queries
    must not see: everything before the thought token except the matched fixed
    prefix.  The prefix is matched token-by-token from the first real token, so a
    left-truncated row simply gets no visible prefix (never a visible history token).
    """
    batch, length = input_ids.shape
    device = input_ids.device
    valid = attention_mask.bool()
    thought_pos = thought_positions(input_ids, thought_token_id)
    pos = torch.arange(length, device=device).unsqueeze(0)
    first_valid = valid.long().argmax(dim=1, keepdim=True)  # left padding at eval, 0 at train

    prefix = torch.as_tensor(list(visible_prefix_ids), dtype=input_ids.dtype, device=device)
    n = int(prefix.numel())
    visible_prefix = torch.zeros(batch, length, dtype=torch.bool, device=device)
    if n > 0:
        idx = (first_valid + torch.arange(n, device=device).unsqueeze(0)).clamp(max=length - 1)
        match = (input_ids.gather(1, idx) == prefix.unsqueeze(0)) & (idx < thought_pos.unsqueeze(1))
        match = match.long().cumprod(dim=1).bool()  # stop at the first mismatch
        visible_prefix.scatter_(1, idx, match)

    before_thought = pos < thought_pos.unsqueeze(1)
    block = valid & before_thought & ~visible_prefix
    return block, thought_pos


def bottleneck_mask_and_positions(
    attention_mask: torch.Tensor,
    block: torch.Tensor,
    thought_pos: torch.Tensor,
    query_len: int,
    dtype: torch.dtype,
):
    """4D additive mask ``[B, 1, q, k]`` and position ids ``[B, q]`` for the last
    ``query_len`` positions of a sequence whose full 2D padding mask is
    ``attention_mask`` (``[B, K]``).  ``block`` covers the prompt (``[B, P]``,
    ``P <= K``); generated positions are never blocked.
    """
    batch, key_len = attention_mask.shape
    device = attention_mask.device
    if block.size(0) != batch:
        raise ValueError(f"bottleneck block batch {block.size(0)} != mask batch {batch}")
    if block.size(1) < key_len:
        block = torch.cat(
            [block, torch.zeros(batch, key_len - block.size(1), dtype=torch.bool, device=device)], dim=1
        )
    elif block.size(1) > key_len:
        block = block[:, :key_len]

    valid = attention_mask.bool()
    q_abs = torch.arange(key_len - query_len, key_len, device=device)  # [q]
    k_abs = torch.arange(key_len, device=device)  # [k]
    causal = q_abs.unsqueeze(1) >= k_abs.unsqueeze(0)  # [q, k]
    allowed = causal.unsqueeze(0) & valid.unsqueeze(1)  # [B, q, k]
    is_target_query = q_abs.unsqueeze(0) >= thought_pos.unsqueeze(1)  # [B, q]
    allowed = allowed & ~(is_target_query.unsqueeze(2) & block.unsqueeze(1))
    # Never leave a row empty (pad queries): let every query see itself.
    allowed = allowed | (q_abs.unsqueeze(1) == k_abs.unsqueeze(0)).unsqueeze(0)

    bias = torch.zeros(batch, query_len, key_len, dtype=dtype, device=device)
    bias = bias.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)

    visible = valid & ~block
    positions = (visible.long().cumsum(dim=1) - 1).clamp(min=0)[:, key_len - query_len:]
    return bias, positions

