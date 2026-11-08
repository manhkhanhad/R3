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

Gist variant (``"gist": True``): the K thought tokens are learned vectors
(continuous_thoughts ``source="constant"``) processed in the *same* pass.  The
first K-1 are the latents: their queries may read the whole history at every
layer.  The last one is a readout token on the target side: like the target
tokens it sees only the prefix, the latents and itself, and it predicts the
first target token (otherwise that prediction would come from a latent position
that reads the history directly).  The target therefore reads the history only
through the latents' per-layer keys/values (gisting, Mu et al. 2023).  Position ids are left unchanged (the target positions
reveal the history length; renumbering would distort the thought-to-history
distances).  ``"delete_latents": True`` (evaluation) also hides the thoughts from
the target side.

Soft variant (``"history_bias": b``, a finite negative number): instead of hiding
the blocked keys, the target-side queries get a fixed additive bias ``b`` on their
attention logits (every layer and head), identical in training and generation.
The history stays readable at a cost (e.g. for copying a brand), while ``r`` and
the instruction are unpenalised.  Position ids are left unchanged (the history is
not removed).  ``b -> -inf`` recovers the hard bottleneck, ``b = 0`` plain R3.
"""

from typing import Optional, Sequence

import torch


def thought_positions(input_ids: torch.Tensor, thought_token_id: int, n_thoughts: int = 1) -> torch.Tensor:
    """First thought position per row.  With K continuous thoughts the K thought
    tokens are contiguous; every query from the first one on is target-side."""
    if n_thoughts > 1:
        from continuous_thoughts import first_thought_positions
        return first_thought_positions(input_ids, thought_token_id, n_thoughts)
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
    n_thoughts: int = 1,
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
    thought_pos = thought_positions(input_ids, thought_token_id, n_thoughts)
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
    gist_k: int = 0,
    delete_latents: bool = False,
    history_bias: Optional[float] = None,
    keep_positions: bool = False,
    all_queries: bool = False,
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
    # gist: the latents are thoughts 0..K-2, the readout (thought K-1) is target-side
    readout = thought_pos + max(gist_k - 1, 0)
    is_target_query = q_abs.unsqueeze(0) >= readout.unsqueeze(1)  # [B, q]
    if delete_latents:
        is_latent_key = (k_abs.unsqueeze(0) >= thought_pos.unsqueeze(1)) & (k_abs.unsqueeze(0) < readout.unsqueeze(1))
        block = block | is_latent_key
    # all_queries (history dropout): hide the blocked keys from every query, so no state of
    # this pass (separators, scaffold, kept titles) encodes a dropped item; only r (first pass) does
    penalised = block.unsqueeze(1).expand(-1, query_len, -1) if all_queries else is_target_query.unsqueeze(2) & block.unsqueeze(1)
    if history_bias is None:
        allowed = allowed & ~penalised
    # Never leave a row empty (pad queries): let every query see itself.
    allowed = allowed | (q_abs.unsqueeze(1) == k_abs.unsqueeze(0)).unsqueeze(0)

    bias = torch.zeros(batch, query_len, key_len, dtype=dtype, device=device)
    if history_bias is not None:
        bias = bias.masked_fill(penalised & allowed, float(history_bias))
    bias = bias.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)

    # keep_positions (history dropout): hidden keys keep their position ids, as in the soft variant
    visible = valid if (gist_k or history_bias is not None or keep_positions) else valid & ~block
    positions = (visible.long().cumsum(dim=1) - 1).clamp(min=0)[:, key_len - query_len:]
    return bias, positions

