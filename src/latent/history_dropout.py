"""Stochastic history dropout for the recommendation pass (variational line, option C).

Variational reading of R3's ignored latent: ``r = f(history)`` is a deterministic
function of what the decoder already reads, so ``I(y; r | history) = 0`` and the
decoder's shortcut through the history tokens is the information-theoretically
correct solution (Wang, Blei & Cunningham 2021: collapse iff the latent is
non-identifiable).  Word dropout (Bowman et al. 2016) / modality dropout reverse
the *gradient* preference: if the decoder's view of the history is unreliable, the
only complete summary is ``r``.

Here, in the recommendation pass only, a random subset of the **history title
tokens** is hidden from *every* query of that pass: per row, each item
(``unit = "item"``) or each title token (``unit = "token"``) is dropped with
probability ``p``.  Hiding them from all queries (not only the target side)
matters: otherwise the separators, the ``### Response:`` scaffold and the kept
titles would still encode the dropped items in their deeper-layer keys/values
(they attend causally to the whole history), and the decoder could read them
there.  The fixed template, headers and item separators stay visible; the pooling
pass that computes ``r`` is unchanged (it sees the whole history), so ``r`` is the
only route to a dropped item; position ids are left unchanged.

Unlike the L7 hard bottleneck (everything hidden) and Attempt 1's curriculum (a
train-time-only mode switch), the decoder here always sees *some* history, so it
cannot learn a mode-conditional policy keyed on "history visible or not".  At
inference ``eval_p`` is used (default 0 = full history; set to ``p`` to evaluate
in the training regime).  The block is drawn once per row at prefill and reused
by the KV-cache decode steps, so constrained beam search is consistent.

Config: ``config.latent_bottleneck = {"dropout": spec}`` with ``spec`` from
``build_spec`` (header / response / separator ids as history_pooling.py, plus
``p``, ``unit``, ``eval_p``).  No ``visible_prefix_ids`` needed.
"""

from typing import Optional

import torch

from history_pooling import build_spec as _build_pool_spec, _find


def build_spec(tokenizer, category_text, p, unit="item", eval_p=0.0):
    if unit not in ("item", "token"):
        raise ValueError(f"history dropout unit must be item or token, got {unit}")
    if not 0.0 <= float(p) <= 1.0 or not 0.0 <= float(eval_p) <= 1.0:
        raise ValueError("history dropout probabilities must be in [0, 1]")
    spec = _build_pool_spec(tokenizer, category_text)
    spec.update(p=float(p), unit=unit, eval_p=float(eval_p))
    return spec


def title_items(input_ids: torch.Tensor, attention_mask: torch.Tensor, spec, thought_pos: torch.Tensor):
    """``title`` [B, T] bool (history title tokens) and ``item`` [B, T] long (item
    index of each title token, -1 elsewhere).  Items are delimited by runs of
    separator tokens (quote / comma / space / newline tokens containing a quote).
    Rows whose header or response scaffold cannot be found (left-truncated) get no
    title tokens, i.e. no dropout."""
    batch, length = input_ids.shape
    title = torch.zeros(batch, length, dtype=torch.bool, device=input_ids.device)
    item = torch.full((batch, length), -1, dtype=torch.long, device=input_ids.device)
    sep = torch.tensor(spec["separator_ids"], device=input_ids.device)
    header, response = spec["header_ids"], spec["response_ids"]
    valid = attention_mask.bool()
    for b in range(batch):
        row = input_ids[b]
        t = int(thought_pos[b])
        h = _find(row[:t], header)
        r = _find(row[:t], response, last=True)
        if h < 0 or r < 0 or r <= h + len(header):
            continue
        span = torch.zeros(length, dtype=torch.bool, device=row.device)
        span[h + len(header):r] = True
        is_sep = span & torch.isin(row, sep)
        tokens = span & ~is_sep & valid[b]
        run_start = is_sep & ~torch.cat([is_sep.new_zeros(1), is_sep[:-1]])
        index = run_start.long().cumsum(0) - 1  # the first run is item 0's opening quote
        item[b] = torch.where(tokens, index.clamp(min=0), torch.full_like(index, -1))
        title[b] = tokens
    return title, item


def dropout_block(input_ids: torch.Tensor, attention_mask: torch.Tensor, spec, thought_pos: torch.Tensor,
                  p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Bool [B, T]: history title keys that target-side queries must not see."""
    title, item = title_items(input_ids, attention_mask, spec, thought_pos)
    if p <= 0.0:
        return torch.zeros_like(title)
    if p >= 1.0:
        return title
    device = input_ids.device
    if spec.get("unit", "item") == "token":
        draw = torch.rand(title.shape, generator=generator, device=device) < p
        return title & draw
    n_items = int(item.max().item()) + 1 if title.any() else 0
    if n_items <= 0:
        return torch.zeros_like(title)
    dropped = torch.rand(title.size(0), n_items, generator=generator, device=device) < p
    drop = dropped.gather(1, item.clamp(min=0))
    return title & drop
