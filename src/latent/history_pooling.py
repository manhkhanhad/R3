"""History-only pooling for plain R3 (user design 2026-11-04).

Plain R3 pools ``r`` over every prompt token before ``<|Thought|>``; on Toys almost
all pooling mass lands on the fixed template (instruction / headers), i.e. ``r``
collapses to a template vector.  Here the pooling keys are restricted to the
**title tokens of the history items**: the tokens strictly between the fixed
header ``### User Input: \\nThe user has palyed the following <category>s before:``
and the last ``### Response:`` before the thought, minus the item-boundary tokens
(tokens made only of quote / comma / space / newline characters and containing a
quote, e.g. ``' "'``, ``'",'``, ``'"\\n\\n'``; spaces and commas inside titles stay).  A token that mixes a title character with a quote (e.g.
``')",'``) counts as a title token.  A quote-only token inside a title (e.g. an
inch mark) is dropped too; the check script measures how often this rule differs
from an exact character-offset mask.

The first pass (hidden states) and the recommendation pass are unchanged; only the
pooling mask changes, computed from ``input_ids`` so it applies identically in
training, ranking evaluation and the intervention scripts.  The prompt is unchanged.
"""

import torch

SEPARATOR_CHARS = set('" ,\n')


def build_spec(tokenizer, category_text):
    """tokenizer: the HF tokenizer; category_text: dataset.category (e.g. 'toys and games')."""
    def enc(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return [int(i) for i in ids]
    header = enc(f"### User Input: \nThe user has palyed the following {category_text}s before:")
    response = enc("### Response:")
    separators = []
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        if s and set(s) <= SEPARATOR_CHARS and '"' in s:
            separators.append(i)
    return {"header_ids": header, "response_ids": response, "separator_ids": separators}


def _find(row, pattern, last=False):
    n, m = row.numel(), len(pattern)
    if n < m:
        return -1
    pat = torch.tensor(pattern, device=row.device)
    hits = (row.unfold(0, m, 1) == pat).all(1).nonzero().flatten()
    if hits.numel() == 0:
        return -1
    return int(hits[-1] if last else hits[0])


def history_title_mask(input_ids, attention_mask, spec, thought_token_id):
    """Bool [B, T]: True on history title tokens (valid, before the thought)."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    sep = torch.tensor(spec["separator_ids"], device=input_ids.device)
    header, response = spec["header_ids"], spec["response_ids"]
    for b in range(input_ids.size(0)):
        row = input_ids[b]
        thought = (row == thought_token_id).nonzero().flatten()
        if thought.numel() == 0:
            raise ValueError("history pooling: no thought token in the row")
        t = int(thought[0])
        h = _find(row[:t], header)
        r = _find(row[:t], response, last=True)
        if h < 0 or r < 0 or r <= h + len(header):
            raise ValueError(f"history pooling: header/response not found in row {b} (h={h}, r={r})")
        span = torch.zeros_like(row, dtype=torch.bool)
        span[h + len(header):r] = True
        mask[b] = span & ~torch.isin(row, sep)
    mask &= attention_mask.bool()
    if not mask.any(1).all():
        raise ValueError("history pooling: a row has no history title token")
    return mask
