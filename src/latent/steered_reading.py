"""B1: the latent steers which history items the decoder reads (latent-utilization line).

Plain R3's decoder reads the whole history and ignores ``r``.  Here ``r`` (the
K=1 continuous thought, continuous_thoughts.py) chooses which history items the
recommendation pass may read:

  s_i = <W_q r, W_k mean_{t in item i} h_t> / sqrt(d)      (first-pass hidden states)
  keep the top-k items by s (rows with <= k items keep everything)

In the recommendation pass the tokens of dropped items are hidden from every
other position and position ids are recomputed over the visible tokens, which
is equivalent to deleting the items from the prompt (the setting of the step-0
probe, scripts/history_subset_probe.py).  The first pass that computes ``r``
and the scores sees the full history.  The same masks apply in training,
teacher-forced scoring and KV-cache beam search.

Selection is hard, so next-item cross-entropy does not reach the scores; they are
trained by an auxiliary loss toward the oracle items (the k history items whose
Qwen3 title embedding is most similar to the target's), which uses the target in
training only.

Config: ``config.latent_steered_reading = {"k": int, "rule": "learned" | "recent",
"aux_weight": float, "dim": int}``.  ``rule="recent"`` is the fixed-rule baseline.
"""

import ast
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_dataset import LatentRDataset

MAX_ITEMS = 10  # longest history in the Amazon splits


class SteeredReadingDataset(LatentRDataset):
    """LatentRDataset plus, per row, ``item_index`` (history item of every token,
    -1 elsewhere) and ``item_sim`` (cosine of each history item's title embedding
    to the target's, padded to MAX_ITEMS with -1e4)."""

    def __init__(self, *args, item_embeddings: Optional[torch.Tensor] = None, **kwargs):
        self.item_embeddings = item_embeddings
        super().__init__(*args, **kwargs)

    def pre(self, idx):
        out = super().pre(idx)
        row = self.data.iloc[idx]
        titles = ast.literal_eval(row["history_item_title"])
        history = ", ".join('"' + t + '"' for t in titles)
        data_point = {"input": f"The user has palyed the following {self.category}s before: {history}", "output": ""}
        prompt = self.generate_prompt(data_point)
        hf = self.tokenizer.tokenizer
        enc = hf(prompt, add_special_tokens=False, return_offsets_mapping=True)
        prefix = self.instruction_tokens()
        n_prompt = len(enc["input_ids"])
        if out["input_ids"][len(prefix):len(prefix) + n_prompt] != enc["input_ids"]:
            raise ValueError(f"row {idx}: offset tokenization differs from the dataset's (or the row was truncated)")

        # Item i spans its separator: the space before its opening quote for item 0,
        # ", " for later items (the comma usually merges into the previous closing quote's token).
        start = prompt.index(history)
        spans, pos = [], start
        for i, t in enumerate(titles):
            open_q = pos if i == 0 else pos + 2
            close = open_q + len(t) + 2
            spans.append((open_q - 1 if i == 0 else pos, close))
            pos = close
        index = []
        for s, _ in enc["offset_mapping"]:
            index.append(next((i for i, (a, b) in enumerate(spans) if a <= s < b), -1))
        item_index = [-1] * len(prefix) + index
        item_index += [-1] * (len(out["input_ids"]) - len(item_index))
        out["item_index"] = item_index[-self.max_len:]

        sim = [-1e4] * MAX_ITEMS
        if self.item_embeddings is not None:
            ids = ast.literal_eval(row["history_item_id"])
            e = self.item_embeddings
            values = (e[torch.tensor(ids)] @ e[int(row["item_id"])]).tolist()
            sim[:len(values)] = values
        out["item_sim"] = sim
        return out


class SteeredCollator:
    """Wrap DataCollatorForSeq2Seq: pad ``item_index`` like ``input_ids``."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        features = [dict(f) for f in features]
        index = [f.pop("item_index") for f in features]
        sim = [f.pop("item_sim") for f in features]
        batch = self.base(features)
        batch["item_index"] = pad_index(index, batch["input_ids"].size(1), self.base.tokenizer.padding_side)
        batch["item_sim"] = torch.tensor(sim, dtype=torch.float32)
        return batch


def pad_index(index, width, side):
    rows = []
    for x in index:
        pad = [-1] * (width - len(x))
        rows.append(pad + list(x) if side == "left" else list(x) + pad)
    return torch.tensor(rows, dtype=torch.long)


class SteeredReading(nn.Module):
    def __init__(self, hidden_size: int, spec: dict):
        super().__init__()
        self.k = int(spec["k"])
        self.rule = spec.get("rule", "learned")
        self.aux_weight = float(spec.get("aux_weight", 0.0))
        dim = int(spec.get("dim", 256))
        self.query = nn.Linear(hidden_size, dim)
        self.key = nn.Linear(hidden_size, dim)
        self.float()
        # Set per batch: by LatentModel.forward (training) or the eval loop (generation, B rows).
        self.item_index: Optional[torch.Tensor] = None
        self.item_sim: Optional[torch.Tensor] = None
        self.rule_override: Optional[str] = None  # evaluation: "recent" | "all" | "learned"
        self.block: Optional[torch.Tensor] = None
        self.last_selection: Optional[torch.Tensor] = None
        self.last_aux: Optional[torch.Tensor] = None
        self.last_stats: dict = {}

    def batch_tensors(self, rows: int, device):
        """item_index / item_sim for ``rows`` rows (beam search expands B -> B*beams)."""
        index, sim = self.item_index, self.item_sim
        if index is None:
            return None, None
        if index.size(0) != rows:
            reps = rows // index.size(0)
            index = index.repeat_interleave(reps, 0)
            sim = sim.repeat_interleave(reps, 0) if sim is not None else None
        return index.to(device), (sim.to(device) if sim is not None else None)

    def forward(self, r, prefix_hidden, prefix_positions, item_index, item_sim=None):
        """Select items and build ``self.block`` ([B, L] bool over the prompt).

        r: [B, H] latent; prefix_hidden: [B, W, H] first-pass hidden states,
        left-aligned real prompt tokens; prefix_positions: [B, W] original position
        of each aligned slot (-1 = pad); item_index: [B, L] in original positions.
        """
        n = MAX_ITEMS
        aligned = torch.where(prefix_positions >= 0,
                              item_index.gather(1, prefix_positions.clamp(min=0)), torch.full_like(prefix_positions, -1))
        onehot = (aligned.unsqueeze(-1) == torch.arange(n, device=r.device)).float()  # [B, W, N]
        counts = onehot.sum(1)  # [B, N]
        valid = counts > 0
        means = torch.einsum("bwn,bwh->bnh", onehot, prefix_hidden.float()) / counts.clamp(min=1).unsqueeze(-1)
        q = self.query(r.float())
        keys = self.key(means)
        scores = torch.einsum("bd,bnd->bn", q, keys) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~valid, float("-inf"))
        n_valid = valid.sum(1)
        rule = self.rule_override or self.rule
        positions = torch.arange(n, device=r.device).expand_as(valid)
        if rule == "learned":
            top = scores.topk(min(self.k, n), dim=1).indices
            selected = torch.zeros_like(valid).scatter_(1, top, True)
        elif rule == "recent":
            selected = positions >= (n_valid - self.k).unsqueeze(1)
        elif rule == "all":
            selected = valid.clone()
        else:
            raise ValueError(f"unknown selection rule {rule}")
        short = n_valid <= self.k
        selected = torch.where(short.unsqueeze(1), valid, selected & valid)

        recent = positions >= (n_valid - self.k).unsqueeze(1)
        recent = torch.where(short.unsqueeze(1), valid, recent & valid)
        long_rows = ~short
        stats = {"rows_selecting": float(long_rows.float().mean())}
        if long_rows.any():
            stats["same_as_recent"] = float((selected == recent).all(1)[long_rows].float().mean())
        self.last_aux = None
        if item_sim is not None and long_rows.any():
            sim = item_sim.float().masked_fill(~valid, float("-inf"))
            oracle = torch.zeros_like(valid).scatter_(1, sim.topk(min(self.k, n), dim=1).indices, True) & valid
            stats["same_as_oracle"] = float((selected == oracle).all(1)[long_rows].float().mean())
            stats["oracle_recall"] = float(((selected & oracle).sum(1).float() / oracle.sum(1).clamp(min=1))[long_rows].mean())
            logp = torch.log_softmax(scores, dim=1).masked_fill(~valid, 0.0)
            per_row = -(logp * oracle.float()).sum(1) / oracle.sum(1).clamp(min=1)
            self.last_aux = per_row[long_rows].mean()
        self.last_stats = stats
        self.last_selection = selected
        hidden_item = (item_index >= 0) & ~selected.gather(1, item_index.clamp(min=0))
        self.block = hidden_item
        return selected


def steered_mask_and_positions(attention_mask, block, query_len, dtype):
    """4D additive mask ``[B, 1, q, k]`` and position ids ``[B, q]`` for the last
    ``query_len`` positions: causal + padding, and keys in ``block`` (dropped
    history items) hidden from every query except themselves.  Positions count
    visible tokens only, as if the dropped items were deleted."""
    batch, key_len = attention_mask.shape
    device = attention_mask.device
    if block.size(0) != batch:
        raise ValueError(f"steered block batch {block.size(0)} != mask batch {batch}")
    if block.size(1) < key_len:
        block = torch.cat([block, torch.zeros(batch, key_len - block.size(1), dtype=torch.bool, device=device)], 1)
    elif block.size(1) > key_len:
        block = block[:, :key_len]
    valid = attention_mask.bool()
    q_abs = torch.arange(key_len - query_len, key_len, device=device)
    k_abs = torch.arange(key_len, device=device)
    causal = q_abs.unsqueeze(1) >= k_abs.unsqueeze(0)
    same = (q_abs.unsqueeze(1) == k_abs.unsqueeze(0)).unsqueeze(0)
    allowed = causal.unsqueeze(0) & valid.unsqueeze(1) & ~block.unsqueeze(1)
    allowed = allowed | same
    bias = torch.zeros(batch, query_len, key_len, dtype=dtype, device=device)
    bias = bias.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)
    visible = valid & ~block
    positions = (visible.long().cumsum(dim=1) - 1).clamp(min=0)[:, key_len - query_len:]
    return bias, positions


def load_fp32_steer_params(model, checkpoint: str) -> None:
    """Reload the selector in fp32 (``from_pretrained(torch_dtype=bf16)`` rounds it)."""
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    model.steer.float()
    params = dict(model.steer.named_parameters())
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith("steer."):
                    with torch.no_grad():
                        params[key[len("steer."):]].copy_(f.get_tensor(key).float())
                    loaded += 1
    if files and loaded == 0:
        raise RuntimeError(f"no selector parameters found in {checkpoint}")
