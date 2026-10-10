"""MUX-style supervision of the R3 thought vector (document/mux_latent_experiments.md).

After MUX (arXiv 2607.18264): a latent token x is decoded into the vocabulary with
the model's own (tied) unembedding, f(x) = softmax(W x / tau), and pulled toward a
*multiplexed* target, a weighted superposition of the one-hot tokens of a text span:

    mux(span) = sum_j a_j onehot(token_j),   a_j ∝ rho^(j-1)
    L = L_rec + weight * KL(mux(span) || f(r))

Here the span is the target item's category path, root dropped, segments
leaf-first ("Electric Guitar Parts > Guitar & Bass Accessories > ..."), so the most
specific tokens get the largest weights.

Two ways to build the thought vector ``r`` (config ``latent_mux.builder``):

* ``pool``: R3's attention pooling over the prompt's final hidden states (unchanged);
* ``codi``: CODI-style, the final hidden state at the position before <|Thought|>
  through an MLP + LayerNorm.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset

BUILDERS = ("pool", "codi")
ROOT = "Musical Instruments"


class CodiLatentLayer(nn.Module):
    """r = LayerNorm(MLP(h[thought - 1])); same call signature as SelfAttentionLayer."""

    def __init__(self, hidden_size, end_k=-1):
        super().__init__()
        self.end_k = end_k  # unused; kept so LatentModel can set it
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def reset(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                w = torch.empty(layer.weight.shape).normal_(0, 1 / math.sqrt(layer.in_features), generator=g)
                with torch.no_grad():
                    layer.weight.copy_(w.to(layer.weight))
                    layer.bias.zero_()
        with torch.no_grad():
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        batch = hidden_states.shape[0]
        if thought_id_idx is None:
            index = torch.full((batch,), hidden_states.shape[1] - 1, device=hidden_states.device)
        else:
            index = thought_id_idx - 1
            if (index < 0).any():
                raise ValueError("the thought token cannot be the first position")
        h = hidden_states[torch.arange(batch, device=hidden_states.device), index]
        return self.norm(self.mlp(h))


def category_segments(path):
    """Leaf-first segments of a category path, the constant root dropped."""
    parts = [p.strip() for p in path.split(">") if p.strip()]
    if len(parts) > 1 and parts[0] == ROOT:
        parts = parts[1:]
    return parts[::-1]


def mux_table(paths, valid, tokenizer, rho=0.9, max_tokens=32):
    """Per-item mux targets: token ids [n, L] (pad 0) and weights [n, L] (sum 1; 0 = invalid row).

    Repeated tokens within a path have their weights merged.
    """
    n = len(paths)
    ids = torch.zeros(n, max_tokens, dtype=torch.long)
    weights = torch.zeros(n, max_tokens)
    for i, (path, ok) in enumerate(zip(paths, valid)):
        if not ok:
            continue
        tokens = []
        for seg in category_segments(path):
            tokens += tokenizer.encode(" " + seg, add_special_tokens=False)
        merged = {}
        for j, t in enumerate(tokens[:max_tokens]):
            merged[t] = merged.get(t, 0.0) + rho ** j
        total = sum(merged.values())
        for k, (t, w) in enumerate(merged.items()):
            ids[i, k] = t
            weights[i, k] = w / total
    return ids, weights


def mux_kl(log_probs, ids, weights):
    """KL(mux || p) per row from log p over the vocabulary; rows with zero weight give 0."""
    picked = log_probs.gather(-1, ids)
    cross = -(weights * picked).sum(-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    return cross - entropy


class MuxLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_mux", None)
        if not spec:
            raise ValueError("latent_mux config is required")
        self.mux_builder = spec.get("builder", "pool")
        if self.mux_builder not in BUILDERS:
            raise ValueError(f"builder must be one of {BUILDERS}")
        if self.mux_builder == "codi":
            self.attention = CodiLatentLayer(config.hidden_size)
        self.mux_weight = float(spec.get("weight", 0.0))
        self.mux_tau = float(spec.get("tau", 1.0))
        self.register_buffer("mux_ids", torch.empty(0, 0, dtype=torch.long), persistent=False)
        self.register_buffer("mux_weights", torch.empty(0, 0), persistent=False)
        self._latent = None
        self._stats = []
        self._seen = 0

    def set_mux_table(self, ids, weights):
        device = self.model.embed_tokens.weight.device
        self.mux_ids = torch.as_tensor(ids).long().to(device)
        self.mux_weights = torch.as_tensor(weights).float().to(device)

    def reset_latent_builder(self, seed=0):
        if self.mux_builder == "codi":
            self.attention.reset(seed)

    def latent_log_probs(self, latent):
        """log softmax(W r / tau) with the (tied) unembedding."""
        logits = self.lm_head(latent.to(self.lm_head.weight.dtype)).float()
        return F.log_softmax(logits / self.mux_tau, dim=-1)

    def observe_reasoning_latent(self, latent):
        self._latent = latent
        return latent

    def forward(self, mux_item_id=None, **kwargs):
        self._latent = None
        output = super().forward(**kwargs)
        if mux_item_id is None or output.loss is None or self._latent is None or self.mux_weight <= 0:
            return output
        if self.mux_ids.numel() == 0:
            raise ValueError("mux table has not been set")
        items = mux_item_id.long().to(self.mux_ids.device)
        ids, weights = self.mux_ids[items], self.mux_weights[items]
        ok = weights.sum(-1) > 0
        if ok.any():
            kl = mux_kl(self.latent_log_probs(self._latent[ok]), ids[ok], weights[ok]).mean()
        else:
            kl = output.loss.new_zeros(())
        if self.training:
            self._seen += len(items)
            self._log(kl.detach())
        output.loss = output.loss + self.mux_weight * kl * accumulation_share(kwargs)
        return output

    def _log(self, kl, every=200):
        self._stats.append(float(kl))
        if len(self._stats) >= every:
            print(f"[mux] seen={self._seen} kl={sum(self._stats) / len(self._stats):.3f}", flush=True)
            self._stats = []


def accumulation_share(kwargs):
    """This micro-batch's share of the accumulated batch under HF gradient accumulation.

    With ``num_items_in_batch`` the LM loss is summed over micro-batches, so a
    per-micro-batch mean must be weighted by its share of the label tokens.
    """
    total = kwargs.get("num_items_in_batch")
    labels = kwargs.get("labels")
    if total is None or labels is None:
        return 1.0
    tokens = (labels[..., 1:] != -100).sum()
    return tokens.to(torch.float32) / torch.as_tensor(total, device=tokens.device, dtype=torch.float32)


class MuxLatentRDataset(LatentRDataset):
    """Adds ``mux_item_id`` (the target item) for the category target."""

    def pre(self, idx):
        example = super().pre(idx)
        example["mux_item_id"] = int(self.data.iloc[idx]["item_id"])
        return example
