"""Multi-aspect latent reasoning for R3 with confidence-gated fusion.

K aspect heads each pool the history into their own latent r_k (one
``SelfAttentionLayer`` per aspect, as in plain R3) and predict an aspect of
the target item from it (the item itself, its category, its semantic-id
groups). Each aspect's confidence is read from its predictive distribution,

    c_k = 1 - H(softmax(logits_k)) / log(C_k)        in [0, 1]

and the fused thought is a confidence-gated mixture

    g = softmax(a * c + b),   r = sum_k g_k r_k,     a = softplus(.) > 0

so an aspect whose head is more certain for this example gets more weight
(a > 0 keeps that monotone). ``fusion="mean"`` (uniform g) is the ablation.
The confidences are detached: the recommendation loss cannot make a head
overconfident to win the gate.

Per-item aspect codes are persistent buffers, so a checkpoint carries
everything needed for beam re-scoring at evaluation time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SelfAttentionLayer


class MultiAspectAttention(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.names = list(spec["names"])
        self.num_classes = [int(c) for c in spec["num_classes"]]
        self.fusion = spec.get("fusion", "confidence")
        k = len(self.names)
        self.poolers = nn.ModuleList(SelfAttentionLayer(hidden_size) for _ in range(k))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, c)) for c in self.num_classes
        )
        # gate: softplus(raw_scale) * confidence + bias
        self.gate_raw_scale = nn.Parameter(torch.tensor(float(spec.get("gate_init_scale", 2.0))))
        self.gate_bias = nn.Parameter(torch.zeros(k))
        self.register_buffer("aspect_codes", torch.full((k, int(spec["n_items"])), -1, dtype=torch.long))
        self.last_logits = None
        self.last_latents = None
        self.last_gate = None
        self.last_confidence = None

    @property
    def end_k(self):
        return self.poolers[0].end_k

    @end_k.setter
    def end_k(self, value):
        for p in self.poolers:
            p.end_k = value

    def gate_scale(self):
        return F.softplus(self.gate_raw_scale)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        latents = torch.stack([p(hidden_states, attention_mask, thought_id_idx) for p in self.poolers], 1)  # [B,K,H]
        logits = [head(latents[:, i].to(head[1].weight.dtype)).float() for i, head in enumerate(self.heads)]
        with torch.no_grad():
            conf = torch.stack([
                1 - (-(F.softmax(l, -1) * F.log_softmax(l, -1)).sum(-1)) / math.log(c)
                for l, c in zip(logits, self.num_classes)
            ], 1)  # [B,K]
        if self.fusion == "mean":
            gate = torch.full_like(conf, 1.0 / len(self.names))
        else:
            gate = F.softmax(self.gate_scale() * conf + self.gate_bias, dim=-1)
        fused = (gate.unsqueeze(-1).to(latents.dtype) * latents).sum(1)
        self.last_logits, self.last_gate, self.last_confidence = logits, gate, conf
        self.last_latents = latents
        return fused


def build_multi_aspect_attention(hidden_size, spec):
    return MultiAspectAttention(hidden_size, spec)
