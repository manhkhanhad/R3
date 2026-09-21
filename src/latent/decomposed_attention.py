"""Explicit prior/residual decomposition of the latent thought: r_i = mu + z_i.

Motivation (analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md,
Attempt 6): every collapsed checkpoint shows r_i ~= mu for all i -- a shared
"recommendation prior" component dominates the vector's norm and direction, and
the per-sample residual z_i is negligible. That single fact explains the whole
E1 signature (`mask` removes mu and hurts; `mean`/`shuffle`/`noise` replace
one near-copy of mu with another and do nothing) and the ~1.0 random-pair
cosine. Under plain CE training the optimizer sets the mu:z ratio, and it sets
it to ~0. This layer takes that choice away, structurally:

  1. z_i cannot carry the shared component: it is centered against an EMA of
     the pooled attention output, so any sample-independent content has only
     one place to live -- the learnable `mu`.
  2. z_i cannot shrink relative to mu: it is unit-normalized and rescaled to
     rho * ||mu||, so the mu:z ratio is a fixed hyperparameter, not something
     the optimizer can drive to zero.

The decomposition lives inside the attention module (not in generate_embs) so
that every consumer that calls `model.attention(...)` directly -- the standard
ranking evaluator and the E1 intervention pipeline -- sees the full r without
modification. Selection is driven by `config.latent_decomposition`, which is
saved into the checkpoint's config.json, so `LatentModel.from_pretrained`
builds the right layer automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SelfAttentionLayer


class DecomposedSelfAttentionLayer(SelfAttentionLayer):
    def __init__(self, hidden_size, rho: float = 1.0, ema_momentum: float = 0.99, end_k=-1):
        super().__init__(hidden_size, end_k=end_k)
        self.rho = rho
        self.ema_momentum = ema_momentum
        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.register_buffer("z_mean", torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer("z_mean_initialized", torch.tensor(False))
        self.last_z_unit = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, thought_id_idx) -> torch.Tensor:
        pooled = super().forward(hidden_states, attention_mask, thought_id_idx).float()  # [B, H]

        if self.training:
            with torch.no_grad():
                batch_mean = pooled.mean(dim=0)
                if not bool(self.z_mean_initialized):
                    self.z_mean.copy_(batch_mean)
                    self.z_mean_initialized.fill_(True)
                else:
                    self.z_mean.mul_(self.ema_momentum).add_(batch_mean, alpha=1.0 - self.ema_momentum)

        z_unit = F.normalize(pooled - self.z_mean, dim=-1, eps=1e-6)
        self.last_z_unit = z_unit
        mu = self.mu.float()
        scale = self.rho * mu.norm().detach()
        return mu.unsqueeze(0) + scale * z_unit


class VAEDecomposedSelfAttentionLayer(SelfAttentionLayer):
    """Attempt 7: VAE-shaped encoder, no sampling.  mu_i, z_i = two heads on the
    pooled attention output; r_i = mu_i + z_i.

    The prior N(0, I) is imposed on z by the training model (aggregate KL over a
    queue of recent z), which pins ||z_i|| ~= sqrt(H).  mu_i's *direction* is
    learned but its norm is fixed to rho_mu * sqrt(H), so the mu:z ratio is a
    hyperparameter -- otherwise CE would grow ||mu|| until z is negligible again.
    mu_ema (stop-gradient running mean of mu_i) is kept for the optional
    consistency term (7b) and for logging.
    """

    def __init__(self, hidden_size, rho_mu: float = 1.0, ema_momentum: float = 0.99, end_k=-1):
        super().__init__(hidden_size, end_k=end_k)
        self.rho_mu = rho_mu
        self.ema_momentum = ema_momentum
        self.mu_head = nn.Linear(hidden_size, hidden_size)
        self.z_head = nn.Linear(hidden_size, hidden_size)
        self.register_buffer("mu_ema", torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer("mu_ema_initialized", torch.tensor(False))
        self.last_mu = None
        self.last_z = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, thought_id_idx) -> torch.Tensor:
        # Keep pooled in its native (possibly bf16) dtype for the nn.Linear heads --
        # F.linear requires matching dtypes (unlike the elementwise ops
        # DecomposedSelfAttentionLayer uses, which auto-promote), and upcasting
        # pooled to float32 first breaks under a bf16-loaded model. Only the
        # *outputs* need float32 precision, for the KL/norm math downstream.
        pooled = super().forward(hidden_states, attention_mask, thought_id_idx)  # [B, H]
        mu_norm = self.rho_mu * (pooled.size(-1) ** 0.5)
        mu = F.normalize(self.mu_head(pooled).float(), dim=-1, eps=1e-6) * mu_norm
        z = self.z_head(pooled).float()

        if self.training:
            with torch.no_grad():
                batch_mu = mu.mean(dim=0)
                if not bool(self.mu_ema_initialized):
                    self.mu_ema.copy_(batch_mu)
                    self.mu_ema_initialized.fill_(True)
                else:
                    self.mu_ema.mul_(self.ema_momentum).add_(batch_mu, alpha=1.0 - self.ema_momentum)

        self.last_mu = mu
        self.last_z = z
        return mu + z


def build_decomposed_attention(hidden_size: int, spec: dict) -> SelfAttentionLayer:
    """Build the attention layer named by a checkpoint's `latent_decomposition` config entry."""
    spec = dict(spec)
    kind = spec.pop("kind", "fixed_scale")
    if kind == "fixed_scale":
        return DecomposedSelfAttentionLayer(hidden_size, **spec)
    if kind == "vae":
        return VAEDecomposedSelfAttentionLayer(hidden_size, **spec)
    raise ValueError(f"unknown latent_decomposition kind {kind!r}")
