"""Attempt 9: baseline-preserving latent residual correction.

The deployed second pass remains the ordinary R3 decoder ``F(x, r)``.  The
difference is how ``r`` is constructed::

    r_i = mu + z_i

``mu`` is a fixed train-set mean thought from a strong, already-trained
checkpoint.  ``z_i`` is produced by a new encoder from detached history states.
The encoder's output projection is initialized to exactly zero, so a newly
constructed Attempt-9 model is bit-for-bit equivalent at the thought slot to
the preserved ``F(x, mu)`` baseline.  Training freezes the original model and
updates only the residual encoder; consequently it cannot improve its objective
by rewriting the decoder or the direct history path.

The wrapper retains ``base_attention`` in the checkpoint for provenance and
for an optional sample-base mode, but the default/final design uses the fixed
mean.  ``last_mu`` and ``last_z`` are exposed for the existing extraction and
causal-intervention tooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SelfAttentionLayer


class ResidualCorrectionAttentionLayer(nn.Module):
    """Wrap an existing thought generator with a zero-initialized correction."""

    def __init__(
        self,
        base_attention: nn.Module,
        hidden_size: int,
        bottleneck_size: int = 512,
        residual_scale: float = 1.0,
        use_fixed_mu: bool = True,
        orthogonal_to_mu: bool = False,
        end_k: int = -1,
    ) -> None:
        super().__init__()
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive")
        self.base_attention = base_attention
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.residual_scale = residual_scale
        self.use_fixed_mu = use_fixed_mu
        self.orthogonal_to_mu = orthogonal_to_mu

        # A separate pooling path is intentional.  Its input is detached in
        # forward(), so residual learning cannot reshape the backbone history
        # states consumed directly by the second-pass decoder.
        self.residual_pool = SelfAttentionLayer(hidden_size, end_k=end_k)
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.residual_in = nn.Linear(hidden_size, bottleneck_size)
        self.residual_out = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

        self.register_buffer("fixed_mu", torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer("fixed_mu_initialized", torch.tensor(False))
        self.last_mu: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None

    @property
    def end_k(self) -> int:
        return self.residual_pool.end_k

    @end_k.setter
    def end_k(self, value: int) -> None:
        self.residual_pool.end_k = value
        if hasattr(self.base_attention, "end_k"):
            self.base_attention.end_k = value

    def set_fixed_mu(self, value: torch.Tensor) -> None:
        value = value.detach().float().reshape(-1)
        if value.numel() != self.hidden_size:
            raise ValueError(
                f"fixed mu has {value.numel()} elements; expected {self.hidden_size}"
            )
        self.fixed_mu.copy_(value.to(self.fixed_mu.device))
        self.fixed_mu_initialized.fill_(True)

    def residual_parameters(self):
        """Yield exactly the parameters Attempt 9 is allowed to optimize."""
        for module in (
            self.residual_pool,
            self.residual_norm,
            self.residual_in,
            self.residual_out,
        ):
            yield from module.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # The source thought generator is part of the immutable baseline.  This
        # also prevents EMA-like buffers in decomposed source checkpoints from
        # changing if sample-base mode is used.
        self.base_attention.eval()
        return self

    def _base_mu(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        thought_id_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_fixed_mu:
            if not bool(self.fixed_mu_initialized):
                raise RuntimeError(
                    "Attempt-9 fixed_mu is uninitialized; call set_fixed_mu() "
                    "before training or load a complete Attempt-9 checkpoint"
                )
            return self.fixed_mu.unsqueeze(0).expand(hidden_states.size(0), -1)

        # Optional ablation: preserve the source checkpoint's per-example
        # thought rather than replacing it by the train mean.
        with torch.no_grad():
            return self.base_attention(
                hidden_states.detach(), attention_mask, thought_id_idx
            ).float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        thought_id_idx: torch.Tensor,
    ) -> torch.Tensor:
        mu = self._base_mu(hidden_states, attention_mask, thought_id_idx).float()
        pooled = self.residual_pool(
            hidden_states.detach(), attention_mask, thought_id_idx
        )
        z = self.residual_out(
            F.silu(self.residual_in(self.residual_norm(pooled)))
        ).float()
        z = self.residual_scale * z
        if self.orthogonal_to_mu:
            # Attempt 9a: z may not change the coefficient of the shared prior.
            # This supports both the default global mu and the optional
            # per-example base-thought ablation.
            mu_unit = F.normalize(mu, dim=-1, eps=1e-8)
            z = z - (z * mu_unit).sum(dim=-1, keepdim=True) * mu_unit

        self.last_mu = mu
        self.last_z = z
        return mu + z


def wrap_with_residual_correction(
    base_attention: nn.Module,
    hidden_size: int,
    spec: dict,
) -> ResidualCorrectionAttentionLayer:
    """Build the wrapper from the JSON-serializable config specification."""
    return ResidualCorrectionAttentionLayer(
        base_attention=base_attention,
        hidden_size=hidden_size,
        **dict(spec),
    )
