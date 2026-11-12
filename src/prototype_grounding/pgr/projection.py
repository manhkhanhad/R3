"""Per-prototype projection into R3's hidden space + reconstruction loss (Mechanism step 3).

`proj_up` bridges two spaces that share no geometry a priori: the evidence
bank's 768-dim sentence-embedder space (independently trained, unrelated to
R3) and R3's hidden_size-dim space. Left to downstream CE gradient alone this
mapping is weakly and unevenly supervised (frequently-retrieved prototypes get
many updates, rare ones get almost none) -- see "Cross-space projection
collapse" in the design doc. The fix: train proj_up/proj_down' with a
reconstruction loss against the *full* bank, independent of retrieval
frequency, so every prototype gets dense supervision regardless of how often
it's actually picked during E1 training.

quality_score is concatenated onto the embedding as an extra scalar feature
rather than used as a fusion weight (the weighted-sum fusion design was
replaced by per-prototype concatenation) -- the network learns how much to
trust it via gradient instead of a hand-tuned softmax deciding upfront.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bank import PrototypeBank


def _mlp(in_dim: int, out_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    hidden_dim = hidden_dim or max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class PrototypePairProjector(nn.Module):
    """proj_up: (bank_dim + 1) -> hidden_size, proj_down': hidden_size -> (bank_dim + 1).

    proj_down' exists only to supervise proj_up via reconstruction; it is not
    used at inference/retrieval time.
    """

    def __init__(self, bank_dim: int, hidden_size: int, mlp_hidden: int | None = None):
        super().__init__()
        self.bank_dim = bank_dim
        self.hidden_size = hidden_size
        self.proj_up = _mlp(bank_dim + 1, hidden_size, mlp_hidden)
        self.proj_down = _mlp(hidden_size, bank_dim + 1, mlp_hidden)

    @staticmethod
    def _with_quality(embeddings: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """[*, bank_dim], [*] -> [*, bank_dim + 1]"""
        return torch.cat([embeddings, quality_score.unsqueeze(-1)], dim=-1)

    def project(self, embeddings: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """embeddings: [*, bank_dim], quality_score: [*] -> p: [*, hidden_size]"""
        return self.proj_up(self._with_quality(embeddings, quality_score))

    def reconstruct(self, embeddings: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """Round-trip through proj_up -> proj_down for the reconstruction loss."""
        x = self._with_quality(embeddings, quality_score)
        return self.proj_down(self.proj_up(x))


def reconstruction_loss(projector: PrototypePairProjector, bank: PrototypeBank) -> torch.Tensor:
    """L_recons over the *full* bank -- every prototype, not just retrieved ones.

    bank.py always stores float32 on CPU; projector may be on GPU and/or a lower
    precision dtype (e.g. bf16 in Stage 3's joint fine-tune). Match both before
    calling into the projector's Linear layers, or this crashes on device/dtype
    mismatch -- harmless historically since phase2_pretrain.py only ever ran this on
    CPU float32 end to end, but real once this is called from a GPU-resident model.
    """
    device = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    embeddings = bank.embeddings.to(device=device, dtype=dtype)
    quality_score = bank.quality_score.to(device=device, dtype=dtype)
    target = projector._with_quality(embeddings, quality_score)
    recon = projector.reconstruct(embeddings, quality_score)
    return ((recon - target) ** 2).mean()
