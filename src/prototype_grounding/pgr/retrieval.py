"""Multi-head query projection + per-head MMR retrieval (design doc: Mechanism steps 1-2).

Design choices this module encodes (see prototype_guided_reasoning.md):
- Query heads are orthogonally initialized, not weight-tied: a single stacked
  weight matrix of shape (n_heads * bank_dim, hidden_size) gets
  ``nn.init.orthogonal_``, then splits into n_heads chunks. Because
  n_heads * bank_dim typically exceeds hidden_size, this is a *semi*-orthogonal
  init (rows orthonormal up to rank hidden_size), which is the standard way to
  approximate mutual head orthogonality when the stacked output dim exceeds
  the input dim -- true full orthogonality isn't achievable in that regime.
- Retrieval is per-head MMR, not a shared pool across heads (Mechanism step 2):
  each head's query runs its own greedy MMR pass so heads actually end up with
  different candidate sets, instead of the best-aligned head's query dominating
  a shared top-k.
- Quality is folded into the MMR relevance term (relevance = sim * quality),
  not used as a separate post-hoc weight -- the softmax-blend design that used
  quality as a fusion weight was replaced by per-prototype concatenation
  (Mechanism step 4); quality now only affects which prototypes get selected.
- MMR here is a plain per-example Python loop (not vectorized across batch).
  With N candidates in the low hundreds and k in the single digits this is
  trivial; if batch sizes grow large enough for this to matter, vectorizing
  the greedy loop across the batch dimension is the first thing to revisit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bank import PrototypeBank


class MultiHeadQueryProjector(nn.Module):
    """Projects r (hidden_size) into n_heads queries in the bank's embedding space."""

    def __init__(self, hidden_size: int, bank_dim: int, n_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.bank_dim = bank_dim
        self.n_heads = n_heads
        self.proj = nn.Linear(hidden_size, n_heads * bank_dim, bias=False)
        with torch.no_grad():
            # nn.init.orthogonal_ does a QR decomposition, unsupported for bf16/fp16 on
            # CPU (and unreliable on GPU) -- construct it in fp32 regardless of the
            # module's actual dtype (from_pretrained(torch_dtype=bf16) constructs new
            # params directly in bf16), then cast down.
            w = torch.empty(self.proj.weight.shape, dtype=torch.float32)
            nn.init.orthogonal_(w)
            self.proj.weight.copy_(w.to(dtype=self.proj.weight.dtype, device=self.proj.weight.device))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r: [..., hidden_size] -> queries: [..., n_heads, bank_dim]"""
        q = self.proj(r)
        return q.view(*q.shape[:-1], self.n_heads, self.bank_dim)


def mmr_select(
    query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_quality: torch.Tensor,
    k: int,
    lam: float = 0.5,
    quality_weighted: bool = True,
) -> torch.Tensor:
    """Greedy MMR selection of k candidate indices for a single query.

    query: [D], candidate_embeddings: [N, D], candidate_quality: [N] -> LongTensor [k]
    (or [min(k, N)] if the candidate pool is smaller than k).

    Selection is index-picking, not a differentiable output -- `query` is detached and
    moved onto whatever device `candidate_embeddings` lives on (the bank stays on CPU
    by convention; the model producing `query` may be on GPU) purely for this
    computation. The caller (PrototypeRetriever.retrieve_one) keeps the original,
    device-resident, grad-attached query tensor for anything downstream that needs it
    (e.g. commitment loss) -- this function never sees or returns that tensor.
    """
    n = candidate_embeddings.shape[0]
    k = min(k, n)
    # bank tensors are always float32 (bank.py); the query may be bf16/fp16 if the
    # model producing it is (e.g. from_pretrained(torch_dtype=bf16)) -- match both
    # device and dtype, not just device, or the matmul below fails.
    query = query.detach().to(device=candidate_embeddings.device, dtype=candidate_embeddings.dtype)
    query_n = F.normalize(query, dim=-1)
    cand_n = F.normalize(candidate_embeddings, dim=-1)
    sim_to_query = cand_n @ query_n  # [N]
    relevance = sim_to_query * candidate_quality if quality_weighted else sim_to_query

    selected: list[int] = []
    remaining = set(range(n))
    # sim between all candidate pairs, computed lazily/cached since N is small
    pairwise_sim = cand_n @ cand_n.T  # [N, N]

    for _ in range(k):
        best_idx, best_score = None, None
        for j in remaining:
            if selected:
                div_penalty = max(pairwise_sim[j, s].item() for s in selected)
            else:
                div_penalty = 0.0
            score = lam * relevance[j].item() - (1 - lam) * div_penalty
            if best_score is None or score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx)
        remaining.discard(best_idx)

    return torch.tensor(selected, dtype=torch.long)


def soft_bank_attention(
    query: torch.Tensor, bank_embeddings: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """query: [D], bank_embeddings: [N, D] -> softmax-weighted blend: [D].

    The differentiable half of the Straight-Through Estimator (straight_through_select,
    below) applied to MMR's otherwise non-differentiable selection. Not used for actual
    selection -- mmr_select still does that, via hard argmax/greedy diversity selection,
    completely unchanged. This exists purely to give query_proj (and everything
    upstream: r, the backbone) a gradient path from CE -- something nothing else in
    this retrieval pipeline provides, since MMR's hard argmax blocks it entirely.
    query_proj has never received task-relevant gradient before this; commitment_loss,
    query_diversity_loss, and usage_balance_loss are all proxies about how retrieval
    behaves, never about whether it actually helps predict the right answer.

    Weighted combination uses the RAW bank embeddings (not normalized) so the blend's
    scale matches what a real hard-selected e_j looks like -- normalized similarity is
    only used to compute the softmax weights, consistent with mmr_select's own
    similarity computation.
    """
    device, dtype = query.device, query.dtype
    bank = bank_embeddings.to(device=device, dtype=dtype)
    bank_n = F.normalize(bank, dim=-1)
    query_n = F.normalize(query, dim=-1)
    weights = F.softmax((bank_n @ query_n) / temperature, dim=-1)  # [N]
    return weights @ bank  # [D]


def straight_through_select(e_hard: torch.Tensor, e_soft: torch.Tensor) -> torch.Tensor:
    """Forward value is exactly e_hard (MMR's real, discrete selection -- the model's
    actual behavior is completely unchanged). Backward gradient flows as if e_soft
    (soft_bank_attention's differentiable blend) had been used instead. Standard
    Straight-Through Estimator -- the same trick FACE (and VQ-VAE before it) use for
    their non-differentiable codebook lookup, applied here to MMR's non-differentiable
    top-k selection instead of nearest-codeword lookup. e_hard and e_soft must be
    broadcastable to the same shape.
    """
    return e_hard.detach() + (e_soft - e_soft.detach())


@dataclass
class RetrievalResult:
    indices: torch.Tensor      # [n_heads, k] long, indices into the bank passed to retrieve()
    queries: torch.Tensor      # [n_heads, bank_dim] float, the query that produced each head's picks


class PrototypeRetriever(nn.Module):
    def __init__(self, hidden_size: int, bank_dim: int, n_heads: int, k_per_head: int, lam: float = 0.5):
        super().__init__()
        self.query_proj = MultiHeadQueryProjector(hidden_size, bank_dim, n_heads)
        self.n_heads = n_heads
        self.k_per_head = k_per_head
        self.lam = lam

    def retrieve_one(self, r: torch.Tensor, bank: PrototypeBank) -> RetrievalResult:
        """r: [hidden_size] (single instance, no batch dim)."""
        queries = self.query_proj(r)  # [n_heads, bank_dim]
        per_head_indices = []
        for a in range(self.n_heads):
            idx = mmr_select(
                queries[a], bank.embeddings, bank.quality_score,
                k=self.k_per_head, lam=self.lam,
            )
            per_head_indices.append(idx)
        # pad to uniform k_per_head if the bank was smaller than k (rare, but keep shapes regular)
        max_k = max(len(idx) for idx in per_head_indices)
        padded = torch.stack([
            F.pad(idx, (0, max_k - len(idx)), value=-1) for idx in per_head_indices
        ])
        return RetrievalResult(indices=padded, queries=queries)

    def retrieve(self, r: torch.Tensor, bank: PrototypeBank) -> list[RetrievalResult]:
        """r: [B, hidden_size] -> list of B RetrievalResults (one per batch element)."""
        return [self.retrieve_one(r[b], bank) for b in range(r.shape[0])]
