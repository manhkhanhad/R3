"""Commitment loss for the query heads (Training curriculum, Phase 2).

MMR's top-k selection is a hard argmax: it blocks gradient, so under
retrieval + per-prototype-projection alone, W_1..W_n (MultiHeadQueryProjector)
never receive any training signal -- CE loss only ever sees which prototypes
got selected via proj_up, never a differentiable path back to *which ones got
picked*. FACE hits the identical problem in its residual-quantization step
(nearest-codeword lookup is also a hard argmax) and fixes it with a
Straight-Through Estimator plus a two-sided commitment loss (their eq. 9):
one term pulls the encoder output toward the codeword it selected, the other
pulls that codeword toward the encoder output -- both sides are learnable in
FACE, since their codebook is C = W_c @ C0 with W_c trainable.

That second side doesn't apply here: our "codebook" is the prototype bank's
embeddings, loaded once from a frozen, external, offline pipeline -- nothing
about it is ever updated, so a term pulling bank embeddings toward the query
would have no gradient to act on and would be dead weight. Only the one-sided
term is used: pull the query toward its top-1 retrieved match.

Whether to actually use this (vs. leaving W_1..W_n at their random orthogonal
init forever) is an open decision -- see the design doc's "Head collapse" risk
and Training curriculum, Phase 2.

CONFIRMED IN PRODUCTION, not just a theoretical risk: inspecting real retrieval
against the trained Stage 3 checkpoint (5 real test users with completely different
listening histories -- prog rock, Christian/inspirational, metal, two different oldies
fans) showed each head retrieving nearly the *identical* prototype set regardless of
which user's r it received. This is specifically input-invariance -- a given head
collapsing to (near-)constant output across different r's -- not heads collapsing onto
*each other* for a single r (the 3 heads did retrieve visibly different themes from one
another, for any given user). commitment_loss alone doesn't guard against this: it
pulls a query toward whatever it *already* retrieves, a self-referential signal with
nothing rewarding the query for actually varying with its input -- a constant query
that always retrieves the same fixed high-quality prototype trivially satisfies it.
query_diversity_loss (below) is the fix: it directly penalizes different users' queries
(same head) converging, rather than a generic inter-head weight-orthogonality penalty,
which wouldn't have addressed this specific failure mode.

query_diversity_loss WORKS AT THE LOSS LEVEL BUT DOESN'T FIX RETRIEVAL IN PRACTICE.
Three real training trials (different weights, step counts, sample sizes, with and
without pre-filtering the bank) all converged cleanly at the loss level (~0.01, down
from ~0.97 collapsed) while retrieval across a realistic sample of 15 real test users
stayed collapsed to 2-4 discrete "attractor" prototypes regardless. Root cause: this
loss only penalizes query-vector similarity *within a random ~32-example mini-batch*.
If the population has already collapsed onto a handful of discrete modes, a random
batch of 32 drawn from that population will almost always already contain examples
from more than one mode by pigeonhole -- satisfying "not everything in this batch is
identical" cheaply, without ever pressuring the model away from population-level mode
collapse. It optimizes a proxy (query-vector spread) that doesn't actually track the
thing that matters (which bank items get selected).

usage_balance_loss (below) is the corrected fix: this failure -- a routing/selection
mechanism collapsing onto a handful of "experts" regardless of input -- is structurally
identical to expert-collapse in Mixture-of-Experts routing, a well-studied problem with
an established solution (Switch Transformer / GShard's load-balancing loss). Applied
here: penalize the *retrieval usage distribution* (which bank items actually get
selected, across a large reference pool) for being non-uniform, rather than penalizing
query-vector similarity within a small batch as an indirect proxy for it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from bank import PrototypeBank
from retrieval import RetrievalResult


def commitment_loss(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """query: [..., D] (from W_a, learnable), target: [..., D] (fixed bank embedding).

    ||normalize(target) - normalize(query)||^2, one-sided since target has no gradient
    to receive. Normalized rather than raw -- checked at real scale (real 1.5B model,
    real bank) and found necessary, not optional: at init, an untrained query's norm
    (derived from r, which carries raw LLM hidden-state magnitude) and the bank
    embedding's norm live on completely different scales, so the raw L2 distance came
    out ~5000 against a CE loss of ~7 -- at any commit_weight that doesn't make this
    term negligible, it would dominate and destabilize training from step one.
    Normalizing also makes this consistent with MMR's own geometry (mmr_select already
    selects by cosine similarity, not raw L2), so the loss reinforces the same notion
    of "closeness" retrieval used to pick the target in the first place. Bounded in
    [0, 4] regardless of either vector's raw scale.
    """
    query_n = F.normalize(query, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (target_n - query_n).pow(2).sum(dim=-1).mean()


def commitment_loss_from_retrieval(
    retrieval_results: list[RetrievalResult], bank: PrototypeBank, device: torch.device,
) -> torch.Tensor:
    """Aggregate commitment_loss across a batch's retrieval results.

    Target per head is the *top-1* retrieved match (indices[:, 0] -- MMR's first pick
    is the one that scored highest on relevance before any diversity penalty applied,
    the closest analog to "the codeword the query actually selected" in FACE's setup).
    """
    queries = torch.stack([r.queries for r in retrieval_results])  # [B, n_heads, bank_dim]
    top1_idx = torch.stack([r.indices[:, 0] for r in retrieval_results])  # [B, n_heads]
    # bank.py always stores float32; queries may be bf16/fp16 if the retriever
    # producing them is -- match both, not just device (see retrieval.py's mmr_select
    # for the same class of fix).
    targets = bank.embeddings[top1_idx].to(device=device, dtype=queries.dtype)  # [B, n_heads, bank_dim]
    return commitment_loss(queries, targets)


def query_diversity_loss(queries: torch.Tensor) -> torch.Tensor:
    """queries: [B, n_heads, D], B >= 2. Penalizes queries from *different* batch
    examples (same head) being too similar to each other -- directly targets the
    input-invariance collapse confirmed in production (see module docstring), where a
    head produces near-constant output regardless of which user's r it receives.

    Mean pairwise cosine similarity across the batch dimension, per head, averaged over
    heads -- excludes the (always 1.0) self-similarity diagonal. Minimizing this pushes
    different users' queries apart; it says nothing about queries staying close to any
    particular target, so it's meant to sit alongside commitment_loss, not replace it.
    Requires B >= 2 (a single-example batch has no "different user" to compare against).
    """
    B, n_heads, D = queries.shape
    if B < 2:
        return queries.new_zeros(())
    q_n = F.normalize(queries, dim=-1)  # [B, n_heads, D]
    sim = torch.einsum("bhd,chd->hbc", q_n, q_n)  # [n_heads, B, B]
    off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=queries.device)
    off_diag = sim[:, off_diag_mask].view(n_heads, B, B - 1)
    return off_diag.mean()


def _usage_balance_loss_one_head(
    queries: torch.Tensor, bank_embeddings: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """queries: [B, D] (one head), bank_embeddings: [N, D] (the full bank, or a large
    reference subset -- not just whatever k got MMR-selected).

    Switch-Transformer/GShard-style load-balancing loss, ported from MoE expert routing
    to prototype retrieval -- the two problems are isomorphic (a query selecting from a
    fixed discrete set of "experts"/prototypes, with the same collapse failure mode).

    P_j = mean *soft* routing probability assigned to bank item j across the batch
          (softmax(sim(query, e_j) / temperature), differentiable -- this is what
          gradient actually pushes on).
    f_j = fraction of the batch whose *hard* top choice is item j (stop-gradient: an
          empirical frequency, not something to differentiate through -- the argmax
          producing it isn't differentiable anyway).
    loss = sum_j f_j * P_j   (the textbook Switch-Transformer form multiplies this by N;
    dropped here so the loss sits in a [1/N, 1] range instead of [1, N] -- checked at
    real scale before, in commitment_loss's docstring: an auxiliary loss on a wildly
    different scale from the others dominates or vanishes depending on its weight in a
    way that's hard to reason about, so every loss in this module is kept roughly
    comparable rather than repeating that mistake)

    Standard MoE analysis: minimized (~1/N) at perfect balance (f_j = P_j = 1/N for
    every j); grows toward 1 as routing collapses onto a single item. Because f is
    detached, gradient flows only through P -- exactly what's needed to push routing
    probability down for items already over-selected (high f) and up for under-used
    ones.
    """
    device, dtype = queries.device, queries.dtype
    bank_n = F.normalize(bank_embeddings.to(device=device, dtype=dtype), dim=-1)
    query_n = F.normalize(queries, dim=-1)
    sim = query_n @ bank_n.T  # [B, N]

    soft_probs = F.softmax(sim / temperature, dim=-1)
    P = soft_probs.mean(dim=0)  # [N], differentiable

    n = bank_embeddings.shape[0]
    hard_idx = sim.argmax(dim=-1)  # [B]
    f = torch.zeros(n, device=device, dtype=P.dtype)
    f.scatter_add_(0, hard_idx, torch.ones_like(hard_idx, dtype=P.dtype))
    f = (f / queries.shape[0]).detach()

    return (f * P).sum()


def usage_balance_loss(
    queries: torch.Tensor, bank_embeddings: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """queries: [B, n_heads, D]. Averages _usage_balance_loss_one_head across heads.

    B should be as large as practical -- this is exactly the axis query_diversity_loss
    was starved on (stuck at the training micro-batch size, ~32). Stage 2's warm-start
    can pass a large batch drawn from its whole r_sample_size pool; Stage 3's trainer is
    limited to the training micro_batch_size unless a separate, larger reference batch
    of r's is threaded in specifically for this loss.
    """
    B, n_heads, D = queries.shape
    losses = [
        _usage_balance_loss_one_head(queries[:, a], bank_embeddings, temperature)
        for a in range(n_heads)
    ]
    return torch.stack(losses).mean()
