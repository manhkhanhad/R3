"""Phase 2 driver: pretrain the mapping alone, backbone untouched (Training curriculum, Phase 2).

Two independent things happen here:

1. proj_up/proj_down' reconstruction pretraining against the full prototype
   bank. Needs no forward pass through R3 at all -- pure autoencoder over
   fixed bank embeddings, so it's cheap and runs on CPU in seconds.

2. A retrieval smoke test against the real bank with a synthetic (random)
   r vector, since no trained checkpoint with the retrieval machinery wired
   in exists yet (the frozen-backbone commitment-loss warm-start described in
   the design doc's Phase 2 needs a real checkpoint + GPU and is left as a
   follow-up script, not run here).

Run: python3 phase2_pretrain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from bank import load_prototype_bank
from projection import PrototypePairProjector, reconstruction_loss
from retrieval import PrototypeRetriever
from losses import commitment_loss

# The augmented bank (augment_bank.py output): all 514 rows, re-embedded uniformly
# with Alibaba-NLP/gte-large-en-v1.5 (1024-dim), tagged verified/unverified by tier.
# Local to this package, so a plain __file__-relative path is fine here (unlike the
# raw RALR bank, this isn't a sibling-project path with worktree-depth issues).
BANK_PATH = Path(__file__).resolve().parent / "data" / (
    "CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.augmented.jsonl"
)
HIDDEN_SIZE = 1536  # Qwen2.5-1.5B-Instruct, confirmed from output_dir/.../config.json
N_HEADS = 3
K_PER_HEAD = 3


def pretrain_projector(
    bank, steps: int = 2000, lr: float = 1e-3, hidden_size: int = HIDDEN_SIZE,
) -> PrototypePairProjector:
    projector = PrototypePairProjector(bank_dim=bank.dim, hidden_size=hidden_size)
    opt = torch.optim.Adam(projector.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = reconstruction_loss(projector, bank)
        loss.backward()
        opt.step()
        if step % (steps // 10) == 0 or step == steps - 1:
            print(f"  step {step:5d}  L_recons = {loss.item():.6f}")
    return projector


def collapse_diagnostic(projector: PrototypePairProjector, bank) -> None:
    """Pairwise cosine similarity of proj_up(bank) -- high mean similarity across
    the whole bank would mean prototypes are landing on near-identical points
    after projection, i.e. the cross-space collapse the design doc flags."""
    with torch.no_grad():
        p = projector.project(bank.embeddings, bank.quality_score)  # [N, hidden_size]
        p_n = torch.nn.functional.normalize(p, dim=-1)
        sim = p_n @ p_n.T
        n = sim.shape[0]
        off_diag = sim[~torch.eye(n, dtype=torch.bool)]
    print(f"  proj_up(bank) pairwise cosine sim: mean={off_diag.mean():.4f} "
          f"std={off_diag.std():.4f} max={off_diag.max():.4f}")
    print(f"  (raw bank embeddings, for comparison):")
    e_n = torch.nn.functional.normalize(bank.embeddings, dim=-1)
    sim_raw = e_n @ e_n.T
    off_diag_raw = sim_raw[~torch.eye(n, dtype=torch.bool)]
    print(f"  raw embedding pairwise cosine sim:  mean={off_diag_raw.mean():.4f} "
          f"std={off_diag_raw.std():.4f} max={off_diag_raw.max():.4f}")


def retrieval_smoke_test(bank) -> None:
    torch.manual_seed(0)
    retriever = PrototypeRetriever(
        hidden_size=HIDDEN_SIZE, bank_dim=bank.dim, n_heads=N_HEADS, k_per_head=K_PER_HEAD, lam=0.5,
    )
    r = torch.randn(HIDDEN_SIZE)  # stand-in for a real r_i -- no checkpoint available yet
    result = retriever.retrieve_one(r, bank)
    for a in range(N_HEADS):
        print(f"  head {a}:")
        for idx in result.indices[a].tolist():
            if idx < 0:
                continue
            print(f"    [{bank.tiers[idx]:>10s} {bank.facets[idx]:>8s} q={bank.quality_score[idx]:.2f}] "
                  f"{bank.rationales[idx][:80]}")

    all_selected = result.indices[result.indices >= 0].tolist()
    print(f"  total selected: {len(all_selected)}, unique: {len(set(all_selected))} "
          f"(some overlap across heads is expected with untrained, random-init query "
          f"projections -- this is a raw baseline, not evidence of collapse either way; "
          f"only meaningful once heads are actually trained, per Head collapse risk)")


def commitment_loss_sanity() -> None:
    query = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)  # fixed bank embedding, no grad
    loss = commitment_loss(query, target)
    loss.backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    print(f"  commitment_loss = {loss.item():.4f}, query.grad nonzero: OK")


if __name__ == "__main__":
    print(f"Loading bank from {BANK_PATH}")
    bank = load_prototype_bank(BANK_PATH)
    n_verified = sum(1 for t in bank.tiers if t == "verified")
    print(f"n prototypes = {len(bank)} ({n_verified} verified, {len(bank) - n_verified} unverified), "
          f"dim = {bank.dim}")

    print("\n=== Phase 2a: proj_up/proj_down' reconstruction pretraining ===")
    projector = pretrain_projector(bank)

    print("\n=== Collapse diagnostic ===")
    collapse_diagnostic(projector, bank)

    print("\n=== Retrieval smoke test (random r, real bank) ===")
    retrieval_smoke_test(bank)

    print("\n=== Commitment loss sanity check ===")
    commitment_loss_sanity()
