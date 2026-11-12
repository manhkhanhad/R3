# Experiment Plan v2 — C1 lead + C11 probe (supersedes v1)

**Date**: 2026-10-28 · **Dataset**: Musical (dev, validation-gated); Toys + CDs confirmation. **Metrics**: HR/NDCG @1,3,5,10 (@20 = @10 with 10 beams); paired bootstrap 2,000 resamples seed 42, user-clustered, repeated shuffle permutations.
**Budget**: pilots ≤4 GPU-h (inference only); main sweep 3 training runs (~1 GPU-day each) + 3 matched recent-only baselines (short; same recipe, constant slot); C11 training only on probe-positive branch.
**Process**: every run discussed with the user before implementation/submission; frozen code copies; outputs under `R3/output_dir/`; new trainable modules fp32; log everything in the research log.

## Claims

- **K1 (frontier, restricted to tested splits)**: at some m ∈ {2,5,10}, significant mean/shuffle damage AND noninferiority to plain R3 (paired HR@10, margin δ_NI = 0.005, one-sided CI above −δ_NI) — or all tested m fail, which is the (split-restricted) negative finding. No step-function claims beyond tested points.
- **K2 (r adds value)**: at that m, raw r beats the matched recent-only constant-slot baseline (not just its own interventions).
- **K3 (integrity)**: the older prefix is reachable only through r (exclusive-path checks pass at every m).
- **K4 (C11 viability)**: the L7 contrast direction is (non-)degenerate under CFG decoding — gate for the C11 training method.
- **K5 (diagnostic)**: L7's −11% is search-explained or score-residual (C6).

## Runs (in order)

### P1 — C11 zero-training CFG probe (≤2 GPU-h; L7 checkpoint) → K4
Guided beam decoding, w ∈ {1.5, 3, 5}, null = mean latent. Three separate measurements with separate gates (v2 review): (i) contrast magnitude of δ = logits_r − logits_null; (ii) decoding effect: guided > plain L7, paired CI excluding 0 — **this is the promotion gate**; (iii) raw/mean/shuffle under each w. δ nonzero but no decoding gain = "contrast exists, decoding doesn't exploit it" — C11 training optional (user decision), not closed as degenerate. Validation-selected w frozen before any confirmation evaluation. E3's training uses stop-gradient on the null branch (amplified-incentive design, NOT structural cancellation — rationale corrected in FINAL_PROPOSAL §C11).

### P2 — C6 beam-vs-catalog-scoring audit (≤2 GPU-h) → K5
As v1: teacher-forced catalog scoring for plain R3 AND L7, identical conventions; compare the CHANGES for both models; throughput probe first; inconclusive if subset too small. De-risks C1 (if L7's deficit is mostly search, the frontier's accuracy ceiling rises).

### E1 — C1 sweep: m ∈ {2, 5, 10} (3 × ~1 GPU-day) → K1, K3
L7 recipe frozen; per-m exclusive-path verification (older-prefix swap, max |Δlogit|; recent-window encoder isolation check) BEFORE full eval. Standard + raw/mean/shuffle eval at all cutoffs.

### E2 — Matched recent-only baselines (3 runs, constant learned slot) → K2
Same visible window and recipe per m, with **equal training budget** (identical steps/LR schedule) or demonstrated convergence (validation-loss plateau) — a shorter run does not qualify as a matched control (v2 review, blocking item). Difference raw-vs-baseline with paired bootstrap.

### E3 (branch, only if P1 positive) — C11 training (1–2 runs)
Train on contrasted logits (dual forward pass, ≈2× step cost), identical contrast at inference; optionally warm-start from L7. Full audit battery. If this reaches content dependence at ≥0.1226 with full history visible, C11 becomes the paper's method and C1 the supporting frontier.

### R1 — Replication of the chosen operating point on Toys + CDs.

## Decision gates
- K1 positive at some m AND K2 positive → C1 result stands; write toward "the trade-off is a masking artifact".
- All tested m fail AND P1 shows no exploitable contrast → line pivots to designed-content directions (C8 donor-outcome latent is the ranked next; documented split-restricted negative on bottleneck relaxation).
- P1 positive → E3 regardless of E1 outcome (C11 can win with full visibility).
- Any mask-integrity failure (K3) invalidates that m's run before interpretation — rerun after fix, never reinterpret.

## Rejected from the plan (user decision 2026-10-28)
C10b anchored correction, C28 nine-plus-one splice — score-combination ideas judged trivial; retained in the report's backlog only as diagnostics someone could run, not as contributions.
