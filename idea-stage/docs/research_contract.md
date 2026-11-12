# Research Contract — ACTIVE idea

**Run**: `collab-reasoning-develop-2026-10-28` · **Date**: 2026-10-28

## Selected idea
**Posterior-aware selection and aggregation over parallel cluster-conditioned latent paths** (Idea 1 of `idea-stage/IDEA_REPORT.md`).

**Selection rationale.** It is the only candidate that (i) attacks the measured bottleneck (oracle 0.2275 vs selector ≤ 0.1450 on report rows; likelihood below chance on misses), (ii) reuses all existing infrastructure (decoded paths, oracle checkpoint, predictors), (iii) has a training-free first stage that is directly comparable to two precisely measured baselines (failed marginalization +0; output-RRF +1.2 pt), and (iv) carries three novelty claims the survey and fresh search left standing (PMI path scoring in rec; decoder-posterior supervision of CF-evidence selection; the named oracle–selector gap diagnostic).

## Core claims
1. Rank-space aggregation across cluster paths recovers part of the oracle gap that probability-space marginalization cannot (M1/E0).
2. PMI/noisy-channel correction makes the hypothesis-conditioned likelihood an informative path score on R3 misses (M2/E1).
3. The exact decoder posterior q(c) ∝ p_dec(target|x,c) is a better selector-supervision signal than cluster labels, and a defer head protects R3 hits (M3/E2).
4. Diagnostic: the oracle-vs-selector gap + below-chance conditioned likelihood is a general, nameable failure of latent-reasoning recommenders.

## Minimum convincing evidence
Full-test HR/NDCG @1,3,5,10,20 vs R3 v1 **and** vs output-RRF 0.1346, with gained/lost + exact McNemar; selection accuracy on misses vs 22.7%; risk–coverage of the defer head; ablations per `refine-logs/EXPERIMENT_PLAN.md` Claim 4.

## Pre-registered kill criteria
E0 < +0.5 pt over output-RRF → M1 is a baseline row, not a contribution. E2-diag q(c) ≤ 22.7% on misses → M3 dead; framing shifts to diagnostic + M1/M2.

## Constraints
Pilots need user go-ahead before any code edit (standing instruction). GPU budget ≤ 8 h, partition researchshort. Env: `rrec` conda env, CUDA/12.4.0. Outputs on scratch, results copied back.

## Next step
`/experiment-bridge` against `refine-logs/EXPERIMENT_PLAN.md` once E0/E1a/E2-diag are approved.
