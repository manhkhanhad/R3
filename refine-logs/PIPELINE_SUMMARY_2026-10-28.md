# Pipeline Summary

**Problem**: make R3's decoder depend on the sample's own latent r while ranking ≥ plain R3 (Musical HR@10 0.1226 / NDCG@10 0.0950); mechanism identical in train and inference.
**Final Method Thesis**: an apparently ignored latent contains useful ranking information recoverable by a frozen-baseline-anchored additive readout — latent utilization measured as anchored improvement, not manufacturable swap sensitivity.
**Final Verdict**: READY (conditional lead; promotion gates explicit)
**Date**: 2026-10-28

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution: the anchored complementarity audit (mean intervention returns the frozen baseline by construction) + its demonstration on R3.
- Optional supporting contribution: C25 audit-gaming counterexample; C6/C28 diagnostics; C1 audited frontier as backup.
- Explicitly rejected complexity: decoder adapters, MI penalties, GRPO, multi-variant stacking; one bilinear term only.

## Must-Prove Claims
- K1: sample's CF-infused latent beats frozen baseline AND shuffles on the identical candidate pool.
- K2: CF infusion necessary (matched no-CF control).
- K3: complementary, not substitutive (correction-alone + SASRec-fusion comparators).
- K4 (promotion): HR@10 ≥ 0.1226 vs original plain R3.

## First Runs to Launch (after user agreement — project rule)
1. E0 pool-headroom + asset check (≤1 GPU-h; kill criterion inside).
2. E1 C28 nine-plus-one splice (~0 GPU).
3. E2 C6 beam-vs-catalog audit (≤2 GPU-h).

## Main Risks
- Pool recall ceiling too low → E0 kills C10b cheaply; lead passes to C1.
- SASRec residual signal absent → G4/G5 fail; informative negative.
- Test-set adaptation on Musical → validation-gated decisions; Toys/CDs confirmation.

## Next Action
- Discuss E0–E2 designs with the user, then implement and submit (per project rule); /run-experiment afterwards.
