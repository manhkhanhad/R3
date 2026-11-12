# Pipeline Summary (v2 — 2026-10-28)

**Problem**: make R3's decoder depend on the sample's own latent r while ranking ≥ plain R3 (Musical HR@10 0.1226 / NDCG@10 0.0950); mechanism identical in train and inference.
**Final Method Thesis**: the content-dependence/accuracy trade-off is an artifact of all-or-nothing masking — a recency-split (recent items visible, older history only via r) yields latent dependence at near-baseline accuracy; the audited frontier over the split point is the finding.
**Final Verdict**: READY (v2 cross-model review round in progress; v1 receipts recorded)
**Date**: 2026-10-28 · v1 (C10b lead) superseded by user decision: score-combination ideas trivial.

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (v2)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (v2)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (v2)
- Review summary: `refine-logs/REVIEW_SUMMARY.md` (v1 rounds) + `.aris/traces/research-review/2026-10-28_run02/` (v2 round)
- Contract: `idea-stage/docs/research_contract.md` (v2)

## Contribution Snapshot
- Dominant: the audited dependence–accuracy frontier (C1); if the P1 probe is positive, the contrast-trained objective (C11) becomes the method contribution.
- Supporting: C6 search-vs-scoring diagnostic; C25 counterexample grounding the anchored success criterion.
- Rejected: score-combination rerankers (C10 family, C28 — user verdict), decoder adapters, MI penalties, GRPO.

## Must-Prove Claims
- K1: some m gives significant swap damage at HR@10 ≈≥ 0.1226 (or the frontier is a step function — also a finding).
- K2: at that m, raw r beats the matched recent-only constant-slot baseline.
- K3: older history reachable only through r (per-m integrity checks).
- K4: L7's CFG contrast direction (non-)degenerate — gate for C11 training.

## First Runs to Launch (after user agreement)
1. P1 — C11 zero-training CFG probe on L7 (≤2 GPU-h).
2. P2 — C6 beam-vs-catalog audit (≤2 GPU-h).
3. E1 — C1 sweep m ∈ {2,5,10} (3 × ~1 GPU-day) + E2 matched baselines.

## Main Risks
- Step-function frontier → informative negative; pivot to designed-content (C8 donor-outcome latent).
- Recent-only saturation → detected by E2 baselines.
- CFG collapse at init → exactly what P1 measures before any training.

## Next Action
- Fold the v2 review round's verdict in; then discuss P1/P2/E1 designs with the user → implement → /run-experiment.
