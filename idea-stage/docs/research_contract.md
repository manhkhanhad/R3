# Research Contract: Recency-Split Frontier (C1) + CFG Contrast Probe (C11)

> **Focused working document for the ACTIVE idea.** v2, 2026-10-28: user demoted score-combination ideas (C10b, C28) as trivial; lead is now C1 with C11 as promotion path. Session recovery reads this, not the full idea pool.

## Selected Idea

- **Description**: Recency-split partial bottleneck in R3: target tokens see the instruction, the last m history items, and r pooled over the OLDER prefix only; older items are masked identically in training and KV-cache beam generation (L7's verified machinery). Sweep m ∈ {2,5,10} and map the content-dependence vs accuracy frontier. Alongside: a zero-training CFG probe on L7 (decode on logits_null + w·(logits_r − logits_null)); if the contrast is non-degenerate, promote C11 — training on contrasted logits so history-redundant latent content structurally cancels — to method lead.
- **Source**: IDEA_REPORT.md C1 (lead) + C11 (probe/promotion); user decision 2026-10-28.
- **Selection rationale**: C1 was the triage's top training bet (5/10 PROCEED novelty; the audited frontier is the delta over PerSRec-style efficiency work); C11 is the most mechanism-novel candidate with a named collapse risk the probe tests for ~2 GPU-h.

## Core Claims

1. **K1 (frontier)**: some m gives significant mean/shuffle damage at HR@10 ≈≥ 0.1226 — or the frontier is a step function (either outcome is the finding).
2. **K2 (r adds value)**: at that m, raw r beats a matched recent-only constant-slot baseline, not just its own interventions.
3. **K3 (integrity)**: older history is reachable only through r (exclusive-path verification per m).
4. **K4 (C11 gate)**: the L7 contrast direction is non-degenerate under CFG decoding → opens C11 training (content dependence with FULL history visible, no bottleneck tax).
- Standard: anchored success (beat the matched baseline AND interventions) — swap sensitivity alone is gameable (C25 counterexample).

## Method Summary

R3's decoder ignores r = f(x) (redundant with visible history); only removing the bypass (L7) created content dependence, at −11% HR@10. C1 tests whether the trade-off is an artifact of all-or-nothing masking by keeping recency visible and routing older history exclusively through r — mask identical in train and inference (Attempt-1 lesson). C11 attacks the same redundancy from the objective side: both branches (sample's r vs mean latent) are always computed, and the model is trained and evaluated on their contrast, so content the history already explains cancels; named risk is the collapsed solution logits_r ≡ logits_null, which the zero-training probe measures first.

Run order: P1 C11 probe (≤2 GPU-h) → P2 C6 beam-vs-catalog audit (≤2 GPU-h) → E1 C1 sweep m∈{2,5,10} (3 × ~1 GPU-day) with per-m mask-integrity checks → E2 matched recent-only baselines → E3 C11 training iff P1 positive → R1 Toys/CDs replication.

## Experiment Design

- **Datasets**: Musical (dev; validation-gated decisions, test exploratory), Toys + CDs confirmation.
- **Baselines**: plain R3 (0.1226/0.0950), L7 (0.1089 raw / 0.0295 mean / 0.021 shuffle), plain SFT (0.1009/0.0790, recipe caveat), matched recent-only constant-slot per m.
- **Metrics**: HR/NDCG @1,3,5,10 (@20 = @10 with 10 beams); paired bootstrap 2,000/seed 42, user-clustered, repeated shuffle permutations.
- **Key knobs**: m (2/5/10); CFG w (1.5/3/5); L7 recipe otherwise frozen.
- **Compute**: pilots ≤4 GPU-h inference; 3+3 training runs ~1 GPU-day each; Slurm, frozen code copies, outputs under `R3/output_dir/`, fp32 new modules.

## Baselines

| Method | Dataset | Metric | Score | Source |
|--------|---------|--------|-------|--------|
| plain R3 v1 | Musical | HR@10 / NDCG@10 | 0.1226 / 0.0950 | project |
| L7 bottleneck (m=0 endpoint) | Musical | HR@10 raw/mean/shuffle | 0.1089 / 0.0295 / 0.021 | project, commit 1aedd7e |
| plain SFT | Musical | HR@10 / NDCG@10 | 0.1009 / 0.0790 | project (recipe not verified comparable) |

## Current Results

| Method | Dataset | Metric | Score | Notes |
|--------|---------|--------|-------|-------|

## Key Decisions

- Lead by user decision 2026-10-28: C10b/C28 (score combination) trivial → rejected from the paper story; C1+C11 lead.
- Per-item-visible mask is deterministic per row and identical train/inference (Attempt-1 mode-conditioning lesson).
- Mask-integrity failures invalidate the run before interpretation — fix and rerun, never reinterpret.
- Fail branch: frontier a step function AND probe degenerate → pivot to designed-content direction (C8 donor-outcome latent), documented negative.
- Process: discuss every run with the user before code changes/submission; log in `analysis/document/research_notes/Latent utilization/research_log.md`.

## Status

- [x] Idea selected (v2, user decision 2026-10-28; receipts in `.aris/runs/latent-utilization-2026-10-28.json`)
- [ ] v2 plan review round (in progress)
- [ ] P1 C11 probe (awaiting user go-ahead)
- [ ] P2 C6 audit (awaiting user go-ahead)
- [ ] E1 C1 sweep + E2 baselines
- [ ] E3 C11 training (conditional)
- [ ] Replication (Toys, CDs)
