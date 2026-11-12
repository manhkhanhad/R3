# Experiment Plan — Anchored Latent Utilization (claim-driven)

**Grounded in**: `refine-logs/FINAL_PROPOSAL.md` · reviewer package (gpt-6-astra ultra, session `01a182c1-fe9d`) · **Date**: 2026-10-28
**Dataset**: Musical Instruments (dev; pilot decisions on validation split); Toys/CDs for confirmation. **Metrics**: HR and NDCG @1, @3, @5, @10 (and @20 = @10 with 10 beams), paired bootstrap (2,000 resamples, seed 42), user-clustered, several shuffle permutations.
**Budget**: pilots ≤8 GPU-h total (≤2 GPU-h each); exactly two correction training fits; C1 backup = 3 training jobs (~1 GPU-day each) only on the C10b-fail branch. Expenditure caps, not runtime estimates — stop at cap and report unresolved feasibility.
**Process rule**: every run is discussed with the user before implementation/submission (project CLAUDE.md); all outputs under `R3/output_dir/`; frozen code copies; new trainable modules fp32.

## Claims

- **K1 (anchor)**: the sample's CF-infused latent adds held-out ranking value above a frozen no-effect-R3 baseline on the same candidate universe (G1+G2).
- **K2 (attribution)**: the CF infusion is necessary for K1 (G4).
- **K3 (non-substitution)**: the gain is complementary, not a replacement ranker (G3+G5).
- **K4 (promotion)**: HR@10 ≥ 0.1226 vs original plain R3 on the standard protocol.
- **K5 (diagnostics)**: L7's −11% is search-explained or score-residual (C6); L7 holds rescuing items plain R3 misses (C28).

## Runs (in order; each gates the next)

### E0 — Asset & pool-headroom check (CPU/≤1 GPU-h; validation only) → gates everything
Wider beam pool from the frozen no-effect R3 checkpoint; report Recall(pool), uncorrected B's HR/NDCG on the pool, recoverable-miss count. **Kill criterion**: Recall(pool) − HR@10(B) too small to reach 0.1226 → C10b dies here; lead passes to C1. Also verify: frozen SASRec vectors coverage, title embeddings, cached latents.

### E1 — C28 nine-plus-one splice (≈0 GPU; cached rankings + possibly one L7 inference pass) → K5
Plain-R3 top-9 + L7's best novel item; count rescues AND losses separately; raw/mean/repeated-shuffle on the L7 branch; ΔNDCG@10 = ΔHR@10/log₂11. Positive + intervention-sensitive already meets the line's operational goal in hybrid form.

### E2 — C6 beam-vs-catalog-scoring audit (≤2 GPU-h, fixed subset size decided from a throughput probe) → K5
Teacher-forced catalog scoring for plain R3 AND L7, identical conventions (length/EOS/sequence-score). Compare the CHANGES for both models. If L7 recovers to R3 quality: re-run mean/shuffle under that protocol before any claim. Residual deficit = "scoring deficit", never "representation failure" (confounded).

### E3 — Two correction fits (the only training; hours each, 1 GPU) → K1, K2
(a) CF-infused r̃; (b) matched no-CF control (identical pool, B, e(c), W capacity, loss, selection). No hidden sweeps; validation-selected single configuration each.

### E4 — One locked evaluation (≤2 GPU-h) → K1–K4
On the identical pool: raw CF / raw no-CF / mean (identity check S≡B) / repeated shuffles / uncorrected B / original plain R3 / correction-alone / SASRec-direct + scalar fusion. Report all cutoffs with CIs. Apply gates G1–G5, then the K4 promotion gate.

### E5 (branch) — C1 recency-split frontier (3 × ~1 GPU-day) — only if C10b fails informatively or the user chooses to run both
m ∈ {2,5,10} with L7's frozen recipe; exclusive-path verification (older-prefix swap, max |Δlogit|); matched recent-only constant-slot baseline per m; standard + intervention eval at all cutoffs.

## Ablations / checks bound to claims
- Mean-identity check (once, numeric) — audit integrity.
- Correction-alone (G3) — substitution.
- No-CF matched fit (G4) — attribution.
- SASRec-direct + scalar fusion (G5) — conventional-baseline necessity.
- Simplicity defense: the method is one bilinear term; no deletion check needed beyond G3 (the term IS the method).
- Frontier primitive: none required — the LLM is the fixed substrate; stated plainly.

## Decision gates (results → claims matrix, abridged; full matrix in `.aris/traces/research-review/2026-10-28_run01/review_r1.out`)
- E4 passes G1–G5 and K4 → C10b becomes line lead; replicate on Toys + CDs; write toward "ignored latent carries recoverable ranking information" (RecSys/SIGIR core).
- E4 passes audit but < 0.1226 → diagnostic finding retained; C10b not lead; E5 starts.
- E4 fails (three failure modes distinguished: sensitivity w/o accuracy gain / gain w/o sample-specific benefit / CIs too wide) → retire as lead per mode; E5 starts; the C25 counterexample + audit remain the methodological contribution.
- E1 positive regardless → hybrid result stands on its own (operational goal met in hybrid form).

## Replication
Winner (C10b or C1 operating point) replicates on Toys and CDs with per-dataset bootstrap before any paper claim; Musical final numbers labeled exploratory (test-set adaptation risk).
