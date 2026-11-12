# Final Proposal — Anchored Latent Utilization (C10b lead, C1 backup)

**Date**: 2026-10-28 · **Run**: `latent-utilization-2026-10-28` · **Status**: READY (conditional lead; promotion gates below)

## Problem Anchor (frozen — no later stage may widen it)

In LLM generative recommendation (R3/LatentR³: Qwen2.5-1.5B generates next-item titles from history + a latent thought r), make predictions depend on the sample's own r — mean/shuffle interventions must significantly hurt — while ranking stays at or above plain R3 (Musical Instruments HR@10 0.1226 / NDCG@10 0.0950, test n = 8,313). Any mechanism must be identical in training and inference (including KV-cache beam search). Established: r = f(x) is redundant with the visible history and the decoder bypasses it; latent-side reshaping does not work; the all-or-nothing bottleneck (L7) works at −11% accuracy.

## Final Method Thesis (one sentence)

An apparently ignored latent reliably contains useful ranking information that a simple frozen-baseline-anchored additive readout can recover — making "latent utilization" measurable as *anchored improvement* (beat the frozen baseline AND beat latent interventions on the same candidate universe) instead of manufacturable swap sensitivity.

## Method (C10b — exactly this, nothing more)

1. **Frozen generator**: the no-effect R3 configuration (r rescaled to no effect; HR@10 0.1216) produces a candidate pool via beam search wider than top-10, and frozen scores B(x,c) on that pool. Report the pool's target recall first: max HR@10 = Recall(pool).
2. **Frozen CF-infused producer** (must be specified before fitting — reviewer gap #5): r̃ = g([r; s_u]) where s_u is the user's frozen SASRec vector and g a small fp32 projection; g is trained ONLY inside the correction fit (W and g are the sole trainable parameters; decoder and SASRec frozen). Title encoder e(c): frozen (existing title embeddings). Training loss: softmax cross-entropy over the pool with the target item (rows whose target is absent from the pool are excluded from fitting but INCLUDED in evaluation as unrecoverable misses — prespecified).
3. **Anchored correction**: S(x,c,r̃) = B(x,c) + (r̃−μ)ᵀ W e(c), μ = fixed training-set mean of r̃. At r̃ = μ, S ≡ B (verify the identity once numerically).
4. **Success gates** (complementarity audit; paired bootstrap, user-clustered, several shuffle permutations):
   - G1: raw r̃ beats frozen B on the identical pool (primary metric HR@10; NDCG@10 must not significantly degrade — prespecified).
   - G2: raw r̃ beats shuffled r̃ on the same candidate universe.
   - G3 (non-substitution): combined S beats the fitted correction evaluated alone.
   - G4 (attribution): the CF-infused correction beats a matched no-CF correction (same pool, B, e(c), W capacity, objective, selection).
   - G5 (conventional-fusion comparator): S beats direct SASRec ranking and validation-tuned scalar fusion of SASRec scores with B on the same pool (or that comparator is recorded missing).
5. **Promotion gate to line lead**: final HR@10 ≥ 0.1226 (original plain R3) with G1–G5 passed. Audit passage alone does NOT promote.

## Dominant contribution

The anchored complementarity audit and its demonstration: a construction in which the mean intervention returns the baseline *by construction*, so any raw-over-mean gain is a real improvement — plus the finding (either way) about whether an ignored latent carries recoverable ranking information. Supporting: the audit-gaming counterexample (C25) motivating the criterion; the C6/C28 diagnostics; C1 as the audited-frontier backup.

## Claim boundaries (from novelty check + review; binding for the paper)

- Do not claim "un-gameable"; claim "rules out sensitivity-only success".
- A positive result shows utilization in the corrected ranking pipeline, NOT that the decoder reads latent content.
- Useful dependence ≠ new information: the correction may recover history information the decoder exploits poorly — state which claim the evidence supports.
- Refiner (arXiv 2610.36890) is concurrent; the verifiable delta: reference-centered bilinear correction on a frozen LLM title-generation baseline with the beat-baseline-AND-beat-interventions requirement on identical candidates.

## Explicitly rejected complexity

No new decoder modules, gates, adapters, or losses; no GRPO; no MI penalties; no combination of C10 variants (a/c/d deferred to backlog); no multi-step latent reasoning. The correction is one bilinear term. If the audit fails, the method is retired as lead — not rescued with machinery.

## Backup (pre-committed, not a pivot): C1 recency-split bottleneck

If C10b fails informatively (sensitivity without accuracy gain, or substitution), the lead passes to C1: last m history items visible, older history only via r, masks identical in train and KV-cache inference (L7 machinery); sweep m ∈ {2,5,10}; audited frontier. Required controls: exclusive-information-path verification (older-prefix swap with r fixed, max |Δlogit| check as in L7) and a matched recent-only constant-slot baseline. Novelty 5/10 PROCEED (closest PerSRec 2601.03479, efficiency-motivated — our delta is the audited frontier).

## Remaining risks

- Candidate-pool recall ceiling too low → E0 kills the experiment before any training (named kill criterion).
- SASRec carries no residual signal after title-reading → G4/G5 fail; informative negative (the finding: the latent's recoverable content is history-redundant).
- Test-set adaptation: Musical test has guided many prior decisions → pilot decisions on validation; final numbers labeled exploratory unless an untouched split is used (Toys/CDs replication is the confirmation).
- Project rule: all implementations and job submissions are discussed with the user before code changes.
