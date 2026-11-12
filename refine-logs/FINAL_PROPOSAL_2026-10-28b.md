# Final Proposal v2 — Recency-Split Frontier (C1 lead) + CFG Contrast Probe (C11)

**Date**: 2026-10-28 (v2; supersedes the C10b-lead v1 after user decision: score-combination ideas are trivial — C10b/C28 demoted) · **Run**: `latent-utilization-2026-10-28`

## Problem Anchor (unchanged, frozen)

Make R3's predictions depend on the sample's own latent r — mean/shuffle interventions must significantly hurt — while ranking ≥ plain R3 (Musical HR@10 0.1226 / NDCG@10 0.0950, test n = 8,313). Mechanism identical in training and inference (incl. KV-cache beam search). Established: r = f(x) is redundant with visible history; latent-side reshaping fails; the all-or-nothing bottleneck (L7) yields real content dependence at −11%.

## Final Method Thesis (one sentence)

The content-dependence/accuracy trade-off in latent-conditioned generative recommendation is an artifact of all-or-nothing masking: splitting the history so recent items stay visible while older history is reachable ONLY through r yields an operating point with significant latent dependence at near-baseline accuracy — and the audited frontier over the split point is the finding, whichever way it goes.

## Method (C1 — lead)

1. **Recency-split mask**: target tokens attend to the instruction prefix, the LAST m history items, r (pooled over the OLDER prefix only), and earlier target tokens. Older items masked identically in training and KV-cache beam generation (L7's verified machinery, commit `1aedd7e`).
2. **Sweep** m ∈ {2, 5, 10} (endpoints m=0 → L7 0.1089/0.0295/0.021 and m=full → plain R3 0.1226/n.s. already measured).
3. **Required controls** (from novelty + ultra review; binding):
   - **Exclusive-information-path verification** per m: swap the older prefix with r fixed and recompute — max |Δlogit| must be ~0 (L7-style check, verified 4.8e-4 there); ALSO verify visible recent-item hidden states are computed without attending to the older prefix (no indirect bypass through the encoder side of the shared backbone).
   - **Matched recent-only baseline** per m: same visible window, constant (learned) slot instead of r — separates "r adds predictive value from the older prefix" from "recent items suffice".
4. **Success criterion** (anchored, per C25): at some TESTED m, raw beats BOTH the matched recent-only baseline AND mean/shuffle interventions (paired user-clustered bootstrap, several permutations), AND passes a prespecified **noninferiority test vs plain R3**: paired HR@10 difference with margin δ_NI = 0.005 absolute (one-sided CI above −δ_NI; margin user-adjustable before any run, frozen thereafter). Nonsignificance alone is not noninferiority. Conclusions are restricted to the tested splits m ∈ {2,5,10} — no step-function or continuous-frontier claims beyond them; the validation-selected m (and C11's w) are frozen before any confirmation evaluation.

## Companion probe (C11 — promotion path to method lead)

Zero-training CFG probe on the L7 checkpoint: decode with logits_null + w·(logits_r − logits_null), null = mean latent, w ∈ {1.5, 3, 5}. **Corrected rationale (v2 review)**: writing logits_r = h + δ and logits_null = h, the guided logits are h + wδ — the shared history term h is NOT cancelled; the contrast *amplifies the latent-dependent component δ by w*. The training proposal is therefore: train with loss on h + wδ with **stop-gradient on the null branch**, so gradients flow only through logits_r and the cheapest way to reduce loss is to make δ informative; this is an amplified-incentive argument, not structural cancellation, and a positive probe does NOT establish resistance to collapse during training or transfer to full-history visibility — those are what E3 itself tests.
P1 measures three SEPARATE quantities, each with its own gate: (i) contrast magnitude (is δ nonzero/non-trivial in norm and direction across samples?), (ii) decoding effect (does any w significantly improve HR@10 over plain L7 decoding, paired CI excluding 0?), (iii) intervention behavior under guidance (raw vs mean/shuffle at each w). **Promotion gate = (ii)**; (i) nonzero with (ii) null is recorded as "contrast exists, decoding doesn't exploit it" — C11 training then remains optional (user decision), since failure to improve decoding does not establish degeneracy. Secondary mitigation if training stalls: warm-start from L7.

## Dominant contribution

The audited dependence–accuracy frontier resolving whether partial masking dissolves the trade-off (empirical), plus — if the C11 probe is positive — a training objective under which latent redundancy structurally cancels (method). Supporting: C6 diagnostic (is L7's −11% search or scoring?), C25 counterexample grounding the anchored success criterion.

## Novelty position (receipts recorded)

C1: 5/10 PROCEED — closest PerSRec (2601.03479; compressed-past/explicit-recent, efficiency-motivated; also Rec2PM, RecMem, EARN). Verifiable delta: the joint latent-intervention + accuracy frontier over the visible window in an LLM recommender, answering an established utilization failure. C11: CFG/contrastive decoding (Sanchez 2023; Li et al. 2023) never applied to latent-conditioner collapse; delta is training-time contrast against the model's own null-latent branch.

## Explicitly rejected complexity

Score-combination/fusion rerankers (C10 family, C28 — user verdict: trivial); decoder adapters, MI penalties, GRPO, multi-variant stacking. If no m works and the C11 probe is degenerate, the informative negative (the frontier IS a step function; contrast IS degenerate) is reported, not patched.

## Remaining risks

- Content dependence may collapse as soon as any informative history is visible (named triage risk). The m-sweep measures this at the tested splits; if all tested m fail, that split-restricted negative is reported as such (no step-function claim over the continuum).
- Recent-only may saturate accuracy (older prefix low-value) → r ignored again; the matched recent-only baseline detects this per m.
- Test-set adaptation on Musical → pilot decisions on validation; Toys/CDs confirmation.
- Process: all implementations/job submissions discussed with the user first.
