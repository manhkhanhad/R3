# Idea Discovery Report

**Direction**: R3 (LatentR³) tries to ignore the latent reasoning vector `r` — how do we make the decoder actually use it?
**Date**: 2026-10-28
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Run**: `.aris/runs/latent-utilization-2026-10-28.json`
**Primary context**: `.claude/worktrees/R3LatentUtilization/CLAUDE.md` (research brief, 2026-10-25); `analysis/document/decoder_insensitivity_to_latent.md`

## Executive Summary

> **Revision 2026-10-28 (user decision)**: score-combination ideas (C10b, C28) judged **trivial** and demoted from lead. New lead: **C1 recency-split bottleneck frontier**, with the **C11 zero-training CFG probe** run alongside — a positive probe promotes C11 (training on contrasted logits) to method lead. Revised proposal/plan: `refine-logs/FINAL_PROPOSAL_2026-10-28b.md`, `refine-logs/EXPERIMENT_PLAN_2026-10-28b.md`. The section below records the superseded v1 recommendation for audit continuity.

**Superseded v1 lead (demoted): C10b — anchored latent score correction.** Freeze the no-effect R3 baseline's candidate scores and fit a single bilinear latent correction that is exactly zero at the mean latent, so the mean intervention returns the baseline by construction; success = beat the frozen baseline AND beat shuffles on the identical candidate pool, with promotion to line lead only at HR@10 ≥ 0.1226. Novelty 6/10 PROCEED (closest: concurrent Refiner, arXiv 2610.36890); ultra-effort external review verdict: *proceed with the capped two-run experiment, but do not grant lead status on audit passage alone*. Pre-committed backup: C1 recency-split bottleneck (5/10 PROCEED). First runs (E0 headroom check, E1 nine-plus-one splice, E2 beam-vs-scoring audit, ≤3 GPU-h combined) are **designed and awaiting user agreement** per the project's discuss-before-code rule — no pilots were executed in this run.

## Literature Landscape

**Sources contributed**: arXiv API (helper), WebSearch. Skipped: Zotero/Obsidian MCP (not configured), Gemini (CLI and MCP unavailable). `WARN: local contributed nothing — no PDFs found in papers/, literature/, or a configured paper library. To include yours, add a "## Paper Library" heading to CLAUDE.md followed by the directory path.`

**Verification**: 15/17 candidates `✅ verified (via arxiv)` (`.aris/verify-papers/verified_papers.json`); 2 venue-indexed papers `… VERIFY_PENDING` (title-only: Soft-Thinking superposition ICLR'26 sub; CoderRec AAAI'26).

### Theme A — Latent reasoning for recommendation (the direct competitors)

| Paper | Venue | Method | Key result / claim | Relevance to us | Status |
|---|---|---|---|---|---|
| ReaRec — Think Before Recommend (Tang, Dai et al.) | arXiv 2503.22675 | Feeds the SeqRec encoder's last hidden state back autoregressively for k implicit reasoning steps, with reasoning position embeddings; ERL/PRL training | +30–50% ceiling on 5 datasets, ID-based backbones | The ID-based analogue of R3's latent thought; does not test whether steps are *used* (content dependence) | ✅ arxiv |
| RecRec: Latent Interests Recursive Reasoning | arXiv 2607.12945 | Critiques ReaRec: reasoning tokens share the input space with history; proposes separate reasoning state + dedicated latent-interest space | Separating spaces helps | Supports the view that `r` co-habiting the history's space makes it redundant | ✅ arxiv |
| HiLaR: Hierarchical Latent Reasoning for LLM-based Rec | arXiv 2607.27760 | Coarse-to-fine hierarchy of latent steps; layer-aware RL | Steps contribute unequally | Same family as R3; no content-dependence analysis | ✅ arxiv |
| IntuRec: Intuition-Guided Latent Reasoning | arXiv 2606.27684 | Initializes latent reasoning from top-K beam candidates ("intuition" prior) | Gains on 3 Amazon subsets | Injects *outcome-space* information into the latent — adjacent to our H1 "give r what history lacks" | ✅ arxiv |
| Factorized Latent Reasoning | arXiv 2604.26760 | Splits the latent thought into multiple preference factors attending to different history parts | Single vector can't capture multi-faceted preference | Factorization still only reshapes r = f(x); by our H1 it stays redundant | ✅ arxiv |
| DiffuReason | arXiv 2602.09744 | Diffusion-based generative refinement over latent reasoning; models uncertainty | Flags missing reasoning supervision + deterministic trajectories as the field's weakness | Another "reshape r" method | ✅ arxiv |
| Refiner — Latent Residual Reasoning for Ranking in GR | arXiv 2610.36890 (2026-10-27, **concurrent**) | Residual latent reasoning: cumulative additive corrections to ranking scores; history queries; LoRA PEFT | Improves over one-shot prediction; online A/B | **Effect-shaped** latent steps (each step must change scores) in ID-based GR ranking — closest concurrent work to an effect-based fix; not LLM title generation, not content-dependence-audited | ✅ arxiv |
| ReasoningRec | arXiv 2410.23180 | Larger LLM generates synthetic preference explanations; fine-tune smaller LLM | +12.5% | Text-space reasoning distillation — the explicit-CoT analogue of our privileged-information direction | ✅ arxiv |
| CoderRec (Cross-Scale Collaboration…) | AAAI 2026 | Downstream recommender's representation used as domain-specific latent thought | Implicit inference without CoT annotations | Confirms SASRec-style vectors are viable latent content (our SASRec-slot probe) | … VERIFY_PENDING |

### Theme B — Latent CoT skepticism: the field has confirmed our failure mode

| Paper | Venue | Method | Key result / claim | Relevance | Status |
|---|---|---|---|---|---|
| Coconut — Reasoning in a Continuous Latent Space | arXiv 2412.06769 | The original continuous-thought method (curriculum replacing CoT with latents) | BFS-like latent exploration claim | The paradigm R3 instantiates for rec | ✅ arxiv |
| Do Latent Tokens Think? (Zhang et al.) | arXiv 2512.21711 | Causal + adversarial analysis of Coconut | Latent tokens are near-insensitive to steering; act as uninterpretable placeholders; benchmark gains ride dataset artifacts ("pseudo-reasoning") | **Independent confirmation of R3's intervention findings** (activation dependence without content dependence) | ✅ arxiv |
| The Illusion of Superposition in Latent CoT via Soft Thinking | ICLR 2026 (sub) | Ablates latent tokens in Soft-Thinking/Coconut | Model extracts answers from the question embedding alone: 96.6% accuracy **without any latent tokens** | The bypass-via-visible-input shortcut = exactly our H1 redundancy | … VERIFY_PENDING |
| RGLT — Retrieval Grounding Latent Reasoning | arXiv 2608.14107 | Dense retrieval: names "shortcut reasoning patterns that preserve performance without incremental gains"; fixes with stage-wise CoT reconstruction + **retrieval-effect credit** on incremental gains per latent step | Effect matching 34.2 vs state alignment 30.6 nDCG@10 | The strongest precedent for **effect-based objectives** (brief's direction 3) — in retrieval, not generative rec | ✅ arxiv |

### Theme C — Posterior collapse: the 2016–2020 analogue

| Paper | Venue | Method | Relevance | Status |
|---|---|---|---|---|
| Cyclical Annealing Schedule (Fu et al.) | NAACL 2019 / arXiv 1903.10145 | KL-weight cycling to escape "decoder ignores z" optimum | Classic statement of the failure: strong AR decoder + redundant latent → KL→0. The decoder-side analogue of our problem, with x (history) in the role of the "strong decoder's own context" | ✅ arxiv |
| BN-VAE (Zhu et al.) | ACL 2020 / arXiv 2004.12585 | Batch-norm on posterior params keeps expected KL positive | Constraint-style fix; extends to conditional VAE | ✅ arxiv |

Lesson transferred: posterior-collapse fixes are of two kinds — (i) **weaken the decoder's bypass path** (our L7 bottleneck; costs accuracy) and (ii) **force the latent to carry non-bypassable information** (our H1-exploiting directions). The literature consistently finds (ii) more accuracy-preserving.

### Theme D — Making conditioning inputs matter (causal/MI regularization)

| Paper | Venue | Method | Relevance | Status |
|---|---|---|---|---|
| CEPAE | arXiv 2602.15546 | Conditional AE where the latent duplicates conditioner info → decoder ignores conditioners; fix: entropy penalty lowering MI(latent; conditioners) | Mirror image of our setting: **redundancy between two input channels causes neglect of one**; fix = make channels complementary | ✅ arxiv |
| Adversary-Free Counterfactual Prediction (Tang et al.) | AISTATS 2026 / arXiv 2510.15479 | Variational upper bound minimizing I(Z;T) without adversaries | Technique source for complementarity constraints (minimize I(r; x)-readable content while keeping task-relevant content) | ✅ arxiv |

### Narrative synthesis

The failure R3 hit is now independently documented in three literatures. In latent-CoT for LLMs, 2512.21711 and the Soft-Thinking superposition study show Coconut-style latent tokens are placeholders the decoder can bypass because the visible input already suffices — precisely R3's measured H1 redundancy (`r = f(x)` adds nothing over visible `x`). In VAE text generation this is posterior collapse, and a decade of fixes says constraint tricks that merely reshape the latent (the analogue of Attempts 10/11, L8, L9) do not work when the decoder has a lossless bypass; what works is removing the bypass (R3's L7, with its −11% cost) or making the latent informationally non-redundant. In recommendation specifically, a 2025–2026 wave (ReaRec, RecRec, HiLaR, IntuRec, Factorized, DiffuReason, Refiner, CoderRec) adds latent reasoning steps, but almost none of it audits content dependence; the two exceptions move in our brief's directions — RGLT supervises each latent step by its *incremental retrieval effect* (effect matching beats state alignment 34.2 vs 30.6 nDCG@10), and the concurrent Refiner (2026-10-27) makes each latent step emit an additive score correction.

The structural gaps: (1) nobody in LLM-based *generative* recommendation (title generation) has an objective that makes predictions provably depend on the sample's own latent while preserving accuracy; (2) effect-based credit (RGLT/Refiner) has not been combined with a redundancy diagnosis — they assume steps should matter, we know *why* they don't; (3) privileged information (target-side or cross-user signals the history lacks) as the designed content of `r` is unexplored — IntuRec's beam-candidate prior is the nearest cousin; (4) nobody has tried to close the bottleneck-accuracy gap (our L7) with partial visibility or multi-token latents under a content-dependence audit. Interventional sensitivity (mean/shuffle swaps) as a *first-class training signal* rather than a post-hoc audit appears genuinely open.

## Ranked Ideas

**Generation**: 5 Claude lens shards (method-transfer, contradiction, untested-assumption, scaling-regime, diagnostic; coverage 5/5, no failures) + two independent GPT brainstorms (gpt-6-astra and gpt-6.1-sol, Codex CLI, xhigh; sessions `01a18293-91ae…` / `01a18293-91d4…`). 44 raw candidates → 29 deduped clusters (C1–C29, full annotated set in `idea-stage/codex_triage_bundle.md`). Strong independent convergence: recency-split bottleneck, K-thought capacity, teacher-distribution distillation, donor outcomes as content, anchored CF corrections, and the audit-gaming counterexample were each proposed by 2–5 independent generators.

**Cross-model triage**: gpt-6-astra, xhigh, session `01a182a8-f75d-7783-9c3a-1eac344806c8` (trace: `.aris/traces/idea-creator/2026-10-28_run01/triage_astra2.out`). Feasibility gate dropped nothing; the reviewer eliminated two on objective flaws (see Eliminated Ideas) and allocated pilot slots as below.

### Idea 1 (pilot slot, high-upside — reviewer-protected): C10b — Anchored latent score correction
- **Method**: Freeze the no-effect R3 baseline scores B(x,c). Fit only a correction S(x,c,r) = B(x,c) + (r−μ)ᵀ W e(c), where μ is the fixed training-set mean latent and e(c) a frozen candidate-title embedding; r is CF-infused (SASRec signal in the producer input). At r = μ the correction is exactly zero, so the mean intervention returns the baseline by construction. Train against actual generated beam-candidate pools.
- **Hypothesis**: a sample-specific, CF-informed latent can add held-out ranking value ABOVE a frozen baseline — complementarity, not substitution; the mean-latent anchor makes any raw-over-mean gain a real improvement rather than manufactured sensitivity.
- **Minimum experiment (≤2 GPU-h)**: score extraction from cached beams + candidate embeddings + fitting + intervention eval on Musical. Named feasibility risks: candidate-pool target recall caps HR@10 (reranking the existing top-10 cannot raise HR@10 — report the pool's recall ceiling first); missing cached assets.
- **Novelty**: additive correction itself is weak (Refiner, concurrent, is close); the differentiator is audited complementarity of a latent channel in LLM generative rec. → deep novelty check, Phase 3.
- **Risk**: MEDIUM-HIGH — SASRec may carry no useful residual signal after title reading. Reviewer objection on record: could meet the prediction-level goal while leaving the thought-slot collapse unresolved.
- **Contribution type**: method.
- **Pilot result**: DESIGNED — awaiting user agreement before implementation (project rule: discuss before code changes).

### Idea 2 (pilot slot, highest information): C6 — Beam-search vs scoring audit of L7's deficit
- **Method**: On a query subset, teacher-force-score catalog titles under plain R3 and L7 with identical sequence-score conventions; compare rank-by-score against beam outputs; repeat L7 scoring with mean/shuffled latents. Inference only, no training.
- **Hypothesis**: part of L7's −11% is search/early-token error, not representation loss; if L7 recovers *relative to plain R3* when search error is removed, the gap-closing effort should target decoding, not representation.
- **Minimum experiment (≤2 GPU-h)**: matched catalog scoring on as many queries as fit; if too few to separate the gaps, the pilot is declared inconclusive (reviewer's instruction). A gain for L7 alone is insufficient — plain R3 may gain equally.
- **Novelty**: search-vs-model error decomposition is familiar; the consequence (overturning the bottleneck trade-off) is what matters.
- **Risk**: LOW for an interpretable diagnosis. **Contribution type**: diagnostic.
- **Pilot result**: DESIGNED — awaiting user agreement.

### Idea 3 (pilot slot, ~0 GPU): C28 — Nine-plus-one complementarity
- **Method**: Keep plain R3's top-9; fill position 10 with L7's highest-ranked item absent from them. Evaluate from cached rankings with raw / mean / shuffled L7 latents; baseline branch frozen. Since only position 10 changes, ΔNDCG@10 = ΔHR@10 / log₂(11) (reviewer-derived check).
- **Hypothesis**: L7 holds correct recommendations plain R3 misses; a fixed allocation realizes own-latent-dependent gains without substitution.
- **Minimum experiment**: pure re-analysis of cached per-row results. **Risk**: MEDIUM (models may share errors; intervention effects may be too sparse for significance). **Contribution type**: empirical.
- **Pilot result**: DESIGNED — awaiting user agreement (needs cached L7 + plain-R3 per-row rankings; if absent, one cheap inference pass).

### Next substantial training bets (post-pilot): C1 and C3
- **C1 — Recency-split partial bottleneck** (5-way convergent): last m items visible, older history only via r, identical masks in train and KV-cache beam inference (L7 machinery). The strongest straightforward model paper if an operating point with significant swap-damage and HR@10 ≈ 0.12 exists. Failure mode: restoring enough recency restores the bypass.
- **C3 — K-thought capacity sweep** (3-way convergent, reuses commit 722e37d): K ∈ {1,4,16} under the L7 bottleneck with a repeated-token control; separates channel capacity from producer quality.

### Standing methodological outputs adopted immediately (no GPU)
- **C25 — audit-gaming counterexample** (convergent, both GPT models): swap sensitivity alone is gameable by an identity check; the line's success criterion is therefore *anchored improvement* (raw beats a frozen baseline AND beats mean/shuffle on the same candidate universe), not swap damage alone. Adopted as the evaluation standard for everything above.
- **C22 — per-sample sensitivity atlas** (CPU re-analysis of existing result JSONs) as a cheap stratification add-on.

### Backlog (annotated, not eliminated)
C26 matched-swap granularity, C27 frozen-latent readout, C5 teacher-distribution distillation, C8 donor-outcome latent (upgrade of L5), C29 chronology-via-r, C17 PCA slot-function sweep, C11 CFG probe, C16 R3-vs-SFT localization (important but 4 training runs), C18 value-flow trace, C20 LUS field audit, C21 oracle ladder, C2/C4/C7/C13/C14/C19/C24 (lower pilot value per triage; reasons in the trace).

## Novelty Verification

**Process**: multi-source WebSearch (6 query formulations across both ideas), all cited arXiv IDs pre-verified via `verify_papers.py` (verdict PASS, hallucination rate 0.0); dossier `idea-stage/NOVELTY_DOSSIER.md`; cross-model verdict by **gpt-6-astra** (Codex CLI, xhigh, session `01a182bd-6f9e-7c31-874b-b4575f528a4a`; trace `.aris/traces/novelty-check/2026-10-28_run01/verdict_astra.out`).

### C10b anchored latent score correction — **6/10, PROCEED**
- **Closest prior work**: Refiner (arXiv 2610.36890, concurrent 2026-10-27, cumulative additive latent corrections to ranking scores, ID-based GR) — a race consideration, not a rejection. Also RecRec (2607.12945, evidence-anchored correction gate); audit papers (2512.21711, 2602.03994) against claiming intervention auditing itself as new.
- **Reviewer-verifiable delta**: relative to Refiner, C10b learns a reference-centered bilinear correction on a FROZEN LLM title-generation baseline and requires the raw latent to beat both the unchanged baseline and latent interventions on identical candidates.
- **Claim boundaries imposed by the reviewer**: (1) the construction rules out *sensitivity alone* as success — do not claim "un-gameable"; (2) CF infusion is plausible-useful, not proven non-redundant; (3) a positive result establishes utilization in the corrected ranking pipeline, NOT that the decoder itself reads latent content. Pilot must include a CF-infusion ablation (matched correction without CF signal).

### C1 recency-split partial bottleneck — **5/10, PROCEED**
- **Closest prior work**: PerSRec (arXiv 2601.03479, Meta — compressed past segments + explicit recent segment; efficiency-motivated); Rec2PM (2602.11605), RecMem (NeurIPS'26), EARN (2507.00715); gist tokens (Mu et al. 2023). Architectural core is strongly precedented — the audited frontier is the novelty.
- **Reviewer-verifiable delta**: relative to PerSRec's compressed-past/explicit-recent architecture, C1 sweeps the explicit-history window in R3 and jointly measures latent-intervention damage and accuracy, to determine whether partial masking resolves the established dependence–accuracy trade-off.
- **Required controls**: (1) verify the exclusive information path — visible recent-item hidden states must not encode the older prefix (the L7-style mask-verification test, extended: recompute with older prefix swapped, r fixed); (2) a matched, separately trained recent-only baseline (constant slot) to show the prefix latent adds predictive value, not just sensitivity.

**Both ideas clear the check**; no named paper contains either result. Eliminated at this phase: none.

## External Critical Review

**Reviewer**: gpt-6-astra, reasoning effort **ultra** (Codex CLI, session `01a182c1-fe9d-7800-be4a-d18b304a69a3`; brief `idea-stage/RESEARCH_REVIEW_REQUEST.md`; full trace `.aris/traces/research-review/2026-10-28_run01/review_r1.out`). One round; convergence reached (concrete run-order package + claims matrix delivered).

**Bottom line (verbatim stance)**: *"Proceed with the capped two-run experiment, but do not make C10b the line's lead on audit passage alone."* Promotion to lead requires: (1) meeting plain-R3 quality (0.1226, not just the 0.1216 anchor), (2) demonstrated useful matched-latent dependence, (3) surviving the correction-alone substitution diagnostic and a conventional SASRec-fusion comparator. If those fail, retain the diagnostic finding and retire the method as lead — do not rescue it with added machinery.

### Material gaps identified (all accepted as plan amendments, no modules added)
1. **Audit pass ≠ accuracy goal**: beating the 0.1216 anchor need not reach 0.1226; a wider beam pool changes the baseline ranking, so compare against uncorrected B on the identical pool AND original plain R3; prespecify the primary metric.
2. **Mean = baseline is one test, not two**: at r=μ, S=B exactly — verify the equality once; the substantive evidence is beat-B and beat-shuffle.
3. **Substitution can pass the audit**: a strong correction can dominate B. Cheap diagnostic: evaluate the fitted correction ALONE on the same pool.
4. **Useful dependence ≠ new information**: the correction may recover history information the decoder exploits poorly — valid but different claim. The matched no-CF ablation attributes CF only; nothing here establishes decoder utilization.
5. **Frozen CF-infused producer is underspecified**: if useful CF-infused latents require training another producer, the experiment has changed — specify construction, title encoder, loss, missing-target handling before fitting.
6. **Statistics**: several shuffle permutations; user-clustered paired bootstrap; validation for pilot decisions, untouched confirmation split where possible, else label exploratory.

### Agreed minimum discriminating package (run order)
1. Asset + pool-headroom check (validation only): wider-pool target recall; max HR@10 = Recall(pool); if the ceiling can't meet the goal, the pool kills the experiment.
2. C28 from cached rankings (count rescues AND losses; raw/mean/repeated-shuffle on the L7 branch; ΔNDCG@10 = ΔHR@10/log₂11).
3. Bounded C6 (≤2 GPU-h): compare the CHANGES for both models under catalog scoring; a residual deficit does not isolate representation (training/decoder-use/scoring confounded); if L7 recovers, re-run interventions under that protocol.
4. Two training runs: CF vs matched no-CF correction fits (everything else frozen and matched; no hidden sweep).
5. One locked evaluation: raw CF / raw no-CF / mean / repeated shuffles / same-pool B / original plain R3 / correction-alone / SASRec-direct + scalar-fusion comparator (or record it missing).

### Results-to-claims matrix (abridged; full 8-row matrix in the trace)
- C10b pass + C6 search + C28 yes → C10b gives useful sample-dependent correction; L7's standalone deficit is search-explained under this protocol.
- C10b fail + C6 residual + C28 no → neither combination establishes the benefit; an informative C10b failure retires it as lead (C1 recency-split becomes the lead training bet).
- Three C10b failure modes are distinguished: sensitivity without accuracy gain / accuracy gain without sample-specific benefit / CIs too wide. No row establishes decoder utilization or CF necessity.

### Venue assessment
Audit pass alone is insufficient for a strong methods paper (elementary bilinear residual ranker). RecSys/SIGIR-credible: reproducible gains over original R3 and straightforward fusion baselines across multiple datasets with attributing ablations — "an apparently ignored latent reliably contains useful ranking information recoverable by a simple readout" is the informative finding. NeurIPS-level would need the broader latent-utilization/redundancy/accuracy story across settings.

## Eliminated Ideas

| Idea | Phase eliminated | Reason |
| C10b anchored score correction | Post-review user decision (2026-10-28) | Score-combination mechanism judged trivial by the project owner (consistent with the ultra review's "elementary bilinear residual ranker"); demoted from lead to optional diagnostic. |
| C28 nine-plus-one splice | Post-review user decision (2026-10-28) | Same reason: ensemble/score-combination triviality. |
|---|---|---|
| C23 conditional-MI screening | Phase-4 triage (gpt-6-astra) | Information-theoretically unsound as stated: for a deterministic producer r = f(x), I(r; y \| x) = 0 whether or not the decoder uses r — the screen cannot measure the property the project needs. |
| C9 privileged explanation distillation (LUPI) | Phase-4 triage (gpt-6-astra) | Deployment-validity flaw: a nonzero teacher mixture at inference requires target-conditioned explanations unavailable at deployment; plus documented weak teacher signal (median target rank 2,257/5,841). |
| C15 CEPAE complementarity MI penalty | Phase-4 triage (downweighted to near-reject) | Repeats the already-failed strategy family — reshaping r without changing what the decoder can bypass; retained in backlog only with a named reason required before any run. |
| C12 exogenous-noise channel probe | Phase-4 triage (downweighted) | Independent noise gives target NLL no population-level reason to reward r-content; the oracle result already answers the channel question. |

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (problem anchor, thesis, method spec with gates G1–G5 + promotion gate, claim boundaries, rejected complexity)
- Review summary: `refine-logs/REVIEW_SUMMARY.md` · Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (E0–E5, claims K1–K5, decision gates, replication protocol)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Working contract: `idea-stage/docs/research_contract.md`

## Next Steps

- [ ] Discuss E0 (pool-headroom check), E1 (C28 splice), E2 (C6 audit) designs with the user; on agreement, implement, test, commit, submit (frozen code copy, outputs under `R3/output_dir/`).
- [ ] E3 two correction fits + E4 locked evaluation → apply gates; K4 decides promotion.
- [ ] On the fail branch: E5 C1 recency-split frontier (m ∈ {2,5,10}).
- [ ] Winner replicates on Toys + CDs before any paper claim.
- [ ] Start the new research log at `analysis/document/research_notes/Latent utilization/research_log.md` with this run's status board.
- [ ] Then `/run-experiment` / `/auto-review-loop` as the work matures.

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, review receipts, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->
