# Idea Discovery Report

**Direction**: Read the current collaborative-reasoning idea and develop new ideas in this direction
**Date**: 2026-10-28
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Run id**: `collab-reasoning-develop-2026-10-28` (state in `.aris/runs/`)

## Executive Summary

The collaborative-reasoning line has established that complementary CF information exists (fused-scorer union ceiling +3.9 pt; parallel cluster-path oracle +8.3 pt) but every attempt to *deliver* it into R3's input substitutes for title evidence, and every *selector* tried so far (decoder confidence, cluster predictor, trained verifiers) is confident only where R3 is already right. The best idea from this run is to stop fighting the delivery problem and attack the **selection/aggregation problem over the already-decoded parallel cluster paths**, with three literature-grounded, mutually compatible mechanisms: (1) rank-based MBR/agreement aggregation across paths (training-free; the same move as the +1.2 pt output-level RRF that already works), (2) PMI/noisy-channel correction of the hypothesis-conditioned path likelihood (the principled fix for the measured below-chance self-verification), and (3) a posterior-distilled selector trained on *which evidence makes the decoder succeed* (Hindsight/PDist applied to CF-cluster evidence — an open niche in recsys). Recommended next step: run the CPU-only pilot E0 on the existing 3,000-row decodes, then the staged GPU pilots after go-ahead.

## Literature Landscape

*Folded from the committed survey (c49e2a4): `research_notes/Evidence retrieval for latent recommendation/{complementary_and_new_information, path_selection_and_verifiers, llm_latent_and_retrieval_rec, retriever_training_and_memory}.md` (2026-10-27/28, per-claim verification flags inside). Condensed to what bears on idea generation.*

**1. The oracle–selector gap is the general pattern, and selection is the open problem.**
Large Language Monkeys (arXiv 2407.21787): coverage from repeated sampling rises to 95% while majority-vote/RM selection stays at ~40%; "building robust verifiers remains an open problem". Our 0.1450 (R3 path) vs 0.2275 (oracle path) gap is this exact pattern. No recsys paper reports an oracle-vs-selector gap over generated reasoning paths — a diagnostic opening.

**2. Hypothesis-conditioned likelihood is a documented bad selector; PMI-style corrections are the fix.**
Surface-form competition / domain-conditional PMI (Holtzman et al., EMNLP 2021), MMI reranking, contrastive decoding, calibrate-before-use: raw likelihood prefers a-priori-likely, generic outputs; dividing out the hypothesis-only prior fixes it. Our decoder's top-beam log-prob selects the true-cluster path *below chance* on R3 misses (0.073 vs 0.129) — the strongest local instance of this failure. No rec paper applies PMI/contrastive scoring to select among generated paths.

**3. Aggregate, don't select.**
Self-consistency (ICLR 2023) marginalizes paths by voting; MBR picks the candidate most agreeing with siblings; RouteRec (arXiv 2607.09908) finds learned per-request *selection* between recommenders loses to BM25 while *item-level aggregation* of candidate lists recovers part of the oracle headroom; ComiRec aggregates multi-interest retrievals rather than picking one interest. Locally, probability-space marginalization of paths collapsed to R3 (peaked beams dominate), but **rank-space aggregation was never tried at the path level** — and at the output level, untuned RRF of v1+fused already gives +1.2 pt (p = 2×10⁻⁹).

**4. Train the selector on the decoder's success, not on the label.**
Atlas PDist / RePlug LSR: target distribution q(d) ∝ p_LM(y | x, d), trained with KL into the retriever — "which evidence makes the decoder succeed" differs from "which cluster contains the target". Hindsight (ICLR 2022): a target-conditioned posterior retriever lifts the prior above what marginal-likelihood training ever reaches; with our small fixed corpus (150 clusters) the posterior is computable *exactly*. Learning-to-defer (Mozannar & Sontag, ICML 2020) gives the consistent surrogate for "pick a path or fall back to R3" with offline hit labels. VRec (arXiv 2603.07725) is the closest recsys analogue (group-level verifier steering the latent) and we have already reproduced its ceiling locally: verifiers trained on labels are confident only where the base model is already right. No rec paper trains cluster/evidence selection with a decoder-likelihood posterior — flagged as an open niche in the survey.

**5. What does not work (local + literature agreement).**
Delivering weak CF into the input substitutes rather than adds (local: cf_text swap ratio 4.9 yet McNemar p = 0.48; literature: FreLLM4Rec, CoRA — LLMs attenuate injected CF). Label-trained verifiers/routers saturate where the base is right (local: k=150 and k=20 verifiers; literature: RouteRec selection, SCoRe's reranker sensitivity). Boosting/DRO-style reweighting of the same input adds nothing when all retrievers miss the same rows (survey section (a)).

## Ranked Ideas

> **Revision 2026-10-28 (user feedback: "training-free is trivial").** Idea 1 is now the **trained** part only: the CF cluster is a latent variable, its prior is trained with KL to the exact decoder posterior q(c|x,y) ∝ p_η(c|x)·p_θ(y|x,c), the null option ∅ (R3's own thought) is a learned defer class, and EM training of the decoder is optional. The parts below that need no training, (a) rank fusion and (b) PMI rescoring at inference, are demoted to baselines. Current spec: `refine-logs/FINAL_PROPOSAL.md`.

Generated 8 candidate ideas from the local evidence + landscape; filtered on feasibility (reuse of existing decodes/checkpoints), novelty risk, and compute. Pilots respect PILOT_MAX_HOURS = 2.

### 🏆 Idea 1 (composite lead): Posterior-aware selection and aggregation over parallel cluster-conditioned latent paths

**Hypothesis.** The +8.3 pt oracle headroom over the ~8 decoded cluster paths is recoverable in meaningful part (target ≥ +1.5 pt full-test HR@10 over R3 v1, beating the +1.2 pt output-RRF) by scoring paths with signals the conditioned decoder does not already share, in three stages:

- **(a) Rank-MBR aggregation (training-free).** Fuse the paths' top-10 lists by reciprocal rank, weighted by p(c|x) from the `mix` predictor, with the R3 path as one list. Rank space neutralizes the peaked-beam dominance that made probability-space marginalization collapse to R3; cross-path item agreement is the self-consistency signal (an item found under several hypotheses is likely not an artifact of any one).
- **(b) PMI-corrected path scores.** Replace the anti-informative conditioned likelihood with s(c) = log p(c|x) + log p(items|x,c) − λ·log p(items|c), where p(items|c) comes from a user-ablated decode (or, as a free approximation, within-cluster item popularity). This is the noisy-channel/domain-PMI correction for "every path is confident given its own cluster".
- **(c) Posterior-distilled selector / defer head.** On validation rows, compute the exact decoder posterior q(c) ∝ p_dec(target | x, c) over clusters (teacher-forced, one forward pass per (row, c), subsampled clusters); distill it into the selector with KL (PDist/Hindsight), and train a defer option (Mozannar–Sontag) so the model falls back to the R3 path unless a path clearly wins. Success metric: >22.7% true-path selection on R3 misses (current best), and net full-test gain with the defer gate.

**Why now/here.** Decodes, oracle model, and predictors all exist; (a) is CPU-only on `cluster_paths_decode.py` outputs; the failure it must beat (tuned marginalization = +0, RRF output fusion = +1.2) is precisely measured.
**Novelty claims.** (i) PMI/noisy-channel scoring of latent reasoning paths in recommendation; (ii) decoder-posterior (PDist/Hindsight) supervision for CF-evidence selection; (iii) the oracle-vs-selector gap + below-chance self-verification as a named, measured phenomenon in latent-reasoning recsys. Each flagged as absent in the survey; verified in Phase 3 below.
**Risks.** (a) may inherit RRF's property of helping only when both lists are decent — path lists on misses are mostly wrong (only 24.4% of rows have any target-bearing path); gains cap at roughly that coverage. (b)'s popularity approximation of p(items|c) may be too crude. (c) repeats the verifier failure mode if q(c) on misses is as R3-correlated as labels are — the E2 pilot measures exactly this before any joint training.
**Pilot plan.** E0 (CPU, <1 h): rank-MBR + agreement scoring on the existing 3,000-row decode, tune on rows 0–999, report 1,000–2,999. E1 (CPU first, GPU ≤2 h if promising): PMI with popularity prior, then with a 1,000-row user-ablated decode. E2 (GPU ≤2 h): teacher-forced q(c) on 2,000 validation rows × 24 sampled clusters, measure whether q(c) separates the true path on R3 misses better than 22.7% before training anything.

### Idea 2: Learned per-row defer between R3's list and the fused-CF list (output level) — SAFE BACKUP
Extend the working +1.2 pt RRF by a per-row gate (features: R3 beam margin, fused-scorer margin, history length, agreement between the two lists) trained on validation hit labels with a defer surrogate. RouteRec warns hard selection fails; soft per-row weighting of two *lists* (not eight) with dense labels is the easiest version that could still work. Low novelty (contribution-sized, not paper-sized), near-zero compute, high floor. Pilot: CPU on existing prediction files.

### Idea 3: Posterior-distilled evidence retriever as the training signal for VRec-style guidance
Stage (c) of Idea 1 used upstream: replace the cluster-label CE in the VRec-style verifier with KL to q(c) ∝ p_dec(target|x,c), freezing the decoder (RePlug-style) to avoid the unlearning collapse seen in job 318248. Tests whether the verifier line fails because of the *label*, not the architecture. Kept as a branch of Idea 1(c) rather than a separate pilot.

### Idea 4: Retrieve similar training rows' *successful thought vectors* r as a latent memory (RaSeRec/kNN-LM style)
The survey found no paper retrieving other users' reasoning latents. But all local evidence says slot delivery of weak signals is ignored, and donor-based retrieval substitutes. Paper-only validation; no pilot. **Deprioritized.**

### Idea 5: Constrain decoding to predicted top-K clusters' vocabulary (two-level decode)
Google interest-exploration / ManCAR-style hard constraint. Local evidence against: predicted clusters are right on ~18% of misses at best, and a hard constraint destroys the 76% of rows where R3 is right for non-cluster reasons unless gated — which re-creates the selection problem. **Eliminated on paper.**

### Idea 6: CVaR/DRO or miss-focused reweighting of R3 training
Survey section (a): reweighting the same input has nothing new to fit when all history-only models miss the same rows; the local residual-scorer line already instantiates the negative result. **Eliminated on paper.**

### Idea 7: Diagnostic paper framing (latent-swap probe + oracle–selector gap)
Not a standalone method; folded into Idea 1's analysis section (it is the motivation and the measurement apparatus).

### Idea 8: LLM-embedding user neighbours (LLM-ESR style) as a new complementarity screen source
A different similarity space might surface different users than the history-based donor/kNN sources. Cheap screen-table addition (one new source column), worth doing opportunistically during the next screen rebuild, but the screen's lesson so far (73–82% hit overlap for *every* source) predicts substitution. **Backlog, not a pilot.**

## Novelty Verification

*(Phase 3 — see below; updated after reviewer verdict.)*

## External Critical Review

*(Phase 4 — see below; updated after reviewer verdict.)*

## Eliminated Ideas

| idea | killed at | reason |
|---|---|---|
| 5. cluster-constrained decode | paper filter | predicted cluster right on ~18% of misses; hard constraint re-creates the selection problem it needs |
| 6. CVaR/DRO miss reweighting | paper filter | same-input reweighting; survey (a) + local residual line are the negative result |
| 4. retrieved thought-vector memory | paper filter (deprioritized) | novel per survey, but contradicts every local slot-delivery result; no cheap discriminating pilot |

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [ ] User go-ahead on pilots E0–E2 (per standing instruction: discuss experiment design before code changes)
- [ ] E0 rank-MBR aggregation (CPU, existing decodes)
- [ ] E1 PMI path scoring; E2 exact decoder posterior q(c) measurement
- [ ] Then /experiment-bridge against `idea-stage/docs/research_contract.md`
