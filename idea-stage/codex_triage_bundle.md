# Idea-triage bundle (read fully; instructions at the end)

Context: R3 (LatentR³) generative recommendation. Latent thought r = f(x) (attention-pooled from history x) spliced as a <|Thought|> token into a Qwen2.5-1.5B decoder that generates the next item title. Decoder ignores r's CONTENT (H1 redundancy confirmed: mean/shuffle swaps change HR@10 ≤0.0015 n.s.; a target-category oracle in the slot redirects 89% of top-10; a SASRec vector is used but substitutes for title-reading, 0.1196 < 0.1226). Goal: predictions depend on the sample's own r (mean/shuffle hurt significantly) AND ranking ≥ plain R3 (Musical HR@10 0.1226 / NDCG@10 0.0950). Known frontier point: L7 latent-only bottleneck (history hidden from target tokens, identical train/inference) → strong content dependence (raw 0.1089 / mean 0.0295 / shuffle 0.021) at −11% accuracy. Known failures: anything that only reshapes r (InfoNCE, queues, GRPO); train/inference-mismatched curricula (mode-conditional policy). Compute: 1-GPU Slurm jobs, days-scale training; pilot budget ≤2 GPU-h per idea (eval-only pilots on existing checkpoints preferred). Existing assets: plain-R3 / L7 / SASRec-slot / oracle-slot checkpoints, intervention harness (raw/mask/mean/shuffle), Coconut-style K-thought bottleneck code (fresh, untested at scale), L5 donor retrieval, SASRec vectors, CDs teacher explanations + embeddings (weak target signal), paired-bootstrap eval.

Below is the full deduped, budget-feasible candidate set (union of 5 Claude lens shards + two GPT brainstorms, gpt-6-astra and gpt-6.1-sol; near-duplicates merged — convergent proposals noted). Annotations: prior_work (nearest literature), so_what (why the answer matters either way), effort.

## Candidates

**C1. Recency-split partial bottleneck ("keep recent visible, older history only via r")** [lens:scaling + lens:contradiction + lens:method-transfer(gist) + astra#1 + sol#3 — 5-way convergent]
Target tokens see the instruction, the last m history items, and r pooled over the older prefix; the older items are masked identically in training and KV-cache beam generation (L7's verified masking machinery). Variants: fraction-p mask, last-k items, gist-style pooling of the truncated prefix (Mu et al. gist tokens). Hypothesis: content dependence survives (older info only reachable through r) while recency recovers most of L7's −11%. prior_work: gist tokens; L7 itself; no one has mapped this frontier under a content-dependence audit. so_what: if any m gives Δ(swap) significant and HR@10 ≈0.12, the "trade-off" was an artifact of all-or-nothing masking; if the curve is a step function, redundancy is binary → pivot to exogenous content. effort: days per point, 2–3 points.

**C2. Stochastic per-item Bernoulli bottleneck** [lens:untested-assumption]
Each history item independently hidden from target tokens with prob p, same masking distribution at train and inference (and a secondary full-visibility eval to quantify the mismatch penalty). Decoder can't know which items are hidden → must read r as insurance. prior_work: dropout/scheduled sampling analogues; Attempt-1 failure looms (mode detection). so_what: tests whether uncertainty about the bypass (not its removal) suffices. effort: days; 3 points.

**C3. Latent-capacity sweep under L7: K thought tokens (1→4→16)** [lens:scaling + astra#3 + sol#5 — 3-way convergent]
Reuse fresh K-thought bottleneck code (commit 722e37d); parallel pooling queries vs repeated-token control; whole-block swaps. prior_work: RGLT uses 16 latent tokens; Coconut. so_what: separates "not enough channel capacity" from "wrong content/producer"; directly tests brief direction 1. effort: days per K.

**C4. Latent information-budget sweep (project r to d∈{64,256}; optional VQ)** [lens:scaling]
Companion to C3 at the single-vector level: where do accuracy and dependence saturate in bits? effort: days per point.

**C5. Teacher-distribution distillation into the bottleneck** [astra#4 + sol#6 — convergent]
Train L7-style student to match frozen plain-R3's predictive distribution over candidate titles (cached sparse teacher distributions / candidate sets), not just the single observed next title. Hypothesis: the single target is a noisy compression objective; the teacher's ranking distribution tells r which distinctions to preserve → closes part of the −11% without reopening the bypass. prior_work: knowledge distillation; ReasoningRec distills text explanations; nobody distills the decision distribution into a latent bottleneck. so_what: if it closes ≥half the gap, bottleneck+distillation becomes the method; if not, information loss is intrinsic. effort: ~1 week.

**C6. Beam-search vs scoring audit of L7's deficit** [astra#5]
Teacher-force-score catalog titles under plain R3 and L7 on a subset; compare rank-by-score vs beam output. Is L7's −11% a representation deficit or a search/early-token deficit? Inference-only. so_what: redirects the whole gap-closing effort (decoding fix vs representation fix) for ~0 GPU cost. effort: days.

**C7. Item-code output representation (fixed short codes instead of titles)** [astra#6]
Bijection item↔short common-token code; bottleneck generates codes. Does removing lexical burden disproportionately help the bottlenecked model? prior_work: semantic IDs / TIGER in generative rec (well-trodden as a method, untested as a bottleneck interaction). risk: changes the output space, breaking comparability with all existing numbers. effort: days–weeks.

**C8. Donor-outcome latent: cross-user outcomes as r's designed content** [astra#7 + sol#7 + brief's L5 — convergent]
Retrieve donors (available at query time), encode their observed next-item titles, pool into r; query's own titles stay visible; fine-tune with retrieval in the loop (identical at inference). Audit vs popularity-matched shuffled donor outcomes with retrieval fixed. prior_work: L5 (+0.0023 significant on CDs, inference-only, gate frozen at 0.5 by bf16 bug); IntuRec beam prior. so_what: direct test of H1's prescription (give r what history lacks) with training in the loop and the bf16 bug fixed. effort: 1–2 weeks.

**C9. Privileged explanation distillation (LUPI)** [lens:method-transfer]
Teacher sees target-conditioned explanation (CDs data exists); student r regresses teacher embedding; decoder trained on teacher embeddings in slot, mixed teacher/student ratio fixed train+test. prior_work: generalized distillation (Lopez-Paz); ReasoningRec. named risk: documented weak target signal in those embeddings (median target rank 2,257/5,841). effort: weeks.

**C10. Non-redundant CF signal through an effect-shaped/anchored channel** [lens:contradiction(sasrec-channels) + lens:method-transfer(rglt-effect-credit) + astra#8 + sol#8 + sol#9 — 5-way convergent family]
Family resolving the "used instead-of, not in-addition" SASRec result. Mechanisms: (a) SASRec vector as an ADDITIONAL token next to r; (b) anchored additive score correction S = B(x,c) + (r−μ)ᵀW e(c), exactly zero at r=μ, baseline frozen; (c) SASRec as attention address (biases attention over history items) rather than injected value; (d) RGLT-style effect-credit margin on target-vs-hard-negative likelihood ratios with CF-infused r. prior_work: RGLT effect matching (34.2 vs 30.6 nDCG@10, dense retrieval); Refiner (concurrent, additive score corrections, ID-based). so_what: gap 2+3 combined — effect-based credit applied to provably non-redundant content in LLM generative rec. effort: days (b) to 2 weeks (c,d).

**C11. CFG dual-path contrastive decoding on the latent** [lens:method-transfer]
Always compute logits twice (sample's r vs null latent); train AND infer on the contrast logits_null + w·(logits_r − logits_null). Redundant content cancels structurally; only r's addition gets gradient. prior_work: CFG for LLMs (Sanchez 2023), contrastive decoding; never for latent-conditioner collapse. named risk: if logits_r−logits_null≈0 at init, training may stall in the collapsed optimum. Zero-training CFG probe on existing checkpoints first. effort: days.

**C12. Exogenous-noise channel probe (can NLL ever reward r-content?)** [lens:untested-assumption]
Inject per-row seeded noise z into r; auxiliary fp32 head reconstructs z from the thought-position hidden state. Separates "channel closed" from "content redundant" for the first time. so_what: tells every other method whether objective design or information design is the binding constraint. effort: days.

**C13. Injection-interface comparison (embedding slot vs prefix-KV vs logit bias)** [lens:untested-assumption]
Same r producer, three injection mechanisms, same audits. so_what: if KV/logit-bias make the same r content-used, the interface — not redundancy — was binding, and L7's frontier is a bottleneck artifact. effort: weeks.

**C14. r-gated LoRA adapter (weight-space conditioning)** [lens:untested-assumption]
Rank-8 LoRA on top-4 layers' MLP, gate g=σ(wᵀr): swapped r mis-gates by construction, history fully visible. Control: g fixed to mean at inference. named risk: gate collapses to constant (measurable via control). effort: weeks.

**C15. CEPAE complementarity MI penalty** [lens:method-transfer]
Penalize I(r; x-readable content) (probe reconstructing history-title embeddings) while keeping I(r; target) up via InfoNCE. named risk: still latent-side shaping — the brief's core lesson says the decoder can ignore even complementary content. effort: days.

**C16. R3-vs-SFT gain localization (ablation ladder)** [lens:contradiction]
Matched-recipe runs: SFT no slot / learned constant slot / frozen random slot / plain R3. Where does the +0.02 actually come from? so_what: redefines the baseline every latent-utilization method must beat; explains the rescaled-r 0.1216 result. effort: days (4 short runs).

**C17. Slot-function PCA sweep (how many dof does the decoder read?)** [lens:contradiction]
Inference-only on plain R3: slot ∈ {raw, zero, mean, norm-matched mean, top-k PC projections, matched gaussian, donor}. If k≤2 PCs recover raw performance, r is a 2-dof calibration signal. effort: days, inference only.

**C18. Layer-wise value-flow trace (attention-sink vs content flow)** [lens:untested-assumption + lens:contradiction — merged]
Per-layer value-zeroing, value-swap, and attention-knockout at the thought position; cross-sample variance of W_v·r vs W_v·oracle. Identifies where content dies; yields a per-layer audit metric for all future runs. effort: days, inference only.

**C19. Decoder-scale bypass regime (0.5B vs 1.5B)** [lens:scaling]
Posterior-collapse prediction: weaker decoder leans on r more. effort: days (1 training run).

**C20. LUS field audit: standardized Latent Utilization Score applied to the 2025-26 latent-rec wave** [lens:diagnostic]
LUS = intervention delta normalized by one-history-item-drop delta; apply to plain R3, L7, SASRec/oracle slots + cheap ReaRec-style and Coconut-style reproductions. Is the wave placebo? prior_work: 2512.21711 did this for Coconut in LLM reasoning; nobody for rec. named risk: reproduction fidelity. effort: weeks.

**C21. Oracle information ladder (usability vs redundancy map)** [lens:diagnostic]
Slot ladder: target-category oracle → target-title embedding → SASRec → donor outcomes → redundant controls; measure swap sensitivity vs history-probe R². so_what: ranks candidate contents for the fix; Spearman ρ between non-redundancy and usability tests H1 quantitatively. effort: weeks (needs 2 extra short fine-tunes).

**C22. Per-sample sensitivity atlas (incl. history-length strata)** [lens:diagnostic + lens:scaling — merged]
Per-row sensitivity scores from existing result JSONs regressed on history length, item popularity, category entropy, SASRec confidence. Mostly CPU re-analysis. so_what: "where r matters" structure = free evaluation stratification for every future run; if flat, per-sample gating is dead. effort: days, near-zero GPU.

**C23. Conditional-MI necessity test (I(r;y|x) screening)** [lens:diagnostic]
Probe-based conditional MI for plain-R3 r, L7 r, SASRec, oracle. Training-free screen for any proposed r-producer. effort: days.

**C24. Training-dynamics autopsy (when does sensitivity die?)** [lens:diagnostic]
Intervention battery at every checkpoint of one plain-R3 re-train. Is there an early content-use phase (posterior-collapse dynamics)? effort: days.

**C25. Audit-gaming negative control (identity-check counterexample)** [astra#10 + sol#10 — convergent]
Constructive model that passes raw≥baseline + mean/shuffle-collapse by consistency-checking r against f(x), with zero useful latent prediction. Theory note: predictions off the consistency manifold are underdetermined by training risk. so_what: sharpens the success criterion for the whole line (and the field): swap sensitivity alone is gameable; motivates anchored-improvement endpoints (cf. C10b). effort: days, ~0 GPU.

**C26. Matched-swap granularity audit (what does L7's r encode?)** [astra#9]
Swap latents between users matched on coarse category/semantic neighborhood: does L7 only encode coarse interests? effort: days, inference only.

**C27. Frozen-latent readout probe (is the deficit in r or in generation?)** [sol#2]
Catalog classifier on frozen L7 latents vs plain-R3 latents at matched capacity. Localizes the −11% before vs after the latent interface. effort: days.

**C28. Nine-plus-one complementarity slot** [sol#1]
Keep plain R3's top-9, fill slot 10 with L7's best novel item; audit the added slot. Direct complementarity test from cached rankings. effort: days, ~0 GPU.

**C29. Canonical bag-of-titles: chronology only via r** [astra#2 + sol#4 — convergent]
Decoder sees alphabetically sorted title bag; r pooled from chronological sequence. r's designed job = temporal/order information. Audit incl. chronology-permuted r. named risk: Amazon data may carry little order signal (that negative would itself explain why temporal latents struggle). effort: ~1 week.

## Instructions

For each candidate, make the strongest case both ways:
- What is the best case FOR it — what would make this the paper people cite?
- What's the strongest objection a reviewer would raise?
- What's the most likely failure mode?
- Is the prior_work note a real novelty problem, or differentiable?
- Rank by expected information and upside within the pilot budget (≤2 GPU-h per pilot, ≤8 GPU-h total, eval-only pilots on existing checkpoints preferred) — which results would matter most, whichever way they come out?
- Which 2-3 would you actually work on, and why?

Rank; do not rewrite. An objection is answered or recorded as a named risk on the idea — never absorbed by adding a module, a gate, or a qualifier. A bold idea with a named risk outranks a hedged idea with none, and complexity added since the brainstorm is a red flag, not progress. And do not let your picks be uniformly the safest — if the top set is all LOW-risk, name the high-upside idea that most deserves a pilot slot and what result would convince you.

Note the overall goal constraint when ranking: the line's deliverable is a model whose predictions depend on its own r at ranking ≥ plain R3 — diagnostics are valuable insofar as they de-risk or redirect that goal (or are independently publishable).
