# Novelty dossier — two top ideas (judge each separately)

Setting for both: R3 (LatentR³, arXiv 2505.19092, ICLR'26): a latent thought r = f(x) attention-pooled from a user's interaction history x is spliced as a <|Thought|> token into a Qwen2.5-1.5B decoder that generates the next item title (Amazon datasets, HR/NDCG@k over beam-searched titles). Established: the decoder ignores r's content (mean/shuffle swaps n.s.) because r is redundant with the visible history; an all-or-nothing latent-only bottleneck (L7) forces content dependence at −11% HR@10. Project goal: content dependence AND accuracy ≥ plain R3.

## Idea 1 — C10b: Anchored latent score correction with a complementarity audit

Method: freeze a no-effect generative-rec baseline's candidate scores B(x,c). Fit ONLY an additive correction S(x,c,r) = B(x,c) + (r−μ)ᵀ W e(c): μ = fixed training-set mean latent, e(c) = frozen candidate-title embedding, r CF-infused (SASRec signal in the producer). At r=μ the correction is exactly zero, so the "mean latent" intervention returns the baseline by construction. Success criterion (the complementarity audit): raw r beats the frozen baseline AND beats mean/shuffled r on the same candidate universe — swap sensitivity alone is explicitly rejected as gameable (a model can consistency-check r against f(x) and fail deliberately on mismatch).

Core claims:
1. An additive latent-effect channel anchored to exactly zero at a reference latent, on a FROZEN LLM generative-rec baseline, so any raw-over-mean gain is a real improvement (not manufactured sensitivity).
2. The complementarity audit (beat-baseline AND beat-swaps on one candidate universe) as the success criterion for latent utilization in generative rec.
3. Using a CF-infused latent as designed non-redundant content for that channel.

## Idea 2 — C1: Recency-split partial bottleneck with a content-dependence/accuracy frontier

Method: target tokens attend to the instruction, the LAST m history items, and r pooled over the OLDER prefix; older items are masked from target tokens identically in training and KV-cache beam-search generation (verified masking machinery from L7). Sweep m to map the frontier between content dependence (mean/shuffle swap damage) and ranking accuracy. Purpose is NOT efficiency: it is to find an operating point where the latent is provably load-bearing at near-baseline accuracy.

Core claims:
1. A recency-split attention bottleneck in an LLM recommender where the latent is the ONLY path to older history, identical in train and inference.
2. The audited frontier (swap-damage vs HR/NDCG as a function of m) as the finding — testing whether the content-dependence/accuracy trade-off is an artifact of all-or-nothing masking.

## Papers found in the multi-source search (all arXiv IDs verified to exist)

Closest to C1 (architecture):
- PerSRec (arXiv 2601.03479, Meta): splits long histories into segments, compresses past segments into learnable "personalized expert" tokens, prediction uses compressed tokens + explicit items from the most recent segment. Motivation: efficiency for long sequences (HSTU/HLLM backbones). No content-dependence audit; no claim about latent utilization.
- Rec2PM (arXiv 2602.11605): recurrent "Preference Memory" tokens compress long histories; memory + working-memory attention. Efficiency-motivated.
- RecMem (NeurIPS'26 poster): recurrent forget-gated memory slots; 5.6× fewer decoder tokens, retains 88–97% Recall. Efficiency-motivated.
- EARN (arXiv 2507.00715): register tokens compress user history in first k layers; later layers see only registers. Efficiency/KV-cache-motivated; notes attention sinks.
- Gist tokens (Mu et al. 2023): instruction compression, the paradigm ancestor.

Closest to C10b (mechanism):
- Refiner (arXiv 2610.36890, 2026-10-27, concurrent): residual latent reasoning for ranking in ID-based generative rec — each latent step emits a cumulative additive correction to ranking scores. Not LLM title generation; no frozen-baseline anchor at a reference latent; no complementarity/intervention audit.
- RecRec (arXiv 2607.12945): recursive latent inference with an evidence-anchored correction gate, small ID-based models. Correction-gate flavor, not a frozen-baseline additive latent channel; no audit.
- EGLR (arXiv 2601.13533): entropy-guided latent reasoning in generative re-ranking. Different mechanism (entropy control of latent reasoning in list generation).
- GR2 (arXiv 2602.07774, Meta): generative reasoning reranker, semantic IDs + RL. Different mechanism.
- RGLT (arXiv 2608.14107): effect-matching credit for latent steps in dense retrieval (not rec, not generative titles).

Audit literature (both ideas' evaluation angle):
- Do Latent Tokens Think? (arXiv 2512.21711): causal/adversarial audit of Coconut latents in LLM reasoning — placeholders, minimal steering sensitivity.
- Bypassing the Rationale: Causal Auditing of Implicit Reasoning in LMs (arXiv 2602.03994): patching-based audits with control patches in LLM reasoning.
- No paper found applying intervention audits (mean/shuffle/swap) to latent channels in RECOMMENDATION, and none using an anchored-zero construction to make the audit un-gameable.

## Questions for you
For each idea separately: Is it novel? What is the closest prior work? What is the delta, stated in one sentence a reviewer could verify? Then give per-idea: novelty score /10 (anchor: 5/10 = clear neighbors but defensible delta worth a pilot; 1–3 reserved for results a named published paper already contains), recommendation (PROCEED / PROCEED WITH CAUTION / ABANDON), key differentiator, and the prior work a reviewer would cite against it.

=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
