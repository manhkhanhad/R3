# Review brief — C10b "anchored latent score correction" (idea-stage, pre-implementation)

You are reviewing a RESEARCH IDEA with its validation plan, before implementation. Pilots are designed but not yet run. Your job: find where the idea or its evaluation plan is broken, and name the cheapest discriminating next experiment. The core hypothesis is not up for rewriting; objections become named risks or decisive tests, not added modules.

## Problem (established evidence, verified in-project)

R3 (LatentR³, ICLR'26): latent thought r = f(x) attention-pooled from user history x, spliced as a <|Thought|> token into Qwen2.5-1.5B which generates the next item title (Amazon Musical Instruments, test n=8,313; HR@10 0.1226 / NDCG@10 0.0950 plain). The decoder ignores r's content:
- mean/shuffle swaps of r change HR@10 ≤0.0015 (n.s.); zeroing r hurts (activation dependence without content dependence).
- H1 confirmed: r=f(x) is redundant with visible x. A target-category oracle in the slot redirects 89% of top-10. Attention to r is high (3–11×); scale is not the cause.
- SASRec user vector in the slot IS used (swap costs 27% on a subset) but overall HR@10 drops to 0.1196: substitution for title-reading, not addition.
- A decoder-side margin loss bought 9× sensitivity at −4% HR with no information added: sensitivity is manufacturable.
- All latent-side reshaping (InfoNCE, queues, category alignment, GRPO) changed nothing. A latent-only attention bottleneck (L7, history hidden from target tokens identically in train and KV-cache beam inference) produced real content dependence (raw 0.1089 / mean 0.0295 / shuffle 0.021) at −11% accuracy.
- Caveat: plain SFT without latent = 0.1009/0.0790; R3 with r rescaled to no effect = 0.1216 — R3's gain over SFT may not come from latent content at all (recipes not verified comparable).

Goal of the line: predictions depend on the sample's own r (mean/shuffle hurt significantly) AND ranking ≥ plain R3.

## The idea under review — C10b

Freeze the no-effect R3 baseline's candidate scores B(x,c) (the 0.1216 configuration: r rescaled to no effect — a plain-R3-quality generator provably independent of r). Fit ONLY an additive correction:
  S(x,c,r) = B(x,c) + (r−μ)ᵀ W e(c)
μ = fixed training-set mean latent; e(c) = frozen candidate-title embedding; W trained (fp32). r is CF-infused (SASRec signal concatenated into the producer input). Candidates c come from the baseline's beam pool. At r=μ the correction is exactly zero: the "mean" intervention returns the baseline by construction.

Success criterion (complementarity audit, adopted after an audit-gaming counterexample C25 showed swap-sensitivity alone is satisfiable by an identity check): raw r must (a) beat the frozen baseline on held-out data AND (b) beat mean/shuffled r on the same candidate universe, with paired-bootstrap CIs.

Known boundaries (from the novelty check, gpt-6-astra, 6/10 PROCEED): closest work Refiner (arXiv 2610.36890, concurrent, cumulative additive latent corrections, ID-based GR; no frozen-baseline anchor, no audit); don't claim "un-gameable"; CF infusion is not proven non-redundant; a positive result shows utilization in the corrected ranking pipeline, not that the decoder reads latent content. Required ablation: matched correction without CF infusion.

Named feasibility risks: candidate-pool recall ceiling (reranking the baseline's top-10 cannot raise HR@10 — beam pool must be wider and its target-recall reported first); missing cached assets.

## Companion pilots (same budget envelope, ≤8 GPU-h total)
- C28: keep plain R3's top-9, fill slot 10 with L7's best novel item; cached-ranking re-analysis; raw/mean/shuffle on the L7 branch. ΔNDCG@10 = ΔHR@10/log₂(11).
- C6: teacher-forced catalog scoring vs beam output for plain R3 AND L7 under identical score conventions (is L7's −11% representation or search error?).
Backup method (next training bet): C1 recency-split bottleneck — last m items visible, older history only via r, identical masks train/inference; audited frontier over m; novelty 5/10 PROCEED (closest PerSRec 2601.03479, efficiency-motivated); required controls: exclusive-information-path verification; matched recent-only baseline.

## Compute & assets
1-GPU Slurm jobs; existing checkpoints (plain R3, no-effect R3, L7, SASRec-slot, oracle-slot), intervention harness, beam outputs, SASRec vectors, paired-bootstrap eval. Fresh K-thought bottleneck code exists (unused here).

## Questions
1. Logical gaps or unjustified claims in C10b or its success criterion?
2. Missing experiments that would strengthen (or cheaply kill) it — what is the minimum discriminating package, in run order, within ≤8 GPU-h of pilots plus ≤2 training runs?
3. Is the contribution sufficient for a top venue (RecSys/SIGIR/NeurIPS-track) if the audit passes? If not, what result would make it sufficient?
4. Results-to-claims matrix: what claim is allowed under each outcome of (C10b pass/fail × C6 search-vs-representation × C28 complementarity yes/no)?
5. Your bottom line: proceed or abandon this idea as the line's lead?

=== SCOPE LIMITS (these bound what you PROPOSE, never what you look for) ===
Report anything that is actually wrong here. Then keep the fix in scope:
1. This is a RESEARCH-WORKFLOW review, not a security review. Assume a cooperating operator.
2. NO speculative machinery: do not add modules, gates, flags, or frameworks unless evidence shows a current defect they fix.
3. NO corner-case obsession unless the case arises here.
4. A clear sentence beats a scored table nobody maintains.
Say plainly when something is correct. Do not manufacture findings. Be brutally honest. If, after genuinely trying to break it, the plan holds up, say so clearly.
