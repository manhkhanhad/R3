# Explicit (textual CoT) reasoning recommenders, 2024–2026: where the gains come from (contrast class for latent-reasoning recommenders)

Verification legend. **[FT]** means the claim was read from the arXiv HTML full text. A small model summarized the fetched pages, and for long papers only the first ~100k characters were read, so appendices are sometimes missing. Exact numbers should be re-checked against the PDF before they are quoted in a paper. **[ABS]** means the claim comes from the abstract page or a search snippet only.

Coverage: R²ec, OneRec-Think, ThinkRec, ReRec, Reason4Rec, Rec-R1, RecGPT, SLIM, RDRec, KAR, LangPTune and DeepRec. On the text-to-latent side: SCoTER, LASAR and EvoReason, plus the latent-only contrast papers LatentR3, IntuRec, HiLaR and FLR. KAR (first posted June 2023) and SLIM/RDRec (2024) fall at or before the start of the 2024–2026 window. They are kept because the task named them.

## Q1. Source of reasoning traces (teacher / self / RL / target-conditioned vs history-only)

### Takeaway
The traces come from three sources:
- **Teacher distillation:** ChatGPT/GPT-3.5, DeepSeek-R1, QwQ-32B, GPT-5, or Llama-2-7B in RDRec.
- **Self-generated, target-conditioned rationales:** OneRec-Think.
- **RL-discovered traces with no rationale supervision:** R²ec, Rec-R1, ReRec, DeepRec and LangPTune.

Several "teacher" pipelines leak the target, either through the generation prompt or through hindsight filtering. In those cases part of the trace's information is hindsight knowledge that is not available at test time.

### Cited Findings
- **R²ec [FT].** Trained "without reasoning annotations". Trajectories are sampled from the policy conditioned only on the user prompt (history), and the item is chosen after the reasoning tokens, so the reasoning never sees the target. — [R²ec](https://arxiv.org/html/2505.16994)
- **OneRec-Think [FT].** No external teacher: "we query our pre-aligned model to generate a rationale" for each target interaction.
  - The traces are **target-conditioned (hindsight)**. The prompt asks why a user with the pruned history (top-10 most similar items) would interact with the target item.
  - On the open-source datasets, manually constructed category-based CoT is used instead. — [OneRec-Think](https://arxiv.org/html/2510.11639)
- **ThinkRec [FT].** Teacher is QwQ-32B. A few thousand training instances were re-queried until the teacher's prediction was correct, and the last explanation was kept.
  - The prompts include the target item and the ground-truth answer, so the traces are target-conditioned, post-hoc rationalizations. — [ThinkRec](https://arxiv.org/html/2505.15091)
- **Reason4Rec [FT].** Teacher is ChatGPT (GPT-3.5-turbo).
  - Reasons are generated from historical aspect-preference summaries.
  - They are then filtered by a reward model that compares them to the **target item's review**.
  - On retry, the **true rating is given as a hint**, which leaks the label into the training rationales. — [Reason4Rec](https://arxiv.org/html/2502.02061)
- **RecGPT (Taobao) [FT].** DeepSeek-R1 produced about 90.0k reasoning-enhanced samples, which were manually reduced to 19.0k for distillation. A further 21.1k self-generated, quality-filtered samples were added during "self-training evolution". QwQ-32B is used only as a data-cleaning judge. — [RecGPT](https://arxiv.org/html/2507.22879)
- **SLIM [FT].** Teacher is ChatGPT (gpt-3.5-turbo). The CoT prompts are filled with the user's history only, and the target item is not given to the teacher. — [SLIM](https://arxiv.org/html/2403.04260)
- **RDRec [FT].** Teacher is Llama-2-7b. The rationale is generated from **each interaction's own review**: "A user bought an item and said '{review}'…". It is therefore target-conditioned per training interaction, and the paper does not say whether test-label reviews are excluded. — [RDRec](https://arxiv.org/html/2405.10587)
- **KAR [FT].** Uses "API of a widely-used LLM", with the model not named. Preference reasoning is built from profile, behavior history and scenario factors. The paper does not use target-conditioning. — [KAR](https://arxiv.org/html/2306.10933)
- **Rec-R1 [FT].** Qwen-2.5-3B-Instruct trained with RL "without access to GPT-4o outputs". The prompt asks for `<think>` reasoning before the JSON query. Only the SFT baseline uses GPT-4o data. — [Rec-R1](https://arxiv.org/html/2503.24289)
- **ReRec [FT].** Self-generated through GRPO rollouts (G=5). No teacher and no target-conditioned trace generation is reported. — [ReRec](https://arxiv.org/html/2604.07851v1)
- **DeepRec [FT].** RL from Qwen2.5-7B-Base. The "cold-start" stage is RL with format, invocation and diversity rewards, not SFT on teacher data. — [DeepRec](https://arxiv.org/html/2505.16810v1)
- **LangPTune [FT].** The base LLM (Gemma-2B-it or Llama-3-8B-it) writes profiles from the user's history and reviews. These are then RL-optimized with downstream NDCG@20 as the reward. — [LangPTune](https://arxiv.org/html/2410.18870v2)
- **SCoTER [FT].** DeepSeek-R1 generates the chains, conditioned on the interaction sequence. The next item is used only in a "Validate" phase that scores whether a chain's recommendations hit the target, which is hindsight selection. A Qwen3-8B student then regenerates the chains offline. — [SCoTER](https://arxiv.org/html/2511.19514v6)
- **LASAR [FT].** "Teacher CoT reasoning text is generated by GPT-5". The visible text does not say whether the CoT is target-conditioned. — [LASAR](https://arxiv.org/html/2605.10207)

### Inferences
- In Reason4Rec, ThinkRec, OneRec-Think, RDRec and (via selection) SCoTER, the training rationales contain hindsight information about the target. In those systems the reasoning text partly acts as a privileged-information or label-smoothing channel during training. Part of any gain may come from that channel rather than from inference-time reasoning.
- Teacher-world-knowledge pipelines (SLIM, KAR, RecGPT, SCoTER, LASAR) inject knowledge external to the interaction data. RL-only pipelines (R²ec, Rec-R1, ReRec, DeepRec) can only elicit knowledge already in the backbone, plus extra test-time computation.

### Gaps
- ThinkRec's own backbone size is not stated in the visible text.
- The KAR generation LLM is unnamed in the paper.
- LASAR's CoT conditioning sits in an appendix that was truncated.

## Q2. Training recipe (SFT on rationales, RL reward design, distillation)

### Takeaway
Two families dominate:
- **SFT/distillation on teacher rationales:** SLIM, RDRec, Reason4Rec, ThinkRec and RecGPT.
- **GRPO-style RL with rank-metric rewards:** R²ec, Rec-R1, ReRec, OneRec-Think stage 3, DeepRec and LangPTune.

The RL rewards are almost always the recommendation metric itself (NDCG or hit), sometimes with dense shaping terms.

### Cited Findings
- **R²ec [FT].** RecPO, compared under both GRPO and RLOO.
  - Reward R = β·R_c + (1−β)·R_d with β = 0.05. R_d is NDCG@1000 of the target item's rank; R_c is a softmax similarity over all items.
  - Settings: G = 4, no KL term, temperature 1.5.
  - It has a separate recommendation head (item embedding scoring), and only the highest-advantage trajectory gets gradient on the recommendation step. — [R²ec](https://arxiv.org/html/2505.16994)
- **OneRec-Think [FT].** Four stages:
  1. Itemic token warm-up (~6B tokens on industrial data).
  2. Multi-task alignment.
  3. SFT on rationale + target with joint NLL.
  4. GRPO with a "Rollout-Beam" reward: the maximum matched itemic tokens over a beam of K = 32. Settings: G = 16, KL β = 0.001. — [OneRec-Think](https://arxiv.org/html/2510.11639)
- **ThinkRec [FT].** SFT only, with no RL. Mixed loss: cross-entropy on reasoning samples (0.2 sampling ratio) and BCE on the "Yes" logit for recommendation samples (0.8). LoRA r = 8 on q/v projections, plus mixture-of-experts LoRA gating per user. — [ThinkRec](https://arxiv.org/html/2505.15091)
- **Reason4Rec [FT].** Llama3-8B with three QLoRA adapters (summarizer, reasoner, predictor) trained sequentially by SFT on teacher outputs. A separate reward-model adapter, trained on 8,000 pairs, filters the reasons. — [Reason4Rec](https://arxiv.org/html/2502.02061)
- **Rec-R1 [FT].** GRPO with a rule-based reward equal to the metric (NDCG@100 for search, NDCG@K for sequential recommendation).
  - Cost: about $0.48 and 210 s of training, versus about $15.60 for the GPT-4o-distilled SFT pipeline, i.e. less than 1/30 of the cost.
  - Retention: SFT loses more than 27 points on IFEval, while Rec-R1 holds or slightly improves. — [Rec-R1](https://arxiv.org/html/2503.24289)
- **ReRec [FT].** GRPO-style objective (lr 5e-6, KL 0.01, G = 5).
  - Reward r = NDCG + 0.01·QAS + 0.01·PAS, where QAS is attribute overlap with the ground truth and PAS is LightGCN cosine similarity.
  - A segment-level advantage penalizes reasoning segments that discuss the wrongly recommended item (w = 0.3), and an online curriculum sets query difficulty. — [ReRec](https://arxiv.org/html/2604.07851v1)
- **DeepRec [FT].** Reinforce++ in two stages: 20 cold-start steps, then 80 recommendation-oriented steps.
  - Process rewards: format, invocation count with M = 3 calls to the traditional recommender (TRM), and diversity of the generated preferences.
  - Outcome rewards: point-wise similarity, hit, and rank. — [DeepRec](https://arxiv.org/html/2505.16810v1)
- **LangPTune [FT].** "RLSO" alternates two steps for K = 5 iterations:
  - Profile-policy optimization: squared loss over pairs of sampled profiles, with reward = normalized NDCG@20.
  - InfoNCE training of the Mxbai embedding recommender. — [LangPTune](https://arxiv.org/html/2410.18870v2)
- **RecGPT [FT].** Training stages:
  1. Curriculum multi-task fine-tuning on 16 subtasks.
  2. Reasoning-enhanced pre-alignment, i.e. distillation from DeepSeek-R1.
  3. Self-training evolution judged by a Human-LLM judge.

  No RL. Backbones: Qwen3-14B and TBStars-MoE-42B-A3.5B. — [RecGPT](https://arxiv.org/html/2507.22879)
- **SLIM [FT].** LoRA SFT of LLaMA2-7B on 1–2k ChatGPT rationales per dataset. The rationales are encoded with BERT and fused into GRU4Rec, SASRec or SRGNN, or matched to item text with no ID backbone. — [SLIM](https://arxiv.org/html/2403.04260)
- **RDRec [FT].** T5-small (P5/POD-style) trained multi-task on sequential, top-N, explanation and rationale-generation tasks. — [RDRec](https://arxiv.org/html/2405.10587)
- **KAR [FT].** Generated text is encoded offline (ChatGLM). A hybrid-expert adaptor maps it to augmented vectors that are fed to 9 CTR models and 4 reranking models. — [KAR](https://arxiv.org/html/2306.10933)

### Inferences
- In every RL paper, the reasoning text and the metric-aligned RL objective are introduced together. That matters for Q3.
- Rec-R1's cost and retention results suggest that a metric-reward RL stage, not teacher rationales, drives much of the "reasoning" benefit in the RL family.

### Gaps
- R²ec's default advantage estimator (GRPO or RLOO) for the main table is not clearly stated in the visible text.

## Q3. Reasoning vs no-reasoning ablations, and whether they are confounded

### Takeaway
Where a "reasoning vs no reasoning" ablation exists, the gains are modest:
- About 15% relative in R²ec.
- About +0.003–0.007 absolute on Beauty in OneRec-Think.
- MAE −0.009 to −0.020 in Reason4Rec.
- 2–18% UAUC in ThinkRec.
- About 5–25% top-N H@10 in RDRec.

These ablations are confounded almost everywhere: removing reasoning also removes the RL stage, the auxiliary loss, or the extra supervised text. No paper runs the clean control of "same RL or same supervision, but no reasoning text". The one exception is Rec-R1's SFT-format ablation, which finds no consistent benefit from reasoning.

### Cited Findings
- **R²ec [FT].** "w/o Reasoning" removes the reasoning tokens and trains only with in-batch contrastive loss, so it also removes RecPO. Reasoning and RL are therefore confounded.
  - Numbers:
    - CDs H@20 0.0945 → 0.1086 and N@20 0.0456 → 0.0525.
    - Games H@20 0.0748 → 0.0853.
    - **Instruments H@5 0.0176 → 0.0264**.
  - Reported average gain is "roughly 15%".
  - Caveat: the fetched table may mix backbones, so check the PDF. — [R²ec](https://arxiv.org/html/2505.16994)
- **OneRec-Think [FT]**, Amazon Beauty:

  | Variant | R@5 | R@10 | N@5 | N@10 |
  |---|---|---|---|---|
  | Base | 0.0460 | 0.0654 | 0.0314 | 0.0377 |
  | +Itemic Alignment | 0.0532 | 0.0735 | 0.0342 | 0.0402 |
  | +Reasoning | 0.0563 | 0.0791 | 0.0398 | 0.0471 |

  - Itemic alignment (no reasoning) gives a gain similar in size to adding reasoning.
  - The +R step also adds SFT on rationales and GRPO with a different reward, and there is no RL-without-reasoning control. No seeds or variances are reported. — [OneRec-Think](https://arxiv.org/html/2510.11639)
- **ThinkRec [FT]**, w/o think vs full:

  | Dataset | UAUC | NDCG@5 |
  |---|---|---|
  | ML1M | 0.6599 → 0.6775 | 0.7570 → 0.7747 |
  | Yelp | 0.5865 → 0.6065 | 0.8402 → 0.8585 |
  | Book | 0.4692 → 0.5705 | 0.6284 → 0.6858 |

  - Anomaly: "w/o both" scores 0.5017 UAUC on Book, which is higher than "w/o think". The authors call this counterintuitive and attribute it to overfitting.
  - These are single runs, and the ablation also removes the reasoning loss and its data. — [ThinkRec](https://arxiv.org/html/2505.15091)
- **Reason4Rec [FT]**, without reasoning vs full:

  | Dataset | MAE | RMSE |
  |---|---|---|
  | Music | 0.5529 → 0.5442 | 0.7742 → 0.7722 |
  | Book | 0.6225 → 0.6029 | 0.8394 → 0.8345 |

  These are single seeds, and label information leaks into the rationales (Q1). — [Reason4Rec](https://arxiv.org/html/2502.02061)
- **RDRec [FT]**, without rationales vs full, H@10 (sequential / top-N):

  | Dataset | Without rationales | Full RDRec |
  |---|---|---|
  | Sports | 0.0566 / 0.2977 | 0.0596 / 0.3683 |
  | Beauty | 0.0705 / 0.2777 | 0.0743 / 0.3380 |
  | Toys | 0.0768 / 0.2200 | 0.0802 / 0.2375 |

  The gains are much larger for top-N than for sequential recommendation. — [RDRec](https://arxiv.org/html/2405.10587)
- **Rec-R1 [FT]**, SFT-format ablation on ESCI, no-reasoning JSON vs reasoning JSON:

  | Domain | No reasoning | Reasoning |
  |---|---|---|
  | Games | 26.87 | 25.70 |
  | Baby | 22.83 | 19.66 |
  | Office | 26.10 | 27.34 |
  | Sports | 26.42 | 24.82 |

  - Reasoning is worse in 3 of 4 domains.
  - On GSM8K, the no-reasoning JSON variant scores 68.1 vs 35.7 for the reasoning variant.
  - There is no RL-without-think ablation. — [Rec-R1](https://arxiv.org/html/2503.24289)
- **RecGPT [FT].** No controlled reasoning ablation. The closest comparison is DeepSeek-R1 vs Qwen3-Base on human-judged pass rates, e.g. item-tag prediction 80.0% vs 33.7%, but those models differ in many ways.
  - Online, over one month with 1% traffic per group: CTR +6.33%, IPV +9.47%, CICD (category diversity of clicks) +6.96%, DT +4.82%. — [RecGPT](https://arxiv.org/html/2507.22879)
- **ReRec, DeepRec and SLIM [FT].** None reports an explicit no-reasoning ablation.
  - DeepRec ablates RL, cold start, two-stage RL and data selection. Capping TRM calls at M = 1 degrades performance.
  - DeepRec beats TedRec by about +5% Recall@5 on ML-1M (0.1550 → 0.1629) and about +13% on Games (0.0598 → 0.0678). — [DeepRec](https://arxiv.org/html/2505.16810v1); [ReRec](https://arxiv.org/html/2604.07851v1); [SLIM](https://arxiv.org/html/2403.04260)
- **LangPTune [FT].**
  - RL-tuned profiles improve on zero-shot profiles from the same LLM: NDCG@20 Movie-TV/Llama 0.0231 → 0.0494 (+114%), and Books/Gemma 0.1743 → 0.2736 (+57%).
  - **A no-profile, no-bottleneck contrastive model ("NoProfile-CL") reaches 0.0543**, higher than LangPTune's 0.0494 on Movie-TV. Forcing a text bottleneck therefore costs accuracy.
  - On Movie-TV, LangPTune trails MCL (0.0563) and LightGCN (0.0525). — [LangPTune](https://arxiv.org/html/2410.18870v2)
- **KAR [FT].** Relative AUC gains of +0.94% to +1.49% across CTR backbones. Per the text, "reasoning" knowledge contributes more than "factual" knowledge, but the exact values are only in Fig. 5.
  - Online: +7% recall (news), and +1.7% song plays (music). — [KAR](https://arxiv.org/html/2306.10933)

### Inferences
- The size of the gain attributable to the reasoning text itself is unidentified in every RL paper. R²ec's roughly 15% is a "reasoning + RL" vs "contrastive only" comparison.
- LangPTune's NoProfile-CL result and Rec-R1's format ablation are the closest evidence that the text bottleneck is not where accuracy comes from.
- Taken together, the evidence suggests most gains come from:
  - (a) RL directly optimizing the ranking metric;
  - (b) extra world/item knowledge from teacher LLMs, which shows up clearly in KAR, SLIM, RecGPT and SCoTER, all of which add teacher text to a strong ID backbone;
  - (c) hindsight or label leakage during rationale construction.

  Gains specifically from inference-time verbal deliberation look small.

### Gaps
- No paper found runs "RL with an empty or random think block" or "SFT on shuffled or other-user rationales" as a control.
- No paper reports variance across seeds for the reasoning ablation.

## Q4. Does the prediction demonstrably depend on the reasoning content? (perturbation / swap / truncation / faithfulness)

### Takeaway
No paper in the set runs a counterfactual intervention on the rationale: perturbing, swapping, truncating or replacing it with another user's rationale and measuring the change in prediction. The evidence offered is either correlational or qualitative:
- Reasoning quality correlates with accuracy.
- Case studies show "alignment" between rationale and output.

ThinkRec's recommendation prompt even emits "Yes/No" before the analysis, so the reasoning cannot cause the label.

### Cited Findings
- **R²ec [FT].** No perturb, truncate or swap experiments in the visible text. The paper argues that reasoning "directly reshapes" the final hidden state h_T. — [R²ec](https://arxiv.org/html/2505.16994)
- **OneRec-Think [FT].** The only evidence is a qualitative consistency figure (Fig. 5) showing "strong alignment" between reasoning and recommended items. The claim that the reasoning is not post-hoc rests on that figure. — [OneRec-Think](https://arxiv.org/html/2510.11639)
- **ThinkRec [FT].**
  - Reason quality is measured with METEOR/BLEURT, i.e. text similarity.
  - The recommendation prompt asks the model to **begin with "Yes" or "No"**, so the label is generated before the reasoning.
  - No faithfulness tests. — [ThinkRec](https://arxiv.org/html/2505.15091)
- **Reason4Rec [FT].** Correlational only:
  - Correct predictions have higher reasoning quality (Table V).
  - Lower error tracks higher reasoning quality (Fig. 5).
  - Table VIII feeds reasons to a different model (the reward model). — [Reason4Rec](https://arxiv.org/html/2502.02061)
- **ReRec [FT].** None. RQ2 perturbs user history, not reasoning content. There is one case study. — [ReRec](https://arxiv.org/html/2604.07851v1)
- **DeepRec, Rec-R1, RecGPT [FT].** No test of dependence on reasoning content is reported. — [DeepRec](https://arxiv.org/html/2505.16810v1); [Rec-R1](https://arxiv.org/html/2503.24289); [RecGPT](https://arxiv.org/html/2507.22879)
- **SLIM and RDRec [FT].** Only step-wise or component ablations are reported (SLIM: Step3 > Step2 > Step1; RDRec: removing preference vs attribute text). There are no shuffled or corrupted rationales. — [SLIM](https://arxiv.org/html/2403.04260); [RDRec](https://arxiv.org/html/2405.10587)
- **SCoTER [FT].** The closest thing to a structure test:
  - Removing positional encoding over the steps lowers Beauty R@10 by 3.71% (0.1187 → 0.1143), with a marginal effect on Sports (0.0889 → 0.0882).
  - Replacing the step-wise embeddings with last-token, mean-pooled or single-vector versions costs 13.23%, 6.91% and 4.04%.
  - There is no content-shuffle or swap test. — [SCoTER](https://arxiv.org/html/2511.19514v6)
- **LangPTune [FT].** The closest thing to an "is the text informative" test is human and GPT-4 next-item choice given the profile.
  - Humans: 58.89% with LangPTune profiles vs 54.67% with zero-shot profiles vs 57.11% with raw metadata. McNemar p = 0.201, not significant.
  - GPT-4: 76.67% vs 73.33% vs 75.56%. — [LangPTune](https://arxiv.org/html/2410.18870v2)
- A related evaluation study of LLM explanations in recommendation reports a "faithfulness–trust gap": simulated users rated clarity 3.86 but trust 2.60. Its faithfulness metric is not an intervention test, and this was checked at search-snippet level only. — [MDPI Electronics 2026](https://www.mdpi.com/2079-9292/15/20/4640)

### Inferences
- The field has not shown that textual recommendation reasoning is causally used. A rationale-swap or truncation test would be a novel and easy contribution, and a fair comparison for a latent model's "does the latent matter" tests. Examples:
  - Feed user A's think block to user B.
  - Truncate the think block at k%.
  - Replace the think block with the empty string at inference in R²ec or Rec-R1.

### Gaps
- R²ec's appendix (extended analysis and case studies) was truncated and might contain such a test.
- I found no reliable rec-specific CoT faithfulness benchmark.

## Q5. Which user/item segments benefit (long-tail, category transitions / novel needs, cold start)?

### Takeaway
Segment analyses are rare. Where they exist, gains concentrate on **sparse users and tail items**:
- SCoTER: tail +30.75% vs head +21.42%; sparsest users +23.90%; online GMV +4.10% for sparse vs +1.49% for dense users.
- SLIM: larger gains for sparse users.
- LatentR3 (a latent model): larger gains on unpopular items.
- Rec-R1: the inductive (unseen-item) setting benefits more.
- RecGPT: category diversity of clicks (CICD) +6.96% online.

No paper analyzes **category transitions / next-item category shift** explicitly. RecGPT's CICD and its intent → tag → item design is the closest thing.

### Cited Findings
- **SCoTER [FT]**, Beauty, SCoTER-H vs HSTU:
  - Items: popularity-quartile Group 0 (tail) +30.75% vs Group 3 (head) +21.42%.
  - Users: sparsest +23.90%, Group 2 +30.24%, densest +16.22%. The user trend is non-monotonic.
  - Online A/B: GMV +4.10% for sparse users vs +1.49% for dense users. — [SCoTER](https://arxiv.org/html/2511.19514v6)
- **SLIM [FT].** User-sparsity groups G1–G5 show larger gains for sparse users. A popularity-bias analysis uses EFD@10 and EPC@10. Exact numbers were not extracted. — [SLIM](https://arxiv.org/html/2403.04260)
- **Rec-R1 [FT].** Sequential recommendation on Beauty has 1,000 inductive and 538 transductive test cases. Appendix E.2.4 says Rec-R1 does better in the inductive (unseen-item) setting. Aggregate numbers were not found in the extracted text. — [Rec-R1](https://arxiv.org/html/2503.24289)
- **RecGPT [FT].** Online CICD (category diversity) +6.96%, and the abstract claims reduced Matthew effect and more long-tail exposure. A popularity-group figure (Fig. 1) is described qualitatively, with no numbers in the text. — [RecGPT](https://arxiv.org/html/2507.22879)
- **LangPTune [FT].** More robust than MCL and LLM2BERT4Rec as training data shrinks (100% → 0.1%), which the authors suggest is useful for cold start. — [LangPTune](https://arxiv.org/html/2410.18870v2)
- **LatentR3 (latent, no CoT) [FT].** Larger relative gains over BIGRec on unpopular items, defined as items outside the top 20% by training frequency. The numbers are only in Fig. 2 and App. G. — [LatentR3](https://arxiv.org/html/2505.19092)
- **No segment analysis reported:** R²ec (cold start and long tail appear only as motivation), OneRec-Think, ThinkRec, ReRec (splits only by query complexity and profile type), DeepRec, RDRec and KAR. Reason4Rec has one qualitative cold-start case. — [R²ec](https://arxiv.org/html/2505.16994); [OneRec-Think](https://arxiv.org/html/2510.11639); [ThinkRec](https://arxiv.org/html/2505.15091); [ReRec](https://arxiv.org/html/2604.07851v1); [DeepRec](https://arxiv.org/html/2505.16810v1); [RDRec](https://arxiv.org/html/2405.10587); [KAR](https://arxiv.org/html/2306.10933); [Reason4Rec](https://arxiv.org/html/2502.02061)

### Inferences
- Where text reasoning helps tail items and sparse users, the plausible mechanism is **teacher world knowledge about items**: semantic relations an ID model cannot learn from few interactions. SCoTER and SLIM both inject DeepSeek-R1 or ChatGPT text into ID backbones. Deliberation per se is a less likely explanation.
- For a latent model whose misses are category transitions and long-tail items, this points to injecting item-semantic or world knowledge (teacher-text anchors as in LASAR or SCoTER, or intent/tag prediction as in RecGPT) rather than adding more latent steps.
- These inferences are not directly tested by any paper.

### Gaps
- No paper measures performance on next items whose category differs from the recent history ("category transition"). This is open.
- Exact per-segment numbers for SLIM, LatentR3 and Rec-R1 are only in figures or appendices that were not extracted.

## Q6. Datasets (Amazon Musical Instruments flagged) and backbone sizes

### Takeaway
Amazon "Instruments" (Musical Instruments) appears in **R²ec, SCoTER, LASAR and LatentR3**. It is not used by OneRec-Think, ThinkRec, Reason4Rec, ReRec, Rec-R1, SLIM, RDRec, KAR, LangPTune or DeepRec. Backbones are mostly 0.6B–8B, with 14B and a 42B-A3.5B mixture-of-experts model in industry.

### Cited Findings
- **R²ec [FT].** Video Games (29,230 users / 10,144 items), CDs (7,701 / 12,024) and **Instruments (15,656 / 10,320)**. Backbones: Gemma2-2B-Instruct and Qwen2.5-3B-Instruct.
  - Latency on an RTX 3090 at batch size 1: R²ec 1.67 s vs SASRec 0.014 s, and 0.0945 s with vLLM. — [R²ec](https://arxiv.org/html/2505.16994)
- **OneRec-Think [FT].** Amazon Beauty, Sports and Toys. Qwen3-1.7B for open-source experiments; "Qwen-8B" in industry.
  - Kuaishou online A/B: App Stay Time +0.159%. — [OneRec-Think](https://arxiv.org/html/2510.11639)
- **ThinkRec [FT].** ML1M, Yelp and Amazon Book. The Llama3-8B baseline backbone, QwQ-32B teacher, and PolyLM-Qwen-7B keyword extractor are named. — [ThinkRec](https://arxiv.org/html/2505.15091)
- **Reason4Rec [FT].** Amazon Digital Music (5,541 users / 3,568 items), Book and Yelp, all for rating prediction. Llama3-8B with QLoRA. — [Reason4Rec](https://arxiv.org/html/2502.02061)
- **ReRec [FT].** RecBench+ Movie and Book queries. Qwen2.5-3B and Llama-3.2-3B. — [ReRec](https://arxiv.org/html/2604.07851v1)
- **Rec-R1 [FT].** ESCI, Amazon-C4 and Amazon Beauty. Qwen2.5-3B. — [Rec-R1](https://arxiv.org/html/2503.24289)
- **RecGPT [FT].** Taobao production. Qwen3-14B and TBStars-MoE-42B-A3.5B. — [RecGPT](https://arxiv.org/html/2507.22879)
- **SLIM [FT].** Amazon Games, Food and Home. LLaMA2-7B student, with Bloomz-560M and 1B also tested; performance rises with size. — [SLIM](https://arxiv.org/html/2403.04260)
- **RDRec [FT].** Amazon Sports, Beauty and Toys. T5-small student, Llama-2-7b teacher. — [RDRec](https://arxiv.org/html/2405.10587)
- **KAR [FT].** MovieLens-1M and Amazon-Books, plus Huawei news and music online. — [KAR](https://arxiv.org/html/2306.10933)
- **LangPTune [FT].** Amazon Movie-TV and Books. Gemma-2B-it and Llama-3-8B-it. — [LangPTune](https://arxiv.org/html/2410.18870v2)
- **DeepRec [FT].** ML-1M and Amazon-2023 Video Games. Qwen2.5-7B-Base, with TedRec as the traditional recommender. — [DeepRec](https://arxiv.org/html/2505.16810v1)
- **SCoTER [FT].** Beauty, **Instruments (24,772 users / 9,922 items / 206,153 interactions)**, Sports and Yelp. TIGER and HSTU backbones, Qwen3-8B student. — [SCoTER](https://arxiv.org/html/2511.19514v6)
- **LASAR [FT].** Beauty, **Instruments** and Sports. Qwen3-0.6B and 1.7B.
  - Instruments NDCG@20: LASAR 0.0730 vs MiniOneRec 0.0677. — [LASAR](https://arxiv.org/html/2605.10207)
- **LatentR3 [FT].** Toys, CDs, Games and **Instruments**. Qwen2.5-1.5B. — [LatentR3](https://arxiv.org/html/2505.19092)

### Inferences
- The dataset statistics differ across papers. R²ec's Instruments has 15,656 users, while SCoTER's has 24,772. Cross-paper number comparisons on "Instruments" are therefore not valid without matching the preprocessing.

### Gaps
- Instruments numbers for R²ec beyond H@5 were not extracted.
- LatentR3's and SCoTER's per-metric Instruments numbers were not extracted.

## Q7. Text→latent compression/distillation of recommendation reasoning (2025–2026)

### Takeaway
There is a clear 2025–2026 line that turns textual CoT into latent or embedding form:
- **SCoTER** (DeepSeek-R1 chains → step-wise embeddings fused into TIGER/HSTU, no online LLM).
- **LASAR** (GPT-5 CoT hidden-state anchors align latent steps via bidirectional KL, then RL).
- **EvoReason** (on-policy distillation of primitive-guided teacher CoT into latent reasoning).

Other latent methods (LatentR3, IntuRec, HiLaR, FLR) use no text teacher. The latent and compressed approaches report matching or exceeding explicit CoT at 20–30× lower latency. LASAR's own ablation shows that CoT alignment adds only about 2–3%.

### Cited Findings
- **SCoTER [FT].**
  - Pipeline: Generate-Validate-Mine → Qwen3-8B student regenerates chains offline → each step embedded (K×D) → order-preserving cross-attention and gated fusion.
  - Offline gains: SCoTER-T vs TIGER +3.75% to +11.59%; SCoTER-H vs HSTU +5.35% to +41.24%. Beauty R@10: HSTU 0.0930 → 0.1187.
  - Online: Tencent Ads GMV +2.14%, with LLM inference cost eliminated online.
  - Removing student distillation drops Beauty R@10 to 0.1008. — [SCoTER](https://arxiv.org/html/2511.19514v6); [SCoTER abs](https://arxiv.org/abs/2511.19514)
- **LASAR [FT].**
  - SFT on Beauty, NDCG@10:
    - Pure SFT: 0.0277.
    - Latent without alignment: 0.0278 (+0.4%).
    - KL alignment to CoT anchors: 0.0285 (+2.9%).
    - Cosine alignment: 0.0277. MSE alignment: 0.0245 (−11.6%).
  - RL stage: latent RL without alignment −2.7%; adding Terminal KL +2.4%; adding REINFORCE depth control a further +3.1%, with mean steps falling from 4.20 to 2.47.
  - Vs explicit CoT (GREAM): Sports NDCG@10 0.0152 vs 0.0118. Per-sample time 0.29 s vs 7.0 s, i.e. more than 20× faster. — [LASAR](https://arxiv.org/html/2605.10207)
- **EvoReason [ABS].** Argues that distilling raw CoT into latent representations is suboptimal because the traces are "redundant and unstable". It extracts reusable reasoning primitives from agentic recommendation trajectories and runs a self-evolving on-policy distillation loop. No numbers on the abstract page. — [EvoReason](https://arxiv.org/abs/2607.29010)
- **LatentR3 [FT]**, latent, no CoT:
  - Reward = negative perplexity of the target given the latent.
  - Toys H@5 0.0701 → 0.0821; CDs 0.0757 → 0.0934.
  - Explicit CoT baselines score far lower: Toys H@5 0.0261 and CDs 0.0302, at about 25–30× the latency.
  - "COT-RL without supervision did not help". — [LatentR3](https://arxiv.org/html/2505.19092)
- **IntuRec, HiLaR and FLR [ABS].** Latent reasoning without a text teacher:
  - IntuRec initializes latent reasoning from a top-K candidate "intuition".
  - HiLaR uses hierarchical preference states with layer-aware process rewards based on marginal target-likelihood gain, on 4 Amazon datasets.
  - FLR uses multi-factor latent thoughts plus GRPO. — [IntuRec](https://arxiv.org/abs/2606.27684); [HiLaR](https://arxiv.org/abs/2607.27760); [FLR](https://arxiv.org/abs/2604.26760)

### Inferences
- LASAR's +0.4% for "latent without alignment" vs +2.9% with KL-to-CoT alignment is the cleanest available estimate of what textual-CoT supervision adds to a latent recommender: small and positive, with the alignment loss choice mattering (MSE hurts).
- SCoTER's larger gains (up to +41% on HSTU) come with a teacher that was hindsight-validated against targets, plus external world knowledge. That fits the pattern that teacher knowledge drives the gains rather than deliberation.
- Explicit CoT baselines in LatentR3 and LASAR (GREAM) underperform latent methods. This is evidence against verbal deliberation being necessary, though baseline tuning quality is unknown.

### Gaps
- Full texts of EvoReason, IntuRec, HiLaR and FLR were not read. Their datasets (including whether Instruments is used) and numbers are unverified.
- No text-to-latent paper reports a test of whether the latent steps carry rationale content, for example probing latents for the teacher's semantic content.
