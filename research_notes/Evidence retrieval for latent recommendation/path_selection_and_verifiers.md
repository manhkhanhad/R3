# Path selection among N candidates: verifiers, aggregation, calibration, and recommendation applications

Verification legend: [V] = arXiv ID and/or venue checked by search or fetch in this session. [M] = cited from prior knowledge. The ID and venue are believed correct but were not re-checked in this session, so confirm them before publishing.

## Q1. Self-consistency, best-of-N with ORMs/verifiers, PRMs, generative verifiers, rerankers, marginalization: gains, and when generator likelihood fails as a selector

### Takeaway
Repeated sampling raises *coverage* (the chance that a correct path exists) far faster than any selector can turn it into accuracy. The selector, not the generator, is the bottleneck. Methods that work better than raw likelihood either (a) marginalize over paths by voting on the final answer, or (b) train a *separate* verifier on correct/incorrect labels, at the outcome level, the step level, or generatively. Raw sequence likelihood fails in documented ways: model errors where the mode is degenerate, length bias, and surface-form competition. These are the same failure family as "conditioning makes every path look confident".

### Cited Findings
- **Coverage vs. selection gap.** Large Language Monkeys (Brown et al., arXiv 2407.21787 [V]): on MATH with Llama-3-8B-Instruct, coverage rises from 79.8% (100 samples) to 95.3% (10,000 samples). Majority voting and reward-model selection plateau and only move from 38.7% to 39.8% over the same range. The authors conclude that "building robust verifiers remains an open problem". On SWE-bench Lite, where an automatic verifier exists, coverage turns directly into accuracy: 15.9% at 1 sample, 56% at 250 samples. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787v3)
- **Self-consistency** (Wang et al., arXiv 2203.11171, ICLR 2023 [V for ID; venue M]): sample diverse CoT paths and take a majority vote over final answers, which is presented as marginalizing out the reasoning path. With PaLM-540B, GSM8K goes from 56.5% to 74.4% (+17.9 absolute points). The paper explicitly contrasts itself with trained verifiers and rerankers as "entirely unsupervised". Note that v1 reported a different figure (+10.6%). — [arXiv 2203.11171](https://arxiv.org/pdf/2203.11171v2)
- **ORM verifiers** (Cobbe et al., arXiv 2110.14168 [V]): sample many solutions and rank them with a trained correctness verifier. A 6B verifier slightly beats a fine-tuned 175B model, "roughly equivalent to a 30x model size increase", and verification "scales more effectively with increased data" than fine-tuning. — [arXiv 2110.14168](https://arxiv.org/pdf/2110.14168); [OpenAI blog](https://openai.com/index/solving-math-word-problems/)
- **PRMs** (Lightman et al., "Let's Verify Step by Step", arXiv 2305.20050 [V]; ICLR 2024 [M]): the process-supervised RM solves 78% of a representative MATH test subset, and process supervision "significantly outperforms outcome supervision". The PRM800K dataset has 800K step labels. The 78.2% best-of-1860 figure comes from a secondary source. — [arXiv 2305.20050](https://arxiv.org/pdf/2305.20050); [MATH-500 wiki (secondary)](https://www.aiwiki.ai/wiki/math-500)
- **Automatic PRM labels** (Math-Shepherd, arXiv 2312.08935 [V for ID]) derives step labels from Monte-Carlo rollouts instead of human annotation. — [arXiv 2312.08935](https://arxiv.org/pdf/2312.08935)
- **Generative verifiers** (GenRM; Zhang et al., arXiv 2408.15240, ICLR 2025 [V]): the verifier is trained by next-token prediction on "Is the answer correct? Yes/No", and the score is P("Yes"). A CoT variant samples verification rationales and averages them by majority vote. In best-of-N, GenRM beats discriminative verifiers, DPO verifiers, and LLM-as-judge: 5%→45.3% on algorithmic tasks, 73%→93.4% on GSM8K, and 28%→44.6% on easy-to-hard MATH. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html); [arXiv 2408.15240](https://arxiv.org/html/2408.15240v1)
- **Limits of best-of-N:** quality is capped by the RM. Best-of-N suffers reward over-optimization, and RMs mis-score OOD inputs. Ensembles of RMs "mitigate but do not eliminate" reward hacking (arXiv 2312.09244 [V for ID]). — [Survey arXiv 2505.02686](https://arxiv.org/pdf/2505.02686); [arXiv 2312.09244](https://arxiv.org/pdf/2312.09244)
- **Near-duplicate candidates:** candidates often differ in wording but not in meaning, which weakens selection. Diversify them by varying hyperparameters or models. — [Foundations of LLMs, arXiv 2501.09223](https://arxiv.org/pdf/2501.09223)
- **Likelihood failure, surface-form competition** (Holtzman et al., arXiv 2104.08315, EMNLP 2021 [V]): the highest-probability answer "isn't always right". Probability mass is split across paraphrases, and some answers are a-priori likely. Domain-conditional PMI, log P(y|x)/P(y|domain), gives consistent zero-shot gains over raw and calibrated scoring across GPT-2 and GPT-3 models. In a fixed-premise multiple-choice setting it is proportional to P(x|y), i.e. "scoring by premise". — [arXiv 2104.08315](https://arxiv.org/pdf/2104.08315); [ACL Anthology](https://preview.aclanthology.org/setup/2021.emnlp-main.564)
- **Likelihood failure, model errors vs. search errors** (Stahlberg & Byrne, EMNLP 2019, D19-1331 [V for venue]): an NMT study of whether the model's global mode is itself bad. From prior knowledge [M]: the exact-search mode is often the *empty* translation, so beam search's "errors" mask model errors. Confirm this in the paper. — [ACL Anthology D19-1331](https://aclanthology.org/D19-1331.xml)
- [M] Length bias: Murray & Chiang, "Correcting Length Bias in Neural MT" (WMT 2018, arXiv 1808.10006) show that sequence log-prob prefers short outputs and that larger beams make it worse. Unverified this session.
- [M] Mode inadequacy and MBR: Eikema & Aziz, "Is MAP decoding all you need?" (COLING 2020, arXiv 2005.10283) argue that the mode is atypical and propose minimum-Bayes-risk selection, which picks the candidate with the highest expected agreement with other samples. Unverified this session.
- [M] Universal self-consistency (Chen et al., arXiv 2311.17311) extends voting to free-form outputs by asking an LLM to pick the most consistent candidate. Snell et al. (arXiv 2408.03314) "Scaling LLM test-time compute optimally" compare best-of-N, PRM-guided beam search, and revisions, and find the optimal strategy depends on difficulty. Both unverified this session.
- An ICML 2026 talk on reward-weighted reranking with an imperfect judge argues that more aggressive reranking amplifies judge noise, so an optimal N or temperature exists. This is a secondary mention only; title and authors not confirmed. — [ICML 2026 slides](https://icml.cc/media/icml-2026/Slides/65764_QnpaFdQ.pdf)

### Inferences
- The user's setup matches the Large Language Monkeys pattern almost exactly. Oracle path selection (HR@10 0.2275) is far above what likelihood achieves (0.1450). That gap is a selector problem, not a generator problem.
- A decoder trained to "trust the conditioning" scores P(path | user, cluster). This is the hypothesis-conditioned likelihood that PMI analysis warns about: each path is likely *given its own cluster*, so comparing across clusters compares quantities with different, implicit priors. A PMI-style or noisy-channel score, e.g. log P(path|user,c) − log P(path|c), or a direct P(c|user) term, is the principled correction. The cluster predictor's 22.7% is effectively an estimate of P(c|user).
- Below-chance likelihood selection (7.3% vs 12.9%) on base-model misses means likelihood is *anti-correlated* with correctness on hard cases. This is consistent with model-error/mode-inadequacy findings: confident, generic paths win.
- Self-consistency-style marginalization transfers directly. Sum or vote item scores across the 8 paths (MBR over item lists) instead of picking one path. This needs no training and is a strong baseline to report.

### Gaps
- No source found that quantifies "conditioning on a hypothesis makes every path look confident" as a named phenomenon in LLM reasoning. The closest documented analogues are PMI/surface-form competition and the noisy-channel prompting literature (Min et al., arXiv 2108.04106, ACL 2022 [M]).
- Exact PRM vs. ORM numeric deltas from Lightman et al. were not extracted from the paper body.

## Q2. Scoring a hypothesis by consistency with independent evidence (cross-model checks, agreement, contrastive decoding, PMI)

### Takeaway
The robust pattern is to score a candidate by evidence that the generator did *not* produce: a second model, agreement across samples, or the input-specific part of the likelihood. Three forms recur. (1) PMI or contrastive scores divide out what is likely regardless of the input. (2) Agreement and MBR reward candidates that many independent samples or models support. (3) A cross-check uses a different model or another view.

### Cited Findings
- Domain-conditional PMI penalizes answers that are a-priori likely. It equals scoring by P(input | hypothesis), which beats raw and calibrated likelihood. — [Holtzman et al. 2021](https://arxiv.org/pdf/2104.08315)
- Self-consistency selects by agreement across independently sampled paths. — [Wang et al.](https://arxiv.org/pdf/2203.11171v2)
- Kadavath et al.: self-evaluation of P(True) "further improves when models are allowed to consider many of their own samples before predicting the validity of one specific possibility". In other words, agreement with sibling samples is informative evidence. — [arXiv 2207.05221](https://arxiv.org/pdf/2207.05221)
- Hou et al. (LLMRank, arXiv 2305.08845, ECIR 2024 [V]): LLM rankers are biased by candidate position and item popularity. Bootstrapping, i.e. re-ranking under shuffled candidate orders and aggregating, reduces position bias. This is an agreement-under-perturbation check. — [arXiv 2305.08845](https://arxiv.org/html/2305.08845v2)
- [M] Contrastive decoding (Li et al., arXiv 2210.15097, ACL 2023) scores tokens by log p_expert − log p_amateur, which penalizes generic continuations that a weak model also likes. MMI decoding for dialogue (Li et al., NAACL 2016, arXiv 1510.03055) reranks by log p(y|x) − λ log p(y) to suppress bland responses. Calibrate-before-use (Zhao et al., arXiv 2102.09690, ICML 2021) divides out the "content-free" prior. Multi-agent debate (Du et al., arXiv 2305.14325, ICML 2024) cross-checks answers across model instances. None of these four were re-verified this session.
- [M] Semantic entropy (Kuhn et al., arXiv 2302.09664, ICLR 2023) clusters samples by meaning and uses the entropy over clusters as an uncertainty signal. It predicts errors better than token-likelihood measures. Unverified this session.

### Inferences
- For cluster-conditioned paths, the most direct analogues of "independent evidence" are: (a) P(c | user) from a separate cluster predictor, (b) how many of the 8 paths retrieve each item, an MBR/agreement score, (c) a contrastive score of user-conditioned vs. user-agnostic decoding under the same cluster, which removes the cluster prior (the PMI analogue), and (d) a cross-model check against a CF model such as SASRec's score of the path's top items.
- Combining (a) with the path's PMI-corrected likelihood is a cheap Bayesian product, P(c|u)·P(items|u,c)/P(items|c). It is a natural first experiment.

### Gaps
- No recommendation paper found that applies PMI or contrastive scoring to select among generated paths or hypotheses.

## Q3. Calibration, selective prediction, abstention, learning-to-defer: training a verifier that knows when the base fails

### Takeaway
Train the selector or verifier on *labeled outcomes of the base model*, including its failures. Use a loss that pushes predictions toward uncertainty on failures (VRec) or a rejector/deferral loss (L2D). Generator-intrinsic confidence is calibrated only in favorable formats. It degrades out of distribution and on "do I know" questions, which is why external selectors are used.

### Cited Findings
- **VRec** (Lin et al., "Verifiable Reasoning for LLM-based Generative Recommendation", arXiv 2603.07725, Mar 2026, Meta/NUS [V]; preprint, no venue found). It uses a reason–verify–recommend loop over latent reasoning steps, built on LatentR3 with Qwen2.5-1.5B.
  - A *mixture of verifiers* (category, title-semantics clusters, SASRec-CF k-means clusters) with a personalized router. Each verifier is a linear head that predicts a coarse *group-level proxy* label, because true reasoning correctness is unobservable.
  - Feedback is the verifier entropy f = H(p). On negative samples (wrong recommendations) the verifier loss *maximizes entropy*, pushing toward uniform (Eq. 11). This is how it learns to "know when the base fails".
  - Guidance is the predicted class's weight vector. The adjustment is r* = Σ(1−c_i) r + c_i g_i with c_i = 1/f_i (Eq. 10).
  - Monotonicity regularization penalizes entropy increases across steps (Eq. 13). Training is two-stage: verifier pre-training, then joint fine-tuning.
  - Gains over the best baseline: CDs R@10 0.1213 vs 0.0978 (~+24%), Instruments ~+5%, MicroLens R@5 ~+29%, Goodreads R@5 ~+5%. On Goodreads R@10 the 1-step variant beats the full model.
  - The fetch noted an internal inconsistency: one passage says large f means good alignment, but c = 1/f down-weights it. Gains plateau after ~6 steps.
  - — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1); code: github.com/Linxyhaha/Verifiable-Rec (per search snippet)
- **Learning to defer** (Mozannar & Sontag, ICML 2020, PMLR 119:7076–7087, arXiv 2006.01862 [V]): learns a classifier plus a rejector that decides whether to defer to an expert, given only samples of the expert's decisions. It reduces the problem to cost-sensitive learning with a consistent surrogate that generalizes cross-entropy. — [PMLR](https://proceedings.mlr.press/v119/mozannar20b.html); [arXiv 2006.01862](https://arxiv.org/pdf/2006.01862)
- **LLM self-knowledge** (Kadavath et al., arXiv 2207.05221 [V]): large models are well-calibrated on MC and T/F questions "in the right format". P(True) self-evaluation works. P(IK) generalizes partially but struggles with calibration on new tasks. — [arXiv 2207.05221](https://arxiv.org/pdf/2207.05221)
- [M] Selective classification (Geifman & El-Yaniv, NeurIPS 2017, arXiv 1705.08500) and SelectiveNet (arXiv 1901.09192, ICML 2019) give risk–coverage trade-offs with a learned selection head. Modern NNs are miscalibrated, and temperature scaling fixes much of this (Guo et al., ICML 2017, arXiv 1706.04599). The GPT-4 technical report (arXiv 2303.08774) shows that RLHF/post-training degrades calibration. Multi-expert L2D: Verma, Barrejón & Nalisnick (AISTATS 2023, arXiv 2210.16955). None were re-verified this session.

### Inferences
- The user's problem is a learning-to-defer problem with K=8 "experts" (cluster-conditioned paths) plus the base model as a default. A rejector/selector trained with the Mozannar–Sontag surrogate on *which path hit* is the formal fit. The training label is available offline because hit/miss is known per path.
- VRec's trick of entropy maximization on failures is a selective-prediction signal. Applied to path selection, the selector should be trained on base-model misses specifically, where likelihood is below chance, rather than on all users. Otherwise easy cases dominate.
- Note that the target statistic (22.7% cluster-predictor accuracy vs 12.9% chance) is a coarse group-level proxy, the same design choice as VRec's proxy objective.

### Gaps
- VRec does not analyze the generator's own confidence as a selector, so no direct published comparison of "decoder log-prob vs. verifier" exists in recsys.
- Could not confirm the exact sign convention in VRec Eq. 10 (internal inconsistency flagged).

## Q4. Recommendation-specific: reranking generated candidates, LLM-as-ranker + verification, multi-path/multi-interest aggregation, reward models, test-time scaling

### Takeaway
Recsys is converging on the same lesson: a *separate* verifier or reranker over a generator's or retriever's candidates, plus test-time scaling via latent reasoning. Recent latent-reasoning recommenders are moving from depth to *width* (parallel paths with learned aggregation). No published recsys work found evaluates *selection* of a single path against oracle path choice.

### Cited Findings
- **"Recommendation Retrievers Need Verifiers"** (Zhang et al., arXiv 2609.12270, 10 Sep 2026, preprint [V]): a lightweight generative verifier added after training to an existing retriever. It scores a top-K candidate by the likelihood of its identifier tokens given the retriever's query state, trained by next-token cross-entropy with no negatives. It improves Recall@10 for SASRec, GRU4Rec, NextItNet, and MiniOneRec on Amazon and YaMBDa. Per the authors, the gains are not explained only by added content features. The abstract has no numbers. — [arXiv 2609.12270](https://arxiv.org/abs/2609.12270)
- **Parallel Latent Reasoning (PLR)** (Tang, Chen et al., arXiv 2601.03153, Jan 2026, Renmin/Alibaba, preprint [V]): parallel latent streams from learnable trigger tokens, kept diverse by a "global reasoning regularization", then combined by "mixture-of-reasoning-streams aggregation". It scales width instead of depth to avoid overthinking. A secondary summary claims >5% gains; the abstract gives no numbers and does not detail the aggregator. — [arXiv 2601.03153](https://arxiv.org/abs/2601.03153); [alphaxiv overview](https://alphaxiv.org/overview/2601.03153v1)
- **LARES** (arXiv 2505.16865 [V]): depth-recurrent latent reasoning for sequential recommendation, building on ReaRec ([M] arXiv 2503.22675, "Think before recommend"). These are depth-wise test-time scaling. — [arXiv 2505.16865](https://arxiv.org/pdf/2505.16865)
- **Other latent-reasoning recommenders.** ManCAR (Feb 2026) constrains latent reasoning trajectories to the interaction-graph manifold to prevent "latent drift". DiffuReason (arXiv 2602.09744) refines the first latent hypothesis with diffusion and GRPO. "Reasoning While Recommending" (arXiv 2601.13533) uses entropy to set latent-reasoning length in generative re-ranking. IDs come from search snippets; contents are not deeply verified. — [DiffuReason](https://arxiv.org/html/2602.09744v1); [arXiv 2601.13533](https://arxiv.org/abs/2601.13533v1); [ManCAR (emergentmind)](https://www.emergentmind.com/papers/2602.20093)
- **VRec** (see Q3) is the closest recsys analogue of a verifier steering latent reasoning. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- **LLM-as-ranker biases** (LLMRank, ECIR 2024): position and popularity bias, and poor perception of history order. Bootstrapping over shuffled orders mitigates them. — [arXiv 2305.08845](https://arxiv.org/html/2305.08845v2)
- [M] Multi-interest recommenders such as MIND (arXiv 1904.08030, CIKM 2019) and ComiRec (arXiv 2005.09347, KDD 2020) produce K interest vectors. ComiRec *aggregates* retrieved items across interests with a controllable diversity-aware aggregation module rather than picking one interest. This is the recsys precedent for "aggregate, don't select". Unverified this session.

### Inferences
- Two recsys-native options follow from this literature. (i) Aggregate: ComiRec/PLR-style merging of the 8 paths' item lists, with weights from a cluster predictor or router. This avoids the hard-selection failure altogether, and under HR@10 merging can capture items from several paths. (ii) Verify: a post-hoc generative verifier (2609.12270-style) or VRec-style group-level verifier scores each path's items given the user, independent of the decoder's conditioned likelihood.
- Given a 22.7% top-1 cluster predictor, soft weighting (top-2/top-3 mixture) is likely to recover more of the oracle gap than hard argmax. This is an inference, not a published result.

### Gaps
- No published reward model trained specifically for best-of-N selection over LLM-generated recommendation reasoning paths was found. The search for "reward model for LLM recommendation best-of-N" returned only general LLM work.
- PLR aggregator details and numbers were not retrievable from the abstract page.
- No recsys paper reports an oracle-vs-selector gap analogous to the user's 0.1450→0.2275.
