# Latent-reasoning, explicit-reasoning, verifiable-reasoning and retrieval-augmented LLM sequential recommenders (2023–2026)

Scope note: about 15 tool calls. Primary arXiv HTML was read for LatentR3, VRec, ReaRec, SCoRe and ManCAR. Everything else comes from abstracts or search snippets and is marked that way. Venues are marked (verified) only when a proceedings page or an official listing appeared in the results.

## Q1. Which works retrieve evidence, examples or reasoning from OTHER users or samples, and how do they choose what to retrieve?

### Takeaway
Retrieval from other users is now an established design. SCoRe and "What LLMs Miss" retrieve similar users, ManCAR uses a graph-neighbor candidate set, and RRCM learns retrieval with RL. In the most detailed cases, *selecting* the evidence (reranking, self-assessment, learned retrieval policy) matters as much as retrieving it, and adding more retrieved users quickly hurts. VRec does not retrieve from users. It gives the latent a *group-level* (k-means CF/title/category cluster) prototype predicted by a verifier, which is the closest analogue to "inject the target's CF cluster".

### Cited Findings
**SCoRe: "LLMs as Better Recommenders with Natural Language Collaborative Signals: A Self-Assessing Retrieval Approach"** (Xin, Sun, Wang, Zhang, Xiong; arXiv 2505.19464, May 2025; no venue stated)
- Retrieval: a pretrained SASRec embeds each user's sequence, and the top K_c=5 inner-product neighbours supervise a contrastive fine-tune of an mpnet text retriever (CAR) over users' histories rendered as text. At inference it retrieves K_e=10 users. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- Selection: a "Self-Assessing Reranker" (SARE). An LLM (GLM-4-Plus, 10k sampled users) writes what *other* preference characteristics would help predict the target. bge-reranker-large is trained with InfoNCE to rank the retrieved users against that assessment. This is supervised contrastive training, not RL. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- Injection: the top K_s=2 users' histories go into the prompt as natural language ("Other users with similar preferences have given high ratings to …"). Backbone is Llama-3.1-8B-Instruct + LoRA, scoring CTR-style Yes/No. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- Results (AUC/UAUC), Amazon-Games: SASRec 0.6111/0.5800, LLaRA 0.6346/0.6035, CoLLM 0.6797/0.5854, BinLLM 0.6756/0.5961, SCoRe 0.6933/0.6353. On ML-1M, BinLLM has higher AUC (0.7423 vs 0.7367), while SCoRe has the best UAUC (0.7190). — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- Selection matters. SCoRe-base (generic retriever), SCoRe-cf (CF similarity only) and the variant without the reranker all underperform the full model (numbers are only in a figure). On Games, performance *falls* as K_e grows, and the best K_s is 1 on Games and 2 on ML-1M. The authors cite lost-in-the-middle and say unreranked retrieval includes "not beneficial" instances. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- The model does use the evidence, but the support is qualitative: on one ML-1M example, attention visualisation shows the LLM attends to the natural-language collaborative info while soft-token CF embeddings get little attention. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)

**"What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals"** (arXiv 2505.20730; from search snippet only). It retrieves the users most similar to the target and supplies their past ratings at inference, comparing this against reasoning-based prompts. — [arXiv 2505.20730](https://arxiv.org/html/2505.20730v2)

**RRCM: "Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation"** (Li, Belligundu, Wei, Yang, Wang, Ghosh; arXiv 2605.07129, v1 May 2026, v2 Sep 2026; no venue)
- Per instance, the model starts from a light history context and chooses among: recommend directly, retrieve collaborative evidence, retrieve item metadata, or both. Memories are natural language behind one retrieval interface. — [arXiv abs 2605.07129](https://arxiv.org/abs/2605.07129v2)
- The retrieval policy is trained with GRPO on an **outcome-only ranking reward**, so retrieval is judged by final top-k quality. Motivation: prior methods don't learn *when* collaborative evidence is needed or *which* is most valuable under a context budget. Datasets and numbers were not retrieved. — [arXiv abs 2605.07129](https://arxiv.org/abs/2605.07129v2); [search snippet](https://arxiv.org/pdf/2605.07129)

**RALLRec+** (arXiv 2503.20430, Mar 2025). Retrieval-augmented LLM rec combined with representation learning and "reasoning-empowered" retrieval. Snippet only. — [arXiv 2503.20430](https://arxiv.org/abs/2503.20430)

**CRAG** (arXiv 2502.14137). Collaborative retrieval for LLM conversational recommenders. Snippet only. — [arXiv 2502.14137](https://arxiv.org/abs/2502.14137)

**G-CRS** (arXiv 2503.06430). Graph exploration finds similar users' past conversations, which serve as few-shot in-context demonstrations, training-free. Snippet only. — [arXiv 2503.06430](https://arxiv.org/html/2503.06430v1)

**ManCAR: "Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation"** (Yang et al., Xiamen Univ. + Shopee; arXiv 2602.20093, Feb 2026; no venue). Not an LLM: a 2-layer Transformer.
- Evidence: a global Swing-style item-item co-interaction graph. Each user's candidate set is their recent items plus k-hop graph neighbours. A rank-based teacher prior puts the most mass on the target (rank 0) and the rest on neighbours by edge weight. The candidate set is also injected as auxiliary context. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)
- At each latent step, a KL term pulls the step's item distribution toward this teacher. The teacher sharpens over steps. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)
- Amazon-2023 NDCG@10: CDs 0.0282 vs LARES 0.0192 vs SASRec 0.0132; Video 0.0275 vs 0.0219; Toys 0.0108 vs 0.0082; Office +15.65%; Arts +12.87%; Music only +2.56% (Recall@10 −3.56% vs PRL). Gains are larger on denser datasets. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)
- Ablation: removing the teacher prior drops CDs NDCG@10 from 0.0282 to 0.0212 and Video from 0.0275 to 0.0231. Most of the gain comes from the CF-neighbourhood supervision. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)

**IntuRec: "Intuition-Guided Latent Reasoning for LLM-Based Recommendation"** (arXiv 2606.27684, Jun 2026; authors from Beihang/USTC/Meta/CUHK/NUS; abstract and secondary summary only)
- Stage 1: the LLM recommender produces a top-K candidate list (the "intuition"). Stage 2: self- and cross-attention encode that list into an embedding that initialises latent reasoning. This addresses hidden states "misaligned with target item embeddings" in earlier latent recommenders. — [arXiv 2606.27684](https://arxiv.org/pdf/2606.27684)
- A secondary summary says it was evaluated on Amazon CDs/Toys/Games, peaks at K≈15 candidates, and needs about 1 reasoning step. These details are unverified against the paper. — [search summary / emergentmind](https://www.emergentmind.com/papers/2606.27684)

**VRec** (details in Q2). Evidence is not retrieved per user. Instead, per-aspect verifiers predict which k-means group (CF or title cluster, k ∈ {15, 20, 25}; categories: 27 for CDs, 19 for Instruments) the target falls in, and the classifier weight column of the predicted group is used as a "preference prototype" mixed into the reasoning embedding. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)

### Inferences
- Two families emerge. (a) Instance-level retrieval of other users' histories as text (SCoRe, "What LLMs Miss", RRCM, G-CRS). (b) Group- or neighbourhood-level CF priors that shape a latent (VRec prototypes, ManCAR graph teacher, IntuRec candidate-set embedding). Family (b) maps most directly onto an R3-style single latent token.
- In every detailed case, the selector (reranker, verifier, RL policy, teacher) is where the effort goes, and more raw retrieved evidence tends to hurt. This matches the user's finding that choosing the right evidence (e.g., the target's CF cluster) is the bottleneck.
- IntuRec and ManCAR both make the latent start from, or be constrained by, a candidate set that likely contains the target. This puts information absent from the prompt into the latent, the condition under which the user found the decoder actually uses it.

### Gaps
- I found no paper that retrieves other users' *reasoning traces* or latent thought vectors (as opposed to their histories) as evidence for the current user. This may be unexplored.
- RRCM, RALLRec+, "What LLMs Miss", CRAG: datasets and numbers not retrieved.
- RecRanker (hybrid ranking prompt with retrieved-user signals) was not researched within budget. Its venue and mechanism are unverified here.

## Q2. Which works supervise or verify the latent reasoning, and do they report that the model otherwise ignores or collapses its latent?

### Takeaway
LatentR3 (R3) does no analysis of whether the latent is used. Its ablations even show that the reasoning layer without RL gives a small part of the gain. ReaRec explicitly documents latent degradation: without supervision, step states become near-identical (cosine 0.98–1.00). VRec, ManCAR, LASAR and IntuRec are each built around the claim that unconstrained latents become homogeneous or drift, and each adds external supervision (group verifier, CF-graph teacher, CoT anchors, candidate prior). None of them reports a probe showing the decoder *ignores* the latent, such as counterfactual swapping or zeroing.

### Cited Findings
**LatentR3: "Reinforced Latent Reasoning for LLM-based Recommendation"** (Zhang, Xu, Zhao, Wang, Feng, He, Chua; arXiv 2505.19092; ICLR 2026 poster, verified via [ICLR virtual site](https://iclr.cc/virtual/2026/poster/10008315) and [mlanthology](https://mlanthology.org/iclr/2026/zhang2026iclr-reinforced/))
- Mechanism: an attention layer, LatentRATT, sits on the LLM's final hidden states and emits latent token(s) in the input-embedding space. Backbone is Qwen2.5-1.5B, applied on top of BIGRec and D3. — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)
- Training: stage 1 is full-model SFT warm-up with next-token loss. Stage 2 is a modified GRPO that updates **only LatentRATT** (so the KL term is zero), samples latents with Gaussian reparameterisation noise, uses reward = negative perplexity of the ground-truth item (continuous), and uses a batch-mean rather than group-mean advantage. It reports about 1/5 the cost of standard GRPO. Standard GRPO restricted to LatentRATT was "significantly worse". — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)
- Token count K ∈ {0, 1, 2, 4, 8}: **one latent token is best** on both datasets. More tokens give no gain, possibly because a larger latent space is harder to learn. — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)
- The paper states that RL from scratch is unstable and prone to collapse in the high-dimensional latent space, which is why SFT initialises it. — [search snippet of arXiv 2505.19092](https://arxiv.org/pdf/2505.19092)
- Ablation, H@5 Toys/CDs: full 0.0821/0.0934; w/o reasoning (= BIGRec) 0.0701/0.0757; w/o LatentRATT 0.0772/0.0705; **w/o RL (SFT only) 0.0804/0.0830**. On Toys, most of the gain is already there with SFT plus the extra layer. — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)
- There is no analysis of what the latent encodes, whether the decoder uses it, or whether it collapses. The authors acknowledge that latents are less interpretable than CoT. — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)

**VRec: "Verifiable Reasoning for LLM-based Generative Recommendation"** (Lin, Zeng, Yu, Xia, Zhang, Singh, Liu, Wang, Feng, Chua, Wang; Meta MRS + NUS; arXiv 2603.07725, 8 Mar 2026; no venue found; code at github.com/Linxyhaha/Verifiable-Rec)
- The paper frames the problem as reason→recommend without intermediate checks, which leads to "homogeneous" reasoning and error accumulation. It proposes reason→verify→recommend. **Built on a pretrained LatentR3 recommender**, Qwen2.5-1.5B with full fine-tuning. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- Verifiers: a mixture of MLP verifiers for category (metadata labels), title-semantic and CF aspects (k-means groups, k ∈ {15, 20, 25}), with a personalised router. Feedback is the entropy f = H(p) of the verifier's group prediction. Guidance g_i is the last-layer weight column of the predicted class, used as a prototype. The update is r* = Σ (1−c_i)·r + c_i·g_i with c_i = 1/f_i. How the weights are normalised across verifiers is unclear from the text. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- Training: stage 1 pretrains the verifier with cross-entropy on positives plus entropy *maximisation* on negatives (weight α). Stage 2 uses L = L_rec + β·L_verifier + γ·L_m, where L_m penalises entropy increasing across reasoning steps (monotonicity). — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- Results (Amazon CDs and **Instruments**, MicroLens, Goodreads; 8:1:1 temporal split; max sequence length 10):
  - CDs R@5: VRec 0.0963 vs LatentR3 0.0801 vs SASRec 0.0784 (+20% over LatentR3).
  - Instruments R@5: 0.1098 vs LatentR3 0.1022 (+7%); N@5 0.0944 vs 0.0881.
  - Goodreads R@5: 0.0366 vs 0.0349 (+5%).
  - MicroLens R@10: 0.0235 vs 0.0160.
  - Baselines include SASRec, TIGER, LETTER, SETRec, D3, LatentR3 and LatentTTS (not BIGRec).
  - Reported p < 0.01.
  - — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- Ablations (CDs; figure only): removing the verifier, using a single verifier, removing the router, removing L_m, or removing verifier pretraining each hurts. Adding verifiers cumulatively raises R@5: 0.0854 (category) → 0.0940 (+title) → 0.0973 (+CF). On MicroLens, the CF-only verifier beats title-only. Steps help up to a plateau at 6–10. Added inference cost is about 0.59%. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- Collapse or homogeneity is shown only qualitatively (t-SNE, case study). No quantitative collapse metric, no sensitivity sweep over number of groups, and no long-tail analysis. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)

**ReaRec: "Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation"** (Tang, Dai et al., Renmin Univ. + Alibaba; arXiv 2503.22675, v3 Aug 2025; venue not stated on the arXiv page, so unverified)
- Mechanism: the last hidden state is fed back autoregressively into the sequence encoder for K steps, with reasoning position embeddings. Backbones are SASRec, BERT4Rec, UniSRec and MoRec (not LLMs). — [arXiv 2503.22675](https://arxiv.org/html/2503.22675v3)
- ERL: averages all step states and adds a KL regulariser pushing apart step-wise output distributions (λ in 0.001–0.1; performance drops when λ > 0.05). PRL: progressive temperature annealing plus reasoning-aware contrastive learning (Gaussian noise γ=0.01, InfoNCE). — [arXiv 2503.22675](https://arxiv.org/html/2503.22675v3)
- **Reported degradation:** without these objectives, step states become near-identical. For the RPE variant, similarity of the final output to the prior two steps is 1.00 and 0.98. Naive multi-step reasoning *underperforms* the base model. With α=1.0 the model copies the prior state (shortcut). Without RCL, errors accumulate (a target's rank goes from #12 to #22). — [arXiv 2503.22675](https://arxiv.org/html/2503.22675v3)

**LARES** (Liu, Zheng, Wang, Zhao, Wang, Chen, Wen; arXiv 2505.16865, May 2025; venue unverified). Depth-recurrent pre-block plus core-block that refines *all* input tokens each step. Trained with self-supervised trajectory- and step-level alignment, then RL. Reports large margins on four datasets. Abstract only. — [arXiv 2505.16865](https://arxiv.org/abs/2505.16865v1)

**LASAR** (arXiv 2605.10207, May 2026; abstract only). For Semantic-ID generative rec it names "representation drift due to lack of reasoning-chain supervision". It constrains each latent step via bidirectional KL to hidden-state anchors extracted from CoT text, plus a policy head for adaptive depth. Trained SFT→RL. It roughly halves the average number of steps and is about 20× faster than explicit CoT. — [arXiv 2605.10207](https://arxiv.org/pdf/2605.10207)

**ManCAR**: the CF-graph teacher KL is the supervision (see Q1). Drift is motivated qualitatively and not measured directly. KL between adjacent steps falls sharply after training (figure only). Adaptive halting uses KL(p_{t−1} || p_t) < ε. Mean steps: CDs 1.84, Toys 3.58. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)

**Collaborative signals fading inside LLMs (non-latent but relevant):**
- "Beyond Semantic Understanding" / FreLLM4Rec (Wang et al.; arXiv 2508.10312; no venue): LLM recommenders "progressively weaken the inherent collaborative signals" from injected ID embeddings layer by layer and overemphasise semantic correlations. Classic Transformers preserve them. Spectral filtering gives up to +8.00% NDCG@10. — [arXiv 2508.10312](https://arxiv.org/abs/2508.10312)
- CoRA (arXiv 2408.10645) reports that CoLLM's performance on ML-1M "significantly declined after introducing textual information", i.e., soft-token CF/ID alignment interferes with text semantics. — [arXiv 2408.10645](https://arxiv.org/pdf/2408.10645)
- LLaRA (SIGIR 2024; venue via [Paper Digest](https://www.paperdigest.org/certificate/d23a229466211be2/pdf)) uses a curriculum from text-only prompts to hybrid ID+text prompts, so the LLM learns to incorporate the ID embeddings. It reports no ablation of whether the IDs are used after training. — [arXiv 2312.02445](https://arxiv.org/html/2312.02445v4)

### Inferences
- The literature consistently implies, but never directly tests, that unsupervised latents collapse or carry little. ReaRec's 0.98–1.00 step similarity is the clearest quantitative evidence. LatentR3's own ablation (SFT-only reaches most of the gain on Toys) leaves open whether the RL-trained latent carries much beyond extra capacity.
- VRec's gains over LatentR3 are smallest on Instruments (+7% R@5), the domain closest to the user's setup, and largest on CDs. Group-prototype guidance helps, but the ceiling depends on how well the verifier predicts the target's group, which matches the user's diagnosis that evidence selection is the bottleneck.
- FreLLM4Rec and CoRA show the same pattern for soft CF tokens as the user found for the R3 latent: the LLM decoder tends to discount injected non-text vectors unless they are protected or made indispensable. SCoRe's attention case study points the same way, with soft tokens ignored and natural-language CF used.

### Gaps
- None of the papers found runs a counterfactual probe of whether the decoder uses the latent (zeroing it, shuffling it across users, or measuring mutual information with the target). This appears to be an open analysis gap.
- VRec: no sweep over the number of groups k, and no accuracy of the verifier's group prediction reported in the text read.
- Venues for ReaRec, LARES and VRec are unverified (not found in proceedings pages within budget).

## Q3. What gains do these methods report over strong sequential baselines (SASRec, plain SFT LLM rec), and are gains concentrated on hard or long-tail users and items?

### Takeaway
Gains over the base LLM recommender are typically 5–25% relative (LatentR3 +10–22%, VRec +5–20% over LatentR3). Gains over SASRec are larger but vary by dataset. Where reported, gains concentrate on unpopular items and short-history users. ReaRec shows extra reasoning *hurts* long-history users and popular items.

### Cited Findings
- **LatentR3, Amazon H@5/N@5:**
  - Toys: SASRec 0.0601/0.0458, BIGRec 0.0701/0.0508, +LatentR3 0.0821/0.0600, D3 0.0830/0.0610, D3+LatentR3 0.0898/0.0670.
  - CDs: SASRec 0.0841/0.0622, BIGRec 0.0757/0.0616, +LatentR3 0.0934/0.0754, D3 0.1122/0.0906, D3+LatentR3 0.1137/0.0915.
  - Games: SASRec 0.0416, BIGRec 0.0461, +LatentR3 0.0580, D3 0.0608, D3+LatentR3 0.0716.
  - Average relative improvement is 21.8% over BIGRec and 10.4% over D3. On CDs, D3+LatentR3 is barely above D3.
  - "Significantly larger improvements on unpopular items" (bottom 80% by frequency), with values shown only in a figure.
  - — [arXiv 2505.19092](https://arxiv.org/html/2505.19092v1)
- **VRec:** see Q2. CDs +20% R@5 over LatentR3 and +23% over SASRec; Instruments +7%; Goodreads +5%. — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- **ReaRec:** average +7.49% at about 3.51% extra latency (best at 2 steps). On SASRec, ERL +6.76% and PRL +8.21%; on UniSRec, ERL +12.29% and PRL +10.43%. The "30–50%" figure is a post-hoc *oracle* best-step ceiling, not a deployable gain. Long-tail item group IG-1 (short-sequence users) gains 12.08% → 16.35% → 18.69% NDCG@20 as steps increase. Long-sequence users and popular items degrade with more steps ("overthinking"). — [arXiv 2503.22675](https://arxiv.org/html/2503.22675v3)
- **ManCAR:** +12–47% NDCG@10 over LARES on most Amazon-2023 categories, but only +2.56% on Music. Gains are larger on denser data, and cold start is not studied. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)
- **SCoRe:** small AUC gains over CoLLM/LLaRA/BinLLM (about +1.8% on ML-1M averaged), clearer on Amazon-Games UAUC (0.6353 vs 0.6035 for LLaRA). — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- **Explicit-reasoning recommenders:**
  - **Rec-R1** (Lin, Wang, Qian; TMLR 2025, verified via [mlanthology](https://mlanthology.org/tmlr/2025/lin2025tmlr-recr1/)): RL on LLM text output using a frozen black-box recommender's score as reward, with no GPT-distilled SFT. Tasks are product search, sequential rec and re-ranking. It reportedly preserves general abilities that SFT erodes. — [arXiv 2503.24289](https://arxiv.org/html/2503.24289v3)
  - **R2Rec "Reason-to-Recommend"** (Zhao, Xu, Li; arXiv 2506.05069): samples interaction chains from the user-item graph (i.e., other users' paths) and turns them into "interaction-of-thought" via masked prompting, then SFT on annotated traces followed by RL. Reports +10.48% HR@1 on average over baselines and +131.81% over the original LLM on three datasets. Abstract only. — [arXiv 2506.05069](https://arxiv.org/pdf/2506.05069)
  - **R²ec** (You et al.; NeurIPS 2025, verified via [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/5a05d477fef5a43c805d3377700efa40-Abstract-Conference.html)): dual-head model (reasoning generation plus item head), trained with RecPO RL and no annotated reasoning. Abstract only. — [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117677)
  - **SLIM** (Wang et al., 2024, arXiv 2403.04260): distils teacher CoT rationales into a small LM (about 4% of the teacher's parameters), whose rationales are embedded for ID-based and ID-agnostic sequential rec. — [arXiv 2403.04260](https://arxiv.org/pdf/2403.04260)
  - **IGR-SR** (arXiv 2512.14034): argues reasoning supervised only by the next item is "unstable and shallow" and anchors it to extracted intents. Reports +7.13%. Snippet only. — [alphaxiv 2512.14034](https://www.alphaxiv.org/abs/2512.14034)
- **Collaborative-signal injection:** CoLLM (arXiv 2310.19488) maps an external CF model's embeddings into the LLM token space. BinLLM (arXiv 2406.03210; likely AAAI per an [ojs.aaai.org link](https://ojs.aaai.org/index.php/AAAI/article/view/33334/35489), unverified) encodes CF embeddings as binary text. E4SRec (arXiv 2312.02443) injects ID embeddings for sequential rec; its venue is unverified. — [arXiv 2310.19488](https://arxiv.org/pdf/2310.19488); [arXiv 2406.03210](https://arxiv.org/html/2406.03210v1); [arXiv 2312.02443](https://arxiv.org/pdf/2312.02443)

### Inferences
- Latent and CF-evidence methods help most where the history alone is least informative (short histories, unpopular items). The likely reason is that they then add information absent from the prompt, which fits the user's finding.
- On CDs, D3+LatentR3 adds only +1.3% H@5 over D3, so latent reasoning gains shrink over a strong base. Expect modest headroom on a 1.5B model over a strong SFT baseline unless the latent carries new evidence.

### Gaps
- No long-tail or user-group breakdown was found for VRec, SCoRe or ManCAR.
- Exact numbers for E4SRec, CoLLM, BinLLM, RecRanker and R2Rec were not retrieved.

## Q4. Any reported failure modes: reasoning collapse, shortcut learning, latents ignored by the decoder?

### Takeaway
Reported failure modes:
- **Step homogenisation or copying:** ReaRec measured it; VRec and LASAR report it qualitatively.
- **RL instability and collapse** without SFT warm-start: LatentR3.
- **Error accumulation and overthinking** for easy or long-history cases: ReaRec.
- **Drift away from target-item space:** IntuRec, LASAR, ManCAR.
- **Collaborative or soft-token signal attenuated by the LLM:** FreLLM4Rec, CoRA, SCoRe's attention case study.
- **Noisy or excess retrieved evidence hurting:** SCoRe.

### Cited Findings
- ReaRec: near-identical step states (similarity 1.00/0.98); naive multi-step reasoning is below the base model; shortcut copying at α=1.0; errors accumulate without RCL; overthinking on long-sequence users and popular items. — [arXiv 2503.22675](https://arxiv.org/html/2503.22675v3)
- LatentR3: RL from scratch is "unstable and prone to collapse" in the latent space, so it uses SFT initialisation. Restricting standard GRPO to the latent layer only was "significantly worse". — [arXiv 2505.19092](https://arxiv.org/pdf/2505.19092); [arXiv 2505.19092 HTML](https://arxiv.org/html/2505.19092v1)
- VRec: without verification, reasoning becomes "homogeneous" and accumulates errors (t-SNE only). — [arXiv 2603.07725](https://arxiv.org/html/2603.07725v1)
- IntuRec: prior latent reasoners start from hidden states "misaligned with target item embeddings". — [arXiv 2606.27684](https://arxiv.org/pdf/2606.27684)
- LASAR: "representation drift" from the lack of chain supervision. — [arXiv 2605.10207](https://arxiv.org/pdf/2605.10207)
- ManCAR: intermediate states evolve without explicit constraints. Removing the CF teacher costs about 25% NDCG@10 on CDs. — [arXiv 2602.20093](https://arxiv.org/html/2602.20093)
- FreLLM4Rec: LLM layers progressively weaken the injected collaborative signal. — [arXiv 2508.10312](https://arxiv.org/abs/2508.10312)
- CoRA: CoLLM degrades on ML-1M when text is added, showing fusion interference. — [arXiv 2408.10645](https://arxiv.org/pdf/2408.10645)
- SCoRe: performance declines with more retrieved users (Games), attributed to lost-in-the-middle. In its example, soft CF tokens receive little attention. — [arXiv 2505.19464](https://arxiv.org/html/2505.19464)
- Related, not recommendation: LatentTTS ("Parallel Test-Time Scaling for Latent Reasoning Models", You et al., arXiv 2510.07745; ACL 2026 via the [ACL Anthology](https://aclanthology.org/2026.acl-long.2069/)) samples latent trajectories with MC-dropout or Gaussian noise and selects them with a step-wise contrastive latent reward model. VRec uses LatentTTS as a baseline. — [HF 2510.07745](https://huggingface.co/papers/2510.07745)

### Inferences
- The user's observation that the decoder ignores the latent unless it carries information absent from the prompt is consistent with all the above. It is not stated or tested as such in any paper found, so it could be a novel, publishable diagnostic: for example, a counterfactual latent swap, or comparing an oracle-cluster latent with a predicted-cluster one.
- Methods that work (VRec prototypes, ManCAR teacher, IntuRec candidate prior, SCoRe text evidence) all inject *externally selected collaborative information*, not more self-computation. The improvement seems to come from the evidence, not from extra "reasoning" depth.

### Gaps
- No paper reports a quantitative "latent ignored by decoder" test (ablation by zeroing or swapping the latent at inference, or gradient or attention attribution to the latent token) for LatentR3-style LLM recommenders.
- WhisperRec ("Latent Reasoning for Efficient Foundation Recommendation Models", arXiv 2607.26621) appeared in search but was not read.
