# LLM-backbone latent reasoning for recommendation (2024–2026): per-paper dossier

Verification legend: **[FT]** means the detail was read in the arXiv HTML full text through a fetch-and-extract tool, which summarises a page rather than returning it verbatim, so treat quoted numbers as transcribed. **[ABS]** means the detail comes only from an abstract, search snippet or secondary citation. Research date: 2026-10-11.

## Q0. Which papers exist? (inventory)

### Takeaway
I found 13 directly relevant papers from 2025–2026 where an LLM (or a T5 generative recommender) reasons through continuous latent states before emitting titles or semantic IDs (SIDs). Ten have an LLM backbone. Two use small T5 backbones but are close relatives. Three latent-reasoning papers on non-LLM sequential recommenders are noted for completeness. The field clusters around the Qwen2.5-1.5B + title + Amazon {Toys, CDs, Games, Instruments} protocol that LatentR3 established, and a Qwen3 + SID line built on OneRec-Think/MiniOneRec.

### Cited Findings
**LLM backbone, latent reasoning (core set)**
1. **LatentR3**: "Reinforced Latent Reasoning for LLM-based Recommendation", Zhang, Xu, Zhao, Wang, Feng, He, Chua (NUS/USTC/CUHK), arXiv 2505.19092 (v2). [FT] Its ethics statement mentions the "ICLR Code of Ethics", but the venue is not confirmed. — [arXiv](https://arxiv.org/html/2505.19092)
2. **FLR**: "Factorized Latent Reasoning for LLM-based Recommendation", Gao, Huang, Wang, Liu, Zeng, Yao, arXiv 2604.26760. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
3. **LASAR**: "Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation", Chen et al. (Beihang/Baidu), arXiv 2605.10207 (v2). [FT, first ~100k characters of ~142k] — [arXiv](https://arxiv.org/html/2605.10207)
4. **PauseRec**: "Implicit Reasoning for Large Language Model-based Generative Recommendation", He, Collins, Kumar, Li, Shah, Loveland, arXiv 2606.14142. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
5. **IntuRec**: "Intuition-Guided Latent Reasoning for LLM-Based Recommendation", Liu, Bai, Zhao, Zhang, Wang, Feng, Rong, arXiv 2606.27684. [FT] This is the same NUS/USTC group as LatentR3. — [arXiv](https://arxiv.org/html/2606.27684)
6. **RecGPT-V3 (Latent Intent Reasoning component)**: Taobao RecGPT team technical report, arXiv 2607.15591. [FT, first ~100k characters] — [arXiv](https://arxiv.org/html/2607.15591)
7. **LaRec**: "Unleashing LLM-based Latent Reasoning for Generative Recommendation", Xia, Lin, Yang, Zhong, Chen, Ren, Hu (UCAS/Xiaohongshu), arXiv 2607.24617. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
8. **HiLaR**: "Hierarchical Latent Reasoning for LLM-based Recommendation", Hu et al. (XJTLU/Xiaohongshu/PKU/BJTU), arXiv 2607.27760. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
9. **EvoReason**: "Self-Evolving Reasoning Primitive-Guided On-Policy Distillation for Latent Reasoning in Generative Recommendation", Zhuang et al. (Kuaishou), arXiv 2607.29010. [FT] — [arXiv](https://arxiv.org/html/2607.29010)

**Generative recommendation with latent reasoning on a T5 backbone (not an LLM, but SID-based and closely related)**
10. **S²GR**: "Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation", Guo et al. (Kuaishou), arXiv 2601.18664. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
11. **IBA / "Where Reasoning Matters"**: "Rethinking Latent Reasoning in Semantic ID-based Generative Recommendation", Yang, Gao, Wang, Yu (Chongqing/Griffith), arXiv 2607.12425. [FT] — [arXiv](https://arxiv.org/html/2607.12425)

**Latent reasoning on non-LLM sequential recommenders (abstract-level, for completeness)**
12. **ReaRec**: "Think before Recommend" (Tang et al.), arXiv 2503.22675. It runs implicit multi-step reasoning over hidden user representations by feeding the last hidden state back into the recommender. [ABS] CogRec's related work says it is "a conventional sequential recommender" that generates no text or SIDs. A search snippet calls it "NeurIPS 2026" without confirmation. — [CogRec arXiv](https://arxiv.org/html/2607.24402); [OneRec-Think arXiv](https://arxiv.org/html/2510.11639v1)
13. **LARES**: "Latent Reasoning for Sequential Recommendation", Liu, Zheng, Wang, Zhao, Wang, Chen, Wen, arXiv 2505.16865. It uses depth-recurrent (looped) latent reasoning that "refines all input tokens at each implicit reasoning step". Training is self-supervised pre-training with trajectory- and step-level alignment, followed by RL post-training. [ABS] — [arXiv abs](https://arxiv.org/abs/2505.16865)
14. **PLR**: "Parallel Latent Reasoning for Sequential Recommendation", arXiv 2601.03153. It is known only from HiLaR's reference list. [ABS, title only] — [HiLaR arXiv](https://arxiv.org/html/2607.27760)

**Checked and excluded (explicit text reasoning, not latent)**
- **OneRec-Think** (arXiv 2510.11639) uses in-text reasoning. It positions itself against "implicit" reasoners that lack readable rationales. — [arXiv](https://arxiv.org/html/2510.11639v1)
- **CogRec** (arXiv 2607.24402) uses Qwen3-1.7B with 4-level SIDs. Its "fast/slow" reasoning is a discrete text trace of Match/LateralJump/Explore operations, and it states no continuous mode exists. [FT] — [arXiv](https://arxiv.org/html/2607.24402)
- **SIDReasoner** (arXiv 2603.23183) and **TwiSTAR** are explicit reasoning over SIDs. I saw them at abstract or citation level only. — [arXiv](https://arxiv.org/html/2603.23183v1)
- **LRT**: "Rethinking LLM Reasoning: From Explicit Trajectories to Latent Representations", ICLR 2026. It is general-purpose latent reasoning, not recommendation. — [ICLR](https://www.iclr.cc/virtual/2026/poster/10010844)
- These candidates were seen but **not read**: "Beyond Matching: Category-Guided Latent Intent Reasoning for Generative Retrieval in E-Commerce" (arXiv 2606.07075), "From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment" (arXiv 2609.29983; its HTML returned 404 and the PDF was unparseable), "Learning Better Reasoning for Generative Recommendation with Semantic IDs" (arXiv 2609.29973), and "TGR" (arXiv 2609.00986). — [search results](https://arxiv.org/pdf/2606.07075)

### Inferences
- **Naming collision.** HiLaR's results tables label a baseline "LaRec" with numbers attributed to LARES. This is not the same paper as LaRec (2607.24617). Anyone citing "LaRec" numbers from HiLaR should check which paper they refer to.
- Two benchmark families dominate:
  - **(a) The LatentR3 protocol:** Qwen2.5-1.5B, titles, Amazon Toys/CDs/Games/Instruments, chronological 8:1:1 split, history length 10. LatentR3, FLR, IntuRec, HiLaR and LaRec use it.
  - **(b) The SID protocol:** Qwen3, 4×256 SIDs, Amazon Beauty/Sports/Toys/Instruments with leave-one-out. LASAR, PauseRec and EvoReason use it.

### Gaps
- I could not do an exhaustive Semantic Scholar citation crawl of LatentR3 within the tool budget, so further follow-ups probably exist.
- I did not read in full: PLR, ReaRec, LARES, 2606.07075, 2609.29983, 2609.29973.

## Q1. Backbone and output space

### Takeaway
All LLM papers use Qwen at 0.6B–14B. The title-generation line uses Qwen2.5-1.5B with trie-constrained beam search over titles. The SID line uses Qwen3-0.6B or 1.7B with 4-level SID tokens added to the vocabulary. No paper uses LLaMA or a scoring head as the primary output.

### Cited Findings
- **LatentR3:**
  - Backbone: Qwen2.5-1.5B (Qwen2.5-3B with LoRA in Appendix E).
  - Output: item text (titles), plugged into BIGRec and D3 with constrained decoding. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **FLR:**
  - Backbone: Qwen2.5-1.5B, frozen during GRPO.
  - Output: titles via prefix-trie beam search. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- **LASAR:**
  - Backbone: Qwen3-0.6B fully fine-tuned, scaled to Qwen3-1.7B with LoRA.
  - Output: SIDs with M=4 levels and 256-entry RQ-KMeans codebooks, added to the vocabulary. [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- **PauseRec:**
  - Backbone: Qwen3-1.7B.
  - Output: 4-level SIDs under constrained decoding, with `<pause>` tokens between `<think>` and `</think>`. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **IntuRec:**
  - Applied on top of BIGRec and D3 (titles). Baselines use Qwen2.5-1.5B; the paper does not state IntuRec's own backbone explicitly.
  - Beam size 10 with constrained decoding. [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **RecGPT-V3:**
  - Backbone: a "hybrid-modal foundation model" extending Qwen3-14B with SID tokens.
  - Output: item *tags* (intent tags), not items. [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- **LaRec:**
  - Backbone: Qwen2.5-1.5B.
  - Output: item identifiers ("ID or Title") via trie-constrained beam search. It does not use SIDs. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- **HiLaR:**
  - Backbone: frozen Qwen2.5-1.5B.
  - Output: titles with catalog-constrained decoding. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- **EvoReason:**
  - Backbone: the OneRec-Think SFT Qwen3-1.7B.
  - Output: 4×256 SIDs with beam size 10. [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- **S²GR:**
  - Backbone: T5 with 4 encoder and 4 decoder layers (TIGER configuration).
  - Output: 3-level × 256 SIDs. Item embeddings come from Qwen3-Embedding-4B. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
- **IBA / Where Reasoning Matters:**
  - Backbone: T5-small.
  - Output: 4-token RQ-VAE SIDs under a prefix trie. [FT] — [arXiv](https://arxiv.org/html/2607.12425)

### Inferences
- Effects in this literature have been demonstrated almost exclusively at the 0.6–1.7B scale. The one exception is RecGPT-V3 at 14B, and it predicts tags rather than items.

### Gaps
- IntuRec's exact backbone for its own method is not explicitly stated in the text I read.

## Q2. How latent steps are produced

### Takeaway
Mechanisms span five families:
- **(i)** One or a few autoregressive latent tokens produced by an extra attention head over final hidden states: LatentR3, IntuRec, HiLaR, LaRec.
- **(ii)** Iterated attention-pooled "thought" slots: FLR.
- **(iii)** Coconut-style full hidden-state feedback loops with adaptive depth: LASAR, EvoReason.
- **(iv)** Fixed learned tokens whose content comes only from contextual attention, with no fed-back state: PauseRec, RecGPT-V3.
- **(v)** Per-SID-position latent refinement in T5 decoders: S²GR, IBA.

### Cited Findings
- **LatentR3:**
  - A "LatentRATT" attention layer on top of the final decoding layer reads the final-layer hidden states, including earlier latent tokens. Its last-position output becomes the next latent token, so this is autoregressive feedback, not pooling.
  - Main setting is **N=1** latent token; Appendix H tests 0, 1 and 2. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **FLR:**
  - A `<|Thought|>` token is updated over T=2 iterations.
  - K learnable query prototypes (K=3 for CDs, K=4 for the others) attention-pool the LLM hidden states using RoPE keys.
  - An MLP softmax gate merges the factors, and the result replaces the thought embedding. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- **LASAR:**
  - The last-layer hidden state at the final prompt token (h0) seeds a loop. Each step feeds the previous hidden state back in place of a token embedding, as sequential forward passes with no projection layer.
  - A 2-layer MLP "Policy Head" on h0 picks N ∈ {1..8}.
  - The answer is then decoded reusing the KV cache. [FT] — [arXiv](https://arxiv.org/html/2605.10207)
  - LASAR explicitly argues that "simply inserting additional learnable tokens or adding single-shot cross-attention does not constitute" latent reasoning. [ABS snippet] — [arXiv](https://arxiv.org/html/2605.10207)
- **PauseRec:**
  - k=5 literal `<pause>` tokens (sweep 1/3/5/10) share **one learned embedding**.
  - Hidden states are computed normally and are not fed back as inputs. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **IntuRec:**
  - Autoregressive: r1 = LLM(x)[-1], and each rn is conditioned on the prompt and earlier states.
  - r1 is replaced by r1+, the output of cross-attention from r1 to encoded top-K candidate items.
  - N is tuned over {1..5}; **N=1 is best**, with a slight decline at higher N. [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **RecGPT-V3:**
  - Up to 10 latent tokens: K = min(⌈N/C⌉, K_max) with C=20 and K_max=10.
  - They are "introduced as new vocabulary tokens whose embeddings are randomly initialized".
  - The text describes no hidden-state feedback. [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- **LaRec:**
  - K=6 continuous latent states z_1..z_K, each conditioned on z_<k. The text does not detail the wiring.
  - A lightweight "Reasoning Head" maps states into the input-embedding space.
  - Performance peaks at K=6. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- **HiLaR:**
  - K=4 sequential states, each sampled from a Gaussian policy conditioned on the history and the earlier states. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- **EvoReason:**
  - M latent tokens start from "primitive-aware slots" and are refined recurrently with z(t) = Proj(f_θ(...)) for t=1..T.
  - The text does not give M or T. [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- **S²GR:**
  - One continuous thinking token per SID level, t_l = Decoder(·) + p_l, placed before each SID code. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
- **IBA:**
  - A "Dual-Axis Refinement" combines horizontal recurrence with FiLM vertical semantic alignment.
  - Per-position budgets are set by a prior such as (3,2,1,0) with a per-user allocator.
  - Finding: earlier SID positions gain most from refinement, while "positions 3 and 4 are already near-saturated at Step 0". [FT] — [arXiv](https://arxiv.org/html/2607.12425)

### Inferences
- Several papers find the optimum at very few latent steps: LatentR3 uses N=1, IntuRec's best is N=1, and LASAR's RL drives the mean down to ~2.5 steps.
- That is consistent with latents acting as a small learned bias or extra query rather than as multi-step computation. No paper tests this directly.

### Gaps
- Exact wiring details are missing for LaRec (how z_k is fed back) and for EvoReason (M and T).

## Q3. Training signal

### Takeaway
Training signals fall into four patterns:
- **Recommendation loss only:** PauseRec, or as the SFT warm-up in most papers.
- **Teacher-CoT hidden-state alignment:** LASAR uses GPT-5 with symmetric KL; LaRec uses Qwen-72B with InfoNCE; EvoReason uses QwQ-32B with KV-L1 plus on-policy distillation; RecGPT-V3 uses DeepSeek-V3.2 with masked-segment reconstruction.
- **Structured-target alignment:** HiLaR uses MSE to quantized temporal preference codes; S²GR uses contrastive alignment to codebook centroids.
- **RL:** GRPO-style with Gaussian-noise exploration in latent space appears in LatentR3, FLR, LaRec and HiLaR. Token-level GRPO plus REINFORCE for depth appears in LASAR, and GRPO on SID accuracy in EvoReason.

### Cited Findings
- **LatentR3:**
  - SFT warm-up uses full-model next-token loss on the target.
  - Its "LR-GRPO" stage updates only LatentRATT with the LLM frozen, so the KL term is 0.
  - Exploration samples r^k = r + ε with ε~N(0,σ²).
  - The reward is negative perplexity of the ground-truth item, −exp(−mean log-prob).
  - Advantages use a batch-mean baseline with L2 normalisation.
  - The authors say RL from scratch "can lead to instability and potential collapse", which motivates the SFT warm-up. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **FLR:**
  - L_rec plus three regularisers: orthogonality (Gram matrix vs identity), attention diversity, and gate entropy, combined with uncertainty weighting.
  - GRPO with Gaussian noise on the thought embedding; the first sample in each group is unperturbed.
  - Reward is 0.1·mean log-prob + 1.0·exact match.
  - Asymmetric clipping (0.2, 0.28) and a reverse-KL penalty.
  - No teacher is used. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- **LASAR:**
  - SFT loss is CE + α·step-wise **symmetric KL between softmax-normalised hidden states** of latent step t and CoT segment t.
  - The CoT teacher is **GPT-5**. Segments are split with bge-small, and anchors are the same backbone's last-token hidden states, precomputed offline.
  - The Policy Head is trained with CE on the number of CoT segments.
  - The RL objective is GRPO (r_rule + r_NDCG, with KL to a reference) + terminal-only KL (h_N to the last anchor) + REINFORCE on N with an EMA baseline, a −λN cost and an entropy bonus. [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- **PauseRec:**
  - Three stages: continued pre-training (CPT); pause-embedding pretraining that trains only the `<pause>` embedding, inserted at 10% random positions; and implicit SFT with the loss masked at pause positions.
  - No rationales and no RL. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **IntuRec:**
  - Stage 1 is next-token CE.
  - Stage 2 is L_rec + λ·L_intu, a BPR-style log-sigmoid of the cosine difference.
  - No RL. [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **RecGPT-V3:**
  - Distils **DeepSeek-V3.2** CoT (~2.3–2.8k tokens; the paper's own figures are inconsistent).
  - The curriculum is warm-up CE on sequences with spans replaced by latent tokens, followed by multi-granularity masked-segment reconstruction.
  - Tag data supervises "only the output tokens to spare the latent positions from collapse".
  - RL uses GRPO with reward r_ctr (from the production ranker) × alignment, diversity and length gates. [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- **LaRec:**
  - Teacher: "a powerful reasoning model like Qwen-72B" writes target-oriented reverse CoT from the user history plus the ground-truth target.
  - Teacher states are the same LLM's final-layer hidden states on the CoT steps.
  - Losses are step-level **InfoNCE** alignment plus a process-direction hinge on cos(z_k − z_{k−1}, e_tgt), with α=γ=1.
  - RL is GRPO with reward 𝕀(hit) + λ·cos(Emb(pred), Emb(target)), λ=1.
  - Exploration is "anchored": offsets come from a Gaussian mixture centred on the user's historical item embeddings, σ=1.5. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- **HiLaR:**
  - SFT is L_rec + 0.1·MSE(Proj(τ^k), e_{u,k}). The targets e_{u,k} are cumulative residual-quantized codes of the frozen LLM's end-of-history hidden state, supervised by K temporal windows.
  - GRPO with G=6 and per-level Gaussian noise (0.12, 0.08, 0.05, 0.025).
  - Rewards: task rewards (hit/prefix/F1); a collaborative reward from a frozen **SASRec**; and a process reward R_align + λ_gain·Σρ_k·clip(Δ log-prob from adding level k).
  - KL coefficient 0.05. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- **EvoReason:**
  - GRPO with reward equal to token-level SID matching accuracy.
  - Confidence-gated on-policy distillation from a same-backbone teacher that sees **QwQ-32B** traces.
  - L1 alignment between R-KV-compressed teacher KV states and student latent KV states.
  - A primitive library (K=5) induced from ReAct-style trajectories and updated from latent-reasoner behaviour. [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- **S²GR:**
  - L_Rec + L_Think. L_Think combines per-level contrastive alignment to K-means centroids of the codebook, InfoNCE between a global vector and the target's averaged SID embedding, and a cosine tie.
  - No RL and no teacher. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
- **LARES:** self-supervised pre-training (trajectory- and step-level alignment) followed by RL post-training. [ABS] — [arXiv](https://arxiv.org/abs/2505.16865)

### Inferences
- LASAR's alignment ablation (Q5) is the only head-to-head comparison of alignment losses. Symmetric KL on softmaxed hidden states beats cosine (no gain) and MSE (harmful).

### Gaps
- Several hyperparameters are unreported: σ in LatentR3; τ1, β, ε and n in LaRec; ρ_k and b in HiLaR; the λ values in EvoReason.

## Q4. Is NEW information injected into the latent beyond the prompt?

### Takeaway
Yes in most post-2025 papers, which is the main axis of differentiation from LatentR3. LatentR3 and PauseRec inject nothing beyond the prompt and gradients from the target. Later papers add one of four things:
- **Teacher CoT hidden states:** LASAR, LaRec, EvoReason, RecGPT-V3.
- **The model's own top-K candidate items:** IntuRec.
- **CF signals:** HiLaR's SASRec reward; LaRec's history-anchored exploration.
- **Codebook structure:** S²GR, IBA.

### Cited Findings
- **LatentR3:** the latent is computed only from the prompt's hidden states. The training signal is target perplexity, and there is no teacher, retrieval or CF input. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **FLR:** "No teacher model is described". It needs no CoT annotations. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- **PauseRec:** no rationale supervision. Pause tokens share one embedding, so the only per-example content comes from attention over the prompt. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **LASAR:** GPT-5 CoT is distilled into hidden-state anchors. This is training-time only, and the teacher is "not used at inference". [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- **LaRec:**
  - Qwen-72B reverse CoT is conditioned on the **ground-truth target**. The authors assert the CoT contains no target information but do not verify it.
  - "Target-oriented vs non-target CoT" on Instruments H@10 is 0.1238 vs 0.1225.
  - RL exploration is anchored at the user's historical item embeddings. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- **IntuRec:**
  - It injects the LLM's **own top-K beam candidates**, the "self-intuition", into r1 via cross-attention. This is test-time extra computation, roughly a two-pass scheme.
  - SASRec candidates did worse: Recall@5 0.0900 vs 0.0986 on CDs.
  - Swapping in random negative candidates gives 0.0953, and low-popularity candidates give 0.0937 (full model 0.0986). [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **HiLaR:** temporal-window quantized preference codes serve as alignment targets, and a frozen SASRec provides the collaborative reward. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- **EvoReason:** QwQ-32B traces feed a teacher pass whose KV states supervise the student, plus a primitive library. [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- **RecGPT-V3:** DeepSeek-V3.2 CoT is distilled, and production-ranker CTR scores serve as the RL reward. [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- **S²GR:** codebook centroid structure is built from an item co-occurrence graph. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)

### Inferences
- Only IntuRec injects new *inference-time* information (extra candidate retrieval through a first beam pass). All teacher-based methods inject information only through training-time supervision. At inference the latent is still a function of the prompt alone, plus extra forward passes in the loop methods.

### Gaps
- No paper isolates "extra computation" from "extra information". For example, none tests a loop of the same depth without teacher alignment against one with it, while holding compute fixed. LASAR's no-alignment variant (+0.4%) comes closest.

## Q5. Datasets (Instruments flagged) and gain vs a no-latent ablation

### Takeaway
Amazon Instruments appears in LatentR3, FLR, HiLaR, LaRec, LASAR, IBA and IntuRec (appendix only). Reported gains over no-reasoning baselines are modest:
- Single-digit to ~17% relative on Toys/CDs.
- On Instruments: ~+2% (FLR's gain over LatentR3, +0.3% H@5) up to ~12% (LaRec vs no latent reasoning, H@10).
Several papers lack a clean same-architecture "w/o latent" ablation.

### Cited Findings
**LatentR3** [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- Datasets: Toys (6,299 items), CDs (5,841), Games (5,308), **Instruments (5,030 items, 66,500 train)**.
- BIGRec→+LatentR3 on Toys, H@5/N@5: 0.0701/0.0508 → 0.0821/0.0600. On CDs, H@5 goes 0.0757 → 0.0934. The average relative gain is 17.0% on BIGRec and 8.4% on D3.
- Ablation, Toys H@5: w/o Reasoning 0.0701; w/o RL (SFT only) 0.0804; full 0.0821.
- Ablation, CDs H@5: w/o Reasoning 0.0757; SFT-only 0.0830; full 0.0934.
- Inconsistency: the text says "all three datasets" although four are listed.
- My inference: on Toys, most of the gain over no reasoning comes already from SFT; RL adds +0.0017 H@5.

**FLR vs LatentR3** (H@5/H@10/N@5/N@10) [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- Instruments: 0.1029/0.1214/0.0882/0.0941 → 0.1032/0.1248/0.0886/0.0955. The smallest gain is H@5 +0.29%.
- Games: 0.0593/0.0889/0.0419/0.0515 → 0.0639/0.0908/0.0462/0.0548.
- There is no w/o-latent row.

**LASAR** [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- Datasets: Beauty, **Instruments**, Sports (5-core, leave-one-out).
- Instruments N@20: 0.0730 vs MiniOneRec 0.0677 (+7.8%).
- Beauty N@10/HR@10: 0.0303/0.0563 vs MiniOneRec 0.0295/0.0542.
- Ablation, Beauty N@10 against pure SFT at 0.0277:
  - Latent with **no alignment**: 0.0278 (+0.4%).
  - KL alignment: 0.0285.
  - Cosine alignment: 0.0277.
  - MSE alignment: 0.0245 (−11.6%).
  - Plus RL with terminal KL and REINFORCE: 0.0303, with mean N falling from 4.20 to 2.47.
- Sports HR@10: adaptive depth 2.80% vs fixed N=4 1.93%.
- Cost: ~0.29 vs 0.27 GPU-sec/sample against MiniOneRec, and about 20× faster than CoT generation (7.0 s).

**PauseRec** [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- Datasets: Beauty, Sports, Toys. **No Instruments.**
- Beauty H@5 vs next-item SFT: 0.0568 vs 0.0533. It "improves every metric over the next-item SFT baseline".
- It beats OneRec-Think (explicit CoT + RL) on 10 of 12 metrics, up to +6.22% (Toys H@5 0.0615 vs 0.0579).
- Explicit CoT without RL does *not* beat SFT: 0.0524 vs 0.0533.

**IntuRec** [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- Datasets: CDs, Toys, Games. **Instruments** and Books appear only in Appendix A.
- Instruments Recall@5: LatentR3-B 0.0251 vs IntuRec-B 0.0332. These numbers are far below other papers' Instruments numbers, which suggests a different split or metric.
- CDs ablation: w/o IRI 0.0822 vs full 0.0986.

**LaRec** [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- Datasets: Toys, **Instruments**, MovieLens-100K, proprietary Industry.
- Instruments H@10/N@10:
  - Full: 0.1238/0.0963.
  - **w/o latent reasoning: 0.1106/0.0893**.
  - w/o step alignment (SA): 0.1142.
  - w/o process alignment (PA): 0.1190.
  - w/o RL: 0.1229.
  - w/o anchored exploration (PGD): 0.1233.
- Toys w/o LR: 0.0900/0.0566 vs 0.1073/0.0695.
- Industry H@10: 0.3654 → 0.4090.
- Online A/B: conversion +2.93%.

**HiLaR** [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- Instruments H@5/H@10/N@5/N@10: HiLaR 0.1129/0.1320/0.0976/0.1038 vs LatentR3 0.1032/0.1241/0.0894/0.0981.
- Ablations appear only in figures, with no w/o-latent variant.

**EvoReason** [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- Datasets: Beauty, Sports, Industrial. **No Instruments.**
- Beauty R@10: EvoReason 0.0919, LASAR 0.0829, LatentR3 0.0776.
- There is no w/o-latent ablation.

**S²GR** [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
- Beauty HR@10: 0.0785 vs w/o Reason 0.0736, and vs TIGER 0.0728.
- **No Instruments.**

**IBA** [FT] — [arXiv](https://arxiv.org/html/2607.12425)
- **Instruments** R@10/N@10: TIGER 0.0853/0.0629 → IBA 0.0975/0.0725. This is not a same-architecture zero-step control.

**RecGPT-V3** [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- HR@30 on an unnamed test set:
  - No reasoning: 0.3050.
  - Explicit CoT SFT: 0.3508.
  - Latent: 0.3462.
  - Plus RL: 0.3693.

### Inferences
- Cross-paper numbers for the "same" Instruments setting do not agree. LatentR3 as reported by FLR has H@5 0.1029; as reported by HiLaR, 0.1032; LaRec's TallRec baseline has H@5 0.0926. Re-implementation variance in this literature is about the same size as the claimed gains.
- On Instruments specifically, the gains of latent variants over LatentR3 are smallest. FLR attributes this to a "lower-dimensional intent space".

### Gaps
- Most papers report no seed variance. Significance tests appear in LaRec (p<0.001), HiLaR (p<0.05) and LASAR (bootstrap; Beauty only marginal).

## Q6. Do they test whether predictions depend on latent CONTENT at inference? Collapse reports?

### Takeaway
**No paper in this set runs a direct inference-time intervention on latent content**: none swaps latents across users, zeroes them, adds noise, or shuffles them at test time. The closest evidence is indirect:
- K=0 / "w/o reasoning" retraining ablations (LatentR3, LaRec, S²GR).
- Source swaps of the injected *input* rather than the latent (IntuRec).
- Qualitative attention maps (PauseRec).
- Noise used only as RL exploration.
Collapse is mentioned as a motivation (LatentR3, RecGPT-V3) or as factor or attention correlation (FLR), but never measured as "the model ignores the latent".

### Cited Findings
- **LatentR3:**
  - No swap, perturbation, zeroing or noise tests at inference. Noise appears only as RL exploration.
  - It mentions that RL from scratch "can lead to instability and potential collapse", without measurement. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **FLR:**
  - No swap or zeroing test.
  - Collapse is discussed as factor degeneration: without the regularisers, average |corr| between factors is ≈0.44 vs 0.03 with them, which "visually corroborates the phenomenon of global degeneration".
  - It describes "mode collapse" of attention heads in the pre-trained baseline. [FT] — [arXiv](https://arxiv.org/html/2604.26760v1)
- **LASAR:**
  - No swap, zeroing or noise experiments.
  - Representation drift is inferred only indirectly: latent reasoning with no alignment gives +0.4%, and MSE alignment hurts by −11.6%. [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- **PauseRec:** "does not replace, shuffle, or remove pause content at inference". It offers one qualitative attention example, in which later pauses "focus on a smaller set of historical SID tokens". There is no collapse analysis. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **IntuRec:** it swaps the *intuition source* (random negatives 0.0953, low-popularity 0.0937 vs 0.0986). It does not zero, noise or swap the latent itself. [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **LaRec:**
  - The only perturbation-like test replaces anchored exploration with Gaussian noise during training: 0.1233 vs 0.1238.
  - No zeroing, swapping or shuffling of latent states. [FT] — [arXiv](https://arxiv.org/html/2607.24617v1)
- **HiLaR:** "None are reported". Its latent analysis is limited to layer-wise gains and history-length buckets. [FT] — [arXiv](https://arxiv.org/html/2607.27760)
- **EvoReason:** none reported. [FT] — [arXiv](https://arxiv.org/html/2607.29010)
- **RecGPT-V3:** none reported. It supervises "only the output tokens to spare the latent positions from collapse", but never measures collapse. Decodability of latents back to text is shown only in a qualitative case study. [FT] — [arXiv](https://arxiv.org/html/2607.15591)
- **S²GR:**
  - None reported.
  - It claims that without L_Think the thinking tokens "degenerate into latent states without explicit meaning", with Beauty HR@10 0.0740 vs 0.0785, but does not test this directly.
  - Its collapse analysis concerns the *codebook* (CUR/ICR), not the thinking tokens. [FT] — [arXiv](https://arxiv.org/html/2601.18664v1)
- **IBA:**
  - None reported.
  - It notes that extra refinement at late SID positions may "disturb reliable representations".
  - The "w/o Predictor" variant "remains competitive" (R@10 0.0958 vs 0.0975). [FT] — [arXiv](https://arxiv.org/html/2607.12425)

### Inferences
- A content-dependence test (swap latents across users, zero them, or replace them with the batch mean at inference) is a clear open gap across the whole sub-field.
- Several findings would be consistent with partly ignored or near-constant latents, though none establishes it:
  - (a) Optimal N=1 in LatentR3 and IntuRec.
  - (b) LASAR's no-alignment latent loop gives only +0.4%.
  - (c) PauseRec gains come from a single shared embedding, so per-example content arises only through attention.
  - (d) Small RL deltas on top of SFT, e.g. LatentR3 Toys at +0.0017 H@5 and LaRec w/o RL at −0.0009 H@10.

### Gaps
- I did not read the appendices of LASAR (~42k characters unread) or RecGPT-V3 (~20k unread), so an intervention study there cannot be fully ruled out. None appeared in the main text.

## Q7. Stated limitations

### Takeaway
The authors' stated limitations centre on interpretability (latents are unreadable), small datasets and backbones, teacher dependence, and the sequential cost of hidden-state loops. Many papers have no limitations section.

### Cited Findings
- **LatentR3:** "relatively small datasets" because of cost; latent reasoning is less interpretable than explicit CoT. [FT] — [arXiv](https://arxiv.org/html/2505.19092)
- **LASAR:** Coconut-style loops "prevent teacher forcing", so forward passes are sequential; it relies on a teacher LLM for offline CoT; future work is listed as more efficient loops and conversational recommendation. [FT] — [arXiv](https://arxiv.org/html/2605.10207)
- **PauseRec:** design not exhaustively tuned; offline evaluation only; pauses are "less directly readable"; possible popularity bias; a single backbone; small absolute gains with no seed variance. [FT] — [arXiv](https://arxiv.org/html/2606.14142)
- **IntuRec:** plans larger datasets and alternative intuition generators; projecting traditional-model candidates is "challenging"; training is slower than BIGRec (0.6191 vs 0.3559 s/it). [FT] — [arXiv](https://arxiv.org/html/2606.27684)
- **IBA:** future work is adaptive depth by item semantics or prefix uncertainty. As limitations, it uses only T5-small and a single seed (42). [FT] — [arXiv](https://arxiv.org/html/2607.12425)
- **No limitations section** in LaRec, FLR, HiLaR, EvoReason, S²GR or RecGPT-V3, per the full-text extraction. [FT] — [LaRec](https://arxiv.org/html/2607.24617v1); [FLR](https://arxiv.org/html/2604.26760v1); [HiLaR](https://arxiv.org/html/2607.27760); [EvoReason](https://arxiv.org/html/2607.29010); [S²GR](https://arxiv.org/html/2601.18664v1)
- **RecGPT-V3 efficiency claim:** the abstract claims "200×" token-cost reduction, but the measured output went from 2,840 to 122 tokens (~23×) and time from 1,020 s to 295 s (3.46×). [FT] — [arXiv](https://arxiv.org/html/2607.15591)

### Inferences
- Recurring unstated weaknesses:
  - A single 1.5B or 1.7B backbone.
  - No seeds or variance reported.
  - Inconsistent baseline numbers across papers.
  - Teacher CoT conditioned on the target (LaRec), which risks label leakage that was never checked.
  - No latent-content intervention tests.

### Gaps
- No OpenReview reviews were found for any of these papers, so reviewer-identified limitations are unknown.
