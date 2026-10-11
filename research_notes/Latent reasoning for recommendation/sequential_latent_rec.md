# Implicit / Latent Multi-Step Reasoning in Non-LLM Sequential Recommenders (2025–2026): Per-Paper Dossier

Verification legend: **[FT]** = read from arXiv full text (HTML), using an extraction model over the page, so the numbers should be spot-checked against the PDF tables before they are quoted in a paper. **[ABS]** = abstract/metadata only. Every paper here is an arXiv preprint unless a venue is stated. LARES, PLR and DiffuReason show an unfilled ACM template venue. RecRec states RecSys 2026.

Papers covered:
- **Core (ID-based / SASRec-style, full text read):** ReaRec (ERL/PRL), LARES, PLR, ManCAR, DiffuReason, RecRec.
- **Adjacent (full text read):** STREAM-Rec, a T5-style generative model with residual reasoning traces.
- **Adjacent (abstract only):** OnePiece (industrial), DS-Frame, LASAR (LLM/semantic ID), IBA "Where Reasoning Matters" (semantic ID).
- **Excluded as LLM-based:** LaRec, Hierarchical Latent Reasoning, and Intuition-Guided Latent Reasoning.

## How latent reasoning steps are produced, and how step count scales

### Takeaway
There are four main patterns:
1. **Autoregressive last-state feedback.** Used by ReaRec, PLR (with parallel streams), ManCAR and DiffuReason.
2. **Depth recurrence over all tokens.** Used by LARES.
3. **A separate recursive reasoning state over a set of latent interests.** Used by RecRec.
4. **Residual-to-target pseudo-token traces.** Used by STREAM-Rec.

Almost every fixed-depth method peaks at about 2–4 steps. Only RecRec (which supervises 9 steps) and ManCAR (adaptive halting) push past that.

### Cited Findings
**ReaRec (Tang et al., arXiv 2503.22675) [FT]**
- The last-position final-layer output is fed back as the next input token, plus a learned K×d reasoning position embedding (Eq. 4). KV caching keeps the cost low. — [ReaRec](https://arxiv.org/html/2503.22675)
- Without the reasoning position embeddings ("Naive"), the model does not beat the base. — [ReaRec](https://arxiv.org/html/2503.22675)
- The best K is typically 2 (Table 4 note: "Step-2"). Performance drops for all variants beyond about 2 steps (Fig. 5). — [ReaRec](https://arxiv.org/html/2503.22675)
- Latency on SASRec is +3.51% at Step-2 (5.6761 s → 5.8752 s, A100) and +10.61% at Step-5. — [ReaRec](https://arxiv.org/html/2503.22675)

**LARES (Liu, Zheng, Wang, Zhao, Wen + Meituan; arXiv 2505.16865) [FT]**
- A pre-block maps item and position embeddings to a hidden sequence H. A shared core block is then applied recurrently to all token states: T_i = C(LN(f(T_{i-1}, H))). H is re-injected at every step, T_0 is Gaussian noise, and the output is T_k[-1]. — [LARES](https://arxiv.org/html/2505.16865)
- During training, depth k is sampled from a log-normal-Poisson distribution (σ=0.5, mean ≈ k̄+1), truncated at 3k̄. Inference uses a fixed k̄+1. — [LARES](https://arxiv.org/html/2505.16865)
- In the sweep over {1,2,3,4,6,8}, performance rises then falls. The best depth is 4 on Instrument and 3 on Scientific. — [LARES](https://arxiv.org/html/2505.16865)

**PLR – Parallel Latent Reasoning (Tang, Chen et al., RUC + Alibaba; arXiv 2601.03153) [FT]**
- Width scaling: M learnable trigger tokens perturb the last state (h0 + τ_m), and each stream reasons T steps in an isolated attention mask, ReaRec-style. — [PLR](https://arxiv.org/html/2601.03153)
- Each stream's steps are mean-pooled. A gated mixture (MoRS), conditioned on h0, aggregates the streams. Inference uses h0 + z_rea, while training uses z_rea only. — [PLR](https://arxiv.org/html/2601.03153)
- Both M and T peak at 2 when grid-searched over 1–5. — [PLR](https://arxiv.org/html/2601.03153)

**ManCAR (Yang et al., arXiv 2602.20093) [FT]**
- Encoder setup as in ReaRec: 2-layer pre-norm Transformer. A shared reasoning module produces each state from the history and earlier states. — [ManCAR](https://arxiv.org/html/2602.20093)
- Each state is rescaled to the mean item-embedding norm. — [ManCAR](https://arxiv.org/html/2602.20093)
- Adaptive halting stops when KL(p_{t-1} ‖ p_t) < ε. — [ManCAR](https://arxiv.org/html/2602.20093)
- Train steps vs. average inference steps: CDs 5 / 1.84, Arts 1 / 1, Toys 4 / 3.58, Grocery 2 / 1.74. — [ManCAR](https://arxiv.org/html/2602.20093)

**DiffuReason (Jiang et al., Tencent; arXiv 2602.09744) [FT]**
- The first "thinking token" is the backbone's final hidden state. Later tokens come from a 2-layer decoder that attends to the backbone states, and they are pooled into a condition vector. — [DiffuReason](https://arxiv.org/html/2602.09744)
- A diffusion refiner (16 denoising steps) then outputs a Gaussian anchor, and G=8 samples are drawn around it. — [DiffuReason](https://arxiv.org/html/2602.09744)
- Default CoT length is 3 (tested 1–5). It peaks at 3, then saturates or degrades. Backbones tested: SASRec, HSTU, TIGER. — [DiffuReason](https://arxiv.org/html/2602.09744)

**RecRec (Deng, Fu et al., RecSys 2026; arXiv 2607.12945) [FT]**
- A context compressor distils backbone states into R=8 latent interests M. A separate reasoning state S (R×d) is updated over n=3 inner passes per step, and M is updated via a GRU gate. — [RecRec](https://arxiv.org/html/2607.12945v1)
- S is never used for prediction. Scoring uses log-mean-exp over the interests. — [RecRec](https://arxiv.org/html/2607.12945v1)
- Training uses 9 steps (3 supervision rounds × 3). — [RecRec](https://arxiv.org/html/2607.12945v1)
- Inference depth is free, and the best N extrapolates past 9: 15 on S-Shop, 12 on V-Shop, 16 on Steam. On MicroLens it peaks around N≈4 and then declines. — [RecRec](https://arxiv.org/html/2607.12945v1)

**STREAM-Rec (Zhang et al., RUC + Tencent; arXiv 2504.09627) [FT]**
- Backbone is a T5-style encoder-decoder (4+4 layers, d=256), not SASRec. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
- Each step computes r_i = MLP(t − s_{i−1}), the residual to the target. The nearest codebook token becomes a pseudo-label "thinking" token. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
- Length is fixed: l=6 on Instruments, l=4 on Scientific. There is no sweep. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)

**OnePiece (Shopee; arXiv 2509.18091) [ABS]**
- Pure-Transformer industrial retrieval and ranking with "block-wise latent reasoning", where reasoning bandwidth is set by block size. — [OnePiece](https://huggingface.co/papers/2509.18091)

**DS-Frame, "Recommender System as Slow and Fast Thinkers" (arXiv 2609.02671) [ABS]**
- A learned selector routes each sample between a fast path and a slow path of "iterative latent refinement", under a compute budget. — [DS-Frame](https://arxiv.org/abs/2609.02671)

**LASAR (arXiv 2605.10207) [ABS; LLM/semantic ID]**
- A policy head predicts per-sample latent depth. This "nearly halves" the average step count. — [LASAR](https://arxiv.org/abs/2605.10207)

**IBA, "Where Reasoning Matters" (arXiv 2607.12425) [ABS; semantic ID]**
- Allocates latent refinement steps across semantic ID positions by information gain. Earlier positions carry more information gain and benefit more. — [IBA](https://arxiv.org/abs/2607.12425)

### Inferences
- The "best depth ≈ 2–4" pattern is robust across ReaRec, LARES, PLR and DiffuReason. Fixed-depth latent reasoning in ID-based seq-rec currently gives "a couple of extra refinement passes", not long chains.
- ManCAR's average of about 1.7–1.8 inference steps on CDs and Grocery reinforces this.
- RecRec is the only paper reporting gains well past the trained depth. Its reasoning state is decoupled from the prediction state, which is plausibly why.

### Gaps
- OnePiece and DS-Frame mechanism details were not verified from full text.
- The ε value in ManCAR is not reported in the portion read.

## Training signal and anti-collapse / diversity mechanisms

### Takeaway
All methods use next-item cross-entropy. Collapse is fought with four kinds of tool:
- **Divergence between step distributions:** pairwise KL in ReaRec-ERL and PLR.
- **Step-indexed temperature and contrastive noise:** ReaRec-PRL.
- **Contrastive trajectory or step alignment plus GRPO:** LARES.
- **External priors or diversity regularisers:** ManCAR's graph teacher, RecRec's interest-diversity regulariser.

DiffuReason is the exception: it found that adding KL hurts.

### Cited Findings
**ReaRec-ERL**
- The user representation is the mean of step outputs r_0..r_K. Loss is L_Rec + λ·L_KL, where L_KL is a pairwise KL between step item-distributions that pushes them apart (λ ∈ {0.001…0.1}). — [ReaRec](https://arxiv.org/html/2503.22675)
- The authors name the failure mode explicitly: the model can "take shortcuts by directly copying the previous reasoning output", causing "pattern collapse". — [ReaRec](https://arxiv.org/html/2503.22675)

**ReaRec-PRL**
- Each step k gets its own cross-entropy loss with temperature τ_k = τ·α^(K−k), which sharpens progressively. Inference uses only r_K. — [ReaRec](https://arxiv.org/html/2503.22675)
- Reasoning-aware contrastive learning adds Gaussian noise (γ=0.01) to reasoning inputs with an InfoNCE loss. — [ReaRec](https://arxiv.org/html/2503.22675)
- α=1.0 (no annealing) leads to collapse, and α=10 degrades performance. — [ReaRec](https://arxiv.org/html/2503.22675)

**LARES**
- Self-supervised pre-training: L_Rec + α·L_TLA + γ·L_SLA. — [LARES](https://arxiv.org/html/2505.16865)
  - Trajectory-level alignment (TLA) is InfoNCE between two trajectories of equal length. Mismatched lengths made the model degrade toward short-chain reasoning.
  - Step-level alignment (SLA) aligns a random intermediate step with the final step.
- RL post-training uses GRPO with G=4 rollouts, an NDCG/Recall reward on the last step, and KL β ∈ {0.5, 1}. — [LARES](https://arxiv.org/html/2505.16865)
- RL data is filtered to drop cases where the target is not in the top 100. — [LARES](https://arxiv.org/html/2505.16865)
- There is no explicit collapse analysis. — [LARES](https://arxiv.org/html/2505.16865)

**PLR**
- Loss: L_NIP + L_RCL + λ·L_KL. — [PLR](https://arxiv.org/html/2601.03153)
  - The global pairwise KL runs over all T×M states (λ=0.1 best).
  - The RCL term uses dropout views of hidden states and attention scores.
  - Anti-collapse also comes from perturbed trigger initialisations and stream isolation.
- Ablation on SASRec/CDs, R@20: full 0.0873; w/o MoRS 0.0785; w/o RCL 0.0782; w/o KL 0.0853. — [PLR](https://arxiv.org/html/2601.03153)

**ManCAR**
- At every step, a KL term pulls the model distribution toward a teacher prior over candidates. Candidates come from a Swing item graph around recent items, and the teacher sharpens with step index (γ decreasing). — [ManCAR](https://arxiv.org/html/2602.20093)
- τ grows with step, the opposite direction to ReaRec. A decreasing schedule was worse: NDCG@10 0.0273 vs. 0.0282 on CDs. — [ManCAR](https://arxiv.org/html/2602.20093)
- Ablation without the teacher prior: CDs 0.0212 vs. 0.0282 for the full model. — [ManCAR](https://arxiv.org/html/2602.20093)
- Caveat found in the text: the teacher ranks the ground-truth target first during training, so the prior is label-informed. — [ManCAR](https://arxiv.org/html/2602.20093)

**DiffuReason**
- Losses: cross-entropy on the anchor, MSE from the anchor to the target embedding, and a GRPO-style loss with a binary Hit@1 reward. — [DiffuReason](https://arxiv.org/html/2602.09744)
- There is no KL term. Adding KL (weight 0.1) hurt, and MSE weight α=2 suppressed exploration. — [DiffuReason](https://arxiv.org/html/2602.09744)

**RecRec**
- Two stages: train the backbone and compressor, then train the reasoner only with per-round cross-entropy, detached rounds and 1-step BPTT. — [RecRec](https://arxiv.org/html/2607.12945v1)
- The interest diversity regulariser penalises off-diagonal cosine similarity among interests. It cuts inter-interest cosine from 0.82–0.90 to 0.14–0.75. — [RecRec](https://arxiv.org/html/2607.12945v1)
- Other stabilisers: a GRU gate and a zero-initialised reasoner. — [RecRec](https://arxiv.org/html/2607.12945v1)

**STREAM-Rec**
- Training stages, in order: pretraining → SFT on residual-derived traces (with DPO against a one-step decoder plus a state-target contrastive loss) → GRPO. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
- GRPO rewards cover format, exact match, in-batch rank, thinking likelihood, and thinking ranking against 50 hard negatives. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)

**LASAR [ABS]**
- Step-wise bidirectional KL to hidden-state anchors from text CoT, against drift. — [LASAR](https://arxiv.org/abs/2605.10207)

### Inferences
- The field openly treats collapse/homogenisation of steps as the central failure mode. The main remedies are explicit diversity pressure (KL apart, cosine penalties) or an external target (graph prior, MSE to the target).
- Neither kind of remedy shows that the step content itself is causally used.

### Gaps
- No paper reports self-distillation between steps as the main signal. STREAM-Rec's DPO against a one-step reference is the closest.
- No paper reports an ablation of KL "apart" vs. KL "together".

## Mechanisms that add NEW information or computation beyond the input sequence

### Takeaway
Most methods add only computation (extra passes with shared weights). ManCAR is the clearest case of genuinely new information during reasoning, via a global item graph. DiffuReason adds stochasticity, and the RL methods add reward signal at training time only.

### Cited Findings
- **ReaRec, LARES, PLR, RecRec:** the reasoning conditions only on the input history plus learned embeddings (reasoning position embeddings, trigger tokens, a random S0). — [ReaRec](https://arxiv.org/html/2503.22675); [LARES](https://arxiv.org/html/2505.16865); [PLR](https://arxiv.org/html/2601.03153); [RecRec](https://arxiv.org/html/2607.12945v1)
- **ManCAR** injects collaborative-neighbourhood candidates from a global (Swing) interaction graph as extra context, plus a graph-based teacher. — [ManCAR](https://arxiv.org/html/2602.20093)
  - Removing the context drops CDs NDCG@10 from 0.0282 to 0.0234.
  - A fair control, ContextBERT4Rec (BERT4Rec given the same graph context), scores 0.0182, so the gain is not just the extra context.
- **DiffuReason** uses only the interaction history. External LLM embeddings (Qwen3-Embedding-8B) appear only in semantic ID construction. — [DiffuReason](https://arxiv.org/html/2602.09744)
- **STREAM-Rec** pretrains with targets padded by two retrieved similar items, and reasoning traces are built using the target (training only). — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
- **ReaRec** with UniSRec/MoRec backbones uses LLaMA-3.1-8B text embeddings (PCA to 768). This is a backbone input, not the reasoning step. — [ReaRec](https://arxiv.org/html/2503.22675)

### Inferences
- The reported gains of pure-computation methods are modest. ReaRec averages about 7.5%, and LARES and PLR are typically +2–7% over the best baseline.
- The method with an external information source reports far larger gains: ManCAR claims up to +46.88% NDCG@10. That is consistent with the hypothesis that "more passes over the same information" has limited headroom.
- ManCAR's training-time label leak in the teacher, and its unclear inference candidate construction, need checking.

### Gaps
- No paper found adds retrieval or memory of other users' outcomes at inference beyond ManCAR's graph neighbourhood.
- OnePiece's "structured context engineering" (industrial side features) was not verified in full text.

## Datasets (Musical Instruments flagged), gain vs zero-step, and heterogeneity by popularity / length / user

### Takeaway
Musical Instruments is used by LARES, ManCAR ("Music"), DiffuReason and STREAM-Rec, but not by ReaRec, PLR or RecRec. On Instruments, latent reasoning gains are small, and ManCAR is beaten by ReaRec-PRL on R@10.

ReaRec's heterogeneity claim is verified only partly. Gains grow with steps for short-sequence users and unpopular (long-tail) items, but decline for long-sequence users and popular items ("overthinking"). DS-Frame [ABS] claims the opposite direction for longer histories.

### Cited Findings
**Datasets per paper**
- **ReaRec:** Yelp; Amazon 2023 Video & Games, Software, CDs & Vinyl, Baby & Products. No Instruments. — [ReaRec](https://arxiv.org/html/2503.22675)
- **LARES:** Amazon 2023 Instrument (57,439 users / 24,587 items / 511,836 actions), Video Games, Baby, Industrial & Scientific. Uses max length 20 and average length ≈ 8–9. — [LARES](https://arxiv.org/html/2505.16865)
- **PLR:** CDs & Vinyl, Movies & TV, Video & Games. No Instruments. — [PLR](https://arxiv.org/html/2601.03153)
- **ManCAR:** CDs, Video, Office, Arts, Grocery, Toys, Musical Instruments (labelled "Music"). — [ManCAR](https://arxiv.org/html/2602.20093)
- **DiffuReason:** Beauty, Instruments, Sports, Video & Games. — [DiffuReason](https://arxiv.org/html/2602.09744)
- **RecRec:** S-Shop, V-Shop, MicroLens-100K, Steam. — [RecRec](https://arxiv.org/html/2607.12945v1)
- **STREAM-Rec:** Instruments and Scientific. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)

**Gains on Instruments**
- **LARES vs. best baseline:** NDCG@10 0.0336 vs. 0.0314 (+7.0%), R@10 0.0636 vs. 0.0598 (+6.4%). — [LARES](https://arxiv.org/html/2505.16865)
  - A no-alignment, no-RL ablation scores 0.0290 NDCG@10.
- **DiffuReason-SASRec:** only +3.73% R@5 on Instruments, vs. +19.64% on Beauty. — [DiffuReason](https://arxiv.org/html/2602.09744)
- **ManCAR on Music:** R@10 is 0.0217 vs. PRL 0.0225, i.e. −3.56% vs. ReaRec-PRL. — [ManCAR](https://arxiv.org/html/2602.20093)
- **STREAM-Rec vs. TIGER:** R@5 +19.52%, N@10 +17.92%. However, SASRec's R@5 (0.0352) already beats TIGER's (0.0333). — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)

**Gains vs. zero-step / no reasoning**
- **ReaRec:** about 7.49% average across metrics. SASRec gains are ERL +6.76% and PRL +8.21%; UniSRec gains are ERL +12.29% and PRL +10.43%. — [ReaRec](https://arxiv.org/html/2503.22675)
  - Gains are not uniform. Baby SASRec-ERL is −2.16%, and Video UniSRec-PRL is −0.08%.
- **PLR, SASRec R@20 zero-step → PLR:** CDs 0.0762 → 0.0873, Movies 0.0664 → 0.0701, Video 0.0937 → 0.1033. — [PLR](https://arxiv.org/html/2601.03153)
  - Against the best baseline, several cells are negative, e.g. SASRec/CDs N@10 −0.37% and UniSRec/Video N@10 −3.40%.
- **RecRec, NDCG@20 at N=0 → N=1:** S-Shop 0.0966 → 0.103, Steam 0.0496 → 0.058. Most of the gain comes in the first steps. — [RecRec](https://arxiv.org/html/2607.12945v1)
- **DiffuReason, no-CoT → full (R@10):** Beauty 0.1013 → 0.1117, Sports 0.0807 → 0.0931. — [DiffuReason](https://arxiv.org/html/2602.09744)

**Heterogeneity by user, sequence length and popularity**
- **ReaRec (Yelp, Fig. 4):** short-sequence users and unpopular items keep improving as steps increase. For example, item group IG-1 gains 12.08%, 16.35%, 18.69% across steps. Long-sequence users and popular items decline with more steps. — [ReaRec](https://arxiv.org/html/2503.22675)
- **DiffuReason:** sparse users benefit most from CoT (UG-0 +17.13% on Sports). For the least-popular items, step 2 is 4.50% worse than step 1, and diffusion recovers this to a +22.49% overall gain. — [DiffuReason](https://arxiv.org/html/2602.09744)
- **DS-Frame [ABS]:** gains are larger on "users with longer histories or less mainstream item profiles". — [DS-Frame](https://arxiv.org/abs/2609.02671)
- **No subgroup analysis:** LARES, PLR, RecRec, ManCAR (density discussed only qualitatively) and STREAM-Rec. — [LARES](https://arxiv.org/html/2505.16865); [PLR](https://arxiv.org/html/2601.03153); [RecRec](https://arxiv.org/html/2607.12945v1); [ManCAR](https://arxiv.org/html/2602.20093)

### Inferences
- The "gains concentrated on long-tail/hard users" premise holds for ReaRec in the sense of short-history users and unpopular items, and DiffuReason partly corroborates the sparse-user part.
- It is not evidence that reasoning helps "hard" users in general. Long-history users get hurt by extra steps in ReaRec.
- On Instruments specifically, latent reasoning margins over strong baselines are small (≈ 4–7%), and they are not consistent across methods.

### Gaps
- No paper reports per-step gains by popularity on Instruments.
- ReaRec does not report a "hard user" (by baseline loss) segmentation.
- The precise ReaRec Fig. 4 numbers for user groups are only partially available in the text.

## Do steps degrade, collapse, or actually matter? (similarity, perturbation, swapping; when more steps hurt)

### Takeaway
Step homogenisation and collapse is documented, most directly by ReaRec's similarity heatmaps and PLR's theory, and "more steps hurt beyond 2–4" is near-universal.

**No paper found performs causal tests of step content** (perturbing, swapping or zeroing reasoning states at inference to show that predictions depend on them).

### Cited Findings
**ReaRec**
- Fig. 9: with reasoning position embeddings only, steps are highly homogeneous. The final output's similarity to the previous two steps is about 1.00 and 0.98. PRL and ERL reduce this. — [ReaRec](https://arxiv.org/html/2503.22675)
- Fig. 10: ERL without KL shows more overlap across steps 0–3. — [ReaRec](https://arxiv.org/html/2503.22675)
- Beyond about 2 steps, performance drops. The authors attribute this to "overthinking" for long-sequence users and popular items. — [ReaRec](https://arxiv.org/html/2503.22675)
- An oracle that picks the best step per user raises the ceiling by "approximately 30%–50%" (Fig. 2, Yelp, K=2). In other words, different steps are right for different users, but the model cannot select among them. — [ReaRec](https://arxiv.org/html/2503.22675)

**PLR**
- Theory claims that excessive depth collapses all streams to a common fixed point under Lipschitz contraction ("diversity decay"). The Lipschitz constant is never measured, and the paper has no direct similarity measurement. — [PLR](https://arxiv.org/html/2601.03153)
- The oracle-selection ceilings of PLR and ReaRec are comparable (NDCG@20 ≈ 0.0402–0.0416). — [PLR](https://arxiv.org/html/2601.03153)

**LARES**
- Rise-then-fall with depth, called "overthinking" on simple sequences. — [LARES](https://arxiv.org/html/2505.16865)
- Mismatched-length TLA degrades the model toward short-chain reasoning. — [LARES](https://arxiv.org/html/2505.16865)

**ManCAR**
- Motivated by "latent drift" off the item manifold. Fig. 6 shows adjacent-step KL dropping sharply, i.e. the predictive distribution converges, which is used for halting. Drift is not measured directly. — [ManCAR](https://arxiv.org/html/2602.20093)

**DiffuReason**
- Peaks at 3 steps. The authors attribute decline to latent noise. — [DiffuReason](https://arxiv.org/html/2602.09744)

**RecRec**
- Critiques ReaRec and LARES for collapsing the prediction into a single d-dimensional vector, with reasoning sharing space with the history. — [RecRec](https://arxiv.org/html/2607.12945v1)
- Stable past training depth on 3/4 datasets, but declines after N≈4 on MicroLens. — [RecRec](https://arxiv.org/html/2607.12945v1)
- No step-similarity or perturbation analysis. — [RecRec](https://arxiv.org/html/2607.12945v1)

**Robustness test (PLR)**
- PLR degrades least when 10–30% of test interactions are removed (CDs). This is a test of input perturbation, not of the reasoning states. — [PLR](https://arxiv.org/html/2601.03153)

### Inferences
- Together, ReaRec's near-1.0 step similarity, ManCAR's converging distributions and PLR's fixed-point argument suggest that latent steps tend to converge to a fixed point of the shared block. Extra steps are then mostly redundant refinements.
- The large oracle gap in ReaRec suggests that the information to pick better items exists across steps, but it is not exploited.
- A perturbation or step-swap study would be a genuinely novel contribution.

### Gaps
- None of the papers read reports interventions on step content, cross-user step swapping, or shuffling the order of reasoning states.
- No paper compares against a same-FLOPs deeper non-recurrent transformer. A third-party review flags this for LARES. — [Pith review](https://pith.science/paper/2505.16865) (secondary; not read in full)

## Limitations and failure analyses

### Takeaway
The limitations recur across papers: fixed depth (adaptive halting is future work), overthinking on easy users and items, shared encoder/reasoner parameters, no causal analysis, small or few Amazon datasets, no variance reporting, and theory with unverified assumptions.

### Cited Findings
- **ReaRec:**
  - Self-stated limitations: needs adaptive depth; parameter sharing between encoding and reasoning causes "task ambiguity"; no inference-time scaling law; no theory; it is "an initial exploratory effort". — [ReaRec](https://arxiv.org/html/2503.22675)
  - Negative cells: Baby SASRec-ERL −2.16%. — [ReaRec](https://arxiv.org/html/2503.22675)
- **LARES:**
  - Fixed inference depth, and outcome-only RL reward (process reward is future work). — [LARES](https://arxiv.org/html/2505.16865)
  - About 10% slower than SASRec at matched FLOPs. — [LARES](https://arxiv.org/html/2505.16865)
  - Uses ERL/PRL as baselines but does not cite them as a separate "ReaRec" row. — [LARES](https://arxiv.org/html/2505.16865)
- **PLR:**
  - Ablations only on SASRec/CDs, and some cells regress vs. the best baseline. — [PLR](https://arxiv.org/html/2601.03153)
  - Train/inference output mismatch (z_rea vs. h0 + z_rea). — [PLR](https://arxiv.org/html/2601.03153)
  - Theorem 4.4 is stated as an equality but proven only as a bound. — [PLR](https://arxiv.org/html/2601.03153)
  - Cost: FLOPs +5.22% and latency +5.80% vs. SASRec. — [PLR](https://arxiv.org/html/2601.03153)
- **ManCAR:**
  - Label-informed teacher, and unclear construction of the inference candidate set. — [ManCAR](https://arxiv.org/html/2602.20093)
  - Weaker on sparse datasets, and loses to PRL on Music R@10. — [ManCAR](https://arxiv.org/html/2602.20093)
  - No variance reporting, and the contraction assumption is unverified. — [ManCAR](https://arxiv.org/html/2602.20093)
- **DiffuReason:**
  - About 2.2–2.3× training time per batch and about +10% inference latency. — [DiffuReason](https://arxiv.org/html/2602.09744)
  - No significance tests, and the Hit@1 reward is mismatched with the R/N metrics. — [DiffuReason](https://arxiv.org/html/2602.09744)
  - Trails ReaRec on Sports N@5 (0.0153 vs. 0.0157). — [DiffuReason](https://arxiv.org/html/2602.09744)
  - Online A/B test: GMV +0.79%. — [DiffuReason](https://arxiv.org/html/2602.09744)
- **RecRec:**
  - No learned halting. — [RecRec](https://arxiv.org/html/2607.12945v1)
  - Reasoning and prediction share the same R×d shape. — [RecRec](https://arxiv.org/html/2607.12945v1)
  - Ablations on SASRec and two datasets only, and no direct comparison with concurrent work (PLR, ManCAR). — [RecRec](https://arxiv.org/html/2607.12945v1)
- **STREAM-Rec:**
  - Fixed reasoning length, and reliance on manual priors in trace annotation. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
  - RL from cold start is suboptimal. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
  - Compared to TIGER rather than to latent-reasoning baselines. — [STREAM-Rec](https://arxiv.org/html/2504.09627v1)
- **OnePiece [ABS]:**
  - Industrial result: over +2% GMV/UU in Shopee search. A team write-up attributes the gains over "ReaRec+PA" to block-wise reasoning bandwidth and progressive training. — [OnePiece](https://huggingface.co/papers/2509.18091)

### Inferences
- The evidence base is heavily concentrated in two groups: RUC/Alibaba (ReaRec, PLR) and RUC/Meituan (LARES), plus ManCAR's group.
- All work is on small Amazon/Yelp-scale offline benchmarks without variance reporting. Gains of ≈ 2–8% should be treated as within plausible tuning noise unless they are replicated.

### Gaps
- Full texts not read: OnePiece, DS-Frame, LASAR, IBA, and the LLM-based latent reasoning recommenders (LaRec 2607.24617, Hierarchical Latent Reasoning 2607.27760, Intuition-Guided 2606.27684, judged out of scope).
- A Semantic Scholar citation list of ReaRec was not systematically enumerated, so additional low-visibility follow-ups may exist.
- ReaRec's peer-reviewed venue was not confirmed.
