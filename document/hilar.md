# HiLaR on R3: hierarchical latent reasoning re-implementation

**Paper.** Hu et al., "Hierarchical Latent Reasoning for LLM-based Recommendation (HiLaR)", arXiv
2607.27760. The code repository (github.com/hupeiyu21/HiLaR) only says "Coming Soon", so
every detail the paper leaves out is a choice, listed below.

**Reported (Table 1, Instruments; LatentR3 preprocessing and split, the same as ours):**

| method | H@5 | H@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| LatentR3 | 0.1032 | 0.1241 | 0.0894 | 0.0981 |
| VRec | 0.1056 | 0.1234 | 0.0916 | 0.0970 |
| FLR | 0.1104 | 0.1302 | 0.0952 | 0.1015 |
| HiLaR | 0.1129 | 0.1320 | 0.0976 | 0.1038 |

Our R3 v1 (SFT): H@10 0.1226, N@10 0.0950.

## What HiLaR is (and how it differs from LaRec)

HiLaR does not distil text CoT. The "teacher" for the K latent steps is a
**hierarchy of quantised user codes**:
- a residual quantizer over a frozen LLM's user state;
- each level is trained to predict a progressively more recent part of the history;
- the last level is trained to predict the next item.

The latent steps are aligned to the cumulative code embeddings (stage 2), then refined
with GRPO using step-wise process rewards (stage 3).

## Stage 1: temporal-guided hierarchical quantizer (`src/latent_distill/hilar_quantizer.py`)

| component | paper | here |
|---|---|---|
| user state h_u | frozen LLM, state at the end-of-history token of a time-aware prompt | Qwen2.5-1.5B-Instruct, final normed layer, last token of a chat prompt listing "[YYYY-MM-DD] title" lines. **CHOICE:** standardised per dimension, because final-layer states are anisotropic (cosine ≈ 0.98 between random users) |
| quantizer | residual, K = 4 levels, codebooks of 256 | same; per-level codebooks in the full 1,536-d space (no encoder is described); k-means init on 8,192 training rows |
| reconstruction | ‖h_u − e_{u,K}‖² | same (mean over dimensions) |
| temporal targets | history split into K chronological equal-count windows; level k < K: items of windows k..K; level K: next item i⁺ | same (`np.array_split`; short histories leave the middle windows empty, and those levels are skipped) |
| temporal loss | multi-label BCE with negative sampling, level decoders Dec_k | **CHOICE:** Dec_k linear over the 5k-item vocabulary; 100 uniform negatives per row; equal level weights |
| VQ loss | codebook + commitment | ‖sg(r) − q‖² + 0.25 ‖r − sg(q)‖² (the commitment term is inert without an encoder) |
| training | not given | Adam, lr 1e-3, batch 512, 20 epochs, training split only (5% held out for diagnostics) |

**What the teacher encodes.** Code assignment depends on h_u only, so the temporal loss
trains the codebook vectors and decoders, not the assignment. A level-4 code therefore
means "what users with this code buy next": a collaborative summary learned from all
training users, with no copy of the row's own target. The student can in principle
compute it from the history, but only by learning the population-level regularity the
quantizer captured.

**Diagnostics** (`stats.json`): codes used and perplexity per level; reconstruction R²
per level; held-out HR@10 of the level-4 decoder vs popularity.

## Stage 2: hierarchical latent alignment (`src/latent/hilar_latent.py`)

    L_SFT = L_rec + λ · (1/K) Σ_k ‖Proj(τ_k) − e_{u,k}‖²,   λ = 0.1 (paper)

| component | paper | here |
|---|---|---|
| latent states τ_k | K continuous states μ(H_u, τ^{<k}); generation mechanism not given | the K-step machinery of the LaRec re-implementation: τ_k = head(state before slot k), and slot k's input is the soft token softmax(lm_head(τ_k)) @ E. Every step conditions on the history and earlier steps |
| Proj | learnable | linear 1,536 → 1,536 |
| ‖·‖² | squared L2 | **CHOICE:** mean over dimensions (a sum over 1,536 dimensions would make λ = 0.1 dwarf the CE) |
| trained parameters | latent module + projection, backbone frozen; lr 5e-6, batch 4 | **R3 v1's fixed recipe**: full fine-tuning from Qwen2.5-1.5B-Instruct, lr 3e-4, batch 128, 3 epochs, early stopping, cutoff 512. This keeps hyperparameters fixed across all our variants |

**Evaluation.**
- Standard constrained 10-beam ranking on the full test set (`rationale_eval.py --mode larec`).
- **Latent-swap control** on the first 2,000 rows: each user receives another user's whole latent chain. The paper has no such test.

## Stage 3: hierarchical reward-guided GRPO (not implemented yet)

- Gaussian policy over the latent states, σ_k = (0.12, 0.08, 0.05, 0.025), with G = 6 rollouts.
- Rewards:
  - final: exact hit, common prefix, token-F1 of the title, clipped SASRec score;
  - process: target log-likelihood gain per latent level, plus cosine to e_{u,k};
  - λ_proc = 0.2, KL 0.05.
- Note from our RL probe (latent_distillation.md): R3's own latent-noise RL got a 0.2% reward spread because the decoder ignored `r`. Stage 3 is only worth building if the stage-2 model's predictions depend on its latents (swap control).

## Runs

| job | what | status |
|---|---|---|
| 320269 | smoke stage 1–2 | failed: the held-out diagnostic fed all 4 levels to the decoder (top-10 over levels); fixed, end-to-end test added |
| 320568 | smoke stage 1–2, rerun | passed (4 min). Initial alignment MSE 22 vs unit-variance targets, from a standard-init projection on unnormalised states, so the projection is now initialised near zero |
| 320569 / 320570 | smoke GRPO probe / train | failed: `forward` cleared the sampled chain on every decoding step; the chain now persists in `last_chain` (test added) |
| 321364 / 321365 | smoke GRPO probe / train, rerun | passed. Latent noise changes the decoded title (1.75–3.5 distinct titles per 6 rollouts); within-group reward std 0.001–0.02 (no hits on a 4-step model); latent KL to stage 2 grows ~73 per update (summed over 4 × 1,536 dims, σ down to 0.025): to monitor |
| 321366 | full stage 1–2 | stage 1 done; **preempted** at stage-2 step 134 (no checkpoint) |
| 321654 | resubmission with `--requeue` + resume from epoch checkpoints | failed in 1 s: `ls` of the (absent) checkpoints aborted under `pipefail`; fixed |
| 321667 | **full stage 2** + test eval + swap control | done (14 h 16 min); results below |
| 321668 | **GRPO probe** on the stage-2 checkpoint (50 batches × 4 users × 6 rollouts, no updates) | done; results below |

## Results (2026-10-14)

### Stage 1: quantizer (66,500 training rows, 20 epochs)

| level | codes used / 256 | perplexity | reconstruction R² of h_u |
|---:|---:|---:|---:|
| 1 | 256 | 232 | 0.74 |
| 2 | 254 | 162 | 0.80 |
| 3 | 253 | 106 | 0.84 |
| 4 | 249 | 23 | 0.86 |

The hierarchy is well used, and each level reconstructs more of the user state. But **the
level-4 ("next item") decoder predicts held-out next items worse than popularity: HR@10 0.057
vs 0.100**. Its training loss was still falling (0.52 → 0.45 over the last 8 epochs). The code
for a user's history is not a good predictor of their next purchase for unseen users.

### Stage 2: alignment SFT (job 321667)

Validation CE per epoch: 0.4613 / **0.3995** / 0.4231 (R3 v1: 0.4632 / 0.3982 / 0.4212). The
alignment MSE fell from about 1 to about 0.4.

| model (test, 8,313 rows) | HR@1 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| R3 v1 | 0.0747 | 0.1020 | **0.1226** | **0.0950** |
| HiLaR stage 2 (ours) | 0.0294 | 0.0869 | 0.1205 | 0.0713 |
| *paper: LatentR3 / HiLaR (after GRPO)* | — | *0.1032 / 0.1129* | *0.1241 / 0.1320* | *0.0981 / 0.1038* |

- **HR@10 is unchanged** (gained 119 / lost 136 hits vs v1), but **NDCG@10 drops 25%**.
  - Hits move down the list: hits by rank 1/2 are 244/243 for HiLaR vs 621/74 for R3. Mean rank on common hits is 3.64 vs 2.23.
  - When the target is at rank 2, rank 1 is usually a sibling with a near-identical title: "D'Addario EXL120-7 …" (76 rows), "D'Addario EJ26 …" (70). The model reaches the right product family but loses the final fine-grained choice.
  - It is not a decoding artefact: beams are in descending score order for every row.
- **The latents' content is not used.** Swapping in another user's whole latent chain on the first 2,000 rows gives HR@10 0.1550 → 0.1540, with 88.9% identical top-1. This is the same picture as R3's single latent.

### Stage 3 gate: GRPO signal probe (job 321668, paper's σ = 0.12/0.08/0.05/0.025)

| 5-batch window | 1–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 | 31–35 | 36–40 | 41–45 | 46–50 |
|---|---|---|---|---|---|---|---|---|---|---|
| distinct titles per user (of 6 rollouts) | 1.15 | 1.00 | 1.20 | 1.05 | 1.15 | 1.00 | 1.00 | 1.00 | 1.20 | 1.10 |
| within-group std of the final reward | 0.012 | 0.000 | 0.015 | 0.003 | 0.012 | 0.000 | 0.000 | 0.000 | 0.010 | 0.002 |

- With the paper's noise scale, the 6 rollouts of a user almost always decode to the **same title**. In half of the windows the final reward has zero within-group variance, so the advantages are noise.
- The process reward varies little too: align ≈ 0.90 throughout.
- The large "gain" term (target log-likelihood −0.95 → −0.30) compares latents against *empty* slots. That shows the model needs something in the slots, not which latent it gets.
- **Conclusion: GRPO at the paper's settings has almost no learning signal on this model.** This is the same failure mode as R3's own RL (latent_distillation.md). Running the 2,000-step GRPO is not justified as is.

### Reading

On our R3 setup, HiLaR's stages 1–2 do not reproduce the paper's gains:
- HR@10 matches R3, NDCG@10 is worse, and the latents' content is unused.
- The paper's GRPO stage has no reward variance to work with.

Likely reasons:
1. The stage-1 teacher's next-item level generalises worse than popularity, so aligning to it adds no predictive information.
2. As with every variant so far, information derived from the same history is redundant for the decoder.
3. Our stage 2 fine-tunes the whole model with R3's recipe, while the paper trains only the latent module (unreleased code, details unknown).
