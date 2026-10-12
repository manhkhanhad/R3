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
| 320269 | smoke: quantizer (512 rows, 2 epochs) → stage 2 (512 rows, 1 epoch) → eval 16 rows + swap | submitted |
