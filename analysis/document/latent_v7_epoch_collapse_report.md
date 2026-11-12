# Latent-collapse assessment for `latent_v7` checkpoints 384, 768, and 1152

## Executive verdict

**All three `latent_v7` checkpoints are strongly shared-direction and exact-template concentrated. The trajectory is non-monotonic through checkpoint 768, followed by a large recovery in centered dimensional diversity at checkpoint 1152.**

Only 21.75%–89.63% of stored FP16 thoughts are unique, and the ten most common templates cover 1.13%–28.66% of train examples. Checkpoint 768 is the most template concentrated. Centered entropy effective rank moves from 3.39 to 5.78 to 11.82; checkpoint 1152 is substantially broader, but its exact-template concentration remains high.

All checkpoints remain anisotropic: random-pair cosine is 0.946–0.992, and the global-mean norm is 97.2%–99.7% of the mean sample norm. The last checkpoint improves residual diversity without eliminating the common carrier direction.

E1 does not match the simple activation-without-content pattern uniformly across all three checkpoints; the checkpoint-level confidence intervals below should be used for the causal conclusion.

**Addendum (§7):** checkpoint 1152 was further fine-tuned with attention-only GRPO (`nomul_attn_lr_5e-5`). Unlike the `latent_v4` GRPO run (see `latent_v4_epoch_collapse_report.md` §7, where the same recipe produced a catastrophic collapse), this `v7` GRPO run leaves raw ranking and train NLL essentially unchanged from checkpoint 1152, and reproduces the same activation-without-content utilization pattern (zero-masking significantly hurts; every content-preserving control is statistically indistinguishable from raw). What it does change is exact-template collapse: the single most common stored thought jumps from covering 0.27% of train examples at checkpoint 1152 to 12.76% after GRPO, and the top-10 templates go from 2.36% to 23.97% coverage, with centered dimensional spread falling rather than rising. See §7 for full evidence — this is a different, more benign failure mode than `v4`'s, and the contrast between the two runs indicates this GRPO configuration's effect on the checkpoint is highly run-dependent rather than a fixed, predictable transformation.

## Scope and reproducibility

- Dataset: `CDs_and_Vinyl`
- Geometry and extraction likelihood split: train, 49,251 examples
- Causal ranking split: test, 6,157 examples
- Architecture output: one 1,536-dimensional stored thought per interaction
- Stored dtype: FP16
- Train split SHA-256: `864f652064a08cd9d9b492e11cacceafba074cc8f4b1c3469b10a4bcdaba8da4`
- Test split SHA-256: `944634cd2b9248a4a579f222151c6df140935f56f5ea731cfe4ac21c240e1999`
- Pairwise samples: 100,000 per diagnostic, seed 42
- PCA: centered and fitted on all 49,251 thoughts independently at each checkpoint
- Ranking: constrained 10-beam catalog generation; paired bootstrap 95% intervals over all 6,157 test examples, 2,000 resamples, seed 42

Primary artifacts:

- [Epoch comparison summary](../results/latent_distribution/latent_v7_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_v7_epoch_collapse_trajectory.png)
- Checkpoint statistics: [384](../results/latent_distribution/latent_v7_checkpoint_384/train_latent_distribution.json), [768](../results/latent_distribution/latent_v7_checkpoint_768/train_latent_distribution.json), [1152](../results/latent_distribution/latent_v7_checkpoint_1152/train_latent_distribution.json)
- Distribution overviews: [384](../figures/latent_v7_checkpoint_384_latent_distribution_overview.png), [768](../figures/latent_v7_checkpoint_768_latent_distribution_overview.png), [1152](../figures/latent_v7_checkpoint_1152_latent_distribution_overview.png)
- All-vector PCA views: [384](../figures/latent_v7_checkpoint_384_latent_pca_all_scatter.png), [768](../figures/latent_v7_checkpoint_768_latent_pca_all_scatter.png), [1152](../figures/latent_v7_checkpoint_1152_latent_pca_all_scatter.png)
- E1 ranking artifacts: [384](../results/intervention/latent_v7_checkpoint_384/), [768](../results/intervention/latent_v7_checkpoint_768/), [1152](../results/intervention/latent_v7_checkpoint_1152/)

## 1. Epoch trajectory

| Diagnostic | Step 384 (epoch 1) | Step 768 (epoch 2) | Step 1152 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 1.174714 | 0.738813 | 0.367138 |
| Zero-mask train target NLL | 1.470935 | 0.979081 | 1.122569 |
| Zero-mask NLL penalty | +0.296221 | +0.240268 | +0.755431 |
| Mean sample norm | 114.36 | 78.19 | 76.50 |
| Global-mean vector norm | 113.96 | 77.21 | 74.38 |
| Global-mean norm / mean sample norm | 0.9965 | 0.9875 | 0.9723 |
| Mean cosine to global mean | 0.9959 | 0.9873 | 0.9724 |
| Mean random-pair cosine | 0.9918 | 0.9748 | 0.9456 |
| Exact FP16-unique vectors | 10,711 | 44,143 | 42,486 |
| Unique fraction | 21.75% | 89.63% | 86.26% |
| Top-10-template coverage | 28.66% | 1.13% | 2.36% |
| Largest-template coverage | 4.25% | 0.27% | 0.97% |
| PC1 centered variance | 55.51% | 46.66% | 27.36% |
| Top-two centered variance | 84.54% | 66.55% | 41.73% |
| Entropy effective rank | 3.39 | 5.78 | 11.82 |
| Participation ratio | 2.51 | 3.62 | 7.71 |
| Components for 90% variance | 4 | 6 | 12 |

Raw train NLL improves monotonically, but collapse metrics do not. Effective rank falls from 3.39 to 5.78, then rises to 11.82. At checkpoint 1152, PC1 explains 27.36% of centered variance and 12 components are needed for 90%, compared with 6 at checkpoint 768. This is a substantial late recovery in dimensional diversity, not monotonic collapse.

## 2. Template concentration versus dimensional concentration

Exact-template collapse is strong at every checkpoint. The unique-vector counts are 10,711, 44,143, and 42,486 among 49,251 samples. The largest single templates cover 4.25%, 0.27%, and 0.97% of the train split.

Dimensional concentration and exact repetition move differently late in training. Checkpoint 1152 expands to effective rank 11.82 and reduces the first-ten-PC share to 87.76%, yet only 86.26% of stored vectors are unique. A broader centered span therefore does not imply recovery from discrete template reuse.

## 3. User-, target-, and preference-linked structure

| Diagnostic | Step 384 | Step 768 | Step 1152 |
|---|---:|---:|---:|
| Same-user cosine (absolute) | 0.9928 | 0.9744 | 0.9455 |
| Same-target cosine (absolute) | 0.9924 | 0.9753 | 0.9461 |
| Same CF-cluster cosine (absolute) | 0.9919 | 0.9750 | 0.9458 |
| Adjusted variance associated with user | 3.24% | 0.07% | -0.72% |
| Adjusted variance associated with target | 6.17% | 1.03% | 1.30% |
| Adjusted variance associated with CF cluster | 1.41% | 0.28% | 0.25% |

The table reports absolute within-group cosine values. Random-pair baselines are 0.9918, 0.9748, and 0.9456. User identity is the strongest of the measured grouping variables at all checkpoints; target-associated variance is smaller, and conventional collaborative-filtering cluster structure is weak. These associations show that the latent bank is not arbitrary noise, but they do not establish that the decoder uses the variation; that causal question is tested by E1 below.

The exact-template utility eta-squared values are not treated as reliable utility evidence because the many small template groups make that statistic structurally prone to overfitting.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 384

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.057008 | 0 | 0.040144 | 0 |
| Zero / mask | 0.043203 | -0.013805 [-0.017216, -0.010395] | 0.026031 | -0.014114 [-0.016528, -0.011635] |
| Global train mean | 0.057171 | +0.000162 [-0.000487, +0.000975] | 0.040206 | +0.000062 [-0.000316, +0.000444] |
| Shuffled user | 0.057333 | +0.000325 [+0.000000, +0.000812] | 0.040359 | +0.000215 [-0.000062, +0.000512] |
| Shuffled target | 0.057333 | +0.000325 [+0.000000, +0.000812] | 0.040359 | +0.000215 [-0.000058, +0.000516] |
| Gaussian noise 0.25x | 0.057658 | +0.000650 [-0.000325, +0.001624] | 0.040532 | +0.000388 [-0.000039, +0.000855] |
| Gaussian noise 0.50x | 0.057658 | +0.000650 [-0.000325, +0.001787] | 0.040538 | +0.000394 [-0.000029, +0.000825] |
| Gaussian noise 1.00x | 0.057820 | +0.000812 [+0.000000, +0.001787] | 0.040892 | +0.000748 [+0.000327, +0.001273] |

### Checkpoint 768

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.099886 | 0 | 0.069828 | 0 |
| Zero / mask | 0.080234 | -0.019652 [-0.024038, -0.015430] | 0.054469 | -0.015359 [-0.018450, -0.012301] |
| Global train mean | 0.099724 | -0.000162 [-0.001299, +0.000975] | 0.069633 | -0.000195 [-0.000651, +0.000291] |
| Shuffled user | 0.099237 | -0.000650 [-0.001624, +0.000162] | 0.069337 | -0.000491 [-0.000966, -0.000034] |
| Shuffled target | 0.099237 | -0.000650 [-0.001624, +0.000162] | 0.069337 | -0.000491 [-0.000938, -0.000008] |
| Gaussian noise 0.25x | 0.098587 | -0.001299 [-0.002436, -0.000321] | 0.069376 | -0.000452 [-0.000941, +0.000075] |
| Gaussian noise 0.50x | 0.098749 | -0.001137 [-0.002436, +0.000162] | 0.069564 | -0.000264 [-0.000803, +0.000271] |
| Gaussian noise 1.00x | 0.098749 | -0.001137 [-0.002599, +0.000325] | 0.069574 | -0.000254 [-0.000904, +0.000404] |

### Checkpoint 1152

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.114017 | 0 | 0.078223 | 0 |
| Zero / mask | 0.088192 | -0.025824 [-0.031346, -0.020302] | 0.054821 | -0.023402 [-0.026915, -0.019890] |
| Global train mean | 0.113692 | -0.000325 [-0.001787, +0.001137] | 0.078153 | -0.000070 [-0.000642, +0.000589] |
| Shuffled user | 0.114991 | +0.000975 [-0.000487, +0.002436] | 0.078535 | +0.000312 [-0.000332, +0.000948] |
| Shuffled target | 0.114991 | +0.000975 [-0.000325, +0.002274] | 0.078535 | +0.000312 [-0.000309, +0.000962] |
| Gaussian noise 0.25x | 0.114341 | +0.000325 [-0.000975, +0.001787] | 0.078465 | +0.000242 [-0.000392, +0.000819] |
| Gaussian noise 0.50x | 0.114666 | +0.000650 [-0.000812, +0.002111] | 0.078634 | +0.000411 [-0.000230, +0.001068] |
| Gaussian noise 1.00x | 0.115478 | +0.001462 [-0.000162, +0.003248] | 0.078739 | +0.000516 [-0.000258, +0.001272] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these tables provide point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 384

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.026961 | 0.038330 | 0.044177 | 0.057008 | 0.026961 | 0.033539 | 0.035957 | 0.040144 |
| Zero / mask | 0.011044 | 0.025662 | 0.031834 | 0.043203 | 0.011044 | 0.019799 | 0.022343 | 0.026031 |
| Global train mean | 0.027124 | 0.037843 | 0.044989 | 0.057171 | 0.027124 | 0.033313 | 0.036248 | 0.040206 |
| Shuffled user | 0.027286 | 0.038330 | 0.044502 | 0.057333 | 0.027286 | 0.033638 | 0.036189 | 0.040359 |
| Shuffled target | 0.027286 | 0.038330 | 0.044502 | 0.057333 | 0.027286 | 0.033638 | 0.036189 | 0.040359 |
| Gaussian noise 0.25x | 0.027286 | 0.038493 | 0.044827 | 0.057658 | 0.027286 | 0.033740 | 0.036354 | 0.040532 |
| Gaussian noise 0.50x | 0.027448 | 0.038006 | 0.044827 | 0.057658 | 0.027448 | 0.033556 | 0.036352 | 0.040538 |
| Gaussian noise 1.00x | 0.027936 | 0.038330 | 0.045314 | 0.057820 | 0.027936 | 0.033941 | 0.036835 | 0.040892 |

#### Checkpoint 768

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.046126 | 0.068215 | 0.078447 | 0.099886 | 0.046126 | 0.058766 | 0.062952 | 0.069828 |
| Zero / mask | 0.034595 | 0.050836 | 0.062368 | 0.080234 | 0.034595 | 0.043992 | 0.048666 | 0.054469 |
| Global train mean | 0.045802 | 0.068215 | 0.077960 | 0.099724 | 0.045802 | 0.058646 | 0.062650 | 0.069633 |
| Shuffled user | 0.045639 | 0.067728 | 0.079259 | 0.099237 | 0.045639 | 0.058300 | 0.062967 | 0.069337 |
| Shuffled target | 0.045639 | 0.067728 | 0.079259 | 0.099237 | 0.045639 | 0.058300 | 0.062967 | 0.069337 |
| Gaussian noise 0.25x | 0.045964 | 0.068540 | 0.078122 | 0.098587 | 0.045964 | 0.058868 | 0.062767 | 0.069376 |
| Gaussian noise 0.50x | 0.046289 | 0.068053 | 0.077960 | 0.098749 | 0.046289 | 0.058766 | 0.062840 | 0.069564 |
| Gaussian noise 1.00x | 0.046289 | 0.067890 | 0.078285 | 0.098749 | 0.046289 | 0.058727 | 0.062947 | 0.069574 |

#### Checkpoint 1152

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.049537 | 0.074712 | 0.090791 | 0.114017 | 0.049537 | 0.064102 | 0.070742 | 0.078223 |
| Zero / mask | 0.029235 | 0.050512 | 0.063830 | 0.088192 | 0.029235 | 0.041447 | 0.046962 | 0.054821 |
| Global train mean | 0.049537 | 0.075199 | 0.090791 | 0.113692 | 0.049537 | 0.064367 | 0.070762 | 0.078153 |
| Shuffled user | 0.049700 | 0.074874 | 0.090304 | 0.114991 | 0.049700 | 0.064243 | 0.070582 | 0.078535 |
| Shuffled target | 0.049700 | 0.074874 | 0.090304 | 0.114991 | 0.049700 | 0.064243 | 0.070582 | 0.078535 |
| Gaussian noise 0.25x | 0.049700 | 0.075361 | 0.091116 | 0.114341 | 0.049700 | 0.064508 | 0.070966 | 0.078465 |
| Gaussian noise 0.50x | 0.049862 | 0.075686 | 0.091278 | 0.114666 | 0.049862 | 0.064730 | 0.071118 | 0.078634 |
| Gaussian noise 1.00x | 0.049862 | 0.074874 | 0.089979 | 0.115478 | 0.049862 | 0.064346 | 0.070524 | 0.078739 |

All 24 generation runs returned 6,157 paired records and zero catalog violations. Raw ranking improves with training: HR@10 is 0.0570, 0.0999, and 0.1140; NDCG@10 is 0.0401, 0.0698, and 0.0782.

Zero-mask HR@10 deltas are -0.013805, -0.019652, and -0.025824; corresponding NDCG@10 deltas are -0.014114, -0.015359, and -0.023402. Across the mean, shuffle, and noise controls, the largest absolute point changes are 0.001462 HR@10 and 0.000748 NDCG@10. The confidence intervals above determine whether those changes are distinguishable from zero.

The shuffled-user and shuffled-target decoded top-10 lists differ on 0, 1, and 0 of 6,157 examples at checkpoints 384, 768, and 1152. Where they are effectively identical, they should be treated as one control rather than independent evidence.

### Activation-versus-content interpretation

Train zero masking raises NLL by 0.2962, 0.2403, and 0.7554. Held-out E1 separately asks whether ranking depends on the activation's identity. A damaging zero intervention combined with null mean/shuffle/noise effects is activation dependence without sample-specific content dependence—the causal criterion for latent-utilization collapse used in the reference report.

## 5. Diagnosis

1. **Exact-template collapse: supported at all checkpoints.** Only 21.75%–89.63% of stored FP16 thoughts are unique, and the top ten templates cover 1.13%–28.66%.
2. **Shared-direction collapse: supported at all checkpoints.** Random-pair cosine is 0.946–0.992, and the mean vector retains 97.2%–99.7% of a typical sample norm.
3. **Centered dimensional collapse: strongest at checkpoint 768 and substantially weaker at checkpoint 1152.** Effective ranks are 3.39, 5.78, and 11.82.
4. **Functional/utilization collapse: mixed.** At least one checkpoint or content control departs from the uniform null pattern; use the paired intervals above.

The most defensible overall conclusion is that `latent_v7` learns increasingly accurate recommendations and, by checkpoint 1152, a much broader centered residual space. It nevertheless retains severe exact-template reuse and a large common direction. Geometric diversity and causal utilization must be evaluated separately.

## 6. Limitations and next action

- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.
- Exact duplicate counts are at FP16 storage precision; cosine and PCA diagnostics operate in float32 after loading.
- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the full train bank.
- Zero masking is off-manifold; mean and shuffled interventions are in-distribution controls, while Gaussian noise tests continuous perturbations.
- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking can miss probability changes that do not alter the constrained top-10 order.

The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This tests whether token probabilities share the ranking result.

---

## 7. Post-GRPO addendum: `nomul_attn_lr_5e-5`

**Following checkpoint 1152, the attention module was further fine-tuned with attention-only GRPO (same recipe as `latent_v4`'s addendum: `lr=5e-5`, `beta=0.01`, 1 epoch, `num_generations=8`, base LLM and `lm_head` frozen). Unlike `latent_v4`'s GRPO run, this one does not break the model: raw ranking and train likelihood are essentially unchanged from checkpoint 1152, and the same activation-without-content utilization collapse persists. What changed substantially is exact-template reuse, which increased sharply, while centered geometric spread narrowed rather than widened.**

### 7.1 Scope

- Checkpoint: `output_dir/CDs_and_Vinyl_v7/nomul_attn_lr_5e-5` (sha256 `db0c798b…`), warm-started from `latent_v7/latent/checkpoint-1152` (confirmed via `config.json`'s `_name_or_path`).
- Same protocol as checkpoints 384/768/1152: geometry on the 49,251-example train split, E1 causal intervention (raw, mask, mean, shuffle_user, shuffle_target, noise 0.25×/0.5×/1.0×) on the 6,157-example test split, identical split hashes (train `864f652…`, test `944634c…`).
- Artifacts: [geometry](../results/latent_distribution/latent_v7_grpo_lr5e-5/train_latent_distribution.json), [train extraction summary](../evidence/CDs_and_Vinyl/latent_v7_grpo_lr5e-5/train/summary.json), [E1 ranking results](../results/intervention/latent_v7_grpo_lr5e-5/).

**Verification against the repo's own eval pipeline:** `output_dir/CDs_and_Vinyl_v7/nomul_attn_lr_5e-5/metric.log` contains a single block (NDCG@10=0.079055, HR@10=0.115316), unlike `v4`'s log which had two stale entries from an unrelated checkpoint (see the `v4` report §7.1 note and the earlier debugging session). Recomputing HR/NDCG directly from `final_result.json` with the repo's own `calc.py` logic reproduces that block exactly, and matches this section's E1 raw row (HR@10=0.115153, NDCG@10=0.078862) to within beam-search batching noise. No stale-log issue here.

### 7.2 Geometry: more exact-template reuse, less centered spread — the opposite pattern from `v4`

| Diagnostic | Checkpoint 1152 (pre-RL) | `nomul_attn_lr_5e-5` (post-RL) |
|---|---:|---:|
| Entropy effective rank | 11.82 | 2.98 |
| Participation ratio | 7.71 | 1.73 |
| PC1 centered variance | 27.36% | 75.22% |
| Components for 90% variance | 12 | 4 |
| Mean random-pair cosine | 0.9456 | 0.9453 |
| Global-mean norm / mean sample norm | 0.9723 | 0.9741 |
| Exact FP16-unique fraction | 86.26% | 57.37% |
| Top-10-template coverage | 2.36% | 23.97% |
| Largest single template coverage | 0.27% | 12.76% |

Shared-direction anisotropy (random-pair cosine, mean-norm ratio) is essentially unchanged. What moved is exact-template reuse: the single largest stored thought now accounts for 12.76% of the 49,251 train examples (up from 0.27%), and centered dimensional spread *narrowed* (effective rank fell from 11.82 to 2.98) rather than widening as it did in the `v4` GRPO run (21.32 → 360.91). GRPO here pushed the policy toward reusing a small number of thoughts far more often, not toward the high-variance, noise-injected regime seen in `v4`.

### 7.3 Train likelihood: essentially unchanged from checkpoint 1152

| Diagnostic | Checkpoint 1152 (pre-RL) | `nomul_attn_lr_5e-5` (post-RL) |
|---|---:|---:|
| Raw train target NLL | 0.367138 | 0.360697 |
| Zero-mask train target NLL | 1.122569 | 1.122569 |
| Zero-mask NLL penalty (masked − raw) | +0.755431 | +0.761871 |
| Fraction of examples where raw beats zero-mask | majority (penalty positive) | 98.05% |

Both raw and zero-mask NLL match checkpoint 1152 almost to the fourth decimal place. This GRPO run did not push the attention module into a harmful regime the way `v4`'s did — it left the SFT-era activation-dependence property intact.

### 7.4 Causal ranking: same collapse pattern as checkpoint 1152, no meaningful net change

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.115153 | 0 | 0.078862 | 0 |
| Zero / mask | 0.088192 | **−0.026961 [−0.032483, −0.021760]** | 0.054821 | **−0.024041 [−0.027599, −0.020545]** |
| Global train mean | 0.115316 | +0.000162 [−0.001137, +0.001462] | 0.079043 | +0.000181 [−0.000428, +0.000813] |
| Shuffled user | 0.115153 | +0.000000 [−0.001462, +0.001462] | 0.079084 | +0.000222 [−0.000445, +0.000905] |
| Shuffled target | 0.115153 | +0.000000 [−0.001462, +0.001462] | 0.079084 | +0.000222 [−0.000424, +0.000871] |
| Gaussian noise 0.25x | 0.114666 | −0.000487 [−0.001949, +0.000812] | 0.078655 | −0.000207 [−0.000796, +0.000370] |
| Gaussian noise 0.50x | 0.114666 | −0.000487 [−0.001787, +0.000812] | 0.078978 | +0.000116 [−0.000492, +0.000734] |
| Gaussian noise 1.00x | 0.113854 | −0.001299 [−0.003411, +0.000650] | 0.078798 | −0.000064 [−0.000924, +0.000827] |

(All 8 generation runs returned 6,157 paired records, zero catalog violations, checkpoint/split hashes matched; shuffled-user and shuffled-target predictions are identical on all 6,157 examples, so treat them as one control.)

#### Complete metrics at all cutoffs

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.050512 | 0.075037 | 0.089329 | 0.115153 | 0.050512 | 0.064667 | 0.070537 | 0.078862 |
| Zero / mask | 0.029235 | 0.050512 | 0.063830 | 0.088192 | 0.029235 | 0.041447 | 0.046962 | 0.054821 |
| Global train mean | 0.050674 | 0.075199 | 0.089654 | 0.115316 | 0.050674 | 0.064872 | 0.070798 | 0.079043 |
| Shuffled user | 0.050512 | 0.075849 | 0.089816 | 0.115153 | 0.050512 | 0.065200 | 0.070924 | 0.079084 |
| Shuffled target | 0.050512 | 0.075849 | 0.089816 | 0.115153 | 0.050512 | 0.065200 | 0.070924 | 0.079084 |
| Gaussian noise 0.25x | 0.050512 | 0.074712 | 0.089329 | 0.114666 | 0.050512 | 0.064483 | 0.070494 | 0.078655 |
| Gaussian noise 0.50x | 0.050999 | 0.075849 | 0.089492 | 0.114666 | 0.050999 | 0.065274 | 0.070858 | 0.078978 |
| Gaussian noise 1.00x | 0.051324 | 0.075849 | 0.089654 | 0.113854 | 0.051324 | 0.065351 | 0.070984 | 0.078798 |

Zero-masking is worse than raw at every cutoff, including HR@1 (0.0292 vs 0.0505). Every content-preserving control tracks raw closely across the whole cutoff range, including HR@1/NDCG@1 where all values sit within about half a point of raw — the same near-no-op pattern seen at HR@10/NDCG@10 holds from the tightest cutoff outward, not just at @10.

Raw HR@10/NDCG@10 (0.1152/0.0789) are statistically indistinguishable from checkpoint 1152's (0.1140/0.0782) — this GRPO run is a near-no-op for ranking accuracy. Zero-masking is significantly harmful, exactly as at every SFT checkpoint, and every content-preserving control (mean, shuffle, noise up to 1.0×) has a confidence interval containing zero. This is the same "activation-without-content" latent-utilization collapse already established at checkpoint 1152 — GRPO neither fixed it nor broke it further.

### 7.5 Diagnosis

This run contrasts sharply with `latent_v4`'s GRPO addendum, despite using the identical training recipe warm-started from an equivalent checkpoint. Where `v4`'s run pushed the policy into a regime where the raw activation actively damaged both train likelihood and held-out ranking, `v7`'s run left both essentially untouched and only reallocated probability mass toward reusing a small number of stored thoughts more often (exact-template collapse worsened sharply while centered geometric spread narrowed). Neither run repaired the underlying functional collapse: in both cases, and at every SFT checkpoint in both report families, the decoder needs *some* nonzero latent activation but does not measurably use its sample-specific content for ranking.

Taken together, the `v4` and `v7` GRPO addenda show that this attention-only GRPO configuration's effect on the trained policy is not a fixed, predictable transformation of the checkpoint — the same hyperparameters and warm-start point produced a catastrophic collapse in one run and a comparatively inert (if template-reinforcing) outcome in another. This is consistent with the GRPO objective's lack of an absolute anchor to the SFT baseline (group-relative advantage, weak `beta=0.01` KL term — see `RECON_NOTES.md` §2 rows 7–9 and the `v4` report §7.5): nothing in the training signal constrains how far, or in which direction, the policy is allowed to drift from its starting point, so the specific outcome appears sensitive to run-level variation (seed realization in sampling/hardware, exact optimization trajectory) rather than deterministic given the hyperparameters alone.

### 7.6 Limitations

- This is a single run at a single learning rate (`5e-5`); as with `v4`, no sweep was executed, and this is the only GRPO run of `v7` observed, so run-to-run variance within `v7` itself (as opposed to across `v4`/`v7`) is not characterized.
- As with the SFT checkpoints, geometry is measured on train and causal ranking on test; the two splits are not jointly perturbed in the same run.
- The train NLL comparison uses the same teacher-forced target as the rest of this report family; it was not re-verified against a beam-search generation NLL.
