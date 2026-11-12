# Latent-collapse assessment for `latent_v6` checkpoints 384, 768, and 1152

## Executive verdict

**All three `latent_v6` checkpoints are strongly shared-direction and exact-template concentrated. The trajectory is non-monotonic through checkpoint 768, followed by a large recovery in centered dimensional diversity at checkpoint 1152.**

Only 15.74%–18.38% of stored FP16 thoughts are unique, and the ten most common templates cover 21.89%–27.75% of train examples. Checkpoint 768 is the most template concentrated. Centered entropy effective rank moves from 15.44 to 12.31 to 57.47; checkpoint 1152 is substantially broader, but its exact-template concentration remains high.

All checkpoints remain anisotropic: random-pair cosine is 0.825–0.907, and the global-mean norm is 90.1%–95.3% of the mean sample norm. The last checkpoint improves residual diversity without eliminating the common carrier direction.

The completed E1 intervention establishes activation dependence without beneficial sample-specific content dependence at all three checkpoints. Zero masking significantly reduces HR@10 and NDCG@10, but replacing the thought with the global train mean or an unrelated train thought does not worsen ranking. Gaussian perturbations are also indistinguishable from raw, except that 0.25x noise slightly improves HR@10 at checkpoint 768 while leaving NDCG@10 unchanged. **Latent-utilization collapse is therefore strongly supported at checkpoints 384, 768, and 1152.**

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

- [Epoch comparison summary](../results/latent_distribution/latent_v6_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_v6_epoch_collapse_trajectory.png)
- Checkpoint statistics: [384](../results/latent_distribution/latent_v6_checkpoint_384/train_latent_distribution.json), [768](../results/latent_distribution/latent_v6_checkpoint_768/train_latent_distribution.json), [1152](../results/latent_distribution/latent_v6_checkpoint_1152/train_latent_distribution.json)
- Distribution overviews: [384](../figures/latent_v6_checkpoint_384_latent_distribution_overview.png), [768](../figures/latent_v6_checkpoint_768_latent_distribution_overview.png), [1152](../figures/latent_v6_checkpoint_1152_latent_distribution_overview.png)
- All-vector PCA views: [384](../figures/latent_v6_checkpoint_384_latent_pca_all_scatter.png), [768](../figures/latent_v6_checkpoint_768_latent_pca_all_scatter.png), [1152](../figures/latent_v6_checkpoint_1152_latent_pca_all_scatter.png)
- E1 ranking artifacts: [384](../results/intervention/latent_v6_checkpoint_384/), [768](../results/intervention/latent_v6_checkpoint_768/), [1152](../results/intervention/latent_v6_checkpoint_1152/)

## 1. Epoch trajectory

| Diagnostic | Step 384 (epoch 1) | Step 768 (epoch 2) | Step 1152 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 1.141141 | 0.721615 | 0.344010 |
| Zero-mask train target NLL | 1.991558 | 1.456953 | 1.859016 |
| Zero-mask NLL penalty | +0.850417 | +0.735338 | +1.515006 |
| Mean sample norm | 96.97 | 86.18 | 97.63 |
| Global-mean vector norm | 92.43 | 79.93 | 87.98 |
| Global-mean norm / mean sample norm | 0.9532 | 0.9275 | 0.9012 |
| Mean cosine to global mean | 0.9524 | 0.9293 | 0.9084 |
| Mean random-pair cosine | 0.9069 | 0.8637 | 0.8246 |
| Exact FP16-unique vectors | 9,051 | 7,752 | 8,097 |
| Unique fraction | 18.38% | 15.74% | 16.44% |
| Top-10-template coverage | 21.89% | 27.75% | 26.37% |
| Largest-template coverage | 3.57% | 5.28% | 5.64% |
| PC1 centered variance | 37.57% | 38.90% | 24.31% |
| Top-two centered variance | 59.07% | 64.66% | 36.39% |
| Entropy effective rank | 15.44 | 12.31 | 57.47 |
| Participation ratio | 5.00 | 4.37 | 11.38 |
| Components for 90% variance | 43 | 33 | 226 |

Raw train NLL improves monotonically, but collapse metrics do not. Effective rank falls from 15.44 to 12.31, then rises to 57.47. At checkpoint 1152, PC1 explains 24.31% of centered variance and 226 components are needed for 90%, compared with 33 at checkpoint 768. This is a substantial late recovery in dimensional diversity, not monotonic collapse.

## 2. Template concentration versus dimensional concentration

Exact-template collapse is strong at every checkpoint. The unique-vector counts are 9,051, 7,752, and 8,097 among 49,251 samples. The largest single templates cover 3.57%, 5.28%, and 5.64% of the train split.

Dimensional concentration and exact repetition move differently late in training. Checkpoint 1152 expands to effective rank 57.47 and reduces the first-ten-PC share to 62.13%, yet only 16.44% of stored vectors are unique. A broader centered span therefore does not imply recovery from discrete template reuse.

## 3. User-, target-, and preference-linked structure

| Diagnostic | Step 384 | Step 768 | Step 1152 |
|---|---:|---:|---:|
| Same-user cosine (absolute) | 0.9241 | 0.8882 | 0.8508 |
| Same-target cosine (absolute) | 0.9095 | 0.8678 | 0.8289 |
| Same CF-cluster cosine (absolute) | 0.9083 | 0.8651 | 0.8275 |
| Adjusted variance associated with user | 12.54% | 16.26% | 8.09% |
| Adjusted variance associated with target | 4.01% | 4.02% | 3.37% |
| Adjusted variance associated with CF cluster | 1.23% | 1.07% | 1.01% |

The table reports absolute within-group cosine values. These are high, but the random-pair baselines are also 0.9069, 0.8637, and 0.8246. Same-user similarity is consistently the highest of the measured groupings; target and conventional collaborative-filtering cluster similarity remain closer to the random baseline. These associations show that the latent bank is not arbitrary noise, but they do not establish decoder use; that causal question is tested by E1 below.

The exact-template utility eta-squared values are not treated as reliable utility evidence because the many small template groups make that statistic structurally prone to overfitting.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 384

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.059932 | 0 | 0.042499 | 0 |
| Zero / mask | 0.039955 | -0.019977 [-0.024367, -0.015592] | 0.025025 | -0.017474 [-0.020523, -0.014289] |
| Global train mean | 0.060094 | +0.000162 [-0.000812, +0.001137] | 0.042451 | -0.000048 [-0.000550, +0.000441] |
| Shuffled user | 0.060419 | +0.000487 [-0.000650, +0.001624] | 0.042491 | -0.000008 [-0.000539, +0.000561] |
| Shuffled target | 0.060419 | +0.000487 [-0.000650, +0.001624] | 0.042491 | -0.000008 [-0.000519, +0.000555] |
| Gaussian noise 0.25x | 0.060419 | +0.000487 [-0.000325, +0.001462] | 0.042810 | +0.000311 [-0.000138, +0.000787] |
| Gaussian noise 0.50x | 0.059932 | +0.000000 [-0.001137, +0.000975] | 0.042493 | -0.000006 [-0.000540, +0.000558] |
| Gaussian noise 1.00x | 0.060419 | +0.000487 [-0.000650, +0.001624] | 0.042705 | +0.000206 [-0.000381, +0.000771] |

### Checkpoint 768

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.097775 | 0 | 0.070970 | 0 |
| Zero / mask | 0.067078 | -0.030697 [-0.035894, -0.025662] | 0.041416 | -0.029555 [-0.033518, -0.025846] |
| Global train mean | 0.097937 | +0.000162 [-0.000975, +0.001299] | 0.070726 | -0.000245 [-0.000902, +0.000371] |
| Shuffled user | 0.098587 | +0.000812 [-0.000650, +0.002274] | 0.070793 | -0.000177 [-0.000975, +0.000541] |
| Shuffled target | 0.098587 | +0.000812 [-0.000650, +0.002274] | 0.070793 | -0.000177 [-0.000965, +0.000562] |
| Gaussian noise 0.25x | 0.099237 | +0.001462 [+0.000325, +0.002599] | 0.071304 | +0.000334 [-0.000205, +0.000931] |
| Gaussian noise 0.50x | 0.098749 | +0.000975 [-0.000325, +0.002274] | 0.071194 | +0.000223 [-0.000505, +0.000904] |
| Gaussian noise 1.00x | 0.098587 | +0.000812 [-0.000325, +0.002111] | 0.071267 | +0.000297 [-0.000259, +0.000823] |

### Checkpoint 1152

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.112230 | 0 | 0.079863 | 0 |
| Zero / mask | 0.075199 | -0.037031 [-0.042716, -0.031509] | 0.040701 | -0.039162 [-0.043562, -0.034778] |
| Global train mean | 0.111743 | -0.000487 [-0.001787, +0.000975] | 0.079689 | -0.000174 [-0.000767, +0.000475] |
| Shuffled user | 0.111255 | -0.000975 [-0.002436, +0.000487] | 0.079419 | -0.000444 [-0.001091, +0.000188] |
| Shuffled target | 0.111255 | -0.000975 [-0.002436, +0.000487] | 0.079419 | -0.000444 [-0.001061, +0.000213] |
| Gaussian noise 0.25x | 0.110931 | -0.001299 [-0.002761, +0.000000] | 0.079629 | -0.000234 [-0.000802, +0.000334] |
| Gaussian noise 0.50x | 0.111418 | -0.000812 [-0.002274, +0.000487] | 0.080141 | +0.000278 [-0.000356, +0.000924] |
| Gaussian noise 1.00x | 0.111905 | -0.000325 [-0.001787, +0.001137] | 0.080047 | +0.000184 [-0.000618, +0.001004] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these tables provide point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 384

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.028423 | 0.041091 | 0.047588 | 0.059932 | 0.028423 | 0.035863 | 0.038540 | 0.042499 |
| Zero / mask | 0.013968 | 0.022414 | 0.027611 | 0.039955 | 0.013968 | 0.018892 | 0.021031 | 0.025025 |
| Global train mean | 0.027936 | 0.041904 | 0.047588 | 0.060094 | 0.027936 | 0.036110 | 0.038423 | 0.042451 |
| Shuffled user | 0.027773 | 0.041416 | 0.047751 | 0.060419 | 0.027773 | 0.035828 | 0.038428 | 0.042491 |
| Shuffled target | 0.027773 | 0.041416 | 0.047751 | 0.060419 | 0.027773 | 0.035828 | 0.038428 | 0.042491 |
| Gaussian noise 0.25x | 0.028585 | 0.041579 | 0.047751 | 0.060419 | 0.028585 | 0.036188 | 0.038732 | 0.042810 |
| Gaussian noise 0.50x | 0.028098 | 0.041741 | 0.048238 | 0.059932 | 0.028098 | 0.036089 | 0.038745 | 0.042493 |
| Gaussian noise 1.00x | 0.028585 | 0.041579 | 0.048075 | 0.060419 | 0.028585 | 0.036103 | 0.038765 | 0.042705 |

#### Checkpoint 768

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.049700 | 0.068377 | 0.079584 | 0.097775 | 0.049700 | 0.060485 | 0.065076 | 0.070970 |
| Zero / mask | 0.022089 | 0.038168 | 0.048400 | 0.067078 | 0.022089 | 0.031192 | 0.035399 | 0.041416 |
| Global train mean | 0.048725 | 0.069027 | 0.079584 | 0.097937 | 0.048725 | 0.060471 | 0.064790 | 0.070726 |
| Shuffled user | 0.048563 | 0.068540 | 0.079259 | 0.098587 | 0.048563 | 0.060167 | 0.064542 | 0.070793 |
| Shuffled target | 0.048563 | 0.068540 | 0.079259 | 0.098587 | 0.048563 | 0.060167 | 0.064542 | 0.070793 |
| Gaussian noise 0.25x | 0.048887 | 0.069352 | 0.079909 | 0.099237 | 0.048887 | 0.060757 | 0.065090 | 0.071304 |
| Gaussian noise 0.50x | 0.049212 | 0.069514 | 0.078447 | 0.098749 | 0.049212 | 0.060937 | 0.064620 | 0.071194 |
| Gaussian noise 1.00x | 0.049375 | 0.069677 | 0.079747 | 0.098587 | 0.049375 | 0.061099 | 0.065223 | 0.071267 |

#### Checkpoint 1152

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.054247 | 0.077148 | 0.088680 | 0.112230 | 0.054247 | 0.067548 | 0.072286 | 0.079863 |
| Zero / mask | 0.012506 | 0.037518 | 0.051486 | 0.075199 | 0.012506 | 0.027309 | 0.033054 | 0.040701 |
| Global train mean | 0.054572 | 0.076336 | 0.088680 | 0.111743 | 0.054572 | 0.067219 | 0.072272 | 0.079689 |
| Shuffled user | 0.054085 | 0.076986 | 0.088842 | 0.111255 | 0.054085 | 0.067300 | 0.072200 | 0.079419 |
| Shuffled target | 0.054085 | 0.076986 | 0.088842 | 0.111255 | 0.054085 | 0.067300 | 0.072200 | 0.079419 |
| Gaussian noise 0.25x | 0.054734 | 0.077148 | 0.089167 | 0.110931 | 0.054734 | 0.067749 | 0.072662 | 0.079629 |
| Gaussian noise 0.50x | 0.055547 | 0.077310 | 0.089329 | 0.111418 | 0.055547 | 0.068172 | 0.073078 | 0.080141 |
| Gaussian noise 1.00x | 0.055059 | 0.076498 | 0.089167 | 0.111905 | 0.055059 | 0.067544 | 0.072751 | 0.080047 |

All 24 generation runs returned 6,157 paired records and zero catalog violations. Raw ranking improves with training: HR@10 is 0.0599, 0.0978, and 0.1122; NDCG@10 is 0.0425, 0.0710, and 0.0799.

Zero-mask HR@10 deltas are -0.019977, -0.030697, and -0.037031; corresponding NDCG@10 deltas are -0.017474, -0.029555, and -0.039162. All six intervals exclude zero. In contrast, mean and shuffled replacements never significantly worsen either metric. All noise intervals also include zero except checkpoint 768's 0.25x HR@10 effect, which is significantly positive: +0.001462 [0.000325, 0.002599]. Its NDCG@10 interval crosses zero. Across all content-preserving and content-changing controls, no intervention provides evidence that the raw sample-specific thought identity improves ranking.

The noise null result is not explained by insufficient displacement. Expected 1.0x Gaussian L2 distances are 97.33, 87.60, and 98.14 at checkpoints 384, 768, and 1152, compared with natural shuffled-thought mean distances of 37.41, 41.12, and 50.85. Perturbations roughly 1.9–2.6 times the natural shuffled distance still leave ranking statistically unchanged.

The shuffled-user and shuffled-target decoded top-10 lists differ on 0, 0, and 0 of 6,157 examples at checkpoints 384, 768, and 1152. Where they are effectively identical, they should be treated as one control rather than independent evidence.

### Activation-versus-content interpretation

Train zero masking raises NLL by 0.8504, 0.7353, and 1.5150, while held-out zero masking significantly damages ranking. The latent slot therefore supplies a useful nonzero activation. However, a global mean, an unrelated valid thought, or large continuous noise does not worsen ranking; the only significant nonzero content-control effect is a small HR@10 improvement from 0.25x noise at checkpoint 768. This is activation dependence without beneficial sample-specific content dependence—the causal criterion for latent-utilization collapse used in the reference report.

## 5. Diagnosis

1. **Exact-template collapse: supported at all checkpoints.** Only 15.74%–18.38% of stored FP16 thoughts are unique, and the top ten templates cover 21.89%–27.75%.
2. **Shared-direction collapse: supported at all checkpoints.** Random-pair cosine is 0.825–0.907, and the mean vector retains 90.1%–95.3% of a typical sample norm.
3. **Centered dimensional collapse: strongest at checkpoint 768 and substantially weaker at checkpoint 1152.** Effective ranks are 15.44, 12.31, and 57.47.
4. **Functional/utilization collapse: strongly supported at all checkpoints.** Zeroing significantly damages held-out ranking, while mean and shuffled replacements never worsen it and even large-noise interventions do not degrade it. The isolated positive HR@10 noise effect at checkpoint 768 does not indicate useful sample-specific content.

The most defensible overall conclusion is that `latent_v6` learns increasingly accurate recommendations and, by checkpoint 1152, a much broader centered residual space. It nevertheless retains severe exact-template reuse and a large common direction, while E1 shows that the decoder does not benefit from the recovered sample-specific variation. Geometric diversity has improved, but causal latent reasoning has not.

## 6. Limitations and next action

- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.
- Exact duplicate counts are at FP16 storage precision; cosine and PCA diagnostics operate in float32 after loading.
- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the full train bank.
- Zero masking is off-manifold; mean and shuffled interventions are in-distribution controls, while Gaussian noise tests continuous perturbations.
- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking can miss probability changes that do not alter the constrained top-10 order.

The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This tests whether token probabilities share the ranking result.
