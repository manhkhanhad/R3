# Latent-collapse assessment for `toys_and_games_v0` checkpoints 422, 844, and 1263

## Executive verdict

**Unlike either the CDs_and_Vinyl or Musical_Instruments report families, `toys_and_games_v0`'s geometry improves nearly monotonically across every collapse metric measured. There is no dip-then-recover pattern and no late regression into template reuse — checkpoint 1263 is the least collapsed checkpoint on every geometric axis, though it is still far from an isotropic, fully diverse latent space.**

Only 43.26%–56.74% of stored FP16 thoughts are unique, and the ten most common templates cover 16.24%–49.14% of train examples. Checkpoints 422 and 844 are both heavily template-concentrated and nearly tied (48.70% vs. 49.14% top-10 coverage); checkpoint 1263 is a roughly three-fold improvement (16.24%). Centered entropy effective rank rises monotonically from 58.54 to 65.19 to 300.84 — the largest jump comes at the final checkpoint, and dimensional diversity is worst at the *first* checkpoint, not the middle one.

All checkpoints remain anisotropic, but decreasingly so: random-pair cosine falls monotonically from 0.719 to 0.538 to 0.423, and the global-mean norm falls from 85.0% to 73.7% to 65.3% of the mean sample norm. Every measured checkpoint here is more diverse on these axes than any checkpoint in the CDs_and_Vinyl or Musical_Instruments reports, though 0.42 random-pair cosine is still well above the near-zero value expected of unrelated 1,536-dimensional vectors.

E1 does not match the simple activation-without-content pattern uniformly across all three checkpoints; the checkpoint-level confidence intervals below should be used for the causal conclusion.

## Scope and reproducibility

- Dataset: `Toys_and_Games`
- Geometry and extraction likelihood split: train, 53,898 examples
- Causal ranking split: test, 6,738 examples
- Architecture output: one 1,536-dimensional stored thought per interaction
- Stored dtype: FP16
- Train split SHA-256: `02eae8dd4a0fa2aa3c8168f448e55dab9dfb8e269fc9d0bd2f5dacdcfee23aac`
- Test split SHA-256: `a0914268088386500fb5c85d5c3d183b76bd3b70d34d3dd65facf58e0cd8fec0`
- Pairwise samples: 100,000 per diagnostic, seed 42
- PCA: centered and fitted on all 53,898 thoughts independently at each checkpoint
- Ranking: constrained 10-beam catalog generation; paired bootstrap 95% intervals over all 6,738 test examples, 2,000 resamples, seed 42

Primary artifacts:

- [Epoch comparison summary](../results/latent_distribution/latent_toys_and_games_v0_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_toys_and_games_v0_epoch_collapse_trajectory.png)
- Checkpoint statistics: [422](../results/latent_distribution/toys_and_games_v0_checkpoint_422/train_latent_distribution.json), [844](../results/latent_distribution/toys_and_games_v0_checkpoint_844/train_latent_distribution.json), [1263](../results/latent_distribution/toys_and_games_v0_checkpoint_1263/train_latent_distribution.json)
- Distribution overviews: [422](../figures/toys_and_games_v0_checkpoint_422_latent_distribution_overview.png), [844](../figures/toys_and_games_v0_checkpoint_844_latent_distribution_overview.png), [1263](../figures/toys_and_games_v0_checkpoint_1263_latent_distribution_overview.png)
- All-vector PCA views: [422](../figures/toys_and_games_v0_checkpoint_422_latent_pca_all_scatter.png), [844](../figures/toys_and_games_v0_checkpoint_844_latent_pca_all_scatter.png), [1263](../figures/toys_and_games_v0_checkpoint_1263_latent_pca_all_scatter.png)
- Interactive t-SNE (all train thoughts, `visualize_reasoning.py --method tsne`, perplexity 30, seed 42, hover shows `user_id`): [422](../figures/toys_and_games_v0_checkpoint_422_raw_latent_tsne_plotly.html), [844](../figures/toys_and_games_v0_checkpoint_844_raw_latent_tsne_plotly.html), [1263](../figures/toys_and_games_v0_checkpoint_1263_raw_latent_tsne_plotly.html)
- Interactive PCA (same tool, `visualize_reasoning.py --method pca`, first two components, hover shows `user_id`): [422](../figures/toys_and_games_v0_checkpoint_422_raw_latent_pca_plotly.html), [844](../figures/toys_and_games_v0_checkpoint_844_raw_latent_pca_plotly.html), [1263](../figures/toys_and_games_v0_checkpoint_1263_raw_latent_pca_plotly.html)
- E1 ranking artifacts: [422](../results/intervention/toys_and_games_v0_checkpoint_422/), [844](../results/intervention/toys_and_games_v0_checkpoint_844/), [1263](../results/intervention/toys_and_games_v0_checkpoint_1263/)

## 1. Epoch trajectory

| Diagnostic | Step 422 (epoch 1) | Step 844 (epoch 2) | Step 1263 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 0.615809 | 0.418278 | 0.287901 |
| Zero-mask train target NLL | 0.658880 | 0.426287 | 0.304419 |
| Zero-mask NLL penalty | +0.043071 | +0.008009 | +0.016518 |
| Mean sample norm | 77.81 | 100.94 | 115.04 |
| Global-mean vector norm | 66.16 | 74.42 | 75.08 |
| Global-mean norm / mean sample norm | 0.8504 | 0.7372 | 0.6526 |
| Mean cosine to global mean | 0.8480 | 0.7332 | 0.6497 |
| Mean random-pair cosine | 0.7192 | 0.5385 | 0.4227 |
| Exact FP16-unique vectors | 23,315 | 23,807 | 30,580 |
| Unique fraction | 43.26% | 44.17% | 56.74% |
| Top-10-template coverage | 48.70% | 49.14% | 16.24% |
| Largest-template coverage | 13.94% | 9.38% | 12.82% |
| PC1 centered variance | 29.98% | 21.08% | 10.22% |
| Top-two centered variance | 40.10% | 41.75% | 16.33% |
| Entropy effective rank | 58.54 | 65.19 | 300.84 |
| Participation ratio | 9.30 | 10.75 | 57.37 |
| Components for 90% variance | 228 | 277 | 494 |

Raw train NLL and centered-geometry collapse metrics improve together here. Effective rank rises from 58.54 to 65.19 to 300.84 — a modest early gain followed by a much larger one at the final checkpoint. PC1's centered-variance share falls the same way: 29.98% → 21.08% → 10.22%, and components needed for 90% variance rise from 228 to 277 to 494. Unlike the CDs_and_Vinyl and Musical_Instruments reports, there is no dip at the middle checkpoint to recover from — the trajectory is monotonic (or very close to it) on every centered-geometry metric measured.

## 2. Template concentration versus dimensional concentration

Exact-template collapse is strong at every checkpoint, though it eases substantially at the last one. The unique-vector counts are 23,315, 23,807, and 30,580 among 53,898 samples — checkpoints 422 and 844 are nearly tied and both much worse than checkpoint 1263. The largest single templates cover 13.94%, 9.38%, and 12.82% of the train split, respectively.

Dimensional concentration and exact repetition move together here, unlike in the other report families. Checkpoint 1263 expands to effective rank 300.84 and reduces the first-ten-PC share to 27.27% (from ~58% at both earlier checkpoints), and its exact-uniqueness and top-10-template coverage improve in the same direction (56.74% unique and 16.24% top-10 coverage, versus roughly 43-44% unique and ~49% top-10 coverage earlier). The improvement is real but incomplete: even at checkpoint 1263, a single template still covers 12.82% of the train split and 43.26% of stored vectors are exact duplicates of another.

## 3. User-, target-, and preference-linked structure

| Diagnostic | Step 422 | Step 844 | Step 1263 |
|---|---:|---:|---:|
| Same-user cosine (absolute) | 0.7370 | 0.5563 | 0.4980 |
| Same-target cosine (absolute) | 0.7298 | 0.5488 | 0.4403 |
| Same CF-cluster cosine (absolute) | 0.7220 | 0.5417 | 0.4280 |
| Adjusted variance associated with user | 1.70% | 1.02% | 7.78% |
| Adjusted variance associated with target | 3.22% | 2.04% | 2.26% |
| Adjusted variance associated with CF cluster | 1.11% | 0.66% | 0.96% |

The table reports absolute within-group cosine values. Random-pair baselines are 0.7192, 0.5385, and 0.4227. User identity is the strongest of the measured grouping variables at all checkpoints; target-associated variance is smaller, and conventional collaborative-filtering cluster structure is weak. These associations show that the latent bank is not arbitrary noise, but they do not establish that the decoder uses the variation; that causal question is tested by E1 below.

The exact-template utility eta-squared values are not treated as reliable utility evidence because the many small template groups make that statistic structurally prone to overfitting.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 422

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.079549 | 0 | 0.047760 | 0 |
| Zero / mask | 0.076729 | -0.002820 [-0.005640, +0.000004] | 0.044350 | -0.003410 [-0.004943, -0.001956] |
| Global train mean | 0.079994 | +0.000445 [-0.000742, +0.001633] | 0.048007 | +0.000247 [-0.000167, +0.000723] |
| Shuffled user | 0.078510 | -0.001039 [-0.002226, +0.000148] | 0.047184 | -0.000576 [-0.001182, -0.000039] |
| Shuffled target | 0.078510 | -0.001039 [-0.002230, +0.000148] | 0.047184 | -0.000576 [-0.001240, -0.000043] |
| Gaussian noise 0.25x | 0.079549 | +0.000000 [-0.001336, +0.001187] | 0.047857 | +0.000096 [-0.000418, +0.000644] |
| Gaussian noise 0.50x | 0.078658 | -0.000890 [-0.002078, +0.000297] | 0.047494 | -0.000266 [-0.000755, +0.000270] |
| Gaussian noise 1.00x | 0.078955 | -0.000594 [-0.001781, +0.000742] | 0.047419 | -0.000341 [-0.000892, +0.000165] |

### Checkpoint 844

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.108341 | 0 | 0.066372 | 0 |
| Zero / mask | 0.108341 | +0.000000 [-0.002523, +0.002523] | 0.066360 | -0.000012 [-0.001263, +0.001288] |
| Global train mean | 0.108341 | +0.000000 [-0.001633, +0.001633] | 0.066611 | +0.000239 [-0.000495, +0.000985] |
| Shuffled user | 0.108341 | +0.000000 [-0.001484, +0.001484] | 0.066766 | +0.000395 [-0.000293, +0.001140] |
| Shuffled target | 0.108341 | +0.000000 [-0.001484, +0.001633] | 0.066766 | +0.000395 [-0.000314, +0.001102] |
| Gaussian noise 0.25x | 0.108786 | +0.000445 [-0.000890, +0.001781] | 0.066829 | +0.000457 [-0.000148, +0.001056] |
| Gaussian noise 0.50x | 0.108341 | +0.000000 [-0.001336, +0.001336] | 0.066607 | +0.000236 [-0.000332, +0.000829] |
| Gaussian noise 1.00x | 0.109083 | +0.000742 [-0.000742, +0.002375] | 0.066880 | +0.000509 [-0.000233, +0.001298] |

### Checkpoint 1263

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.106115 | 0 | 0.067831 | 0 |
| Zero / mask | 0.103295 | -0.002820 [-0.005640, -0.000148] | 0.066464 | -0.001368 [-0.002591, -0.000132] |
| Global train mean | 0.105076 | -0.001039 [-0.002671, +0.000594] | 0.067339 | -0.000492 [-0.001164, +0.000176] |
| Shuffled user | 0.107153 | +0.001039 [-0.000742, +0.002820] | 0.067854 | +0.000023 [-0.000675, +0.000719] |
| Shuffled target | 0.107153 | +0.001039 [-0.000890, +0.002820] | 0.067854 | +0.000023 [-0.000663, +0.000774] |
| Gaussian noise 0.25x | 0.106115 | +0.000000 [-0.001336, +0.001187] | 0.067998 | +0.000167 [-0.000352, +0.000668] |
| Gaussian noise 0.50x | 0.105521 | -0.000594 [-0.001781, +0.000594] | 0.067372 | -0.000459 [-0.001002, +0.000084] |
| Gaussian noise 1.00x | 0.105373 | -0.000742 [-0.002523, +0.001039] | 0.067493 | -0.000338 [-0.001062, +0.000400] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these tables provide point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 422

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.024785 | 0.040516 | 0.055506 | 0.079549 | 0.024785 | 0.033914 | 0.040057 | 0.047760 |
| Zero / mask | 0.020778 | 0.037251 | 0.051944 | 0.076729 | 0.020778 | 0.030278 | 0.036293 | 0.044350 |
| Global train mean | 0.024785 | 0.041110 | 0.055654 | 0.079994 | 0.024785 | 0.034288 | 0.040201 | 0.048007 |
| Shuffled user | 0.024191 | 0.040516 | 0.055358 | 0.078510 | 0.024191 | 0.033695 | 0.039755 | 0.047184 |
| Shuffled target | 0.024191 | 0.040516 | 0.055358 | 0.078510 | 0.024191 | 0.033695 | 0.039755 | 0.047184 |
| Gaussian noise 0.25x | 0.024785 | 0.040962 | 0.056100 | 0.079549 | 0.024785 | 0.034195 | 0.040363 | 0.047857 |
| Gaussian noise 0.50x | 0.024933 | 0.040665 | 0.055654 | 0.078658 | 0.024933 | 0.034023 | 0.040134 | 0.047494 |
| Gaussian noise 1.00x | 0.024488 | 0.040071 | 0.055358 | 0.078955 | 0.024488 | 0.033620 | 0.039853 | 0.047419 |

#### Checkpoint 844

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.034135 | 0.061443 | 0.077916 | 0.108341 | 0.034135 | 0.049829 | 0.056592 | 0.066372 |
| Zero / mask | 0.034283 | 0.059810 | 0.078510 | 0.108341 | 0.034283 | 0.049048 | 0.056744 | 0.066360 |
| Global train mean | 0.034580 | 0.061146 | 0.078213 | 0.108341 | 0.034580 | 0.049961 | 0.056954 | 0.066611 |
| Shuffled user | 0.034728 | 0.061294 | 0.078362 | 0.108341 | 0.034728 | 0.050110 | 0.057142 | 0.066766 |
| Shuffled target | 0.034728 | 0.061294 | 0.078362 | 0.108341 | 0.034728 | 0.050110 | 0.057142 | 0.066766 |
| Gaussian noise 0.25x | 0.034432 | 0.061739 | 0.078213 | 0.108786 | 0.034432 | 0.050204 | 0.056980 | 0.066829 |
| Gaussian noise 0.50x | 0.034728 | 0.061739 | 0.078510 | 0.108341 | 0.034728 | 0.050196 | 0.057042 | 0.066607 |
| Gaussian noise 1.00x | 0.034580 | 0.061591 | 0.078658 | 0.109083 | 0.034580 | 0.050068 | 0.057106 | 0.066880 |

#### Checkpoint 1263

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.039181 | 0.062036 | 0.077768 | 0.106115 | 0.039181 | 0.052338 | 0.058755 | 0.067831 |
| Zero / mask | 0.038290 | 0.061443 | 0.076877 | 0.103295 | 0.038290 | 0.051674 | 0.057963 | 0.066464 |
| Global train mean | 0.038736 | 0.061739 | 0.077768 | 0.105076 | 0.038736 | 0.052006 | 0.058577 | 0.067339 |
| Shuffled user | 0.038142 | 0.061888 | 0.078807 | 0.107153 | 0.038142 | 0.051841 | 0.058777 | 0.067854 |
| Shuffled target | 0.038142 | 0.061888 | 0.078807 | 0.107153 | 0.038142 | 0.051841 | 0.058777 | 0.067854 |
| Gaussian noise 0.25x | 0.039329 | 0.061146 | 0.078065 | 0.106115 | 0.039329 | 0.052006 | 0.058974 | 0.067998 |
| Gaussian noise 0.50x | 0.037993 | 0.062333 | 0.078658 | 0.105521 | 0.037993 | 0.052087 | 0.058728 | 0.067372 |
| Gaussian noise 1.00x | 0.038587 | 0.062036 | 0.078658 | 0.105373 | 0.038587 | 0.052119 | 0.058946 | 0.067493 |

All 24 generation runs returned 6,738 paired records and zero catalog violations. Raw ranking improves with training: HR@10 is 0.0795, 0.1083, and 0.1061; NDCG@10 is 0.0478, 0.0664, and 0.0678.

Zero-mask HR@10 deltas are -0.002820, +0.000000, and -0.002820; corresponding NDCG@10 deltas are -0.003410, -0.000012, and -0.001368. Across the mean, shuffle, and noise controls, the largest absolute point changes are 0.001039 HR@10 and 0.000576 NDCG@10. The confidence intervals above determine whether those changes are distinguishable from zero.

The shuffled-user and shuffled-target decoded top-10 lists differ on 2, 1, and 3 of 6,738 examples at checkpoints 422, 844, and 1263. Where they are effectively identical, they should be treated as one control rather than independent evidence.

### Activation-versus-content interpretation

Train zero masking raises NLL by only 0.0431, 0.0080, and 0.0165 — an order of magnitude smaller than the penalties seen in the CDs_and_Vinyl and Musical_Instruments reports, indicating a much weaker activation-dependence signal here overall. Checkpoint 844 is a striking exception: masking changes train NLL by only 0.008 and produces an exactly zero HR@10 delta on held-out ranking (+0.000000), meaning the decoder at that checkpoint gets essentially no measurable benefit from a nonzero latent activation at all, let alone its sample-specific content. Checkpoints 422 and 1263 show small, borderline-significant zero-mask penalties instead. Held-out E1 separately asks whether ranking depends on the activation's identity; at no checkpoint here does any content-preserving control (mean, shuffle, noise) show a robust, significant improvement over raw, so there is no evidence at any checkpoint that the decoder uses sample-specific content — only weak and inconsistent evidence that it needs any activation at all.

## 5. Diagnosis

1. **Exact-template collapse: supported at all checkpoints.** Only 43.26%–56.74% of stored FP16 thoughts are unique, and the top ten templates cover 16.24%–49.14%.
2. **Shared-direction collapse: supported at all checkpoints.** Random-pair cosine is 0.423–0.719, and the mean vector retains 65.3%–85.0% of a typical sample norm.
3. **Centered dimensional collapse: worst at checkpoint 422 and improves monotonically through checkpoint 1263.** Effective ranks are 58.54, 65.19, and 300.84 — there is no dip at the middle checkpoint in this dataset, unlike the CDs_and_Vinyl and Musical_Instruments reports.
4. **Functional/utilization collapse: mixed.** At least one checkpoint or content control departs from the uniform null pattern; use the paired intervals above.

The most defensible overall conclusion is that `toys_and_games_v0` learns increasingly accurate recommendations alongside a genuinely and monotonically improving centered geometry: effective rank, PC1 concentration, random-pair cosine, and (from checkpoint 844 onward) exact-template reuse all move in the healthier direction together, with the largest gains concentrated at the final checkpoint. This is a materially different trajectory from the other datasets in this report family, where geometric and template-collapse metrics decoupled or reversed late in training. The improvement is nevertheless incomplete — checkpoint 1263 still has under 57% unique vectors and a single template covering nearly 13% of train examples — and, per point 4, does not obviously translate into the decoder causally using sample-specific content.

## 6. Limitations and next action

- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.
- Exact duplicate counts are at FP16 storage precision; cosine and PCA diagnostics operate in float32 after loading.
- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the full train bank.
- Zero masking is off-manifold; mean and shuffled interventions are in-distribution controls, while Gaussian noise tests continuous perturbations.
- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking can miss probability changes that do not alter the constrained top-10 order.

The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This tests whether token probabilities share the ranking result.
