# Latent-collapse assessment for `musical_instruments_v0` checkpoints 519, 1038, and 1557

## Executive verdict

**All three `musical_instruments_v0` checkpoints are strongly shared-direction and exact-template concentrated. Unlike the CDs_and_Vinyl report family, centered geometric diversity here improves early and then plateaus, while exact-template collapse gets dramatically worse at the final checkpoint rather than recovering.**

Only 25.05%–77.44% of stored FP16 thoughts are unique, and the ten most common templates cover 5.74%–55.35% of train examples. Checkpoint 1557 (the final epoch) is by far the most template concentrated — its top-10 templates alone cover 55.35% of train examples, roughly ten times checkpoint 1038's 5.74%. Centered entropy effective rank rises from 2.03 (checkpoint 519) to 2.69 (checkpoint 1038) and then holds essentially flat at 2.68 (checkpoint 1557); dimensional diversity does not track the collapse in unique-vector count at all.

All checkpoints remain anisotropic: random-pair cosine is 0.890–0.943, and the global-mean norm is 94.4%–97.2% of the mean sample norm. The last checkpoint improves residual diversity without eliminating the common carrier direction.

Zero masking significantly reduces HR@10 and NDCG@10 at every checkpoint, while the global-mean, shuffled, and Gaussian-noise controls never significantly worsen ranking. The decoder therefore depends on a nonzero latent activation but does not benefit from the measured sample-specific content. **E1 strongly supports latent-utilization collapse at all three checkpoints.**

## Scope and reproducibility

- Dataset: `Musical_Instruments`
- Geometry and extraction likelihood split: train, 66,500 examples
- Causal ranking split: test, 8,313 examples
- Architecture output: one 1,536-dimensional stored thought per interaction
- Stored dtype: FP16
- Train split SHA-256: `6497a7a06f9d090ec3d35b1f3a18032e00e0b5a0ceb3bb510ed80a62ac5cd5fd`
- Test split SHA-256: `d98cc4d2fa70db7633bfb65bbbb5e1f8549f01dad86ee96ce76c9ec2106bc281`
- Pairwise samples: 100,000 per diagnostic, seed 42
- PCA: centered and fitted on all 66,500 thoughts independently at each checkpoint
- Ranking: constrained 10-beam catalog generation; paired bootstrap 95% intervals over all 8,313 test examples, 2,000 resamples, seed 42

Primary artifacts:

- [Epoch comparison summary](../results/latent_distribution/latent_musical_instruments_v0_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_musical_instruments_v0_epoch_collapse_trajectory.png)
- Checkpoint statistics: [519](../results/latent_distribution/musical_instruments_v0_checkpoint_519/train_latent_distribution.json), [1038](../results/latent_distribution/musical_instruments_v0_checkpoint_1038/train_latent_distribution.json), [1557](../results/latent_distribution/musical_instruments_v0_checkpoint_1557/train_latent_distribution.json)
- Distribution overviews: [519](../figures/musical_instruments_v0_checkpoint_519_latent_distribution_overview.png), [1038](../figures/musical_instruments_v0_checkpoint_1038_latent_distribution_overview.png), [1557](../figures/musical_instruments_v0_checkpoint_1557_latent_distribution_overview.png)
- All-vector PCA views: [519](../figures/musical_instruments_v0_checkpoint_519_latent_pca_all_scatter.png), [1038](../figures/musical_instruments_v0_checkpoint_1038_latent_pca_all_scatter.png), [1557](../figures/musical_instruments_v0_checkpoint_1557_latent_pca_all_scatter.png)
- Interactive t-SNE (all train thoughts, `visualize_reasoning.py --method tsne`, perplexity 30, seed 42, hover shows `user_id`): [519](../figures/musical_instruments_v0_checkpoint_519_raw_latent_tsne_plotly.html), [1038](../figures/musical_instruments_v0_checkpoint_1038_raw_latent_tsne_plotly.html), [1557](../figures/musical_instruments_v0_checkpoint_1557_raw_latent_tsne_plotly.html)
- Interactive PCA (same tool, `visualize_reasoning.py --method pca`, first two components, hover shows `user_id`): [519](../figures/musical_instruments_v0_checkpoint_519_raw_latent_pca_plotly.html), [1038](../figures/musical_instruments_v0_checkpoint_1038_raw_latent_pca_plotly.html), [1557](../figures/musical_instruments_v0_checkpoint_1557_raw_latent_pca_plotly.html)
- E1 ranking artifacts: [519](../results/intervention/musical_instruments_v0_checkpoint_519/), [1038](../results/intervention/musical_instruments_v0_checkpoint_1038/), [1557](../results/intervention/musical_instruments_v0_checkpoint_1557/)

## 1. Epoch trajectory

| Diagnostic | Step 519 (epoch 1) | Step 1038 (epoch 2) | Step 1557 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 0.392596 | 0.309391 | 0.241733 |
| Zero-mask train target NLL | 0.818712 | 0.771051 | 0.805350 |
| Zero-mask NLL penalty | +0.426116 | +0.461660 | +0.563617 |
| Mean sample norm | 87.49 | 88.73 | 100.05 |
| Global-mean vector norm | 85.03 | 85.48 | 94.42 |
| Global-mean norm / mean sample norm | 0.9719 | 0.9633 | 0.9437 |
| Mean cosine to global mean | 0.9709 | 0.9621 | 0.9435 |
| Mean random-pair cosine | 0.9428 | 0.9257 | 0.8904 |
| Exact FP16-unique vectors | 36,170 | 51,499 | 16,660 |
| Unique fraction | 54.39% | 77.44% | 25.05% |
| Top-10-template coverage | 19.05% | 5.74% | 55.35% |
| Largest-template coverage | 5.80% | 1.04% | 35.77% |
| PC1 centered variance | 78.05% | 71.60% | 72.84% |
| Top-two centered variance | 95.09% | 82.39% | 82.22% |
| Entropy effective rank | 2.03 | 2.69 | 2.68 |
| Participation ratio | 1.56 | 1.86 | 1.82 |
| Components for 90% variance | 2 | 3 | 3 |

Raw train NLL improves monotonically, but collapse metrics do not track it. Effective rank rises from 2.03 to 2.69 between checkpoints 519 and 1038, then holds flat at 2.68 by checkpoint 1557 — centered dimensional diversity plateaus rather than continuing to improve. At checkpoint 1557, PC1 still explains 72.84% of centered variance and 3 components are needed for 90%, identical to checkpoint 1038's 3. Meanwhile exact-template reuse gets much worse over the same interval (top-10 coverage 5.74% → 55.35%), so the best-fitting summary is a plateau in centered geometry alongside a sharp late regression into discrete template reuse, not a late recovery.

## 2. Template concentration versus dimensional concentration

Exact-template collapse is strong at every checkpoint and is worst at the final one. The unique-vector counts are 36,170, 51,499, and 16,660 among 66,500 samples — checkpoint 1557 has the fewest unique vectors of the three despite being the most-trained. The largest single templates cover 5.80%, 1.04%, and 35.77% of the train split, respectively; at checkpoint 1557 a single stored thought is reused for over a third of all training examples.

Dimensional concentration and exact repetition move independently late in training. Checkpoint 1557's centered PCA geometry (effective rank 2.68, first-ten-PC share 99.90%) is nearly identical to checkpoint 1038's (2.69, 99.94%), yet its exact-uniqueness collapses from 77.44% to 25.05% and its top-10-template coverage rises nearly tenfold (5.74% → 55.35%). A stable centered span therefore provides no protection against a late, severe regression into discrete template reuse.

## 3. User-, target-, and preference-linked structure

| Diagnostic | Step 519 | Step 1038 | Step 1557 |
|---|---:|---:|---:|
| Same-user cosine (absolute) | 0.9401 | 0.9237 | 0.8859 |
| Same-target cosine (absolute) | 0.9433 | 0.9269 | 0.8922 |
| Same CF-cluster cosine (absolute) | 0.9431 | 0.9263 | 0.8909 |
| Adjusted variance associated with user | 4.15% | 5.12% | 4.21% |
| Adjusted variance associated with target | 1.36% | 1.23% | 1.74% |
| Adjusted variance associated with CF cluster | 0.18% | 0.30% | 0.20% |

The table reports absolute within-group cosine values. Random-pair baselines are 0.9428, 0.9257, and 0.8904. User identity is the strongest of the measured grouping variables at all checkpoints; target-associated variance is smaller, and conventional collaborative-filtering cluster structure is weak. These associations show that the latent bank is not arbitrary noise, but they do not establish that the decoder uses the variation; that causal question is tested by E1 below.

The exact-template utility eta-squared values are not treated as reliable utility evidence because the many small template groups make that statistic structurally prone to overfitting.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 519

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.114880 | 0 | 0.088959 | 0 |
| Zero / mask | 0.091062 | -0.023818 [-0.027908, -0.019728] | 0.060371 | -0.028588 [-0.031507, -0.025593] |
| Global train mean | 0.115361 | +0.000481 [-0.000842, +0.001684] | 0.089290 | +0.000330 [-0.000234, +0.000916] |
| Shuffled user | 0.116204 | +0.001323 [+0.000000, +0.002767] | 0.089509 | +0.000550 [-0.000065, +0.001169] |
| Shuffled target | 0.116204 | +0.001323 [-0.000120, +0.002646] | 0.089509 | +0.000550 [-0.000051, +0.001185] |
| Gaussian noise 0.25x | 0.115722 | +0.000842 [-0.000241, +0.002045] | 0.089410 | +0.000450 [-0.000034, +0.000955] |
| Gaussian noise 0.50x | 0.116083 | +0.001203 [-0.000120, +0.002526] | 0.089480 | +0.000521 [-0.000065, +0.001143] |
| Gaussian noise 1.00x | 0.115843 | +0.000962 [-0.000481, +0.002289] | 0.089375 | +0.000416 [-0.000200, +0.001020] |

### Checkpoint 1038

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.121857 | 0 | 0.094444 | 0 |
| Zero / mask | 0.097678 | -0.024179 [-0.028630, -0.019969] | 0.070991 | -0.023453 [-0.026544, -0.020458] |
| Global train mean | 0.121617 | -0.000241 [-0.001684, +0.001203] | 0.094529 | +0.000084 [-0.000657, +0.000842] |
| Shuffled user | 0.122218 | +0.000361 [-0.001083, +0.001684] | 0.094800 | +0.000356 [-0.000289, +0.001135] |
| Shuffled target | 0.122218 | +0.000361 [-0.000962, +0.001684] | 0.094800 | +0.000356 [-0.000300, +0.001056] |
| Gaussian noise 0.25x | 0.121015 | -0.000842 [-0.002406, +0.000722] | 0.094114 | -0.000330 [-0.000973, +0.000352] |
| Gaussian noise 0.50x | 0.121617 | -0.000241 [-0.001684, +0.001203] | 0.094305 | -0.000139 [-0.000795, +0.000532] |
| Gaussian noise 1.00x | 0.120654 | -0.001203 [-0.002887, +0.000364] | 0.093991 | -0.000453 [-0.001127, +0.000247] |

### Checkpoint 1557

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.123060 | 0 | 0.095716 | 0 |
| Zero / mask | 0.091904 | -0.031156 [-0.036088, -0.026104] | 0.064864 | -0.030852 [-0.034726, -0.027252] |
| Global train mean | 0.124023 | +0.000962 [-0.000481, +0.002406] | 0.095775 | +0.000059 [-0.000679, +0.000754] |
| Shuffled user | 0.124263 | +0.001203 [-0.000361, +0.002767] | 0.096337 | +0.000621 [-0.000210, +0.001476] |
| Shuffled target | 0.124263 | +0.001203 [-0.000481, +0.002767] | 0.096337 | +0.000621 [-0.000162, +0.001479] |
| Gaussian noise 0.25x | 0.123782 | +0.000722 [-0.000601, +0.002045] | 0.095952 | +0.000236 [-0.000476, +0.000926] |
| Gaussian noise 0.50x | 0.124023 | +0.000962 [-0.000481, +0.002286] | 0.096040 | +0.000324 [-0.000334, +0.000960] |
| Gaussian noise 1.00x | 0.124504 | +0.001444 [-0.000241, +0.003128] | 0.096126 | +0.000410 [-0.000404, +0.001213] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these tables provide point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 519

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.069770 | 0.083604 | 0.096475 | 0.114880 | 0.069770 | 0.077821 | 0.083053 | 0.088959 |
| Zero / mask | 0.032600 | 0.061109 | 0.071695 | 0.091062 | 0.032600 | 0.049705 | 0.054074 | 0.060371 |
| Global train mean | 0.070131 | 0.084446 | 0.095874 | 0.115361 | 0.070131 | 0.078360 | 0.083007 | 0.089290 |
| Shuffled user | 0.070011 | 0.084205 | 0.096235 | 0.116204 | 0.070011 | 0.078195 | 0.083091 | 0.089509 |
| Shuffled target | 0.070011 | 0.084205 | 0.096235 | 0.116204 | 0.070011 | 0.078195 | 0.083091 | 0.089509 |
| Gaussian noise 0.25x | 0.070011 | 0.084927 | 0.096235 | 0.115722 | 0.070011 | 0.078587 | 0.083151 | 0.089410 |
| Gaussian noise 0.50x | 0.070131 | 0.084326 | 0.096355 | 0.116083 | 0.070131 | 0.078252 | 0.083148 | 0.089480 |
| Gaussian noise 1.00x | 0.070131 | 0.084085 | 0.095754 | 0.115843 | 0.070131 | 0.078179 | 0.082930 | 0.089375 |

#### Checkpoint 1038

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.073981 | 0.090220 | 0.100926 | 0.121857 | 0.073981 | 0.083282 | 0.087692 | 0.094444 |
| Zero / mask | 0.050523 | 0.066763 | 0.079514 | 0.097678 | 0.050523 | 0.059887 | 0.065173 | 0.070991 |
| Global train mean | 0.074101 | 0.090461 | 0.101528 | 0.121617 | 0.074101 | 0.083494 | 0.088028 | 0.094529 |
| Shuffled user | 0.074101 | 0.090701 | 0.102370 | 0.122218 | 0.074101 | 0.083614 | 0.088418 | 0.094800 |
| Shuffled target | 0.074101 | 0.090701 | 0.102370 | 0.122218 | 0.074101 | 0.083614 | 0.088418 | 0.094800 |
| Gaussian noise 0.25x | 0.073740 | 0.090701 | 0.101287 | 0.121015 | 0.073740 | 0.083418 | 0.087745 | 0.094114 |
| Gaussian noise 0.50x | 0.073740 | 0.090100 | 0.100686 | 0.121617 | 0.073740 | 0.083180 | 0.087539 | 0.094305 |
| Gaussian noise 1.00x | 0.073981 | 0.090220 | 0.100926 | 0.120654 | 0.073981 | 0.083234 | 0.087619 | 0.093991 |

#### Checkpoint 1557

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.074943 | 0.091423 | 0.103573 | 0.123060 | 0.074943 | 0.084522 | 0.089470 | 0.095716 |
| Zero / mask | 0.044268 | 0.061109 | 0.073860 | 0.091904 | 0.044268 | 0.053838 | 0.059109 | 0.064864 |
| Global train mean | 0.074823 | 0.091062 | 0.103452 | 0.124023 | 0.074823 | 0.084139 | 0.089196 | 0.095775 |
| Shuffled user | 0.075304 | 0.092506 | 0.104294 | 0.124263 | 0.075304 | 0.085133 | 0.089942 | 0.096337 |
| Shuffled target | 0.075304 | 0.092506 | 0.104294 | 0.124263 | 0.075304 | 0.085133 | 0.089942 | 0.096337 |
| Gaussian noise 0.25x | 0.075063 | 0.091664 | 0.103452 | 0.123782 | 0.075063 | 0.084592 | 0.089400 | 0.095952 |
| Gaussian noise 0.50x | 0.075063 | 0.092145 | 0.103452 | 0.124023 | 0.075063 | 0.084848 | 0.089449 | 0.096040 |
| Gaussian noise 1.00x | 0.075063 | 0.091664 | 0.103332 | 0.124504 | 0.075063 | 0.084592 | 0.089333 | 0.096126 |

All 24 generation runs returned 8,313 paired records and zero catalog violations. Raw ranking improves with training: HR@10 is 0.1149, 0.1219, and 0.1231; NDCG@10 is 0.0890, 0.0944, and 0.0957.

Zero-mask HR@10 deltas are -0.023818, -0.024179, and -0.031156; corresponding NDCG@10 deltas are -0.028588, -0.023453, and -0.030852. Across the mean, shuffle, and noise controls, the largest absolute point changes are 0.001444 HR@10 and 0.000621 NDCG@10. The confidence intervals above determine whether those changes are distinguishable from zero.

The shuffled-user and shuffled-target decoded top-10 lists differ on 9, 9, and 7 of 8,313 examples at checkpoints 519, 1038, and 1557. Where they are effectively identical, they should be treated as one control rather than independent evidence.

### Activation-versus-content interpretation

Train zero masking raises NLL by 0.4261, 0.4617, and 0.5636. Held-out E1 separately asks whether ranking depends on the activation's identity. A damaging zero intervention combined with null mean/shuffle/noise effects is activation dependence without sample-specific content dependence—the causal criterion for latent-utilization collapse used in the reference report.

## 5. Diagnosis

1. **Exact-template collapse: supported at all checkpoints, and most severe at the final one.** Only 25.05%–77.44% of stored FP16 thoughts are unique, and the top ten templates cover 5.74%–55.35% — both the worst uniqueness and the worst template concentration occur at checkpoint 1557, not at an earlier checkpoint.
2. **Shared-direction collapse: supported at all checkpoints.** Random-pair cosine is 0.890–0.943, and the mean vector retains 94.4%–97.2% of a typical sample norm; both decline modestly and monotonically with more training.
3. **Centered dimensional collapse: worst at checkpoint 519 and then plateaus.** Effective ranks are 2.03, 2.69, and 2.68 — a sizeable early improvement (519→1038) followed by no further change (1038→1557).
4. **Functional/utilization collapse: strongly supported at all checkpoints.** Zeroing damages ranking while mean, shuffled, and large-noise interventions never significantly worsen it.

The most defensible overall conclusion is that `musical_instruments_v0` learns increasingly accurate recommendations, but its geometric trajectory is not the same story as CDs_and_Vinyl's: centered dimensional diversity improves early and then plateaus rather than continuing to broaden, and exact-template reuse gets substantially *worse* at the final checkpoint even as raw ranking keeps improving. Geometric diversity, exact-template reuse, and causal utilization move independently here and must be evaluated separately.

## 6. Limitations and next action

- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.
- Exact duplicate counts are at FP16 storage precision; cosine and PCA diagnostics operate in float32 after loading.
- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the full train bank.
- Zero masking is off-manifold; mean and shuffled interventions are in-distribution controls, while Gaussian noise tests continuous perturbations.
- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking can miss probability changes that do not alter the constrained top-10 order.

The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This tests whether token probabilities share the ranking result.
