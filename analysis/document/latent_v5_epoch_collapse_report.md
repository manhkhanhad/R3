# Latent-collapse assessment for `latent_v5` checkpoints 384, 768, and 1152

## Executive verdict

**Yes. `latent_v5` exhibits geometric latent collapse at all three checkpoints, with a changing failure form and a severe majority-mass/template collapse by checkpoint 1152.**

Checkpoint 384 is strongly low-dimensional: PC1 explains 51.75% of centered variance, four components explain 90%, and entropy effective rank is 4.22. Checkpoint 768 has a much broader covariance spectrum, with effective rank 53.74 and 190 components for 90% variance, but its typical vectors remain nearly identical: median random-pair cosine is 0.999798 and median shuffled-vector distance is only 1.62. Its apparently healthy PCA is driven by a minority tail of distant vectors rather than uniformly diverse thoughts.

Checkpoint 1152 shows the clearest collapse. Only 3,910 of 49,251 stored thoughts (7.94%) are unique at FP16 precision; the ten most common templates cover 54.60% of the train set; the largest template alone covers 12.50%; and 93.19% of sampled unrelated pairs have cosine above 0.999. Median cosine to the global mean is 0.999852 and median shuffled-vector distance is 0.55. These statistics show that the typical thought belongs to an almost identical, highly repetitive core.

Checkpoint 1152's entropy effective rank of 40.59 does not contradict this verdict. Its centered-norm median is only 1.55 while the 99th percentile is 91.72. A small set of extreme vectors contributes variance in many directions and inflates covariance-based rank, while most examples remain collapsed. Effective rank must therefore be read together with template concentration, pairwise quantiles, and centered-distance quantiles.

Train raw target NLL improves monotonically from 1.1493 to 0.7119 to 0.3298, so recommendation likelihood improves while the representation develops a severely collapsed majority mode. The completed E1 ranking experiment separates activation from content: zero masking significantly reduces HR@10 and NDCG@10 at every checkpoint, while the global-mean and Gaussian-noise controls are indistinguishable from raw. Shuffled thoughts never worsen ranking; at checkpoint 768 they slightly improve HR@10 while leaving NDCG@10 statistically unchanged. **The decoder needs a nonzero latent activation but shows no evidence that the sample-specific thought identity improves ranking. Latent-utilization collapse is therefore strongly supported at all three checkpoints.**

## Scope and reproducibility

- Dataset: `CDs_and_Vinyl`
- Geometry and extraction likelihood split: train, 49,251 examples
- Causal ranking split: test, 6,157 examples
- Checkpoints: `output_dir/CDs_and_Vinyl_v5/latent/checkpoint-{384,768,1152}`
- Samples per checkpoint: 49,251
- Thought dimension: 1,536; one stored thought per interaction
- Stored dtype: FP16
- Train split SHA-256: `864f652064a08cd9d9b492e11cacceafba074cc8f4b1c3469b10a4bcdaba8da4`
- Test split SHA-256: `944634cd2b9248a4a579f222151c6df140935f56f5ea731cfe4ac21c240e1999`
- Pairwise diagnostics: 100,000 sampled pairs, seed 42
- PCA and group-variance diagnostics: centered and fitted on all 49,251 thoughts independently at each checkpoint
- Ranking: constrained 10-beam catalog generation; paired bootstrap 95% intervals over all 6,157 test examples, 2,000 resamples, seed 42

Checkpoint hashes:

| Step | Checkpoint SHA-256 |
|---:|---|
| 384 | `02e19477b718099baa145d82c7dc618446d5fb04af40ef22ccdedd136e8ddeba` |
| 768 | `eb55f4efa53d9b4def2ca256ace43a26e23749ccc7c811b4fd5b43c917aa52cd` |
| 1152 | `5ac44611ba1a1be8a3cf2809c5ceebd15d338311f450cc5398db6217b4fae701` |

Primary artifacts:

- [Epoch comparison summary](../results/latent_distribution/latent_v5_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_v5_epoch_collapse_trajectory.png)
- Per-checkpoint statistics: [384](../results/latent_distribution/latent_v5_checkpoint_384/train_latent_distribution.json), [768](../results/latent_distribution/latent_v5_checkpoint_768/train_latent_distribution.json), [1152](../results/latent_distribution/latent_v5_checkpoint_1152/train_latent_distribution.json)
- Distribution overviews: [384](../figures/latent_v5_checkpoint_384_latent_distribution_overview.png), [768](../figures/latent_v5_checkpoint_768_latent_distribution_overview.png), [1152](../figures/latent_v5_checkpoint_1152_latent_distribution_overview.png)
- All-vector PCA views: [384](../figures/latent_v5_checkpoint_384_latent_pca_all_scatter.png), [768](../figures/latent_v5_checkpoint_768_latent_pca_all_scatter.png), [1152](../figures/latent_v5_checkpoint_1152_latent_pca_all_scatter.png)
- Interactive t-SNE views: [384](../figures/latent_v5_checkpoint_384_tsne.html), [768](../figures/latent_v5_checkpoint_768_tsne.html), [1152](../figures/latent_v5_checkpoint_1152_tsne.html)
- E1 ranking artifacts: [384](../results/intervention/latent_v5_checkpoint_384/), [768](../results/intervention/latent_v5_checkpoint_768/), [1152](../results/intervention/latent_v5_checkpoint_1152/)

## 1. Epoch trajectory

| Diagnostic | Step 384 (epoch 1) | Step 768 (epoch 2) | Step 1152 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 1.149280 | 0.711866 | 0.329771 |
| Zero-mask train target NLL | 1.973567 | 1.356354 | 1.141819 |
| Zero-mask NLL penalty | +0.824287 | +0.644489 | +0.812048 |
| Mean sample norm | 94.01 | 69.39 | 79.99 |
| Global-mean vector norm | 91.82 | 67.32 | 78.71 |
| Global-mean norm / mean sample norm | 0.9767 | 0.9702 | 0.9840 |
| Mean cosine to global mean | 0.9736 | 0.9707 | 0.9874 |
| Median cosine to global mean | 0.9868 | 0.998548 | 0.999852 |
| Mean random-pair cosine | 0.9478 | 0.9418 | 0.9746 |
| Median random-pair cosine | 0.9730 | 0.999798 | 0.999981 |
| Random pairs with cosine > 0.999 | 18.42% | 57.59% | 93.19% |
| Exact FP16-unique vectors | 25,466 | 36,890 | 3,910 |
| Unique fraction | 51.71% | 74.90% | 7.94% |
| Samples in repeated templates | 58.29% | 31.42% | 93.89% |
| Top-10-template coverage | 7.57% | 2.91% | 54.60% |
| Largest-template coverage | 1.27% | 0.56% | 12.50% |
| Median shuffled-vector L2 | 23.02 | 1.62 | 0.55 |
| Mean shuffled-vector L2 | 26.07 | 14.34 | 5.47 |
| PC1 centered variance | 51.75% | 33.06% | 32.64% |
| Top-two centered variance | 78.36% | 41.28% | 39.62% |
| Entropy effective rank | 4.22 | 53.74 | 40.59 |
| Participation ratio | 2.86 | 8.23 | 8.28 |
| Components for 90% variance | 4 | 190 | 113 |

The mean-only trajectory understates the failure. Mean random-pair cosine changes modestly from 0.9478 to 0.9418 between checkpoints 384 and 768, but the median jumps from 0.9730 to 0.999798. The distribution becomes more heterogeneous: most vectors enter a near-identical core while a minority moves far enough away to lower the mean and broaden PCA.

Checkpoint 1152 amplifies the core and adds exact repetition. Relative to checkpoint 768, uniqueness falls by 66.96 percentage points, top-10 coverage rises by 51.68 points, median shuffled distance falls from 1.62 to 0.55, and the fraction of random pairs above 0.999 cosine rises from 57.59% to 93.19%. This is strong evidence of collapse during the third epoch despite continued train-NLL improvement.

## 2. Checkpoint-specific geometric diagnosis

### Checkpoint 384: low-dimensional collapse

Checkpoint 384 has the smallest effective rank and the most concentrated eigenspectrum. Four PCs explain 90% of centered variance, and the first two explain 78.36%. Exact repetition is already material: 58.29% of samples belong to a repeated FP16 template, though the mass is distributed across many templates rather than dominated by a handful.

Its pairwise distribution is less extreme than later checkpoints. Median shuffled distance is 23.02 and only 18.42% of random pairs exceed 0.999 cosine. The main failure is therefore a narrow shared subspace with moderate template reuse, not yet the single dominant near-identical core seen later.

### Checkpoint 768: broad outlier covariance around a collapsed core

Checkpoint 768 looks healthier by exact uniqueness and PCA: 74.90% of vectors are unique, effective rank is 53.74, and 190 PCs are needed for 90% variance. Those aggregate statistics conceal a sharply concentrated typical case. Median random-pair cosine is 0.999798, median cosine to the global mean is 0.998548, and median shuffled distance is only 1.62 in a space whose mean vector norm is 67.32.

The centered-norm distribution exposes the mixture: the median is 4.84, but the 95th percentile is 46.66 and the maximum is 92.10. A minority tail supplies much of the high-dimensional covariance. The checkpoint has escaped simple exact-template collapse but not majority-mass or shared-direction collapse.

### Checkpoint 1152: severe template and majority-mass collapse

Checkpoint 1152 maps 49,251 interactions to only 3,910 exact FP16 vectors. The largest template contains 6,158 samples, the five largest cover 38.57%, and the ten largest cover 54.60%. More than 93% of unrelated sampled pairs exceed 0.999 cosine, and half of shuffled pairs are separated by only 0.55 in L2 despite a mean sample norm of 79.99.

The representation is not uniformly constant. The centered-norm median is 1.55, but the 99th percentile is 91.72 and the maximum is 120.99. These rare outliers explain why full-bank PCA reports effective rank 40.59 and 113 components for 90% variance. For the typical example, however, the thought is effectively drawn from a small set of nearly collinear templates. Template counts and robust quantiles are the appropriate primary evidence at this checkpoint.

## 3. Preference-linked structure

| Diagnostic | Step 384 | Step 768 | Step 1152 |
|---|---:|---:|---:|
| Same-user cosine (absolute) | 0.9569 | 0.9473 | 0.9794 |
| Same-target cosine (absolute) | 0.9495 | 0.9434 | 0.9770 |
| Same CF-cluster cosine (absolute) | 0.9485 | 0.9428 | 0.9756 |
| Adjusted variance associated with user | 11.67% | 3.00% | 9.63% |
| Adjusted variance associated with target | 2.87% | 1.83% | 6.86% |
| Adjusted variance associated with CF cluster | 0.79% | 0.41% | 1.20% |

The table reports absolute within-group cosine, as requested. These values are high for every grouping, but the random-pair baselines are also 0.9478, 0.9418, and 0.9746. Absolute similarity therefore mostly reflects the shared direction and must not be read as preference separation. Same-user pairs are consistently the most similar of the three groupings; target and train-only CF-neighborhood similarity are closer to the random baseline.

Adjusted group-centroid variance is larger than the pairwise gaps suggest, especially for user at checkpoints 384 and 1152 and target at checkpoint 1152. In a core-plus-outlier distribution, these variance fractions can be driven disproportionately by the minority tail. They show some structured residual variation, not that the dominant collapsed core contains rich preference reasoning.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 384

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.060094 | 0 | 0.042316 | 0 |
| Zero / mask | 0.047913 | -0.012181 [-0.016079, -0.008283] | 0.029594 | -0.012723 [-0.015461, -0.009900] |
| Global train mean | 0.059445 | -0.000650 [-0.001787, +0.000329] | 0.042095 | -0.000221 [-0.000686, +0.000251] |
| Shuffled user | 0.059932 | -0.000162 [-0.001299, +0.000812] | 0.042262 | -0.000055 [-0.000512, +0.000371] |
| Shuffled target | 0.059932 | -0.000162 [-0.001141, +0.000812] | 0.042262 | -0.000055 [-0.000509, +0.000375] |
| Gaussian noise 0.25x | 0.060257 | +0.000162 [-0.000650, +0.000975] | 0.042293 | -0.000023 [-0.000447, +0.000387] |
| Gaussian noise 0.50x | 0.059607 | -0.000487 [-0.001624, +0.000650] | 0.042235 | -0.000082 [-0.000547, +0.000372] |
| Gaussian noise 1.00x | 0.059607 | -0.000487 [-0.001787, +0.000812] | 0.042229 | -0.000087 [-0.000680, +0.000444] |

### Checkpoint 768

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.098912 | 0 | 0.072047 | 0 |
| Zero / mask | 0.070814 | -0.028098 [-0.032812, -0.023226] | 0.046795 | -0.025252 [-0.028797, -0.021667] |
| Global train mean | 0.098749 | -0.000162 [-0.001462, +0.001137] | 0.072025 | -0.000022 [-0.000627, +0.000593] |
| Shuffled user | 0.100211 | +0.001299 [+0.000325, +0.002599] | 0.072540 | +0.000494 [-0.000025, +0.001031] |
| Shuffled target | 0.100211 | +0.001299 [+0.000162, +0.002599] | 0.072540 | +0.000494 [-0.000033, +0.001044] |
| Gaussian noise 0.25x | 0.098912 | +0.000000 [-0.001137, +0.001137] | 0.072046 | -0.000001 [-0.000441, +0.000487] |
| Gaussian noise 0.50x | 0.099074 | +0.000162 [-0.000975, +0.001299] | 0.072229 | +0.000182 [-0.000510, +0.000821] |
| Gaussian noise 1.00x | 0.099724 | +0.000812 [-0.000162, +0.001787] | 0.071885 | -0.000162 [-0.000793, +0.000404] |

### Checkpoint 1152

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.112717 | 0 | 0.080572 | 0 |
| Zero / mask | 0.079584 | -0.033133 [-0.038980, -0.027607] | 0.052833 | -0.027739 [-0.031821, -0.023708] |
| Global train mean | 0.112392 | -0.000325 [-0.001787, +0.001299] | 0.080402 | -0.000170 [-0.000838, +0.000503] |
| Shuffled user | 0.112555 | -0.000162 [-0.001624, +0.001137] | 0.080265 | -0.000307 [-0.000917, +0.000280] |
| Shuffled target | 0.112555 | -0.000162 [-0.001462, +0.001137] | 0.080265 | -0.000307 [-0.000888, +0.000289] |
| Gaussian noise 0.25x | 0.112717 | +0.000000 [-0.001462, +0.001462] | 0.080521 | -0.000051 [-0.000703, +0.000589] |
| Gaussian noise 0.50x | 0.111580 | -0.001137 [-0.002924, +0.000491] | 0.080062 | -0.000510 [-0.001242, +0.000162] |
| Gaussian noise 1.00x | 0.111580 | -0.001137 [-0.002765, +0.000487] | 0.080052 | -0.000519 [-0.001375, +0.000264] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these complete tables provide point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 384

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.027936 | 0.041091 | 0.047263 | 0.060094 | 0.027936 | 0.035662 | 0.038199 | 0.042316 |
| Zero / mask | 0.014780 | 0.028098 | 0.036544 | 0.047913 | 0.014780 | 0.022417 | 0.025905 | 0.029594 |
| Global train mean | 0.028098 | 0.040767 | 0.046938 | 0.059445 | 0.028098 | 0.035581 | 0.038104 | 0.042095 |
| Shuffled user | 0.028098 | 0.041254 | 0.047101 | 0.059932 | 0.028098 | 0.035782 | 0.038172 | 0.042262 |
| Shuffled target | 0.028098 | 0.041254 | 0.047101 | 0.059932 | 0.028098 | 0.035782 | 0.038172 | 0.042262 |
| Gaussian noise 0.25x | 0.027936 | 0.040929 | 0.047263 | 0.060257 | 0.027936 | 0.035538 | 0.038138 | 0.042293 |
| Gaussian noise 0.50x | 0.028423 | 0.040767 | 0.047263 | 0.059607 | 0.028423 | 0.035637 | 0.038307 | 0.042235 |
| Gaussian noise 1.00x | 0.028748 | 0.040279 | 0.046614 | 0.059607 | 0.028748 | 0.035470 | 0.038078 | 0.042229 |

#### Checkpoint 768

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.050024 | 0.070164 | 0.081858 | 0.098912 | 0.050024 | 0.061774 | 0.066554 | 0.072047 |
| Zero / mask | 0.027611 | 0.043853 | 0.055222 | 0.070814 | 0.027611 | 0.037050 | 0.041719 | 0.046795 |
| Global train mean | 0.050024 | 0.070164 | 0.081858 | 0.098749 | 0.050024 | 0.061795 | 0.066568 | 0.072025 |
| Shuffled user | 0.050024 | 0.070489 | 0.082183 | 0.100211 | 0.050024 | 0.061958 | 0.066717 | 0.072540 |
| Shuffled target | 0.050024 | 0.070489 | 0.082183 | 0.100211 | 0.050024 | 0.061958 | 0.066717 | 0.072540 |
| Gaussian noise 0.25x | 0.050024 | 0.070326 | 0.081371 | 0.098912 | 0.050024 | 0.061855 | 0.066370 | 0.072046 |
| Gaussian noise 0.50x | 0.050187 | 0.070976 | 0.081046 | 0.099074 | 0.050187 | 0.062283 | 0.066385 | 0.072229 |
| Gaussian noise 1.00x | 0.049050 | 0.070326 | 0.081208 | 0.099724 | 0.049050 | 0.061453 | 0.065891 | 0.071885 |

#### Checkpoint 1152

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.055547 | 0.076661 | 0.089979 | 0.112717 | 0.055547 | 0.067826 | 0.073313 | 0.080572 |
| Zero / mask | 0.031996 | 0.049700 | 0.061556 | 0.079584 | 0.031996 | 0.042230 | 0.047059 | 0.052833 |
| Global train mean | 0.054897 | 0.076661 | 0.090141 | 0.112392 | 0.054897 | 0.067693 | 0.073235 | 0.080402 |
| Shuffled user | 0.054897 | 0.076498 | 0.089167 | 0.112555 | 0.054897 | 0.067548 | 0.072740 | 0.080265 |
| Shuffled target | 0.054897 | 0.076498 | 0.089167 | 0.112555 | 0.054897 | 0.067548 | 0.072740 | 0.080265 |
| Gaussian noise 0.25x | 0.055384 | 0.076661 | 0.089492 | 0.112717 | 0.055384 | 0.067809 | 0.073086 | 0.080521 |
| Gaussian noise 0.50x | 0.054897 | 0.076986 | 0.089492 | 0.111580 | 0.054897 | 0.067813 | 0.072957 | 0.080062 |
| Gaussian noise 1.00x | 0.055059 | 0.076986 | 0.089979 | 0.111580 | 0.055059 | 0.067830 | 0.073184 | 0.080052 |

All 24 generation runs returned 6,157 paired records and zero catalog violations. Raw ranking improves strongly with training: HR@10 rises from 0.0601 to 0.0989 to 0.1127, and NDCG@10 rises from 0.0423 to 0.0720 to 0.0806.

Zero masking is significantly destructive at every checkpoint, with HR@10 changes of -0.0122, -0.0281, and -0.0331 and NDCG@10 changes of -0.0127, -0.0253, and -0.0277. In contrast, the global-mean and all Gaussian-noise confidence intervals contain zero. Shuffling also never harms ranking. At checkpoint 768, shuffled-user and shuffled-target each improve HR@10 by 0.001299 (95% intervals exclude zero), but their NDCG@10 intervals include zero. This small beneficial HR effect strengthens, rather than weakens, the conclusion that the raw sample-specific identity is not needed for ranking.

Shuffled-user and shuffled-target yield identical ranks at every test example for all checkpoints. Their decoded top-10 lists are identical everywhere except one example at checkpoint 768, and even there the target rank is unchanged. They are therefore one effective ranking control, not independent replications.

### Activation-versus-content interpretation

Zero masking raises train target NLL by 0.8243, 0.6445, and 0.8120 at checkpoints 384, 768, and 1152, and significantly damages held-out ranking. The latent slot therefore supplies a useful nonzero activation. However, a global mean, an unrelated train thought, or Gaussian noise as large as 1.0x leaves held-out ranking statistically unchanged, apart from the small positive shuffle HR@10 effect at checkpoint 768. This is activation dependence without beneficial sample-specific content dependence—the causal definition of latent-utilization collapse used in the reference report.

## 5. Diagnosis

1. **Shared-direction collapse: strongly supported at every checkpoint.** The global-mean vector retains 97.0%–98.4% of the mean sample norm, and typical cosine alignment becomes extreme after checkpoint 384.
2. **Centered dimensional collapse: strongly supported at checkpoint 384; heterogeneous at checkpoints 768 and 1152.** Later effective ranks are raised by minority tails and do not describe the typical example.
3. **Exact-template collapse: moderate at checkpoint 384, weakest at checkpoint 768, and severe at checkpoint 1152.**
4. **Causal utilization collapse: strongly supported at all three checkpoints.** Zeroing significantly damages ranking, while mean replacement and even large Gaussian perturbations do not; shuffled thoughts do not reduce ranking at any checkpoint.

The combined evidence distinguishes two failures. Geometry shows a narrow or majority-collapsed representation, especially at checkpoint 1152. E1 shows that the decoder uses the latent slot as a generic carrier but does not benefit from which sample-specific thought occupies it. Better raw accuracy and the outlier-inflated effective rank therefore do not imply usable latent reasoning.

## 6. Limitations and next action

- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted independently at each checkpoint; scalar summaries are comparable, but coordinates are not pointwise aligned across epochs.
- Exact-template counts use FP16 storage precision. Float32 cosine and distance diagnostics independently confirm the dominant-core phenomenon.
- Pairwise diagnostics use 100,000 reproducible samples; PCA and group-centroid variance use all 49,251 vectors.
- Full-bank effective rank is variance-weighted and sensitive to rare high-distance observations. It should not be interpreted as the number of dimensions used by a typical sample in these skewed distributions.
- Direct row-level t-SNE can visually spread identical inputs because of its repulsive objective. The interactive views are exploratory; exact counts, robust pairwise quantiles, distances, and PCA statistics are the primary collapse evidence.
- Zero masking is off-manifold, but mean and shuffled replacements provide in-distribution controls. Gaussian noise provides continuous perturbation controls.
- E1 evaluates constrained top-10 ranking, not held-out teacher-forced NLL. It can miss probability changes that do not alter the decoded order.

`latent_v5` improves raw train likelihood and held-out ranking across epochs but does not learn a uniformly diverse or causally used thought space. Checkpoint 384 is severely low-rank; checkpoint 768 creates a near-identical majority core plus a diverse tail; and checkpoint 1152 collapses most examples into a small, almost collinear template vocabulary. At all three checkpoints, E1 finds activation dependence without evidence that sample-specific thought identity improves ranking.

The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), separately from content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This will test whether the underlying token probabilities are as insensitive to thought identity as the constrained ranking is.
