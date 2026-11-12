# Latent-collapse assessment for `latent_v4` checkpoints 384, 768, and 1152

## Executive verdict

**All three `latent_v4` checkpoints retain a strong shared direction and low-dimensional centered geometry, but none exhibits the severe exact-template collapse seen in the earlier SFT/RL endpoints. The geometric trajectory improves overall with training rather than collapsing progressively.**

Checkpoint 384 is the most dimensionally collapsed: PC1 explains 63.81% of centered variance, five components explain 90%, and the entropy effective rank is 4.23. Checkpoint 768 remains concentrated but broadens to effective rank 11.34 and 32 components for 90% variance. Checkpoint 1152 is the least dimensionally collapsed, with effective rank 21.32, PC1 at 37.96%, and 72 components required for 90% variance.

At the same time, all checkpoints are strongly anisotropic. Mean random-pair cosine stays between 0.883 and 0.908, and the global-mean norm is 93.9%–95.7% of the mean sample norm. Thus, the final checkpoint is healthier than the earlier checkpoints in centered diversity, but it is not an isotropic or unconcentrated latent space.

Exact FP16 uniqueness is high at every checkpoint (92.92%–98.06%), and the top ten templates cover at most 2.10% of samples. This rules out a small repeated-template vocabulary as the main failure mechanism. As in the reference checkpoint-768 report, exact uniqueness and geometric diversity must be separated: checkpoint 384 is 94.58% unique but has an effective rank of only 4.23.

The completed E1 test-ranking intervention establishes the same functional failure at all three checkpoints. Zero masking significantly reduces HR@10 and NDCG@10, but replacing the thought with the global train mean or an unrelated train thought has no significant effect. Gaussian perturbations are also indistinguishable from raw, even at 1.0x scale, whose expected L2 displacement is more than twice the natural shuffled-thought distance. The decoder therefore needs a nonzero latent activation but makes almost no measurable ranking use of its sample-specific content. **Latent-utilization collapse is strongly supported at checkpoints 384, 768, and 1152 despite the progressive improvement in centered geometric diversity.**

**Addendum (§7):** checkpoint 1152 was further fine-tuned with attention-only GRPO (`nomul_attn_lr_5e-5`). That run does not fix latent-utilization collapse — it produces a worse failure. Raw HR@10/NDCG@10 fall roughly 7×/6× versus checkpoint 1152, geometric effective rank rises almost 17× (a variance-injection effect from GRPO's own exploration noise, not learned diversity), and zero-masking or noising the RL-tuned latent now *significantly improves* ranking and train likelihood relative to using it as trained. See §7 for the full evidence and a likely mechanism (group-relative GRPO advantage with no absolute anchor to the SFT baseline).

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

- [Epoch comparison summary](../results/latent_distribution/latent_v4_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_v4_epoch_collapse_trajectory.png)
- Checkpoint statistics: [384](../results/latent_distribution/latent_v4_checkpoint_384/train_latent_distribution.json), [768](../results/latent_distribution/latent_v4_checkpoint_768/train_latent_distribution.json), [1152](../results/latent_distribution/latent_v4_checkpoint_1152/train_latent_distribution.json)
- Distribution overviews: [384](../figures/latent_v4_checkpoint_384_latent_distribution_overview.png), [768](../figures/latent_v4_checkpoint_768_latent_distribution_overview.png), [1152](../figures/latent_v4_checkpoint_1152_latent_distribution_overview.png)
- All-vector PCA views: [384](../figures/latent_v4_checkpoint_384_latent_pca_all_scatter.png), [768](../figures/latent_v4_checkpoint_768_latent_pca_all_scatter.png), [1152](../figures/latent_v4_checkpoint_1152_latent_pca_all_scatter.png)
- E1 ranking artifacts: [384](../results/intervention/latent_v4_checkpoint_384/), [768](../results/intervention/latent_v4_checkpoint_768/), [1152](../results/intervention/latent_v4_checkpoint_1152/)

## 1. Epoch trajectory

| Diagnostic | Step 384 (epoch 1) | Step 768 (epoch 2) | Step 1152 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 1.141430 | 0.715405 | 0.341776 |
| Zero-mask train target NLL | 1.814777 | 1.169955 | 1.487041 |
| Zero-mask NLL penalty | +0.673347 | +0.454550 | +1.145265 |
| Mean sample norm | 89.39 | 80.03 | 87.58 |
| Global-mean vector norm | 83.92 | 76.60 | 82.48 |
| Global-mean norm / mean sample norm | 0.9389 | 0.9571 | 0.9417 |
| Mean cosine to global mean | 0.9396 | 0.9531 | 0.9413 |
| Mean random-pair cosine | 0.8829 | 0.9081 | 0.8854 |
| Exact FP16-unique vectors | 46,582 | 48,295 | 45,763 |
| Unique fraction | 94.58% | 98.06% | 92.92% |
| Top-10-template coverage | 2.10% | 0.42% | 0.77% |
| Largest-template coverage | 0.56% | 0.27% | 0.27% |
| PC1 centered variance | 63.81% | 52.20% | 37.96% |
| Top-two centered variance | 77.59% | 67.42% | 55.52% |
| Entropy effective rank | 4.23 | 11.34 | 21.32 |
| Participation ratio | 2.30 | 3.34 | 5.50 |
| Components for 90% variance | 5 | 32 | 72 |

Training substantially improves raw train NLL, while centered diversity also increases. From checkpoint 384 to 1152, effective rank grows about fivefold, the PC1 share falls by 25.85 percentage points, and the number of components needed for 90% variance rises from 5 to 72. This directly contradicts a hypothesis of monotonic geometric collapse across these three checkpoints.

The shared offset does not improve monotonically. Random-pair cosine and the mean-norm ratio peak at checkpoint 768, making it the most directionally aligned checkpoint, even though its centered effective rank is higher than checkpoint 384. Checkpoint 1152 then becomes both less aligned and more diverse in centered coordinates.

## 2. Template concentration versus dimensional concentration

The checkpoints do not collapse into a small set of exact stored templates. Even the least unique checkpoint, 1152, contains 45,763 distinct FP16 vectors among 49,251 samples. The largest exact template contains 275 samples at checkpoint 384 and 133 samples at checkpoints 768 and 1152.

Dimensional concentration is nevertheless severe, especially early. At checkpoint 384, the first ten centered PCs explain 97.20% of variance and the first 32 explain 99.20%. High exact uniqueness therefore reflects many distinct points lying in a very narrow subspace, not broadly distributed reasoning states. By checkpoint 1152, the corresponding shares fall to 75.49% and 85.05%, indicating a material but incomplete recovery.

## 3. User-, target-, and preference-linked structure

| Diagnostic | Step 384 | Step 768 | Step 1152 |
|---|---:|---:|---:|
| Same-user cosine | 0.9134 | 0.9318 | 0.9105 |
| Same-user minus random | +0.0305 | +0.0236 | +0.0251 |
| Same-target cosine | 0.8895 | 0.9145 | 0.8924 |
| Same-target minus random | +0.0066 | +0.0063 | +0.0069 |
| Same CF-cluster cosine | 0.8855 | 0.9105 | 0.8885 |
| Same CF-cluster minus random | +0.0026 | +0.0024 | +0.0030 |
| Adjusted variance associated with user | 22.02% | 22.36% | 17.34% |
| Adjusted variance associated with target | 4.29% | 6.16% | 5.69% |
| Adjusted variance associated with CF cluster | 1.60% | 2.05% | 1.75% |

The residual geometry is primarily user-linked. Same-user pairs consistently exceed random-pair similarity by roughly 0.024–0.031, and adjusted user-associated variance is 17%–22%. Target structure is smaller, while conventional collaborative-filtering cluster structure is weak. The train thoughts are therefore not arbitrary noise, but their strongest measured organization is user identity rather than target or CF-neighborhood structure.

The exact-template utility eta-squared values are not reliable evidence of utility alignment because almost every vector is unique. With nearly one template group per sample, that statistic is structurally overfit.

## 4. Causal intervention ranking evidence

Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are paired bootstrap 95% intervals over the per-example metric contribution.

### Checkpoint 384

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.054734 | 0 | 0.040595 | 0 |
| Zero / mask | 0.032159 | -0.022576 [-0.026474, -0.018678] | 0.022007 | -0.018588 [-0.021629, -0.015575] |
| Global train mean | 0.054734 | +0.000000 [-0.001137, +0.001137] | 0.040605 | +0.000010 [-0.000416, +0.000462] |
| Shuffled user | 0.054410 | -0.000325 [-0.001462, +0.000812] | 0.040505 | -0.000090 [-0.000586, +0.000382] |
| Shuffled target | 0.054410 | -0.000325 [-0.001462, +0.000812] | 0.040505 | -0.000090 [-0.000586, +0.000382] |
| Gaussian noise 0.25x | 0.054410 | -0.000325 [-0.001299, +0.000650] | 0.040464 | -0.000131 [-0.000456, +0.000185] |
| Gaussian noise 0.50x | 0.054247 | -0.000487 [-0.001787, +0.000650] | 0.040298 | -0.000297 [-0.000756, +0.000129] |
| Gaussian noise 1.00x | 0.055384 | +0.000650 [-0.000650, +0.002111] | 0.040655 | +0.000060 [-0.000495, +0.000590] |

### Checkpoint 768

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.099724 | 0 | 0.070525 | 0 |
| Zero / mask | 0.076986 | -0.022738 [-0.027773, -0.017541] | 0.049660 | -0.020864 [-0.024442, -0.017460] |
| Global train mean | 0.100211 | +0.000487 [-0.000812, +0.001787] | 0.070502 | -0.000022 [-0.000594, +0.000572] |
| Shuffled user | 0.099886 | +0.000162 [-0.000975, +0.001299] | 0.070488 | -0.000037 [-0.000545, +0.000514] |
| Shuffled target | 0.099886 | +0.000162 [-0.000975, +0.001299] | 0.070488 | -0.000037 [-0.000545, +0.000514] |
| Gaussian noise 0.25x | 0.100698 | +0.000975 [-0.000162, +0.002436] | 0.070578 | +0.000053 [-0.000534, +0.000651] |
| Gaussian noise 0.50x | 0.100536 | +0.000812 [-0.000325, +0.002111] | 0.070541 | +0.000016 [-0.000524, +0.000596] |
| Gaussian noise 1.00x | 0.099886 | +0.000162 [-0.001137, +0.001462] | 0.070467 | -0.000058 [-0.000578, +0.000463] |

### Checkpoint 1152

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.115153 | 0 | 0.080583 | 0 |
| Zero / mask | 0.077148 | -0.038006 [-0.044177, -0.031992] | 0.048349 | -0.032234 [-0.036317, -0.028134] |
| Global train mean | 0.115153 | +0.000000 [-0.001624, +0.001624] | 0.080926 | +0.000343 [-0.000324, +0.001042] |
| Shuffled user | 0.114179 | -0.000975 [-0.002436, +0.000325] | 0.080477 | -0.000106 [-0.000741, +0.000534] |
| Shuffled target | 0.114179 | -0.000975 [-0.002436, +0.000325] | 0.080477 | -0.000106 [-0.000741, +0.000534] |
| Gaussian noise 0.25x | 0.114666 | -0.000487 [-0.001787, +0.000812] | 0.080699 | +0.000115 [-0.000403, +0.000647] |
| Gaussian noise 0.50x | 0.114666 | -0.000487 [-0.001949, +0.001137] | 0.080574 | -0.000009 [-0.000684, +0.000647] |
| Gaussian noise 1.00x | 0.115966 | +0.000812 [-0.000812, +0.002436] | 0.081080 | +0.000497 [-0.000254, +0.001255] |

### Complete metrics at all cutoffs

The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential comparison against raw; these complete tables provide the corresponding point estimates at cutoffs 1, 3, 5, and 10.

#### Checkpoint 384

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.029722 | 0.038980 | 0.043365 | 0.054734 | 0.029722 | 0.035117 | 0.036906 | 0.040595 |
| Zero / mask | 0.014780 | 0.019328 | 0.024038 | 0.032159 | 0.014780 | 0.017479 | 0.019401 | 0.022007 |
| Global train mean | 0.030047 | 0.038818 | 0.043365 | 0.054734 | 0.030047 | 0.035070 | 0.036936 | 0.040605 |
| Shuffled user | 0.029722 | 0.039630 | 0.043365 | 0.054410 | 0.029722 | 0.035378 | 0.036915 | 0.040505 |
| Shuffled target | 0.029722 | 0.039630 | 0.043365 | 0.054410 | 0.029722 | 0.035378 | 0.036915 | 0.040505 |
| Gaussian noise 0.25x | 0.029885 | 0.039305 | 0.043528 | 0.054410 | 0.029885 | 0.035211 | 0.036938 | 0.040464 |
| Gaussian noise 0.50x | 0.029722 | 0.038655 | 0.043040 | 0.054247 | 0.029722 | 0.034827 | 0.036644 | 0.040298 |
| Gaussian noise 1.00x | 0.029560 | 0.038655 | 0.043528 | 0.055384 | 0.029560 | 0.034831 | 0.036822 | 0.040655 |

#### Checkpoint 768

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.045802 | 0.069352 | 0.081858 | 0.099724 | 0.045802 | 0.059597 | 0.064705 | 0.070525 |
| Zero / mask | 0.027611 | 0.047263 | 0.057820 | 0.076986 | 0.027611 | 0.039053 | 0.043386 | 0.049660 |
| Global train mean | 0.045639 | 0.069839 | 0.082833 | 0.100211 | 0.045639 | 0.059610 | 0.064879 | 0.070502 |
| Shuffled user | 0.045477 | 0.070002 | 0.082020 | 0.099886 | 0.045477 | 0.059802 | 0.064693 | 0.070488 |
| Shuffled target | 0.045477 | 0.070002 | 0.082020 | 0.099886 | 0.045477 | 0.059802 | 0.064693 | 0.070488 |
| Gaussian noise 0.25x | 0.045314 | 0.069190 | 0.082020 | 0.100698 | 0.045314 | 0.059293 | 0.064520 | 0.070578 |
| Gaussian noise 0.50x | 0.045314 | 0.069514 | 0.082995 | 0.100536 | 0.045314 | 0.059413 | 0.064877 | 0.070541 |
| Gaussian noise 1.00x | 0.045639 | 0.069027 | 0.082020 | 0.099886 | 0.045639 | 0.059353 | 0.064650 | 0.070467 |

#### Checkpoint 1152

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.052948 | 0.078447 | 0.092415 | 0.115153 | 0.052948 | 0.067590 | 0.073321 | 0.080583 |
| Zero / mask | 0.026636 | 0.044340 | 0.056846 | 0.077148 | 0.026636 | 0.036700 | 0.041809 | 0.048349 |
| Global train mean | 0.053435 | 0.079259 | 0.092253 | 0.115153 | 0.053435 | 0.068282 | 0.073594 | 0.080926 |
| Shuffled user | 0.053110 | 0.079422 | 0.092578 | 0.114179 | 0.053110 | 0.068201 | 0.073554 | 0.080477 |
| Shuffled target | 0.053110 | 0.079422 | 0.092578 | 0.114179 | 0.053110 | 0.068201 | 0.073554 | 0.080477 |
| Gaussian noise 0.25x | 0.053110 | 0.078935 | 0.092902 | 0.114666 | 0.053110 | 0.068021 | 0.073738 | 0.080699 |
| Gaussian noise 0.50x | 0.052785 | 0.079259 | 0.092415 | 0.114666 | 0.052785 | 0.068064 | 0.073431 | 0.080574 |
| Gaussian noise 1.00x | 0.052948 | 0.078610 | 0.092415 | 0.115966 | 0.052948 | 0.067905 | 0.073559 | 0.081080 |

All 24 generation runs returned 6,157 records and zero catalog violations. Raw ranking improves strongly with training: HR@10 rises from 0.0547 to 0.0997 to 0.1152, and NDCG@10 rises from 0.0406 to 0.0705 to 0.0806.

Zero masking is significantly destructive at every checkpoint, and its ranking penalty is largest at checkpoint 1152. In contrast, every mean, shuffled, and noise confidence interval contains zero. Their absolute point changes are at most 0.000975 HR@10 and 0.000497 NDCG@10, while zero-mask changes reach -0.038006 and -0.032234. Sample-specific thought identity therefore remains causally irrelevant to measured ranking even as raw accuracy and geometric effective rank improve.

The perturbation null result is not explained by insufficient displacement. Expected 1.0x Gaussian L2 distances are 89.69, 80.56, and 87.87 for checkpoints 384, 768, and 1152, compared with natural shuffled-vector mean distances of 39.48, 29.70, and 35.39. Perturbations more than twice as large as a natural shuffle still leave ranking statistically unchanged.

The shuffled-user and shuffled-target variants have identical ranks for every test example at each checkpoint. Their decoded top-10 lists differ for only one of 6,157 examples per checkpoint, so they should be treated as one effective ranking control rather than independent evidence.

### Activation-versus-content interpretation

The train extraction and held-out ranking agree. Zero masking raises train NLL by 0.6733, 0.4546, and 1.1453 at checkpoints 384, 768, and 1152, while test ranking also falls substantially. The thought slot therefore supplies a useful nonzero activation. However, a global mean, an unrelated valid thought, or large continuous noise preserves ranking. This is activation dependence without sample-specific content dependence—the causal definition of latent-utilization collapse used in the reference report.

The zero-mask penalty is non-monotonic from checkpoints 384 to 768 and then peaks at 1152. Raw train NLL and raw test ranking improve with training, while centered geometry becomes more diverse. These axes can therefore improve while the decoder continues to ignore which valid thought it receives.

## 5. Diagnosis

1. **Severe exact-template collapse: not supported.** More than 92.9% of stored vectors are unique at all three checkpoints, and top-10 coverage stays below 2.11%.
2. **Shared-direction collapse: supported at all checkpoints.** Random thoughts have mean cosine around 0.89–0.91, and the global mean retains about 94%–96% of a typical vector's norm.
3. **Centered dimensional collapse: strongest at checkpoint 384 and progressively weaker thereafter.** Effective rank rises from 4.23 to 11.34 to 21.32, with matching improvements across PC concentration and components-for-90%-variance.
4. **Functional/utilization collapse: strongly supported at all three checkpoints.** Zeroing significantly damages held-out ranking, while mean, shuffled, and even large-noise interventions do not.

The most defensible overall conclusion is that `latent_v4` avoids the discrete-template failure but retains a strong common carrier and a low-dimensional residual space. Training through checkpoint 1152 substantially broadens that residual space and improves recommendation accuracy, yet the decoder does not use the recovered sample-specific variation. Geometric improvement alone has therefore not fixed causal latent reasoning.

## 6. Limitations and next action

- Geometry and NLL utilities are measured on train thoughts; ranking interventions are measured on the held-out test split.
- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.
- Exact duplicate counts are at FP16 storage precision, though cosine and PCA diagnostics operate in float32 after loading and independently show concentration.
- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the complete train bank.
- Zero masking is off-manifold, but the mean and shuffled interventions provide in-distribution controls. Gaussian noise tests local and larger continuous perturbations.
- The deterministic shuffled-user and shuffled-target selectors produce effectively identical ranking controls here and should not be counted as independent replications.
- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking generation can miss probability changes that do not alter the constrained top-10 order.

The next action is to run the paired validation-NLL intervention at all three checkpoints. Track activation dependence, \(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, \(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). E1 ranking already shows that content changes do not alter constrained ordering; validation NLL will test whether the underlying token probabilities are equally insensitive.

---

## 7. Post-GRPO addendum: `nomul_attn_lr_5e-5`

**Following checkpoint 1152, the attention module was further fine-tuned with attention-only GRPO (`train_noise_grpo.py`, `lr=5e-5`, `beta=0.01`, 1 epoch, `num_generations=8`, Gaussian exploration scale 1.5, base LLM and `lm_head` frozen). The result is not a fix for latent-utilization collapse — it is a new, more severe failure: the RL-tuned latent's raw activation is now actively harmful to the frozen decoder, and zeroing or noising it significantly improves ranking.**

### 7.1 Scope

- Checkpoint: `output_dir/CDs_and_Vinyl_v4/nomul_attn_lr_5e-5` (sha256 `67ce359f…`), warm-started from `latent_v4/latent/checkpoint-1152`.
- Same protocol as checkpoints 384/768/1152: geometry on the 49,251-example train split, E1 causal intervention (raw, mask, mean, shuffle_user, shuffle_target, noise 0.25×/0.5×/1.0×) on the 6,157-example test split, identical split hashes (train `864f652…`, test `944634c…`).
- Artifacts: [geometry](../results/latent_distribution/latent_v4_grpo_lr5e-5/train_latent_distribution.json), [train extraction summary](../evidence/CDs_and_Vinyl/latent_v4_grpo_lr5e-5/train/summary.json), [E1 ranking results](../results/intervention/latent_v4_grpo_lr5e-5/), [distribution overview figure](../figures/latent_v4_grpo_lr5e-5_latent_distribution_overview.png), [PCA scatter](../figures/latent_v4_grpo_lr5e-5_latent_pca_all_scatter.png).

**Note on `output_dir/CDs_and_Vinyl_v4/nomul_attn_lr_5e-5/metric.log`:** that file (written by the repo's own `merge.py`/`calc.py` pipeline, not by this report) contains three appended blocks and only the *last* is trustworthy for this checkpoint. `merge.py` reads shard predictions from `./output_dir/CDs_and_Vinyl_v4/{0,1}.json` — a path shared, byte-for-byte, between `launch_run.sh`'s SFT-eval block and its RL-eval block. The first two blocks (NDCG@10≈0.070, HR@10≈0.100) are the SFT-only `latent` checkpoint's numbers (confirmed by the `./output_dir/CDs_and_Vinyl_v4/latent/final_result.json` path Slurm job 246220 printed immediately before them, `run_logs/R3.246220.out:9488-9491`); they leaked into this file because the earlier crashed job (253170, see the debugging session above) had `noise_eval.py` fail with `OSError` *before* writing fresh shard predictions, so `merge.py` silently re-merged the stale SFT shard files under the GRPO checkpoint's `final_result.json` path — the same missing-error-handling pattern documented earlier in this conversation, this time at the eval stage. The third block (NDCG@10=0.012939, HR@10=0.016567) is from job 253424, the run that actually produced today's checkpoint, and matches this section's raw row (0.012753/0.016242, small differences from beam-search batching) and an independent direct recomputation from `final_result.json`. There is no evidence of an earlier *working* GRPO checkpoint that was overwritten — only mislabeled SFT numbers.

### 7.2 Geometry moved a lot — in the opposite direction from what "fixing collapse" would look like

| Diagnostic | Checkpoint 1152 (pre-RL) | `nomul_attn_lr_5e-5` (post-RL) |
|---|---:|---:|
| Entropy effective rank | 21.32 | **360.91** |
| Participation ratio | 5.50 | 104.46 |
| PC1 centered variance | 37.96% | 7.16% |
| Components for 90% variance | 72 | 482 |
| Mean random-pair cosine | 0.8854 | 0.7411 |
| Global-mean norm / mean sample norm | 0.9417 | 0.8623 |
| Exact FP16-unique fraction | 92.92% | 78.55% |
| Top-10-template coverage | 0.77% | 1.10% |

GRPO's Gaussian exploration (scale 1.5 on every non-baseline sample) massively expands the centered geometry — effective rank rises almost 17×, and the shared-direction anisotropy that persisted through all three SFT checkpoints finally weakens. But exact-template collapse gets slightly *worse* (fewer unique vectors, more mass in top templates), so this is not a clean "geometric collapse resolved" story; it looks like RL-driven variance injection, not learned diversity.

### 7.3 Train likelihood: the raw thought is now actively poisonous

| Diagnostic | Checkpoint 1152 (pre-RL) | `nomul_attn_lr_5e-5` (post-RL) |
|---|---:|---:|
| Raw train target NLL | 0.341776 | **6.850221** |
| Zero-mask train target NLL | 1.487041 | 1.486371 |
| Zero-mask NLL penalty (masked − raw) | +1.145265 | **−5.363851** |
| Fraction of examples where raw beats zero-mask | majority (penalty positive) | **0.66%** |

Zero-mask NLL is essentially unchanged from checkpoint 1152 (1.4864 vs 1.4870) — the "no signal" floor is the same. What changed is the raw path: it used to be far better than zero (that was the collapse — a useful but non-specific activation); after GRPO it is far *worse* than zero. On 99.3% of train examples, deleting the RL-tuned thought produces a better teacher-forced likelihood than keeping it.

### 7.4 Causal ranking: zeroing or noising the RL policy beats using it as trained

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.016242 | 0 | 0.012753 | 0 |
| Zero / mask | 0.077310 | **+0.061069 [+0.054734, +0.067565]** | 0.048222 | **+0.035469 [+0.031356, +0.039704]** |
| Global train mean | 0.005847 | −0.010395 [−0.013805, −0.007142] | 0.002789 | −0.009964 [−0.012943, −0.007423] |
| Shuffled user | 0.015105 | −0.001137 [−0.003577, +0.001141] | 0.011111 | −0.001642 [−0.003418, +0.000114] |
| Shuffled target | 0.015105 | −0.001137 [−0.003577, +0.001299] | 0.011111 | −0.001642 [−0.003362, +0.000140] |
| Gaussian noise 0.25x | 0.017379 | +0.001137 [−0.000487, +0.002761] | 0.013121 | +0.000369 [−0.000821, +0.001671] |
| Gaussian noise 0.50x | 0.019165 | **+0.002924 [+0.000650, +0.005197]** | 0.012824 | +0.000071 [−0.001655, +0.001728] |
| Gaussian noise 1.00x | 0.022251 | **+0.006009 [+0.003086, +0.008771]** | 0.014350 | +0.001597 [−0.000186, +0.003502] |

(All 8 generation runs returned 6,157 paired records, zero catalog violations, checkpoint/split hashes matched; shuffled-user and shuffled-target predictions differ on only 1/6,157 examples, so treat them as one control as in the checkpoint tables above.)

#### Complete metrics at all cutoffs

| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.010070 | 0.012344 | 0.013318 | 0.016242 | 0.010070 | 0.011441 | 0.011825 | 0.012753 |
| Zero / mask | 0.025987 | 0.044827 | 0.055871 | 0.077310 | 0.025987 | 0.036725 | 0.041275 | 0.048222 |
| Global train mean | 0.000325 | 0.002761 | 0.003573 | 0.005847 | 0.000325 | 0.001734 | 0.002063 | 0.002789 |
| Shuffled user | 0.007146 | 0.011694 | 0.012181 | 0.015105 | 0.007146 | 0.009973 | 0.010162 | 0.011111 |
| Shuffled target | 0.007146 | 0.011694 | 0.012181 | 0.015105 | 0.007146 | 0.009973 | 0.010162 | 0.011111 |
| Gaussian noise 0.25x | 0.008771 | 0.013968 | 0.014780 | 0.017379 | 0.008771 | 0.011943 | 0.012272 | 0.013121 |
| Gaussian noise 0.50x | 0.005847 | 0.014780 | 0.015430 | 0.019165 | 0.005847 | 0.011355 | 0.011628 | 0.012824 |
| Gaussian noise 1.00x | 0.006172 | 0.016079 | 0.016567 | 0.022251 | 0.006172 | 0.012359 | 0.012562 | 0.014350 |

At every cutoff, zero-masking beats raw by a wide margin (e.g. HR@1 rises from 0.0101 to 0.0260, more than 2.5×), while the global-mean substitution is worse than raw at every cutoff, and shuffled/noise controls track raw closely at HR@1/NDCG@1 and only diverge (mildly, and in the helpful direction for noise) as the cutoff widens — consistent with the HR@10/NDCG@10 pattern above holding across the whole cutoff range rather than being an artifact of one cutoff.

Raw HR@10/NDCG@10 collapse from 0.1152/0.0806 at checkpoint 1152 to **0.0162/0.0128** — a roughly 7× and 6× drop respectively — even though this run started from checkpoint 1152's weights. Zero-masking recovers most of that gap (HR@10 0.0773, about two-thirds of the pre-RL raw level) and is significant in the *helpful* direction, the opposite sign from every SFT checkpoint. Larger Gaussian noise also significantly helps HR@10, monotonically with scale. Only the global-mean substitution is significantly harmful, and shuffled-identity controls remain statistically neutral, as at the SFT checkpoints.

### 7.5 Diagnosis

This is not the SFT-style "activation-dependence without content-dependence" pattern — it is worse. The GRPO update did not teach the attention module to use sample-specific content (shuffled controls are still neutral, consistent with checkpoints 384–1152), and it did not preserve the SFT-era property that *some* nonzero activation helps. Instead, the specific raw output the policy converged to is now off-distribution enough to actively damage both teacher-forced likelihood and held-out ranking, while the noise the GRPO exploration itself injects at generation time happens to land in a less damaging region.

This is consistent with the RL setup documented in `RECON_NOTES.md` (§2 rows 7–9, Flag 5): the GRPO advantage is a *group-relative* baseline computed from each batch's own unperturbed sample, normalized within-group, with only a weak KL penalty (`beta=0.01`) against the SFT policy. Nothing in that objective anchors the policy to be at least as good as the SFT starting point in absolute terms — a systematic, group-wide degradation in the unperturbed arm is invisible to a relative-advantage estimator as long as the *ranking* of perturbed variants within each group stays informative. The result observed here is consistent with that failure mode: the policy drifted to a raw output that is uniformly worse than an unperturbed or randomly perturbed alternative, without the training signal ever penalizing that drift directly.

### 7.6 Limitations

- This is a single run at a single learning rate (`5e-5`, the only one active in `scripts/launch_run.sh`); the commented-out sweep (`1e-4`, `1e-5`, `5e-4`) was not executed, so it is unknown whether this failure is specific to this learning rate or general to this GRPO configuration.
- `scripts/launch_run.sh` shares eval shard-file paths (`output_dir/CDs_and_Vinyl_v4/{0,1}.json`) between its SFT-eval and RL-eval blocks and has no error checking after either `torchrun` or `noise_eval.py`. A crashed eval step silently leaves `merge.py`/`calc.py` to score stale shard files from an unrelated checkpoint under the current run's `final_result.json` and `metric.log`, as happened for this checkpoint's `metric.log` (see the note in §7.1). Any future reading of that file should check the printed source path in the corresponding `run_logs/R3.<jobid>.out`, not just take the last block on faith.
- As with the SFT checkpoints, geometry is measured on train and causal ranking on test; the two splits are not jointly perturbed in the same run.
- The train NLL comparison is at the same teacher-forced target used throughout this report family; it was not re-verified against a beam-search generation NLL.
