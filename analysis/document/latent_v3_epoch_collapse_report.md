# Latent-collapse assessment for `latent_v3` checkpoints 384, 768, and 1152

## Executive verdict

**Yes. All three `latent_v3` checkpoints exhibit severe representational collapse: almost all examples reuse exact FP16 templates, unrelated thoughts are strongly aligned, and centered variance occupies only a few effective dimensions.**

The geometry improves modestly over training but never becomes healthy. Entropy effective rank increases from 3.21 at checkpoint 384 to 5.73 at checkpoint 768 and 6.95 at checkpoint 1152. Mean random-pair cosine falls from 0.9453 to 0.9420 to 0.8957. However, only 1.75%–2.73% of 49,251 stored train thoughts are unique, 98.0%–99.2% of examples belong to repeated templates, and the ten most common templates cover 63.69%–73.50% of the train set.

The failure form changes across epochs. Checkpoint 384 is the most dimensionally collapsed, with PC1 explaining 68.54% of centered variance and only four components required for 90%. Checkpoint 768 has the most concentrated template vocabulary: its largest template covers 21.00% and its top ten cover 73.50%. Checkpoint 1152 is the least directionally and dimensionally collapsed, but it still contains only 1,002 unique FP16 vectors, its top ten templates cover 66.22%, and seven components explain 90% of centered variance.

Zero masking raises train target NLL at every checkpoint, establishing activation dependence. It does not establish sample-specific content dependence. No retained v3 mean, shuffled-thought, noise, validation-NLL, or constrained-ranking intervention artifacts were found, and the checkpoint weights have been deleted. **Causal utilization collapse therefore cannot be tested or reconstructed from the retained v3 artifacts.**

## Scope, retained evidence, and reproducibility

- Dataset and split: `CDs_and_Vinyl`, train
- Samples per checkpoint: 49,251
- Thought dimension: 1,536; one stored thought per interaction
- Stored dtype: FP16
- Train split SHA-256: `864f652064a08cd9d9b492e11cacceafba074cc8f4b1c3469b10a4bcdaba8da4`
- Pairwise diagnostics: 100,000 sampled pairs, seed 42
- PCA and group-variance diagnostics: centered and fitted on all 49,251 thoughts independently at each checkpoint

The original model files no longer exist, but their hashes were recorded during extraction:

| Step | Recorded checkpoint | Recorded SHA-256 |
|---:|---|---|
| 384 | `output_dir/CDs_and_Vinyl_v3/latent/checkpoint-384` | `f4b6e034e26a727174f176baf68125f900cc0d16d334cc6cb4433360d5760871` |
| 768 | `output_dir/CDs_and_Vinyl_v3/latent/checkpoint-768` | `b7c6a188047f6da13324de8143f971e3db73441389c2696b56b50136f51d8e1a` |
| 1152 | `output_dir/CDs_and_Vinyl_v3/latent/checkpoint-1152` | `e5b4bbc244b15ff651d06925ab58b015029bcef347c48b83305be2086b063916` |

The extracted thought banks are sufficient to recompute template counts, pairwise geometry, PCA, and metadata-linked statistics. The saved per-example raw and zero-mask train NLL columns preserve the extraction-time activation analysis. Model-backed likelihood or generation interventions cannot be rerun without restoring the checkpoints.

Primary artifacts:

- [Epoch comparison summary](../results/latent_distribution/latent_v3_epoch_comparison/summary.json)
- [Epoch trajectory figure](../figures/latent_v3_epoch_collapse_trajectory.png)
- Extraction summaries: [384](../evidence/CDs_and_Vinyl/latent_v3_checkpoint_384/train/summary.json), [768](../evidence/CDs_and_Vinyl/latent_v3_checkpoint_768/train/summary.json), [1152](../evidence/CDs_and_Vinyl/latent_v3_checkpoint_1152/train/summary.json)
- Per-checkpoint geometry: [384](../results/latent_distribution/latent_v3_checkpoint_384/train_latent_distribution.json), [768](../results/latent_distribution/latent_v3_checkpoint_768/train_latent_distribution.json), [1152](../results/latent_distribution/latent_v3_checkpoint_1152/train_latent_distribution.json)
- Distribution overviews: [384](../figures/latent_v3_checkpoint_384_latent_distribution_overview.png), [768](../figures/latent_v3_checkpoint_768_latent_distribution_overview.png), [1152](../figures/latent_v3_checkpoint_1152_latent_distribution_overview.png)
- All-vector PCA views: [384](../figures/latent_v3_checkpoint_384_latent_pca_all_scatter.png), [768](../figures/latent_v3_checkpoint_768_latent_pca_all_scatter.png), [1152](../figures/latent_v3_checkpoint_1152_latent_pca_all_scatter.png)
- Interactive t-SNE views: [384](../figures/latent_v3_checkpoint_384_raw_latent_tsne_plotly.html), [768](../figures/latent_v3_checkpoint_768_raw_latent_tsne_plotly.html), [1152](../figures/latent_v3_checkpoint_1152_raw_latent_tsne_plotly.html)

## 1. Epoch trajectory

| Diagnostic | Step 384 (epoch 1) | Step 768 (epoch 2) | Step 1152 (epoch 3) |
|---|---:|---:|---:|
| Raw train target NLL | 1.141149 | 0.712631 | 0.338499 |
| Zero-mask train target NLL | 2.011184 | 1.121504 | 1.116929 |
| Zero-mask NLL penalty | +0.870036 | +0.408873 | +0.778431 |
| Mean sample norm | 99.33 | 82.91 | 80.71 |
| Global-mean vector norm | 96.53 | 80.60 | 76.24 |
| Global-mean norm / mean sample norm | 0.9718 | 0.9722 | 0.9446 |
| Mean cosine to global mean | 0.9722 | 0.9703 | 0.9462 |
| Mean random-pair cosine | 0.9453 | 0.9420 | 0.8957 |
| Exact FP16-unique vectors | 862 | 1,346 | 1,002 |
| Unique fraction | 1.75% | 2.73% | 2.03% |
| Samples in repeated templates | 99.23% | 98.03% | 98.73% |
| Largest-template coverage | 15.67% | 21.00% | 15.90% |
| Top-10-template coverage | 63.69% | 73.50% | 66.22% |
| Top-20-template coverage | 75.57% | 82.37% | 77.86% |
| Mean shuffled-vector L2 | 30.82 | 26.45 | 33.79 |
| PC1 centered variance | 68.54% | 55.75% | 48.78% |
| Top-two centered variance | 80.41% | 74.16% | 67.42% |
| Top-10 centered variance | 99.63% | 95.08% | 94.13% |
| Entropy effective rank | 3.21 | 5.73 | 6.95 |
| Participation ratio | 2.03 | 2.84 | 3.46 |
| Components for 90% variance | 4 | 7 | 7 |

Raw train NLL improves monotonically, but geometric recovery is limited. From checkpoint 384 to 1152, random-pair cosine falls by 0.0496 and effective rank roughly doubles, yet more than 98.7% of samples at the endpoint still belong to repeated templates. The first ten PCs explain 94.13% of endpoint variance in a nominally 1,536-dimensional space.

Template concentration is non-monotonic. It worsens from checkpoint 384 to 768 and partially recovers at checkpoint 1152. The epoch-2 checkpoint is therefore the most template-concentrated even though checkpoint 384 is the most dimensionally concentrated.

## 2. Geometric diagnosis

### Exact-template collapse

All three checkpoints fail the basic diversity gate. Checkpoint 384 maps 49,251 interactions to 862 stored vectors; checkpoint 768 maps them to 1,346; and checkpoint 1152 maps them to 1,002. High-frequency reuse is not a minor quantization tail: between 63.69% and 73.50% of all examples are captured by only ten exact FP16 templates.

Checkpoint 768 is the strongest template failure. One vector is reused by about one-fifth of the train set, and twenty vectors cover 82.37%. Checkpoint 1152 reduces that concentration somewhat, but its 66.22% top-10 coverage remains severe.

### Shared-direction and dimensional collapse

The global-mean vector retains 94.46%–97.22% of a typical vector's norm, while unrelated random pairs have mean cosine 0.8957–0.9453. Thus, template identities occupy a narrow directional cone around a large shared offset.

Centered PCA independently confirms the collapse. Effective ranks of 3.21–6.95 and participation ratios of 2.03–3.46 mean that most centered variance behaves like only a few balanced dimensions. Even at the least collapsed endpoint, seven components capture 90% and ten capture 94.13% of variance.

The agreement among exact-template counts, pairwise cosine, and centered PCA rules out an explanation based only on FP16 equality. Quantization may merge nearby vectors, but it cannot explain the extreme alignment and low-dimensional float32 geometry after loading.

## 3. Preference-linked residual structure

| Diagnostic | Step 384 | Step 768 | Step 1152 |
|---|---:|---:|---:|
| Same-user cosine | 0.9609 | 0.9506 | 0.9151 |
| Same-user minus random | +0.0156 | +0.0086 | +0.0193 |
| Same-target cosine | 0.9478 | 0.9444 | 0.8994 |
| Same-target minus random | +0.0025 | +0.0024 | +0.0037 |
| Same CF-cluster cosine | 0.9463 | 0.9429 | 0.8970 |
| Same CF-cluster minus random | +0.0009 | +0.0009 | +0.0013 |
| Adjusted variance associated with user | 23.83% | 14.12% | 15.84% |
| Adjusted variance associated with target | 3.86% | 3.49% | 4.00% |
| Adjusted variance associated with CF cluster | 1.37% | 0.96% | 1.16% |

Some residual structure is user-linked: same-user pairs are consistently more similar than random pairs, and adjusted user-associated variance is 14%–24%. Target-linked structure is much smaller, while conventional collaborative-filtering cluster structure is weak. These associations show that the few residual directions are not pure random noise, but they do not demonstrate rich or causally used recommendation reasoning.

The exact-template utility eta-squared statistic is not a reliable semantic measure here. Template frequency and group imbalance can make template identity appear predictive without showing that the decoder uses template-specific information.

## 4. Activation dependence and causal status

Zero masking raises train NLL by 0.8700, 0.4089, and 0.7784 at checkpoints 384, 768, and 1152. The nonzero thought slot therefore affects prediction likelihood at every epoch. The penalty is smallest at checkpoint 768 despite that checkpoint having the strongest top-template concentration, demonstrating that activation magnitude and representation diversity are separate axes.

A zero vector is off-manifold and cannot distinguish meaningful sample-specific reasoning from a generic carrier activation. The defining E1 controls—global train mean, shuffled-user, shuffled-target, calibrated noise, and held-out constrained ranking—were not retained for v3. The model weights are also unavailable, so these controls cannot now be generated.

Accordingly:

1. **Exact-template collapse: strongly supported at all three checkpoints.**
2. **Shared-direction collapse: strongly supported at all three checkpoints.**
3. **Centered dimensional collapse: strongly supported, with modest improvement across epochs.**
4. **Activation dependence: supported on the train split.**
5. **Causal utilization collapse: unresolved because in-distribution interventions are unavailable.**

## 5. Retained endpoint ranking shards

Two retained files, `output_dir/CDs_and_Vinyl_v3/0.json` and `1.json`, contain 3,078 and 3,079 prediction records. Concatenating them in shard order produces 6,157 records with the following raw metrics:

| Metric | Value |
|---|---:|
| HR@1 | 0.047101 |
| HR@3 | 0.068865 |
| HR@5 | 0.081046 |
| HR@10 | 0.097775 |
| NDCG@1 | 0.047101 |
| NDCG@3 | 0.059641 |
| NDCG@5 | 0.064639 |
| NDCG@10 | 0.070074 |

The directory placement suggests that these are v3 endpoint predictions, but the shards do not embed a checkpoint path or SHA-256. They should therefore be treated as retained run-level evidence, not cryptographically attributable to checkpoint 1152. They provide no intervention comparison.

## 6. Limitations

- The checkpoint weights have been deleted. Their identities survive only as hashes recorded in extraction and geometry summaries, so the model-backed experiments cannot be independently rerun.
- No v3 validation or test E1 intervention artifacts were found.
- Geometry and zero-mask NLL are train-split measurements and do not establish held-out causal behavior.
- PCA axes are fitted independently at each checkpoint; scalar summaries are comparable, but coordinates are not pointwise aligned across epochs.
- Exact equality is measured at FP16 storage precision, although cosine and PCA independently confirm collapse after float32 loading.
- Direct row-level t-SNE can visually separate identical inputs because of its repulsive objective. Exact counts, cosine, distance, and PCA statistics remain the primary evidence.
- The retained endpoint ranking shards lack embedded checkpoint provenance.

## Conclusion

The retained artifacts are sufficient for a strong representational diagnosis: `latent_v3` remains severely template-, direction-, and dimension-collapsed throughout training. Later epochs broaden the residual subspace and reduce mean alignment, but they do not restore sample-level latent diversity. Checkpoint 768 is the most concentrated in a few templates, and checkpoint 1152 remains severely collapsed despite achieving the best raw train NLL.

The retained evidence is not sufficient for the stronger causal conclusion reached in the v4 report. Activation dependence is present, but only restoring the original model weights would allow mean, shuffle, noise, and held-out ranking interventions to determine whether the decoder used the small remaining sample-specific variation.
