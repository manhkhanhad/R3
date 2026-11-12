# Latent-reasoning collapse assessment for `latent_dev/checkpoint-768`

## Executive verdict

**Yes, checkpoint-768 exhibits the causal/functional form of latent reasoning collapse, but it does not exhibit the severe geometric template collapse documented for the earlier SFT and RL checkpoints.**

The decoder needs a nonzero thought activation: replacing the thought with zero raises validation target NLL by 0.2601 and significantly reduces test ranking. However, replacing it with the global train mean or an unrelated train thought has essentially no effect on NLL or ranking. Even perturbations whose expected L2 distance is comparable to or larger than a natural train-thought shuffle do not reliably affect prediction. This is the defining causal symptom in [Latent Reasoning Collapse in R3 Recommendation Models](latent_reasoning_collapse_r3.md): activation dependence without sample-specific content dependence.

Unlike the previously diagnosed checkpoints, checkpoint-768 has a substantially more diverse latent representation. Of 49,251 stored train thoughts, 48,282 (98.03%) are unique at FP16 precision; mean random-pair cosine is 0.6443 rather than 0.9855–0.9905; and centered-PCA entropy effective rank is 69.81 rather than 2.56–4.82. Therefore, its failure is best described as **latent-utilization collapse or decoder bypass**, not representational/template collapse. The thought encoder produces varied vectors, but the recommendation decoder makes almost no use of which valid vector it receives.

Checkpoint-768 is also the epoch-2 checkpoint from the same SFT training run as the 3-epoch SFT model. Comparing these two endpoints strongly suggests that the final SFT epoch is associated with severe geometric collapse and growing reliance on a shared nonzero activation. This is a longitudinal hypothesis supported by the endpoints, not yet proof that the third epoch itself caused the change; checkpoints within that epoch must be evaluated to establish the trajectory.

## Scope and reproducibility

- Dataset: `CDs_and_Vinyl`
- Architecture: `LatentModel`, hidden dimension 1,536, one thought
- Checkpoint: `output_dir/CDs_and_Vinyl/latent_dev/checkpoint-768`
- Training position: step 768, epoch 2 of the same run whose SFT endpoint was trained for 3 epochs
- Checkpoint SHA-256: `6c4636d5c3d9bc4e04e30919fd27e15f2febc4b9828d67fba8599d4cd7394f17`
- Train split: 49,251 examples; SHA-256 `864f652064a08cd9d9b492e11cacceafba074cc8f4b1c3469b10a4bcdaba8da4`
- Validation split: 6,156 examples; SHA-256 `71a46f1546ab35eacb74a91b859b6c75a199a65ccc2ecfb3c36f6f8b5b8d3acd`
- Test split: 6,157 examples; SHA-256 `944634cd2b9248a4a579f222151c6df140935f56f5ea731cfe4ac21c240e1999`
- Intervention seed: 42
- Pairwise geometry sample: 100,000 pairs, seed 42
- PCA: centered and fitted on all 49,251 train thoughts

Primary artifacts:

- [Validation NLL intervention summary](../results/intervention/latent_dev_checkpoint_768/valid_summary.json)
- [Checkpoint-768 train extraction summary](../evidence/CDs_and_Vinyl/latent_dev_checkpoint_768/train/summary.json)
- [Latent-distribution statistics](../results/latent_distribution/latent_dev_checkpoint_768/train_latent_distribution.json)
- [Three-epoch SFT train extraction summary](../evidence/CDs_and_Vinyl/sft_only/train/summary.json)
- [Three-epoch SFT latent-distribution statistics](../results/latent_distribution/sft_only/train_latent_distribution.json)
- [Latent-distribution overview](../figures/latent_dev_checkpoint_768_latent_distribution_overview.png)
- [All-vector PCA view](../figures/latent_dev_checkpoint_768_latent_pca_all_scatter.png)

## 1. Causal intervention evidence

Positive delta NLL means worse than the raw thought. Confidence intervals are paired bootstrap 95% intervals over all validation examples.

| Intervention | Validation NLL | Delta NLL vs raw | 95% CI | Paired bootstrap p |
|---|---:|---:|---:|---:|
| Raw | 1.137609 | 0 | — | — |
| Zero / mask | 1.397719 | +0.260109 | [+0.252626, +0.267557] | <0.001 |
| Global train mean | 1.137719 | +0.000110 | [-0.000111, +0.000327] | 0.312 |
| Shuffled user | 1.137819 | +0.000210 | [-0.000036, +0.000452] | 0.094 |
| Shuffled target | 1.137782 | +0.000173 | [-0.000080, +0.000429] | 0.185 |
| Gaussian noise 0.25x | 1.137729 | +0.000120 | [-0.000032, +0.000275] | 0.125 |
| Gaussian noise 0.50x | 1.137539 | -0.000070 | [-0.000243, +0.000098] | 0.422 |
| Gaussian noise 1.00x | 1.137510 | -0.000099 | [-0.000326, +0.000122] | 0.387 |

Zeroing is strongly destructive, but all in-distribution replacements and noise perturbations are statistically indistinguishable from raw. The mean/shuffle changes are three orders of magnitude smaller than the zero-mask change.

The test-set constrained-ranking results agree. Intervals below are paired bootstrap 95% intervals for the change relative to raw.

| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw | 0.101186 | 0 | 0.073543 | 0 |
| Zero / mask | 0.085106 | -0.016079 [-0.020465, -0.011856] | 0.058896 | -0.014646 [-0.017750, -0.011636] |
| Global train mean | 0.101186 | +0.000000 [-0.001462, +0.001462] | 0.073673 | +0.000131 [-0.000520, +0.000778] |
| Shuffled user | 0.100861 | -0.000325 [-0.001787, +0.001137] | 0.073722 | +0.000179 [-0.000578, +0.000898] |
| Shuffled target | 0.100861 | -0.000325 [-0.001787, +0.001137] | 0.073722 | +0.000179 [-0.000578, +0.000898] |
| Gaussian noise 0.25x | 0.100536 | -0.000650 [-0.002111, +0.000812] | 0.073361 | -0.000181 [-0.000846, +0.000496] |
| Gaussian noise 0.50x | 0.100536 | -0.000650 [-0.002111, +0.000812] | 0.073229 | -0.000314 [-0.000965, +0.000324] |
| Gaussian noise 1.00x | 0.101023 | -0.000162 [-0.001787, +0.001462] | 0.073522 | -0.000020 [-0.000750, +0.000718] |

All generation variants returned 6,157 records with zero catalog violations. The raw checkpoint achieves HR@1/3/5/10 of 0.051324/0.070976/0.082833/0.101186 and NDCG@1/3/5/10 of 0.051324/0.062724/0.067609/0.073543.

### Causal interpretation

The thought slot matters, but thought identity does not. A valid thought behaves like an enabling activation or soft prompt rather than a causally used, example-specific reasoning state. This is direct evidence that checkpoint-768 fails the diagnosis document's primary acceptance condition: mean, shuffled-user, and shuffled-target thoughts should significantly worsen held-out likelihood and ranking if their sample-specific content is being used.

## 2. Latent geometry evidence

Checkpoint-768 is far less geometrically collapsed than either comparison checkpoint in the diagnosis document.

| Diagnostic | Checkpoint-768 | Earlier SFT | Earlier RL |
|---|---:|---:|---:|
| Train thoughts | 49,251 | 49,251 | 49,251 |
| Mean sample norm | 66.43 | — | — |
| Global-mean vector norm | 53.22 | 72.59 | 95.34 |
| Global-mean norm / mean sample norm | 0.8011 | 0.9926 | 0.9952 |
| Mean cosine to global mean | 0.8035 | — | 0.9952 |
| Mean random-pair cosine | 0.6443 | 0.9855 | 0.9905 |
| Exact FP16-unique vectors | 48,282 | 2,337 | 348 |
| Unique fraction | 98.03% | 4.74% | 0.71% |
| Top-10-template coverage | 0.41% | 32.21% | 82.12% |
| Largest-template coverage | 0.27% | — | 15.61% |
| PC1 centered variance | 23.76% | 48.27% | 77.05% |
| Top-two centered variance | 31.27% | 67.76% | 89.85% |
| Entropy effective rank | 69.81 | 4.82 | 2.56 |
| Participation ratio | 13.79 | 3.35 | 1.63 |
| Components for 90% variance | 200 | — | — |

This checkpoint passes the spirit of the diagnosis document's basic diversity gate. It is still anisotropic—23.76% of centered variance lies in PC1 and participation ratio 13.79 remains small relative to 1,536 coordinates—but it is not a small set of nearly collinear templates. Nearly all stored thoughts are distinct, and unrelated thoughts are separated by substantial angles and distances.

The natural shuffled-thought L2 distance averages 54.49. Expected Gaussian perturbation distances are 16.82, 33.63, and 67.26 for noise scales 0.25x, 0.50x, and 1.00x. Thus, the null causal result cannot be explained by interventions being too small: the 1.00x perturbation is larger than a typical natural shuffle and still leaves NLL and ranking unchanged.

## 3. Preference-linked structure

The representation contains some measurable group structure:

| Pair/group diagnostic | Result |
|---|---:|
| Random-pair cosine | 0.6443 |
| Same-user cosine | 0.6911 |
| Same-target cosine | 0.6567 |
| Same train-only CF-cluster cosine | 0.6492 |
| Adjusted variance associated with user | 9.98% |
| Adjusted variance associated with target | 3.54% |
| Adjusted variance associated with CF cluster | 1.05% |

Same-user similarity exceeds random similarity by 0.0468, but same-target and CF-neighborhood differences are only 0.0124 and 0.0050. The encoder therefore does not produce random noise; it retains user-linked structure and weak target/preference structure. The causal intervention shows that the downstream decoder does not materially use this variation for recommendation.

The reported exact-template eta-squared statistic (0.9877) should not be interpreted as strong template–utility alignment. Because 98% of vectors are unique, template identity has almost one group per sample and severely overfits this statistic.

## 4. Diagnosis relative to the documented failure mode

The evidence separates two phenomena:

1. **Representational/template collapse: not supported for checkpoint-768.** Its train thoughts are diverse by exact counts, cosine similarity, and centered effective rank. This is a major improvement over both earlier checkpoints.
2. **Causal utilization collapse: strongly supported.** Zeroing the slot hurts, while substituting distant, valid, sample-mismatched thoughts does not. The likely mechanism remains an unrestricted history-to-decoder bypass: the decoder obtains sample-specific information directly from history and uses the thought mainly as a generic activation.

Accordingly, saying only that checkpoint-768 “does not collapse” would be misleading. Its latent vectors no longer collapse geometrically, but its recommendation behavior still collapses functionally to near-indifference among valid thoughts. Conversely, describing it as the same 348-template failure seen in the earlier RL checkpoint would also be incorrect.

## 5. Training-duration observation and hypothesis

### Training lineage

Checkpoint-768 and the 3-epoch SFT checkpoint use the same model and training code. Checkpoint-768 is the epoch-2 state, while `output_dir/CDs_and_Vinyl/latent` is the epoch-3 SFT endpoint. This makes their difference more informative than a comparison between independently trained models, although only the two endpoints around the final epoch are currently measured.

### Observed endpoint changes

The train NLL values below use the same 49,251-example extraction protocol. Geometry statistics use the same full-bank PCA and 100,000-pair protocol.

| Diagnostic | Step 768 (epoch 2) | SFT endpoint (epoch 3) | Endpoint change |
|---|---:|---:|---:|
| Raw train target NLL | 0.695813 | 0.714917 | +0.019104 (worse) |
| Zero-mask train target NLL | 0.989908 | 1.795241 | +0.805333 |
| Zero-mask penalty over raw | 0.294095 | 1.080324 | +0.786229; 3.67x larger |
| Global-mean vector norm | 53.22 | 72.59 | +36.4% |
| Global-mean norm / mean sample norm | 0.8011 | 0.9926 | Approaches one shared direction |
| Mean random-pair cosine | 0.6443 | 0.9855 | +0.3412 |
| Exact FP16-unique vectors | 48,282 | 2,337 | -95.2% |
| Unique fraction | 98.03% | 4.74% | -93.29 percentage points |
| Top-10-template coverage | 0.41% | 32.21% | +31.80 percentage points |
| PC1 centered variance | 23.76% | 48.27% | +24.51 percentage points |
| Top-two centered variance | 31.27% | 67.76% | +36.49 percentage points |
| Entropy effective rank | 69.81 | 4.82 | -93.1% |
| Participation ratio | 13.79 | 3.35 | -75.7% |

Two changes occur together across the training endpoints:

1. The latent representation contracts from a diverse, moderately anisotropic space into a highly collinear and repetitive space.
2. The decoder's dependence on the presence of a nonzero latent activation becomes much stronger: the train zero-mask penalty increases by a factor of 3.67.

The raw train NLL is slightly worse at the later SFT endpoint, so the stronger activation dependence is not accompanied by better raw train likelihood in this endpoint comparison. This should be treated as a possible overtraining or optimization warning, not as evidence that the larger carrier is beneficial.

### Proposed training dynamics

The observations support the following working hypothesis:

```text
At epoch 2:
diverse thoughts exist -> decoder ignores most thought identity -> weak shared carrier

Third SFT epoch:
unused variation contracts -> shared direction/norm grows -> zero-mask penalty increases

Later verifier-free RL:
the same shortcut is amplified -> severe template and dimensional collapse
```

This explains why checkpoint-768 can have healthy geometry but already fail the shuffled-thought causal test. The utilization failure appears before the severe geometric collapse: because the decoder does not benefit from sample-specific variation, the recommendation objective supplies little pressure to preserve it. Continued optimization can then strengthen a generally useful shared activation while allowing unused dimensions and templates to collapse.

The relationship is therefore asymmetric. Severe geometric collapse makes sample-specific causal dependence unlikely, but geometric diversity does not guarantee that the decoder uses the available information. Checkpoint-768 demonstrates the second case directly.

### Status of the hypothesis

**Supported observations:** checkpoint-768 and the 3-epoch SFT endpoint are on the same stated training trajectory; geometry is dramatically more collapsed at the later endpoint; and the same-protocol train zero-mask penalty is 3.67 times larger.

**Still assumptions:** collapse progresses during the third epoch rather than changing abruptly at one update; the additional epoch itself is the cause rather than a particular learning-rate, data-order, or optimizer phase; and shuffled-thought content dependence remains near zero from epoch 2 to epoch 3. Two endpoints cannot establish these claims.

The decisive follow-up is to evaluate every available intermediate checkpoint using the same train geometry and validation intervention protocols. Track:

\[
A_t=\operatorname{NLL}_{zero,t}-\operatorname{NLL}_{raw,t}
\]

as activation dependence and

\[
C_t=\operatorname{NLL}_{shuffle,t}-\operatorname{NLL}_{raw,t}
\]

as sample-specific content dependence, alongside random-pair cosine, unique-template concentration, effective rank, and raw validation NLL. The proposed mechanism predicts that effective rank and uniqueness fall, \(A_t\) rises, and \(C_t\) remains near zero. If \(C_t\) is initially positive and later disappears, the stronger conclusion would be that training actively destroys previously used content rather than merely collapsing variation that was never used.

## 6. Limitations

- The validation NLL shuffle variants are deterministic random train-vector replacements, but the current likelihood implementation does not explicitly enforce different-user or different-target membership. The test-ranking implementation does enforce the named exclusion.
- Test `shuffle_user` and `shuffle_target` produced identical predictions and ranks for all 6,157 examples. Their deterministic selector starts from the same candidate and changes it only when the relevant exclusion collides; consequently, they are not independent ranking controls and should count as one effective result.
- Teacher-forced validation NLL and constrained test ranking use different splits and cannot be paired with each other.
- Geometry demonstrates diversity and weak preference association, not semantic reasoning quality or sufficiency.
- This report does not compare checkpoint-768 against an accuracy-matched model with the direct history-to-decoder path removed, so the bypass mechanism is strongly supported but not uniquely proven.
- The training-duration claim currently rests on the epoch-2 and epoch-3 endpoints. It establishes an association across the final epoch, not within-epoch monotonicity or causation.
- The complete shuffled-thought E1 protocol has been run for checkpoint-768 and the later RL checkpoint, but not yet for the 3-epoch SFT endpoint under the same validation/test protocol.

## Conclusion and next action

Checkpoint-768 shows that the two failures are distinct. At epoch 2, the model has diverse latent vectors but already lacks **causal reasoning dependence**. By the epoch-3 SFT endpoint, the same training trajectory is also severely **geometrically collapsed** and has a much larger zero-mask penalty. The evidence therefore supports the hypothesis that the final SFT epoch converts causally unused variation into a stronger shared activation carrier, while stopping short of proving the within-epoch trajectory or causation from two endpoints.

The next priority is not another generic diversity loss. The checkpoint already has substantial latent diversity. Instead, training should force the decoder to use that diversity: constrain or remove the direct history bypass and add an in-distribution counterfactual objective that makes the correct thought outperform mean, shuffled-user, shuffled-target, and nearby wrong thoughts. Before prototype mining, rerun E1 and require those in-distribution interventions—not only zero masking—to significantly worsen validation NLL and test ranking.
