# Avoiding Latent Reasoning Collapse and Shortcut Learning in R3

> Prior diagnosis: [Latent Reasoning Collapse in R3 Recommendation Models](latent_reasoning_collapse_r3.md), [latent_v4](latent_v4_epoch_collapse_report.md), [latent_v5](latent_v5_epoch_collapse_report.md), [latent_v6](latent_v6_epoch_collapse_report.md).

## Experiment map

1. [Attempt 1: annealed attention-pattern schedule](#attempt-1), including the [alpha=0.75 follow-up](#attempt-1-alpha075) and [final-epoch equal three-way follow-up](#attempt-1-threeway)
2. [Attempt 2: in-distribution counterfactual margin loss](#attempt-2)
3. [Attempt 3: multi-negative InfoNCE, temperature curriculum, and geometry regularization](#attempt-3)
4. [Attempt 4: CASE-inspired causal alignment losses](#attempt-4)
5. [Attempt 5: gradient masking on the history-to-target attention path](#attempt-5)
6. [Attempt 6: explicit $\mu/z$ decomposition with structural isolation](#attempt-6)
7. [Attempt 7: VAE-shaped encoder with an aggregate KL prior on $z$](#attempt-7), including [7a](#attempt-7a), [7b](#attempt-7b), the [KL/curriculum ablations](#attempt-7-followup-ab), the [no-curriculum ablation](#attempt-7-no-curriculum), and the [$z$-only geometry correction](#attempt-7-z-method)
8. [Attempt 8: memory-bank semantic alignment for $z$](#attempt-8)
9. [Attempt 9: baseline-preserving latent residual correction](#attempt-9), including the [9a orthogonal-subspace follow-up](#attempt-9a)
10. [Attempt 10: collaborative multi-positive InfoNCE on plain R3](#attempt-10)
11. [Attempt 11: SASRec-similarity-weighted collaborative InfoNCE](#attempt-11)

Conceptual interlude: [Reframing $r_i = \mu + z_i$](#reframing-mu-z).

## Problem statement

Every checkpoint variant analyzed so far (SFT, RL, `latent_v4`, `latent_v5`, `latent_v6`, across steps 384/768/1152) shows the same **functional/utilization collapse**: zeroing the thought vector `r` significantly damages held-out ranking and likelihood, but replacing `r` with the global mean, a shuffled in-distribution thought, or even large Gaussian noise does not. The decoder needs *some* valid-scale activation in the thought slot, but shows no measurable dependence on *which* valid thought it receives.

Geometric diversity (effective rank, exact-template uniqueness, PCA concentration) varies substantially across these variants and does **not** track this functional collapse. Some of the geometrically healthiest checkpoints (e.g. `latent_v4` checkpoint 1152, effective rank 21.32, 92.9% unique vectors) show the *largest* zero-mask penalty yet the *same* null result for shuffled/mean replacement. Geometric spread and causal necessity are separate axes, and no variant tried so far has fixed the causal one.

**This is a property of the published LatentR³ design, not an implementation bug in this codebase.** LatentR³ (arXiv:2505.19092), §3.1 and Eq. 1, explicitly defines the architecture as `x → LLM(x) → r`, then `LLM(x, r) → ŷ` — the final generation step is designed by the original authors to condition on the raw history `x` **and** the thought `r` together, not `r` alone: "we concatenate `x` and `r` as the input to the LLM for next-item prediction." `LatentModel.forward` in `src/latent/layers.py` faithfully implements this: `r` is computed from history via a correctly history-masked pooling attention (`SelfAttentionLayer`), then spliced into the input embeddings, and the second forward pass uses ordinary unrestricted causal attention over the full sequence. Every variant in this repo inherits the same structural property: the decoder is never forced to route information through `r`, because it can already see `x` directly. Standard next-token likelihood provides no training pressure to prevent `r` from collapsing into a generic, low-information activation carrier — a constant or low-rank vector can satisfy "the decoder needs *some* nonzero input in this slot" while carrying almost no sample-specific content.

Verifier-free RL (as used in this project's RL-tuned checkpoints) makes this worse, not better: it increases the thought's norm and its alignment with the global mean while improving train likelihood, consistent with RL amplifying the shared-carrier shortcut rather than learning a semantically structured reasoning space.

## Note: what the `noise_*` interventions actually measure

Referenced throughout every attempt below (`src/prototype_grounding/experiment.py`, `intervention`/`generate`): each `noise_<scale>` mode perturbs the real thought as `r_noisy = r + N(0,1) × std × scale`, where `std` is a single scalar — the population-wide, per-element standard deviation across all extracted training thoughts (`train_vec.std()`, not per-dimension) — and `scale ∈ {0.25, 0.5, 1.0}`. All norms reported anywhere in this document (`mean_vector_norm`, per-example norms, `expected_gaussian_noise_l2`) are **L2 (Euclidean)** — PyTorch's `.norm()` default.

This is not a subtle perturbation. Because `scale` multiplies a std computed from each checkpoint's own population, the resulting noise norm lands at a consistent *fraction of that checkpoint's own typical vector norm* regardless of its absolute scale:

| Checkpoint | Per-element `std` | Typical vector L2 norm | noise_0.25 L2 (% of norm) | noise_0.5 L2 (% of norm) | noise_1.0 L2 (% of norm) |
|---|---:|---:|---:|---:|---:|
| scheduled (1152) | 2.07 | 73.2 | 20.3 (28%) | 40.7 (56%) | 81.3 (111%) |
| margin=0.2 (768) | 2.23 | 73.7 | 21.9 (30%) | 43.7 (59%) | 87.5 (119%) |
| margin=0.5 (768) | 4.50 | 176.3 | 44.1 (25%) | 88.2 (50%) | 176.5 (100%) |
| contrastive V0 (1152) | 9.04 | 343.3 | 88.6 (26%) | 177.2 (52%) | 354.5 (103%) |
| contrastive V2 (768) | 1.52 | 58.9 | 14.9 (25%) | 29.7 (50%) | 59.5 (101%) |

`noise_1.0` injects a random perturbation with L2 norm **roughly equal to the vector's own norm** — comparable in magnitude to the entire signal, not a nudge. That every checkpoint except contrastive V0 shows a negligible ranking effect even at this scale is correspondingly stronger evidence of collapse than "the noise was too weak to matter" would suggest: the decoder tolerates a random reshuffling of `r`'s direction at full-signal magnitude, so long as its rough scale is preserved (since `mask` — an actual zero — still hurts).

For reference, the raw per-element value distributions these `std`s are drawn from (computed directly from each checkpoint's stored `thoughts.pt`; min/max/mean/std exact over all elements, percentiles from a 200k-element subsample):

| Checkpoint | min | max | mean | std | 1%ile | 99%ile |
|---|---:|---:|---:|---:|---:|---:|
| scheduled (1152) | −20.9 | 19.0 | −0.028 | 2.07 | −6.0 | 5.8 |
| margin=0.2 (768) | −20.9 | 16.8 | 0.047 | 2.23 | −5.3 | 5.6 |
| margin=0.5 (768) | −18.9 | 19.3 | −0.019 | 4.50 | −11.1 | 11.0 |
| contrastive V0 (1152) | **−58.0** | **52.3** | 0.021 | **9.04** | **−26.0** | **25.8** |
| contrastive V2 (768) | −6.3 | 7.2 | −0.018 | 1.52 | −3.5 | 3.8 |

Every checkpoint's per-element distribution is roughly zero-centered and Gaussian-ish. Contrastive V0's range is a clear outlier — individual elements reach ~26× its own std at the 1st/99th percentile tails, consistent with the near-1-dimensional, "exploded" geometry reported for it below (effective rank 1.37, PC1 92%): a handful of directions carry unusually large magnitude rather than variance being spread evenly across dimensions.

## Where this leaves us

The shortcut is architecturally permitted by the method itself, and nothing in the current training objective (plain next-token CE, then perplexity-reward RL) counteracts it. Fixing this will require either changing what the decoder is allowed to see, changing what the loss rewards, or both. The attempts below test both directions.

<a id="attempt-1"></a>
## 1. Attempt 1: annealed attention-pattern schedule

### 1.1 Approach

#### 1.1.1 Original schedule

A training-time curriculum that schedules which path(s) the decoder may use, via a scalar `alpha` that ramps `0 → 1` linearly over the first `num_epochs - 1` epochs and then holds `≥ 1` for the final epoch. Per training batch, one random draw `u ~ U(0,1)` selects a mode:

- `alpha ≥ 1` → **both**: unrestricted attention, i.e. vanilla R3 (the deployed configuration).
- `alpha < 1` and `u ≤ alpha` → **reasoning_only**: recommendation-position tokens can attend to the thought token and each other, but not to raw history — a hard bottleneck forcing gradient through `r`.
- `alpha < 1` and `u > alpha` → **history_only**: recommendation-position tokens can attend to history and each other, but not to the thought token — trains the shortcut path in isolation.

Only the second (recommendation) forward pass is masked this way; the first pass that computes `r` from history is untouched. Eval/generation always uses `both`, so validation loss stays comparable to the deployed configuration throughout training.

Implemented additively as `ScheduledLatentModel` (`src/latent/scheduled_attention.py`) subclassing `LatentModel`, plus a separate training entry point (`src/latent/latent_attention_train_scheduled.py`) with an epoch-driven `TrainerCallback`; `layers.py`/`reasoning_dataset.py`/`latent_attention_train.py` are untouched.

Two implementation issues were caught and fixed before trusting any result from this approach:

1. **Residual-stream leak in `history_only`.** Blocking attention to the thought token isn't sufficient to make predictions independent of `r` — the very first predicted token's query sits *at* the thought position, so it inherits `r`'s value through the transformer's residual connection regardless of attention masking (confirmed empirically: a logit diff of 0.34 from perturbing only `r` while attention to it was blocked). Fixed by replacing `r` with the plain pre-splice special-token embedding at that position whenever `history_only` is drawn, verified by the gradient into the `r`-generating layer going to exactly `0.0` in that mode.
2. **`EarlyStoppingCallback(patience=1)`, inherited unchanged from `latent_attention_train.py`, is incompatible with the curriculum.** Eval loss is expected to be non-monotonic during the ramp (it legitimately got worse from epoch 1 to epoch 2 in testing), and patience=1 killed training before epoch 3 — the one phase that actually matters — ever ran. Removed for this variant; `load_best_model_at_end` was also disabled, since the "best" eval epoch by loss is not necessarily the alpha≥1 epoch this curriculum exists to produce.

<a id="attempt-1-alpha075"></a>
#### 1.1.2 Follow-up A: cap alpha at 0.75 so `both` is never trained

Run after [Attempt 2](#attempt-2) and [Attempt 3](#attempt-3), directly motivated by this attempt's original diagnosis: the mode-conditional-shortcut failure requires the model to actually experience unrestricted `both` attention during training in order to learn a policy conditioned on it. Testing that directly: `AttentionScheduleCallback` was extended with an `alpha_max` parameter (`src/latent/latent_attention_train_scheduled.py`) — when `alpha_max < 1.0`, alpha ramps linearly up to that cap and holds there instead of continuing past 1.0 into the deterministic "both" phase, so the model trains exclusively on the stochastic `reasoning_only`/`history_only` mix for the entire run and only ever encounters `both` at eval/inference time. Submitted with `--ramp_epochs 3 --alpha_max 0.75` (single-GPU, same DDP rationale as the original Attempt 1 run). First attempt (job 257500) was preempted at epoch 2 by a higher-priority job sharing the node; resubmitted from scratch on a higher-priority queue as job 257592, completing in 2h59m. Training was notably more stable than every other attempt: eval loss (1.229 → 1.041 → 1.070) only regressed by +0.029 at epoch 3, versus +0.2 or more in every margin/contrastive/schedule variant that ever trains under `both` — consistent with the idea that exposure to `both` itself, not just its position in the curriculum, is implicated in that regression. `load_best_model_at_end=False` for this script, so the saved checkpoint is the true final epoch-3 state, never having seen `both` attention during training at all.

<a id="attempt-1-threeway"></a>
#### 1.1.3 Follow-up B: final-epoch equal three-way curriculum

This follow-up tests whether exposing the decoder to its deployed joint path only after two epochs of separated-path training can recover ranking without losing the sample-specific dependence produced by the alpha=0.75 curriculum.

The experiment contains only the attention curriculum—no μ/z decomposition, KL, contrastive loss, residual correction, or other attempt:

- **epochs 1–2:** alpha increases linearly from 0 to 0.75; each training batch uses either `history_only` or `reasoning_only`, never both;
- **epoch 3:** `history_only`, `reasoning_only`, and unrestricted `both` are sampled independently with probability 1/3 each;
- **evaluation/inference:** always uses the deployed `both` path.

The intended mechanism was: first force each information path to become independently usable, then spend the last epoch teaching the decoder how to combine them in the exact configuration used at inference. Training plus standard ranking completed in job 283592; geometry and all eight causal ranking interventions completed in job 292536.

### 1.2 Experiment results

#### 1.2.1 Original schedule

#### 1.2.2 Training run

Full `CDs_and_Vinyl` dataset (49,251 train examples), `Qwen2.5-1.5B-Instruct` backbone, 3 epochs, batch size 128 (micro-batch 4), 2×L40S GPUs via DDP (job 248939, ~1h13m). Eval loss (always measured under `both`) improved monotonically: 1.274 → 1.073 → 1.020 across epochs 1–3, with no instability at the ramp→`both` transition — if anything, training loss dropped and stabilized sharply right at that transition (e.g. from a noisy ~2.5–4 range during the ramp to a smooth ~1.2–2.1 immediately after), suggesting the model adapts to unrestricted attention easily rather than fighting it.

Caveat: this run used 2-GPU DDP, under which each rank draws its own random mode independently (Python's `random` module is not rank-synchronized) — the intended per-batch schedule was effectively per-GPU-shard for this run, adding noise to the realized mixing ratio without changing the qualitative outcome below.

#### 1.2.3 Geometric result: not obviously improved

Checkpoint-1152 geometry (`analysis/results/latent_distribution/latent_scheduled_checkpoint_1152/`), compared against the existing checkpoint-1152 reports:

| Diagnostic | v4 (1152) | v5 (1152) | v6 (1152 range) | **scheduled** |
|---|---:|---:|---:|---:|
| Unique-vector fraction | 92.92% | 7.94% | 15.74–18.38% | **14.95%** |
| Top-10 template coverage | 0.77% | 54.60% | 21.89–27.75% | **58.22%** |
| Largest single template | 0.27% | 12.50% | — | **13.90%** |
| Entropy effective rank | 21.32 | 40.59 | 57.47 | **14.02** |
| PC1 share of variance | 37.96% | — | — | **48.75%** |
| Random-pair cosine | 0.8854 | — | 0.825–0.907 | **0.9034** |

Template concentration (58% top-10 coverage, 14% largest template) is closer to `v5`'s severe-collapse profile than to `v4`'s healthy one, and effective rank (14.0) is lower than both `v5` and `v6` at checkpoint 1152. The schedule did not improve geometric diversity.

#### 1.2.4 Causal-intervention result: collapse persists

E1 ranking intervention on the held-out test split (n=6,157; constrained 10-beam generation; `src/prototype_grounding/experiment.py generate`, job 249601). Raw performance:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0458 | 0.0684 | 0.0788 | 0.0952 | 0.0458 | 0.0590 | 0.0632 | 0.0685 |
| mask | 0.0268 | 0.0492 | 0.0630 | 0.0854 | 0.0268 | 0.0396 | 0.0452 | 0.0525 |
| mean | 0.0456 | 0.0684 | 0.0781 | 0.0939 | 0.0456 | 0.0589 | 0.0629 | 0.0679 |
| shuffle_user | 0.0456 | 0.0681 | 0.0788 | 0.0942 | 0.0456 | 0.0588 | 0.0631 | 0.0682 |
| shuffle_target | 0.0456 | 0.0681 | 0.0788 | 0.0942 | 0.0456 | 0.0588 | 0.0631 | 0.0682 |
| noise_0.25 | 0.0458 | 0.0677 | 0.0784 | 0.0947 | 0.0458 | 0.0586 | 0.0630 | 0.0683 |
| noise_0.5 | 0.0460 | 0.0681 | 0.0780 | 0.0945 | 0.0460 | 0.0588 | 0.0629 | 0.0682 |
| noise_1.0 | 0.0466 | 0.0679 | 0.0789 | 0.0949 | 0.0466 | 0.0590 | 0.0635 | 0.0686 |

Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42, paired on identical `sample_id`/`user_id`/`target`):

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI |
|---|---:|---|---:|---|
| mask | **−0.0097** | **[−0.0143, −0.0054]** | **−0.0161** | **[−0.0193, −0.0130]** |
| mean | −0.0013 | [−0.0029, 0.0003] | −0.0006 | [−0.0014, 0.0001] |
| shuffle_user | −0.0010 | [−0.0024, 0.0003] | −0.0004 | [−0.0010, 0.0002] |
| shuffle_target | −0.0010 | [−0.0024, 0.0003] | −0.0004 | [−0.0010, 0.0002] |
| noise_0.25 | −0.0005 | [−0.0016, 0.0006] | −0.0003 | [−0.0008, 0.0003] |
| noise_0.5 | −0.0006 | [−0.0023, 0.0008] | −0.0003 | [−0.0010, 0.0004] |
| noise_1.0 | −0.0003 | [−0.0019, 0.0011] | 0.0001 | [−0.0007, 0.0008] |

`mask` is the only mode whose CI excludes zero. Every content-preserving intervention (`mean`, both `shuffle_*` variants, all three noise levels) has a CI straddling zero for both HR@10 and NDCG@10 — statistically indistinguishable from `raw`. This is the identical activation-dependence-without-content-dependence signature seen in SFT, RL, `v4`, `v5`, and `v6`. **The attention-schedule approach did not fix latent reasoning collapse.**

![Scheduled checkpoint-1152 E1 ranking by intervention](../figures/latent_scheduled_e1_ranking_grouped_bars.png)

Notably, the `mask` penalty here (HR@10 −10.2%, NDCG@10 −23.4% relative) is smaller than `v4` checkpoint-1152's zero-mask penalty (HR@10 −33%, NDCG@10 −40% relative), even though geometric collapse is worse. Combined with the observation that training loss dropped sharply right when the schedule reverted to `both`, the most likely explanation is that the curriculum trained a **more capable shortcut**, not a more necessary `r`: dedicating full, isolated gradient signal to `history_only` batches makes the direct history→decoder path stronger on its own than it ever gets in vanilla joint training, and nothing prevents the model from learning a **mode-conditional policy** — rely on `r` only when history is structurally blocked, ignore it when history is available — since the three attention patterns are trivially distinguishable from the input structure itself. Forcing dependence in the restricted regime does not transfer to the merged regime; it just makes the alternative path more attractive there.

This diagnosis is tested directly in [Follow-up A](#attempt-1-alpha075), which caps alpha below 1.0 so the model never sees `both` during training.

#### 1.2.5 Follow-up A: alpha=0.75

##### 1.2.5.1 Geometric result: healthiest random-pair cosine measured in this investigation

| Diagnostic | margin=0.2 (768, best prior causal result) | **alpha=0.75 scheduled (1152)** |
|---|---:|---:|
| Unique-vector fraction | 54.11% | **95.55%** |
| Top-10 template coverage | 15.33% | **0.57%** |
| Entropy effective rank | 110.01 | **57.19** |
| PC1 share of variance | 17.89% | **21.21%** |
| Random-pair cosine | 0.7376 | **0.4790** |

Random-pair cosine of 0.48 is the lowest (least collinear, most diverse) measured anywhere in this investigation.

##### 1.2.5.2 Causal-intervention result: genuine content-dependence with no activation-dependence — the inverse of every prior collapse signature

E1 ranking intervention on the held-out test split (n=6,157; job 257759). Raw performance:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0456 | 0.0687 | 0.0804 | 0.1038 | 0.0456 | 0.0589 | 0.0636 | 0.0712 |
| mask | 0.0494 | 0.0723 | 0.0861 | 0.1064 | 0.0494 | 0.0625 | 0.0681 | 0.0746 |
| mean | 0.0396 | 0.0609 | 0.0744 | 0.0968 | 0.0396 | 0.0519 | 0.0574 | 0.0647 |
| shuffle_user | 0.0398 | 0.0629 | 0.0757 | 0.0970 | 0.0398 | 0.0531 | 0.0584 | 0.0652 |
| shuffle_target | 0.0398 | 0.0629 | 0.0757 | 0.0970 | 0.0398 | 0.0531 | 0.0584 | 0.0652 |
| noise_0.25 | 0.0453 | 0.0681 | 0.0809 | 0.1048 | 0.0453 | 0.0584 | 0.0636 | 0.0713 |
| noise_0.5 | 0.0458 | 0.0685 | 0.0799 | 0.1043 | 0.0458 | 0.0588 | 0.0635 | 0.0713 |
| noise_1.0 | 0.0453 | 0.0694 | 0.0806 | 0.1028 | 0.0453 | 0.0589 | 0.0635 | 0.0707 |

Raw ranking (HR@10=0.1038, NDCG@10=0.0712) is close to margin=0.2's — task performance well preserved despite the training/eval attention-pattern mismatch.

Paired bootstrap 95% CIs vs. `raw`:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | +0.0026 | [−0.0008, 0.0060] | **+0.0035** | **[0.0016, 0.0054]** | NDCG only, positive |
| **mean** | **−0.0070** | **[−0.0104, −0.0036]** | **−0.0065** | **[−0.0086, −0.0046]** | **yes** |
| **shuffle_user** | **−0.0068** | **[−0.0102, −0.0032]** | **−0.0059** | **[−0.0080, −0.0040]** | **yes** |
| **shuffle_target** | **−0.0068** | **[−0.0102, −0.0034]** | **−0.0059** | **[−0.0079, −0.0040]** | **yes** |
| noise_0.25 | 0.0010 | [−0.0005, 0.0024] | 0.0001 | [−0.0007, 0.0009] | no |
| noise_0.5 | 0.0005 | [−0.0015, 0.0024] | 0.0002 | [−0.0008, 0.0012] | no |
| noise_1.0 | −0.0010 | [−0.0032, 0.0011] | −0.0005 | [−0.0017, 0.0008] | no |

![Scheduled alpha=0.75 checkpoint-1152 E1 ranking by intervention](../figures/latent_scheduled_alpha075_e1_ranking_grouped_bars.png)

This is a genuinely different signature from every other checkpoint tested in this entire investigation, including the successful ones. `mean`/`shuffle_user`/`shuffle_target` are all cleanly significant with tight, non-borderline CIs — real content-dependence. But `mask` is **not harmful** (its HR@10 CI straddles zero, and its NDCG@10 delta is significantly *positive*) — the model does not need *some* valid-scale activation in the thought slot the way every other checkpoint (including margin=0.2 and contrastive V0) does. This decouples the two axes that were previously confounded in every "success": content-dependence here cannot be explained away as "the decoder just needs *a* valid-scale vector, and it happens to also prefer the right one" — a zero vector is entirely fine, but the *specific coherent wrong* content in `mean`/`shuffle_*` measurably hurts. Noise remains non-significant at all three scales, including `noise_1.0` (full-magnitude random corruption) — consistent with the sensitivity being about *semantic identity* specifically, not generic scale/magnitude disruption, and reinforcing that this isn't simply "any perturbation hurts."

Combined with the healthiest geometry measured in this investigation and fully preserved task performance, this looks like the strongest and cleanest result so far — arguably better evidence of genuine causal necessity than margin=0.2's, precisely because it isn't confounded by an activation-dependence effect that a trivial "carry *some* signal" shortcut could also produce.

#### 1.2.6 Follow-up B: final-epoch equal three-way curriculum

##### 1.2.6.1 Ranking result

Standard full-beam evaluation (`latent_attention_eval.py`, `n=6,157`) produced:

| metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| NDCG | 0.0503 | 0.0609 | 0.0658 | **0.0722** |
| HR | 0.0503 | 0.0689 | 0.0807 | **0.1007** |

The E1 `raw` condition closely reproduces it (HR@10=0.1013, NDCG@10=0.0725). Full intervention results are:

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0503 | 0.0690 | 0.0799 | **0.1013** | 0.0503 | 0.0610 | 0.0655 | **0.0725** |
| mask | 0.0413 | 0.0590 | 0.0716 | 0.0905 | 0.0413 | 0.0513 | 0.0565 | 0.0626 |
| mean | 0.0495 | 0.0687 | 0.0796 | 0.1015 | 0.0495 | 0.0606 | 0.0650 | 0.0721 |
| shuffle_user | 0.0489 | 0.0684 | 0.0791 | 0.1004 | 0.0489 | 0.0601 | 0.0645 | 0.0713 |
| shuffle_target | 0.0489 | 0.0684 | 0.0791 | 0.1004 | 0.0489 | 0.0601 | 0.0645 | 0.0713 |
| noise_0.25 | 0.0502 | 0.0703 | 0.0804 | 0.1009 | 0.0502 | 0.0615 | 0.0657 | 0.0723 |
| noise_0.5 | 0.0502 | 0.0698 | 0.0802 | 0.1005 | 0.0502 | 0.0613 | 0.0656 | 0.0721 |
| noise_1.0 | 0.0505 | 0.0702 | 0.0807 | 0.1005 | 0.0505 | 0.0616 | 0.0660 | 0.0724 |

Paired bootstrap 95% intervals use 2,000 resamples with seed 42. Delta is `raw − variant`, so a positive value means the intervention hurts ranking:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | significant? |
|---|---:|---|---:|---|:---:|
| **mask** | **+0.0109** | **[+0.0065, +0.0153]** | **+0.0098** | **[+0.0070, +0.0127]** | **yes, both** |
| mean | −0.0002 | [−0.0026, +0.0021] | +0.0004 | [−0.0007, +0.0014] | no |
| shuffle_user | +0.0010 | [−0.0016, +0.0034] | +0.0011 | [−0.0003, +0.0025] | no |
| shuffle_target | +0.0010 | [−0.0015, +0.0036] | +0.0011 | [−0.0001, +0.0024] | no |
| noise_0.25 | +0.0005 | [−0.0011, +0.0019] | +0.0001 | [−0.0007, +0.0010] | no |
| noise_0.5 | +0.0008 | [−0.0010, +0.0024] | +0.0003 | [−0.0004, +0.0010] | no |
| noise_1.0 | +0.0008 | [−0.0011, +0.0028] | +0.0001 | [−0.0007, +0.0009] | no |

This is a clean **activation-dependence without content-dependence** result. Replacing the latent by zero significantly damages both metrics, but replacing it by the global mean, a different sample's latent, or substantial Gaussian noise is statistically neutral. The decoder needs a nonzero, in-distribution-looking activation at the thought position; it does not measurably need the correct sample-specific content.

The third-epoch joint-path exposure also does not deliver its ranking objective. It scores HR@10/NDCG@10=0.1013/0.0725 in E1, versus 0.1038/0.0712 for the earlier alpha=0.75-only schedule and 0.1152/0.0806 for unrestricted `v4`. Relative to alpha=0.75, one metric moves slightly down and the other slightly up; relative to baseline, both remain materially lower. There is no evidence of recovered recommendation performance.

##### 1.2.6.2 Geometry result

Despite the causal null result, this checkpoint has exceptionally broad global geometry:

| diagnostic | v4 baseline | alpha=0.75 only | final three-way |
|---|---:|---:|---:|
| entropy effective rank | 21.3 | 57.2 | **113.3** |
| participation ratio | 5.5 | 13.0 | **25.4** |
| PC1 variance fraction | 0.380 | 0.212 | **0.133** |
| components for 90% variance | 72 | 222 | **318** |
| random-pair cosine | 0.885 | 0.479 | **0.422** |
| cosine to global mean | 0.941 | 0.692 | **0.650** |
| user adjusted η² | 0.173 | **0.396** | 0.315 |
| target adjusted η² | 0.057 | **0.248** | 0.182 |
| CF-cluster adjusted η² | 0.018 | **0.188** | 0.128 |
| unique-vector fraction (fp16) | 0.929 | **0.956** | 0.916 |

The final three-way schedule spreads variance across far more dimensions than either comparison and lowers global cosine concentration. However, its user/target/cluster organization is weaker than the alpha=0.75-only schedule, and its exact-vector uniqueness is also lower. Most importantly, none of the dramatic global-rank improvement translates into sensitivity to the correct latent content. This is another direct demonstration that high effective rank and low random-pair cosine can coexist with a causally interchangeable latent.

**Verdict:** the final-epoch equal three-way curriculum fails its intended trade-off. It does not recover baseline ranking, and it converts the earlier alpha=0.75 schedule's content-sensitive signature into a stronger but content-agnostic dependence on latent presence. The result is consistent with the unrestricted `both` batches reopening the direct-history shortcut or allowing a generic thought-slot activation strategy, but this experiment does not isolate which mechanism dominates. Either way, the geometry cannot rescue the causal conclusion: this schedule should not replace alpha=0.75 when the objective is sample-specific latent quality.

Artifacts: `analysis/results/intervention/latent_scheduled_final_threeway_checkpoint_1152/` and `analysis/results/latent_distribution/latent_scheduled_final_threeway_checkpoint_1152/train_latent_distribution.json`.

<a id="attempt-2"></a>
## 2. Attempt 2: in-distribution counterfactual margin loss

### 2.1 Approach

Motivated directly by attempt 1's failure mode: rather than restricting attention during training and hoping it transfers to the unrestricted deployed configuration, compute a loss term **directly under `both` attention** (the actual deployed configuration) that requires the correct thought to beat a wrong one:

\[
-\log P(y \mid x, r_{\text{correct}}) + m \le -\log P(y \mid x, r_{\text{wrong}})
\]

`r_wrong` is another example's thought from the same training batch (shuffled by one position), detached before use so the hinge term shapes the decoder and the correct branch's own `r`, but not the borrowed negative's generation. This directly targets the exact number that failed above (shuffle/mean delta ≈ 0 under `both`) instead of relying on a training-time proxy.

Implemented additively as `MarginLatentModel` (`src/latent/margin_loss_attention.py`) and `src/latent/latent_attention_train_margin.py`, using vanilla unrestricted attention throughout (no schedule) to isolate this approach's effect. Verified on a toy model: gradient flows into the `r`-generating layer from both branches, the batch-size-1 edge case is a harmless no-op, and the negative branch's detach is structural (not just intended).

#### 2.1.1 Implementation issue caught before trusting any result: silent gradient-accumulation scale bug

The first full training run (job 249779) completed, but its logged `'loss'` (~10.7–11.3) did not match the model's own reported `ce_loss + margin_loss` (~0.45–0.58) — roughly an order of magnitude off. Root cause: `transformers.Trainer` decides whether to divide the loss by `gradient_accumulation_steps` before backward based purely on whether the model's `forward()` signature has a `**kwargs` catch-all (`trainer.py`, `model_accepts_loss_kwargs`) — it assumes such a model must already be doing its own `num_items_in_batch`-aware normalization, and skips its own compensating division. `MarginLatentModel.forward()` has `**kwargs` (inherited from the original signature) but computes its own scalar loss manually, never using `num_items_in_batch`. Result: Trainer skipped its division, we never did it ourselves, so **the effective gradient was `gradient_accumulation_steps`× (32, for this run) too large for every optimizer step** — not a display artifact, an actual training bug. This plausibly explains that run's eval_loss trajectory getting *worse* in epoch 3 (1.220 → 1.036 → 1.258) rather than improving, consistent with destabilizingly-large effective steps late in training. (Vanilla and scheduled-attention checkpoints are unaffected — they delegate loss computation entirely to `Qwen2ForCausalLM`'s built-in path, which correctly consumes `num_items_in_batch`.)

Fixed by storing the real `gradient_accumulation_steps` as a model attribute at training-script setup and dividing the returned loss by it inside `forward()`'s training branch (default `1`, a no-op, for eval/generation and the unit tests). Verified on the toy model: loss and gradient norm both scale by exactly `1/grad_accum_steps` as expected. The original run's checkpoint was discarded without analysis (confounded by the near-30x effective learning-rate spike, not a fair test of the margin-loss idea) and retraining was submitted (`scripts/latent_margin_full.sbatch`, full `CDs_and_Vinyl`, 2×GPU DDP, `margin=0.2`, `margin_weight=1.0`, job 250502).

The fixed run's eval loss (1.234 → 1.036 → 1.250) still regressed in epoch 3 exactly as the buggy run had (1.220 → 1.036 → 1.258) — ruling out the gradient-scale bug as the cause of that regression; it appears to be an intrinsic property of the margin-loss dynamic (`margin_loss` crept up across epochs: 0.201 → 0.203 → 0.253, consistent with `margin_weight=1.0` pulling somewhat against the main task objective as training progresses). `load_best_model_at_end=True` kept the epoch-2 checkpoint (`checkpoint-768`, eval_loss 1.036) in both runs; all results below use this checkpoint.

### 2.2 Experiment results

#### 2.2.1 Geometric result: healthiest of any checkpoint tested

Checkpoint-768 geometry (`analysis/results/latent_distribution/latent_margin_checkpoint_768/`):

| Diagnostic | v4 (1152) | v5 (1152) | v6 (1152) | scheduled (1152) | **margin (768)** |
|---|---:|---:|---:|---:|---:|
| Unique-vector fraction | 92.92% | 7.94% | 15.74–18.38% | 14.95% | **54.11%** |
| Top-10 template coverage | 0.77% | 54.60% | 21.89–27.75% | 58.22% | **15.33%** |
| Entropy effective rank | 21.32 | 40.59 | 57.47 | 14.02 | **110.01** |
| PC1 share of variance | 37.96% | — | — | 48.75% | **17.89%** |
| Components for 90% variance | 44 | — | — | — | **250** |
| Random-pair cosine | 0.8854 | — | 0.825–0.907 | 0.9034 | **0.7376** |

Effective rank of 110 is roughly 2–5× every prior checkpoint's best result, with the lowest PC1 concentration and lowest random-pair cosine seen in this investigation.

#### 2.2.2 Causal-intervention result: genuine content-dependence, for the first time

E1 ranking intervention on the held-out test split (n=6,157; job 250597). Raw performance:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0573 | 0.0788 | 0.0906 | 0.1082 | 0.0573 | 0.0696 | 0.0745 | 0.0802 |
| mask | 0.0468 | 0.0694 | 0.0806 | 0.1002 | 0.0468 | 0.0598 | 0.0644 | 0.0708 |
| mean | 0.0490 | 0.0700 | 0.0827 | 0.1002 | 0.0490 | 0.0612 | 0.0664 | 0.0721 |
| shuffle_user | 0.0515 | 0.0726 | 0.0845 | 0.1028 | 0.0515 | 0.0636 | 0.0685 | 0.0744 |
| shuffle_target | 0.0515 | 0.0726 | 0.0845 | 0.1028 | 0.0515 | 0.0636 | 0.0685 | 0.0744 |
| noise_0.25 | 0.0573 | 0.0783 | 0.0908 | 0.1072 | 0.0573 | 0.0694 | 0.0745 | 0.0798 |
| noise_0.5 | 0.0572 | 0.0781 | 0.0901 | 0.1077 | 0.0572 | 0.0693 | 0.0742 | 0.0799 |
| noise_1.0 | 0.0583 | 0.0788 | 0.0906 | 0.1080 | 0.0583 | 0.0702 | 0.0750 | 0.0807 |

Raw ranking (HR@10=0.1082, NDCG@10=0.0802) is close to `v4` checkpoint-1152's baseline (HR@10=0.1152, NDCG@10=0.0806) — unlike the attention schedule, this approach did not cost task performance.

Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42):

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0080 | [−0.0114, −0.0045] | −0.0094 | [−0.0123, −0.0066] | yes |
| **mean** | **−0.0080** | **[−0.0107, −0.0054]** | **−0.0081** | **[−0.0108, −0.0057]** | **yes** |
| **shuffle_user** | **−0.0054** | **[−0.0081, −0.0028]** | **−0.0058** | **[−0.0081, −0.0035]** | **yes** |
| **shuffle_target** | **−0.0054** | **[−0.0081, −0.0026]** | **−0.0058** | **[−0.0081, −0.0035]** | **yes** |
| noise_0.25 | −0.0010 | [−0.0023, 0.0003] | −0.0003 | [−0.0012, 0.0005] | no |
| noise_0.5 | −0.0005 | [−0.0019, 0.0008] | −0.0002 | [−0.0009, 0.0005] | no |
| noise_1.0 | −0.0002 | [−0.0018, 0.0015] | 0.0005 | [−0.0004, 0.0014] | no |

**For the first time across every checkpoint tested (SFT, RL, `v4`, `v5`, `v6`, the attention-schedule attempt), `mean` and both `shuffle_*` variants have 95% CIs that exclude zero** — `mean`'s effect size (−0.0080 HR@10) essentially matches `mask`'s (−0.0080). The decoder no longer just needs *some* valid activation in the thought slot; substituting a wrong-but-valid thought now measurably hurts, at a magnitude comparable to removing it outright. Noise interventions remain non-significant, consistent with every prior checkpoint — isotropic Gaussian noise is a weaker, more generic corruption than swapping in another real thought, and this was never expected to be the sensitive test.

![Margin=0.2 checkpoint-768 E1 ranking by intervention](../figures/latent_margin_e1_ranking_grouped_bars.png)

Combined with the geometric result (far higher effective rank, far lower concentration than anything tested before) and the fact that raw ranking performance was not sacrificed, this is the first approach in this investigation that looks like a genuine fix for latent reasoning collapse rather than a different flavor of the same shortcut.

<a id="attempt-2-margin05"></a>
#### 2.2.3 Margin=0.5 follow-up: raising the margin made things worse, not better

To test whether a larger target gap would strengthen the effect further, the same setup was retrained with `margin=0.5` (everything else identical: `margin_weight=1.0`, single random in-batch negative, `checkpoint-768` again selected as best-eval). Result: **a regression, not an improvement.** Trained twice (the first checkpoint was lost mid-evaluation to an unrelated infrastructure issue — its directory was removed by something outside this pipeline — so it was retrained from scratch for a clean, fully consistent record; both runs' `checkpoint_sha256` differ, confirming they are genuinely different training runs, not a cache artifact); numbers below are from the second, complete run.

Geometry collapsed far more severely than margin=0.2 in both runs, though the specific signature varied between them — consistent with margin=0.5 being a genuinely unstable optimization regime rather than one reproducible failure point:

| Diagnostic | margin=0.2 (768) | margin=0.5 run 1 (768) | **margin=0.5 run 2 (768, reported below)** |
|---|---:|---:|---:|
| Unique-vector fraction | 54.11% | 15.91% | **2.66%** |
| Top-10 template coverage | 15.33% | 34.37% | **13.04%** |
| Entropy effective rank | 110.01 | 6.97 | **12.84** |
| PC1 share of variance | 17.89% | 54.28% | **31.44%** |
| Random-pair cosine | 0.7376 | 0.9891 | **0.9986** |

Raw ranking performance also dropped (HR@10 0.1082→0.1025, NDCG@10 0.0802→0.0734), and the full E1 intervention (n=6,157, all 8 modes, all sha256-matched to the same checkpoint) showed the content-dependence gains had vanished entirely — and replicated cleanly across both training runs:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0494 | 0.0718 | 0.0845 | 0.1025 | 0.0494 | 0.0623 | 0.0676 | 0.0734 |
| mask | 0.0141 | 0.0281 | 0.0378 | 0.0505 | 0.0141 | 0.0222 | 0.0262 | 0.0304 |
| mean | 0.0494 | 0.0716 | 0.0846 | 0.1025 | 0.0494 | 0.0622 | 0.0675 | 0.0733 |
| shuffle_user | 0.0495 | 0.0713 | 0.0827 | 0.1020 | 0.0495 | 0.0621 | 0.0668 | 0.0731 |
| shuffle_target | 0.0495 | 0.0713 | 0.0827 | 0.1020 | 0.0495 | 0.0621 | 0.0668 | 0.0731 |
| noise_0.25 | 0.0495 | 0.0715 | 0.0841 | 0.1028 | 0.0495 | 0.0621 | 0.0674 | 0.0734 |
| noise_0.5 | 0.0494 | 0.0713 | 0.0833 | 0.1028 | 0.0494 | 0.0621 | 0.0670 | 0.0733 |
| noise_1.0 | 0.0502 | 0.0724 | 0.0836 | 0.1026 | 0.0502 | 0.0629 | 0.0676 | 0.0737 |

Paired bootstrap 95% CIs vs. `raw`:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0520 | [−0.0590, −0.0453] | −0.0430 | [−0.0477, −0.0380] | yes |
| mean | 0.0000 | [−0.0015, 0.0015] | −0.0001 | [−0.0008, 0.0007] | no |
| shuffle_user | −0.0005 | [−0.0019, 0.0008] | −0.0003 | [−0.0011, 0.0004] | no |
| shuffle_target | −0.0005 | [−0.0019, 0.0008] | −0.0003 | [−0.0010, 0.0004] | no |
| noise_0.25 | 0.0003 | [−0.0011, 0.0018] | 0.0000 | [−0.0008, 0.0008] | no |
| noise_0.5 | 0.0003 | [−0.0011, 0.0018] | −0.0001 | [−0.0010, 0.0007] | no |
| noise_1.0 | 0.0002 | [−0.0018, 0.0021] | 0.0003 | [−0.0007, 0.0013] | no |

Every content-preserving mode's CI includes zero — even `noise_1.0`, whose CI excluded zero for contrastive V0 below, stays flat here. Meanwhile `mask`'s penalty grew *larger* than margin=0.2's (−51%/−43% relative here, vs. −7%/−12% at margin=0.2) — the model became more dependent on *some* activation being present while becoming completely indifferent to its content. This is a more extreme version of the same failure every other checkpoint showed, not a partial success, and it reproduced across two independent training runs despite their differing specific geometric signatures.

![Margin=0.5 checkpoint-768 E1 ranking by intervention](../figures/latent_margin_m0.5_e1_ranking_grouped_bars.png)

Interpretation: `margin_loss` was pinned exactly at the 0.5 ceiling for the entire training run and never moved (compare margin=0.2, where it similarly plateaued near its own, lower ceiling) — a hinge loss applies constant, non-saturating gradient pressure for as long as the margin is unmet, unlike a softmax-based loss which saturates gracefully as separation grows. With a target the single-random-negative scheme couldn't realistically satisfy, the model was pushed toward a degenerate solution that satisfies the loss's letter (make wrong-`r` predictions generically worse, inflating the `mask` penalty) without the intended semantic separation, letting `r`'s content collapse entirely since nothing else constrained its diversity. This matches a documented failure mode in the triplet/metric-learning literature: a margin larger than the achievable distance can scramble the learned features rather than sharpen them.

<a id="attempt-3"></a>
## 3. Attempt 3: multi-negative InfoNCE, temperature curriculum, and geometry regularization

### 3.1 Approach

Directly motivated by the margin=0.5 failure mechanism above. Three changes, implemented as an ablation ladder so each can be attributed independently rather than confounded in one combined run:

- **Multi-negative InfoNCE** (replaces the single-random-negative hinge): every other example in the same micro-batch serves as a negative (3 negatives at `micro_batch_size=4`, not 1), built via the same deterministic roll-and-splice trick generalized to every nonzero shift, giving `micro_batch_size` total forward passes per micro-batch. Negatives remain detached. Loss is softmax cross-entropy (`-NLL/temperature` as logits) over {correct, wrong_1, ..., wrong_{B-1}} instead of a hinge over one comparison.
- **Temperature curriculum** (replaces the fixed hinge margin): `temperature` anneals linearly from `temp_start` to `temp_end` over training. A softmax loss is always differentiable and saturates gracefully as separation grows, so there is no "unreachable target" failure mode analogous to margin=0.5 — this directly targets the confirmed mechanism above.
- **VICReg-style geometry regularizer**: a variance term (penalizes any dimension of `r` whose per-batch std falls below a target) and a covariance term (penalizes correlation between dimensions), computed directly on the positive thought vectors and decoupled from the contrastive loss. Directly optimizes the same effective-rank/PC1-concentration diagnostics used throughout this investigation, independent of whatever the contrastive loss does. Caveat: at `micro_batch_size=4` the covariance estimate has only 3 degrees of freedom for a 1536-dim vector and is noisy per-step; kept at low weight (`geom_cov_weight=0.04`, matching the VICReg paper's var:cov ratio) as a cumulative-over-many-steps nudge rather than a precise per-step signal.

Implemented additively as `ContrastiveGeometryLatentModel` (`src/latent/contrastive_geometry_attention.py`) and `src/latent/latent_attention_train_contrastive.py`, including the same `grad_accum_steps` fix established in [Attempt 2](#attempt-2) (required again here since this model also has `**kwargs` in its `forward()` signature). Verified on a toy model: contrastive loss at initialization matches `log(batch_size)` exactly (no separation yet, as expected), loss and gradient norm scale by exactly `1/grad_accum_steps`, and the batch-size-1 edge case is a harmless no-op.

Three variants submitted to isolate each component's contribution, all full `CDs_and_Vinyl`, 2×GPU DDP, 3 epochs:

| Variant | Job | `temp_start`→`temp_end` | Geometry regularizer |
|---|---|---|---|
| V0 — InfoNCE only | 252238 | 0.1 → 0.1 (fixed, hard) | off |
| V1 — InfoNCE + curriculum | 252239 | 1.0 → 0.1 | off |
| V2 — full combination | 252220 | 1.0 → 0.1 | on (`geom_var_weight=1.0`, `geom_cov_weight=0.04`) |

V0 fixes temperature at the eventual hard target from step 1 with no ramp — the InfoNCE analogue of margin=0.5's fixed-hard-target setup — so if V0 alone trains without collapsing, that tells us the softmax's smooth saturation avoids the hinge's failure mode even without a curriculum, simplifying the eventual recipe; if V0 collapses like margin=0.5 did, that tells us the curriculum (V1) is genuinely load-bearing.

### 3.2 Experiment results

#### 3.2.1 Training dynamics: the curriculum (V1, V2) never separated; the fixed hard temperature (V0) did — via an extreme shortcut

All three completed 3 epochs cleanly (exit 0). `eval_loss` followed the same familiar non-monotonic shape as every margin run for V1/V2 (≈1.22 → ≈1.04 → ≈1.25, worse at epoch 3), so `load_best_model_at_end` kept `checkpoint-768` (epoch 2) for both. **V0 was the only run with monotonically improving `eval_loss`** (1.597 → 1.279 → 1.183), so it kept `checkpoint-1152` (the final epoch) — the first checkpoint in this investigation where the fully-trained final state was the one used.

The `contrastive_loss` trajectories diverged sharply between the curriculum runs and V0:

| Variant | epoch 1 | epoch 2 | epoch 3 | vs. `log(4)=1.386` (no-separation baseline) |
|---|---:|---:|---:|---|
| V1 (curriculum) | 1.384 | 1.386 | **0.432** | stuck at baseline until the very end |
| V2 (curriculum + geometry) | 1.387 | 1.387 | 1.355 | never moved off baseline |
| V0 (fixed hard temp) | 1.408 | **0.347** | 0.347 | separated by epoch 2, held stable |

The curriculum spends most of training at soft temperatures with little separating pressure, and only reaches its hard end-point (`temp_end=0.1`) right at the end — V2 never got there in time to separate at all (worse: its `cov_loss` actively *increased* over training, 0.85→1.04→1.12, and `var_loss` stayed flat, so the geometry regularizer gained no traction either). V1 did start separating once temperature reached ~0.1 late in epoch 3, but by then eval_loss had already regressed, so the checkpoint that survived (`load_best_model_at_end`, epoch 2) predates that separation entirely — the *good* checkpoint (epoch 3) was deleted by `save_total_limit=1`'s rotation before it could be evaluated.

V0, given the hard temperature the whole time, did separate — but its `cov_loss` diagnostic (unconstrained, since V0 has no geometry regularizer) exploded from ~169k at epoch 2 to ~103k at epoch 3, orders of magnitude beyond V1/V2's ~1–30. That turned out to be a real warning sign, not noise.

#### 3.2.2 Geometric result: V0 exploded into a near-1-dimensional shortcut; V2 looked the healthiest ever measured; V1's surviving checkpoint is collapsed

| Diagnostic | margin=0.2 (768, best prior result) | V0 (1152) | V1 (768, pre-separation) | V2 (768) |
|---|---:|---:|---:|---:|
| Mean vector L2 norm | ~74 | **343.3** | 78.8 | 58.9 |
| Unique-vector fraction | 54.11% | 5.42% | **1.83%** | **98.08%** |
| Top-10 template coverage | 15.33% | 31.32% | **68.07%** | **0.42%** |
| Entropy effective rank | 110.01 | **1.37** | **3.95** | **316.13** |
| PC1 share of variance | 17.89% | **92.44%** | 41.56% | **1.95%** |
| Random-pair cosine | 0.7376 | 0.9827 | 0.9760 | 0.9804 |

V2's centered-PCA numbers (effective rank 316, PC1 1.95%, 98% unique) are the best ever measured in this investigation — but sit oddly alongside a raw pairwise cosine of 0.98 (near-parallel vectors) and a mean-vector-norm ratio near 1.0, suggesting most of each vector's magnitude comes from a large shared direction with a rich residual on top, rather than straightforwardly diverse vectors. V0's numbers are the opposite extreme: the most collapsed geometry measured anywhere in this investigation (effective rank 1.37, 92% of variance in one component), alongside an exploded mean norm (343 vs. the usual ~60–80) — consistent with the `cov_loss` explosion seen during training. V1's surviving (pre-separation) checkpoint is collapsed in the ordinary way, similar in character to `v5`/scheduled/margin=0.5.

#### 3.2.3 Causal-intervention result: V0 shows the most complete positive signal yet, at the cost of roughly halving raw performance; V1 and V2 both fail

**V2 (full combination) — clean failure**, matching the training-time signal:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0549 | 0.0773 | 0.0864 | 0.1085 | 0.0549 | 0.0679 | 0.0716 | 0.0787 |
| mask | 0.0326 | 0.0510 | 0.0607 | 0.0757 | 0.0326 | 0.0433 | 0.0473 | 0.0521 |
| mean | 0.0547 | 0.0778 | 0.0879 | 0.1090 | 0.0547 | 0.0680 | 0.0721 | 0.0789 |
| shuffle_user | 0.0546 | 0.0771 | 0.0872 | 0.1082 | 0.0546 | 0.0676 | 0.0717 | 0.0785 |
| shuffle_target | 0.0546 | 0.0771 | 0.0872 | 0.1082 | 0.0546 | 0.0676 | 0.0717 | 0.0785 |
| noise_0.25 | 0.0547 | 0.0773 | 0.0864 | 0.1083 | 0.0547 | 0.0677 | 0.0714 | 0.0785 |
| noise_0.5 | 0.0551 | 0.0768 | 0.0867 | 0.1083 | 0.0551 | 0.0676 | 0.0716 | 0.0785 |
| noise_1.0 | 0.0549 | 0.0773 | 0.0866 | 0.1083 | 0.0549 | 0.0678 | 0.0716 | 0.0787 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0328 | [−0.0378, −0.0278] | −0.0266 | [−0.0305, −0.0229] | yes |
| mean | 0.0005 | [−0.0006, 0.0016] | 0.0002 | [−0.0005, 0.0008] | no |
| shuffle_user | −0.0003 | [−0.0016, 0.0010] | −0.0003 | [−0.0009, 0.0005] | no |
| shuffle_target | −0.0003 | [−0.0016, 0.0010] | −0.0003 | [−0.0010, 0.0004] | no |
| noise_0.25 | −0.0002 | [−0.0015, 0.0011] | −0.0002 | [−0.0007, 0.0002] | no |
| noise_0.5 | −0.0002 | [−0.0016, 0.0013] | −0.0002 | [−0.0008, 0.0004] | no |
| noise_1.0 | −0.0002 | [−0.0018, 0.0015] | −0.0000 | [−0.0008, 0.0008] | no |

Only `mask` is significant, despite the striking geometry — reinforcing, once again, that geometric diversity is neither necessary nor sufficient for causal necessity.

![Contrastive V2 checkpoint-768 E1 ranking by intervention](../figures/latent_contrastive_v2_e1_ranking_grouped_bars.png)

**V0 (fixed hard temperature) — the most thorough positive signal in this investigation, but at real cost:**

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0266 | 0.0364 | 0.0413 | 0.0521 | 0.0266 | 0.0323 | 0.0343 | 0.0379 |
| mask | 0.0055 | 0.0135 | 0.0188 | 0.0266 | 0.0055 | 0.0100 | 0.0122 | 0.0147 |
| mean | 0.0182 | 0.0239 | 0.0281 | 0.0370 | 0.0182 | 0.0215 | 0.0232 | 0.0261 |
| shuffle_user | 0.0182 | 0.0237 | 0.0281 | 0.0357 | 0.0182 | 0.0213 | 0.0231 | 0.0255 |
| shuffle_target | 0.0182 | 0.0237 | 0.0281 | 0.0357 | 0.0182 | 0.0213 | 0.0231 | 0.0255 |
| noise_0.25 | 0.0258 | 0.0359 | 0.0408 | 0.0536 | 0.0258 | 0.0316 | 0.0337 | 0.0378 |
| noise_0.5 | 0.0237 | 0.0304 | 0.0349 | 0.0487 | 0.0237 | 0.0275 | 0.0294 | 0.0337 |
| noise_1.0 | 0.0206 | 0.0261 | 0.0307 | 0.0422 | 0.0206 | 0.0239 | 0.0257 | 0.0294 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0255 | [−0.0302, −0.0210] | −0.0232 | [−0.0268, −0.0196] | yes |
| **mean** | **−0.0151** | **[−0.0198, −0.0104]** | **−0.0119** | **[−0.0153, −0.0084]** | **yes** |
| **shuffle_user** | **−0.0164** | **[−0.0210, −0.0117]** | **−0.0124** | **[−0.0157, −0.0092]** | **yes** |
| **shuffle_target** | **−0.0164** | **[−0.0211, −0.0120]** | **−0.0124** | **[−0.0158, −0.0092]** | **yes** |
| noise_0.25 | 0.0015 | [−0.0010, 0.0041] | −0.0001 | [−0.0016, 0.0013] | no |
| noise_0.5 | −0.0034 | [−0.0075, 0.0005] | −0.0042 | [−0.0070, −0.0014] | partial (NDCG only) |
| **noise_1.0** | **−0.0099** | **[−0.0140, −0.0057]** | **−0.0086** | **[−0.0116, −0.0054]** | **yes** |

This is the first checkpoint anywhere in this investigation showing a **noise dose-response**: harm grows monotonically from `noise_0.25` (not significant) through `noise_0.5` (borderline) to `noise_1.0` (clearly significant on both metrics), rather than staying uniformly negligible. Combined with `mask`/`mean`/`shuffle_user`/`shuffle_target` all being significant, this is the broadest, most internally-consistent evidence of genuine content-dependence found so far — `shuffle_user`/`shuffle_target`'s effect size (−31–33% relative) closely matches `mean`'s (−29–31%), which specifically rules out "mean is just an unfamiliar off-template scale" as an alternative explanation (a real, in-distribution substitution hurts just as much as an artificial average).

The cost: raw performance is roughly **halved** relative to every healthier checkpoint (HR@10 0.0521 vs. ~0.095–0.115 elsewhere), consistent with the near-1-dimensional, exploded geometry above. V0 appears to have learned a representation the model is genuinely, strongly dependent on — but a severely damaged one that generalizes far worse, not a usable fix. Across every approach tried so far, a trade-off curve is now visible: margin=0.2 gets modest-but-real causal dependence while fully preserving task performance; V0 gets strong, thorough causal dependence at roughly half the task performance; margin=0.5, V1, and V2 get neither (failing via two distinct mechanisms — hinge non-saturation for margin=0.5, insufficient time-at-hard-temperature for the curriculum runs). A natural next step is sweeping the fixed temperature between V0's harsh 0.1 and softer values to look for a better point on this curve.

![Contrastive V0 checkpoint-1152 E1 ranking by intervention](../figures/latent_contrastive_v0_e1_ranking_grouped_bars.png)

**V1 (InfoNCE + curriculum, no geometry) — evaluated on the wrong checkpoint by necessity, reproduces V2's null result:**

Only `checkpoint-768` (epoch 2, pre-separation) survived training; the epoch-3 checkpoint where `contrastive_loss` actually dropped to 0.432 was deleted by `save_total_limit=1` before it could be evaluated (`load_best_model_at_end` selects by `eval_loss`, which regressed at epoch 3). Evaluating the available checkpoint reproduces the same null pattern as V2, exactly as expected from its pre-separation training state:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0539 | 0.0770 | 0.0898 | 0.1077 | 0.0539 | 0.0673 | 0.0726 | 0.0784 |
| mask | 0.0255 | 0.0489 | 0.0637 | 0.0859 | 0.0255 | 0.0390 | 0.0450 | 0.0522 |
| mean | 0.0539 | 0.0776 | 0.0911 | 0.1077 | 0.0539 | 0.0677 | 0.0732 | 0.0786 |
| shuffle_user | 0.0539 | 0.0771 | 0.0906 | 0.1080 | 0.0539 | 0.0674 | 0.0730 | 0.0786 |
| shuffle_target | 0.0539 | 0.0771 | 0.0906 | 0.1080 | 0.0539 | 0.0674 | 0.0730 | 0.0786 |
| noise_0.25 | 0.0541 | 0.0773 | 0.0898 | 0.1078 | 0.0541 | 0.0676 | 0.0727 | 0.0785 |
| noise_0.5 | 0.0541 | 0.0776 | 0.0898 | 0.1075 | 0.0541 | 0.0678 | 0.0728 | 0.0785 |
| noise_1.0 | 0.0544 | 0.0770 | 0.0901 | 0.1074 | 0.0544 | 0.0675 | 0.0729 | 0.0785 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0218 | [−0.0263, −0.0170] | −0.0262 | [−0.0297, −0.0228] | yes |
| mean | 0.0000 | [−0.0015, 0.0015] | 0.0002 | [−0.0006, 0.0010] | no |
| shuffle_user | 0.0003 | [−0.0011, 0.0018] | 0.0002 | [−0.0005, 0.0010] | no |
| shuffle_target | 0.0003 | [−0.0011, 0.0018] | 0.0002 | [−0.0005, 0.0010] | no |
| noise_0.25 | 0.0002 | [−0.0010, 0.0013] | 0.0002 | [−0.0004, 0.0008] | no |
| noise_0.5 | −0.0002 | [−0.0016, 0.0011] | 0.0001 | [−0.0006, 0.0010] | no |
| noise_1.0 | −0.0003 | [−0.0016, 0.0010] | 0.0001 | [−0.0007, 0.0008] | no |

![Contrastive V1 checkpoint-768 E1 ranking by intervention](../figures/latent_contrastive_v1_e1_ranking_grouped_bars.png)

Only `mask` is significant; every content-preserving mode's CI includes zero, essentially identical to V2's pattern. This result doesn't bear on whether the curriculum approach itself works — it bears on the checkpoint-selection bug that discarded the one checkpoint that could have tested it. Re-running V1 with `load_best_model_at_end=False` (mirroring how V0 naturally kept its final epoch) would be needed to properly evaluate the curriculum's own separated end-state, analogous to V0's.

<a id="attempt-4"></a>
## 4. Attempt 4: CASE-inspired causal alignment losses

### 4.1 Approach

Adapted from CASE (arXiv:2607.18820), "Causal Alignment and Structural Enforcement" for faithful chain-of-thought in text LLMs. Their setup maps onto ours directly: instruction `Z` ↔ our history `x`, CoT `X` ↔ our thought `r`, answer `Y` ↔ our target `y`, and they face the identical architectural problem — the decoder can see `Z` directly, bypassing `X`. Their training objective is a sum of four losses:

```
L_train = L_main + L_X-int + L_Z-int + L_Z-drop
L_main  = E[-log p(x|z) - log p(y|z,x)]                (Eq. 5, standard joint)
L_X-int = E[-log p(y_i'|z_i,x_i')]                      (Eq. 6, counterfactual-CoT)
L_Z-int = E[-log p(y_i|z_i',x_i)]                       (Eq. 7, biased-instruction)
L_Z-drop= E[-log p(y_i|∅,x_i)]                          (Eq. 8, empty-instruction)
```

`L_X-int` pairs the *same* instruction with a genuinely valid *different* (CoT, answer) pair generated by a stronger external LLM — we have no such oracle for recommendation (nothing can generate a valid alternate "this history really does lead to item Y′" pair for us), so **this term is dropped for this attempt**, per instruction. The other two translate directly:

- **`L_Zdrop`** (our name for the `Z-drop` adaptation): predict `y` from `r` alone, with history structurally hidden from the target positions. This reuses the exact `reasoning_only` attention-bias mechanism validated in [Attempt 1](#attempt-1) (`scheduled_attention.py`), but as a **permanent auxiliary loss added every step**, not an annealed curriculum phase — directly testing whether that distinction matters, since [Attempt 1](#attempt-1)'s failure was specifically attributed to the curriculum annealing back to unrestricted access.
- **`L_Zint`** (our name for the `Z-int` adaptation): CASE poisons the instruction with a fixed adversarial claim ("An MIT professor believes the answer is {wrong}") while keeping the correct (CoT, answer) fixed, training the model to resist the injected bias. We have no natural-language slot to inject a textual claim into mid-training, so this is approximated at the embedding level: another example's thought vector (detached, borrowed via the same roll-and-splice trick used in the margin/contrastive attempts) is spliced into a history-region position — specifically the token immediately preceding the thought token, which is always real content, never padding — while the *correct* `r` stays in its normal slot and the true labels are unchanged. This tests whether the decoder can be misled by a plausible-looking but wrong "pooled summary" signal sitting in the history stream when the correct `r` is still available where it's supposed to be.

```
L_train = L_main + zdrop_weight · L_Zdrop + zint_weight · L_Zint     (zdrop_weight = zint_weight = 1.0, unweighted like CASE)
```

No margin, no temperature anywhere — every term is plain NLL, matching CASE's design. `L_main` and `L_Zint` use unrestricted ("both") attention (matching the deployed configuration); only `L_Zdrop` uses the restricted mask.

Implemented additively as `CausalAlignmentLatentModel` (`src/latent/causal_alignment_attention.py`) and `src/latent/latent_attention_train_causal.py`, including the same `grad_accum_steps` fix established in [Attempt 2](#attempt-2). Verified on a toy model: the `L_Zdrop` attention mask actually blocks history from reaching target logits (perturbing a history-region embedding changes `L_main`'s target logits but leaves `L_Zdrop`'s completely unchanged, exactly as intended) and the poison-source detach is structurally correct (confirmed in isolation; a same-tensor gradient-conflation artifact in the first attempt at this check turned out to be a flawed test, not a bug — the position used as a poison source is *also* independently used for its own row's prediction, so its total gradient is legitimately nonzero regardless of the detach).

Full `CDs_and_Vinyl`, 2×GPU DDP (safe here since the only per-example randomness, the poison-source roll, is deterministic and computed independently per rank from its own local micro-batch, unlike [Attempt 1](#attempt-1)'s stochastic mode draw). Two infrastructure failures before a clean run, neither a bug in this code: a first attempt (job 257573) hit `RuntimeError: DDP expects same model across all ranks, but Rank 1 has 344 params, while rank 0 has inconsistent 0 params` after a 30-minute NCCL collective timeout on node `album`, which had independently caused unrelated jobs to crash instantly earlier the same day — a node-level infrastructure issue, not a code defect; resubmitted excluding that node as job 257622, which completed cleanly in 3h02m.

Training was stable throughout: `zdrop_loss` consistently ran higher than `main_loss`/`zint_loss` (the bottleneck-only task is genuinely harder, as intended), but **`zint_loss` tracked `main_loss` almost identically for the entire run** (e.g. epoch-3 batches: 0.424 vs. 0.426) — an early warning, confirmed by the eventual result, that the single-position embedding-level poison wasn't posing much of a training challenge. `load_best_model_at_end=True` kept `checkpoint-768` (epoch 2, same pattern as most prior attempts).

### 4.2 Experiment results

#### 4.2.1 Geometric result: the best measured in this investigation

| Diagnostic | alpha=0.75 scheduled (1152, prior best) | **causal-alignment (768)** |
|---|---:|---:|
| Unique-vector fraction | 95.55% | 95.55% |
| Top-10 template coverage | 0.57% | 0.57% |
| Entropy effective rank | 57.19 | **79.68** |
| PC1 share of variance | 21.21% | **15.41%** |
| Random-pair cosine | 0.4790 | **0.4393** |

#### 4.2.2 Causal-intervention result: `L_Zdrop` induced real activation-dependence; `L_Zint` did not induce content-dependence

E1 ranking intervention on the held-out test split (n=6,157; job 257862). Raw performance:

| Metric | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0547 | 0.0747 | 0.0864 | 0.1044 | 0.0547 | 0.0663 | 0.0712 | 0.0770 |
| mask | 0.0265 | 0.0507 | 0.0630 | 0.0797 | 0.0265 | 0.0405 | 0.0456 | 0.0510 |
| mean | 0.0546 | 0.0742 | 0.0861 | 0.1043 | 0.0546 | 0.0660 | 0.0709 | 0.0768 |
| shuffle_user | 0.0539 | 0.0744 | 0.0867 | 0.1041 | 0.0539 | 0.0658 | 0.0709 | 0.0765 |
| shuffle_target | 0.0539 | 0.0744 | 0.0867 | 0.1041 | 0.0539 | 0.0658 | 0.0709 | 0.0765 |
| noise_0.25 | 0.0546 | 0.0749 | 0.0866 | 0.1035 | 0.0546 | 0.0664 | 0.0712 | 0.0767 |
| noise_0.5 | 0.0542 | 0.0754 | 0.0864 | 0.1043 | 0.0542 | 0.0665 | 0.0711 | 0.0769 |
| noise_1.0 | 0.0544 | 0.0745 | 0.0867 | 0.1044 | 0.0544 | 0.0662 | 0.0712 | 0.0769 |

Raw ranking (HR@10=0.1044, NDCG@10=0.0770) is strong, close to `v4`'s best vanilla baseline.

Paired bootstrap 95% CIs vs. `raw`:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0247 | [−0.0299, −0.0198] | −0.0260 | [−0.0297, −0.0224] | yes |
| mean | −0.0002 | [−0.0018, 0.0015] | −0.0002 | [−0.0010, 0.0006] | no |
| shuffle_user | −0.0003 | [−0.0019, 0.0013] | −0.0005 | [−0.0013, 0.0003] | no |
| shuffle_target | −0.0003 | [−0.0019, 0.0013] | −0.0005 | [−0.0014, 0.0003] | no |
| noise_0.25 | −0.0010 | [−0.0024, 0.0005] | −0.0003 | [−0.0010, 0.0004] | no |
| noise_0.5 | −0.0002 | [−0.0015, 0.0011] | −0.0001 | [−0.0007, 0.0005] | no |
| noise_1.0 | 0.0000 | [−0.0015, 0.0016] | −0.0001 | [−0.0010, 0.0007] | no |

![Causal-alignment checkpoint-768 E1 ranking by intervention](../figures/latent_causal_e1_ranking_grouped_bars.png)

`mask` is large and clearly significant (−23.6%/−33.8% relative) — `L_Zdrop`'s permanent bottleneck loss produced real, substantial activation-dependence, the strongest such penalty outside `v4`'s own baseline. But every content-preserving mode is null: `mean`/`shuffle_user`/`shuffle_target` all sit within noise of `raw`, reproducing the classic activation-without-content collapse signature despite the best geometry measured anywhere in this investigation. Consistent with the training-time warning sign (`zint_loss ≈ main_loss` throughout), `L_Zint`'s embedding-level poison — another example's thought vector spliced into a single history-region position — evidently wasn't a strong enough adversarial signal: the model most likely learned to structurally discount that one anomalous position as an outlier, without ever needing genuine cross-referencing against `r`'s specific content to do so. This is a different failure mode from CASE's own text-level adversarial injection (a fabricated claim woven into fluent instruction text, much harder to detect structurally than a single out-of-place embedding), and suggests that if this direction is revisited, the poison needs to look less like an anomaly and more like plausible content — e.g. corrupting multiple positions, or drawing the poison from a *semantically similar* rather than a uniformly random other example.

Net result: `L_Zdrop` (the permanent bottleneck loss) replicates activation-dependence on its own, comparable to what the schedule and margin-loss approaches also achieve; `L_Zint` as implemented did not deliver the content-dependence CASE's own ablations attribute mostly to their inference-time structural enforcement and `L_X-int` (the oracle-dependent term dropped for this attempt) rather than to `L_Z-int` alone — consistent with our result here.

<a id="attempt-5"></a>
## 5. Attempt 5: gradient masking on the history-to-target attention path

### 5.1 Approach

Adapted from "Faithfulness as Information Flow" (arXiv:2605.24286), Eq. 8. Every approach so far that achieved genuine content-dependence did so by also reducing what the decoder can *compute*: the attention schedule and CASE's `L_Zdrop` both structurally hide history from the decoder some or all of the time (real forward-pass information loss), and [Attempt 3](#attempt-3)'s V0 only achieved strong content-dependence by collapsing `r` into a damaged, near-1-dimensional code that roughly halved task performance. Gradient masking targets the actual trade-off directly: the *forward* pass is completely unchanged (decoder sees history and `r` together, full capacity, bit-identical to vanilla R3) — only the *backward* pass is altered, so the optimizer gets no gradient credit for routing information through the direct history→target attention edges specifically. Every other edge (target→`r`, target→target, and everything in the first pass that computes `r`) is completely unaffected.

Formally (their Eq. 8), for pre-softmax attention score `z_ij` (query `i`, key `j`):

```
z~_ij = sg(G_ij · z_ij) + (1 - G_ij) · z_ij,   G_ij = 1[i is a target position AND j is a history position]
```

where `sg` is stop-gradient. Since `sg(x)` has the same *value* as `x` but zero gradient, `z~_ij == z_ij` numerically everywhere (forward pass identical to vanilla), while `G_ij=1` entries contribute no gradient to the Q/K projections that produced them.

This cannot be expressed as a bias added to the standard `attention_mask` argument (a bias affects forward and backward identically; here they must differ) — it requires intercepting the raw attention-score computation inside every transformer layer. Implemented in `src/latent/gradient_mask_attention.py` by monkey-patching the module-level `eager_attention_forward` function that `Qwen2Attention.forward()` looks up by name at every call (requires `attn_implementation="eager"`); the patched function is a strict no-op wrapper around the original when a thread-local flag is inactive (default), so importing this module is harmless everywhere else, and only alters behavior inside an explicit `with` scope in `GradientMaskLatentModel.forward()`'s training branch. Because this only changes gradient flow — not the loss computation or the inputs — the standard `Qwen2ForCausalLM.forward(..., labels=...)` built-in loss path is used completely unchanged: no custom NLL, no extra forward passes, and critically no `grad_accum_steps` bug (unlike the margin/contrastive/causal attempts), since Trainer's own normalization applies exactly as it does for vanilla `LatentModel`.

Verified with two levels of rigor before trusting any training result:
- **Isolated unit test of the core mechanism** (synthetic Q/K/V tensors, no model): restricting target queries to attend *only* to history keys and checking their gradient is exactly `0.0` confirms the blocking is complete with no leakage, while history-query gradients (never subject to blocking) and the value-vector gradient (never blocked, by design) remain nonzero everywhere, confirming the "gentler than full masking" property is real, not just in principle.
- **Full-model test**: `GradientMaskLatentModel`'s forward logits are bit-identical (`max abs diff = 0.0`) to vanilla `LatentModel` given the same weights, and gradient into the `r`-generating layer is nonzero, confirming target predictions still learn via the unblocked target→`r` path.

Submitted as job 258026 (full `CDs_and_Vinyl`, 2×GPU DDP — safe here since the masking is purely deterministic, a function of `thought_idx` only, with no randomness anywhere unlike the attention schedule).

### 5.2 Experiment results

#### 5.2.1 Result: the most severe geometric collapse measured in this investigation

Training completed cleanly (job 258026, 1h37m) with a notably healthy-looking loss trajectory — eval loss *improved monotonically* across all three epochs (2.456 → 1.484 → 1.213), the only run besides the alpha=0.75 schedule to do so; every other attempt regressed at epoch 3. `load_best_model_at_end=True` kept the final epoch (`checkpoint-1152`), consistent with monotonic improvement. Raw `grad_norm` was highly elevated early in training (1700+ at the start, settling to the 50–200 range by epoch 3) — in hindsight a symptom of the underlying problem below, not just early-training noise.

Geometry (`analysis/results/latent_distribution/latent_gradmask_checkpoint_1152/`) tells an unambiguous story on its own:

| Diagnostic | V0 (previous worst, [Attempt 3](#attempt-3)) | **Gradient Mask (1152)** |
|---|---:|---:|
| Unique-vector fraction | 5.42% | **0.13%** |
| Top-10 template coverage | 31.32% | **87.53%** |
| Entropy effective rank | 1.37 | **1.17** |
| PC1 share of variance | 92.44% | **97.10%** |
| Random-pair cosine | 0.9827 | **0.9998** |
| Components for 90% variance | 1 | **1** |

Random-pair cosine of 0.9998 means essentially every one of the 49,251 training vectors points in virtually the identical direction — the most complete representational collapse measured anywhere in this investigation, worse even than [Attempt 3](#attempt-3)'s V0.

**Likely mechanism, and a real limitation of the design rather than a bug**: the gradient mask (Eq. 8) only cuts gradient through the attention-*score* component of blocked (target, history) pairs; the *value*-vector pathway is deliberately left open (that asymmetry is what makes it "gentler" than full masking, per the source paper's own design). But the value/output projection weights are shared across every position in the network, not specific to the blocked edges — so the model apparently found an alternate route to keep exploiting history's content through that unblocked pathway, without the attention weights ever needing credit for it. That would explain both the collapse and the unusually clean training trajectory: the model wasn't struggling to route around the block, it simply stopped needing `r` almost entirely, letting it collapse to a near-constant instead of being forced to make it meaningful.

The E1 causal-intervention job (258128) was cancelled after the `raw` mode, before completing the remaining seven modes — the geometric result alone was judged decisive enough (given the >0.999 cosine and effective-rank-1.17 numbers) that the remaining ~3 hours of compute for `mask`/`mean`/`shuffle_*`/`noise_*` confirmation wasn't worth spending. Predicted (not measured) outcome based on the geometry: `mask` would likely still show a real effect (zero is a genuinely different value from the collapsed near-constant), but `mean`/`shuffle_*` would likely show ~0 (swapping one already-near-identical vector for another changes almost nothing) — the same activation-without-content signature as every failed attempt, in its most extreme form.

**Verdict: gradient masking as implemented does not fix collapse**, and plausibly makes it worse than doing nothing, by leaving an unblocked value-pathway shortcut available while removing the (weaker) score-pathway signal that at least other approaches partially relied on. A more complete version would need to also block gradient through the value pathway for the same blocked positions — a more invasive change than the source paper's design, and not obviously better than the two approaches that already worked ([Attempt 2](#attempt-2)'s margin loss at margin=0.2, and [Attempt 1 Follow-up A](#attempt-1-alpha075)'s alpha=0.75 schedule).

<a id="reframing-mu-z"></a>
## Reframing: $r_i = \mu + z_i$ — a prior that overwhelms the reasoning

Stepping back from the five attempts, one decomposition explains every collapse signature at once. Write each generated latent as

    r_i = μ + z_i

where μ is a shared component encoding general recommendation-task knowledge (a *prior*: "what tends to come next in this domain"), and z_i is the sample-specific reasoning for user/history i. Under plain CE training the optimizer chooses the μ:z ratio, and it chooses ≈0 — μ alone already explains most of the loss reduction, so nothing rewards z_i for varying. This is the posterior-collapse mechanism of latent-variable models (Bowman et al. 2015), pointed at the *internal* split of r rather than the external x-vs-r shortcut.

It accounts for the full observed pattern with a single mechanism rather than three coincidences:
- random-pair cosine ≈ 1: r_i ≈ μ for every i;
- `mask` hurts: zeroing r removes μ, the one load-bearing component;
- `mean` / `shuffle` / `noise` do nothing: each replaces one near-copy of μ with another (μ + z_j ≈ μ + z_i when ‖z‖ ≪ ‖μ‖);
- ranking metrics keep improving with more training: μ is itself a learned, history-conditioned quantity, and a better prior alone raises HR/NDCG — the loss curve says the objective is being optimized, not *which* component is doing it.

Note this also reframes margin=0.2 ([Attempt 2](#attempt-2)): it is not considered a success under this lens — the `noise_*` deltas were not significant and random-pair cosine stayed high, i.e. z_i stayed small.

### Probe: `z_alone` vs `noise_matched` (job 258369 → 258393)

Three of the four natural conditions are already in the E1 suite: μ+z_i = `raw`, μ = `mean`, μ+z_j = `shuffle`. The untested one is **z_i alone** (r_i minus the train mean). Two new `generate` modes were added to `src/prototype_grounding/experiment.py`:
- `z_alone`: `(q − μ̂) / ‖q − μ̂‖ · mean‖r‖` — the μ-free residual, re-normed to the typical raw-r norm so a magnitude mismatch is not misread as a content effect.
- `noise_matched`: a random direction at the same fixed norm — the null control. Removing μ removes the "task scaffold" the decoder was trained against, so `z_alone` is expected to be bad *regardless* of checkpoint; the informative quantity is the **gap `z_alone` − `noise_matched`**, which isolates whether z_i's direction carries decodable content while the missing-scaffold penalty hits both equally.

Run on the three checkpoints still on disk (margin=0.2 and the v4 baseline have been deleted from `output_dir/`): alpha=0.75 (working), causal L_Zdrop (activation- without content-dependence), gradmask (worst collapse).

| checkpoint | raw | mean | z_alone | noise_matched | z_alone − noise_matched |
|---|---|---|---|---|---|
| alpha=0.75 | HR@10 0.1038 / NDCG@10 0.0712 | 0.0968 / 0.0647 | 0.0960 / 0.0609 | 0.0812 / 0.0487 | **+0.0148 / +0.0122** |
| causal (Zdrop) | 0.1044 / 0.0770 | 0.1043 / 0.0768 | 0.0997 / 0.0684 | 0.1013 / 0.0735 | **−0.0016 / −0.0051** |
| gradmask | missing (job 258128 cancelled before `raw`) | missing | 0.0195 / 0.0143 | 0.0156 / 0.0109 | +0.0039 / +0.0034 (both near-floor) |

Independent confirmation of the μ+z hypothesis, without relying on any of the machinery from [Attempt 6](#attempt-6) or [Attempt 7](#attempt-7): on the one checkpoint with previously-established genuine content-dependence (alpha=0.75), the μ-free residual's *direction* clearly beats a norm-matched *random* direction — the gap survives the "missing scaffold" confound this experiment was designed to control for, since both conditions lack μ equally. On the checkpoint with previously-established activation-dependence *without* content-dependence (causal L_Zdrop), `z_alone` is statistically indistinguishable from (if anything slightly worse than) `noise_matched` — precisely what "no real content in the residual" predicts. On gradmask, both are near the performance floor, consistent with total collapse: removing μ leaves essentially nothing to distinguish, real or random. Three checkpoints, three outcomes, all landing exactly where the framework predicts they should.

<a id="attempt-6"></a>
## 6. Attempt 6: explicit $\mu/z$ decomposition with structural isolation

### 6.1 Approach

#### 6.1.1 Why not simpler variants

Two obvious designs were considered and rejected before this one:
- **Shared learnable μ + margin loss.** With μ identical across rows, margin's pairwise comparison sees `r_i − r_j = z_i − z_j`; μ cancels, so the gradient into z_i is *identical* to [Attempt 2](#attempt-2)'s. Cosmetically cleaner, same optimization — no reason to expect a different outcome.
- **Warm-up, then freeze μ.** Nothing stops the attention module from regrowing a constant offset inside z_i; the generator re-learns μ′ within z_i and the remaining steps are R3 with an unused parameter (R3 is the special case μ = 0).

The deeper point: the collapsing force is the CE loss itself. Any design where z_i is shaped only by CE will collapse z_i whatever is done to μ. The isolation therefore has to be structural on z_i.

#### 6.1.2 Design (`src/latent/decomposed_attention.py`, `decomposed_contrastive_attention.py`, `latent_attention_train_decomposed.py`)

1. **Centering — z_i cannot carry the shared component.** `z_i = f(h_i) − EMA[f(h)]` (momentum 0.99, stop-gradient). Any sample-independent content has exactly one place to live: the learnable vector μ. This is what makes "μ = prior, z_i = per-sample" a fact of the architecture rather than an interpretation of a trained model.
2. **Fixed scale — z_i cannot shrink relative to μ.** `r_i = μ + ρ·‖μ‖·ẑ_i` with ẑ_i unit-normed. The μ:z ratio ρ is a hyperparameter, not something the optimizer can drive to 0. With ρ≈1 and spread z_i the pairwise cosine of r cannot approach 1, and perturbing z_i must move r_i by a non-trivial amount. (‖μ‖ is detached in the scale; μ is initialised from the `<|Thought|>` embedding direction at norm 1.0 and free to grow via CE.)
3. **Task-coupled pressure — the decoder must actually read z_i.** 1+2 only guarantee z_i is large and distinct in geometry; the decoder could still treat it as noise (V2's failure). So: roll-and-splice InfoNCE from [Attempt 3](#attempt-3) (decoder must score μ+z_i above μ+z_j for all other j in the micro-batch; μ cancels, so the pressure lands on z only), plus VICReg variance+covariance on ẑ·√H to forbid the 1-D sample-identity code V0 found. (The covariance term alone would not prevent that — a single active dimension has zero covariance with the rest; the per-dim variance floor is what does.)

Why this is not a re-run of [Attempt 3](#attempt-3): V0 pushed all of **r** apart, and since r *was* the prior, distinctness destroyed the recommendation knowledge → performance halved. Here μ receives only the CE gradient and is untouched by the contrastive term, so distinctness is demanded only of z_i. The prior is protected; the reasoning part is forced to matter.

The alpha=0.75 attention schedule stays on to close the other shortcut (decoder reading raw history instead of r). Contrastive/geometry terms are skipped on `history_only` batches, where r is discarded at the thought position. Single-GPU only (per-batch mode draw; EMA buffer not DDP-synchronised).

The decomposition lives *inside* the attention module and is selected by a `latent_decomposition` flag saved in the checkpoint's `config.json` (branch added to `LatentModel.__init__` in `src/latent/layers.py` and `src/grpo_attention_tuning/model.py`), so the standard ranking evaluator and the E1 pipeline — both of which call `model.attention(...)` directly — see the full r without modification. Verified with a CPU toy model: `‖r−μ‖ = ρ‖μ‖` exactly; EMA centering converges; all three schedule modes give finite loss with `history_only` zeroing the z-terms; save → reload through both vanilla loaders reconstructs the decomposed layer with `max|Δr| = 2e-7`.

#### 6.1.3 Runs

ρ is doing real work — too small and we are back to "noise doesn't matter", too large and the decoder is overwhelmed as in V0 — so it is swept rather than guessed. Temperature fixed at 0.1 (V0's setting, the only one that produced dependence; the curriculum never separated), the cleanest test of the hypothesis that V0 failed *because* the contrastive force hit the prior.

| job | ρ | output |
|---|---|---|
| 258391 | 1.0 | `output_dir/CDs_and_Vinyl_R3_decomposed_rho1.0/latent` |
| 258392 | 0.5 | `output_dir/CDs_and_Vinyl_R3_decomposed_rho0.5/latent` |

Success bar (the user's criteria, not the earlier ones): random-pair cosine well below the current ~0.98, `noise_*`/`shuffle` deltas significant, raw HR/NDCG near baseline. Results pending.

### 6.2 Experiment results

#### 6.2.1 Result: task performance collapses for both ρ, and r becomes actively harmful rather than merely useless

E1 ranking intervention on the held-out test split (n=6,157; jobs 258443/258444). Full per-mode breakdown:

**ρ=1.0**

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0148 | 0.0214 | 0.0242 | 0.0338 | 0.0148 | 0.0185 | 0.0196 | 0.0227 |
| mask | 0.0172 | 0.0242 | 0.0292 | 0.0404 | 0.0172 | 0.0212 | 0.0233 | 0.0268 |
| mean | 0.0159 | 0.0203 | 0.0227 | 0.0300 | 0.0159 | 0.0184 | 0.0194 | 0.0218 |
| shuffle_user/target | 0.0114 | 0.0174 | 0.0213 | 0.0294 | 0.0114 | 0.0149 | 0.0165 | 0.0191 |
| noise_0.25 | 0.0154 | 0.0216 | 0.0242 | 0.0330 | 0.0154 | 0.0189 | 0.0199 | 0.0228 |
| noise_0.5 | 0.0151 | 0.0205 | 0.0240 | 0.0317 | 0.0151 | 0.0182 | 0.0197 | 0.0221 |
| noise_1.0 | 0.0141 | 0.0206 | 0.0248 | 0.0322 | 0.0141 | 0.0179 | 0.0196 | 0.0219 |

**ρ=0.5**

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0219 | 0.0351 | 0.0426 | 0.0577 | 0.0219 | 0.0296 | 0.0326 | 0.0375 |
| mask | 0.0227 | 0.0344 | 0.0429 | 0.0577 | 0.0227 | 0.0296 | 0.0330 | 0.0378 |
| mean | 0.0409 | 0.0577 | 0.0666 | 0.0835 | 0.0409 | 0.0507 | 0.0544 | 0.0598 |
| shuffle_user/target | 0.0252 | 0.0393 | 0.0474 | 0.0645 | 0.0252 | 0.0333 | 0.0366 | 0.0422 |
| noise_0.25 | 0.0236 | 0.0365 | 0.0453 | 0.0583 | 0.0236 | 0.0310 | 0.0346 | 0.0388 |
| noise_0.5 | 0.0231 | 0.0375 | 0.0453 | 0.0596 | 0.0231 | 0.0314 | 0.0346 | 0.0392 |
| noise_1.0 | 0.0244 | 0.0356 | 0.0435 | 0.0585 | 0.0244 | 0.0309 | 0.0342 | 0.0390 |

Geometry:

| | ρ=1.0 | ρ=0.5 |
|---|---|---|
| effective rank | 81.9 | 3.9 |
| top eigenvalue fraction (PC1) | 0.211 | 0.675 |
| components for 90% variance | 156 | 3 |
| random-pair cosine to mean | 0.731 | 0.925 |

Both collapse task performance far below baseline (~0.10 HR@10 / ~0.07 NDCG@10 in every prior attempt) — ρ=1.0 to roughly a third, ρ=0.5 to roughly half. Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42), both metrics:

**ρ=1.0**

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0067 | [−0.0104, −0.0032] | −0.0041 | [−0.0064, −0.0019] | yes (both) |
| mean | +0.0037 | [+0.0006, +0.0068] | +0.0009 | [−0.0007, +0.0026] | HR@10 only |
| shuffle_user/target | +0.0044 | [+0.0006, +0.0080] | +0.0036 | [+0.0013, +0.0061] | yes (both) |
| noise_0.25/0.5/1.0 | all n.s. | | all n.s. | | no |

**ρ=0.5**

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | +0.0000 | [−0.0054, +0.0057] | −0.0003 | [−0.0041, +0.0034] | no |
| mean | −0.0258 | [−0.0304, −0.0213] | −0.0223 | [−0.0259, −0.0189] | yes (both) |
| shuffle_user/target | −0.0068 | [−0.0115, −0.0023] | −0.0047 | [−0.0082, −0.0012] | yes (both) |
| noise_0.25/0.5/1.0 | all n.s. | | all n.s. | | no |

(Delta sign convention: raw − variant, so a positive delta at ρ=1.0 means the variant *outperforms* raw — the "harmful r" signature — while a negative delta at ρ=0.5 means raw *outperforms* the variant, the more familiar direction but for the wrong reason, as the verdict below explains.)

Two distinct, and equally bad, failure signatures:

- **ρ=1.0: harmful but content-sensitive.** `mask` significantly *beats* raw on both metrics (removing r entirely is better than using it — r is net-harmful), yet `mean`/`shuffle` significantly *underperform* raw (using your *own* r beats a generic replacement) — `shuffle` on both metrics, `mean` on HR@10 only (its NDCG@10 CI includes zero, so that particular comparison is directionally consistent but not itself conclusive). Both true at once means the model did learn something genuinely sample-specific — content-dependence is real — but whatever it learned actively hurts more than it helps, worse than having no reasoning signal at all. The healthy-looking geometry (effective rank 82, PC1 21%) is not fake, exactly — it's just organized around a signal that damages the prediction.
- **ρ=0.5: your own content is the worst option available.** `mask` ≈ raw (removing r changes nothing), but `mean` and `shuffle` both significantly *beat* raw — replacing your personalized r with the population average, or with a random stranger's r, produces better recommendations than your own. This is the most severe form of "actively adversarial" content: not merely uninformative, but specifically counterproductive for the example it was computed for.

Both patterns point to the same mechanism: forcing a fixed μ:z magnitude ratio removes the graceful "mostly ignore z" solution that plain CE training would otherwise find, and combined with the temperature=0.1 contrastive pressure (V0's setting, chosen deliberately to test whether V0's failure was about hitting the *prior* rather than the *pressure itself*) the optimizer is pushed into a regime that actively encodes something harmful rather than either helpful or inert content. Recall this pressure only ever tests "is *a* z distinguishable", never "is *my own* z good for me" — nothing in [Attempt 6](#attempt-6)'s objective forbids a solution where distinctiveness and helpfulness point in opposite directions, and that is exactly what happened.

**Verdict: [Attempt 6](#attempt-6) fails outright, for both ρ.** Neither the hard-normalization mechanism nor the ρ value is worth tuning further in isolation; the fixed-scale approach itself removes too much of the optimizer's room to find a benign trade-off. Between the three mu/z mechanisms tried so far (6: hard normalization, [7a](#attempt-7a): pure KL, [7b](#attempt-7b): KL + μ-consistency), only [7b](#attempt-7b) — the one with the softest structural constraint and an added identifiability anchor rather than a hard-forced ratio — currently shows the theoretically "correct" ordering (raw > mask > mean ≈ shuffle, all pointing the right direction). See below for [7b](#attempt-7b)'s full result once its E1 run completes.

<a id="attempt-7"></a>
## 7. Attempt 7: VAE-shaped encoder with an aggregate KL prior on $z$

### 7.1 Approach

A second reading of the μ+z decomposition, proposed independently of [Attempt 6](#attempt-6): instead of a hard-normalized deterministic residual, treat r_i as a VAE latent — `mu_i, z_i = attention(x_i)` (two heads on the same pooled output, no sampling), `r_i = mu_i + z_i`, and pull the *distribution* of z_i toward a prior N(0, I) with a KL term, rather than hard-normalizing it.

#### 7.1.1 Why this is a different mechanism from [Attempt 6](#attempt-6), not a relabeling

A literal per-example KL is undefined for a deterministic z_i (it's a Dirac delta). The standard fix (Wasserstein Autoencoders / InfoVAE) is to match the *aggregate* distribution of z over a batch/queue to the prior via moment matching:

    KL(N(m, diag(s²)) ‖ N(0, I)) = ½ Σ_d (s_d² + m_d² − 1 − log s_d²)

- `m → 0`: shared content is pushed out of z (soft analogue of [Attempt 6](#attempt-6)'s hard EMA-centering).
- `s² → 1` **with a log barrier**: any dimension shrinking toward 0 incurs unbounded penalty — a strictly stronger anti-collapse guarantee than [Attempt 6](#attempt-6)'s linear VICReg hinge, and it directly forbids the 1-D sample-identity code V0 found. It is also two-sided (penalizes both shrinking and exploding), unlike the hinge.
- With H=1536, unit per-dimension variance concentrates ‖z_i‖ tightly around √1536 ≈ 39 — the prior pins the *magnitude* of z as a side effect of pinning its distribution.

#### 7.1.2 The identifiability problem, and why there are two variants

Nothing in the objective distinguishes `mu_i` from `z_i` — both are outputs of the same encoder, and only the sampling step (deliberately dropped, per this attempt's stated goal) usually gives a VAE's two channels different roles. With `r_i = mu_i + z_i` deterministic, CE is free to route all sample-specific content through the unconstrained `mu_i` channel while z_i's aggregate distribution is shaped into a decorrelated, unit-variance cloud that carries nothing — the prior is satisfiable by informative *or* uninformative z. This produces the collapse signature by construction: geometry that looks improved (z mechanically lowers pairwise cosine) with interventions on z doing nothing.

Also closed: `‖mu_i‖` must be fixed (`rho_mu · √H`, direction still learned), or CE can simply grow `‖mu‖` until the prior-pinned `‖z‖` is negligible by comparison — the collapse re-entering through the mean instead of by shrinking z.

Two variants, run in parallel:
- **[7a](#attempt-7a)**: pure prior only (`mu_consistency_weight=0`) — the direct test of whether the distributional prior alone (with `‖mu‖` fixed) makes z matter.
- **[7b](#attempt-7b)**: [7a](#attempt-7a) + a soft identifiability anchor — `mu_i` penalized toward its own EMA across training (`mu_consistency_weight=0.5`), closing the loophole above without hard-wiring μ to a single global parameter (keeping the "mu, z = encoder(x)" shape the user specified, unlike [Attempt 6](#attempt-6)'s literal `nn.Parameter`).

Expectation going in: [7a](#attempt-7a) is likely to reproduce the V2 pattern (good geometry, no causal effect) precisely because of the identifiability gap above; [7b](#attempt-7b) is the version that plausibly works. Running both makes that a measured result rather than an assumption.

#### 7.1.3 Implementation (`src/latent/decomposed_attention.py::VAEDecomposedSelfAttentionLayer`, `vae_decomposed_attention.py`, `latent_attention_train_vae.py`)

`VAEDecomposedSelfAttentionLayer` adds `mu_head`/`z_head` linear layers on the existing pooled-attention output; `mu` is unit-direction-normalized then rescaled to the fixed norm, `z` is unconstrained. A `build_decomposed_attention(hidden_size, spec)` factory in `decomposed_attention.py` dispatches on a `kind` field in the `latent_decomposition` config entry (`"fixed_scale"` → [Attempt 6](#attempt-6)'s layer, `"vae"` → this one; [Attempt 6](#attempt-6)'s already-running checkpoints omit `kind`, which defaults to `"fixed_scale"`, so 258391/258392 are unaffected), so both attempts' checkpoints load correctly through the same `layers.py`/`grpo model.py` branch and the standard evaluator / E1 pipeline need no changes for either.

`VAEDecomposedLatentModel` (built on `ScheduledLatentModel`, same alpha=0.75 schedule as [Attempt 6](#attempt-6)) computes the aggregate KL over `[current micro-batch ∪ a queue of up to 256 recent detached z]` rather than the micro-batch alone — a micro-batch of 4 gives a 3-degree-of-freedom variance estimate, far too noisy for a term with a log barrier; the queue gives a stable estimate while gradient still flows through the current rows. KL / consistency terms are skipped on `history_only` batches, where r is discarded and z has no path to the loss.

Verified with a CPU toy model: `‖mu_i‖` exactly equals the fixed target for every row; `r = mu + z` exactly; all three schedule modes give finite loss with `history_only` correctly zeroing the KL/consistency terms; isolating the KL term alone (no competing CE) and descending only `z_head`'s parameters decreases it monotonically over 50 steps, confirming the gradient direction is correct. One real finding surfaced by the toy test, not a bug: at `kl_weight=1.0`, CE's gradient into `z_head` (~0.32 in the toy model) considerably exceeds the isolated KL gradient's effect over a handful of steps — the identifiability tension above showing up quantitatively, not just in principle. Whether `kl_weight=1.0` is strong enough over a full 1152-step run (with Adam rather than raw SGD) is exactly what the two runs below will show; `z_mean_sq`/`z_var` are logged every step specifically to monitor this. Save → reload through both vanilla loaders reconstructs the layer with `max|Δr| = 7e-7`.

**Bug (overnight, unattended)**: job 258440 ([7a](#attempt-7a)) completed training successfully (`checkpoint-1152`, `model.safetensors` both valid — loss trajectory ce=0.878/kl=0.075 at end of epoch 3) but the chained standard eval crashed: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`. Cause: `VAEDecomposedSelfAttentionLayer.forward` upcast `pooled` to float32 *before* the `mu_head`/`z_head` linear layers (bf16 weights, since the model loads with `torch_dtype=bfloat16`) — `F.linear` requires matching dtypes, unlike the elementwise ops `DecomposedSelfAttentionLayer` ([Attempt 6](#attempt-6)) uses, which auto-promote and never hit this. It never surfaced during training because `Trainer`'s bf16 autocast smooths over it, and never surfaced in the CPU toy test because that test's toy model was never cast to bf16. Fixed by upcasting each head's *output* to float32 instead of its input (the matmul now runs in native bf16, matching precision for the KL/norm math happens after). Re-verified with a bf16 toy model, exercising the exact crashing call path (`generate_embs` and full `model.generate()` outside any autocast context) — both pass. No retraining needed: the checkpoint was already valid, so only the standard eval was re-run (job 258606), with its E1 pipeline re-queued as 258607.

#### 7.1.4 Runs

| job | variant | mu_consistency_weight | output |
|---|---|---|---|
| 258440 | [7a](#attempt-7a) | 0.0 | `output_dir/CDs_and_Vinyl_R3_vae7a/latent` |
| 258441 → 258557 | [7b](#attempt-7b) | 0.5 | `output_dir/CDs_and_Vinyl_R3_vae7b/latent` |

Same success bar as [Attempt 6](#attempt-6): random-pair cosine well below baseline, `noise_*`/`shuffle` deltas significant, raw HR/NDCG near baseline.

<a id="attempt-7-followup-ab"></a>
#### 7.1.5 Follow-up: remove KL or loosen the curriculum

[7b](#attempt-7b) works causally but sits below the unrestricted baseline: HR@10 0.0966 / NDCG@10 0.0650 vs `v4`'s 0.1152 / 0.0806 (−16% / −19%). Decomposing that gap from results already in hand:

- The alpha=0.75 schedule alone (no μ/z) scores 0.1038 / 0.0712 — a ~10-12% tax by itself, roughly 60% of [7b](#attempt-7b)'s gap.
- [Attempt 8](#attempt-8) shares [7b](#attempt-7b)'s schedule, encoder, `rho_mu` and `mu_consistency=0.5`, differs only in `kl_weight` (1.0 → 0) plus its alignment loss, and lands at 0.1057 / 0.0735 — *above* schedule-alone. So the μ/z machinery is not inherently costly; the prime suspect for [7b](#attempt-7b)'s remaining ~7% is the KL prior. Mechanically plausible: a log-barrier KL to N(0, I) pins unit variance on all 1536 dimensions, i.e. ‖z‖ ≈ 39 ≈ ‖μ‖ of isotropic, prior-shaped, largely uninformative signal the decoder must learn to tolerate.
- Note the μ-consistency anchor is *not* a suspect: [7a](#attempt-7a) (weight 0) → [7b](#attempt-7b) (weight 0.5) raised raw HR@10 from 0.0869 to 0.0966.

Two single-variable runs, each differing from [7b](#attempt-7b) in exactly one setting:

| run | job | change vs [7b](#attempt-7b) | question |
|---|---|---|---|
| A | 259793 (E1: 259795) | `kl_weight` 1.0 → 0.0 | Is the KL prior the cost? Does the μ-anchor alone keep `raw > mask`? |
| B | 259794 (E1: 259796) | `alpha_max` 0.75 → 0.9 | How much of the schedule's tax can be recovered while keeping the x-shortcut defense? |

Outputs `output_dir/CDs_and_Vinyl_R3_vae7b_nokl/` and `..._a09/`; tags `latent_vae7b_nokl_checkpoint_1152`, `latent_vae7b_a09_checkpoint_1152`. Results pending.

<a id="attempt-7-no-curriculum"></a>
#### 7.1.6 Follow-up: disable the attention curriculum

Run A and Run B (above) each perturbed one ingredient of [7b](#attempt-7b) (KL, schedule tightness) and showed both are load-bearing. This run goes further: remove the schedule's contribution entirely rather than loosen it, to isolate how much of [7b](#attempt-7b)'s causal necessity comes from the μ/z mechanism itself versus from the schedule's x-shortcut guard.

**Change vs. [7b](#attempt-7b)**: `ramp_epochs=0, alpha_max=1.0` instead of `ramp_epochs=3, alpha_max=0.75`. Per `AttentionScheduleCallback`, alpha reaches 1.0 by the second training step and the schedule then always draws `"both"` (unrestricted history + latent, simultaneously) for the rest of training — no `reasoning_only`/`history_only` mixing at all, verified directly against the callback (only the single step at `epoch=0.0` briefly draws `history_only`, before any gradient update). KL (1.0) and the μ-consistency anchor (0.5) are unchanged from [7b](#attempt-7b).

**Prediction, given Run B's result**: Run B (alpha=0.9 — still mostly "both") already showed the causal signature nearly vanish (`mask` n.s., `shuffle` n.s., only a tiny `mean` effect). With the curriculum fully disabled from the start, I'd expect this to go further — either a complete return to the "nothing matters" baseline pattern, or possibly the more interesting case: does the μ/z mechanism (KL + anchor) produce *any* independent activation- or content-dependence on its own, with zero schedule assistance? Either answer is informative: a clean null result would show the schedule was doing all the causal-necessity work in [7b](#attempt-7b); any surviving effect would isolate what the μ/z mechanism contributes unaided.

Submitted as job 271274 (`output_dir/CDs_and_Vinyl_R3_vae7b_nocurriculum/`), E1 pipeline queued as 271275, dependent on successful completion. Results pending.

<a id="attempt-7-z-method"></a>
#### 7.1.7 Methodology correction: analyze $z$ rather than only $r=\mu+z$

Every geometry number reported for the μ/z-decomposition attempts ([7a](#attempt-7a), [7b](#attempt-7b), [Attempt 8](#attempt-8), [Runs A/B](#attempt-7-followup-ab), and the [curriculum-disabled run](#attempt-7-no-curriculum)) was computed on the combined thought vector `r_i` spliced into the decoder, not on `z_i` in isolation. Since `‖μ_i‖` is architecturally fixed (`VAEDecomposedSelfAttentionLayer` unit-normalizes then rescales `μ_i` to a constant target norm `rho_mu · √H` on every forward pass), any norm or directional variance measured in `r` is necessarily a property of `z` and/or the angle between `μ` and `z` — but running the diagnostics on the sum conflates "z is genuinely diverse" with "μ's large fixed norm dominates the sum regardless of z." This was exactly the confound the earlier `z_alone` vs. `noise_matched` E1 probe was built to control for, on the causal-intervention side; it was never extended to the geometry pipeline.

Concretely observed: the curriculum-disabled run's `r`-geometry showed a striking bimodal pattern (two norm clusters ~46-47 and ~66-67; two cosine-to-mean clusters ~0.65 and ~0.77). Checking the *identity-group* structure in the same report showed this split does not correlate with any tested identity dimension (`same_user`/`same_target`/`same_cf_cluster` pairwise cosine distributions are statistically indistinguishable from `random` pairs) — i.e. a generic two-point-mass degeneracy, not a meaningful two-cluster split of user preferences. Worth being precise about that distinction going forward: "bimodal" is not the same claim as "collapsed into two meaningful clusters," and the pairwise-similarity-by-identity-group panel is what actually distinguishes them.

**Fix**: `src/prototype_grounding/experiment.py::extract` now also saves `z_only.pt` alongside `thoughts.pt` whenever the checkpoint exposes `model.attention.last_z` (mu/z-decomposed checkpoints only; free to capture, since it's already computed as a side effect of the existing `model.attention(...)` call inside `thought_embeds`, no extra forward pass). `latent_distribution_report` takes a new `--component {r,z}` flag (default `r`, preserving all prior results and file paths unchanged) that runs the identical diagnostic code on `z_only.pt` instead, writing to separate output files (`train_latent_distribution_z.json` / `_z_arrays.npz`) so nothing is overwritten.

Applied retroactively via a lightweight re-extraction (job 276414) to the three mu/z checkpoints still on disk (Run A `vae7b_nokl`, Run B `vae7b_a09`, curriculum-disabled `vae7b_nocurriculum`) — [7a](#attempt-7a), [7b](#attempt-7b), [Attempt 8](#attempt-8), and [Attempt 6](#attempt-6)'s checkpoints were already uploaded to gdrive and deleted locally, so z-only geometry for those would need a re-download first. Results pending.

### 7.2 Experiment results

<a id="attempt-7a"></a>
#### 7.2.1 Result: 7a confirms the identifiability-gap prediction exactly

E1 ranking intervention on the held-out test split (n=6,157; job 258607, checkpoint-1152). Full per-mode breakdown:

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0390 | 0.0568 | 0.0666 | 0.0869 | 0.0390 | 0.0492 | 0.0532 | 0.0598 |
| mask | 0.0437 | 0.0640 | 0.0788 | 0.1007 | 0.0437 | 0.0556 | 0.0617 | 0.0687 |
| mean | 0.0309 | 0.0484 | 0.0586 | 0.0778 | 0.0309 | 0.0410 | 0.0453 | 0.0514 |
| shuffle_user/target | 0.0312 | 0.0500 | 0.0601 | 0.0768 | 0.0312 | 0.0421 | 0.0462 | 0.0516 |
| noise_0.25 | 0.0393 | 0.0577 | 0.0677 | 0.0866 | 0.0393 | 0.0499 | 0.0540 | 0.0601 |
| noise_0.5 | 0.0390 | 0.0557 | 0.0664 | 0.0859 | 0.0390 | 0.0485 | 0.0529 | 0.0592 |
| noise_1.0 | 0.0393 | 0.0577 | 0.0677 | 0.0869 | 0.0393 | 0.0498 | 0.0540 | 0.0601 |

Standard test-set evaluation agrees closely (NDCG@[1,3,5,10] = 0.0391/0.0497/0.0535/0.0598, HR@[1,3,5,10] = 0.0391/0.0573/0.0666/0.0862), as expected for the same unrestricted model.

Geometry:

| | value |
|---|---|
| effective rank | 24.5 |
| top eigenvalue fraction (PC1) | 0.365 |
| components for 90% variance | 103 |
| group-identity structure (adjusted η²) | user_id 0.38, target_item 0.22 (both well above permutation baseline) |

Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42), both metrics:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0138 | [−0.0188, −0.0086] | −0.0089 | [−0.0122, −0.0055] | yes (both) |
| mean | +0.0091 | [+0.0058, +0.0123] | +0.0084 | [+0.0060, +0.0110] | yes (both) |
| shuffle_user/target | +0.0101 | [+0.0067, +0.0136] | +0.0082 | [+0.0058, +0.0107] | yes (both) |
| noise_0.25/0.5/1.0 | ≈0.0000–0.0010 | all contain 0 | ≈−0.0003–0.0006 | all contain 0 | no |

The geometry is genuinely good by every measure used in this investigation — best effective rank yet, lowest PC1 concentration, and (new this attempt) z's variance is demonstrably structured around real user/target identity, not just spread out arbitrarily. And yet `mask`, `mean`, and both `shuffle` modes are all *significantly better than raw* — removing or replacing z improves predictions — while `noise_*` shows no effect at all, meaning perturbing the existing z doesn't matter because whatever it currently contains is already actively unhelpful, not merely fragile. This is the cleanest confirmation in the whole investigation that **geometric health is neither necessary nor sufficient for causal necessity** — [7a](#attempt-7a) is a negative instance of the same principle V2 demonstrated positively: gorgeous geometry, provably harmful content. Exactly the mechanism predicted before training: with `mu_i` unconstrained and per-example, CE routed the useful signal there, leaving the KL-shaped `z` free to become adversarial noise the decoder has to route around.

**Verdict: [Attempt 7a](#attempt-7a) fails, confirming the identifiability-gap theory with real data, not just toy-model reasoning.**

**Incident (overnight, unattended)**: job 258441 ([7b](#attempt-7b)) crashed at step 768/1152 during the epoch-2 checkpoint save — `safetensors` write failed with `Disk quota exceeded`, a per-account quota distinct from raw filesystem capacity (639T free at the time). `save_total_limit=1` deletes the old checkpoint only *after* the new one saves successfully, so the moment of writing checkpoint-768 transiently requires roughly 2× a single checkpoint's footprint (~8.7GB); four attempts (6×2, 7×2) progressing in lockstep on the same account pushed this over the limit. The script had no `set -e`, so training's crash didn't stop the job — it fell through into the standard-eval/merge/calc steps against a nonexistent checkpoint, producing a cascade of unrelated-looking `FileNotFoundError`s that would have obscured the real cause without reading the full log. Fixed by: freeing ~17GB by deleting `optimizer.pt` (5.8GB each, no longer needed) from three *finished, fully E1-evaluated* checkpoints (causal, gradmask, scheduled_alpha075 — model weights untouched, confirmed present after deletion); resubmitting [7b](#attempt-7b) as job 258557 with `--resume_from_checkpoint checkpoint-384` (no retraining lost beyond the crashed epoch); adding `set -Eeuo pipefail` to the resubmitted script so a recurrence fails cleanly instead of cascading. E1 pipeline re-queued as 258558, dependent on 258557.

<a id="attempt-7b"></a>
#### 7.2.2 Result: 7b is the cleanest, most complete content-dependence signature in the whole investigation

E1 ranking intervention on the held-out test split (n=6,157; job 258558, checkpoint-1152). Raw performance and full per-mode breakdown:

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0406 | 0.0616 | 0.0736 | 0.0966 | 0.0406 | 0.0527 | 0.0576 | 0.0650 |
| mask | 0.0383 | 0.0562 | 0.0679 | 0.0906 | 0.0383 | 0.0487 | 0.0535 | 0.0608 |
| mean | 0.0300 | 0.0479 | 0.0586 | 0.0794 | 0.0300 | 0.0405 | 0.0448 | 0.0515 |
| shuffle_user | 0.0326 | 0.0499 | 0.0580 | 0.0794 | 0.0326 | 0.0426 | 0.0460 | 0.0529 |
| shuffle_target | 0.0326 | 0.0499 | 0.0580 | 0.0794 | 0.0326 | 0.0426 | 0.0460 | 0.0529 |
| noise_0.25 | 0.0413 | 0.0617 | 0.0734 | 0.0962 | 0.0413 | 0.0531 | 0.0579 | 0.0653 |
| noise_0.5 | 0.0409 | 0.0616 | 0.0732 | 0.0953 | 0.0409 | 0.0529 | 0.0576 | 0.0647 |
| noise_1.0 | 0.0409 | 0.0616 | 0.0728 | 0.0944 | 0.0409 | 0.0529 | 0.0575 | 0.0645 |

Standard test-set evaluation (`latent_attention_eval.py` + `calc.py`, full beam search) agrees closely: NDCG@[1,3,5,10] = 0.0409/0.0530/0.0578/0.0652, HR@[1,3,5,10] = 0.0409/0.0619/0.0736/0.0962 — consistent with the E1 pipeline's `raw` mode above, as expected since both run the same model unrestricted.

Geometry:

| | value |
|---|---|
| effective rank | 8.0 |
| top eigenvalue fraction (PC1) | 0.419 |
| components for 90% variance | 7 |
| user_id structure (adjusted η²) | **0.51** (higher than [7a](#attempt-7a)'s 0.38) |
| target_item structure (adjusted η²) | **0.36** (higher than [7a](#attempt-7a)'s 0.22) |

Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42), both metrics:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | +0.0060 | [+0.0023, +0.0097] | +0.0043 | [+0.0022, +0.0064] | yes (both) |
| **mean** | **+0.0172** | **[+0.0132, +0.0214]** | **+0.0135** | **[+0.0104, +0.0167]** | **yes (both)** |
| **shuffle_user** | **+0.0172** | **[+0.0132, +0.0216]** | **+0.0121** | **[+0.0093, +0.0151]** | **yes (both)** |
| **shuffle_target** | **+0.0172** | **[+0.0132, +0.0216]** | **+0.0121** | **[+0.0093, +0.0151]** | **yes (both)** |
| noise_0.25 | +0.0005 | [−0.0010, +0.0019] | −0.0002 | [−0.0009, +0.0004] | no |
| noise_0.5 | +0.0013 | [−0.0002, +0.0029] | +0.0003 | [−0.0003, +0.0010] | no |
| noise_1.0 | +0.0023 | [+0.0003, +0.0044] | +0.0006 | [−0.0003, +0.0015] | HR@10 only |

(Delta sign convention: raw − variant, so positive means the variant *underperforms* raw — the theoretically expected direction for a genuine content-dependence effect.)

Every mode points the theoretically correct direction, for the first time among the three μ/z mechanisms tried: `raw > mask > mean ≈ shuffle`, and the noise sweep is monotonically graded with perturbation size. Worth being precise rather than overstating it: `noise_1.0` reaches significance on HR@10 but not on NDCG@10 — the effect at that perturbation level is real but small and metric-sensitive, unlike the mean/shuffle effect (17.8% relative HR@10 drop, 20.8% relative NDCG@10 drop), which is significant and substantial on both metrics simultaneously. That mean/shuffle effect being roughly 3× the mask effect (6.2% HR@10 drop) is the core finding: z's *specific* content matters more than its mere presence — real personalization signal, not just "something is there."

The geometry resolves a question left open by the μ-consistency design: does anchoring `mu_i` toward its own EMA push *useful* content out of z, or just generic content? The answer is the latter — [7b](#attempt-7b)'s user/target-identity structure (η² 0.51/0.36) is *stronger* than [7a](#attempt-7a)'s (0.38/0.22) despite z's overall spread being more concentrated (effective rank 8.0 vs 24.5). The anchor didn't just fix the causal-dependence failure; it left behind a z whose remaining variance is more tightly organized around genuinely identity-relevant differences, not less.

**Verdict: [Attempt 7b](#attempt-7b) works — the first mu/z-decomposition attempt (of three: hard-normalization, pure-KL, KL+anchor) to produce genuine, statistically robust, correctly-signed content-dependence.** Comparable in kind to the two prior successes (margin=0.2, alpha=0.75), and arguably the most complete signature yet, since it shows both activation-dependence (mask matters, modestly) and content-dependence (mean/shuffle/noise matter, substantially) in a single coherent, gradient story rather than one or the other. Confirms the identifiability-gap theory from the other direction: closing the loophole ([7a](#attempt-7a) → [7b](#attempt-7b)) is sufficient to flip a confirmed failure into a genuine success, using exactly the mechanism predicted (a soft anchor, not a hard-forced ratio).

#### 7.2.3 Follow-up results: remove KL or loosen the curriculum

##### 7.2.3.1 Result: both single-variable changes break 7b's property, in two different ways — 7b is a genuine sweet spot, not an arbitrary point

Full per-mode metrics and bootstrap 95% CIs (both HR@10 and NDCG@10), n=6,157:

**Run A (KL off, `mu_consistency=0.5` kept, `alpha=0.75` kept)**

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0401 | 0.0565 | 0.0692 | 0.0906 | 0.0401 | 0.0496 | 0.0548 | 0.0617 |
| mask | 0.0413 | 0.0630 | 0.0741 | 0.0968 | 0.0413 | 0.0538 | 0.0583 | 0.0656 |
| mean | 0.0403 | 0.0565 | 0.0690 | 0.0895 | 0.0403 | 0.0497 | 0.0548 | 0.0614 |
| shuffle_user/target | 0.0395 | 0.0562 | 0.0685 | 0.0890 | 0.0395 | 0.0492 | 0.0542 | 0.0608 |
| noise_0.25/0.5/1.0 | ~0.040 | ~0.057 | ~0.069 | ~0.090 | ~0.040 | ~0.050 | ~0.055 | ~0.062 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0062 | [−0.0109, −0.0013] | −0.0039 | [−0.0070, −0.0010] | yes (both) |
| mean | +0.0011 | [+0.0002, +0.0023] | +0.0003 | [−0.0004, +0.0010] | HR@10 only |
| shuffle_user/target | +0.0016 | [+0.0005, +0.0029] | +0.0009 | [+0.0003, +0.0017] | yes (both) |
| noise_0.25/0.5/1.0 | all n.s. | | all n.s. | | no |

**Run B (`alpha_max=0.9`, KL and anchor both kept as in [7b](#attempt-7b))**

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0481 | 0.0669 | 0.0807 | 0.1017 | 0.0481 | 0.0591 | 0.0647 | 0.0714 |
| mask | 0.0481 | 0.0663 | 0.0784 | 0.0994 | 0.0481 | 0.0586 | 0.0636 | 0.0703 |
| mean | 0.0481 | 0.0655 | 0.0793 | 0.0996 | 0.0481 | 0.0584 | 0.0641 | 0.0705 |
| shuffle_user/target | 0.0482 | 0.0653 | 0.0802 | 0.0997 | 0.0482 | 0.0584 | 0.0645 | 0.0707 |
| noise_0.25/0.5/1.0 | ~0.048 | ~0.067 | ~0.080 | ~1.01 | ~0.048 | ~0.059 | ~0.065 | ~0.071 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | +0.0023 | [−0.0003, +0.0049] | +0.0010 | [−0.0004, +0.0025] | no |
| mean | +0.0021 | [+0.0002, +0.0042] | +0.0008 | [−0.0002, +0.0019] | HR@10 only, tiny |
| shuffle_user/target | +0.0019 | [−0.0002, +0.0042] | +0.0007 | [−0.0004, +0.0017] | no |
| noise_0.25/0.5/1.0 | all n.s. | | all n.s. | | no |

Geometry:

| | Run A (no KL) | [7b](#attempt-7b) (reference) | Run B (α=0.9) |
|---|---|---|---|
| effective rank | **1.5** | 8.0 | **35.3** |
| top eigenvalue fraction (PC1) | **0.932** | 0.419 | **0.208** |
| cosine to global mean | **0.99999** | ~0.72 ([7b](#attempt-7b), from earlier table) | 0.739 |
| user_id / target_item adjusted η² | **−0.044 / −0.064** (below permutation baseline) | 0.51 / 0.36 | 0.359 / 0.179 |

**Run A: removing the KL collapses the geometry more severely than any other μ/z attempt** — effective rank 1.5, cosine 0.99999, identity structure statistically indistinguishable from noise (negative adjusted η²) — worse than [7a](#attempt-7a) (rank 24.5) and worse even than gradmask ([Attempt 5](#attempt-5), rank 1.17, the previous worst in this investigation). And it **reverts to the `mask`-beats-`raw` inversion** (HR@10 Δ=−0.0062, NDCG@10 Δ=−0.0039, both significant) — the exact failure signature of [7a](#attempt-7a) and [Attempt 8](#attempt-8). So KL is not simply a tax on [7b](#attempt-7b)'s performance (the hypothesis this run was designed to test) — it is *necessary*, jointly with the anchor, for both the geometry and the correctly-signed causal result. Removing it doesn't isolate the anchor's contribution cleanly; it destroys the mechanism.

**Run B: the best geometry of any μ/z attempt in this investigation, paired with essentially no causal effect.** Effective rank 35.3 beats [7a](#attempt-7a)'s 24.5; PC1 concentration (0.208) is the lowest measured; identity structure (η² 0.36/0.18) is solid. Yet only `mean`'s HR@10 reaches significance, with a tiny effect size (+0.0021), and it does not replicate on NDCG@10 or on `shuffle`. This is the single cleanest "great geometry, causally inert" pairing in the whole investigation — cleaner than [7a](#attempt-7a)'s, because here nothing else changed except the schedule's tightness, isolating exactly which knob is responsible.

**Verdict: [7b](#attempt-7b) (alpha=0.75, KL=1.0, anchor=0.5) is a genuine sweet spot, not an arbitrary point on a tunable curve.** Both single-variable perturbations tested — loosen the schedule (Run B), remove the KL (Run A) — independently break the property that makes [7b](#attempt-7b) work, in two different directions: Run B trades causal necessity for raw performance; Run A loses both simultaneously. The x-path deficit identified earlier is real ([7b](#attempt-7b)'s x-path is 15-20% below what [Attempt 8](#attempt-8) achieves under the same schedule), but neither of these two natural fixes closes it without cost. Closing it without cost, if possible at all, needs a different kind of change — the decoupled-latent-encoder architecture change (this section, above) is the candidate currently on record; information-advantage mechanisms are being investigated separately.

##### 7.2.3.2 Where 7b's performance gap actually is: the x-path, not r

For schedule-trained models `mask` (thought slot zeroed) is an in-distribution condition — `history_only` batches train exactly it — so it measures the **x-path alone**, and `raw − mask` measures **what r adds on top of x**. (For the unrestricted `v4` baseline `mask` is out-of-distribution and not comparable.) HR@10:

| checkpoint | raw | x-path alone (`mask`) | r adds | mean / shuffle_user |
|---|---:|---:|---:|---|
| v4 baseline | 0.1152 | (0.0771, OOD) | — | 0.1152 / 0.1142 |
| alpha=0.75 only | 0.1038 | 0.1064 | −0.0026 | 0.0968 / 0.0970 |
| [7a](#attempt-7a) (KL, no anchor) | 0.0869 | 0.1007 | −0.0138 | 0.0778 / 0.0768 |
| **[7b](#attempt-7b) (KL + anchor)** | 0.0966 | **0.0906** | **+0.0060** | 0.0794 / 0.0794 |
| 8 (anchor + alignment, no KL) | 0.1057 | 0.1108 | −0.0050 | 0.0958 / 0.0955 |

Three observations:
1. **[7b](#attempt-7b) is the only schedule-trained model in which r is net-helpful** (`raw > mask`); in alpha-only, [7a](#attempt-7a), and [Attempt 8](#attempt-8), the decoder does better with the thought slot zeroed even when r's content is demonstrably real (as in [Attempt 8](#attempt-8)). [7b](#attempt-7b)'s representation is good *and* the decoder combines it productively with x.
2. **[7b](#attempt-7b)'s masked condition is weak, but this does not prove its raw-performance gap is "entirely in the x-path."** The second pass is the joint nonlinear function `F(x, r)`, and ranking metrics are non-additive. `raw - mask` is a checkpoint-specific intervention effect, not a portable contribution that can be added to another model's masked score. [Attempt 8](#attempt-8) is the direct counterexample: it has the strongest masked score but adding its correct `r` makes performance worse.
3. **Gradient leakage remains a plausible, not proven, contributor.** `mu_head`/`z_head` are single linear layers on the backbone's pooled final state. KL/anchor gradients therefore reach the same history representations the decoder consumes directly. The two KL models have the two weakest masked conditions, but the joint `F(x, r)` interaction means improving those representations alone need not improve raw performance.

**Candidate design if the hypothesis holds — a decoupled latent encoder**: compute μ/z from a *detached* pooled state through a small MLP+LayerNorm encoder with its own parameters, so latent-space gradients stop at the encoder. This can protect the existing representation, but there is no justified arithmetic such as "masked score + the old r gain = new raw score": it must be evaluated on the complete joint `F(x, r)` prediction. The general statement to test is: *latent-space objectives should be isolated from the representation the decoder's direct history path consumes, while the correct latent must still improve the joint prediction.*

**Decision gate — Runs A/B (jobs 259793/259794) before building it.** Run A (KL off, weights still shared) is the direct test: if its x-path recovers to ~0.11 the leak hypothesis is confirmed (and if its r turns net-negative like [Attempt 8](#attempt-8), the KL is also what makes r *usable* — so isolating rather than removing it is the configuration that gets both). If Run A's x-path stays near 0.09, the hypothesis is wrong and decoupling is dropped. Run B (alpha_max 0.9) bounds the schedule's own tax independently.

#### 7.2.4 Follow-up result: disable the attention curriculum

##### 7.2.4.1 Result: the most extreme activation-dependence-without-content-dependence signature in this investigation

Full per-mode metrics and bootstrap 95% CIs (n=6,157, job 271275):

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0544 | 0.0762 | 0.0900 | 0.1139 | 0.0544 | 0.0671 | 0.0728 | 0.0805 |
| mask | 0.0206 | 0.0413 | 0.0549 | 0.0809 | 0.0206 | 0.0325 | 0.0381 | 0.0464 |
| mean | 0.0546 | 0.0767 | 0.0897 | 0.1143 | 0.0546 | 0.0675 | 0.0728 | 0.0807 |
| shuffle_user/target | 0.0541 | 0.0760 | 0.0898 | 0.1152 | 0.0541 | 0.0670 | 0.0727 | 0.0809 |
| noise_0.25/0.5/1.0 | ~0.055 | ~0.076 | ~0.090 | ~0.114 | ~0.055 | ~0.067 | ~0.073 | ~0.081 |

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| **mask** | **+0.0330** | **[+0.0278, +0.0385]** | **+0.0341** | **[+0.0303, +0.0381]** | **yes (both, by far the largest effect measured this session)** |
| mean | −0.0005 | [−0.0024, +0.0015] | −0.0002 | [−0.0011, +0.0005] | no |
| shuffle_user/target | −0.0013 | [−0.0031, +0.0006] | −0.0004 | [−0.0011, +0.0003] | no |
| noise_0.25/0.5/1.0 | all n.s. | | all n.s. | | no |

Raw performance (HR@10=0.1139, NDCG@10=0.0805) essentially matches baseline (0.1152/0.0806) — confirming the schedule was the *entire* source of [7b](#attempt-7b)'s performance cost, not the μ/z mechanism itself. But `mask` removing 29% of HR@10 while `mean`/`shuffle`/every `noise` level shows zero effect is the purest, most extreme version of the exact "activation-dependence without content-dependence" signature that motivated this whole investigation in the first place: the decoder needs *something* non-zero at the thought position (an OOD-input sensitivity, not a personalization signal), but genuinely does not care what that something is.

Consistent with the geometry from the same checkpoint (effective rank 1.28, the worst measured, and a two-point-mass degeneracy uncorrelated with any identity dimension): with the schedule fully removed, the μ/z mechanism's own pressures (KL, anchor) are entirely insufficient on their own to prevent collapse. The schedule was not adding a modest assist on top of a mechanism that mostly worked by itself — it was doing essentially all of the causal-necessity work.

**Verdict: the Run A / Run B / curriculum-disabled sequence together fully characterize [7b](#attempt-7b)'s trade-off curve.** Loosening the schedule trades causal necessity for raw performance smoothly and monotonically ([7b](#attempt-7b) → Run B → curriculum-disabled), while removing the KL (Run A) breaks the mechanism outright regardless of schedule. There is no configuration in this family that gets both baseline performance and genuine content-dependence — [7b](#attempt-7b)'s alpha=0.75 operating point is very close to the best trade-off available within this design, not an arbitrary or undertuned choice. Closing the remaining gap requires a structurally different mechanism (the decoupled-latent-encoder idea, or an information-advantage approach), not further tuning within this family.

#### 7.2.5 z-only geometry results (job 276414)

| | Run A (no KL) | Run B (α=0.9) | curriculum-disabled |
|---|---|---|---|
| effective rank: r → z | 1.5 → 2.0 | 35.3 → 34.9 | 1.28 → 1.28 |
| top eigenvalue fraction: r → z | 0.87 → 0.87 | 0.208 → 0.211 | 0.939 → 0.939 |
| user_id η²: r → z | ~0 → −0.006 | 0.359 → 0.359 | 0.183 → 0.183 |
| target_item η²: r → z | ~0 → −0.058 | 0.178 → 0.178 | 0.036 → 0.036 |
| **cosine-to-mean: r → z** | 0.99999 → 0.99999 | **0.739 → 0.410** | **0.730 → 0.031** |

**Rank and identity-structure (η²) are essentially invariant to the r-vs-z choice in all three cases.** Expected in hindsight: PCA and η² are both computed relative to the data's own mean, hence invariant to any constant shift applied uniformly to every example. This is reassuring for everything concluded from r-based geometry earlier in this document — none of it was an artifact of μ dominating the sum; z's own rank and identity structure match what r showed throughout.

**`cosine_to_global_mean` is the one diagnostic that isn't invariant, and it reveals something the r-based plots concealed for the curriculum-disabled checkpoint specifically.** z's cosine-to-mean there is 0.031 — near-orthogonal to its own population mean — while simultaneously having effective rank 1.28 (98%+ of variance in one direction). Both can only be true together if z has collapsed into **two roughly antipodal point-masses along a single axis** (`+c·v` and `−c·v` for some direction `v`), with group sizes/magnitudes that roughly cancel in the mean, making the mean itself near-zero and cosine-to-a-near-zero-vector numerically small regardless of actual direction. The earlier r-based figure's bimodal-but-both-positively-aligned picture (cosine 0.65 / 0.77 to the mean) was μ's own consistent direction masking this: once μ is subtracted out, z's true structure is a **sign-flip collapse**, a strictly more precise (and worse) diagnosis than "two similar templates" — the residual isn't just uninformative, it's binarized onto a single learned axis with no other structure at all.

Run B shows the opposite correction: z's cosine-to-mean (0.410) is *lower* than r's (0.739), meaning z alone is *more* diverse than r's number suggested — μ was adding a bit of apparent extra alignment on top of an already-healthy, non-degenerate z. Run A shows no correction either way (already maximally collapsed, cosine ≈1 for both r and z).

**Methodological takeaway for future geometry reports on μ/z checkpoints**: rank/η²/PC1-concentration can be read directly off `r`, since they're translation-invariant and z-based numbers will match. `cosine_to_global_mean` (and any other translation-*variant* statistic) must be computed on `z` directly — reading it off `r` can both overstate collapse severity (Run B: r looked less diverse than z actually is) and understate it (curriculum-disabled: r's bimodal-but-aligned picture hid a sharper antipodal collapse in z).

<a id="attempt-8"></a>
## 8. Attempt 8: memory-bank semantic alignment for $z$

### 8.1 Approach

A third mechanism for the same μ+z decomposition, motivated by a specific hypothesis about *what kind* of structure z should have: the same user, or the same target item, should produce similar reasoning. Rather than "z should be spread out" (VICReg, [Attempt 6](#attempt-6)) or "z should match a fixed reference distribution" (KL, [Attempt 7](#attempt-7)), this asks z to be organized by task-relevant identity.

#### 8.1.1 Why this is routed through the decoder, not computed in raw z-space

The literal proposal — pull z_i toward z_j for same-user/same-target j, push it away from an unrelated z_k — is naturally expressed as a distance in z-space (cosine or L2). That would inherit exactly the gap that broke [Attempt 7a](#attempt-7a): a constraint on z's own geometry doesn't force the *decoder* to depend on it. So instead:

- **Positive** (same `user_id` or same `target_item`): splice the candidate's z onto the **anchor's own μ** — `r = μ_i + z_j` — and minimize the resulting NLL against the **anchor's own true target**. This asks a decoder-coupled question: does this other identity's reasoning still work in my context? μ is *not* shared/cancelled here (unlike [Attempt 6](#attempt-6)'s negative construction) — it's the anchor's own per-example mean, held fixed while z is swapped, isolating whether z's content is portable across identities that should plausibly reason alike.
- **Negative** (different `user_id` AND different `target_item`): same splice, hinge-pushed up to a margin (reusing [Attempt 2](#attempt-2)'s validated margin mechanism) — an unrelated identity's reasoning should predict badly.
- **Quality weighting** (the user's addition): each positive is weighted by `stopgrad(exp(-own_nll_j))`, where `own_nll_j` is candidate j's own per-example loss from when its z was cached. A candidate whose own reasoning didn't lead to a correct prediction is down-weighted as a positive — a good anchor shouldn't be dragged toward a low-quality reference.

#### 8.1.2 Implementation (`src/latent/semantic_align_dataset.py`, `semantic_alignment_attention.py`, `latent_attention_train_semantic.py`)

`SemanticAlignLatentRDataset` (subclass of `LatentRDataset`, additive only) attaches a stable hash of `user_id` and the raw `item_id` to each training example; verified these survive `DataCollatorForSeq2Seq`/`HFDataset` unchanged as plain `[batch]` int64 tensors (`remove_unused_columns=False` was already set in every training script for this reason).

`SemanticAlignmentLatentModel` (built on `VAEDecomposedLatentModel`, reusing [Attempt 7](#attempt-7)'s `mu_head`/`z_head` encoder unchanged) maintains a 20,000-entry memory bank of `(z, user_hash, target_hash, own_nll)` tuples from recent steps (a MoCo-style queue — needed because with `micro_batch_size=4` and ~6865 users / ~5830 items over 49251 rows, same-identity pairs essentially never co-occur in-batch; a 4096-entry bank was the initial choice, raised to 20,000 (~117.6MB, negligible on a 40GB+ GPU) after computing the hit-rate math: modeling same-identity co-occurrence as Poisson, 4096 entries (8.3% of the 49,251-row dataset) gives only ~45%/~50% P(>=1 hit) for same-user/same-target respectively (expected hits 0.60/0.70), while 20,000 (40.6% of data) gives ~95%/~97% (expected hits 2.92/3.42) -- close to saturating both axes rather than leaving a large fraction of anchors with no positive at all most steps). Each step: sample one positive index per anchor (`same_user | same_target`, random tie-break) and one negative (`~same_user & ~same_target`); anchors with no valid positive/negative in the current bank state (common early in training) are masked out of that term, not zero-filled. `kl_weight` defaults to 0 ([Attempt 7](#attempt-7)'s prior is off, to isolate this mechanism's own effect); `mu_consistency_weight` defaults to 0.5, kept on since preliminary [Attempt 7](#attempt-7) results favor it.

Verified with a CPU toy model (bf16 throughout, learning from the [Attempt 7a](#attempt-7a) dtype-bug incident): empty-bank step correctly gives zero pos/neg loss with no crash; a bank entry sharing `user_hash` is correctly found and weighted; gradients reach both `mu_head` and `z_head`; `history_only` schedule batches correctly skip both the alignment terms and the bank push (z has no path to the loss there); bank size cap is respected; eval-mode `generate_embs`/`model.generate()` work under bf16. Save→reload through both loaders reproduces the layer; a reload check under bf16 showed a small (~1%) numerical drift that a float32 control run isolated to bf16 CPU matmul non-determinism from serialization changing memory layout/kernel dispatch (not a logic bug — float32 reload is exact, `max|Δr| = 0.0`).

#### 8.1.3 Run

Submitted as job 258747 with `bank_size=4096`, then cancelled after 3.5 minutes and resubmitted as job 258749 with `bank_size=20000` once the hit-rate math above showed 4096 was leaving too many anchors without a positive most steps (single GPU, alpha=0.75 schedule retained, μ-consistency=0.5, `pos_weight=neg_weight=1.0`, `margin=2.0`). E1 pipeline queued as 258750, dependent on successful completion. Results pending.

### 8.2 Experiment results

#### 8.2.1 Result: the strongest, most robust content-dependence signal of the session — but r is still net-harmful

E1 ranking intervention on the held-out test split (n=6,157; job 258750, checkpoint-1152). Full per-mode breakdown:

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0489 | 0.0710 | 0.0806 | 0.1057 | 0.0489 | 0.0615 | 0.0654 | 0.0735 |
| mask | 0.0562 | 0.0770 | 0.0888 | 0.1108 | 0.0562 | 0.0682 | 0.0731 | 0.0802 |
| mean | 0.0369 | 0.0612 | 0.0728 | 0.0958 | 0.0369 | 0.0513 | 0.0560 | 0.0633 |
| shuffle_user/target | 0.0390 | 0.0585 | 0.0703 | 0.0955 | 0.0390 | 0.0504 | 0.0553 | 0.0634 |
| noise_0.25 | 0.0500 | 0.0698 | 0.0806 | 0.1049 | 0.0500 | 0.0614 | 0.0657 | 0.0735 |
| noise_0.5 | 0.0494 | 0.0682 | 0.0804 | 0.1033 | 0.0494 | 0.0603 | 0.0653 | 0.0727 |
| noise_1.0 | 0.0445 | 0.0664 | 0.0797 | 0.0983 | 0.0445 | 0.0571 | 0.0626 | 0.0685 |

Raw performance (HR@10=0.1057, NDCG@10=0.0735) is the best of any attempt this session, at or above the ~0.10/0.07 baseline — the first μ/z mechanism to cost essentially nothing on task performance.

Geometry:

| | value |
|---|---|
| effective rank | 11.3 |
| top eigenvalue fraction (PC1) | 0.224 |
| components for 90% variance | 8 |
| random-pair cosine to mean | **0.972** (high — comparable to several collapsed checkpoints) |
| user_id structure (adjusted η²) | 0.45 |
| target_item structure (adjusted η²) | 0.29 |
| unique-vector fraction | 0.981 (highest of any attempt) |

Paired bootstrap 95% CIs vs. `raw` (2,000 resamples, seed 42), both metrics:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | Significant? |
|---|---:|---|---:|---|:---:|
| mask | −0.0050 | [−0.0086, −0.0013] | −0.0067 | [−0.0092, −0.0043] | yes (both) |
| **mean** | **+0.0099** | **[+0.0065, +0.0132]** | **+0.0101** | **[+0.0080, +0.0122]** | **yes (both)** |
| **shuffle_user/target** | **+0.0102** | **[+0.0067, +0.0140]** | **+0.0101** | **[+0.0077, +0.0125]** | **yes (both)** |
| noise_0.25 | +0.0008 | [−0.0011, +0.0028] | −0.0001 | [−0.0012, +0.0010] | no |
| noise_0.5 | +0.0024 | [−0.0002, +0.0050] | +0.0008 | [−0.0007, +0.0023] | no |
| **noise_1.0** | **+0.0075** | **[+0.0034, +0.0117]** | **+0.0050** | **[+0.0026, +0.0074]** | **yes (both)** |

Two things are both true here, and the honest verdict has to hold both at once:

**The content-dependence signal is the cleanest and most robust of the whole investigation.** `mean`, `shuffle_user/target`, and `noise_1.0` are all significant on *both* HR@10 and NDCG@10 — the first attempt where the noise sweep's top level clears significance on both metrics simultaneously ([7b](#attempt-7b)'s `noise_1.0` was HR@10-only). The effect sizes are substantial (mean/shuffle ≈9.5% relative HR@10 drop) and the noise sweep is cleanly graded (noise_0.25/0.5 not significant, noise_1.0 significant) — textbook content-dependence, and it costs essentially nothing in raw performance.

**But `mask` significantly *beats* raw** — the same inversion seen in [Attempt 6](#attempt-6) (ρ=1.0) and [7a](#attempt-7a): removing r entirely outperforms using it. So the trained r is simultaneously informative (its specific content matters, more clearly than in any prior attempt) and net-costly (the model would score higher overall if it ignored r altogether). The memory-bank alignment mechanism succeeded at making z's content matter — the semantic positive/negative pressure clearly worked — but didn't succeed at making that content net-*helpful*, only net-*present*.

The geometry adds a genuinely new data point to this investigation's recurring theme, from a new angle: random-pair cosine (0.972) is high enough that on its own it would be read as a collapse signature comparable to several failed checkpoints, yet real, substantial, statistically robust content-dependence coexists with it. Geometric summary statistics — cosine included, not just effective rank/PC1 — are not reliable stand-ins for the causal test.

**Verdict: [Attempt 8](#attempt-8) partially succeeds.** It is the strongest demonstration in this investigation that z's *specific content* is read and used by the decoder (mean/shuffle/noise all robust), but it does not meet the original goal of making r causally necessary *without cost* — `mask` beating `raw` means the decoder would still be better off ignoring the reasoning path entirely. The quality-weighted memory-bank mechanism is worth keeping as the source of the content-dependence signal; what's still missing is something that makes the decoder's *use* of r beneficial, not just responsive to it — plausibly the DSN-style reconstruction idea discussed earlier ([Attempt 7](#attempt-7)'s design notes), or a direct penalty when `mask` outperforms `raw` during training, are the natural next steps if this line is continued.

<a id="attempt-9"></a>
## 9. Attempt 9: baseline-preserving latent residual correction

### 9.1 Approach

#### 9.1.1 Motivation

[7b](#attempt-7b) proves that a sample-specific latent can be causally useful, but its joint recommendation performance remains below the unrestricted baseline. Improving the masked condition alone is not the objective: R3 predicts with the nonlinear joint decoder `F(x, r)`, so a stronger `F(x, 0)` does not imply a stronger `F(x, r)`. [Attempt 8](#attempt-8) demonstrates the failure directly (`mask > raw`). [Attempt 9](#attempt-9) therefore preserves a complete high-performing joint decoder and gives the new latent only one job: learn a correction that reduces the remaining recommendation loss.

The source is the curriculum-disabled [7b](#attempt-7b) checkpoint (`HR@10=0.1139`, `NDCG@10=0.0805`). Its `mean` intervention is indistinguishable from raw (`0.1143/0.0807`), so its train-set mean thought is an empirically validated shared scaffold `mu`, not a guessed replacement.

#### 9.1.2 Baseline-preserving residual correction

Let `F_theta0` be the completed source checkpoint and let

    mu = mean_train(r_i)
    z_i = A_phi(stopgrad(H_theta0(x_i)))
    r_i^(9) = mu + z_i
    y_hat_i = F_theta0(x_i, r_i^(9))

`A_phi` is a new history-pooling attention layer followed by `LayerNorm -> Linear -> SiLU -> Linear`. Its final linear layer is initialized to exactly zero, hence `z_i=0` at initialization and [Attempt 9](#attempt-9) initially reproduces the source model's measured `F(x, mu)` condition exactly. The source backbone, source thought generator, and second-pass decoder are all frozen; only `A_phi` is trainable. Its input history states are detached as a structural guard, not merely protected by the optimizer configuration.

This is deliberately a **latent-space residual**, not an additive decomposition of ranking metrics or a separate logit model. The unchanged R3 second pass still consumes `x` and `mu+z_i` together and may use their full nonlinear interaction. Freezing `F_theta0` only ensures residual learning cannot improve its objective by damaging or rewriting the already-working recommendation model.

Training uses ordinary target CE only — no margin, KL, geometry regularizer, or attention curriculum. This differs from [Attempt 2](#attempt-2): it does not ask the correct latent merely to beat a wrong latent, and it does not provide a trainable wrong branch that can be made artificially bad. Since every original parameter is frozen, any reduction in CE must be mediated by the new `z_i` correction.

Implemented additively in:

- `src/latent/residual_correction_attention.py`: checkpoint-loadable `mu+z` wrapper, detached residual encoder, zero initialization;
- `src/latent/latent_attention_train_residual.py`: freezes the source checkpoint and trains only the residual path;
- `scripts/latent_attempt9_residual_full.sbatch`: full training plus standard ranking evaluation;
- `scripts/R3_attempt9_residual_e1_full.sbatch`: r/z geometry and nine-mode E1 evaluation;
- both vanilla checkpoint loaders reconstruct the wrapper from `config.latent_residual_correction`.

The E1 suite adds `mu_only`, which substitutes the exact persistent `mu` buffer. This is the primary control:

- `raw > mu_only`: the learned sample-specific correction is net helpful;
- `raw > mean/shuffle`: its identity-specific content matters;
- `raw >= source baseline`: it translates into competitive recommendation performance;
- `mask` remains diagnostic but is not the Attempt-9 baseline, because it removes the already-validated shared scaffold as well as the correction.

<a id="attempt-9a"></a>
#### 9.1.3 Follow-up 9a: orthogonal prior/residual subspaces

[Attempt 9](#attempt-9) leaves the correction unconstrained, so it can decompose as `z_i = c_i mu + z_i_perp`. The parallel term can change only the strength of the generic scaffold — a possible useful calibration effect, but also a possible return to activation-scale shortcut learning. [Attempt 9a](#attempt-9a) is a strict single-variable ablation that removes this degree of freedom:

    u_mu = mu / ||mu||
    z_i = z_raw_i - (z_raw_i^T u_mu) u_mu
    r_i = mu + z_i

Therefore `mu^T z_i = 0` for every example and `||r_i||^2 = ||mu||^2 + ||z_i||^2`. The coefficient of `mu` is fixed; any learned correction must occupy the complementary subspace. Everything else is identical to [Attempt 9](#attempt-9): same source checkpoint and train mean, detached residual encoder, zero initialization, frozen original model, CE objective, hyperparameters, standard evaluation, and nine E1 modes.

The trainer now logs `residual_parallel_energy_fraction`. It is exactly zero up to numerical precision for [9a](#attempt-9a); for unconstrained [Attempt 9](#attempt-9) it measures whether the learned correction primarily rescaled the prior. The comparison is interpreted as follows:

- 9 and [9a](#attempt-9a) similar: [Attempt 9](#attempt-9) naturally learned outside the prior direction;
- 9 better with real content-dependence: prior-strength modulation was useful;
- [9a](#attempt-9a) better: removing the shared-direction shortcut improved personalization;
- both `raw ~= mu_only`: the frozen decoder has no exploitable sample-specific correction in either space.

### 9.2 Experiment results

#### 9.2.1 Attempt 9 status

**Status: implemented and unit-verified (zero-init baseline preservation, detached backbone gradient, residual-only full-model backward, and save/reload through both model loaders).** The existing evidence artifact loads as a 1536-d mean vector with `L2=38.8868`. Full training/evaluation is job 277732; its dependent E1 evaluation is job 277733.

#### 9.2.2 Attempt 9a status

Implemented through `orthogonal_to_mu` in the shared residual layer/trainer, with separate launchers `scripts/latent_attempt9a_orthogonal_full.sbatch` and `scripts/R3_attempt9a_orthogonal_e1_full.sbatch`. **Status: implemented and unit-verified; five [Attempt 9](#attempt-9)/[9a](#attempt-9a) tests pass. Full training/evaluation not yet submitted.**

<a id="attempt-10"></a>
## 10. Attempt 10: collaborative multi-positive InfoNCE on plain R3

### 10.1 Approach

This attempt returns to the undecomposed plain-R3 latent `r_i` and directly imposes the collaborative-filtering structure proposed for sample-level reasoning: two samples should have compatible latent representations when they belong to the same user or have the same target item. It does not use the μ/z decomposition, an attention curriculum, an oracle latent, or a warm-up phase.

#### 10.1.1 Intuition: sample-specific reasoning with collaborative structure

The goal is not to make every sample use one global reasoning vector. Each interaction still has its own latent `r_i`, because the particular history and next-item event are sample-specific. The additional assumption is that reasoning should have *partially shared structure* along two axes:

- **same user:** two interactions from user `u` should encode some stable preference signal, even when their target items differ;
- **same target:** two different users who reach item `t` should encode some compatible item-property signal, even when their histories differ.

This is analogous to collaborative filtering, but applied to latent reasoning rather than directly to user/item embeddings. It supplies supervision that ordinary next-item CE does not: CE only asks whether each individual `r_i` participates in predicting its own target, whereas collaborative InfoNCE also asks whether the *relations among latent vectors* agree with known user/item relations.

A single raw-space cosine loss would be too rigid. Same-user and same-target relations can conflict: two samples may share a user but not a target, or a target but not a user. The model therefore learns two separate projection heads:

$$
q_i^{u}=\operatorname{normalize}\!\left(g_u(r_i)\right),
\qquad
q_i^{t}=\operatorname{normalize}\!\left(g_t(r_i)\right).
$$

Here, $g_u$ and $g_t$ are independent `LayerNorm -> Linear(1536, 256)` mappings. The user head can retain the dimensions useful for preference similarity while the target head can retain dimensions useful for item similarity. Both losses still backpropagate into the common `r_i`, so the deployed latent is jointly shaped by the two relations; the projection heads themselves are not used for recommendation inference.

#### 10.1.2 Multi-positive InfoNCE objective

For anchor `i`, define relation-specific positives from other samples (`sample_id != sample_id_i`):

$$
P_i^{u}=\{j \neq i : u_j=u_i\},
\qquad
P_i^{t}=\{j \neq i : t_j=t_i\}.
$$

A candidate is a negative only when it shares **neither** relation:

$$
N_i=\{j \neq i : u_j \neq u_i \;\land\; t_j \neq t_i\}.
$$

The AND condition is important. Treating a same-target example as a negative for the user loss (or a same-user example as a negative for the target loss) would make the two objectives contradict one another. For relation $a\in\{u,t\}$, define the temperature-scaled cosine logit as

$$
s_{ij}^{a}=\frac{\left(q_i^{a}\right)^{\!\top}q_j^{a}}{\tau}.
$$

The per-anchor multi-positive loss is

$$
\mathcal{L}_i^{a}
=-\log
\frac{\displaystyle\sum_{p\in P_i^{a}}\exp\!\left(s_{ip}^{a}\right)}
{\displaystyle\sum_{j\in P_i^{a}\cup N_i}\exp\!\left(s_{ij}^{a}\right)}.
$$

Unlike a single-positive loss, this maximizes the total probability assigned to *any* valid same-user/same-target example, so one arbitrarily sampled positive is not treated as the only correct explanation. An anchor with no available positive or negative is skipped for that relation rather than given a fabricated zero vector. The complete training objective is

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+0.1\,\mathcal{L}_{\mathrm{user}}
+0.1\,\mathcal{L}_{\mathrm{target}}.
$$

Here, $\tau=0.1$, with at most 16 positives per relation and 256 negatives per anchor.

#### 10.1.3 Why the FIFO bank and correctness gate are needed

The training micro-batch contains only four samples, while the dataset contains thousands of users and items. Same-user or same-target pairs would therefore occur too rarely for an in-batch-only contrastive loss. A non-persistent FIFO corpus holds the 10,000 most recent eligible candidates as `(r, user_id, target_id, sample_id, NLL)`. Every new micro-batch can contrast its four anchors against this much larger recent corpus. Once full, new entries overwrite the oldest entries; the queue is run-local and is not restored from a checkpoint.

Not every target-labeled latent is a trustworthy positive. A sample enters the candidate bank only when the normal recommendation decoder predicts **every supervised target token correctly** (`token_accuracy=1.0`). This gate applies to both positive and negative candidates and prevents a latent that currently produces the wrong answer from teaching other anchors merely because its metadata matches. The anchor itself does not have to pass the gate: an incorrect anchor can still receive a gradient toward high-quality related examples.

The update order for each training step is:

1. Plain R3 produces `r_i` and runs the normal history-plus-latent recommendation pass.
2. The decoder logits give `L_CE`, per-example NLL, and exact target-token accuracy.
3. Current correct examples are appended to the detached FIFO candidates for that step.
4. User and target InfoNCE losses are computed for each anchor with eligible pairs.
5. Correct current latents are detached and inserted into the FIFO for future steps.

Stored/candidate latents are detached, so gradients cannot flow backward through old examples or create an ever-growing computation graph. Contrastive gradients flow through the current anchor `r_i` into the shared R3 latent producer, and into the two projection heads. At inference, the queue, metadata, correctness gate, and InfoNCE computation disappear; recommendation uses the ordinary candidate-independent R3 path with the learned raw `r_i`. The objective therefore changes training supervision without adding an inference-time retrieval corpus or candidate-relative scorer.

The losses are active for all three epochs with no explicit two-epoch warm-up. This avoids waiting until ordinary R3 has already collapsed before asking InfoNCE to separate the vectors. There is nevertheless a natural data-dependent start: at the first steps the queue is empty and anchors without positives contribute no contrastive term; the signal becomes denser as correct examples accumulate.

### 10.2 Experiment results

#### 10.2.1 Run and ranking result

Training and standard ranking completed in job 283515. By the end of training the FIFO was full (10,000 entries), and both relations continued to produce eligible positives, so the reported outcome is not an empty-bank or inactive-loss artifact. Standard full-beam test ranking (`n=6,157`) was:

| metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| NDCG | 0.0533 | 0.0675 | 0.0738 | **0.0813** |
| HR | 0.0533 | 0.0780 | 0.0932 | **0.1166** |

The strongest plain-R3 reference (`v4`, checkpoint-1152) is HR@10=0.1152 and NDCG@10=0.0806. Collaborative InfoNCE is therefore slightly higher in point estimate (+0.0014 HR@10, +0.0007 NDCG@10), while the main defensible conclusion is that it **preserves baseline recommendation performance**. The difference is too small to call an improvement without a paired confidence interval or repeated seeds.

#### 10.2.2 Geometry result (job 292397)

The completed train-set report covers all 49,251 examples. Comparison against the same `v4` plain-R3 reference:

| diagnostic | v4 baseline | collaborative InfoNCE |
|---|---:|---:|
| entropy effective rank | 21.3 | **111.0** |
| participation ratio | 5.5 | **32.2** |
| PC1 variance fraction | 0.380 | **0.112** |
| components for 50% variance | 2 | **13** |
| components for 90% variance | 72 | **269** |
| random-pair cosine | 0.885 | **0.359** |
| cosine to global mean | 0.941 | **0.599** |
| same-user pair cosine | 0.911 | **0.818** |
| same-target pair cosine | 0.892 | **0.492** |
| user adjusted η² | 0.173 | **0.476** |
| target adjusted η² | 0.057 | **0.156** |
| CF-cluster adjusted η² | 0.018 | **0.105** |
| unique-vector fraction (fp16) | 0.929 | 0.916 |

The important quantities are the gaps relative to unrelated pairs, not the absolute same-group cosine. Under baseline R3, same-user (0.911) and same-target (0.892) pairs are barely separated from random pairs (0.885). With collaborative InfoNCE, the random reference falls to 0.359 while same-user pairs remain at 0.818 and same-target pairs at 0.492. The adjusted group-effect statistics agree: user structure rises by 0.303 and target structure by 0.099. Because the geometry report measures raw `r`, not the auxiliary projection-head outputs, this organization is present in the deployed latent itself rather than existing only inside a discarded training head.

The user relation is much stronger than the target relation. This is plausible because repeated samples from one user share stable preference/history information, while different users reaching the same next item can have heterogeneous histories and reasons. It also suggests that equal auxiliary weights do not imply equal realized structure; whether the weaker target alignment should be strengthened is an empirical question, not something justified by geometry alone.

There is one mixed diagnostic: the exact-vector unique fraction decreases slightly (0.929 → 0.916), and the largest repeated template remains 133 samples. Thus the objective greatly improves global rank and identity organization without eliminating every exact duplicate. This is not a reason to reject the result, but it prevents the stronger claim that all forms of geometric collapse disappeared.

#### 10.2.3 Causal-intervention result (job 296325)

All eight E1 ranking interventions completed on the held-out test split (`n=6,157`):

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0538 | 0.0783 | 0.0936 | **0.1178** | 0.0538 | 0.0680 | 0.0743 | **0.0820** |
| mask | 0.0383 | 0.0588 | 0.0707 | 0.0893 | 0.0383 | 0.0501 | 0.0549 | 0.0609 |
| mean | 0.0551 | 0.0789 | 0.0937 | 0.1173 | 0.0551 | 0.0687 | 0.0747 | 0.0823 |
| shuffle_user | 0.0542 | 0.0788 | 0.0927 | 0.1166 | 0.0542 | 0.0684 | 0.0741 | 0.0818 |
| shuffle_target | 0.0542 | 0.0788 | 0.0927 | 0.1166 | 0.0542 | 0.0684 | 0.0741 | 0.0818 |
| noise_0.25 | 0.0534 | 0.0776 | 0.0929 | 0.1155 | 0.0534 | 0.0674 | 0.0737 | 0.0809 |
| noise_0.5 | 0.0541 | 0.0776 | 0.0936 | 0.1156 | 0.0541 | 0.0676 | 0.0742 | 0.0813 |
| noise_1.0 | 0.0538 | 0.0775 | 0.0929 | 0.1152 | 0.0538 | 0.0675 | 0.0737 | 0.0809 |

Paired bootstrap 95% intervals use 2,000 resamples with seed 42. Delta is `raw − variant`, so a positive value means the intervention hurts ranking:

| mode | HR@10 Δ | 95% CI | NDCG@10 Δ | 95% CI | significant? |
|---|---:|---|---:|---|:---:|
| **mask** | **+0.0284** | **[+0.0227, +0.0339]** | **+0.0212** | **[+0.0177, +0.0248]** | **yes, both** |
| mean | +0.0005 | [−0.0015, +0.0023] | −0.0002 | [−0.0012, +0.0007] | no |
| shuffle_user | +0.0011 | [−0.0010, +0.0032] | +0.0002 | [−0.0009, +0.0012] | no |
| shuffle_target | +0.0011 | [−0.0010, +0.0032] | +0.0002 | [−0.0008, +0.0012] | no |
| noise_0.25 | +0.0023 | [+0.0007, +0.0039] | +0.0011 | [+0.0003, +0.0019] | yes, both |
| noise_0.5 | +0.0021 | [+0.0005, +0.0039] | +0.0008 | [−0.0001, +0.0015] | HR@10 only |
| noise_1.0 | +0.0026 | [+0.0005, +0.0047] | +0.0011 | [+0.0000, +0.0023] | yes, both |

The dominant signal is **activation dependence, not identity-content dependence**. Zeroing the thought vector causes a large, highly significant drop (24.1% relative HR@10 and 25.8% relative NDCG@10). However, replacing each sample's latent with the global train mean or another sample's latent is statistically neutral on both metrics. Thus the decoder requires a nonzero latent in the thought slot but does not measurably require the correct user's or target's latent content.

The small noise effects do not reverse that conclusion. Gaussian perturbations reduce HR@10 by only 0.0021–0.0026, are not monotonic with scale, and are roughly an order of magnitude smaller than the mask effect. They show local sensitivity to moving the vector off its learned manifold, not sensitivity to which valid sample-specific reasoning vector is supplied. The natural on-manifold controls—mean and shuffling—remain the stronger test of identity-specific content, and both are null.

This result cleanly separates **representation organization** from **decoder utilization**. Collaborative InfoNCE raises effective rank from 21.3 to 111.0 and strongly increases user/target group structure in raw `r`, so the auxiliary loss did shape the deployed latent rather than only its projection heads. Nevertheless, the recommendation decoder can obtain essentially the same ranking from a global or mismatched latent. A plausible mechanism is that the contrastive gradients organize directions used by the training-only projection heads while the decoder continues to rely mainly on a shared activation/scaffold component; the present experiment supports this explanation but does not uniquely prove it.

**Final verdict: collaborative InfoNCE partially succeeds as representation learning but fails the causal latent-quality objective on CDs and Vinyl.** It preserves baseline ranking and produces the healthiest global geometry measured among plain-R3 variants, yet neither ranking improvement over baseline nor causal use of the correct sample-specific content is established. This is another strong counterexample to using geometry as the success criterion: even geometry explicitly organized by user and target identity can be causally interchangeable to the downstream decoder. The cross-dataset Toys and Musical Instruments replications should determine whether this failure pattern is stable, but the current CDs result does not justify claiming that collaborative InfoNCE improves recommendation through better-used latent reasoning.

Artifacts: `analysis/results/latent_distribution/latent_collaborative_infonce/train_latent_distribution.json`, `analysis/results/intervention/latent_collaborative_infonce/`, and `output_dir/CDs_and_Vinyl_R3_collaborative_infonce/latent/final_result.json`.

<a id="attempt-11"></a>
## 11. Attempt 11: SASRec-similarity-weighted collaborative InfoNCE

### 11.1 Approach

Attempt 10 defined collaborative structure through **exact metadata matches**: a positive had to share the anchor's user or its target item. This is a hard, binary notion of relatedness. It ignores that two different users with near-identical recent histories are collaboratively related, and that two different but frequently co-consumed items (e.g. two string sets of the same gauge family) should also encode compatible item-property reasoning. It also gives every matching positive the same importance.

Attempt 11 keeps the plain-R3 model, the FIFO bank, the correctness gate, the two training-only projection heads, and the inference path of Attempt 10 unchanged. Only the **pair definition and weighting** change: both relations now come from a frozen SASRec trained on the same split (`Verifiable-Rec/code/baselines/results/sasrec/Musical_Instruments_5_2015-1-2018-11`, test NDCG@10 = 0.0802, HR@10 = 0.0978).

#### 11.1.1 Two soft collaborative relations

For every training row $i$, SASRec supplies two frozen, L2-normalized vectors:

- $h_i\in\mathbb{R}^{128}$: SASRec's final hidden state after reading the row's history (last 10 items). This is exactly the representation SASRec scores the next item with, so similar $h$ means *similar behavioral sequence*, whether or not the users are the same.
- $e_{t_i}\in\mathbb{R}^{128}$: SASRec's item embedding of the row's target item $t_i$. Similar $e$ means *collaboratively similar target items*; the same item has similarity 1.

The two relation similarities between anchor $i$ and candidate $j$ are

$$
S^{\mathrm{seq}}_{ij}=h_i^{\top}h_j,
\qquad
S^{\mathrm{item}}_{ij}=e_{t_i}^{\top}e_{t_j}.
$$

They replace Attempt 10's relations one-for-one: **same user → similar sequence**, **same target → similar target item**.

#### 11.1.2 Pair selection

For relation $a\in\{\mathrm{seq},\mathrm{item}\}$, the positive set is the top-$k$ ($k=16$) candidates by $S^a_{ij}$ among those with $S^a_{ij}\ge\delta_{+}$ ($\delta_{+}=0.3$), excluding the anchor's own stale bank entries. A negative must be dissimilar under **both** relations:

$$
N_i=\{j:\;S^{\mathrm{seq}}_{ij}<\delta_{-}\;\land\;S^{\mathrm{item}}_{ij}<\delta_{-}\},\qquad \delta_{-}=0.2,
$$

subsampled to at most 256. This is the soft analogue of Attempt 10's AND rule: a sequence-neighbor is never pushed away by the item loss and vice versa. The gap $\delta_{-}<\delta_{+}$ leaves an ambiguous band (0.2–0.3) that is neither positive nor negative.

The thresholds come from the measured SASRec geometry on the Musical Instruments train set (66,500 rows, 5,010 catalog items, 0 unknown targets; 4,000 anchors vs. a 10,000-row pool):

| space | random-pair mean | p50 | p90 | p99 | top-1 mean | top-16 mean |
|---|---:|---:|---:|---:|---:|---:|
| sequence $h$ | 0.054 | 0.047 | 0.247 | 0.453 | 0.783 | 0.576 |
| target item $e$ | 0.041 | 0.036 | 0.199 | 0.366 | 0.924 | 0.537 |

So $\delta_{-}=0.2$ is roughly the random-pair p90 (most of the bank remains available as negatives), while $\delta_{+}=0.3$ is above p90 of random pairs and well below the typical 16th-nearest-neighbor similarity (~0.55), so top-16 positives are usually available.

#### 11.1.3 Similarity-weighted InfoNCE

With the same projection heads as Attempt 10 ($q^{a}_i=\operatorname{normalize}(g_a(r_i))$, $s^a_{ij}=q^{a\top}_i q^a_j/\tau$, $\tau=0.1$), each selected positive $p$ gets its own InfoNCE term against the shared negatives,

$$
\ell^{a}_{ip}=-\log\frac{\exp(s^a_{ip})}{\exp(s^a_{ip})+\sum_{n\in N_i}\exp(s^a_{in})},
$$

and is weighted by its SASRec similarity $w^a_{ip}=\max(S^a_{ip},0)^{\gamma}$ ($\gamma=1$):

$$
\mathcal{L}^{a}=\frac{\sum_{i}\sum_{p\in P^a_i} w^a_{ip}\,\ell^a_{ip}}{\sum_{i}\sum_{p\in P^a_i} w^a_{ip}},
\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+0.1\,\mathcal{L}^{\mathrm{seq}}+0.1\,\mathcal{L}^{\mathrm{item}}.
$$

Two design choices differ from Attempt 10's multi-positive form:

- **Per-positive terms instead of one log-sum over positives.** Attempt 10's $-\log\sum_p\exp(s_{ip})/\ldots$ can be satisfied by the single easiest positive; with graded relatedness the per-positive form makes each neighbor count in proportion to its weight, and positives do not compete with one another in the denominator.
- **Micro-batch-level weight normalization.** Normalizing by the total weight over all anchors (not per anchor) means a closer neighbor of one anchor matters more than a marginal one, *and* an anchor with genuinely close neighbors contributes more than an anchor whose best neighbors barely pass $\delta_{+}$.

#### 11.1.4 Implementation

- `src/latent/sasrec_pair_features.py`: loads the Cornac pickle on CPU (it was saved from CUDA), encodes each CSV row's history with `SASRecModel._encode`, and returns normalized sequence vectors, target item indices, and the normalized item table. Running it as a script prints the similarity table above.
- `src/latent/sasrec_collaborative_dataset.py`: extends Attempt 10's dataset with `sasrec_sequence` (128-d) and `sasrec_item` per row.
- `src/latent/sasrec_collaborative_attention.py::SASRecCollaborativeLatentModel`: plain R3 plus `sequence_projection` / `item_projection` heads, a non-persistent FIFO storing `(r, h, target item, user, sample)`, and the frozen item table as a non-persistent buffer. The correctness gate (`token_accuracy = 1.0`) and detached bank are unchanged from Attempt 10.
- `src/latent/latent_attention_train_sasrec_collaborative.py`: trainer; logs `sequence_/item_infonce_loss`, `*_positive_rate`, and the mean SASRec similarity of the selected positives (`*_positive_similarity`).
- `tests/test_sasrec_collaborative_attention.py`: FIFO order, similarity weighting, top-$k$ selection, and pair masks (self exclusion, both-relation negatives).
- Run: `scripts/R3_attempt11_sasrec_collab_musical_full.sbatch` (Musical Instruments; overrides via `MIN_POSITIVE_SIMILARITY`, `MAX_NEGATIVE_SIMILARITY`, `SIMILARITY_WEIGHT_POWER`, `EXCLUDE_SAME_USER_POSITIVES`).

At inference nothing changes: the saved checkpoint is evaluated by the ordinary `LatentModel` path; SASRec, the heads, and the bank are training-only.

#### 11.1.5 What would count as success

The Attempt 10 lesson applies directly: better-organized geometry is not enough. The decisive evidence is the causal-intervention suite — `mean` and `shuffle_*` replacements must significantly hurt ranking relative to `raw` — together with ranking at least matching the plain-R3 Musical Instruments baseline. A natural additional diagnostic is whether raw-$r$ cosine correlates with SASRec similarity ($h$ and $e$) on held-out pairs, which directly tests whether the soft CF structure transferred into the deployed latent.

### 11.2 Experiment results

#### 11.2.1 Run and ranking result (job 304495)

Training completed (3 epochs, 7.7 h). Both relations were active throughout: once the FIFO filled (10,000 entries during epoch 2), ~100% of anchors had sequence positives and ~99% had item positives; over the last 50 logged steps the selected positives had mean SASRec similarity 0.59 (sequence) and 0.58 (item), and ~17% of samples passed the correctness gate. Final losses: CE 0.23 (train), validation loss 0.504 → **0.402** → 0.408 over epochs 1–3; sequence InfoNCE fell to ~1.2 while item InfoNCE stayed ~2.9, mirroring Attempt 10's stronger-user/weaker-target asymmetry.

Standard full-beam test ranking (`n=8,313`), with paired bootstrap (2,000 resamples, seed 42) of Attempt 11 minus each reference:

| run | HR@10 | NDCG@10 | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---:|---:|---|---|
| **Attempt 11** | **0.1208** | **0.0953** | | |
| plain R3 v0 | 0.1209 | 0.0941 | −0.0001 [−0.0042, +0.0042] | +0.0012 [−0.0012, +0.0036] |
| plain R3 v1 | 0.1226 | 0.0950 | −0.0018 [−0.0063, +0.0023] | +0.0002 [−0.0021, +0.0025] |
| Attempt 10 | 0.1208 | 0.0937 | +0.0000 [−0.0042, +0.0041] | +0.0016 [−0.0009, +0.0039] |
| Attempt 10 rerun | 0.1184 | 0.0945 | +0.0024 [−0.0020, +0.0067] | +0.0008 [−0.0018, +0.0031] |
| plain SFT | 0.1009 | 0.0790 | +0.0198 [+0.0149, +0.0249] | +0.0162 [+0.0132, +0.0193] |

Every difference from plain R3 and Attempt 10 is inside its confidence interval, and the two Attempt 10 seeds differ from each other by about as much (0.0024 HR@10). **Attempt 11 preserves but does not improve ranking.**

Two observations bound what this objective could achieve:

- **The teacher is weaker than the student.** On the same test rows SASRec's exported top-10 (history excluded) reaches HR@10 = 0.0921, versus 0.1208 for Attempt 11. Hit overlap: both 635, R3-only 369, SASRec-only 131; the oracle union is 0.1365. SASRec therefore carries some complementary signal (131 rows), but pair selection only transfers its *neighborhood structure*, not those hits.
- **The pair loss reaches the decoder only indirectly.** As in Attempt 10, the InfoNCE gradients shape `r` through training-only projection heads, while Attempt 10's intervention study showed the decoder barely uses `r`'s sample-specific content. Changing which pairs are positive does not change that bottleneck; the causal-intervention suite has not yet been run for Attempt 11.
