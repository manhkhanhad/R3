# Latent Verification

## Objective

Determine whether generated latent reasoning `r` contains *useful sample-specific information*, then train a verifier that can identify or improve such reasoning without rewarding the history-to-decoder shortcut. This work builds on [Attempt 10](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10), whose collaborative InfoNCE improves latent geometry but not demonstrated causal use of the correct latent.

The reference [VRec paper](https://arxiv.org/html/2603.07725v1) motivates reason–verify–recommend, verifier pre-training, and adaptive feedback. Its group-preference prediction and entropy-based confidence are not directly sufficient here: an R3 decoder can recommend correctly from the raw history while ignoring the identity of `r`. Correctness alone therefore cannot label a latent as useful.

## Proposed model design (not implemented)

1. **Generate a counterfactual corpus.** For each training history `x` and target `y`, compare the decoder under its own generated latent `r(x)` with a train-set mean latent and several independently sampled, matched-scale training latents. Keep `x`, `y`, and decoder weights fixed within each comparison. Record target-token NLL, exact-token correctness, and, where affordable, the target's recommendation rank. A usable label is `NLL(x, y, r_control) - NLL(x, y, r(x))`; positive means the generated latent helped relative to that control. Require consistent benefit across multiple controls rather than treating one favorable draw as proof.
2. **Pre-train a conditional verifier.** Let `V(x, r)` estimate counterfactual utility or the probability of a meaningful positive utility margin. Use pairwise comparisons among latents evaluated on the *same* history, with a calibrated regression or ranking objective. At inference, `V` receives neither the target `y`, outcome labels, decoder logits computed with `y`, nor stored example IDs. It must not predict only sample difficulty, latent norm, or whether the decoder's answer happened to be correct.
3. **Introduce adaptive verification cautiously.** First test a bounded gate between `r(x)` and a fixed baseline latent, with a frozen decoder and verifier. If the verifier genuinely identifies helpful latents, test a small norm-controlled correction `r* = r + gate(x,r) · correction(x,r)`; use stop-gradient or alternating updates and regularize the correction against arbitrary activation-scale changes. The correction must have no access to `y` at inference. Do not jointly optimize an unrestricted verifier, correction, and decoder from the start: they could collude through another shortcut.
4. **Retain the causal acceptance test.** On held-out histories, require positive paired `raw − mean` and `raw − shuffle` HR/NDCG effects, with uncertainty intervals, while preserving overall ranking. Verifier accuracy, latent geometry, and a mask penalty are supporting diagnostics, not substitutes. Also check whether a verifier-selected/corrected latent beats the unverified latent *and* the mean baseline on the same examples.

The training split should supply verifier labels; validation selects thresholds and architecture; test remains untouched until final evaluation. Cross-fit or use a held-out label-generation split if a learned verifier is evaluated on the same generated examples used for its training. Record checkpoint hashes because the on-disk checkpoint can differ from previously published results.

## Feasibility check — 2026-09-24

### Setup

- Dataset: CDs and Vinyl. Current on-disk Attempt-10 checkpoint: `output_dir/CDs_and_Vinyl_R3_collaborative_infonce/latent`, SHA-256 `f078edc5c68a2cd6d74a4b061b6d79359c8926f9475a9d70fbdcf427b6c54edd`.
- A fresh extraction produced 2,048 training latents to estimate the mean and supply shuffle controls, plus the first 512 validation examples for paired target-token NLL interventions. The train mean here is a *subset estimate*, not the full-corpus mean. `src/prototype_grounding/experiment.py` computed per-example mean supervised-token NLL; it did not run a new ranking evaluation.
- Artifacts: `analysis/evidence/CDs_and_Vinyl/latent_verification_feasibility/{train,valid}/` and `analysis/results/intervention/latent_verification_feasibility/valid_summary.{json,npz}`. The intervention's cached validation latents use fp16, as in the existing pipeline. Bootstrap intervals in the summary use 2,000 paired resamples, seed 42.
- Important provenance caveat: the earlier Attempt-10 E1 ranking artifacts report checkpoint SHA-256 `4750fc27ddc42dec86def25d45701d186fee30c3f90d933d33723e6f55d574e2`. They describe a *different checkpoint file* and must not be pooled with the fresh NLL results.

### Model-backed validation NLL

The table uses `control NLL − raw NLL`; positive means the sample's own generated latent helps. Raw mean NLL is **1.2530** across 512 validation examples.

| Control | Mean NLL | Mean gain | 95% paired CI for gain | Gain > 0 | Gain > 0.01 |
|---|---:|---:|---:|---:|---:|
| Zero mask | 2.9353 | +1.6822 | [+1.6167, +1.7494] | 512/512 | 512/512 |
| Train-subset mean | 1.2529 | −0.00017 | [−0.00095, +0.00062] | 245/512 | 57/512 |
| Shuffle user control | 1.2526 | −0.00046 | [−0.00120, +0.00032] | 240/512 | 53/512 |
| Shuffle target control | 1.2528 | −0.00020 | [−0.00100, +0.00062] | 263/512 | 52/512 |

The own latent beats **both** the mean and the first shuffled control by more than 0.01 NLL on only **22/512 (4.3%)** examples; it loses to both by more than 0.01 on 21/512. The per-example gain relative to the mean has standard deviation 0.0090, median −0.00012, and maximum +0.0417 NLL. Threshold 0.01 is descriptive, not a calibrated semantic-success threshold. Small fp16/model-evaluation differences could affect near-zero labels; a reproducibility check with repeated controls and an effect-size margin is necessary before verifier training.

### Existing held-out ranking, separate checkpoint

The earlier E1 intervention records allow a paired sample-level check on 6,157 test examples, all with the *older* `4750…` checkpoint hash. Relative to the mean control, raw changes the recorded target rank for 184/6,157 (3.0%) examples: raw alone hits @10 in 20 cases, while mean alone hits @10 in 17. HR@10 difference is +0.00049, with a paired 2,000-resample bootstrap interval of approximately [−0.00146, +0.00260]. Relative to `shuffle_user`, raw alone hits @10 in 26 cases and shuffle alone in 19. By contrast, raw beats the zero mask at @10 in 248 cases while mask beats raw in 73. This reproduces the activation-dependence-versus-identity-dependence distinction in [Attempt 10](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10).

### Interpretation and next decision

**Observed:** The current checkpoint's correct latent has a large benefit over a zero vector but essentially no mean benefit over valid-scale, mismatched latents on this validation sample. A small tail of favorable counterfactuals exists, but many comparably sized adverse cases exist too. The older checkpoint's held-out rankings likewise contain few identity-sensitive cases.

**Inference:** Simply labeling latents “good” when the decoder answers correctly would primarily teach the verifier the existing shortcut. The current generated-latent corpus does not yet show a robust, abundant *causal-utility* signal for verifier pre-training. This does not prove that no useful latent exists; NLL and top-10 rank capture different effects, the fresh NLL sample is only 512 examples, and the mean control is estimated from 2,048 training examples.

**Next gate before model implementation:** Repeat matched-checkpoint counterfactual evaluation with a full train mean, multiple independent shuffled controls, and a larger train/validation sample; measure stable positive and negative label rates at preregistered margins. If robust positives remain rare, first generate stronger candidate latents (for example, bounded perturbations or a separately trained correction) and verify that their benefit survives held-out, on-manifold controls. Treat target-aware latent optimization as a diagnostic upper bound only, never as an inference-time input or unqualified verifier label.

## Original R3 phenomenon: CDs and Vinyl v4, last epoch — 2026-09-24

### Why this checkpoint

The preceding feasibility run used an Attempt-10 checkpoint, which cannot establish the phenomenon in the *original* R3 model. This analysis instead uses the unmodified v4 SFT checkpoint at `~/scratchDirectory/R3/output_dir/CDs_and_Vinyl_v4/latent/checkpoint-1152` (epoch-end step 1152), SHA-256 `395111175b0cc904e999642354c48690f56eb1d7f2ca21e481f4e3163b366e04`. The literal `~/scratchDirectory/R3/output` directory is empty; the checkpoint is under `output_dir` in that scratch tree. The hash matches both the earlier full 49,251-example train-latent extraction and all existing v4 step-1152 E1 ranking records. This is *not* the later `nomul_attn_lr_5e-5` GRPO checkpoint.

New model-backed extraction and paired interventions cover the first **1,024 validation** examples. They use the full, checkpoint-matched train mean and two distinct deterministic random train-latent controls. Each condition retains the same history and target. Per-example NLL is mean supervised target-token cross-entropy. Artifacts: `analysis/evidence/CDs_and_Vinyl/latent_v4_checkpoint_1152/valid/` and `analysis/results/intervention/latent_v4_checkpoint_1152/valid_summary.{json,npz}`. The intervention script stores extracted latents in fp16, then evaluates them in the model's bf16 thought slot. Paired CIs use 2,000 bootstrap resamples.

### Target-likelihood response on validation

`Gain = control NLL − own-r NLL`; positive means the original generated latent helps. Own-r mean NLL is **1.34008**.

| Control | Mean NLL | Mean gain (95% paired CI) | Gain > 0 | Gain > 0.01 | Gain < −0.01 |
|---|---:|---:|---:|---:|---:|
| Zero mask | 2.33668 | +0.99660 [+0.95085, +1.04429] | 972/1,024 | 969/1,024 | 48/1,024 |
| Full-train mean | 1.33962 | −0.00045 [−0.00136, +0.00043] | 493/1,024 | 149/1,024 | 172/1,024 |
| Random train latent A (`shuffle_user` in E1) | 1.33978 | −0.00029 [−0.00126, +0.00066] | 494/1,024 | 168/1,024 | 198/1,024 |
| Random train latent B (`shuffle_target` in E1) | 1.33983 | −0.00025 [−0.00122, +0.00066] | 499/1,024 | 154/1,024 | 177/1,024 |

Only **75/1,024 (7.3%)** beat *both* the full-train mean and random control A by more than 0.01 NLL; **101/1,024 (9.9%)** lose to both by more than 0.01. These descriptive counts are not stable good/bad verifier labels yet: 0.01 is an arbitrary threshold, each random control is one draw, and near-zero differences may be sensitive to numerical precision. The mean-control per-example gains have standard deviation 0.01497, versus a mean of −0.00045, so aggregate neutrality masks small positive and negative effects.

Simple geometry does not separate those effects. On validation, Spearman correlations of `mean NLL − own-r NLL` with latent norm, cosine to the full-train mean, and L2 distance to that mean are approximately **+0.085, +0.016, and −0.008**, respectively. Correlation with the zero-mask NLL gain is approximately **−0.030**. Thus distance from the mean and dependence on a nonzero thought are poor proxies for whether the *correct* thought helps. These are exploratory associations, not a trained verifier test.

### Held-out ranking and output changes

The previously completed E1 records provide paired raw/mean/shuffle/mask rankings for **6,157 test** examples at this exact hash. Re-analysis at the per-example level gives:

| Control | Target rank changes vs own `r` | Top-1 item changes | Raw-only HR@10 hits | Control-only HR@10 hits | Raw − control HR@10 (95% CI) |
|---|---:|---:|---:|---:|---:|
| Full-train mean | 97/6,157 (1.6%) | 348/6,157 (5.7%) | 11 | 11 | +0.00000 [−0.00162, +0.00162] |
| Random train latent A | 111/6,157 (1.8%) | 365/6,157 (5.9%) | 13 | 7 | +0.00097 [−0.00049, +0.00227] |
| Zero mask | 617/6,157 (10.0%) | 4,688/6,157 (76.1%) | 302 | 68 | +0.03801 [+0.03183, +0.04418] |

The mean replacement changes the full ordered top-10 prediction list for **4,420/6,157 (71.8%)** cases, even though the target rank changes in only 97. This is an important qualification: `r` **does affect outputs**, particularly the ordering or membership of non-target candidates, but swapping its sample identity rarely changes *task-relevant target ranking*. A changed list is not automatically an improvement; without candidate-score margins, some changes could reflect near-ties. The target's hit@10 changes only 22 times under the mean control (11 wins each way).

The small **97/6,157** target-rank-change fraction needs a denominator qualification: **5,448/6,157 (88.5%)** targets are absent from raw top-10, and the E1 record assigns every absent target the same sentinel rank (`1,000,000,000`). Their *true* rank beyond ten is not measured. For the **709** targets actually found in raw top-10, mean replacement changes recorded rank in **86 (12.1%)**. Conditional on raw target position, changes are **5/326 (1.5%)** at rank 1, **40/243 (16.5%)** at ranks 2–5, and **41/140 (29.3%)** at ranks 6–10. A further 11 previously absent targets enter the top-10, while 11 leave it. Thus well-ranked targets are more stable; targets near the lower top-10 ranks are less stable, even though aggregate HR@10 is exactly unchanged.

List churn is also weighted toward lower positions: under mean replacement, the predicted item changes at position 1 for **348/6,157 (5.7%)** cases, versus **2,278/6,157 (37.0%)** at position 9. The average overlap of the two top-10 *sets* is **9.46 items**, so much of the 71.8% ordered-list change is reordering or replacement of only one item. This pattern is compatible with small score perturbations around close candidates, but the stored ranking records do not contain the full candidate-score margins needed to establish that mechanism.

### Comment and implication for verification

The evidence supports **activation dependence plus weak target-aligned identity dependence**, not the absolute claim that `r` has no effect. The zero vector is strongly harmful; valid-scale substitutions leave mean target likelihood and held-out target ranking nearly unchanged. Small helpful and harmful subsets exist, so an adaptive verifier may still be appropriate, but these effects are not identified by basic norm or mean-distance diagnostics.

The analysis does **not** compare the mathematical distribution of latent vectors with the full output-logit distribution. NLL tests the observed target tokens; E1 tests constrained top-10 generation. The stronger hypothesis—that latent directions are poorly aligned with decision-relevant output directions—remains an inference, not a proven distributional fact. A next diagnostic would measure first-target-token logit changes or output-distribution divergence under own/mean/shuffle latents, then ask whether those changes improve the target's probability rather than merely reshuffle alternatives.

For verifier design, separate (i) whether `r` faithfully represents user/target information, (ii) whether it changes the output distribution, and (iii) whether that change improves the recommendation. Correct output alone, healthy latent geometry alone, and output change alone each fail to establish (iii). Train and calibrate an adaptive verifier only after obtaining repeatable, same-history counterfactual labels or an independently validated semantic-alignment target.

## Candidate replacement and target-free selection — 2026-09-24

### Question and protocol

Does replacing v4's own latent with a *better* latent improve recommendation, and can an inference-time selector find it? First, we computed answer-aware **oracle upper bounds** among the existing raw, mean, and two shuffled controls on the full 6,157-example v4 test ranking. Next, `scripts/latent_verification_candidates.py` scored a larger train-latent candidate set on the first 256 validation histories, keeping each history, target, and decoder fixed. Each history had its own latent, the full-train mean, four random train latents, up to four same-user train latents, and up to four same-target train latents. Same-user candidates exist for 184/256 histories; same-target candidates exist for 254/256. Same-target retrieval **uses the answer** and is only a diagnostic oracle pool, not an inference-time mechanism.

Candidate selection by lowest target NLL uses the answer and is also an oracle, not a trained verifier. To test whether those gains are predictable without the target, `scripts/latent_verification_probe.py` cross-fit a standardized ridge-regression selector over five user-grouped folds. Its inputs were candidate source (mean/same-user/random), own and candidate norms, their distance/cosines to each other and the train mean, and history length. **No target ID, target NLL, same-target candidate, or outcome label enters selector inference.** This is a minimal geometry-based selector, not a semantic or learned-history verifier.

Artifacts: `analysis/results/latent_verification/latent_v4_checkpoint_1152_candidates_n256_s42.{json,npz}` and `analysis/results/latent_verification/latent_v4_checkpoint_1152_candidates_n256_s42_probe.{json,npz}`. Both use the original v4 checkpoint hash `39511117…`. Recomputed own-latent NLL matches the earlier validation NLL exactly for these 256 examples (`max absolute difference = 0`).

### Results

| Diagnostic | Own latent | Answer-aware best | Difference |
|---|---:|---:|---:|
| Test HR@10, raw/mean/two shuffles, 6,157 examples | 0.11515 | 0.11743 | +0.00227; 14 previously missed targets recovered |
| Test NDCG@10, same candidate pool | 0.08058 | 0.08254 | +0.00196 |
| Validation target NLL, all expanded candidates, 256 examples | 1.28400 | 1.27130 | −0.01270 |
| Validation target NLL, inference-available expanded pool only | 1.28400 | 1.27294 | −0.01106 |

The test ranking oracle is an *upper bound for that small existing candidate pool*: it chooses the best recorded target rank separately for each example and cannot be implemented without the target. The expanded-bank oracle uses NLL rather than recommendation rank, so its 0.01270 gain does **not** imply a corresponding HR/NDCG gain. Best-of-four target-NLL gains over the own latent are **0.00735** for same-target candidates on 254 available histories and **0.00787** for random candidates on all 256; this does not show a special benefit from same-target retrieval. Best-of-four gains favor selecting the minimum of several noisy alternatives, so they must not be interpreted as ordinary candidate quality.

The five-fold, user-held-out geometry selector chooses a non-own latent for **211/256** histories when its predicted NLL gain exceeds zero. Its measured mean gain is only **+0.00096 NLL** relative to always using the own latent, with paired bootstrap 95% CI **[−0.00034, +0.00246]**; 36 selections help by more than 0.01 while 29 hurt by more than 0.01. A conservative predicted-gain threshold of 0.01 selects only one case and yields negligible aggregate gain. Thus this simple selector does **not** establish reliable adaptive verification, despite an answer-aware oracle ceiling of +0.01106 NLL in its permitted pool.

### Interpretation

An oracle can find alternative latents that slightly lower target NLL, but replacing `r` with a supposedly “good” latent does **not** ensure better recommendation. The ranking oracle's ceiling among the existing controls is only 14 additional hits in 6,157 examples, and the target-free selector does not recover a statistically established NLL gain. The present test does not prove that a richer verifier cannot work: the candidate pool is small, the selector sees only simple geometry/history length, and the cross-fit sample is 256 histories. It also does not test ranking for the newly retrieved candidates. The next gate is to learn preference-alignment signals or a stronger history-conditioned verifier on a larger training corpus, then evaluate its selected latents on *paired held-out recommendation rank*, with the target excluded from selection.

## Same-user positive refinement ranking gate — 2026-09-24

Before training a verifier or corrector, test whether moving a latent toward a related, training-correct latent can improve **held-out constrained recommendation rank** with the decoder frozen. The exploratory jobs use the original v4 checkpoint (`checkpoint-1152`, hash checked against cached thoughts), not the modified Collaborative Reasoning model. They start from the first 256 validation histories and their four pre-sampled same-user train candidates. Each candidate's teacher-forced target-token accuracy is measured under its own train history; eligibility requires at least 0.8, matching the agreed queue gate. This is a proxy for prediction correctness, **not** proof that the latent caused the correct tokens. The 64-history pilot picks the highest train accuracy (then lowest train NLL); the broader run picks the highest cosine similarity to the validation latent among eligible candidates. Neither selection uses the validation target.

For each fixed history, constrained 10-beam catalog generation compares its own latent, the selected same-user latent, norm-matched 0.25 and 0.50 interpolations toward that latent, a correctness-gated different-user latent, and the global train mean. The model and history stay fixed. Report HR/NDCG, paired raw-only versus intervention-only top-10 hits, and the answer-aware best-per-history rank as a *diagnostic ceiling*, never an inference policy. A gain for same-user blends over both raw and different-user/mean controls would justify developing a target-free verifier or bounded correction; absent a gain, related positive alignment alone is insufficient. This small, selected sample is exploratory and requires larger validation with uncertainty estimates before any efficacy claim.

Implementation: `scripts/latent_verification_refinement_oracle.py`; launcher: `scripts/latent_verification_refinement_oracle.sbatch`. Pilot Slurm job **302935** and broader job **302954** both completed successfully. Artifacts: `analysis/results/latent_verification/latent_v4_checkpoint_1152_refinement_oracle_n64_s42.json` and `analysis/results/latent_verification/latent_v4_checkpoint_1152_refinement_oracle_similar_n256_s42.json`.

| Broader similarity-selected run, 181 eligible histories | HR@10 | NDCG@10 | Target entered/leaves top 10 vs raw |
|---|---:|---:|---:|
| Own latent | 0.1160 | 0.0742 | — |
| Same-user token-accurate latent | 0.1160 | 0.0727 | 0 / 0 |
| Norm-matched 0.25 blend | 0.1160 | 0.0723 | 0 / 0 |
| Norm-matched 0.50 blend | 0.1160 | 0.0712 | 0 / 0 |
| Different-user token-accurate latent | 0.1105 | 0.0721 | 0 / 1 |
| Full train mean | 0.1105 | 0.0720 | 0 / 1 |
| Answer-aware best among own, same-user, blends, and mean | 0.1160 | 0.0746 | Oracle ceiling, not a policy |

The accuracy-first 64-history pilot also found **zero** target entries or exits at top 10 for same-user replacement and both blends. In the broader run, the selected same-user latents average 0.919 teacher-forced token accuracy, but replacing with them changes the recorded target rank in only 5/181 cases; the 0.25 and 0.50 blends change it in 4 and 6 cases, respectively. All these changes are *within* the top 10; none recover a missed target. The slight NDCG reductions are descriptive, not evidence of a reproducible harm. The sample is selected for candidate availability, includes only 21 raw top-10 hits, and ranks below ten are not measured. This negative result applies to four sampled **same-user** candidates on frozen original v4, not to all possible title/CF-conditioned positives or a jointly trained corrector. It does, however, weaken the immediate premise that moving toward an already token-accurate, nearby same-user latent is enough to repair v4's target-ranking utilization collapse. A verifier should not be trained on this candidate relation as if it were already demonstrated to improve recommendations.
