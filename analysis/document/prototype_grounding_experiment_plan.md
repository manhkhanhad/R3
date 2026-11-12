# Prototype Grounding for Latent Recommendation: Experimental Plan

> Follow-up diagnosis: [Latent Reasoning Collapse in R3 Recommendation Models](latent_reasoning_collapse_r3.md).

## Locked experimental scope

- Dataset: `CDs_and_Vinyl` only.
- Stage-2 checkpoint: `/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5`.
- Checkpoint architecture: `LatentModel`, hidden size 1536
- Primary metrics: HR and NDCG at `K = {1, 3, 5, 10}`.
- Current scope: single-thought analysis only.
- Multi-thought analysis: deferred until a multi-thought checkpoint is provided.
- Output policy: every derived result, log, cache, embedding, prototype, figure, and summary must be saved under `analysis/`. The supplied checkpoint remains in its existing location and is treated as read-only input.

## Progress tracker

Active single-thought protocol: **9/9 resolved (100%)** — seven stages executed; E7 and E8 stopped by the preregistered decision/stop gates.

`[█████████████████████████] 100%`

- [x] **S0 — Scope and checkpoint verification.** Confirmed `nomul_attn_lr_5e-5`, the 2018 5-core splits, metrics, hashes, and artifact policy.
- [x] **E0 — Existing Stage-1 baseline.** Recalculated HR/NDCG from the requested checkpoint's existing predictions.
- [x] **E1 — Causal thought intervention.** Zeroing harms likelihood and ranking; in-distribution replacement does not harm likelihood.
- [x] **E1b — Latent-distribution follow-up.** Confirmed severe template and dimensional collapse in the 49,251 extracted train thoughts.
- [x] **E1c — SFT-only latent-distribution comparison.** SFT is already collapsed, but RL further reduces template diversity and effective rank.
- [x] **E2 — Evidence extraction and utility scoring.** Extracted all train thoughts, scored counterfactual utility, and materialized feasible subsets.
- [x] **E3 — Prototype mining and diagnostics.** Mined all feasible banks/counts/seeds; diagnosed severe utilization collapse and near-duplicates.
- [x] **E4 — Frozen prototype grounding.** Residual grounding was null/slightly worse in NLL and ranking across all three seeds; H2 rejected.
- [x] **E5 — Collaborative-transfer test.** A 512-example exclusion screen was inconclusive; full evaluation stopped because E4 showed no gain to preserve.
- [x] **E6 — Oracle retrieval analysis.** Oracle selection improves NLL while cosine selection does not, isolating retrieval as a bottleneck.
- [x] **E7 — End-to-end Stage-2 grounding.** **Stopped by gate:** the E4 prerequisite failed; training cannot establish the central mined-evidence claim.
- [x] **E8 — Single-thought robustness and subgroup analysis.** **Stopped by the same gate:** there is no validated grounded improvement whose robustness can be tested.
- [ ] **E9 — Multi-thought grounding and depth scaling.** **Deferred; excluded from the active progress denominator.**

After every experiment, update its checkbox above, the experiment journal, and a dedicated result subsection. The subsection must contain the numerical result, a plain-language result description, a researcher comment interpreting the result, limitations, and the next action. Record failed and inconclusive runs as well as successful runs.

## 1. Goal

Validate whether latent reasoning vectors mined from a trained recommender can be converted into reusable, cross-user prototypes that improve next-item recommendation when used to ground newly generated latent reasoning.

The proposed single-thought pipeline is:

1. Load and freeze the supplied one-thought latent-reasoning recommender.
2. Extract latent reasoning vectors from training interactions as an evidence bank.
3. Estimate which evidence vectors causally help recommendation.
4. Cluster useful evidence into shared prototypes.
5. Retrieve prototypes for a new latent thought and use them to adjust that thought.
6. Fine-tune and evaluate the grounded recommender.

The primary scientific claim is not merely that a codebook can improve the model. It is that reasoning evidence shared across users contains reusable recommendation knowledge.

## 2. Research questions

### RQ1: Does the baseline actually use its latent reasoning?

Would removing, shuffling, or perturbing a thought vector reduce target-item likelihood and ranking accuracy?

### RQ2: Do mined prototypes contain reusable evidence?

Can prototypes extracted from training thoughts improve test predictions while the recommender and prototypes remain frozen?

### RQ3: Does evidence quality matter?

Are prototypes built from causally useful thoughts better than prototypes built from all or merely correct thoughts?

### RQ4: Is the knowledge genuinely collaborative?

Do prototypes supported by other users improve recommendations when all evidence from the query user is excluded?

### RQ5: Can trainable grounding improve recommendation without destroying personalization?

Does residual, confidence-gated grounding outperform hard prototype replacement and ordinary continued fine-tuning?

### RQ6: Does grounding support deeper latent reasoning?

Does grounding prevent performance degradation as the number of latent reasoning steps increases? This question is retained for the later multi-thought phase and is not part of the current experiment scope.

## 3. Hypotheses and success criteria

| ID | Hypothesis | Evidence required |
|---|---|---|
| H1 | Latent thoughts causally contribute to prediction. | Masking or shuffling thoughts significantly worsens target NLL and ranking metrics. |
| H2 | Useful-thought prototypes transfer to unseen samples. | Frozen grounding beats the raw baseline, global mean, random, and shuffled-prototype controls. |
| H3 | Prototype value depends on evidence quality. | Useful-evidence prototypes outperform all-thought and correct-only prototypes. |
| H4 | Prototypes encode cross-user knowledge. | Gains remain with query-user evidence excluded and are stronger than random-other-user evidence. |
| H5 | Mined initialization matters. | Mined prototypes outperform an equally sized randomly initialized trainable bank. |
| H6 | Residual grounding preserves personalization. | Residual grounding beats hard replacement while maintaining diverse prototype usage. |
| H7 | Grounding improves reasoning scalability. | Deferred until multi-thought checkpoints are available. |

The central idea should be considered supported only if H1-H4 hold. An improvement that appears only after end-to-end training is insufficient to establish that mined evidence itself is useful.

## 4. Definitions

For user history \(X_u\), target item \(y_u\), and reasoning step \(t\), let the Stage-1 model generate:

\[
r_{u,t}=M_\theta(X_u,r_{u,<t}).
\]

Define counterfactual evidence utility as:

\[
s(r_{u,t})=
\log P(y_u\mid X_u,r_{u,t})-
\log P(y_u\mid X_u,r^{\text{masked}}_{u,t}).
\]

A positive score means the thought increases the target item's probability. Record both this continuous score and whether recommendation rank improves.

Given prototypes \(P=\{p_1,\ldots,p_K\}\), retrieve a soft prototype:

\[
a_k=\operatorname{softmax}
\left(\operatorname{cos}(q(r,X),p_k)/\tau\right),
\qquad
\bar p=\sum_{k\in\operatorname{TopK}}a_kp_k.
\]

Use residual grounding by default:

\[
r^*=r+\alpha W(\bar p-r).
\]

The gate \(\alpha\) may initially be fixed and later learned from retrieval confidence, history, and the raw reasoning vector.

## 5. Data protocol and leakage controls

Use the existing chronological train/validation/test splits. All evidence mining, clustering, prototype fitting, and hyperparameter selection must obey these rules:

- Extract prototype evidence from the training split only.
- Fit normalization, dimensionality reduction, clustering, and retrieval indices on training evidence only.
- Use validation data to choose prototype count, retrieval size, temperature, grounding strength, and loss weights.
- Report test results once using the selected configuration.
- Never use the test target, category, CF cluster, or item identity for prototype selection.
- Target information may be used for offline evidence scoring on training samples and for explicitly labeled oracle analyses only.
- In the cross-user experiment, remove every evidence vector belonging to the query user before retrieval or prototype construction.
- Weight users uniformly during clustering so prolific users do not dominate the bank.

Use only `CDs_and_Vinyl` for the current analysis. Run at least three seeds for stochastic interventions, clustering, retrieval, and training. Other datasets require an explicit future scope change.

## 6. Stage A: Load and validate the latent-reasoning baseline

### A1. Supplied checkpoint

Do not retrain Stage 1. Use the supplied checkpoint at:

`/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5`

Verify and record:

- checkpoint architecture and parameter configuration;
- split identifiers;
- available training arguments and trainer state;
- HR/NDCG at 1, 3, 5, and 10, plus target NLL when available;
- per-sample outputs and ranks.

Do not introduce prototype components in this stage.

The existing checkpoint predictions establish the initial baseline:

| Metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| HR | 0.05262303 | 0.07601104 | 0.09030372 | 0.11060581 |
| NDCG | 0.05262303 | 0.06601825 | 0.07191042 | 0.07838422 |

These values were recalculated over 6,157 examples from the requested checkpoint's existing `final_result.json`. Catalog violation count (`CC`) was 60, matching its `metric.log`. The derived metric artifact is stored at `analysis/results/baseline_1T/metrics.json`.

### A2. Thought intervention test

Evaluate the frozen baseline under:

| Variant | Intervention |
|---|---|
| Raw | Use the original latent thought. |
| Mask | Replace the thought with zero or a learned null vector. |
| Mean | Replace it with the global training-thought mean. |
| Shuffle-user | Use a thought from another randomly selected user. |
| Shuffle-target | Use evidence associated with an unrelated target. |
| Noise | Add Gaussian noise at several signal-to-noise ratios. |
| Wrong-step | Substitute a vector from another reasoning step. |

Primary outputs are change in target NLL, target rank, HR@1/3/5/10, and NDCG@1/3/5/10.

Decision gate: do not proceed to full prototype training unless at least one destructive intervention reliably worsens target NLL and recommendation metrics.


## 7. Stage B: Build the evidence bank

For every training interaction and reasoning step, store:

- sample and user identifiers;
- reasoning-step index;
- raw latent vector;
- user-history length;
- target item identifier;
- target category and CF cluster for analysis;
- baseline target log-probability and rank;
- whether the target appears in top 1/3/5/10;
- masked-thought target log-probability and rank;
- continuous utility score \(s(r)\);
- model checkpoint and extraction configuration.

Prefer a memory-mappable tensor format for vectors and a separate tabular metadata file. Normalize vectors only after saving their original form.

### Evidence subsets

Construct the following banks:

| Bank | Membership rule |
|---|---|
| E-all | Every extracted training thought. |
| E-correct | Thoughts from samples where the target is correctly ranked. |
| E-positive | Thoughts with \(s(r)>0\). |
| E-top | Top 25% of thoughts by \(s(r)\). |
| E-harmful | Thoughts with \(s(r)<0\); negative control. |

Inspect utility distributions by dataset, user activity, item popularity, and reasoning step before clustering.

## 8. Stage C: Mine prototypes

Use L2-normalized embeddings and weighted spherical k-means as the initial method. Weight each vector by a clipped transformation of its utility score while keeping total weight per user approximately equal.

Initial sweep:

- Prototype count `K`: `{32, 64, 128, 256}`.
- Evidence bank: `{E-all, E-correct, E-positive, E-top, E-harmful}`.
- Bank sharing: `{shared across steps, separate per step}`.
- Seed: at least three clustering seeds.

For each prototype, record:

- number of interactions and distinct users;
- mean and variance of evidence utility;
- target/category/CF distributions;
- average distance to assigned evidence;
- dominant-user fraction;
- nearest other prototype similarity.

Drop or merge prototypes with insufficient distinct-user support. Flag banks with severe utilization imbalance or near-duplicate prototypes.

## 9. Stage D: Frozen grounding test

Freeze the Stage-1 model and prototype bank. Ground test thoughts without additional learning.

### Retrieval sweep

- Retrieval: hard nearest versus soft top-k.
- Top-k: `{1, 4, 8}`.
- Temperature: tune on validation data.
- Grounding strength \(\alpha\): `{0, 0.25, 0.5, 0.75, 1.0}`.

### Required controls

| Control | Purpose |
|---|---|
| Raw reasoning | Establish the Stage-1 baseline. |
| Global mean | Test generic smoothing. |
| Random vector | Test arbitrary perturbation. |
| Random training thought | Test whether any in-distribution vector helps. |
| Shuffled prototype | Test whether correct retrieval matters. |
| Random k-means centroids | Control for bank size and computation. |
| Category centroid | Compare against supervised semantic grouping. |
| CF centroid | Compare against conventional collaborative grouping. |

Primary comparison: soft residual grounding using E-top versus raw reasoning, E-all prototypes, shuffled prototypes, and category/CF centroids.

Decision gate: proceed to end-to-end Stage 2 if useful-evidence grounding improves target NLL or ranking consistently across seeds and at least one dataset, without relying on target-aware retrieval.

## 10. Stage E: Collaborative-transfer test

Repeat frozen grounding with progressively stricter evidence access:

1. All training-user evidence allowed.
2. Query-user evidence excluded.
3. Same-target evidence excluded.
4. Both query-user and same-target evidence excluded.
5. Only prototypes supported by at least `m` distinct users, with `m` swept on validation data.

Report results separately for:

- sparse, medium, and active users;
- head, torso, and long-tail target items;
- short and long histories;
- seen and rare preference categories.

If gains disappear when same-user evidence is removed, describe the mechanism as memory-based personalization rather than collaborative prototype transfer.

## 11. Stage F: Oracle analysis

For analysis only, score several retrieved prototypes using the known test target and choose the prototype yielding the highest target likelihood.

Compare:

- raw reasoning;
- similarity retrieval;
- learned retrieval;
- target-aware oracle retrieval.

Interpretation:

- No oracle gain: the bank lacks helpful reasoning directions.
- Oracle gain but no similarity-retrieval gain: retrieval is the bottleneck.
- Both improve: prototype quality and basic retrieval are supported.
- Learned retrieval approaches oracle: the end-to-end grounding mechanism is effective.

Oracle numbers must never be presented as deployable recommendation results.

## 12. Stage G: End-to-end grounding

Initialize the prototype bank from the best frozen experiment. Compare:

| Variant | Description |
|---|---|
| Continued FT | Continue training Stage 1 without grounding. |
| Hard replace | Set \(r^*=p_k\). |
| Fixed residual | Use fixed \(\alpha\) selected on validation data. |
| Learned gate | Predict \(\alpha\) from reasoning and retrieval confidence. |
| Frozen mined bank | Train the recommender and grounding layers only. |
| Trainable mined bank | Fine-tune prototypes with a small learning rate. |
| Random trainable bank | Same capacity with random initialization. |

Recommended objective:

\[
\mathcal L=
\mathcal L_{\mathrm{rec}}
+\lambda_c\mathcal L_{\mathrm{commit}}
+\lambda_u\mathcal L_{\mathrm{utility}}
+\lambda_d\mathcal L_{\mathrm{diversity}}.
\]

Suggested utility objective:

\[
\mathcal L_{\mathrm{utility}}=
\max\left(0,
m-\log P(y\mid X,r^*)+\log P(y\mid X,r)
\right).
\]

Use stop-gradient or a frozen raw branch when computing the comparison term so both sides cannot improve by moving together.

The model should be allowed to abstain from grounding when prototype retrieval is ambiguous. Candidate gate inputs include top similarity, top-1/top-2 margin, retrieval entropy, and neighbor agreement.

## 13. Stage H: Robustness evaluation and deferred reasoning-depth analysis

### H1. Current single-thought robustness analysis

Using the supplied `nomul_attn_lr_5e-5` checkpoint, evaluate:

- truncated histories;
- injected irrelevant interactions;
- sparse-user subsets;
- head, torso, and long-tail target items;
- prototype-bank pruning;
- stale prototypes mined from an earlier extraction checkpoint, if one becomes available.

### H2. Deferred multi-thought analysis

Do not run reasoning-depth experiments yet. When multi-thought checkpoints are trained and provided, evaluate `n_thought` in `{1, 2, 4, 6, 8, 10}` with:

- no grounding;
- grounding at every step;
- grounding at only the first step;
- grounding at only the final step;
- a shared bank;
- step-specific banks.

The desired result is that grounding prevents the degradation of ungrounded latent reasoning as depth grows, rather than merely adding a constant one-step improvement.

## 14. Metrics and statistical reporting

### Recommendation metrics

- HR@1, HR@3, HR@5, and HR@10.
- NDCG@1, NDCG@3, NDCG@5, and NDCG@10.
- Target NLL and target rank.
- Catalog coverage and invalid-generation count.

### Mechanism metrics

- Change in target log-probability caused by grounding.
- Fraction of samples helped and harmed.
- Mean grounding strength.
- Retrieval similarity and margin.
- Prototype utilization entropy and effective prototype count.
- Distinct-user support per prototype.
- Dominant-user fraction.
- Raw-to-grounded vector distance.
- Cluster stability across seeds.

Use paired per-user or per-sample tests for model comparisons, report confidence intervals, and correct for multiple comparisons in large sweeps. Report all seeds, not only the best checkpoint.

## 15. Minimum experiment matrix

Run this matrix before expanding the method:

| Dimension | Values |
|---|---|
| Dataset | CDs_and_Vinyl |
| Reasoning steps | 1 only; multi-thought deferred |
| Evidence bank | E-all, E-correct, E-top, E-harmful |
| Prototype count | 32, 64, 128, 256 |
| Grounding | raw, hard, soft residual |
| Grounding strength | 0, 0.25, 0.5, 0.75, 1 |
| Retrieval | top-1, top-4 |
| User access | all users, other-user-only |
| Seeds | at least 3 |

Use validation results to reduce this matrix before running final test evaluations.

## 16. Execution order

1. Load, verify, and freeze the supplied Stage-1 checkpoint. **Complete.**
2. Record its existing baseline metrics under `analysis/`. **Complete.**
3. Implement thought extraction and intervention evaluation. **Complete.**
4. Verify H1 and calculate evidence utility. **Complete; H1 only holds for thought-slot presence.**
5. Build evidence subsets and prototype diagnostics. **Complete.**
6. Run frozen grounding and controls. **Complete; E4 failed its decision gate.**
7. Run cross-user and oracle analyses. **Complete to the decision boundary.**
8. Select one prototype configuration using validation data. **Complete: E-top/K=32/top-1/residual alpha=0.75 was evaluated but rejected.**
9. Implement residual grounding and abstention. **Residual grounding complete; abstention training stopped because there is no deployable frozen gain.**
10. Run end-to-end training ablations. **Stopped by the Stage-D decision gate.**
11. Evaluate single-thought robustness, sparse users, and long-tail items. **Stopped because no grounded treatment passed E4.**
12. Keep reasoning-depth experiments pending until multi-thought checkpoints are supplied.

## 17. Required artifact layout

All derived outputs must be written below the repository's `analysis/` directory:

- `analysis/config/`: immutable experiment configurations and checkpoint references.
- `analysis/evidence/CDs_and_Vinyl/train/`: latent vectors and metadata.
- `analysis/prototypes/CDs_and_Vinyl/<bank>/<K>/`: centroids, assignments, and diagnostics.
- `analysis/results/baseline_1T/`: supplied-checkpoint baseline summaries.
- `analysis/results/intervention/`: thought-causality results.
- `analysis/results/frozen_grounding/`: frozen prototype experiments.
- `analysis/results/cross_user/`: collaborative-transfer results.
- `analysis/results/oracle/`: retrieval upper-bound analysis.
- `analysis/results/stage2/`: end-to-end grounding experiments.
- `analysis/results/robustness_1T/`: single-thought subgroup and robustness results.
- `analysis/results/scaling/`: deferred multi-thought results.
- `analysis/logs/`: extraction, clustering, evaluation, and training logs.
- `analysis/figures/`: plots and visual diagnostics.
- `analysis/cache/`: reproducible intermediate caches that can be regenerated.

Do not save new analysis artifacts in `code/output_dir`, `run_logs`, or the external dataset directory.

Every result record should include the dataset split hash, Stage-1 checkpoint hash, prototype-bank hash, seed, and complete grounding configuration.

## 18. Stop conditions

Pause and revise the idea if any of the following occurs:

- Thought interventions do not affect predictions.
- Oracle prototypes cannot improve target likelihood.
- Random or shuffled prototypes perform as well as useful-evidence prototypes.
- Gains vanish under other-user-only evidence retrieval.
- Hard or soft grounding causes prototype collapse or large personalization loss.
- Improvements appear only with a larger trainable parameter count and not with mined initialization.

These outcomes are informative: they distinguish reusable collaborative reasoning from ordinary regularization, vector quantization, retrieval memory, or added model capacity.

## 19. Experiment journal

Update this table immediately after every completed, failed, or inconclusive experiment. Link the detailed configuration, metrics, and logs stored under `analysis/`.

| ID | Status | Date | Configuration | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Conclusion | Artifacts |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| S0 | Complete | 2026-09-09 | `LatentModel`; hidden 1536; one thought; requested `nomul_attn_lr_5e-5` checkpoint | — | — | — | — | — | — | — | — | Checkpoint and scope verified; checkpoint SHA-256 begins `b03801e8`. | `analysis/config/prototype_grounding.json` |
| E0 | Complete | 2026-09-09 | Existing predictions; 6,157 test examples; 10 constrained beams | 0.05262303 | 0.07601104 | 0.09030372 | 0.11060581 | 0.05262303 | 0.06601825 | 0.07191042 | 0.07838422 | Requested checkpoint baseline reproduced; 60 catalog violations were counted. | `analysis/results/baseline_1T/metrics.json` |
| E1 | Complete | 2026-09-10 | Full validation paired NLL; zero-thought constrained test ranking | 0.01786584 | 0.03556927 | 0.04937470 | 0.07211304 | 0.01786584 | 0.02799347 | 0.03366867 | 0.04107710 | H1 passes for presence of a nonzero thought but fails for sample-specific content; proceed with generic-template controls. | `analysis/results/intervention/valid_summary.json`; `analysis/results/frozen_grounding/generation_test_all_k32_s42_mask_t1_temp0.1_a1.0.json` |
| E1b | Complete; collapse confirmed | 2026-09-10 | 49,251 train thoughts; 100k pair samples; centered PCA on all rows | — | — | — | — | — | — | — | — | Only 348 exact FP16 templates; random-pair cosine 0.9905; two PCs explain 89.85% of centered variance. | `analysis/results/latent_distribution/train_latent_distribution.json`; `analysis/figures/latent_distribution_overview.png` |
| E1c | Complete; SFT/RL comparison | 2026-09-10 | SFT `latent`; 49,251 train thoughts; identical full-bank PCA protocol | — | — | — | — | — | — | — | — | SFT has 2,337 templates and effective rank 4.82; RL has 348 and rank 2.56, so RL intensifies existing collapse. | `analysis/results/latent_distribution/sft_only/train_latent_distribution.json`; `analysis/figures/sft_only_latent_pca_all_scatter.png` |
| E2 | Complete with unavailable controls | 2026-09-10 | 49,251 train thoughts; raw versus zero utility; CF/user/item metadata | — | — | — | — | — | — | — | — | Mean utility +1.10780; all train utilities positive, so E-harmful is empty; exact E-correct unavailable without train rankings. | `analysis/evidence/CDs_and_Vinyl/train/`; `analysis/results/intervention/evidence_utility_summary.json` |
| E3 | Complete with unavailable controls | 2026-09-10 | E-all/E-top; K=32/64/128/256; seeds 13/42/73 | — | — | — | — | — | — | — | — | Severe stable collapse; E-top/K=32 is least degenerate (~10.7 effective prototypes). | `analysis/prototypes/CDs_and_Vinyl/`; `analysis/results/frozen_grounding/prototype_diagnostics_summary.json` |
| E4 | Complete; rejected | 2026-09-10 | E-top/K=32; top-1 residual; alpha=0.75; seeds 13/42/73 | 0.052298/0.052298/0.052136 | 0.076173/0.076336/0.076011 | 0.089654/0.089816/0.089979 | 0.110768/0.110606/0.110768 | 0.052298/0.052298/0.052136 | 0.066022/0.066082/0.065838 | 0.071586/0.071632/0.071591 | 0.078329/0.078274/0.078234 | No consistent NLL or ranking gain; H2 rejected and Stage-2 gate fails. | `analysis/results/frozen_grounding/` |
| E5 | Inconclusive screen; stopped | 2026-09-10 | 512 validation rows; seed 42; user/target/both exclusions; minimum support 1/5/10 | — | — | — | — | — | — | — | — | Exclusions do not identify a reliable gain; full run is invalidated by E4's null result. | `analysis/results/cross_user/` |
| E6 | Complete; diagnostic only | 2026-09-10 | Full validation; E-top/K=32; top-1; seeds 13/42/73 | — | — | — | — | — | — | — | — | Target-aware oracle improves NLL by about 0.011; cosine retrieval is null/slightly harmful. | `analysis/results/oracle/` |
| E7 | Stopped by gate | 2026-09-10 | End-to-end Stage 2 | — | — | — | — | — | — | — | — | Not run: E4 did not satisfy the explicit prerequisite; trainable gains would not establish mined-evidence utility. | — |
| E8 | Stopped by gate | 2026-09-10 | Single-thought robustness | — | — | — | — | — | — | — | — | Not run: there is no successful grounded treatment to stress-test or stratify. | — |
| E9 | Deferred | — | Multi-thought grounding | — | — | — | — | — | — | — | — | Waiting for multi-thought checkpoints. | — |

### Per-experiment update template

For each journal entry, add a short subsection containing:

- objective and hypothesis;
- exact checkpoint and data split;
- command or configuration file;
- artifact and log paths;
- HR/NDCG at 1, 3, 5, and 10;
- mechanism diagnostics relevant to the experiment;
- comparison against E0;
- outcome: supported, rejected, failed, or inconclusive;
- plain-language result description;
- researcher comment explaining the likely mechanism, important caveats, and whether the result changes the working hypothesis;
- next action.

### E0 result — Existing Stage-1 baseline

- **Objective and hypothesis:** establish the immutable ranking baseline for every later paired comparison.
- **Checkpoint/data:** `output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5`; 6,157-row test split; checkpoint SHA-256 `b03801e85325a091305fd9079e54905d25dd9c440fed24711d5e4936447dc5a5`; test split SHA-256 `944634cd2b9248a4a579f222151c6df140935f56f5ea731cfe4ac21c240e1999`.
- **Command/configuration:** `python src/prototype_grounding/experiment.py manifest`; configuration in `analysis/config/prototype_grounding.json`.
- **Result:** HR@1/3/5/10 = 0.05262303/0.07601104/0.09030372/0.11060581; NDCG@1/3/5/10 = 0.05262303/0.06601825/0.07191042/0.07838422. The supplied prediction file contains 60 generated strings absent from the catalog lookup.
- **Plain-language description:** among 6,157 held-out cases, the requested checkpoint places the target in its ten returned recommendations for 11.06% of cases and ranks it first for 5.26%.
- **Researcher comment:** this exactly reproduces the numbers in the checkpoint's `metric.log`, so the evaluation parser and row alignment are trustworthy. The prior draft's `CC=0` statement was incorrect; the original metric log reports 60 catalog violations. Because there are ten returned beams, HR/NDCG beyond @10 are not informative.
- **Limitations:** E0 uses existing predictions and therefore does not yet measure target NLL or thought causality.
- **Outcome/next action:** baseline supported and locked; complete E1 on validation with paired raw/intervened target NLL, followed by constrained-ranking confirmation on test.

### E1 result — Causal thought intervention

- **Objective and hypothesis:** determine whether the latent vector carries sample-specific reasoning that causally changes the target prediction.
- **Checkpoint/data:** frozen `nomul_attn_lr_5e-5`; all 6,156 validation examples; three noise/shuffle RNGs are deterministically seeded from seed 42 by batch.
- **Command/artifacts:** `python src/prototype_grounding/experiment.py intervention --split valid --batch-size 32`; `analysis/results/intervention/valid_summary.json` and paired arrays in `valid_summary.npz`.
- **Likelihood result:** raw NLL 1.169466. Zero replacement raises NLL to 2.170643 (increase 1.001177; paired bootstrap 95% CI [0.991138, 1.011707], p<0.001). Global mean, shuffled-user, and shuffled-target replacements change NLL by +0.000256, +0.000277, and +0.000099 respectively; all three confidence intervals cross zero. Gaussian noise at scales 0.25/0.5/1.0 changes NLL by -0.000579/-0.002483/-0.006078, meaning noise slightly improves rather than harms teacher-forced likelihood.

| Intervention | Mean target NLL | ΔNLL vs raw ↓ | 95% CI for ΔNLL | Paired bootstrap p | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 1.169466 | 0.000000 | [0, 0] | — | 0.05262303 | 0.07601104 | 0.09030372 | 0.11060581 | 0.05262303 | 0.06601825 | 0.07191042 | 0.07838422 |
| Mask / zero | 2.170643 | +1.001177 | [0.991138, 1.011707] | <0.001 | 0.01786584 | 0.03556927 | 0.04937470 | 0.07211304 | 0.01786584 | 0.02799347 | 0.03366867 | 0.04107710 |
| Global train mean | 1.169722 | +0.000256 | [-0.000066, 0.000570] | 0.106 | 0.05181095 | 0.07584863 | 0.08900439 | 0.11206757 | 0.05181095 | 0.06572239 | 0.07113203 | 0.07850156 |
| Shuffled user | 1.169743 | +0.000277 | [-0.000139, 0.000684] | 0.182 | 0.05197336 | 0.07601104 | 0.08851713 | 0.11060581 | 0.05197336 | 0.06577848 | 0.07092968 | 0.07800062 |
| Shuffled target | 1.169566 | +0.000099 | [-0.000309, 0.000510] | 0.638 | 0.05197336 | 0.07601104 | 0.08851713 | 0.11060581 | 0.05197336 | 0.06577848 | 0.07092968 | 0.07800062 |
| Gaussian noise 0.25× | 1.168888 | -0.000579 | [-0.000888, -0.000264] | 0.001 | 0.05213578 | 0.07584863 | 0.08965405 | 0.11109306 | 0.05213578 | 0.06579974 | 0.07148918 | 0.07834391 |
| Gaussian noise 0.5× | 1.166983 | -0.002483 | [-0.003073, -0.001884] | <0.001 | 0.05213578 | 0.07454929 | 0.08851713 | 0.11011856 | 0.05213578 | 0.06510755 | 0.07090964 | 0.07782224 |
| Gaussian noise 1.0× | 1.163388 | -0.006079 | [-0.007121, -0.005022] | <0.001 | 0.05278545 | 0.07422446 | 0.08802988 | 0.10800715 | 0.05278545 | 0.06516364 | 0.07084596 | 0.07733902 |

Here positive ΔNLL means worse than raw. NLL uses all 6,156 validation examples; ranking uses all 6,157 test examples. All constrained-ranking variants completed in Slurm job `245421`.
- **Plain-language description:** the model strongly needs a nonzero vector in the thought position, but it behaves almost identically when the example's vector is replaced by an average or unrelated training vector. The learned thoughts therefore look much more like a shared activation carrier than sample-specific reasoning evidence.
- **Researcher comment:** this is evidence against the motivating noise-reasoning mechanism. The catastrophic zero result alone is insufficient: zero is far outside the learned vector manifold, while in-distribution shuffles leave predictions unchanged. The small benefit from noise additionally suggests local flatness or mild regularization, not fragile semantic reasoning.
- **Ranking result:** zero-thought HR@1/3/5/10 = 0.01786584/0.03556927/0.04937470/0.07211304 and NDCG@1/3/5/10 = 0.01786584/0.02799347/0.03366867/0.04107710. Relative to E0, absolute HR drops are 0.03475719/0.04044177/0.04092902/0.03849277 and absolute NDCG drops are 0.03475719/0.03802478/0.03824175/0.03730711. The constrained evaluator produced zero invalid catalog strings in this rerun.
- **Limitations:** all constrained-generation variants completed. The ranking implementation formally excludes same-user or same-target evidence for the corresponding shuffle. Teacher-forced NLL and generation metrics are measured on validation and test respectively, so their small changes should not be compared sample-by-sample.
- **Outcome/next action:** H1 is supported for the presence of a learned-manifold thought vector and rejected for sample-specific thought content. The formal gate passes, so E3/E4 proceed, but every prototype result must beat global-mean and shuffled in-distribution controls to support collaborative reasoning.

### E1b result — Latent-reasoning distribution and diversity

- **Objective:** test the proposed explanation for E1: mean and shuffled interventions are harmless because the learned thought distribution has very little diversity.
- **Checkpoint/data/configuration:** all 49,251 train thoughts from `nomul_attn_lr_5e-5`, dimension 1,536; 100,000 sampled pairs; exact stored-vector counts at FP16 precision; centered covariance eigendecomposition on all 49,251 thoughts.
- **Command/artifacts:** `python src/prototype_grounding/experiment.py latent-distribution --seed 42 --pair-samples 100000 --pca-samples 49251`; statistics in `analysis/results/latent_distribution/train_latent_distribution.json`, numerical arrays in `train_latent_distribution_arrays.npz`, and plots in `analysis/figures/latent_distribution_overview.png`, `latent_pca_all_scatter.png`, `latent_pca_weighted_templates.png`, and `latent_template_frequency.png`. Both the PCA fit and projection use every thought. The corrected all-vector figure pairs exact-coordinate log density with a display-only jitter view; the weighted version makes exact overlaps visible through marker area.

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Mean vector norm / mean sample norm | 0.9952 | Almost the full vector magnitude is a shared offset. |
| Mean cosine to global mean | 0.9952 | Individual thoughts point in almost the same direction. |
| Mean random-pair cosine | 0.9905 | Unrelated train thoughts are nearly collinear. |
| Mean same-user cosine | 0.9914 | Only +0.0009 above a random pair. |
| Mean same-target cosine | 0.9906 | Only +0.0002 above a random pair. |
| Exact unique vectors at stored FP16 precision | 348 / 49,251 (0.71%) | The bank consists largely of repeated templates. |
| Samples covered by top 10 exact templates | 82.12% | A very small template set dominates the dataset. |
| Samples covered by top 20 exact templates | 91.57% | The long tail carries little mass. |
| Largest exact template | 7,690 samples (15.61%) | One latent value is reused across thousands of interactions. |
| Centered PCA variance: PC1 / top 2 / top 10 / top 32 | 77.05% / 89.85% / 98.28% / 99.74% | Residual variation occupies a tiny subspace. |
| Entropy effective rank / participation ratio | 2.56 / 1.63 | Effective dimension is near two despite 1,536 coordinates. |
| Median distance from global mean | 10.46 | Natural centered variation is small relative to norm ≈95.34. |
| Median shuffled-vector distance | 16.10 | Shuffling moves within a narrow learned family. |
| Expected noise L2 at 0.25× / 0.5× / 1.0× | 23.87 / 47.73 / 95.47 | Even the smallest noise exceeds typical natural residual variation. |

- **Preference-signal diagnostic:** after subtracting a same-group-size permutation baseline, user identity explains about 10.19% of centered variance, target identity 2.56%, and CF cluster 0.79%. Thus some personalization signal exists, but it is weak compared with the shared low-dimensional structure and is not exploited by the recommendation intervention.
- **Utility association:** zero-mask utility has only weak association with vector norm (Pearson -0.028), centered distance (-0.074), or cosine to the mean (+0.033); exact-template identity explains only 3.37% of utility variance. The utility ranking used for E-top therefore does not separate geometrically distinctive reasoning very well.
- **Plain-language description:** this directly supports the proposed collapse explanation. Most examples reuse a few latent templates; unrelated users and targets receive almost parallel vectors; and nearly all centered variance lies in a handful of directions. The PCA visualization shows discrete template islands rather than a rich continuous reasoning space.
- **Researcher comment:** collapse explains why global-mean and shuffled thoughts remain in-distribution and preserve performance. It is not the whole explanation, however: 0.25× Gaussian noise moves a vector farther than a typical natural shuffle, yet E1 ranking remains nearly unchanged. The downstream decoder is therefore also weakly sensitive to thought *content*, while strongly sensitive to whether the thought slot contains a normal-sized activation at all. This is consistent with a shortcut through the directly visible history tokens.
- **Limitations:** exact duplicate counts are measured after FP16 storage, so nearby BF16 outputs may quantize to the same saved vector. This cannot explain the cosine/PCA collapse, which is also present in float32 analysis. PCA and variance decomposition use the complete train bank; pair statistics use 100,000 draws.
- **Outcome/next action:** the latent-diversity hypothesis is strongly supported, but merely adding a diversity loss may not force causal use. Before prototype mining is retried, train a bottlenecked model in which item prediction cannot bypass the thought through full-history attention, and add a counterfactual dependence gate: shuffled in-distribution thoughts—not only zero—must significantly worsen held-out NLL and ranking.

### E1c result — SFT-only latent distribution versus RL

- **Objective:** determine whether latent collapse is created by verifier-free RL or is already present in the SFT-only checkpoint.
- **Checkpoint/data:** SFT-only checkpoint `output_dir/CDs_and_Vinyl/latent`, SHA-256 `cd5501c9865046ca544795367445317a44085114fd927a4b8e01138096875799`; the same 49,251-row train split and 1,536-dimensional thought representation used in E1b.
- **Commands/artifacts:** extraction used `python src/prototype_grounding/experiment.py extract --split train --batch-size 64 --checkpoint output_dir/CDs_and_Vinyl/latent --tag sft_only`; analysis used `latent-distribution --seed 42 --pair-samples 100000 --pca-samples 49251 --checkpoint output_dir/CDs_and_Vinyl/latent --tag sft_only`. Evidence is under `analysis/evidence/CDs_and_Vinyl/sft_only/train/`; statistics are in `analysis/results/latent_distribution/sft_only/train_latent_distribution.json`; figures are prefixed `analysis/figures/sft_only_`.

| Diagnostic | SFT only | RL `nomul_attn_lr_5e-5` | Interpretation |
|---|---:|---:|---|
| Raw train target NLL ↓ | 0.714917 | 0.687539 | RL improves teacher-forced train fit by 0.027378. |
| Zero-mask train NLL | 1.795241 | 1.795336 | The zero-vector branch is essentially unchanged. |
| Mean zero-mask utility | 1.080324 | 1.107797 | RL increases the advantage of having a learned-manifold activation. |
| Mean vector norm | 72.59 | 95.34 | RL substantially increases latent magnitude. |
| Mean-vector norm / mean sample norm | 0.9926 | 0.9952 | Both are dominated by a shared offset; RL is slightly more concentrated. |
| Mean cosine to global mean | 0.9927 | 0.9952 | Both banks occupy a narrow cone. |
| Mean random-pair cosine | 0.9855 | 0.9905 | RL makes unrelated thoughts more collinear. |
| Exact FP16 templates | 2,337 | 348 | RL reduces stored template diversity by 85.1%. |
| Top-10-template sample coverage | 32.21% | 82.12% | RL concentrates most samples into very few templates. |
| Largest-template coverage | 6.32% | 15.61% | The dominant RL template is much more prevalent. |
| PC1 centered variance | 48.27% | 77.05% | RL collapses residual variation toward one main axis. |
| Top-two centered variance | 67.76% | 89.85% | Two dimensions nearly describe the RL bank. |
| Entropy effective rank | 4.82 | 2.56 | SFT is low-rank, and RL roughly halves the effective dimension. |
| Participation ratio | 3.35 | 1.63 | The same conclusion holds under a second rank measure. |
| Adjusted user-centroid variance | 5.15% | 10.19% | RL retains/increases some user-linked residual structure despite global collapse. |
| Adjusted target-centroid variance | 2.08% | 2.56% | Target-linked geometry remains weak in both checkpoints. |
| Adjusted CF-centroid variance | 0.61% | 0.79% | Conventional collaborative grouping explains less than 1%. |

- **Visual result:** the SFT [all-vector PCA plot](../figures/sft_only_latent_pca_all_scatter.png) contains more visible branches and template islands than the corresponding RL plot, while still showing heavy exact overlap. The frequency-weighted view is `analysis/figures/sft_only_latent_pca_weighted_templates.png`, with overview and template-frequency plots in the same directory.
- **Plain-language description:** collapse does not originate solely from RL. The SFT model already maps 49,251 interactions into a narrow, repeated-template representation. RL improves train NLL mainly while enlarging and concentrating that shared representation, leaving only 348 stored templates and an effective rank near 2.6.
- **Researcher comment:** this refines the working hypothesis. SFT creates the shortcut-compatible latent channel; verifier-free RL exploits it further instead of producing diverse sample-specific reasoning. The near-identical zero-mask NLL across checkpoints, together with the larger RL norm and lower raw NLL, is consistent with RL optimizing a shared activation carrier. It does not demonstrate that RL learned more informative reasoning.
- **Limitations:** exact counts are measured at FP16 storage precision, although cosine and PCA independently confirm collapse. Training NLL is not a held-out causal comparison. PCA axes are fitted separately for each checkpoint, so their coordinates cannot be compared point-by-point; scalar diversity statistics can be compared.
- **Outcome/next action:** the diversity problem should be addressed during SFT, not only during RL. Introduce the bottleneck and sample-specific dependence/diversity objectives in SFT, require the shuffled-thought E1 test to pass, and then use RL only after diversity is maintained.

### E2 result — Evidence extraction and utility scoring

- **Objective and hypothesis:** extract every train-split thought and identify evidence whose presence improves its own target likelihood.
- **Checkpoint/data:** frozen `nomul_attn_lr_5e-5`; all 49,251 chronological training interactions. Original vectors are stored as float16 `[49251, 1536]`; metadata is separate CSV. Train split/checkpoint hashes are recorded in `summary.json`.
- **Command/artifacts:** `python src/prototype_grounding/experiment.py extract --split train --batch-size 8`; vectors and metadata under `analysis/evidence/CDs_and_Vinyl/train/`; utility/subgroup summary under `analysis/results/intervention/evidence_utility_summary.json`; train-only CF clusters under `analysis/cache/`.
- **Result:** mean raw target NLL 0.687539 versus masked NLL 1.795336, giving mean utility +1.107797. Utility quantiles at 0/25/50/75/100% are 0.038829/0.765819/1.082527/1.395745/4.514868. All 49,251 train utilities are positive. Mean utility is 1.0791/1.1071/1.1081 for sparse/medium/active training users and 1.0869/1.1204/1.1189 for tail/torso/head items.
- **Evidence subsets:** E-all=49,251; E-positive=49,251 (identical to E-all); E-top=12,313; E-harmful=0. The exact E-correct subset cannot be formed from the supplied checkpoint because it contains test predictions only, not train-split constrained ranks; it is not silently approximated.
- **Plain-language description:** every extracted training thought is better than an all-zero thought, but usefulness barely varies by user activity or item popularity. There is no naturally harmful evidence under the plan's signed-utility definition.
- **Researcher comment:** together with E1, this suggests utility is dominated by an off-manifold zero baseline. Since E-all and E-positive collapse to the same bank and E-harmful is empty, H3's planned evidence-quality contrast is partially non-identifiable for this checkpoint. E-top remains the only valid utility-selective comparison.
- **Limitations:** product genre/category labels are unavailable, so `target_category` is the dataset-level label. CF clusters are derived leakage-safely from the binary train user-item matrix (rank-32 SVD, 64 clusters). Exact train recommendation ranks would require a costly additional full constrained-generation pass.
- **Outcome/next action:** extraction succeeded; preserve the empty/unavailable controls as findings and mine E-all/E-top banks after E1's ranking decision gate.

### E3 result — Prototype mining and diagnostics

- **Objective and hypothesis:** test whether utility-selected train thoughts form reusable, well-supported, non-duplicate clusters.
- **Configuration:** weighted spherical k-means on L2-normalized vectors; E-all and E-top banks; K=32/64/128/256; seeds 13/42/73; inverse user-frequency weighting. E-positive is identical to E-all, E-harmful is empty, and exact E-correct is unavailable as established in E2.
- **Command/artifacts:** repeated `python src/prototype_grounding/experiment.py mine ...`; centroids, assignments, manifests, and per-prototype diagnostics under `analysis/prototypes/CDs_and_Vinyl/`; aggregate diagnostics in `analysis/results/frozen_grounding/prototype_diagnostics_summary.json`.
- **Result:** E-all effective prototype counts averaged 2.8, 3.0, 13.9, and 14.3 at nominal K=32,64,128,256. Mean nearest-prototype cosine was 0.7855/0.8958/0.9550/0.9776. E-top effective counts were 10.7/13.4/13.5/13.6, with nearest-prototype cosine 0.9303/0.9653/0.9827/0.9913. Median distinct-user support for E-top/K=32 was 8; for most larger banks it fell to 0–2. Every configuration contained at least one fully user-dominated prototype. Matched centroid cosine across seeds ranged from 0.9666 to 0.9998.
- **Plain-language description:** asking for more clusters does not reveal more shared reasoning types. Most evidence falls into a handful of directions, while extra clusters become tiny, duplicated, or single-user islands.
- **Researcher comment:** this is a stable geometric failure mode, consistent with the E1 result that in-distribution thoughts are nearly interchangeable. High cross-seed similarity does not imply healthy clusters here; it shows that the same collapsed structure is repeatedly recovered. E-top/K=32 is selected for downstream testing because it has the best effective usage and user support, not because it is healthy.
- **Limitations:** category-specific banks cannot be assessed because item genre labels are absent. Single-thought makes shared-versus-step-specific banks identical. Empty and unavailable evidence subsets are reported rather than replaced with altered definitions.
- **Outcome/next action:** H3 is presently unsupported and partially non-identifiable. Run E4 with E-top/K=32, all three seeds, hard/residual variants, and aggressive mean/shuffle/random controls; prune low-user-support prototypes in E5.

### E4 result — Frozen prototype grounding

- **Objective and hypothesis:** test whether utility-selected prototypes improve recommendation with the checkpoint and bank frozen. H2 requires consistent improvement over raw reasoning and non-semantic controls.
- **Checkpoint/data:** frozen `nomul_attn_lr_5e-5`; 6,156 validation examples for target NLL and 6,157 test examples for constrained ranking; E-top/K=32; clustering seeds 13/42/73.
- **Command/artifacts:** validation sweep through `scripts/prototype_grounding.sh`; test ranking through Slurm job `245428` using `scripts/e4_ranking_selected.sbatch`. Results and paired arrays are in `analysis/results/frozen_grounding/`; logs are `analysis/logs/E4_screen_rescaled.log`, `E4_full_valid.log`, and `E4_ranking_slurm.245428.log`.
- **Representation correction:** the first exploratory E4 pass inserted unit-normalized spherical-k-means directions into a raw latent space. Because this changed vector scale, those results were invalid and were moved to `analysis/results/frozen_grounding/superseded_unit_centroids/`. The reported rerun stores raw weighted cluster means for intervention and separate unit vectors for cosine lookup.

| Variant | Seed | Validation NLL | Raw − grounded NLL ↑ | 95% paired CI | Helped | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw checkpoint | — | 1.169397 | 0 | — | — | 0.052623 | 0.076011 | 0.090304 | 0.110606 | 0.052623 | 0.066018 | 0.071910 | 0.078384 |
| E-top residual, top-1, α=0.75 | 13 | 1.169433 | -0.000037 | [-0.000179, 0.000114] | 0.4933 | 0.052298 | 0.076173 | 0.089654 | 0.110768 | 0.052298 | 0.066022 | 0.071586 | 0.078329 |
| E-top residual, top-1, α=0.75 | 42 | 1.169426 | -0.000030 | [-0.000176, 0.000127] | 0.4922 | 0.052298 | 0.076336 | 0.089816 | 0.110606 | 0.052298 | 0.066082 | 0.071632 | 0.078274 |
| E-top residual, top-1, α=0.75 | 73 | 1.169406 | -0.000009 | [-0.000150, 0.000140] | 0.4920 | 0.052136 | 0.076011 | 0.089979 | 0.110768 | 0.052136 | 0.065838 | 0.071591 | 0.078234 |

- **Sweep/control result:** the 512-example validation screen covered hard replacement and residual top-1/top-4 with α=0.25/0.5/0.75/1.0 for all seeds. Effects were only about ±0.0005 NLL. At seed 42, global-mean replacement changed NLL by +0.000032, shuffled prototypes by +0.000593, and an arbitrary Gaussian vector by +0.899801. The last result only confirms that off-manifold perturbation is destructive; it is not evidence for semantic retrieval. Category centroids are unavailable because the dataset has no item-category field. CF-centroid and random-k-means controls were not promoted after the proposed E-top treatment itself failed the raw-baseline gate.
- **Plain-language description:** grounding changes almost no validation likelihood and changes only a few test ranks. Every seed has lower NDCG@10 than the raw checkpoint, and none has a statistically reliable NLL gain.
- **Researcher comment:** the outcome agrees with E1 and E3: thoughts occupy a narrow shared manifold, but nearest cluster identity does not add useful per-example information. The tiny rank fluctuations are evaluation noise, not collaborative reasoning transfer. The alpha=0.75 choice was retained across seeds rather than cherry-picking a favorable seed.
- **Limitations:** only the least-collapsed bank was taken to full validation/test after the broad screen. The plan's required semantic category control is structurally unavailable. Ranking confidence intervals were not computed, but all absolute changes are at most 0.00065 and disagree across cutoffs.
- **Outcome/next action:** H2 is rejected. The explicit Stage-D gate fails, so E7 must not be run as confirmation of the central claim. Run E6 only as a diagnostic upper bound and use E5 only to determine whether its small screen signal survives leakage exclusions.

### E5 result — Collaborative-transfer exclusions

- **Objective and hypothesis:** determine whether any apparent frozen-grounding signal depends on evidence from the query user or same target.
- **Configuration/artifacts:** E-top/K=32 seed 42, top-1 residual α=0.75, first 512 validation examples; leave-one-user/target retrieval uses raw-scale train evidence. Results are in `analysis/results/cross_user/`; log is `analysis/logs/E5_screen_rescaled.log`. An earlier unit-scale leave-out implementation was invalidated and moved to `analysis/results/cross_user/superseded_unit_centroids/`.

| Evidence access | Minimum distinct users | Grounded − raw NLL ↓ | 95% paired CI | Paired p | Helped fraction |
|---|---:|---:|---:|---:|---:|
| All evidence | 1 | -0.000261 | [-0.000786, 0.000237] | 0.286 | 0.5059 |
| Query user excluded | 1 | -0.000323 | [-0.000839, 0.000188] | 0.189 | 0.5078 |
| Same target excluded | 1 | -0.000269 | [-0.000795, 0.000225] | 0.271 | 0.5059 |
| User and target excluded | 1 | -0.000324 | [-0.000844, 0.000185] | 0.185 | 0.5078 |
| User and target excluded | 5 | -0.000311 | [-0.000831, 0.000205] | 0.214 | 0.5059 |
| User and target excluded | 10 | -0.000311 | [-0.000831, 0.000205] | 0.214 | 0.5059 |

- **Plain-language description:** excluding potentially memorized evidence does not visibly remove the tiny screen improvement, but every interval includes zero and the result is based on only 512 examples.
- **Researcher comment:** this cannot support H4. The exclusion invariance is equally compatible with genuinely shared information and with the near-interchangeable shared activation template diagnosed in E1. Because E4's full run is null/slightly harmful, there is no established gain whose collaborative source can be tested.
- **Limitations:** this is a single-seed screening analysis, not a promoted full evaluation. Subgroup reporting would be misleading at this sample size and after failure of the upstream efficacy gate.
- **Outcome/next action:** E5 is inconclusive and full promotion is stopped. Revisit H4 only after a revised retriever yields a significant full-validation improvement without target-aware selection.

### E6 result — Oracle retrieval analysis

- **Objective and hypothesis:** separate a deficient prototype bank from a deficient cosine retriever by allowing the known validation target to select the best of eight candidate prototypes. Oracle results are diagnostic and non-deployable.
- **Checkpoint/data/configuration:** frozen `nomul_attn_lr_5e-5`; all 6,156 validation examples; E-top/K=32; candidate-k=8; seeds 13/42/73.
- **Command/artifacts:** `python src/prototype_grounding/experiment.py oracle --split valid --bank top --k 32 --candidate-k 8 ...`; `analysis/results/oracle/valid_top_k32_s{13,42,73}.json`; log `analysis/logs/E6_oracle_full_valid.log`.

| Seed | Raw NLL | Cosine-retrieval NLL | Raw − cosine NLL ↑ | Oracle NLL | Raw − oracle NLL ↑ | Oracle helped fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 1.169397 | 1.169482 | -0.000086 | 1.158646 | 0.010751 | 0.8955 |
| 42 | 1.169397 | 1.169467 | -0.000070 | 1.157695 | 0.011701 | 0.9264 |
| 73 | 1.169397 | 1.169470 | -0.000073 | 1.158239 | 0.011157 | 0.9163 |

- **Plain-language description:** useful directions exist among the nearby candidates, but cosine similarity consistently chooses the wrong one. Knowing the target gives a roughly 0.011 NLL improvement; deployable similarity retrieval is slightly worse than doing nothing.
- **Researcher comment:** the bank is not entirely devoid of counterfactual directions, so a revised project should focus on query-conditioned retrieval or contrastive alignment. However, an oracle gain does not rescue the collaborative-reasoning claim: the selection consumes the answer, and E1 still shows that unrelated thoughts are nearly interchangeable.
- **Limitations:** the oracle directly uses the validation target and has no valid HR/NDCG deployment interpretation. No learned target-free retriever was trained because the frozen-grounding prerequisite failed.
- **Outcome/next action:** prototype quality has a small oracle upper bound, while basic retrieval is rejected. Revise the method before Stage 2, ideally by first training a target-free retriever that beats raw NLL on held-out validation.

### E7 result — End-to-end Stage-2 grounding

- **Status/outcome:** stopped by the preregistered Stage-D decision gate; no training run was launched.
- **Reason and researcher comment:** E4 requires useful-evidence grounding to improve NLL or ranking consistently before end-to-end training. It did neither. A trainable Stage-2 model could exploit added capacity or ordinary continued fine-tuning and would therefore not establish that mined evidence is useful—the plan explicitly labels that evidence insufficient for the central claim.
- **Next action:** revise the representation/retrieval objective, establish a frozen held-out gain, and only then compare continued FT, mined initialization, and equal-capacity random initialization.

### E8 result — Single-thought robustness and subgroup analysis

- **Status/outcome:** stopped under the plan's stop conditions because no grounded treatment passed E4.
- **Reason and researcher comment:** robustness analysis presupposes an effective treatment. Stratifying null fluctuations over sparse users, item popularity, or corrupted histories would multiply comparisons without answering whether prototype grounding works. A stale-checkpoint test is additionally unavailable because only one extraction checkpoint was supplied.
- **Next action:** retain E0 as the reference robustness baseline; activate E8 only after a revised grounded model demonstrates a preregistered full-validation effect.
