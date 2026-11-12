# Collaborative Reasoning

## Purpose

Improve the plain-R3 collaborative reasoning baseline from [Attempt 10](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10) without mistaking a healthier latent distribution for better-used reasoning. This is a **discussion and experiment-planning document**. The directions below are hypotheses, not committed implementations or claimed results; each should be refined before coding.

The central question is: can the model learn a sample-specific latent `r` that is both collaboratively structured **and causally useful** for next-item recommendation?

## Current baseline: Attempt 10

Attempt 10 trains plain R3 with two projection heads and multi-positive InfoNCE: same-user and same-target examples are positives, while examples sharing neither relation are negatives. A correctness-gated FIFO bank supplies pairs despite the small training micro-batch. The auxiliary losses shape the deployed raw `r`, but the heads and bank are absent at inference.

On CDs and Vinyl, the reported standard ranking is HR@10 **0.1166** and NDCG@10 **0.0813**, approximately preserving the strongest plain-R3 reference. Raw-latent effective rank rises from **21.3** to **111.0**; user- and target-associated structure both strengthen. However, held-out E1 ranking is raw HR@10 **0.1178** / NDCG@10 **0.0820**, and replacing each latent with the train mean or a shuffled valid latent does not significantly change either metric. Zero masking causes a large decline. The baseline therefore improves representation organization without establishing that the decoder uses the *correct* sample's reasoning. Full methods, uncertainty intervals, and artifacts are in [Attempt 10](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10).

**Provenance gate:** the published E1 artifacts refer to checkpoint SHA-256 `4750fc27ddc42dec86def25d45701d186fee30c3f90d933d33723e6f55d574e2`, whereas a later local extraction found a different on-disk `CDs_and_Vinyl_R3_collaborative_infonce/latent` hash, `f078edc5c68a2cd6d74a4b061b6d79359c8926f9475a9d70fbdcf427b6c54edd`. Future comparisons must pin and record the actual model file, data split, train-latent bank, and evaluation protocol; results from these two hashes must not be silently pooled.

## What needs improvement

1. **Pair quality:** “same user” and “same target” are coarse relations. They need not indicate the same immediate preference or reasoning, so equal-weight positives can blur distinct interests.
2. **Bank quality:** exact target-token correctness checks the final answer, not whether the latent caused or faithfully represented it. The history-to-decoder path can answer correctly while ignoring `r`.
3. **Aspect imbalance:** Attempt 10's user organization is much stronger than its target organization; equal loss weights do not imply equally useful gradients.
4. **Decoder utilization:** better geometry may remain confined to directions that the second-pass decoder does not use. Contrastive improvement alone need not move target likelihood or ranking.

## Potential directions for discussion

### A. Multi-aspect collaborative positives

Extend user/target identity with preference aspects such as title semantics and train-derived collaborative groups. Let pair weights reflect aspect agreement rather than assigning every same-user or same-target pair identical strength. Preserve separate aspect heads so conflicting relations need not collapse into one cosine space. [VRec](https://arxiv.org/html/2603.07725v1) motivates multi-dimensional preference verification, but its end-to-end results do not by themselves validate this InfoNCE modification.

**First ablation:** change only positive-pair weighting; keep R3, queue size, loss scale, and ranking protocol fixed. Compare raw latent geometry, held-out target NLL, and mean/shuffle ranking interventions. **Risk:** noisy semantic or cluster labels can encode popularity or target metadata rather than transferable preference.

### B. History-conditioned aspect router

Use a small router conditioned only on information available at inference (history and generated `r`) to weight user-, target-, and semantic-aspect losses per example. This may help different users or histories emphasize different preference dimensions. Keep target ID out of router input; regularize against routing everything to the easiest head.

**First ablation:** compare fixed equal weights with a bounded router while leaving pair construction unchanged. **Risk:** the router may suppress hard but useful objectives and further improve auxiliary loss without improving recommendation.

### C. Verifier-informed queue and pair confidence

Pre-train a preference-alignment verifier using available group-level labels, then use calibrated confidence to down-weight questionable positives or queue entries. Keep exact-answer correctness as a separate diagnostic, not the only quality certificate. Compare three gates: correctness-only (baseline), preference-only, and their conjunction or soft combination. Verification must never use the unknown next item at inference.

**First ablation:** freeze the verifier while testing its effect on candidate-bank composition and pair quality, before jointly training it with R3. **Risk:** group-prediction confidence can be high for a spurious preference; confidence alone does not prove `r` helps the decoder. The [Latent Verification](Latent%20Verification.md) analysis shows why target-aware oracle quality and an inference-time selector must be evaluated separately.

### D. Contrastive objectives closer to recommendation utility

Supplement identity-based InfoNCE with a signal that a latent improves its own history's target prediction relative to checkpoint-matched mean or mismatched latents. Keep the comparison within the **same history**, so a correct decoder answer obtained from history alone is not mistaken for useful reasoning. Consider preference-calibrated soft labels or ranking-aware margins only after measuring whether stable positive counterfactuals exist.

**First ablation:** diagnostic collection of paired latent-specific target-NLL effects before any new loss. **Risk:** target-aware comparison is legal for training labels but unavailable for verifier selection at inference; naive joint optimization can manufacture off-manifold or activation-scale shortcuts.

### E. A decoder path that can use verified reasoning

If A–D improve geometry or verifier accuracy but raw/mean/shuffle remain interchangeable, treat utilization as an architectural bottleneck. A bounded latent-conditioned correction to a preserved high-performing decoder is one candidate. This is a separate hypothesis from improving collaborative InfoNCE and should not be bundled into the first contrastive ablation.

**First ablation:** freeze the baseline decoder, train only a small correction path, and require improvement over both the frozen baseline and mean-latent correction. **Risk:** forcing all prediction through `r` has damaged ranking in prior attempts; preserve a safe baseline route.

## Evaluation contract

- Compare each proposal with the **same pinned Attempt-10 checkpoint or matched retraining setup** and vary one mechanism at a time. Record checkpoint hash, seeds, split hashes, train-latent bank provenance, and inference-time inputs.
- Report standard held-out HR/NDCG with paired uncertainty, plus E1-style `raw`, `mask`, `mean`, and shuffled-latent interventions. Rank improvements and `raw > mean/shuffle` are separate acceptance conditions.
- Track raw-`r` geometry, user/target/aspect organization, queue acceptance, and active contrastive gradients as diagnostics—not as success criteria.
- For any learned verifier or router, separate target-aware training labels from target-free inference; evaluate calibration and selection on held-out users/histories. Do not infer causal latent utility from correct answers alone.
- Replicate promising results across seeds and datasets only after a single-change CDs and Vinyl experiment passes the causal gate.

## Discussion order

Start with **A: multi-aspect pair weighting**, because it changes the contrastive supervision at the narrowest layer while preserving the Attempt-10 model and decoder. Discuss B and C after determining whether aspect labels and verifier confidence are reliable. Consider D only with stable paired counterfactual labels, and E if improved representations still fail decoder-utilization interventions. This is a proposed order, not a final design decision.

## Musical Instruments title+CF experiment (submitted; no results yet)

The first implemented variant combines two agreed changes to the **Collaborative Reasoning** training objective: same-user positives receive train-only title-similarity weights, and pairs of different users with CF-neighbor target items become a new positive relation with their own projection head. Same-target positives remain in place; CF positives are excluded from the negative sets of the other heads. The original R3 code and checkpoints are not edited.

Training starts from the **original Musical Instruments R3 v0 epoch-2 checkpoint**, `~/scratchDirectory/R3/output_dir/Musical_Instruments_v0/latent/checkpoint-1038` (trainer epoch ≈1.999; model SHA-256 `d94b019217ddf8551d95fac0739656198ce2542caa052ac583cef8176f16e57e`). Before optimization, a no-gradient pass over shuffled training examples fills the non-persistent FIFO with latents whose supervised target-token accuracy is at least **0.8**, stopping once its 10,000-entry capacity is reached or the train split is exhausted. The same gate is used for subsequent queue insertions during training. The queue is constructed from the epoch-2 model; it is not restored from a checkpoint.

Implementation choices for this first run: title vectors are 64-dimensional normalized train-only TF-IDF/SVD features on unique item titles; CF neighbors are train-only item–user co-consumption cosine neighbors with at least two common users, cosine ≥0.1, and at most 32 neighbors per item. On Musical Instruments training data this yields 5,004 items, 2,676 with at least one CF neighbor, and 8,979 directed neighbor edges. A positive title-weight floor of 0.25 keeps every same-user pair positive. The job continues the epoch-2 model weights for **one additional epoch** with a fresh optimizer at learning rate `5e-5`; user, target, and CF auxiliary coefficients are each 0.1. These settings are initial implementation choices, not evidence that they are optimal.

Launcher: `scripts/latent_collaborative_title_cf_musical_full.sbatch`; Slurm job **302825** (replacing pending job 302809), submitted 2026-09-24 and pending at last check. The GPU constraint includes 3090; the launcher uses micro-batch 2 and queue-prefill batch 4 on 3090, versus 4 and 8 on the other allowed GPUs. Output: `~/scratchDirectory/R3/output_dir/Musical_Instruments_R3_collaborative_title_cf_from_v0_e2/latent`. The launcher records source/model checksums, trains, then runs standard ranking evaluation. Focused tests and a real-checkpoint GPU prefill/forward smoke test passed before submission. **No training or ranking result is available yet.**

This run is **not** a single-variable comparison against the published Attempt-10 Musical checkpoint: it changes initialization, training duration, queue warm-up/gate, title weighting, adds a CF head, and raises the total auxiliary coefficient from 0.2 to 0.3. It can establish feasibility and generate a candidate model, but attribution requires a matched control trained from the same epoch-2 checkpoint with the same prefill, gate, duration, and optimizer settings but the original Attempt-10 loss. Any ranking gain must then also survive raw/mean/shuffle interventions to count as better-used reasoning.
