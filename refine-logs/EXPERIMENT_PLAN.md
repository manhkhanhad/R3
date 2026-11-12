# Experiment Plan: Cluster Evidence as a Trained Latent Variable

Env: `module purge; module load CUDA/12.4.0; conda activate rrec`. Outputs go to `scratchDirectory/R3/output_dir`; copy results back to `analysis/results/collab_reasoning/posterior_prior/`. Report HR/NDCG @1,3,5,10,20 for every result (@20 = @10 decoded with 10 beams).

> These designs are proposals. Code is edited, and jobs are launched, only after user agreement.

## E2-diag: is the decoder posterior a better target than the label? (GPU, ≤2 h, measurement only)

- **What.** Teacher-forced log p_θ(target | x, c) with the frozen `oracle_cfcluster150` model.
  - Rows: 2,000 validation rows.
  - Clusters per row: the 24 candidates (true cluster, mix top-15, 8 random) plus ∅, where ∅ is R3 v1's own path.
  - Reuses the scoring code in `evidence_eval.py`.
- **Outputs.** q per row. Also, for 1,000 of the rows, beam decodes of the top-3 clusters under q, to get conversion labels (does that cluster's path hit?).
- **Kill.** On R3-miss rows, q's top non-∅ cluster converts no more often than the 22.7% predictor baseline.
- **Also report.**
  - How often argmax q equals the label cluster.
  - Mass on ∅, split by R3 hit vs miss rows: it should be high on hits and low on misses that some path converts.

## E3: train the prior on q, decoder frozen (GPU, ~2 h; only if E2-diag passes)

- **Data.** q computed on the validation split (out-of-period, like test). Hold out 20% for early stopping.
- **Head.** MLP over [RMSnorm(r), mix-predictor logits], with 151 outputs.
- **Loss.** KL(q ‖ p_η).
- **Control.** The same head trained with label CE, with ∅ labelled "R3 hit".
- **Inference on test.** Take the top-K by p_η (K ∈ {1, 3, 5}) plus ∅, decode, and combine with the learned weights.
- **Tuning.** Rows 0–999, if anything needs tuning. Report rows 1,000–2,999, comparable with the cluster-path table, and then the full test set.

## E4: EM training of the decoder (GPU, ≤4 h; only if E3 beats the label-CE control)

- **Model.** Start from `oracle_cfcluster150`.
- **Loss.** L = −Σ_c SG(q(c)) log p_θ(y|x,c) + KL(q ‖ p_η).
- **Refresh.** Recompute q each epoch on a training subset, using fold-held-out rows so that q is not over-confident in-sample, the same leakage concern as in the calibration fix.
- **Risk.** The decoder learns to ignore c. Track the swap-cluster KL ratio every epoch, and stop if it falls below 1.

## Baseline block (CPU, existing decodes)

- Training-free path fusion and p(c|x) selection, on the 317369 decodes.
- Output RRF of v1 and fused CF.

These are reported as rows of the table only, not as contributions.

## Order and budget

E2-diag, then a decision, then E3, then a decision, then E4. Total ≤ 8 GPU-hours. Partition `researchshort`, constraint `3090|a40|l40|l40s|a100|h100`.
