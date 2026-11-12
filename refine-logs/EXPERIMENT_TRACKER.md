# Experiment Tracker — cluster evidence as a trained latent variable

| run | config | status | gate / baseline |
|---|---|---|---|
| prereq | retrain R3 v1 and oracle_cfcluster150, lost in the scratch purge; v1 valid preds; re-decode cluster paths on 3k test rows → `cluster_paths/full_retrain`. First chain 341713–341717 failed 2026-10-29 06:00 (home disk write failed at epoch-2 save). Resumed from checkpoint-520 on scratch: 342764 (v1), 342765 (oracle), 342766 (valid preds), 342767 (paths) | DONE 2026-10-30 (R3 path 0.1480, best-path oracle 0.2280 on report rows) | re-baselines R3 path / best-path oracle on the new models |
| E2-diag | exact decoder posterior q over candidates (true ∪ mix/grufull top-15 ∪ 8 random) + ∅; 3k test rows + all valid rows (`scripts/collab_posterior.sbatch`, job 342768) | DONE 2026-10-30: formal gate passes (argmax-q path converts 47.8% of decoded misses) but q ≈ label (argmax q = label on 96.4%; median mix rank 31 vs 31; select-by-q 0.1088 vs select-by-label 0.1073 on misses) → KL(q) vs label-CE contrast has no headroom | kill if top non-∅ q converts ≤ 22.7% of R3 misses |
| E3 | prior head trained with KL(q‖p_η), decoder frozen; vs label-CE control | gated on E2-diag | must beat label-CE control and R3 v1 (0.1450 report rows) |
| E4 | EM: decoder trained on posterior-weighted clusters | gated on E3 | track swap-cluster KL ratio ≥ 1 |
| baseline | training-free path fusion / p(c\|x) selection; output RRF 0.1346 | CPU, any time | table rows only |
