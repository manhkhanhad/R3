# Experiment Tracker — latent-utilization-2026-10-28 (v2: C1 lead + C11 probe)

| ID | What | Claim | Budget | Status | Result |
|---|---|---|---|---|---|
| P1 | C11 zero-training CFG probe on L7 (w∈{1.5,3,5}; Musical, 1,500 rows) | K4 | ≤2 GPU-h | DONE (341711 COMPLETED 2026-10-29; 341710 beam-prefix bug fixed in 29495a3) | Gates: (i) PASS |δ|≈0.6, KL≈0.038; (ii) **FAIL** — w=1.5 +0.004 HR@10 n.s., w=3/5 significantly WORSE; (iii) PASS (raw_cfg3 − mean/shuffle ≈ +0.10 HR@10, sig). Identity check exact (mean_cfg3 δ≈1e-10). C11 NOT promoted; E3 not triggered. |
| P2 | C6 beam-vs-catalog-scoring audit (R3 + L7) | K5 | ≤2 GPU-h | NOT STARTED (awaiting user go-ahead) | — |
| E1 | C1 sweep m∈{2,5,10} + per-m mask-integrity checks | K1,K3 | 3 × ~1 GPU-day | AWAITING DESIGN DISCUSSION | — |
| E2 | Matched recent-only constant-slot baselines (per m) | K2 | 3 short runs | AWAITING DESIGN DISCUSSION | — |
| E3 | C11 training on contrasted logits | K4+ | 1–2 runs | NOT TRIGGERED (P1 promotion gate failed 2026-10-29) | — |
| R1 | Toys + CDs replication of chosen operating point | confirm | TBD | BLOCKED on E1–E3 | — |

Demoted (user, 2026-10-28): C10b anchored correction, C28 splice — trivial score combination. v1 tracker preserved in EXPERIMENT_TRACKER_2026-10-28.md.
Process: discuss each run with the user before implementation; log launches/results in the research log.
