# E05 — From MW to MW-R-H: fixing the copy-row tail with a history-only reference
Back to [main plan](../../main.md). Dates: 2026-11-11 – 2026-11-17. Status: done (Toys design; CDs default seed).

## Question
MW (E04) makes the latent used but loses copy rows, because the history-swap hinge also acts on rows the history already
explains. Which per-row allocation of the hinge keeps the latent used, recovers copy rows and keeps the non-copy gain,
**without** a hand-coded copy rule?

## Setup
- Toys, test n = 6,738 (copy_row 905 by `motive_contrastive.py::copy_row`, used for logging/analysis only, except in
  the -nc runs). Baseline plain R3 (A) 345857: HR@10 0.1052 / NDCG@10 0.0667, copy 0.513, non-copy 0.0420.
- Trainer `src/latent/latent_attention_train_motive.py` + model `src/latent/motive_contrastive.py`, via
  `scripts/margin_wider_collab.sbatch METHOD=motive`, base Qwen/Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs, last
  epoch kept, m = 0.2, WIDER 0.1. Two bases: **MWC-M base** (same-motive InfoNCE `MOTIVE_W=0.1` from epoch 2; M3a, M3b,
  M3a-n, M4, M6, M7, MWC-M-nc) and **MW base** (`MOTIVE_W=0`; MW-nc, MW-R, MW-R-H and variants). Reference MWC-M 362616:
  HR@10 0.1057 / NDCG@10 0.0670, copy 0.486, non-copy 0.0466, latent used (−0.084 to −0.090).
- Codes: 364d6927 (M3a/M3b), 253fba16 (M3a-n), 5ee286e8 (M4), 032f5a2c → 62872468 (M6), a89b8680 (M7), d0b10de8 (skipcopy),
  d15f6b02 (reference scoring), efc11712 (refweight; MW-R and MW-R-H 365504), 8a62b675 (history-only reference retrain),
  4e9c30f0 (MW-3b latent reference), 6b7c02b6 (band, rank^2), 9b1ebf33 (swap gap), f3292730 (TRAIN_SEED, MW-R-H-L).

### Final recipe: MW-R-H (contribution 1)
```
CODE_ROOT=output_dir/margin_wider_collab_code/code_efc11712 METHOD=motive MOTIVE_W=0 CATEGORY=Toys_and_Games
BASE_MODEL=Qwen/Qwen2.5-1.5B MARGIN=0.2 MARGIN_WEIGHT=1.0 WEIGHT=0.1 MARGIN_MODE=refweight REF_WEIGHT_FN=rank REF_TAU=0.5
REF_NLL_FILE=output_dir/margin_wider_collab/reference_nll/Toys_and_Games_nolatent/train.pt   (NUM_EPOCHS 3, LR 3e-4 default)
```
1. **Reference** (`scripts/no_latent_reference.sbatch`, job 365404): R3 without the latent slot (no-latent SFT, base
   Qwen2.5-1.5B, plain R3 prompt minus `<|Thought|>`, verified identical on 5 rows), trained on the same train rows.
   `src/latent/reference_score.py::score` stores NLL_ref,i = (1/|y_i|) Σ_t −log p_ref(y_i,t | x_i, y_i,<t) per train row.
2. **Weight** (`latent_attention_train_motive.py::reference_weights`, fn = rank, power 1):
   w_i = rank(NLL_ref,i) / (N − 1) ∈ [0, 1] over the N = 53,898 train rows (argsort of argsort). Low NLL_ref (the
   history explains the target) → small weight. REF_TAU is used only by `sigmoid` and has no effect with `rank`.
   Toys mean weight 0.500; copy rows 0.181 vs non-copy 0.569.
3. **Loss** (`motive_contrastive.py`, margin_mode refweight), per micro-batch b = 1..4:
   h_b = relu(0.2 + NLL_b(r_b) − NLL_b(sg r_π(b))), π = roll by 1 (two-sided in the decoder);
   L = mean_b NLL_b(r_b) + 1.0 · Σ_b w_b h_b / max(Σ_b w_b, 1e-6) + 0.1 · WIDER(r).
   Inference = plain R3. MW-R-H-L adds `MARGIN_EPOCHS=2 LATE_MARGIN_WEIGHT=0.3` (hinge × 0.3 in epoch 3).

## Results

### Ablation table (Toys; Δ vs plain A as diff (p_t / p_b); copy/non-copy HR@10; latent = raw − mask / mean / shuffle HR@10)

| step | variant (run) | base | HR@10 | NDCG@10 | ΔHR@10 | ΔNDCG@10 | copy | non-copy | latent | verdict |
|---|---|---|---:|---:|---|---|---:|---:|---|---|
| 0 | MW (356322) | MW | .1030 | .0668 | −.0022 (.43/.47) | +.0001 (.96/.94) | .479 | .0447 | .051/.031/.073 | used, copy loss |
| 1 | M3a self-weighted hinge w = sg(1 − e^{−NLL(y\|x,r_other)}) (363141) | MWC-M | .1080 | .0684 | +.0028 (.31/.27) | +.0017 (.30/.29) | .506 | .0463 | .0003/.0009/−.0001 n.s. | **unused** (weight decays with CE) |
| 2 | M3b hinge epoch 1 only (363145) | MWC-M | .1080 | .0690 | +.0028 (.31/.28) | +.0022 (.18/.16) | .516 | .0447 | .0013/.0006/.0010 n.s. | **unused** |
| 3 | M3a-n normalised weight min(2, u/EMA(u)) (363638) | MWC-M | .0960 | .0609 | **−.0092 (.002/.005)** | **−.0059 (.001/<.001)** | .472 | .0377 | .0635/.0414/.0469 | used, ranking sig worse |
| 4 | M4 per-token novel hinge (363511) | MWC-M | .1005 | .0608 | −.0047 (.09/.10) | **−.0059 (.001/.001)** | .482 | .0413 | .033/.033/.032 | used, @5 sig worse |
| 5 | M6 adaptive one-sided min(0.2, 0.5·sg NLL_other) (363594) | MWC-M | not recorded | | | | | | | **dropped**: gap never formed (hinge active 100%, margin 0.181 ≈ m_i 0.180); one-sided stop-gradient hinge = CE; cancelled |
| 6 | M7 Lagrangian δ 0.05 (364879) | MWC-M | .1077 | .0686 | +.0025 (.37/.38) | +.0019 (.25/.24) | .496 | .0475 | .017/.010/.015 | weak (train gap ≠ test) |
| 6 | M7 Lagrangian δ 0.10 (364880) | MWC-M | .1070 | .0686 | +.0018 (.51/.49) | +.0019 (.24/.23) | .502 | .0458 | .005/.008/.009 | weak |
| 7 | MWC-M-nc skip-copy rule (364875) — dev ceiling | MWC-M | .1043 | .0669 | −.0009 (.75/.79) | +.0002 (.91/.88) | .497 | .0434 | .034/.054/.063 | used; copy NOT recovered |
| 7 | **MW-nc skip-copy rule (364876) — dev ceiling** | MW | .1109 | .0697 | **+.0056 (.045/.049)** | +.0030 (.074/.067) | .515 | .0482 | .029/.016/.037 | used, copy recovered |
| 8 | MW-R sigmoid, base-LM reference (365390) | MW | .1036 | .0666 | −.0016 (.55/.57) | −.0001 (.95/.96) | .493 | .0432 | .024/.035/.031 | used, gain lost |
| 8 | MW-R rank, base-LM reference (365389) | MW | .1046 | .0689 | −.0006 (.83/.86) | +.0022 (.19/.19) | .501 | .0432 | .042/.046/.045 | used, gain lost |
| 9 | MW-R-T (MW-3b latent reference 365403) | — | no training run | | | | | | | reference scored only (AUROC below); user required a history-only reference |
| 10 | **MW-R-H rank (365504)** | MW | **.1086** | **.0698** | +.0034 (.22/.23) | +.0031 (.068/.067) | .504 | **.0473** | **.030/.030/.035** | **used, all @5/@10 ≥ R3, non-copy +.0053 sig** |
| 11 | band U = 1.5 (365897) | MW-R-H | .1079 | .0689 | +.0027 (.32/.33) | +.0021 (.19/.18) | .526 | .0430 | .011/.021/.038 | band never active = replicate of 365504 |
| 12 | rank^2 (365898) | MW-R-H | .1027 | .0660 | −.0025 (.37/.41) | −.0008 (.64/.66) | .482 | .0439 | .040/.008/.022 | worse, copy −.031 sig; closed |
| 13 | MW-R-H-L seed 2 (367253) | MW-R-H | .1070 | .0692 | +.0018 (.52/.54) | +.0024 (.14/.14) | .498 | .0463 | .023/.022/.039 | rescues seed 2 |
| 13 | MW-R-H-L default (367254) | MW-R-H | .1072 | .0686 | +.0019 (.49/.49) | +.0019 (.25/.24) | .510 | .0446 | .042/.037/.042 | stable |

p_t/p_b and the non-copy values for 367254 / rank^2 were computed read-only from the existing `eval_raw.json` files
(bootstrap CIs equal the log's). Latent tests: all non-"n.s." entries are significant (bootstrap). M3a-n latent test from
the log; M6 has no result file.

### MW-R-H all cutoffs and seeds (Toys; CDs default seed)

| run | GPU | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy | non-copy | mask/mean/shuffle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plain A 345857 | | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .513 | .0420 | — |
| no-latent reference 365404 | | .0377 | not recorded | .0785 | .1039 | not recorded | .0585 | .0667 | not recorded | not recorded | — |
| MW-R-H 365504 (default) | A40 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | .504 | .0473 | .030/.030/.035 |
| MW-R-H seed 1 365887 | L40S | .0404 | .0663 | .0837 | .1101 | .0553 | .0625 | .0710 | .514 | .0475 | .054/.080/.080 |
| MW-R-H seed 1 368189 | A40 | .0404 | .0647 | .0804 | .1088 | .0544 | .0608 | .0700 | .510 | .0465 | .052/.027/.051 |
| MW-R-H seed 2 365888 | L40S | .0374 | .0635 | .0763 | .0980 | .0525 | .0577 | .0648 | .474 | .0396 | .061/.037/.043 |
| MW-R-H seed 2 366001 | A40 | .0334 | .0574 | .0717 | .0944 | .0471 | .0530 | .0603 | .463 | .0372 | .050/.051/.047 |
| MW-R-H-L seed 2 367253 | A40 | .0381 | .0669 | .0806 | .1070 | .0550 | .0606 | .0692 | .498 | .0463 | .023/.022/.039 |
| MW-R-H-L default 367254 | A40 | .0386 | .0652 | .0809 | .1072 | .0538 | .0602 | .0686 | .510 | .0446 | .042/.037/.042 |
| **CDs** R3_ori stage 1 | | .0495 | .0707 | .0819 | .0989 | .0616 | .0662 | .0717 | .581 | .0638 | — |
| **CDs MW 362561** | A40 | .0525 | .0726 | .0851 | .1049 | .0640 | .0691 | .0754 | .620 | .0674 | .013/.007/.004 |
| **CDs MW-R-H 366023** | A40 | .0555 | .0801 | .0906 | .1121 | .0696 | .0739 | .0808 | .624 | .0748 | .024/.015/.017 |

- Toys vs plain A (p_t / p_b): 365504 ΔHR@5 +.0033 (.19/.20), ΔNDCG@5 +.0030 (.079/.080); seed 1 365887 **ΔNDCG@5 +.0040
  (.022/.012), ΔNDCG@10 +.0043 (.012/.012)**; 368189 **ΔNDCG@10 +.0032 (.041/.031)**; seed 2 365888 **ΔHR@10 −.0073 (.008/.006)**;
  366001 **all four sig below** (ΔHR@10 −.0108, p < .001). 365504 non-copy vs plain A +.0053 **sig**, copy −.009 n.s.
- CDs 366023 vs R3_ori: **ΔHR@5 +.0088 (.001/.002), ΔNDCG@5 +.0078 (<.001), ΔHR@10 +.0132 (<.001) [+.0070, +.0188],
  ΔNDCG@10 +.0091 (<.001)**; vs MW all four sig (ΔHR@10 +.0071, ΔNDCG@10 +.0054). Log: copy +.048 sig, non-copy +.0106 sig
  vs R3_ori (418-row recomputation for this report: .624 − .581 = +.043, .0748 − .0638 = +.0110).
- CDs copy/non-copy values in the table: 418-row split recomputed read-only. The 0.1086/0.1079/0.1101/0.0980/0.0944
  spread over 5 Toys MW-R-H runs (mean 4 runs HR@10 .1028 / NDCG@10 .0665) is the seed spread.

### Reference gate: AUROC of −NLL_ref for the copy rule (train / valid / test)

| reference | job | AUROC | copy vs non-copy rank weight | used for |
|---|---|---|---|---|
| base LM Qwen2.5-1.5B (untrained on rows) | 365378 | .914 / .900 / .892 | .159 vs .574 (sigmoid τ 0.5: .101 vs .586) | MW-R |
| MW-3b final model, latent slot kept (MW-R-T) | 365403 | .878 / .831 / .795 | not recorded | comparison only |
| **history-only no-latent SFT** | **365404** | **.888 / .829 / .785** | **.181 vs .569** (sigmoid ratio .72: useless) | **MW-R-H** |
| CDs history-only | 365885 | .875 / .858 / .866 | not recorded | CDs MW-R-H |

Gate: train AUROC ≥ 0.80. The base LM's lowest-20% NLL captures 73% of copy rows (MW-3b reference 66%); trained
references show memorisation (MW-3b train NLL copy .11 vs non-copy .33, valid .30 vs .73).

### Swap-gap analysis (365511, code 9b1ebf33): gap = NLL(y|x,r_other) − NLL(y|x,r_own), per-token mean, test

| model | copy gap mean (median) [q90] | non-copy gap mean (median) [q90] | copy HR@10 |
|---|---|---|---:|
| WIDER only | +0.003 | −0.000 | not recorded |
| MW-3b | +0.001 | −0.000 | .515 |
| **MW** | **+2.11** (1.24) [5.61] | **+1.40** (0.89) [3.31] | .478 |
| MW-nc | +1.13 (1.08) [2.15] | +1.08 (0.95) [2.11] | .515 |
| MWC-M / MWC-M-nc | +2.27 / +2.31 | +2.13 / +2.59 | — / .497 |
| MW-3b-L4 | +2.75 | +3.00 | .499 |
| M7 δ .05 / .10 | +0.061 (0.005) / +0.210 (0.019) | +0.063 / +0.108 | — |

Train ≈ test for every model (MW train copy +1.93 vs non-copy +1.50).

## Reading
- **Absolute gates fail on scale.** M3a's weight 1 − e^{−NLL} falls with CE on every row (mean 0.87 → 0.32, copy 0.67 →
  0.14), the hinge is never satisfied (margin 0.06) and the latent ends unused; its copy "recovery" is plain R3's.
  M3a-n fixes the scale (mean ≈ 1, copy 0.60 vs non-copy 1.05) but WIDER jumps in epoch 2 (0.054 → 0.477) and ranking
  drops significantly. M3b and MW-3b: once the hinge stops, dependence vanishes. Copy recovery and latent use moved together.
- **Token-level and one-sided fixes fail by construction.** M4: copy rows still carry 16.5% novel tokens, exactly the
  variant tokens that decide sibling slots. M6: the gradient of [m + NLL_pos − sg(NLL_neg)]₊ equals the CE gradient, so
  no contrast forms (dropped by the user 2026-11-12; never reuse). M7 meets its train-gap constraint with μ ≈ 0, but
  test dependence is weak (median gap ≈ 0; a few rows carry it).
- **The rule-based ceiling isolates the cause.** MW-nc (no hinge on copy rows, MW base) recovers copy rows fully (.515)
  with the latent used and HR@10 significantly above R3. With the motive term (MWC-M-nc) copy rows do not recover
  (.497): the shared decoder over-trusts `r`. Hence the adaptive replacement targets the **MW base**. The -nc runs are
  development references only, never a paper method.
- **The swap gap says what MW does wrong:** MW's copy gap (+2.11, q90 5.61) exceeds its non-copy gap (+1.40), i.e. `r`
  carries copy identity; MW-nc has a balanced gap (1.13 vs 1.08, q90 2.15). Copy-row loss tracks the copy-row gap.
- **Reference weighting replaces the rule.** A base-LM reference (MW-R) separates copy rows best (AUROC .91) but moves
  copy rows only halfway (.501) and loses the non-copy gain (.0432). The **history-only** reference (MW-R-H), which knows
  what this model class can read from the history, separates slightly less (AUROC .888 train) yet gives the best
  adaptive result: all @5/@10 above R3 (n.s., NDCG@10 p_t .068), non-copy +.0053 sig, copy −.009 n.s., latent used. It
  approaches the MW-nc ceiling (.1109 / .0697) without a rule. On CDs it is significantly above both R3_ori and MW.
- **Variants that did not help:** band U = 1.5 was never active (MW-R-H's in-training gap is only ≈ 0.44 by the end; U
  was miscalibrated on MW's post-hoc test gap); rank^2 sharpens weights but the hinge stays unsatisfied in epoch 3 and
  copy rows drop (−.031 sig).
- **Robustness.** Seed 2 fails reproducibly (L40S and A40: HR@10 .0980 / .0944, copy .474 / .463). Failure signature:
  epoch-3 margin loss .118–.140 (vs .024), WIDER rising (.28–.44 vs .076), grad norm 4–5 (vs 1.4). GPU is not the cause.
  MW-R-H-L (hinge × 0.3 in epoch 3) rescues seed 2 (.1070, latent used) and keeps the default seed stable (.1072), but
  training is not deterministic on the same seed/GPU (epoch-2 margin .110 vs .195 in the failed run), so part of the
  rescue may be run-to-run variation.

## Paper use
- **Method section:** the MW-R-H formula above (reference NLL → rank weight → weighted hinge + WIDER), the reference
  gate (AUROC table), and the swap-gap analysis as the mechanism (copy gap > non-copy gap under MW).
- **Main ablation table (Toys):** MW → MW-R (base-LM reference) → MW-R-H (history-only reference), plus MW-3b (no late
  hinge) and rank^2 / band as weighting ablations, with copy / non-copy and latent-test columns.
- **Appendix:** M3a, M3b, M3a-n, M4, M7 (failed principled fixes on the MWC-M base); seed table and MW-R-H-L as the
  robustness fix; MW-3b reference scoring (MW-R-T).
- **Dev only:** MW-nc and MWC-M-nc (rule-based ceilings), M6 (dropped, no results), the band run as a replicate.

## Sources
- `analysis/document/research_notes/Latent utilization/research_log.md`: 2299–2335 (M3a/M3b setup, MWC-M2, copy root
  cause), 2336–2354 (M4, margin sweep setup, M6 design), 2412–2428 (MW-R prerequisites, single-checkpoint), 2429–2437
  (first fix results), 2448–2461 (M3a, M3a-n, M6 early warning), 2485–2500 (m 0.05, M3b, M3a-n ranking), 2527–2546
  (MW-3b-L, one-sided hinge dropped, M3a-n / M4 / L4 results), 2597–2602 (skipcopy, M7 design), 2631–2639 (M7, -nc
  results, MW-R step 1), 2640–2670 (MW-R-T, MW-R-H reference 365404, swap gap 365511, MW-R results), 2671–2730 (MW-R-H
  365504, seeds, CDs reference, band, rank^2, CDs 366023, MW-R-H-L 367253/367254), 2739–2744 (CDs seeds, Toys s1 A40).
- `analysis/document/research_notes/Latent utilization/MW_to_MWC_report_2026-11-12.md`.
- Results: `output_dir/margin_wider_collab/Toys_and_Games_margin_0.2_wider_0.1_{motive*,skipcopy,refweight_*}/eval_raw.json`,
  `.../CDs_and_Vinyl_margin_0.2_wider_0.1_refweight_nolatent_rank/`, `.../reference_nll/{Toys_and_Games,Toys_and_Games_mw3b,
  Toys_and_Games_nolatent,CDs_and_Vinyl_nolatent}/`, `.../swap_gap/Toys_and_Games/`; logs `run_logs/R3_mwRH_rank.365504.out`
  (code efc11712, reference file, weights copy .181 / non-copy .569), `R3_mwR_{rank,sigmoid}.36538{9,0}.out`,
  `R3_nolatent_ref.365404.out`, `R3_refscore_mw3b.365403.out`, `R3_swapgap.365511.out`.
- Code: `src/latent/latent_attention_train_motive.py::reference_weights`, `src/latent/reference_score.py::score`,
  `src/latent/motive_contrastive.py` (forward, margin modes), `src/latent/wider_reg.py`, `scripts/margin_wider_collab.sbatch`.
- Memory notes `r3-margin-wider-collab.md`, `MEMORY.md`.
