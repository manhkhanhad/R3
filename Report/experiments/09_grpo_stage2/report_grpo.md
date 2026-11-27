# E09 — R3's stage-2 GRPO and our RL variants on latent-using models
Back to [main plan](../../main.md). Dates: 2026-11-08 – 2026-11-26 (+ beam-NDCG runs submitted 2026-11-27). Status: done (negative); beam-NDCG full-model runs pending.

## Question
R3 has a stage 2: GRPO on the pooling attention, with Gaussian noise on `r` and reward −PPL of the target. Once the decoder reads `r` (MW, MW-R-H), can stage 2 improve ranking while keeping latent use? "R3's GRPO" = the original R3_ori stage-2 code (commit c558b44: noise N(0, 1.5²) on `r`, first copy noise-free, G = 8, −PPL reward, advantage vs the noise-free baseline, β 0, 2 GPUs). Every other variant here is ours.

## Setup
| run | ID | start | method / settings | code |
|---|---|---|---|---|
| stock noise-GRPO | smoke 358298 → 358299 | MW 356322 (Toys) | R3 trainer; lr 1e-5, β 0.01, 8 gen., ε 0.2/0.28, batch 256, 1 epoch, **1 GPU**; KL ≡ 0 (see below) | 15d5523e |
| GRPO + KL anchor | smoke 360236 → 360237 | MW 356322 | same + `KL_ANCHOR=1`: reference = frozen copy that builds its **own** `r`, same noise, TRL k3 | 710e2f81 |
| rank GRPO v1 | smoke 361058 → 361059 / 361060 (cancelled) | MW / R3_ori Toys | beam group (top-8 beams), reward +1 hit / −0.5/log2(k+1), KL anchor β 0.04, fp32 attention, lr 1e-5, no noise | b3f54bd3 |
| rank GRPO v2 | 362509 (eval rerun 363469) / 362510 (post 363709) | MW / R3_ori Toys | v1 + no-hit groups get advantage 0 | bffc90af, c9550c22 |
| rank GRPO v2 CDs | smoke 363659 → 363710 / 363711 | MW 362561 / R3_ori CDs | same | ed0d0635 |
| rank GRPO on MW-R-H | smoke 368197 → 368198 | MW-R-H 365504 | same as 362509 | 1a2053ff |
| rank GRPO arms A / C / A+C | smoke 374797 → 374798 / 374799 / 374800 | MW-R-H 365504 | A: target injected into no-hit groups (G = 16); C: history −0.5 / redundant-beam −0.25; lr 3e-5 | f3409399 |
| HiLaR-style | probe 376905 / 377036 (fixed); smoke 378363 → 378364 | MW-R-H 365504 | Gaussian policy on `r`, σ = 1.0·D/√d, antithetic G = 12; reward hit + .5 prefix + .5 F1 + .1 SASRec z + dLL; latent KL β .05; lr 5e-6 | 6af2c9c8, 0bbb59bc, 55908a7e |
| **exact R3 replica** | smoke 371451 → 371452 (interv. 374331) | R3_ori Toys | original code, 2×A40, lr 1e-5, G 8, β 0, batch 256, micro 8, seed 42 | c558b44 (+ NCCL env 2c33ae98) |
| **exact R3 on MW-R-H** | smoke 380282 → 380283 (lr 5e-5), 380284 (lr 1e-5) | MW-R-H 365504 | same | c558b44, sbatch 09716616 |
| exact R3, 5 seeds | 380284 (s42), 382413–382416 (s1–s4) | MW-R-H 365504 | lr 1e-5 | c558b44 |
| exact R3, lr search | 383021 / 383022 / 383023 | MW-R-H 365504 | lr 2e-5 / 3e-5 / 4e-5, seed 42 | c558b44 |

- Only the pooling attention trains; the decoder is frozen (verified bit-identical).
- Toys test n = 6,738 (copy rows 836 under the `compare.py` rule used by these tables). CDs n = 6,157.

**KL bug (stage-2 KL structurally zero).** Stock R3 computes the reference log-probs from embeddings that already contain the policy's `r` (+ noise). With a frozen decoder, π_ref = π_θ exactly, so KL ≡ 0 with zero gradient and there is no trust region. The anchor fix lets the reference copy compute `r` with the frozen stage-1 attention. CPU check: stock KL 0.0, anchor KL 7.1e-5.

**Audit 2026-11-19.**
- Trainer code is identical to R3_ori except the optional KL anchor.
- R3 runs 2 GPUs: its advantage baseline is the mean noise-free reward of **two** users' groups, normalised over 16 values. All our earlier stock runs (L9, 358299, 360237) used 1 GPU, so **none of them is an exact replica**.
- Rank-GRPO runs are a different method (no noise, beam groups).

**bf16 finding.** In 358299 / 360237 only 13–24% of the attention weights ever changed: bf16 rounds most lr-1e-5 updates away. The rank-GRPO runs use fp32 attention.

## Results

### GRPO on MW (Toys, start 356322)
| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MW stage 1 356322 | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| stock GRPO 358299 (KL ≡ 0) | 0.0223 | 0.0433 | 0.0565 | 0.0781 | 0.0345 | 0.0400 | 0.0469 |
| GRPO + KL anchor 360237 | 0.0362 | 0.0603 | 0.0766 | 0.0962 | 0.0502 | 0.0569 | 0.0632 |
| rank GRPO v2 362509 | 0.0381 | 0.0637 | 0.0809 | 0.1072 | 0.0530 | 0.0601 | 0.0686 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap (t-test p not recorded for these runs):
- **358299 vs stage 1:** HR@10 **−0.0249** [−0.0295, −0.0202], NDCG@10 −0.0199 sig; vs A −0.0272 sig.
  - Latent tests (raw − mode HR@10): mask +0.0255 sig, mean +0.0039 [−0.0009, +0.0086] **n.s.**, shuffle +0.0416 sig, i.e. partial collapse.
  - Last step kl 0.0, grad_norm 0.90.
- **360237 vs stage 1:** HR@10 **−0.0068** [−0.0095, −0.0043], NDCG@10 −0.0036 sig. vs 358299 +0.0181 sig; vs A −0.0091 sig.
  - Latent tests: mask 0.043, mean **0.019** sig, shuffle 0.064 [+0.0580, +0.0708].
  - kl 8.9e-5 → peak 0.186 → 0.031: the anchor is active.
- **362509 (rank GRPO v2):** vs start HR@10 **+0.0042** [+0.0019, +0.0064], NDCG@10 **+0.0018** [+0.0008, +0.0029]; vs R3_ori +0.0049 n.s.; vs A +0.0019 n.s.
  - Latent tests (363469): mask −0.0545, mean −0.0282, shuffle −0.0712, all sig.
- **Rank GRPO v1 (361059 / 361060)** was cancelled at 795 / 750 steps: train group hit rate fell (MW .275 → .212; R3_ori .223 → .175). Rank-only rewards in no-hit groups push down whatever ranks first.

### Rank GRPO v2: MW arm vs content-free R3_ori arm
| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys R3_ori (start) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |
| Toys rank GRPO R3_ori 362510 | 0.0389 | 0.0634 | 0.0793 | 0.1098 | 0.0530 | 0.0595 | 0.0694 |
| CDs R3_ori (start) | .0495 | .0707 | .0819 | .0996 | not recorded | not recorded | .0719 |
| CDs MW 362561 (start) | .0525 | .0726 | .0851 | .1049 | .0640 | .0691 | .0754 |
| CDs rank GRPO MW 363710 | .0536 | .0724 | .0854 | .1069 | .0644 | .0698 | .0766 |
| CDs rank GRPO R3_ori 363711 | .0546 | .0758 | .0880 | .1095 | .0666 | .0717 | .0786 |

| arm | HR@10 | gain vs own start (bootstrap) | shuffle effect |
|---|---|---|---|
| Toys MW 362509 | 0.1072 | +0.0042 sig | −0.071 (used) |
| Toys R3_ori 362510 | 0.1098 | +0.0076 [+0.0046, +0.0104] sig | 0.000 (mask −0.017 sig; mean / shuffle n.s.; 12 distinct `r` over 53,898 rows) |
| CDs MW 363710 | 0.1069 | +0.0019 [−0.0005, +0.0044] n.s. (vs R3_ori +0.0080 sig) | −0.008 (mask / mean / shuffle +0.0135 / +0.0050 / +0.0081 sig) |
| CDs R3_ori 363711 | 0.1095 | +0.0106 [+0.0070, +0.0140] sig | 0 (mask +0.0365 sig; mean / shuffle n.s.) |

Correction 2026-11-22: the R3_ori arm's gain is mostly on non-copy rows (Toys copy +.018 sig, non-copy +.006 sig; non-copy gives +.0053 of the +.0076).

### Toys, starting from MW-R-H 365504 (HR@10 .1086 / NDCG@10 .0698, shuffle −.035)
Full table (`analysis/results/grpo_row_types/full_table.py`; Δ vs start with p_bootstrap / p_t):

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | ΔHR@10 | ΔNDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| start MW-R-H 365504 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | — | — |
| beam rank GRPO 368198 | .0381 | .0681 | .0827 | .1089 | .0555 | .0615 | .0700 | +.0003 (.800 / .768) | +.0002 (.647 / .656) |
| rank GRPO + A 374798 | .0370 | .0652 | .0795 | .1043 | .0534 | .0594 | .0674 | **−.0043** (.002 / .003) | **−.0024** (.001 / .002) |
| rank GRPO + C 374799 | .0315 | .0586 | .0723 | .0959 | .0471 | .0527 | .0603 | **−.0128** (.000 / .000) | **−.0095** (.000 / .000) |
| rank GRPO + A+C 374800 | .0356 | .0631 | .0803 | .1033 | .0516 | .0587 | .0660 | **−.0053** (.000 / .001) | **−.0037** (.000 / .000) |
| HiLaR-style 378364 | .0365 | .0650 | .0812 | .1077 | .0531 | .0597 | .0683 | −.0009 (.500 / .453) | **−.0015** (.020 / .020) |
| R3 exact lr 1e-5 s42 (380284) | .0380 | .0689 | .0828 | .1095 | .0557 | .0614 | .0701 | +.0009 (.170 / .157) | +.0003 (.380 / .412) |
| R3 exact lr 1e-5 s1 | .0380 | .0678 | .0828 | .1095 | .0552 | .0614 | .0700 | +.0009 (.147 / .134) | +.0002 (.494 / .527) |
| R3 exact lr 1e-5 s2 | .0378 | .0681 | .0831 | .1083 | .0553 | .0615 | .0697 | −.0003 (.782 / .655) | −.0001 (.733 / .710) |
| R3 exact lr 1e-5 s3 | .0381 | .0680 | .0831 | .1091 | .0554 | .0616 | .0700 | +.0004 (.543 / .491) | +.0002 (.456 / .492) |
| R3 exact lr 1e-5 s4 | .0380 | .0677 | .0830 | .1089 | .0551 | .0614 | .0698 | +.0003 (.711 / .655) | −.0000 (.953 / .927) |
| R3 exact lr 2e-5 (383021) | .0230 | .0493 | .0668 | .0950 | .0382 | .0454 | .0544 | **−.0137** (.000 / .000) | **−.0153** (.000 / .000) |
| R3 exact lr 3e-5 (383022) | .0071 | .0191 | .0301 | .0473 | .0139 | .0184 | .0240 | **−.0613** (.000 / .000) | **−.0458** (.000 / .000) |
| R3 exact lr 4e-5 (383023) | .0255 | .0481 | .0614 | .0841 | .0385 | .0440 | .0513 | not recorded (far below start) | not recorded |
| R3 exact lr 5e-5 (380283) | .0197 | .0383 | .0484 | .0701 | .0303 | .0345 | .0415 | **−.0386** (.000 / .000) | **−.0283** (.000 / .000) |
| ref R3_ori stage 1 (run 1) | .0377 | .0607 | .0778 | .1023 | .0510 | .0580 | .0659 | −.0064 (.027 / .028) | −.0039 (.019 / .025) |
| ref R3_ori stage 2 lr 5e-5 (run 1) | .0380 | .0628 | .0784 | .1055 | .0522 | .0586 | .0674 | −.0031 (.278 / .276) | −.0024 (.161 / .169) |

**Five seeds at lr 1e-5.** Mean HR@10 +.0004 [−.0005, +.0014] (p .33), NDCG@10 +.0001. No seed differs from the start at any cutoff (each p > .13); the seed spread is .0012 HR@10.

Latent tests (raw − intervention HR@10):

| run | mask | mean | shuffle | notes |
|---|---|---|---|---|
| start MW-R-H 365504 | −.0297 | −.0294 | −.0343 | |
| 368198 | −.031 | −.030 | −.033 | |
| A 374798 | −.0252 | −.0252 | −.0260 | |
| C 374799 | −.0171 | −.0295 | −.0251 | |
| A+C 374800 | −.0242 | −.0255 | −.0261 | |
| HiLaR 378364 | −.0297 | −.0294 | −.0343 | all sig |
| R3 exact lr 1e-5 (s42) | −.030 | −.030 | −.033 | |
| R3 exact lr 5e-5 | +.0086 (masking helps) | −.0101 | −.0065 | latent use destroyed |

The HiLaR row matches the start to four digits; both are copied as recorded in the log.

**Exact R3 replica on R3_ori Toys (371452).**

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| R3 exact 371452 | .0375 | .0619 | .0779 | .1012 | .0515 | .0581 | .0656 |

- vs stage 1: HR@10 −.0010 [−.0027, +.0006], p .25; all CIs span 0.
- vs 362510: significantly worse (HR@10 −.0086).
- Interventions 374331: mask −.0086 sig, mean / shuffle n.s.

**R3_ori's own stage 2** (released checkpoints, `analysis/results/r3_ori_stage2_check/`; stage 2 − stage 1, HR@10 (NDCG@10)):
- Toys run 1: lr 5e-5 +.0033 sig (+.0015 sig); lr 1e-5 −.0004 n.s.; lr 1e-4 +.0046 sig.
- Toys run 2 (lr 5e-5): **collapsed** (HR@10 .0024).
- Toys run 3: +.0009 n.s.
- CDs: run 1 +.0106 sig (+.0087 sig); run 2 +.0002; run 3 −.0016. 3-run mean +.0030 [+.0016, +.0045] sig, driven by run 1.
- Stage-1 spread on Toys: HR@10 .1023 / .1051 / .1082.

### Diagnostics
- **Weight drift** of the 6 pooling-attention tensors (‖W‖ ≈ 54):
  - MW-R-H at lr 1e-5: ‖Δ‖ .20–.25 (0.4%), 5 seed updates mutually aligned (cos .53–.63). At lr 5e-5: 4.44 (8.2%), cos .06 with the 1e-5 direction.
  - R3_ori Toys run 1: 1e-5 → .55, 5e-5 → 2.15, 1e-4 → 4.41. R3_ori CDs at 5e-5: 4.41 / 1.86 / 3.95 (all fine).
  - Drift size does not separate gain from collapse (CDs run 1 4.41 fine vs MW-R-H 4.44 collapsed). The decoder's sensitivity to `r` does.
- **Grad norm** (mean over run): lr 1e-5 .07, 2e-5 .38, 3e-5 .24, 5e-5 .93; not monotone with the outcome. At lr 5e-5 it grew from .03–.05 to 4.8 by epoch .36.
- **Why rank GRPO adds nothing on MW-R-H:**
  - 368198 keeps the same top-1 on 87% of rows; the hit sets differ by 22 / 24 rows.
  - The reward is starved: non-copy train rows have a hit 21% (test 4.9%; memorised), hits at rank ≥ 2 only 14%.
  - MW-R-H already sits at the post-GRPO non-copy level of R3_ori.
  - A injects train targets (overfits). C's redundancy penalty removes same-product variants that copy rows need (copy HR@10 −.068).
- **HiLaR probe** (377036, corrected; σ_rel 1.0 on MW-R-H):
  - noise / ‖r‖ 1.35; 1.52 distinct titles per 12 rollouts; P(≥ 2) .36; hit .303 (clean .301); mean dLL −.137; combined reward std .143.
  - The first probe 376905 had an invalid dLL column (transformers 4.48 ignores logits_to_keep).
  - Training reward .450 → .554, entirely from dLL (−.115 → −.014); noisy hit .141 → .151; drift ‖μ − μ_ref‖ 0.4 → 16.
- **Pending (2026-11-27):** beam-NDCG reward, full-model GRPO on MW-R-H (smoke 383569 passed, code 071e729c; stock-advantage bug under a hit/miss reward fixed with a group baseline). lr 3e-7 / 1e-6 / 3e-6 + fp32 start eval submitted; IDs not recorded.

## Reading
1. **On a model whose decoder reads `r`, stock R3 stage 2 is harmful** without a trust region: 358299 lost −0.025 HR@10 and partly collapsed (the mean swap no longer hurts). That is because the stage-2 KL is structurally zero.
2. A real KL anchor (360237) removes 73% of the damage and restores per-user content, but still ends −0.007 below its start. The −PPL-under-noise reward does not improve ranking once `r` is used.
3. **Rank GRPO** (our design) improves MW on Toys (+0.0042 sig), but it gains more on R3_ori, whose `r` is a constant shared soft prompt (Toys +0.0076, CDs +0.0106). The gain is a generic ranking-objective gain through one shared vector, not use of per-user content. On MW-R-H it is a no-op (368198), and its A / C / A+C extensions are significantly worse. The user dropped this recipe (2026-11-24, code removed in 6284ef8b); it is history only.
4. **R3's exact GRPO** has two regimes:
   - lr ≤ 1e-5: a seed-independent no-op on MW-R-H (5 seeds, mean +.0004 n.s.) and on R3_ori (371452);
   - lr ≥ 2e-5: destructive on MW-R-H (2e-5 −.0137, 3e-5 −.0613, 4e-5 .0841, 5e-5 −.0386, latent use destroyed).

   No lr in between helps. R3_ori's own stage-2 gains are small, inconsistent across runs and below the stage-1 seed spread, so the RL ablation in R3's paper (one run per cell) is not reproduced as a robust effect.
5. **HiLaR-style GRPO** is stable and keeps latent use (shuffle −.034), but gives no gain (NDCG@10 −.0015 sig). Its reward gain is noise robustness (dLL).
6. **Net:** no GRPO variant improves MW-R-H. Stage 2 can be dropped from the method; stage 1 (MW-R-H) already reaches the post-GRPO R3_ori level while using `r` (368198 vs R3_ori HR@10 +.0067 sig).

Caveats: one run per lr in the lr search (non-monotone: 3e-5 lowest); 358299 / 360237 were 1-GPU (not exact); no t-tests for the 11-08 to 11-13 runs.

## Paper use
- **Main / limitations paragraph:** R3's exact stage 2 on MW-R-H (5 seeds at lr 1e-5 = no-op; lr ≥ 2e-5 collapses), plus the R3_ori stage-2 reproducibility check. This justifies reporting stage-1 results only.
- **Appendix:** the KL-structurally-zero bug and anchor fix (358299 → 360237); weight-drift analysis; rank GRPO R3_ori-control finding (a generic gain via a shared soft prompt) as a caution about RL gains in latent recommenders.
- **Dev only / not a method:** rank GRPO v1 / v2, 368198, A / C / A+C (user dropped 2026-11-24), HiLaR-style port, beam-NDCG runs (pending).

## Sources
- Research log `analysis/document/research_notes/Latent utilization/research_log.md`:
  - GRPO on MW design: 1757–1759, 1780;
  - 358299 result and KL-anchor design: 1907–1935;
  - 360237: 2054–2066;
  - rank GRPO design / v1 / v2: 2099–2168;
  - mid-run: 2351;
  - 362509: 2355–2377;
  - 362510 / 363709: 2462–2525;
  - CDs pair: 2541–2595;
  - 368198 setup: 2717–2726; result + audit: 2765–2785;
  - exact replica: 2787–2798, 2850–2862;
  - why rank GRPO fails and A / C arms: 2908–2924;
  - HiLaR probe: 3076–3107, 3222–3226;
  - A / C / A+C / HiLaR results, removal, R3_ori stage-2 check, exact R3 on MW-R-H: 3227–3295;
  - 5 seeds, weight drift: 3324–3336;
  - 2026-11-26 SUMMARY and full table, lr 4e-5: 3411–3477;
  - beam-NDCG submission: 3615–3633.
- Memory: `r3-rank-grpo.md`, `feedback-r3-grpo-means-original-code.md`.
- Result files:
  - `output_dir/grpo_stage2/{toys_margin_wider_grpo,toys_margin_wider_grpo_kl,toys_mw_rankgrpo2,toys_r3ori_rankgrpo2,cds_mw_rankgrpo2,cds_r3ori_rankgrpo2}/`;
  - `output_dir/r3_exact_grpo/`;
  - `analysis/results/{mwrh_rankgrpo,r3_exact_grpo,rank_grpo_AC,r3_exact_mwrh,r3_ori_stage2_check,grpo_row_types,hilar_probe}/`.

