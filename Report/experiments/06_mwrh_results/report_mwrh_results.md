# E06 — MW-R-H main results and robustness
Back to [main plan](../../main.md). Dates: 2026-11-14 – 2026-11-27. Status: done for Toys, CDs, Steam, MovieLens; Goodreads stopped.

## Question
Does MW-R-H (contribution 1) make the decoder use `r`'s content (mean and shuffle significantly hurt) while ranking at least as well as R3, on every dataset and seed? Where it fails, why?

## Setup
- **MW-R-H recipe:**
  - The history-swap hinge (m 0.2) is weighted per row by the history-only reference: rank weights, τ 0.5, sum(w·hinge)/sum(w).
  - WIDER 0.1, 3 epochs (epoch 3 kept), base Qwen2.5-1.5B, lr 3e-4, HF clip 1.0.
  - Settings: `METHOD=motive MOTIVE_W=0 MARGIN_MODE=refweight REF_WEIGHT_FN=rank`.
  - Code: f3292730 (Toys/CDs seeds; `TRAIN_SEED`), 513bae11 (Steam), 7e603e45 (MovieLens), 1fa432de (Goodreads), 66b61b81 (Steam lr/clip), 1186ca19 (MovieLens S/D), bd018410 (S on CDs/Toys).
- **References:** history-only no-latent SFT: Toys 365404, CDs 365885, Steam 369661, MovieLens 369531, Goodreads 368191.
- **Variants:**
  - MW-R-H-L: hinge ×0.3 in epoch 3 (`MARGIN_EPOCHS=2 LATE_MARGIN_WEIGHT=0.3`).
  - MW-R-H-S: + mean((1−w)·NLL with swapped `r`), `COPY_SWAP_W=1`.
  - MW-R-H-D: + mean((1−w)·KL(no-latent ref ‖ model)), `DISTILL_W=1`.
  - MW-nc: hinge off on copy_rows.
  - MW-3b: fixed hinge in epoch 1 only.
  - MW-HN: hinge negative = bank latent of a user with the same last item and a different next item.
- **Tests:** paired t-test p_t and paired bootstrap (2,000, seed 42) over all test rows. Latent tests give Δ = intervention − raw (negative = hurts); * = significant.

## Results

### Toys (n 6,738; copy 905) — robustness / seed table

| run | GPU | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy / non-copy HR@10 | ΔHR@10 vs R3 (p_t / p_b) | ΔNDCG@10 vs R3 (p_t / p_b) | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain R3 (A) | | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .513 / .0420 | – | – | not recorded |
| MW-R-H 365504 (default) | A40 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | .504 / .0473 | +.0034 (.22/.23) | +.0031 (.068/.067) | −.0301* / −.0300* / −.0349* |
| seed 1 365887 | L40S | .0404 | .0663 | .0837 | .1101 | .0553 | .0625 | .0710 | .514 / .0475 | +.0049 (.082/.078) | **+.0043** (.012/.012) | −.0543* / −.0798* / −.0795* |
| seed 2 365888 | L40S | .0374 | .0635 | .0763 | .0980 | .0525 | .0577 | .0648 | .474 / .0396 | **−.0073** (.008/.006) | −.0020 (.24/.23) | −.0611* / −.0370* / −.0426* |
| seed 1 368189 | A40 | .0404 | .0647 | .0804 | .1088 | .0544 | .0608 | .0700 | .510 / .0465 | +.0036 (.19/.19) | **+.0032** (.041/.031) | −.0521* / −.0272* / −.0512* |
| seed 2 366001 | A40 | .0334 | .0574 | .0717 | .0944 | .0471 | .0530 | .0603 | .463 / .0372 | **−.0108** (1.5e-4/<5e-4) | **−.0064** (1.6e-4/<5e-4) | −.0496* / −.0511* / −.0469* |
| MW-R-H-L seed 2 367253 | A40 | .0381 | .0669 | .0806 | .1070 | .0550 | .0606 | .0692 | .498 / .0463 | +.0018 (.52/.54) | +.0024 (.14/.14) | −.0229* / −.0221* / −.0393* |
| MW-R-H-L default 367254 | A40 | .0386 | .0652 | .0809 | .1072 | .0538 | .0602 | .0686 | .510 / .0446 | +.0019 (.49/.49) | +.0019 (.25/.24) | −.0421* / −.0371* / −.0420* |

Additional @5 tests vs R3:
- 365887: NDCG@5 +.0040 significant.
- 366001: HR@5 −.0079 and NDCG@5 −.0054 significant.
- 365504: HR@5 +.0033 (p .19), NDCG@5 +.0030 (p .08).

The band run 365897 (U 1.5) never activated its band, so it is effectively a replicate of 365504: HR@10 .1079 / NDCG@10 .0689, n.s. vs R3, shuffle −.0384*.

Means:
- **3 A40 seeds** (365504 / 368189 / 366001): HR@1/3/5/10 .0374±.0036 / .0634±.0055 / .0783±.0059 / .1039±.0083; NDCG@3/5/10 .0523±.0045 / .0584±.0047 / .0667±.0055; copy .492±.026.
- **4-run mean** (365504 / 365887 / 365888 / 366001): HR@10 .1028, NDCG@10 .0665.
- **All 5 runs**: HR@10 .1040±.0073, NDCG@10 .0672±.0045.

Failure mode of seed 2:
- Hinge unsatisfied late: margin_loss ep2 .192 vs .134, ep3 .125 vs .024 (default seed).
- WIDER jumps in epoch 3: .443 vs .076.
- Grad norm 5.0 vs 1.4.
- 366001 also had a grad spike (~78) and CE 2.46 in late epoch 1.
- Then copy rows collapse (.463–.474).
- The failure appears on both L40S and A40, so the GPU is not the cause.

MW-R-H-L rescues seed 2:
- Curve: margin .170 / .110 / .052, WIDER ep3 .039, grad 1.4.
- But its epochs 1–2 use the same recipe as the failed 366001, and the margin already differs at ep2 (.110 vs .195). So training is nondeterministic, and part of the rescue may be luck.

### CDs (n 6,157; copy 418) — main + seed table

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy / non-copy | ΔHR@10 vs R3_ori (p_t / p_b) | ΔNDCG@10 (p_t / p_b) | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R3_ori | .0494 | .0710 | .0819 | .0989 | .0617 | .0662 | .0717 | .577 / .0641 | – | – | not recorded |
| MW 362561 | .0525 | .0726 | .0851 | .1049 | .0640 | .0691 | .0754 | .620 / .0674 | **+.0060** (.029/.03) | **+.0037** (.033/.035) | −.0125* / −.0068* / −.0039* |
| MW-R-H 366023 | .0555 | .0801 | .0906 | .1121 | .0696 | .0739 | .0808 | .624 / .0748 | **+.0132** (1.4e-5/<5e-4) | **+.0091** (3.1e-6/<5e-4) | −.0239* / −.0146* / −.0167* |
| seed 1 368187 | .0554 | .0768 | .0903 | .1116 | .0677 | .0733 | .0801 | .653 / .0721 | **+.0127** (5e-5/<5e-4) | **+.0084** (2.7e-5/<5e-4) | −.0309* / −.0091* / −.0219* |
| seed 2 368188 | .0499 | .0775 | .0911 | .1121 | .0660 | .0716 | .0784 | .636 / .0739 | **+.0132** (4.6e-5/<5e-4) | **+.0067** (.001/<5e-4) | −.0219* / −.0287* / −.0318* |
| **3-seed mean ± sd** | .0536±.0032 | .0781±.0017 | .0907±.0004 | **.1119±.0003** | .0678±.0018 | .0729±.0012 | **.0797±.0012** | .638±.014 / .0736±.0013 | all seeds significant at all @5/@10 | | used 3/3 |

- HR@5 and NDCG@5 are significantly above R3_ori for every seed.
- vs MW: HR@10 significant in 3/3 seeds, NDCG@10 in 2/3 (seed 2 NDCG n.s.).
- 366023 gains on both row types: copy +.048* and non-copy +.0106*.

### Steam (n 10,303) — collapse test

| run | lr / clip | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| R3 371447 | | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 | −.0141* / +.0003 / −.0007 |
| no-latent 369661 | | .0065 | .0139 | .0208 | .0357 | .0108 | .0136 | .0184 | – |
| MW-R-H 369662 (collapsed) | 3e-4 / default | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 | −.0117* / −.0009 (p .15) / −.0007 (p .35) |
| MW-R-H 383065 | 3e-4 / 1.0 explicit | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 | −.0292* / −.0116* / −.0094* |
| MW-R-H 383066 | 1e-4 / 1.0 | .0116 | .0231 | .0302 | .0425 | .0182 | .0211 | .0251 | −.0079* / +.0009 / −.0026 (p .07/.08) |
| MW-R-H 383067 | 5e-5 / 1.0 | .0102 | .0193 | .0248 | .0360 | .0154 | .0177 | .0213 | −.0066* / −.0034* (p .007) / −.0070* |

Significance vs R3:
- **369662:** HR@1/3/5 and NDCG@3/5/10 significant (NDCG@10 +.0031, p_t .0016 / p_b .001); HR@10 +.0030 (p .09).
- **383065:** HR@1 +.0022, NDCG@3 +.0021, NDCG@5 +.0020 significant; HR@10 +.0005 (.78) and NDCG@10 +.0017 (.096/.090) n.s.
- **383066:** all cutoffs significant (HR@10 +.0067, NDCG@10 +.0059, p < 5e-4), but content is unused.
- **383067:** HR@1 +.0025, HR@3 +.0042, NDCG@3 +.0035, NDCG@5 +.0027, NDCG@10 +.0021 (.024/.032) significant; HR@5 (.10) and HR@10 +.0002 (.91) n.s.
- All three new runs are n.s. vs 369662, except 383066, which is significantly better at @3/@5/@10.

Trajectories:
- 369662: the latents collapse to one direction at steps ~150–200. At that point CE spikes to 4.36 and grad norm reaches 80, and WIDER stays ≈ 1.0 (final cos to the shared direction 1.0000).
- Inference: the WIDER gradient is zero when all latents are identical (z = q), and the hinge cannot separate identical latents, so nothing pulls the run back.
- 383065, at the same settings, did not collapse (WIDER .06 from step ~100). So the collapse is run-to-run instability, not lr alone; a different clip default in 369662 is not ruled out.
- Hinge at epoch 2.25–2.5: 383065 .177, 383066 .162, 383067 **.021** (first Steam run where the hinge is satisfied).

| steps | WIDER 369662 / 383065 / 383066 / 383067 | CE (same order) | hinge (same order) |
|---|---|---|---|
| 101–150 | .761 / .072 / .508 / .432 | 2.55 / 1.85 / 1.65 / 1.69 | .201 / .197 / .198 / .191 |
| 151–200 | **.998** / .062 / .774 / .360 | **4.36** / 1.61 / 1.44 / 1.45 | .198 / .200 / .209 / .197 |
| 301–400 | .967 / .058 / .043 / .607 | 1.71 / 1.40 / 1.28 / 1.29 | .203 / .201 / .202 / .204 |
| ep 1.5–1.75 | .982 / .023 / .017 / .120 | – | .199 / .200 / .194 / .044 |
| ep 2.25–2.5 | 1.000 / .109 / .106 / .086 | .944 / .892 / 1.000 / 1.077 | .201 / .177 / .162 / .021 |

### MovieLens (n 10,164; copy_row = ref-NLL Q1, 2,541)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | hist / top-10 | copy / non-copy HR@10 | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| R3 371446 | .0249 | .0556 | .0806 | .1335 | .0425 | .0527 | .0698 | .48 | .4061 / .0426 | −.0923* / −.0022* / −.0027* |
| R3 seed 1 375356 | .0261 | .0581 | .0819 | .1372 | .0445 | .0542 | .0720 | | | n.s. shuffle |
| MW-R-H 369532 | .0193 | .0492 | .0701 | .1188 | .0363 | .0449 | .0605 | 1.09 | .3318 / .0478 | −.0464* / −.0159* / −.0267* |
| MW-3b 374225 | .0269 | .0617 | .0830 | .1263 | .0469 | .0557 | .0696 | .39 | .3676 / .0459 | +.0019 / +.0013 / +.0011 (n.s.) |
| MW-nc ep2 374720 | .0226 | .0474 | .0695 | .1102 | .0368 | .0458 | .0590 | | | unused (mask +.0020 n.s.) |
| MW-nc 374722 | .0228 | .0557 | .0793 | .1250 | .0417 | .0514 | .0660 | .62 | .3597 / .0467 | −.0414* / −.0335* / −.0463* |
| MW-HN 374776 | .0263 | .0588 | .0841 | .1279 | .0449 | .0552 | .0694 | .39 | .3628 / .0496 | −.0715* / −.0103* / −.0006 n.s. |
| **MW-R-H-S 375420** | .0267 | .0616 | .0887 | **.1380** | .0467 | .0578 | **.0737** | .41 | .4026 / .0498 | −.0346* / −.0356* / −.0381* |
| **MW-R-H-D 375422** | .0276 | .0576 | .0851 | **.1350** | .0446 | .0558 | **.0718** | .45 | .4093 / .0436 | −.0593* / −.0694* / −.0729* |

Significance vs R3 (seed 0):
- **MW-R-H:** significantly below at every cutoff (HR@10 −.0147, p_t 2.9e-7; NDCG@10 −.0093, p_t 4e-9).
- **MW-3b:** HR@3 / NDCG@3 significantly above; HR@10 −.0072 (p .014/.01).
- **MW-nc:** HR@10 −.0086*, NDCG@10 −.0037*, @1–@5 n.s.
- **MW-HN:** HR@10 −.0056 (.05), rest n.s.
- **S:** HR@3 +.0060 (.005/.006), HR@5 +.0082 (.001/.001), NDCG@3/5/10 significant (NDCG@10 +.0039, .012/.016); HR@10 +.0045 (.12). vs seed 1: HR@5 +.0069 and NDCG@5 +.0037 significant, rest n.s.
- **D:** HR@1 +.0028 (.044/.051), HR@5 +.0045 (.045/.042), NDCG@5 +.0031 significant; HR@10 +.0015 (.60). vs seed 1: n.s. at every cutoff.
- S and D are both significantly above MW-R-H at all cutoffs. S − D is n.s. overall; S is better on non-copy HR@10 (+.0063, .019/.016).
- The MW-nc ep2 model never satisfied the hinge, so it is not a valid MW-nc test.
- MW-HN satisfied its hinge against stale bank latents.

Exclude-history decoding (code 3ae4990f):
- R3 373414: .0254/.0569/.0832/.1362, NDCG .0433/.0541/.0712.
- MW-R-H 373415: .0241/.0579/.0823/.1277, NDCG .0433/.0533/.0680.
- MW-R-H excl − R3 excl: n.s. at @1–@5; HR@10 −.0085 (p_t .004) and NDCG@10 −.0033 (.04) remain.
- So re-recommending watched movies explains the whole @1–@5 gap and ~40% of the HR@10 gap.

Diagnosis:
- **Where the loss is.** It sits on history-explained rows (Q1 −.074). MW-R-H re-recommends watched movies: 1.09 history items per top-10 vs .48, and the top-1 is a history item 14.5% vs 3.6% of the time. Last-item successors per top-10 fall from 2.26 to 2.04.
- **What `r` encodes.** Probe 374778: MW-R-H `r` follows the history (gap +.061) and the user (+.107) more than the next item (+.025). `r` encodes "watched A", not "next is B".
- **The latent helps on copy rows, so the decoder is the problem.** Copy-row split of 369532: mean −.0460, shuffle −.0756, mask −.1409 on copy rows. Without `r` the decoder reaches only ~.29 vs no-latent .47 there, so the decoder has lost its history-only reading.
- **This is what S and D fix.** Both restore copy_rows to the R3 level.
  - S keeps copy rows latent-independent (shuffle on copy rows −.0307) and keeps the non-copy gain (+.0072*).
  - D makes copy rows strongly latent-dependent (−.2125) with no non-copy gain (+.0009 n.s.).

### Goodreads (n 9,670) — failed, stopped by user

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|
| R3-Instruct 367929 | .0242 | .0367 | .0433 | .0567 | .0315 | .0342 | .0385 | |
| no-latent base 368191 | not recorded | .0340 | .0405 | .0525 | .0289 | .0316 | .0354 | |
| MW-R-H 368192 | .0188 | .0290 | .0353 | .0454 | .0247 | .0273 | .0306 | −.0125* / −.0094* / −.0061* |
| MW-R-H-L 371445 | .0143 | .0231 | .0297 | .0397 | .0194 | .0221 | .0254 | n.s. (unused) |

- 368192 is significantly below R3-Instruct at all cutoffs (p_t ≤ 4e-5) and below no-latent (HR@10 p_t 2.5e-4).
- The whole gap is on series-continuation rows (8% of test, ~64% of baseline hits): HR@10 .308 vs no-latent .429. Other rows equal R3 (.0224).
- MW-R-H-L is worse still (continuation .271) and drops latent use, which suggests epoch-3 overfit. R3's best epoch here is 2, and MW-R-H keeps epoch 3.
- The NUM_EPOCHS=2 test was never run: the user stopped the line on 2026-11-20.

### Swapped-pass MW-R-H-S on CDs / Toys (2026-11-26) — dropped

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy / non-copy | mask / mean / shuffle Δ HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| CDs S 382429 | .0607 | .0866 | .1002 | .1215 | .0757 | .0813 | .0882 | .6196 / .0852 | .0000 / +.0010 / +.0021 (all n.s.) |
| Toys S 382430 | .0416 | .0706 | .0856 | .1095 | .0585 | .0648 | .0723 | .5050 / .0482 | −.0006 / −.0016 / −.0027 (n.s.; NDCG@10 shuffle −.0032 *) |

- CDs S is above R3_ori at all cutoffs (HR@10 +.0226) and above MW-R-H 366023 at all cutoffs (HR@10 +.0094, p_t 8e-4).
- Toys S is above R3 (A) at HR@1/3/5 and NDCG@3/5/10 (NDCG@10 +.0056, p_t 7e-4); HR@10 +.0043 n.s. vs MW-R-H 365504 it is n.s. except NDCG@5.

Why the latent goes unused (training logs):

| run | hinge ep 0–1 | 1.5–2 | 2.5–3 | swapped CE ep 2.5–3 | latent |
|---|---|---|---|---|---|
| Toys MW-R-H 365504 | .200 | .088 | .020 | – | used |
| Toys S 382430 | .175 | .139 | .093 | .172 | unused |
| CDs S 382429 | .200 | .191 | .175 | .185 | unused |
| MovieLens S 375420 | .199 | .094 | .060 | .219 | used |

- The swapped-pass CE and the hinge pull the same swapped-pass loss in opposite directions.
- On Toys and CDs the decoder learns to predict well with the wrong `r` and the hinge never forms, so the gain is a history-reading regulariser, not latent content.
- On MovieLens the hinge is satisfied and the latent is used.
- This explanation is correlational. Decision (user, 2026-11-26): S is misaligned with the objective. Steam S 382431 was cancelled.

### WIDER / hinge trajectories, MW-R-H default seed (2026-11-26)

| epochs | Steam 369662 | Toys 365504 | CDs 366023 | MovieLens 369532 |
|---|---|---|---|---|
| WIDER 0–.25 | .724 | .444 | .575 | .771 |
| .25–.5 | .957 | .067 | .236 | .490 |
| .5–1 | 1.000 | .051 | .074 | .096 |
| 1–1.5 | .995 | .068 | .108 | .171 |
| 1.5–2 | .991 | .149 | .352 | .250 |
| 2–2.5 | 1.000 | .087 | .230 | .259 |
| 2.5–3 | 1.000 | .069 | .162 | .132 |
| hinge 2.5–3 | .198 | .020 | .062 | .055 |

Latent use follows whether the hinge forms. MovieLens 369532 also shows a grad-norm jump from 2.7 to 10.7 in epoch 3.

### Cross-dataset summary

| dataset | run(s) | ranking vs R3 | latent content used | seeds |
|---|---|---|---|---|
| CDs | MW-R-H 366023 / 368187 / 368188 | **significantly above at all @5/@10 in 3/3** (+.0130 HR@10 mean) | yes, 3/3 | 3, robust |
| Toys | MW-R-H 365504 / 368189 / 366001 (+ L40S 365887 / 365888) | ≥ R3 in 3/5 runs (NDCG@10 significantly above in 2), seed 2 significantly below (2/2 GPUs); A40 mean .1039 ≈ R3 | yes, 5/5 | 5 runs, seed-dependent; MW-R-H-L 2/2 ≈ R3 |
| Steam | 369662 default | NDCG@1–@10 significantly above, HR@10 p .09 | **no** (latents collapsed) | 1 |
| Steam | 383065 (clip 1.0) / 383067 (lr 5e-5) | ≈ R3 at HR@10; 383067 NDCG@10 significantly above | yes / yes | 1 each |
| Steam | 383066 (lr 1e-4) | significantly above at all cutoffs | no | 1 |
| MovieLens | MW-R-H 369532 | **significantly below at all cutoffs** | yes | 1 |
| MovieLens | MW-R-H-S 375420 / MW-R-H-D 375422 | ≥ R3 (S significantly above at @3/@5/NDCG) / = R3 | yes / yes | 1 each |
| Goodreads | 368192 / 371445 | significantly below | yes / no | 1; stopped |

## Reading
- **CDs:** C1 holds strongly and seed-robustly. MW-R-H gains on copy and non-copy rows, and the hinge is satisfied with diverse latents.
- **Toys:** C1 holds on average (A40 mean ≈ R3, content used in every run) but is seed-dependent. A late-unsatisfied hinge plus a WIDER jump breaks copy rows. MW-R-H-L fixes the failing seed, but because training is nondeterministic the fix rests on 2 runs.
- **Steam:** the default run is not a valid C1 result: the latents collapse early and the ranking gain is a regulariser effect.
  - 383067 (lr 5e-5) is the only Steam run with the hinge satisfied, content used and NDCG@10 significantly above R3.
  - The best ranking (383066) does not use the latent. On Steam, ranking and content use still trade off. Single runs only.
- **MovieLens:** plain MW-R-H fails. `r` encodes history identity and the shared decoder loses its history-only reading on copy rows.
  - S and D are the first MovieLens runs that meet the criterion; S is the stronger ranker.
  - S does not transfer to CDs or Toys (latent unused there), so no single recipe meets C1 on all four datasets.
- **Goodreads:** fails on series-continuation rows, with an epoch-3 confound. Stopped.
- **Open issues:**
  - Nondeterminism: same seed and GPU give different curves (Toys, Steam).
  - Epoch-selection mismatch: R3 keeps the best epoch, MW-R-H keeps epoch 3.
  - Single seeds on Steam and MovieLens.
  - Per-dataset lr: Steam needs 5e-5, while 3e-4 is used elsewhere.
  - A recipe that works for both dense-transition (MovieLens) and title-readable (Toys/CDs) data is still missing.

## Paper use
- **Main table:** CDs 3-seed mean; Toys 365504 + 3-A40-seed mean; Steam 383067 (with 369662 shown as the collapsed default); MovieLens 369532 vs MW-R-H-S 375420 / -D 375422. Each row with its R3 baseline and the shuffle / mean latent tests.
- **Robustness / seed table:** the Toys 7-run table and the CDs 3-seed table.
- **Appendix:**
  - Steam collapse test and WIDER trajectories;
  - MovieLens diagnosis (exclude-history, copy-row split, probe) and the MW-nc / MW-3b / MW-HN runs;
  - Goodreads;
  - swapped-pass on CDs/Toys (as a negative result).
- **Dev only:** MW-nc ep2, the band run 365897.

## Sources
- Research log `analysis/document/research_notes/Latent utilization/research_log.md`:
  - MW-R-H Toys/CDs 2026-11-14..17 (l. 2640–2716);
  - 3 seeds and Goodreads 2026-11-18..20 (l. 2727–2805);
  - Steam / MovieLens MW-R-H and diagnosis 2026-11-21..23 (l. 2813–3046);
  - MW-R-H-S / -D 2026-11-24 (l. 3087–3173);
  - S on CDs/Toys, trajectories, Steam collapse test 2026-11-25..26 (l. 3296–3410, 3478–3548).
- Tables: jobs tmp `report_tables.md` (Toys paired tests and interventions), `cds_tables.md`, `toys_swapce.py`, `cds_swapce.py` (rerun 2026-11-27 for the S latent-test signs), `steam_lr_eval.py`, `ml_runs.py`, `xds.py`.
- Per-row results: `R3/output_dir/margin_wider_collab/`.
- Memory: r3-margin-wider-collab, r3-movielens, r3-steam, r3-goodreads.
