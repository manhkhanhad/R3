# From margin + WIDER to collaborative latent reasoning (MW → MWC): model versions and results

Toys_and_Games, test n = 6,738. Report date 2026-11-12.

## Goal and how it is measured

R3 (LatentR³) splices a latent thought vector `r` (attention pooling over the history) into the prompt; plain R3's decoder ignores `r`'s content. The goal of this line is a model whose recommendations **depend on `r`** while ranking **at least as well as plain R3**, with `r` carrying *collaborative* structure: two users who buy the same item should have similar latents; one user's two purchases should have similar latents only when the items are similar.

- **Ranking:** constrained beam search (10 beams), HR and NDCG at @1/3/5/10 (@20 = @10). The primary metrics are @5 and @10.
- **Baseline:** plain R3 retrain "A" (same recipe, no latent losses). Plain-R3 retrains on Toys vary between about 0.102 and 0.108 HR@10.
- **Latent impact (interventions):** the same model evaluated with `r` replaced by a zero vector (mask), the mean training latent (mean), or another user's latent (shuffle). A significant drop means the recommendation uses `r`'s content.
- **Statistics:** paired bootstrap over test rows, 2,000 resamples, seed 42.
- **Copy rows:** 905 test rows (13%) where the target is a same-day near-copy of a history item (same brand, title Jaccard ≥ 0.3). They behave very differently from the other 5,833 rows, so they are reported separately.

## Model versions

All versions share the base: Qwen2.5-1.5B, R3 latent pooling, CE on the target title, lr 3e-4, batch 128, 3 epochs, unchanged prompt and inference.

| model | job | what it adds or changes |
|---|---|---|
| **plain R3 (A)** | 345857 | baseline retrain |
| **WIDER only** | 355749 | cross-user WIDER penalty (weight 0.1) on `r`: pushes latents of different users apart |
| **MW** (margin + WIDER) | 356322 | WIDER + **history-swap margin hinge**: `[0.2 + NLL(y | x, r_own) − NLL(y | x, r_other)]₊`, where `r_other` is another in-batch row's latent with the same history `x`. It forces the decoder to read `r` |
| **MWC-G** | 358546 | MW + **donor grounding**: `r1 = r0 + s·‖r0‖·dir(evidence)` from 32 time-causal donors (other users' latents and their next items); margin on `r1` |
| **MWC-I** | 358547 | MW (epoch 1) then **user + target InfoNCE** (epochs 2–3): positives = same user's rows / other users with the same next item, 10k queue |
| **MWC-I-B** | 359986 | MWC-I with the user term removed and **same-user different-target hard negatives** for the target term |
| **MWC-I-C** | 360056 | MWC-I-B + an **item-prediction head** on `r` (CE over the Qwen3 title table) |
| **G3** | 360502 | MW + **CF-donor evidence**: donors chosen by history overlap (time-causal, other users); their next items form centred evidence in the slot; evidence-swap margin; WIDER on the history latent `r0` |
| **MWC-M** | 362616 | MW (epoch 1) then **same-motive InfoNCE** (epochs 2–3): positives = other users with the same next item, and the same user's rows whose next item is similar (title cos ≥ 0.8); same-user dissimilar rows neutral; full-train latent bank |
| **MWC-M2** | 362632 | MWC-M, but within-user positives replaced by the **top-20 SASRec sequence neighbours** (other users, one row each) with a similar next item |
| **MWC-M m = 0.1 / 0.05** | 363593 / 363201 | MWC-M with a smaller margin m |
| **MWC-M3a** | 363141 | MWC-M with a **self-weighted hinge**: weight `1 − exp(−NLL(y | x, r_other))`, small where the history alone predicts the target |
| **MWC-M3a-n** | 363638 | MWC-M3a with the weight **normalised** by its running mean (mean weight ≈ 1, cap 2) |
| **MWC-M3b** | 363145 | MWC-M with the hinge **only in epoch 1** |
| **MW-3b** | 363603 | MW with the hinge **only in epoch 1** and **no motive loss**: epochs 2–3 are CE + WIDER |

Still running: **MWC-M4** (hinge on novel target tokens only, i.e. tokens absent from the history titles; 363511) and **MWC-M6** (adaptive one-sided margin `min(0.2, 0.5·NLL)` with no gradient through the negative term; 363594).

## Results

### Ranking (all cutoffs; @20 = @10 with 10 beams)

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 (905) | non-copy HR@10 (5,833) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain R3 (A, retrain) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 | 0.513 | 0.0420 |
| WIDER only | 0.0350 | 0.0628 | 0.0790 | 0.1070 | 0.0512 | 0.0578 | 0.0669 | 0.517 | 0.0434 |
| MW (margin + WIDER, 356322) | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 | 0.478 | 0.0447 |
| MWC-G (donor grounding) | 0.0358 | 0.0638 | 0.0798 | 0.1064 | 0.0521 | 0.0587 | 0.0672 | 0.503 | 0.0449 |
| MWC-I (user+target InfoNCE) | 0.0322 | 0.0561 | 0.0718 | 0.0966 | 0.0455 | 0.0520 | 0.0600 | 0.482 | 0.0369 |
| MWC-I-B (target InfoNCE + hard neg.) | 0.0331 | 0.0591 | 0.0726 | 0.1017 | 0.0481 | 0.0536 | 0.0629 | 0.487 | 0.0418 |
| MWC-I-C (B + item head) | 0.0263 | 0.0511 | 0.0677 | 0.0977 | 0.0405 | 0.0474 | 0.0570 | 0.477 | 0.0387 |
| G3 (CF-donor evidence) | 0.0321 | 0.0516 | 0.0665 | 0.0901 | 0.0435 | 0.0496 | 0.0573 | 0.448 | 0.0346 |
| MWC-M (same-motive InfoNCE) | 0.0355 | 0.0640 | 0.0816 | 0.1057 | 0.0521 | 0.0593 | 0.0670 | 0.486 | 0.0466 |
| MWC-M2 (SASRec-neighbour motive) | 0.0343 | 0.0598 | 0.0769 | 0.1045 | 0.0489 | 0.0559 | 0.0648 | 0.494 | 0.0441 |
| MWC-M m=0.1 | 0.0353 | 0.0614 | 0.0803 | 0.1031 | 0.0504 | 0.0582 | 0.0655 | 0.498 | 0.0418 |
| MWC-M m=0.05 | 0.0373 | 0.0649 | 0.0776 | 0.1060 | 0.0534 | 0.0586 | 0.0677 | 0.504 | 0.0442 |
| MWC-M3a (self-weighted hinge) | 0.0371 | 0.0650 | 0.0809 | 0.1080 | 0.0531 | 0.0596 | 0.0684 | 0.506 | 0.0463 |
| MWC-M3a-n (normalised self-weight) | 0.0331 | 0.0589 | 0.0693 | 0.0960 | 0.0480 | 0.0523 | 0.0609 | 0.472 | 0.0377 |
| MWC-M3b (hinge epoch 1 only) | 0.0380 | 0.0649 | 0.0827 | 0.1080 | 0.0535 | 0.0608 | 0.0690 | 0.516 | 0.0447 |
| MW-3b (MW, hinge epoch 1 only, no motive) | 0.0398 | 0.0687 | 0.0856 | 0.1125 | 0.0562 | 0.0632 | 0.0718 | 0.515 | 0.0501 |

### Paired bootstrap vs plain R3 (2,000 resamples, seed 42; * = 95% CI excludes 0)

| model | ΔHR@5 | ΔNDCG@5 | ΔHR@10 | ΔNDCG@10 |
|---|---|---|---|---|
| WIDER only | -0.0006 [-0.0043, +0.0034] | -0.0006 [-0.0029, +0.0017] | +0.0018 [-0.0025, +0.0061] | +0.0002 [-0.0021, +0.0025] |
| MW (margin + WIDER, 356322) | +0.0007 [-0.0042, +0.0058] | +0.0010 [-0.0024, +0.0045] | -0.0022 [-0.0076, +0.0036] | +0.0001 [-0.0031, +0.0035] |
| MWC-G (donor grounding) | +0.0003 [-0.0046, +0.0053] | +0.0002 [-0.0031, +0.0036] | +0.0012 [-0.0043, +0.0064] | +0.0005 [-0.0028, +0.0036] |
| MWC-I (user+target InfoNCE) | -0.0077 [-0.0126, -0.0028] * | -0.0064 [-0.0097, -0.0031] * | -0.0086 [-0.0137, -0.0036] * | -0.0067 [-0.0099, -0.0035] * |
| MWC-I-B (target InfoNCE + hard neg.) | -0.0070 [-0.0113, -0.0024] * | -0.0048 [-0.0079, -0.0017] * | -0.0036 [-0.0083, +0.0015] | -0.0038 [-0.0067, -0.0008] * |
| MWC-I-C (B + item head) | -0.0119 [-0.0168, -0.0068] * | -0.0111 [-0.0144, -0.0076] * | -0.0076 [-0.0132, -0.0021] * | -0.0097 [-0.0130, -0.0064] * |
| G3 (CF-donor evidence) | -0.0131 [-0.0180, -0.0080] * | -0.0088 [-0.0124, -0.0052] * | -0.0151 [-0.0211, -0.0093] * | -0.0095 [-0.0131, -0.0058] * |
| MWC-M (same-motive InfoNCE) | +0.0021 [-0.0031, +0.0074] | +0.0008 [-0.0026, +0.0045] | +0.0004 [-0.0052, +0.0061] | +0.0003 [-0.0030, +0.0039] |
| MWC-M2 (SASRec-neighbour motive) | -0.0027 [-0.0074, +0.0022] | -0.0025 [-0.0056, +0.0009] | -0.0007 [-0.0059, +0.0049] | -0.0019 [-0.0050, +0.0014] |
| MWC-M m=0.1 | +0.0007 [-0.0039, +0.0053] | -0.0003 [-0.0035, +0.0030] | -0.0021 [-0.0071, +0.0033] | -0.0012 [-0.0043, +0.0020] |
| MWC-M m=0.05 | -0.0019 [-0.0065, +0.0027] | +0.0001 [-0.0031, +0.0033] | +0.0007 [-0.0046, +0.0062] | +0.0010 [-0.0022, +0.0040] |
| MWC-M3a (self-weighted hinge) | +0.0013 [-0.0033, +0.0064] | +0.0012 [-0.0020, +0.0045] | +0.0028 [-0.0024, +0.0083] | +0.0017 [-0.0015, +0.0047] |
| MWC-M3a-n (normalised self-weight) | -0.0102 [-0.0150, -0.0055] * | -0.0061 [-0.0098, -0.0028] * | -0.0092 [-0.0148, -0.0034] * | -0.0059 [-0.0092, -0.0025] * |
| MWC-M3b (hinge epoch 1 only) | +0.0031 [-0.0016, +0.0082] | +0.0024 [-0.0007, +0.0060] | +0.0028 [-0.0024, +0.0082] | +0.0022 [-0.0010, +0.0056] |
| MW-3b (MW, hinge epoch 1 only, no motive) | +0.0061 [+0.0012, +0.0110] * | +0.0047 [+0.0014, +0.0081] * | +0.0073 [+0.0018, +0.0126] * | +0.0051 [+0.0017, +0.0082] * |

### Latent interventions: raw − intervention, HR@10 [95% CI] / NDCG@10 (positive = replacing the latent hurts = latent used)

| model | raw HR@10 | zero latent (mask) | mean latent | another user's latent (shuffle) |
|---|---:|---|---|---|
| WIDER only | 0.1070 | +0.0009 [-0.0009, +0.0027] / +0.0006 | -0.0007 [-0.0024, +0.0009] / -0.0000 | -0.0009 [-0.0028, +0.0009] / -0.0003 |
| MW (margin + WIDER, 356322) | 0.1030 | +0.0508 [+0.0450, +0.0567] * / +0.0370 | +0.0309 [+0.0260, +0.0358] * / +0.0233 | +0.0733 [+0.0668, +0.0801] * / +0.0505 |
| MWC-G (donor grounding) | 0.1064 | +0.0562 [+0.0499, +0.0626] * / +0.0407 | +0.0675 [+0.0613, +0.0741] * / +0.0479 | +0.0675 [+0.0610, +0.0742] * / +0.0481 |
| MWC-I (user+target InfoNCE) | 0.0966 | +0.0807 [+0.0741, +0.0876] * / +0.0532 | +0.0772 [+0.0705, +0.0840] * / +0.0507 | +0.0749 [+0.0684, +0.0815] * / +0.0481 |
| MWC-I-B (target InfoNCE + hard neg.) | 0.1017 | +0.0876 [+0.0807, +0.0941] * / +0.0552 | +0.0837 [+0.0770, +0.0905] * / +0.0527 | +0.0831 [+0.0763, +0.0899] * / +0.0534 |
| MWC-I-C (B + item head) | 0.0977 | +0.0381 [+0.0327, +0.0439] * / +0.0288 | +0.0424 [+0.0367, +0.0481] * / +0.0307 | +0.0494 [+0.0433, +0.0557] * / +0.0338 |
| G3 (CF-donor evidence) | 0.0901 | +0.0588 [+0.0531, +0.0649] * / +0.0400 | +0.0237 [+0.0184, +0.0289] * / +0.0145 | +0.0508 [+0.0450, +0.0567] * / +0.0335 |
| MWC-M (same-motive InfoNCE) | 0.1057 | +0.0899 [+0.0827, +0.0972] * / +0.0588 | +0.0882 [+0.0810, +0.0954] * / +0.0579 | +0.0840 [+0.0769, +0.0910] * / +0.0549 |
| MWC-M2 (SASRec-neighbour motive) | 0.1045 | +0.0629 [+0.0568, +0.0695] * / +0.0420 | +0.0515 [+0.0457, +0.0576] * / +0.0370 | +0.0654 [+0.0591, +0.0720] * / +0.0439 |
| MWC-M m=0.1 | 0.1031 | +0.0849 [+0.0778, +0.0919] * / +0.0553 | +0.0782 [+0.0715, +0.0847] * / +0.0504 | +0.0797 [+0.0727, +0.0865] * / +0.0518 |
| MWC-M m=0.05 | 0.1060 | +0.0560 [+0.0497, +0.0617] * / +0.0354 | +0.0102 [+0.0064, +0.0140] * / +0.0074 | +0.0251 [+0.0203, +0.0297] * / +0.0157 |
| MWC-M3a (self-weighted hinge) | 0.1080 | +0.0003 [-0.0021, +0.0027] / -0.0003 | +0.0009 [-0.0015, +0.0033] / +0.0002 | -0.0001 [-0.0025, +0.0022] / -0.0006 |
| MWC-M3a-n (normalised self-weight) | pending | | | |
| MWC-M3b (hinge epoch 1 only) | 0.1088 | +0.0013 [-0.0007, +0.0034] / +0.0005 | +0.0006 [-0.0013, +0.0027] / +0.0004 | +0.0010 [-0.0009, +0.0031] / +0.0007 |
| MW-3b (MW, hinge epoch 1 only, no motive) | 0.1125 | -0.0010 [-0.0030, +0.0009] / -0.0005 | -0.0016 [-0.0039, +0.0004] / -0.0003 | -0.0006 [-0.0028, +0.0015] / +0.0001 |

MWC-M3a-n's interventions are still running. Its ranking is already significantly below plain R3, so it fails regardless.

### Collaborative structure of the latent (motive probe, raw `r`, 20k training rows)

| model | cos same item, other user − random other user | same item among the 10 nearest other-user latents (chance 0.0005) | same user: cos similar items − dissimilar items |
|---|---:|---:|---:|
| MW (356322) | +0.145 | 0.0265 | +0.058 |
| MWC-I-B | +0.211 | 0.0278 | +0.056 |
| **MWC-M** | **+0.273** | **0.0341** | **+0.071** |

## Which model achieves both: better than the baseline *and* a latent that matters?

Criterion: a significant drop under the latent interventions, and HR/NDCG @5 and @10 at or above plain R3. Significance on the ranking is optional.

| model | latent used? (HR@10 drop when `r` is replaced) | Δ vs plain R3: HR@5 / NDCG@5 / HR@10 / NDCG@10 | both? |
|---|---|---|---|
| **MWC-M** | **yes, strongest**: −0.084 to −0.090, all sig | **+0.0021 / +0.0008 / +0.0004 / +0.0003** (n.s.) | **yes**: all four positive, plus collaborative structure in `r` |
| **MWC-G** | yes: −0.056 to −0.068, all sig | +0.0003 / +0.0002 / +0.0012 / +0.0005 (n.s.) | **yes**: all four positive, but the gain is tiny and the donor evidence collapsed to a constant (only the slot is used, not the donors) |
| MW (356322) | yes: −0.031 to −0.073 | +0.0007 / +0.0010 / −0.0022 / +0.0001 | mixed: HR@10 below |
| MWC-M m = 0.05 | weakly: −0.010 to −0.056 | −0.0019 / +0.0001 / +0.0007 / +0.0010 | mixed: HR@5 below |
| MWC-M m = 0.1 | yes: −0.078 to −0.085 | +0.0007 / −0.0003 / −0.0021 / −0.0012 | mixed: mostly below |
| MWC-M2 | yes: −0.052 to −0.065 | −0.0027 / −0.0025 / −0.0007 / −0.0019 | no (below at all four) |
| MWC-I / I-B / I-C, G3, M3a-n | yes | significantly below at several cutoffs | no |
| MW-3b | **no** (all n.s.) | **+0.0061 / +0.0047 / +0.0073 / +0.0051, all significant** | no: best ranking, but `r` is ignored |
| MWC-M3a, MWC-M3b | no (all n.s.) | +0.001 to +0.003 (n.s.) | no: `r` is ignored |

**Answer:**
- **MWC-M** is the model that currently achieves both. Its latent has the strongest impact of any version (replacing `r` costs 0.084–0.090 HR@10 and 0.055–0.059 NDCG@10, all significant). It is at or above plain R3 at every primary cutoff (HR@5 +0.0021, NDCG@5 +0.0008, HR@10 +0.0004, NDCG@10 +0.0003; not significant). Its latent also carries the intended collaborative structure.
- **MWC-G** also meets the letter of the criterion, but its margin over R3 is negligible and its donor evidence was found to be constant, so the collaborative part does not work there.
- **No version yet beats plain R3 *significantly* while using the latent.** The only significant improvement (MW-3b, HR@10 +0.0073, NDCG@10 +0.0051) comes from a model that ignores `r` at test time.

## Why the latent-using models do not pull ahead: the copy-row trade-off

- **Every model that keeps the latent in use loses copy rows.** The history-swap hinge is the cause:
  - models trained with it reach 0.45–0.50 copy HR@10, against 0.50–0.52 for every model without it (WIDER-only 0.517);
  - on a copy row the history alone already predicts the target, so the hinge can only be met by letting `r` override the history;
  - the sibling variants that `r` does not point to then leave beam slots 6–10.
- **The same models gain on non-copy rows.** MWC-M reaches 0.0466 against 0.0420 for plain R3; on non-copy rows whose target is not among the CF donors' outcomes, the gain is +0.0049, significant.
- **The two effects cancel.** MWC-M's copy-row deficit (−0.0036 overall) offsets its non-copy gain (+0.0040 overall).
- **The fixes so far recovered copy rows only by dropping latent use:**
  - M3a and M3b: latent unused;
  - m = 0.05: latent weakly used;
  - M3a-n: ranking hurt.

  M4 (hinge only on novel tokens) is the remaining test of separating the two. Its training log shows the hinge gap forming (0.141 → 0.046) while copy rows are mostly exempt.

## Caveats and next steps

- **Single seed per model.** Plain-R3 retrains on Toys vary between about 0.102 and 0.108 HR@10; MW-3b (0.1125) is above that range, but MWC-M's lead over R3 is within it.
- **Pending:** M4, M6 and M3a-n's interventions.
- **Suggested next steps:**
  - replicate MWC-M and MW-3b (second seed; CDs);
  - if M4 keeps `r` in use while recovering copy rows, it is the candidate for beating R3 with a latent that matters.

Sources: research log `analysis/document/research_notes/Latent utilization/research_log.md`; per-run outputs in `output_dir/margin_wider_collab/`, `output_dir/margin_wider/`, `output_dir/wider/`, `output_dir/guidance_scaffold/`; tables generated by `report_tables.py` from those files.
