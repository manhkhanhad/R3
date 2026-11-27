# E07 — Collaborative terms on the MW base (Toys)
Back to [main plan](../../main.md). Dates: 2026-11-08 – 2026-11-12. Status: done (single seed per run).

## Question
MW (margin + WIDER, Toys 356322) is the first base whose `r` content the decoder reads (shuffle −0.073 HR@10) at R3-level ranking. With the decoder now reading `r`, does a **collaborative** term (contribution 2: ground each latent in other users' latents, because people buy an item for shared reasons) change what `r` carries and keep ranking ≥ plain R3? Which definition of "collaborative positive" works?

## Setup
- Common recipe (= 356322): Qwen2.5-1.5B base, lr 3e-4, batch 128, 3 epochs, history-swap hinge m 0.2 × 1.0, WIDER 0.1. Toys test n = 6,738. Baselines: plain R3 (A) 0.1052 / 0.0667, MW 356322 0.1030 / 0.0668.
- Script `scripts/margin_wider_collab.sbatch` (`METHOD=infonce|grounded|cfg3|motive`); results `output_dir/margin_wider_collab/<name>/`.

| run | ID | code | what it adds to MW |
|---|---|---|---|
| MWC-I | 358547 | b3740008 | original collaborative-reasoning InfoNCE (two-phase, `collaborative_margin_attention.py`): epoch 1 MW only (10k FIFO queue fills, token-accuracy gate 0.5); epochs 2–3 + **user head** (positives = same user's other rows) 0.1 + **target head** (positives = other users with the same next item) 0.1, τ 0.1, 256-d projection, 16 positives / 256 negatives |
| MWC-I-B | 359986 | not recorded | user weight 0; target head + up to 16 **same-user, different-target hard negatives** |
| MWC-I-C | 360056 | b3cb3f0a | B + item-prediction head on `r` (LN + Linear 1536→1536 fp32 → Qwen3 title table, CE over 6,299 items, τ 0.07, weight 0.2, both phases) |
| MWC-G | 358546 | b3740008 | donor grounding: r1 = r0 + s·‖r0‖·dir(Σ a_j V(r0_j, e(y_j))) over 32 time-causal donors (other users' r0 + observed next item); item-grounding CE 0.2 + retrieval KL 0.1; hinge and WIDER on r1; fp32 grounding modules |
| MWC-G3 | 360502 | ba0ccfc9 | CF-prior donor evidence: ≤ 32 other-user train rows sharing ≥ 1 history item, time-causal; values = donor outcome + overlap features (no donor r0); centred evidence; + evidence-swap margin 0.2; WIDER on r0 |
| **MWC-M** | **362616** | 0d58bf29 | **same-motive InfoNCE** (`motive_contrastive.py`, `latent_attention_train_motive.py`), design below |
| MWC-M2 | 362632 | 87278c64 | MWC-M with within-user positives replaced by the top-20 SASRec sequence neighbours (20 distinct other users) with a similar next item |
| MWC-M m 0.1 / 0.05 | 363593 / 363201 | 62872468 / 0d58bf29 | MWC-M with margin 0.1 / 0.05 |

**MWC-M design (the user's definition, 2026-11-11).** Two users buying the same item → similar `r` (shared motive); one user's two purchases → similar `r` only if the items are similar.
- Positives: ≤ 16 **cross-user rows with the exact same next item** + ≤ 16 **same-user rows whose next item has Qwen3 title cos ≥ 0.8**. Same-user dissimilar rows are **neutral** (never in the loss). Negatives: 256 bank rows of other users with a different item.
- Bank: a **persistent full-train bank** of every row's `r` (53.9k rows, indexed by train-row id, written in epoch 1, survives preemption), not a 10k FIFO.
- Schedule: epoch 1 = MW; epochs 2–3 + InfoNCE (weight 0.1, τ 0.1, LN+Linear 256-d). Inference = plain R3.
- Pair statistics (Toys train):
  - a same-item other-user positive exists for 99.8% of rows (median 10);
  - within-user pairs with cos ≥ 0.8 are 16% of pairs, and 40% of rows have at least one (44% incl. repeats);
  - positives available in training: cross 100%, within 44%.

## Results

### Ranking (HR@1 = NDCG@1; @20 = @10 with 10 beams)
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 (905) | non-copy HR@10 (5,833) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain R3 (A) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 | 0.513 | 0.0420 |
| MW (356322) | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 | 0.478 | 0.0447 |
| MWC-I (358547) | 0.0322 | 0.0561 | 0.0718 | 0.0966 | 0.0455 | 0.0520 | 0.0600 | 0.482 | 0.0369 |
| MWC-I-B (359986) | 0.0331 | 0.0591 | 0.0726 | 0.1017 | 0.0481 | 0.0536 | 0.0629 | 0.487 | 0.0418 |
| MWC-I-C (360056) | 0.0263 | 0.0511 | 0.0677 | 0.0977 | 0.0405 | 0.0474 | 0.0570 | 0.477 | 0.0387 |
| MWC-G (358546) | 0.0358 | 0.0638 | 0.0798 | 0.1064 | 0.0521 | 0.0587 | 0.0672 | 0.503 | 0.0449 |
| MWC-G3 (360502, knn) | 0.0321 | 0.0516 | 0.0665 | 0.0901 | 0.0435 | 0.0496 | 0.0573 | 0.448 | 0.0346 |
| **MWC-M (362616)** | 0.0355 | 0.0640 | 0.0816 | **0.1057** | 0.0521 | 0.0593 | **0.0670** | 0.486 | **0.0466** |
| MWC-M2 (362632) | 0.0343 | 0.0598 | 0.0769 | 0.1045 | 0.0489 | 0.0559 | 0.0648 | 0.494 | 0.0441 |
| MWC-M m 0.1 (363593) | 0.0353 | 0.0614 | 0.0803 | 0.1031 | 0.0504 | 0.0582 | 0.0655 | 0.498 | 0.0418 |
| MWC-M m 0.05 (363201) | 0.0373 | 0.0649 | 0.0776 | 0.1060 | 0.0534 | 0.0586 | 0.0677 | 0.504 | 0.0442 |

### Δ vs plain R3 (A), paired bootstrap 95% CI (bold = excludes 0). Paired t-test p: not recorded for these Toys runs.
| model | ΔHR@5 | ΔNDCG@5 | ΔHR@10 | ΔNDCG@10 |
|---|---|---|---|---|
| MW (356322) | +0.0007 [−0.0042, +0.0058] | +0.0010 [−0.0024, +0.0045] | −0.0022 [−0.0076, +0.0036] | +0.0001 [−0.0031, +0.0035] |
| MWC-I | **−0.0077** [−0.0126, −0.0028] | **−0.0064** [−0.0097, −0.0031] | **−0.0086** [−0.0137, −0.0036] | **−0.0067** [−0.0099, −0.0035] |
| MWC-I-B | **−0.0070** [−0.0113, −0.0024] | **−0.0048** [−0.0079, −0.0017] | −0.0036 [−0.0083, +0.0015] | **−0.0038** [−0.0067, −0.0008] |
| MWC-I-C | **−0.0119** [−0.0168, −0.0068] | **−0.0111** [−0.0144, −0.0076] | **−0.0076** [−0.0132, −0.0021] | **−0.0097** [−0.0130, −0.0064] |
| MWC-G | +0.0003 [−0.0046, +0.0053] | +0.0002 [−0.0031, +0.0036] | +0.0012 [−0.0043, +0.0064] | +0.0005 [−0.0028, +0.0036] |
| MWC-G3 | **−0.0131** [−0.0180, −0.0080] | **−0.0088** [−0.0124, −0.0052] | **−0.0151** [−0.0211, −0.0093] | **−0.0095** [−0.0131, −0.0058] |
| **MWC-M** | +0.0021 [−0.0031, +0.0074] | +0.0008 [−0.0026, +0.0045] | +0.0004 [−0.0052, +0.0061] | +0.0003 [−0.0030, +0.0039] |
| MWC-M2 | −0.0027 [−0.0074, +0.0022] | −0.0025 [−0.0056, +0.0009] | −0.0007 [−0.0059, +0.0049] | −0.0019 [−0.0050, +0.0014] |
| MWC-M m 0.1 | +0.0007 [−0.0039, +0.0053] | −0.0003 [−0.0035, +0.0030] | −0.0021 [−0.0071, +0.0033] | −0.0012 [−0.0043, +0.0020] |
| MWC-M m 0.05 | −0.0019 [−0.0065, +0.0027] | +0.0001 [−0.0031, +0.0033] | +0.0007 [−0.0046, +0.0062] | +0.0010 [−0.0022, +0.0040] |

Other comparisons:
- MWC-I − MW HR@10 **−0.0064** [−0.0117, −0.0013].
- MWC-M − MW HR@10 +0.0027 n.s.
- MWC-M2 − MWC-M: **NDCG@5 −0.0034 sig**; HR@5 −0.0047 (CI touches 0); HR@10 −0.0012 n.s.
- MWC-G3 − MW HR@10 **−0.0129** sig.
- MWC-I-B − MW NDCG@10 **−0.0039** [−0.0068, −0.0009].

### Latent tests: raw − intervention, HR@10 [95% CI] / NDCG@10 (positive = latent used; all significant)
| model | mask | mean | shuffle_user | slot eff. rank |
|---|---|---|---|---:|
| MW (356322) | +0.0508 [+0.0450, +0.0567] / +0.0370 | +0.0309 [+0.0260, +0.0358] / +0.0233 | +0.0733 [+0.0668, +0.0801] / +0.0505 | 3.7 |
| MWC-I | +0.0807 [+0.0741, +0.0876] / +0.0532 | +0.0772 [+0.0705, +0.0840] / +0.0507 | +0.0749 [+0.0684, +0.0815] / +0.0481 | 76 |
| MWC-I-B | +0.0876 [+0.0807, +0.0941] / +0.0552 | +0.0837 [+0.0770, +0.0905] / +0.0527 | +0.0831 [+0.0763, +0.0899] / +0.0534 | 20.9 |
| MWC-I-C | +0.0381 / +0.0288 | +0.0424 / +0.0307 | +0.0494 / +0.0338 | 66 |
| MWC-G (slot r1) | +0.0562 / +0.0407 | +0.0675 / +0.0479 | +0.0675 / +0.0481 | 77 |
| MWC-G3 (knn − slot) | +0.059 | +0.024 | +0.051 | not recorded |
| **MWC-M** | +0.0899 [+0.0827, +0.0972] / +0.0588 | +0.0882 [+0.0810, +0.0954] / +0.0579 | **+0.0840** [+0.0769, +0.0910] / +0.0549 | 16.6 |
| MWC-M2 | +0.0629 / +0.0420 | +0.0515 / +0.0370 | +0.0654 / +0.0439 | 50 |
| MWC-M m 0.1 | +0.0849 / +0.0553 | +0.0782 / +0.0504 | +0.0797 / +0.0518 | not recorded |
| MWC-M m 0.05 | +0.0560 / +0.0354 | +0.0102 / +0.0074 | +0.0251 / +0.0157 | not recorded |

### Donor controls (grounding runs), HR@10 / NDCG@10
| run | knn (method) | off (no donors) | donor-content controls | knn − off | knn − control |
|---|---|---|---|---|---|
| MWC-G 358546 | 0.1064 / 0.0672 | 0.1049 / 0.0662 | shuffle_outcome 0.1064 / 0.0672 | +0.0015 [−0.0007, +0.0037] n.s. | 0.0000 n.s. |
| MWC-G3 360502 | 0.0901 / 0.0573 | 0.0739 / 0.0477 | random 0.0233; swap 0.0401 | **+0.0162** sig | swap **+0.0500**, random **+0.0668** sig |

### Motive probe (363016, code f33e7cd6; raw `r`, first 20k train rows; 18,565 same-item anchors; within-user 8,838 similar / 18,110 dissimilar)
| model | cos same item, other user | cos random other user | gap | P@10 same item among nearest other-user latents (chance 0.0005) | within-user cos similar / dissimilar | within gap |
|---|---:|---:|---:|---:|---|---:|
| MW (356322) | 0.228 | 0.084 | +0.145 | 0.0265 | 0.744 / 0.686 | +0.058 |
| MWC-I-B | 0.264 | 0.053 | +0.211 | 0.0278 | 0.742 / 0.686 | +0.056 |
| **MWC-M** | 0.294 | 0.021 | **+0.273** | **0.0341** | 0.794 / 0.724 | **+0.071** |

The first probe, 363014, failed on a code bug. The plain A checkpoint no longer exists, so A is not in the probe.

### Latent-bank quality (360284 MWC-I-B, 360285 MWC-I, 360286 MW; code 02ffaf86; 10k train rows, 800 anchors, k = 4)
- Swap-in utility: u_ij = NLL(y_i | x_i, r_i) − NLL(y_i | x_i, r_j). "Good" = u > −0.1; "better" = u > 0.
- Coverage in the 10k bank: 0.825 of anchors have a same-target other-user row; 0.947 have a same-user other-target row.

| checkpoint | eff. rank | centred cos same-target / same-user / random | same-target positives: mean u / good / better | best-of-4 good / better | same-user other-target: median u / good / better | random: mean u / good |
|---|---:|---|---|---|---|---|
| MW 356322 | 3.7 | 0.157 / 0.547 / 0.010 | −1.26 / 0.12 / 0.04 | 0.45 / 0.20 | −0.006 / 0.66 / 0.26 | −1.72 / 0.15 |
| MWC-I | 76.0 | 0.156 / 0.734 / −0.003 | −1.39 / 0.15 / 0.07 | 0.45 / 0.22 | −0.011 / 0.75 / 0.30 | −1.86 / 0.11 |
| MWC-I-B | 20.9 | 0.208 / 0.596 / 0.006 | −1.17 / 0.10 / 0.05 | 0.36 / 0.17 | −0.009 / 0.68 / 0.26 | −1.55 / 0.06 |

### Random-donor diagnostic (code fabc7e81; 400 anchors × 64 random other-user latents)
| checkpoint (job) | good rate | anchors with ≥ 1 good donor | best donor beats own `r` | corr(u, latent cos) | corr(u, target-title cos) | corr(u, history Jaccard) |
|---|---:|---:|---:|---:|---:|---:|
| MW 356322 (360311) | 0.149 | 0.995 | 0.805 | **0.46** | 0.03 | 0.02 |
| MWC-I (360312) | 0.116 | 0.990 | 0.700 | 0.44 | 0.10 | 0.03 |
| MWC-I-B (360313) | 0.073 | 0.965 | 0.60 | 0.46 | 0.07 | 0.04 |

- 356322: good rate by latent-cos decile 0.56 (top) / 0.00 (bottom).
- 356322 good vs bad donors: latent cos 0.86 vs −0.06; same target 0.001 vs 0.000.
- Best-donor gain: mean +0.021 nats (356322) / +0.022 (MWC-I).

### Chance level of the target InfoNCE
- Multi-positive InfoNCE at equal scores = log((P+N)/P). In a 10k queue: median P 2.6, mean P 4.0, and 0.80 of anchors have a positive. Chance is therefore **4.56** with 256 negatives (B: 4.62 with +16 hard negatives).
- Logged end values: MWC-I 4.02, B 4.10, about 0.5 nats below chance (phase 2 started at 5.3).
- User InfoNCE (MWC-I): 4.32 → 0.28.

### MWC-I copy / non-copy split and geometry (per-row HR@10; copy = same-day, same-brand, title Jaccard ≥ 0.3)
| rows | n | MWC-I | MW | plain A | MWC-G | MWC-I − MW |
|---|---:|---:|---:|---:|---:|---:|
| copy | 905 | 0.4818 | 0.4785 | 0.5127 | 0.5028 | +0.0033 |
| non-copy | 5,833 | 0.0369 | 0.0447 | 0.0420 | 0.0449 | −0.0079 |
| same-brand target | 2,144 | 0.2822 | 0.2957 | 0.3027 | 0.3008 | −0.0135 |

- Top-10 exact-history-item share: A 0.101, MW 0.104, MWC-I 0.127, B 0.127, C 0.139, MWC-M 0.111.
- Centred cos (10k train `r`), MW → MWC-I: same-user consecutive rows 0.804 → **0.865**; same-target cross-user 0.152 → 0.159 (unchanged); ‖r‖ 274 → 134.
- MWC-I trace (n = 300): KL under an `r` swap 1.95, vs 0.48 for a random direction and 0.053 for dropping the last item.

### MWC-M slice analysis (`tmp/slices.py`, paired bootstrap per slice; M − A)
- **Copy rows.** HR@10 0.513 / 0.479 / 0.486 (A / MW / M), **−0.0265 [−0.052, −0.001]**. NDCG@10 −0.0012 n.s. and HR@1 +0.0077, so only ranks 6–10 lose variants; HR@5 +0.0088 and NDCG@5 +0.0104 are n.s. The deficit is already in MW (MW − A on copy rows −0.0343, sig).
- **Non-copy rows.** HR@10 0.0420 / 0.0447 / 0.0466, +0.0046 [−0.0003, +0.0094] (borderline).
  - Non-copy rows whose target is not among the CF-donor outcomes (5,505 rows): 0.0258 → 0.0307, **+0.0049 [+0.0007, +0.0093], sig**.
  - Heavy users (11+ train rows): +0.020 [0.000, +0.044]. Histories of 8+ items: +0.0066.
  - Non-copy HR@1: −0.0026 [−0.0051, 0.0000].
- **Decomposition.** Copy rows cost −0.0036 overall and non-copy rows add +0.0040, so the net is ≈ 0.

### Training logs
- **MWC-M:** hinge 0.20 → 0.02; WIDER 0.70 → 0.04; motive InfoNCE 3.43 → 2.23. Projection cos: same-item other-user 0.36, same-user similar 0.75, negatives 0.18. Eval loss 0.803 / 0.605 / 0.593.
- **MWC-M2:** hinge 0.17 → 0.02; motive 3.06 → 2.14. Projection cos: cross 0.35 / within 0.62 / negatives 0.18. Eval loss 0.785 / 0.633 / 0.612.
- **MWC-G:** the evidence vector collapsed.
  - Pairwise cos across users 0.9995; cos(e_knn, e_random) 0.9998; donor attention entropy 3.46 = ln 32 (uniform).
  - cos(evidence, r0 shared direction) −0.97: WIDER on r1 selected a constant vector that cancels r0's common component.
- **MWC-G3:** the hinge on r0 stayed at 0.200 = m and WIDER on r0 at 1.000, so **r0 collapsed**. The evidence-swap gap grew 0.00 → 0.40 nats.

## Reading
1. **Original CR on MW (MWC-I) makes `r` strongly used but costs ranking**: −0.0086 HR@10 vs A, significant at all four primary cutoffs.
   - The whole loss is on non-copy rows (−0.0079 vs MW); copy rows hold.
   - The term that is learned is the **user head** (4.32 → 0.28). A user's rows are sliding-window prefixes, so pulling them together turns `r` into a user-ID code: same-user cos 0.804 → 0.865, history-item share of the top-10 0.104 → 0.127.
   - The target head stays about 0.5 nats under chance, and the same-target cross-user cos does not move (0.152 → 0.159).
2. **Removing the user head** (B) recovers most of HR@10 (n.s. vs A), but NDCG stays below. The item head (C) hurts further. The InfoNCE-I line is closed.
3. **Bank quality and random donors.**
   - Same-target other-user latents are poor substitutes in effect terms: only 10–15% are good.
   - The user's own other rows are nearly interchangeable: 66–75% are good.
   - Whether another user's `r` is useful tracks only latent closeness (corr 0.46), not a shared target, brand or history.
   - So aligning `r` by target ID carries no collaborative information the decoder can use on Toys. Effect-mined positives would just be self-smoothing.
4. **Donor routes.**
   - MWC-G meets the criterion, but its donor path collapsed to a constant (knn = off = shuffle_outcome). Its content use is r0's, i.e. MW's.
   - MWC-G3's donor evidence is genuinely read (knn − swap +0.050), but it **substitutes** for title reading: r0 collapsed and the history-brand share of the top-10 fell to 0.44 (R3 0.62). Ranking drops −0.015.
5. **Same-motive InfoNCE (MWC-M) is the first collaborative run that meets the criterion.**
   - Ranking equals A at every cutoff (all four primary Δ ≥ 0, n.s.).
   - Content dependence is the strongest of the line: shuffle −0.084, mean −0.088.
   - The raw latent shows the intended structure: same-item cross-user gap +0.273 vs +0.145 (MW), P@10 +29%, and unrelated users pushed apart (random cos 0.021).
   - The non-copy rows where MWC-I lost are now the best of the line (0.0466). The gain is significant on non-copy rows whose target is not a CF-donor outcome (+0.0049).
6. **Why the positive definition matters.**
   - MWC-I treats *every* same-user row as a positive. That rewards what is stable across a user's windows (history items, brands, i.e. identity), lets the decoder use `r` as a "who is this user" key, and so the model re-recommends history and loses novel next items.
   - MWC-M gates same-user positives on item similarity (title cos ≥ 0.8) and leaves dissimilar same-user rows neutral, which removes the identity pressure (history-item share 0.111 vs 0.127).
   - MWC-M also takes cross-user **same-item** positives from a full-train bank, so every row has its shared-motive positives (cross 100%) instead of only those in a 10k queue window (0.80).
   - MWC-M2 (SASRec neighbours instead of same-user similar rows) is weaker (NDCG@5 −0.0034 vs M, sig), so the same-user similar-item rule is the better within-user definition.
7. **Margin sweep with motive.**

   | margin | content dependence (HR@10 drop) | copy HR@10 | non-copy HR@10 |
   |---|---|---:|---:|
   | m 0.2 | ≈ 0.09 | 0.486 | 0.0466 |
   | m 0.1 | 0.08 | 0.498 | 0.0418 |
   | m 0.05 | 0.01–0.06 | 0.504 | 0.0442 |

   A smaller margin trades content use for copy-row recovery roughly proportionally; m 0.2 stays best.

Caveats:
- Single seed per model. The plain-R3 retrain spread on Toys is about 0.102–0.108 HR@10, so MWC-M's parity with R3 is within noise in both directions.
- The probe is on training rows (in-sample for the positives); a held-out probe has not been run.
- Paired t-test p was not recorded.
- The copy-row count is 905 here but 836 under the later `compare.py` rule (2026-11-11); the code that produced 905 was not found.

## Paper use
- **Main result (C2 on the MW base, Toys):**
  - MWC-M 362616: equals R3, shuffle −0.084;
  - probe 363016: cross-user same-item gap +0.273 vs +0.145, P@10 +29%;
  - non-copy slice: +0.0049, significant on non-CF-donor rows.
- **Ablation, positive definition:** MWC-I 358547 (original CR: user + target ID) vs MWC-M (same-motive) on the same base: −0.0086 sig vs equal to R3. MWC-I-B (no user head) and MWC-M2 (SASRec within-positives) are the intermediate rows. The user-identity mechanism (same-user cos, history share, non-copy loss) is the explanation.
- **Motivation / appendix:** the bank-quality study (360284–360286), the random-donor diagnostic (360311–360313) and the InfoNCE chance level. Together they show why target-ID alignment alone carries no usable collaborative information.
- **Appendix / negative results:** MWC-I-C, MWC-G (donor collapse), MWC-G3 (evidence replaces title reading), margin sweep.

## Sources
- Research log `analysis/document/research_notes/Latent utilization/research_log.md`:
  - MWC design: 1747–1755, 1775–1776;
  - MWC-G and MWC-I results / checks / analysis, MWC-I-B and MWC-I-C designs: 1819–1905;
  - trade-off comparison table: 1937–1965;
  - bank quality, chance level, random donors: 1967–2005;
  - G3 design: 2007–2022; G3 results: 2068–2097;
  - B / C results: 2024–2052;
  - InfoNCE row split: 2170–2186;
  - MWC-M / M2 / probe / slices: 2235–2322;
  - margin sweep: 2340–2343, 2486–2487, 2505–2507.
- `analysis/document/research_notes/Latent utilization/MW_to_MWC_report_2026-11-12.md` (tables via `report_tables.py`).
- Result dirs:
  - `output_dir/margin_wider_collab/Toys_and_Games_margin_0.2_wider_0.1_{infonce,infonceB,infonceC,grounded,cfg3,motive,motive_sasrec}/`;
  - `output_dir/margin_wider_collab/Toys_and_Games_margin_{0.05,0.1}_wider_0.1_motive/`;
  - `output_dir/margin_wider/`.
- Design doc: `analysis/document/collaborative_reasoning.md`.

