# E08 — Collaborative terms on the MW-R-H base
Back to [main plan](../../main.md). Dates: 2026-11-22 – 2026-11-27. Status: partial (single seed; Toys/Steam same-motive not run).

## Question
On the contribution-1 base MW-R-H, does adding a collaborative term keep ranking ≥ R3 and MW-R-H while making `r` used (more)? Two positive definitions are tested:
- **same-motive** InfoNCE (MW-R-H + motive, `MOTIVE_W=0.1`; E07's MWC-M);
- the **original collaborative-reasoning** InfoNCE (MW-R-H-CR = MWC-I's user + target heads).

On MovieLens there is a second question: can these terms make `r` encode "next is B" rather than "watched A"?

## Setup
- Base MW-R-H: rank-weighted, history-only-reference hinge (τ 0.5, sum(w·hinge)/sum(w)), margin 0.2, WIDER 0.1, 3 epochs, Qwen2.5-1.5B base, lr 3e-4.
- Collaborative term from epoch 2. Script `scripts/margin_wider_collab.sbatch`.

| run | ID | dataset | code | term |
|---|---|---|---|---|
| MW-R-H + same-motive | 374791 | CDs | 013bd0ac | same-motive InfoNCE w 0.1 (366023 recipe otherwise) |
| MW-R-H + same-motive | 374774 | MovieLens | 013bd0ac | same as above |
| MW-HN | 374776 | MovieLens | 013bd0ac | no InfoNCE; hinge negative = bank latent of another user with the same last item and a different next item (`HINGE_NEG=history`) |
| motive + HN-neg (MWC-M-HN) | 376838 | MovieLens | bd018410 (feature 1caa41c6) | same-motive + up to 16 history-matched bank negatives in the InfoNCE (`MOTIVE_HARD_NEG=16`); hinge keeps its in-batch negative |
| same-motive w 0.3 | 376839 | MovieLens | bd018410 | `MOTIVE_W=0.3` |
| probes | 374778, 376414, 376840 | MovieLens | motive_probe + history_vs_next | |
| **MW-R-H-CR** | 376544 | MovieLens | c3b42206 | original CR InfoNCE: user head (same user's other rows) 0.1 + target head (same next item) 0.1, two-phase, gate 0.5, 10k FIFO queue, 16 positives / head, 256 negatives, `EARLY_STOP=False` |
| MW-R-H-CR | 383923 / 383924 | Toys / CDs | 66b61b81 | same; InfoNCE path verified identical to MWC-I 358547 (`_relation_loss`, same trainer) |
| MW-R-H-CR | 384089 | Steam | 42e0da0a | same (383925 failed at start: KeyError 'Steam'; fix adds 'Steam' to category_dict, prompt text unchanged) |

- Failed starts: 376835 / 376836 (sbatch argument glued, fixed bd018410), probe 376837 cancelled.
- Test n: Toys 6,738 (copy 905), CDs 6,157 (copy 418), Steam 10,303, MovieLens 10,164 in the paired comparisons (copy_row = ref-NLL Q1, 2,541).

## Results

### CDs MW-R-H + same-motive (374791)
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3_ori | .0494 | .0710 | .0819 | .0989 | .0617 | .0662 | .0717 |
| MW-R-H 366023 | .0555 | .0801 | .0906 | .1121 | .0696 | .0739 | .0808 |
| **MW-R-H + motive 374791** | .0547 | .0801 | .0947 | **.1145** | .0692 | .0752 | **.0816** |
| MW-R-H-CR 383924 | .0422 | .0594 | .0695 | .0884 | .0522 | .0564 | .0625 |

- **374791 vs R3_ori:** significant at all cutoffs (HR@10 +.0156, NDCG@10 +.0099).
- **374791 vs MW-R-H 366023:** n.s. everywhere (HR@5 +.0041, p .09; HR@10 +.0024). Paired t / bootstrap p at the other cutoffs: not recorded.
- **Latent tests (374791):** mask −.0369 [.0313, .0426], mean −.0352, shuffle **−.0401**, vs MW-R-H 366023 shuffle −.017.
- **Training (374791):** motive_loss 3.04 → 1.96; cos_cross .067 → .181 vs cos_neg .014.
- Copy split and probe: not recorded.
- **Filled in 2026-11-27** (`tmp/cds_motive.py`, all 6,157 rows; diff (p_t / p_b)):
  - vs R3_ori, all significant: HR@1 +.0054 (.012 / .011), HR@3 +.0091 (4e-4 / .001), HR@5 +.0128, HR@10 +.0156,
    NDCG@3 +.0075, NDCG@5 +.0091, NDCG@10 +.0099 (p < 5e-4 for @5 / @10).
  - vs MW-R-H 366023, all n.s.: HR@1 −.0008 (.68 / .66), HR@3 .0000 (1 / 1), HR@5 +.0041 (.094 / .097), HR@10 +.0024
    (.37 / .36), NDCG@3 −.0003 (.85), NDCG@5 +.0013 (.46 / .45), NDCG@10 +.0008 (.63 / .64).
  - Copy split HR@10: copy .6196, non-copy .0777. vs R3_ori: copy +.0431, non-copy +.0136. vs MW-R-H: copy −.0048,
    non-copy +.0030.
  - Latent tests HR@10 / NDCG@10 (raw − intervention, all p < 5e-4):
    - mask +.0369 / +.0359 (copy +.115, non-copy +.031);
    - mean +.0352 / +.0344 (copy +.105, non-copy +.030);
    - shuffle +.0401 / +.0380 (copy +.136, non-copy +.033).
  - Probe: still not run.

### MovieLens runs
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | hist / top-10 | copy HR@10 | non-copy HR@10 | shuffle HR@10 (copy / non-copy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R3 371446 | .0249 | .0556 | .0806 | .1335 | .0425 | .0527 | .0698 | .48 | .4061 | .0426 | −.0027 |
| MW-R-H 369532 | .0193 | .0492 | .0701 | .1188 | .0363 | .0449 | .0605 | 1.09 | .3318 | .0478 | −.0267 |
| MW-R-H + motive 374774 | .0261 | .0531 | .0761 | .1207 | .0416 | .0509 | .0652 | .90 | .3424 | .0468 | −.0627 (−.1696 / −.0270) |
| MW-HN 374776 | .0263 | .0588 | .0841 | .1279 | .0449 | .0552 | .0694 | .39 | .3628 | .0496 | −.0006 n.s. |
| motive + HN-neg 376838 | .0241 | .0532 | .0729 | .1155 | .0406 | .0486 | .0623 | .91 | .3310 | .0437 | −.0413 (−.1149 / −.0168) |
| motive w 0.3 376839 | .0226 | .0503 | .0710 | .1127 | .0384 | .0469 | .0603 | .80 | .3117 | .0463 | −.0096 |
| **MW-R-H-CR 376544** | .0249 | .0552 | .0818 | .1265 | .0423 | .0532 | .0677 | .57 | .3617 | .0481 | −.0507 (−.1452 / −.0192) |
| MW-R-H-S 375420 (ref., not collaborative) | .0267 | .0616 | .0887 | .1380 | .0467 | .0578 | .0737 | .41 | .4026 | .0498 | −.0381 |

Comparisons (diff (p_t / p_b); * = sig):
- **Motive 374774 vs R3:** HR@10 −.0128*, NDCG@10 −.0045*, HR@5 −.0045 (p .05); @1–@5 otherwise n.s. vs MW-R-H: better at @1, @5 and NDCG; HR@10 +.0020 n.s.
- **Motive 374774 latent tests** (all p < 5e-4): mask −.0727, mean −.0545, shuffle −.0627 HR@10 (NDCG@10 −.0407 / −.0324 / −.0366). This is the largest latent use on MovieLens (R3 −.0027, MW-R-H −.0267).
- **MW-HN 374776 vs R3:** HR@10 −.0056 (.05 / .048); rest n.s.
- **MW-HN 374776 latent tests:** shuffle −.0006 [−.0030, +.0018] n.s.; mean −.0103 sig; mask −.0715 sig. Per-user content is unused.
  - Cause: the hinge is trivially satisfied against **stale** bank latents (margin_loss .02–.08 from epoch 1).
- **CR 376544 vs R3:**
  - @1–@5 n.s.; HR@10 −.0070* (.015 / .006); NDCG@10 −.0021 (.17 / .15); HR@5 +.0060 sig.
  - copy rows −.0445 sig; non-copy +.0055 (.034 / .041).
- **CR 376544 vs other runs:**
  - vs MW-R-H: significantly better at all cutoffs.
  - vs motive 374774: HR@5 +.0057 (.017 / .025), HR@10 +.0058 (.037 / .038), NDCG n.s.
  - MW-R-H-S − CR: HR@3 +.0064, HR@5 +.0070, HR@10 +.0115, NDCG@10 +.0060, all sig.
- **CR 376544 latent tests** (p < 5e-4): mask −.0382, mean −.0437, shuffle −.0507. Hinge satisfied early (margin_loss .003 from epoch 1.5).
- **HN-neg 376838:**
  - vs R3: HR@5 −.0077, HR@10 −.0180, NDCG@5 −.0041, NDCG@10 −.0075, all sig.
  - vs 374774: n.s. (HR@10 −.0052, .06 / .078; NDCG@10 −.0030, .049 / .06).
  - Latent tests (p < 5e-4): mask −.0333, mean −.0405, shuffle −.0413.
  - InfoNCE did not separate the hard negatives: cos_hard ≥ cos_cross throughout (epoch 2.5: positives .619, hard .668, random .579).
- **Weight 0.3 376839:**
  - vs R3: HR@3 −.0053, HR@5 −.0095, HR@10 −.0209, NDCG@5 −.0058, NDCG@10 −.0094, all sig.
  - vs 374774: HR@1 −.0034, HR@5 −.0050, HR@10 −.0081, NDCG@10 −.0049, all sig.
  - Latent tests: mask −.0059 (.004 / .002), mean −.0082, shuffle −.0096 (p < 5e-4).
  - **Confounded:** the hinge was never satisfied (margin_loss .202 → .194) and CE spiked to 1.64 in epoch 0.5–1.0, before the InfoNCE switched on. The run diverged from 374774 early (runs are not deterministic).

### MovieLens probes ("history vs next"; first 20k train rows; 374778, 376414, 376840)
| model | gap_history (same last item, different next) | gap_next (same next, different last) | next / history | cross-user same-item gap | within-user gap |
|---|---|---|---|---|---|
| R3 371446 | +.262 | +.036 | .14 | | +.0074 |
| MW-R-H 369532 | +.061 | +.025 | .41 | | +.107 |
| MW-nc 374722 | +.186 | +.068 | .37 | +.075 | +.054 |
| MW-R-H + motive 374774 | +.318 | +.151 | .48 | +.159 | +.064 |
| + HN-neg 376838 | +.225 | +.102 | .45 | +.108 | |
| weight 0.3 376839 | +.432 | +.168 | .39 | +.176 | |

- MW-R-H same-next retrieval P@10 .0071 (chance .0018; R3 .0284).
- gap_next < gap_history in every model.

### MW-R-H-CR on four datasets (original CR on MW-R-H, lr 3e-4)
Toys 383923 (n = 6,738; copy_row 905):

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| plain R3 (A) | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .5127 | .0420 |
| MW-R-H 365504 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | .5039 | .0473 |
| MW-R-H s1 A40 368189 | .0404 | .0647 | .0804 | .1088 | .0544 | .0608 | .0700 | .5105 | .0465 |
| MW-R-H s2 A40 366001 | .0334 | .0574 | .0717 | .0944 | .0471 | .0530 | .0603 | .4630 | .0372 |
| MWC-I 358547 | .0322 | .0561 | .0718 | .0966 | .0455 | .0520 | .0600 | .4818 | .0369 |
| **MW-R-H-CR 383923** | .0337 | .0637 | .0806 | .1076 | .0509 | .0578 | .0665 | .5061 | .0458 |

- **vs R3 (A):** all n.s. (HR@1 −.0030, p .12; HR@5 +.0010, .68 / .70; HR@10 +.0024, .39 / .40; NDCG@5 −.0006, .72; NDCG@10 −.0002, .90 / .94).
- **vs MW-R-H 365504:** significantly worse at HR@1 −.0046 (.008 / .006), HR@3 −.0045 (.032 / .036), NDCG@3 −.0045, NDCG@5 −.0036 (.018 / .024), NDCG@10 −.0033 (.032 / .038). HR@5 −.0022 (.33) and HR@10 −.0010 (.70) n.s.
- **vs other MW-R-H seeds:** vs 368189, NDCG@10 −.0034 sig, HR@10 n.s.; vs 366001, better at @3–@10 (sig).
- **Latent tests** (raw − intervention, p < 5e-4): mask +.0793 / +.0505, mean +.0706 / +.0469, shuffle **+.0793** / +.0516 HR@10 / NDCG@10. Shuffle on copy rows +.35, non-copy +.037; MW-R-H 365504 shuffle was −.035.

Steam 384089 (n = 10,303):

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| R3 371447 | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 |
| MW-R-H 369662 | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 |
| MW-R-H clip1.0 383065 | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 |
| MW-R-H-CR 384089 | .0071 | .0165 | .0235 | .0361 | .0125 | .0154 | .0195 |

- vs R3: all n.s. (HR@10 +.0003, .86 / .90; NDCG@10 +.0003, .76 / .74).
- vs 383065: HR@1 −.0028 sig (.0055 / .006); rest n.s.
- vs 369662: worse at HR@1/3 and NDCG@3/5/10 (NDCG@10 −.0028, .0025 / .005).
- Latent tests: all n.s., within ±.0007 (mask +.0003, p .77; shuffle +.0007, .35 / .41).

CDs 383924 (n = 6,157; copy_row 418): ranking is in the CDs table above; copy HR@10 .5574, non-copy .0542 (R3_ori .5766 / .0641, MW-R-H .6244 / .0748).
- vs R3_ori: significantly worse at every cutoff (HR@5 −.0123; HR@10 −.0106, .0006 / .003; NDCG@5 −.0098; NDCG@10 −.0092, < 5e-4).
- vs MW-R-H 366023: worse at every cutoff (HR@10 −.0237, NDCG@10 −.0183, p < 5e-4).
- Latent tests @10 (raw − intervention):
  - mask −.0068 HR / −.0058 NDCG: the zero slot is **better** than raw (copy rows −.062);
  - mean +.0122 / +.0106 and shuffle +.0060 / +.0052, both sig, almost all on copy rows (mean +.148, shuffle +.086; non-copy +.002 / +.000).

**Four-dataset summary of MW-R-H-CR**
| dataset | ranking vs R3 | vs MW-R-H | latent content | hinge in training | user / target InfoNCE | criterion |
|---|---|---|---|---|---|---|
| MovieLens 376544 | = @1–@5, HR@10 −.0070 sig | better (all cutoffs) | used (shuffle −.051) | satisfied (.003) | not recorded | no (HR@10) |
| Toys 383923 | = at all cutoffs | NDCG −.003 sig | strongly used (shuffle −.079) | satisfied (.023) | 1.87 → .39 / 4.43 → 3.82 (chance ≈ 4.5) | **yes** |
| CDs 383924 | worse at all cutoffs (HR@10 −.011) | −.024 sig | mixed (mask > raw) | not satisfied (.194–.203) | 3.54 → .70 / 4.4 → 4.0 | no |
| Steam 384089 | = at all cutoffs | HR@1 −.003 sig (vs 383065) | unused (mask neutral) | not satisfied (.20) | 3.57 → 1.44 / 4.16 → 4.20 | no |

Training logs (`tmp/cr_progress.py`):
- **Toys:** hinge .146 → .023 (as MW-R-H 365504); WIDER .287 → .047; positive rates user / target .61–.65 / .80–.84.
- **Steam:** WIDER .724 in epoch 0.5–1.0 (partial collapse, grad norm 60 early), recovering to .05 once the InfoNCE was on.
- **CDs:** CE ends .874 vs MW-R-H .662; MW-R-H's hinge falls to .077 / .062 in epochs 2–3, CR's never does.

## Reading
1. **Same-motive on MW-R-H (CDs 374791)** is the strongest C2 result on this base.
   - All cutoffs are significantly above R3_ori, it equals MW-R-H (n.s., numerically higher at HR@5 / HR@10 / NDCG@5 / NDCG@10), and latent use roughly doubles (shuffle −.040 vs −.017).
   - The collaborative term buys content dependence at no ranking cost. It also aligns same-item cross-user latents (cos_cross .181 vs cos_neg .014).
2. **On MovieLens no collaborative variant beats R3.** The MW-R-H base itself is below R3 there: it re-recommends watched movies (1.09 history items per top-10).
   - Same-motive raises latent use most (−.0627) and moves ranking partly back (better than MW-R-H at @1, @5 and NDCG) but stays below R3 at @10.
   - The probes show the motive term raises gap_next 6× (.025 → .151), but gap_history rises too (.061 → .318): `r` still encodes "watched A" more than "next is B".
   - Neither history-matched negatives (376838) nor a heavier weight (376839, confounded) changes this. MW-HN (hinge against stale bank latents) loses per-user content.
3. **The original CR on MW-R-H works only where the hinge forms (Toys, MovieLens).**
   - On Toys it meets the criterion (equals R3, shuffle −.079 ≈ MWC-I's level). That reverses MWC-I's ranking loss on the MW base: the reference-weighted hinge protects ranking. Still, it is below MW-R-H at the top ranks and on NDCG.
   - On CDs and Steam the user head (the only head that learns; the target head stays at chance on all datasets) competes with the hinge, which never forms. CDs then loses ranking, as MWC-I did on Toys; on Steam the slot becomes fully ignorable.
   - MW-R-H alone is ≥ CR on ranking on every dataset except MovieLens, where CR is better than MW-R-H but still below R3.
4. **Positive definition.** Cross-dataset evidence favours same-motive over the original CR: CDs motive 374791 is .1145 / .0816 vs CR 383924 .0884 / .0625. On MovieLens, CR edges out motive at HR@5 / HR@10 (sig) with less latent use. The direct Toys comparison is missing.

Caveats:
- Every run is single seed, and runs are nondeterministic (376839 diverged from 374774 in epoch 1, before its weight could matter).
- CR uses a 10k FIFO queue with gate 0.5 and two heads; motive uses the full row bank. So CR vs motive differs in more than the positives.
- Steam CR ran at lr 3e-4, which is unstable on Steam.
- MovieLens latent checkpoints of rank / rank_cr / rank_motive / motive_hn / motive_w03 / hardneg were deleted 2026-11-26; results are kept.

## Not yet run (open C2 runs)
- **MW-R-H + same-motive on Toys** (the only same-base comparison with MW-R-H-CR 383923 and MW-R-H 365504) and **on Steam**.
- Seed replicates of CDs 374791 and Toys 383923.
- CR on Steam at lr 5e-5 (where the MW-R-H hinge formed).
- Copy split and motive probe for CDs 374791 and for Toys / CDs CR.
- History-excluded evals of MovieLens 374775 / 374777 (cancelled 2026-11-22 by user, to redo later).
- A held-out motive probe (all probes use training rows).

## Paper use
- **Main (C2 on the contribution-1 base):** CDs MW-R-H + same-motive 374791: significant over R3_ori at all cutoffs, equal to MW-R-H, latent use 2×. Needs a Toys / Steam replication before it can be a headline.
- **Ablation, positive definition:** MW-R-H-CR on four datasets vs MW-R-H + same-motive (CDs, MovieLens), with the training-log reading: hinge forms only on Toys / MovieLens, the target head is at chance everywhere, and the user head competes with the hinge.
- **Appendix / analysis:** MovieLens probes (history vs next), MW-HN stale-negative shortcut, HN-neg and weight-0.3 negatives.
- **Dev only:** MW-R-H-S 375420 (reference row, not collaborative).

## Sources
- Research log `analysis/document/research_notes/Latent utilization/research_log.md`:
  - MW-HN / motive design and CDs submission: 2895–2906;
  - motive probe 374778, MW-R-H copy-row interventions: 2930–2945;
  - CDs 374791 and MovieLens results: 2963–3007;
  - CR submission, probe 376414, follow-ups 376835–376840: 3031–3072;
  - 376544 / 376838 / 376839 / probe 376840: 3174–3221;
  - CR on 3 datasets, Steam fix, progress, Toys / Steam / CDs results and summary: 3521–3612.
- Memory: `r3-margin-wider-collab.md` (lines 47–50).
- Result dirs: `output_dir/margin_wider_collab/{CDs_and_Vinyl,Movielens,Toys_and_Games,Steam}_margin_0.2_wider_0.1_refweight_nolatent_rank_{motive,cr,hardneg,motive_hn,motive_w03}/`.
- Scripts: `tmp/done_1122.py`, `tmp/ml_runs.py`, `tmp/toys_cr.py`, `tmp/cds_cr.py`, `tmp/steam_cr_vs_clip.py`, `tmp/cr_progress.py`.

