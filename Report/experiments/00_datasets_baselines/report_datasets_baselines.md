# E00 — Datasets and baselines
Back to [main plan](../../main.md). Dates: 2026-10 (Musical, Toys, CDs) to 2026-11-23 (cross-dataset analysis). Status: done (SASRec missing for most datasets, see Gaps).

## Question
Which datasets does the paper use, how were they built, what do simple and LLM baselines score on them, and which dataset traits predict where MW-R-H helps?

## Setup
- Split format: one row per (user, next item). History = earlier items (last 10). Global-temporal train / valid / test. Constrained 10-beam title decoding (@20 = @10). Same R3 prompt on every dataset (category noun from the category dict).
- Backbone for R3, no-latent SFT and MW-R-H: base Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs.
  - R3 and no-latent SFT keep the best epoch (early stopping). MW-R-H keeps epoch 3 (a recipe difference, see E06).
  - Exception: the Goodreads, Yelp and MobileRec R3 baselines were trained from Qwen2.5-1.5B-**Instruct**.
- Baselines:
  - popularity (train counts);
  - item-CF (= last-3 co-occurrence, history excluded);
  - last-item-successor table (top-10 train successors of the last history item);
  - SASRec (Cornac, untuned Amazon hyper-parameters: dim 128, 2 blocks, max_len 10);
  - no-latent SFT (`no_latent_ref` jobs, `plain_sft_train.py`; also MW-R-H's history-only reference);
  - R3.
- Row and item counts were counted from the split CSVs and `info/*.txt` on 2026-11-27 (read-only). Everything else is copied from the research log and memory notes.

## Results

### Table 1. Dataset construction and size (paper: main dataset table = first four rows)

| dataset | file id | construction / filtering | #items | train / valid / test rows | copy_row (test) |
|---|---|---|---:|---|---|
| Toys_and_Games | Toys_and_Games_5_2017-1-2018-11 | Amazon 2018, 5-core, 2017-01..2018-11, global-temporal split (train targets ≤ 2018-02-08 ≤ valid ≤ 2018-04-15 ≤ test ≤ 2018-09-28) | 6,299 | 53,898 / 6,737 / 6,738 | 905 (same-day same-brand variant, Jaccard ≥ 0.3) |
| CDs_and_Vinyl | CDs_and_Vinyl_5_2015-10-2018-11 | Amazon 2018, 5-core, 2015-10..2018-11 | 5,841 | 49,251 / 6,156 / 6,157 | 418 (same rule) |
| Steam | Steam_5_2016-1-2017-12 | UCSD (Kang) reviews as interactions, review date = time, first review per (user, game); 2016-01..2017-12; popularity thinning (keep with p = min(1, 2000 / game reviews in window)); 12k users sampled from those with ≥ 5 reviews (seed 42), 5-core; crawl-truncated games dropped (`truncation_grace_days=90`: 921 games, 7.1% of reviews); release freeze `release_before=2017-09-01` (1,980 games, 5.2%) | 4,419 | 82,419 / 10,302 / 10,303 | not defined (target never in history) |
| MovieLens | Movielens_5_2014-1-2015-3 | MovieLens-20M, ratings ≥ 4, 2014-01..2015-03, last 20 ratings per user, 6,700 of 8,174 eligible users (seed 42), 5-core; leading article moved to front of title; description = genres + top-10 tags | 2,916 | 81,308 / 10,163 / 10,164 | 2,541 (= lowest quarter of history-only reference NLL; user naming) |
| Musical Instruments (dev) | Musical_Instruments_5_2015-1-2018-11 | Amazon 2018, 5-core | 5,030 | 66,500 / 8,312 / 8,313 | ~10% same-day same-brand rows (300 / 3,000-row subset) |
| Goodreads (dropped) | Goodreads_5_2017-1-2017-12 | `is_reviewed==1` interactions timed by the review's `date_added` (15.74M timed, 0.5% same-second ties); global first-in-time dedup of (user, work); editions merged into works; 2017, 26k sampled users, 5-core | 7,949 | 77,353 / 9,669 / 9,670 | not defined (series-continuation rows 778 = 8%) |
| Yelp (dropped) | Yelp_5_2021-1-2022-2 | converted from the VRec Yelp split (same rows), history = last 10, ids remapped, " (street)" added to 2,539 duplicate chain names; 3.9% revisit targets | 12,748 | 124,842 / 14,733 / 13,344 | not defined |
| MobileRec (dropped) | Mobilerec_5_2021-1-2022-1 | 2021-01..2022-01, 10k sampled users (seed 42), 5-core, (user, app) duplicates dropped, history 10, developer suffix for duplicate names | 5,017 | 55,424 / 6,928 / 6,929 | not defined |

Steam first version (2026-11-19):
- 4,683 items, 83,064 / 10,383 / 10,383 rows.
- **20.7%** of test targets unseen in train, from crawl truncation and from releases after the cutoff.
- After the fixes: 1.6% (valid 4.6%).
- Ending the window earlier did not help (17–18%).
- Without thinning, popularity HR@10 was .062 vs item-CF .074 (heavy head).

Unseen test targets on the other datasets: Toys 5.0%, CDs 0.3%, MovieLens 0.2%, Goodreads 9.6%.

### Table 2. Baselines on the test set (HR@1/3/5/10, NDCG@3/5/10; HR@1 = NDCG@1)

| dataset | popularity HR@10 | item-CF HR@10 | last-item-successor HR@10 | SASRec (untuned) | no-latent SFT | R3 |
|---|---:|---:|---:|---|---|---|
| Toys | .0110 | .094 (memory note only) | .089 | HR@10 .0883 (VRec Cornac export, history excluded) | 365404: HR@1 .0377, HR@5 .0785, HR@10 .1039, NDCG@5 .0585, NDCG@10 .0667 (@3 not recorded) | plain R3 (A): .0367/.0625/.0795/.1052, .0514/.0584/.0667; R3_ori: .0377/.0607/.0778/.1023, .0510/.0580/.0659 |
| CDs | not recorded | not recorded | .120 | not recorded | 365885: .0417/.0637/.0734/.0945, .0545/.0585/.0653 | R3_ori: .0494/.0710/.0819/.0989, .0617/.0662/.0717 |
| Steam | .0094 | .0401 | .023 | not recorded | 369661: .0065/.0139/.0208/.0357, .0108/.0136/.0184 | 371447: .0077/.0151/.0226/.0358, .0119/.0150/.0192 |
| MovieLens | .0495 | .1282 | .104 (.1494 as a count table, history excluded) | not recorded | 369531: .0253/.0541/.0797/.1332, .0419/.0524/.0695 | 371446: .0249/.0556/.0806/.1335, .0425/.0527/.0698; seed 1 375356: .0261/.0581/.0819/.1372, .0445/.0542/.0720 |
| Musical (dev) | not recorded | not recorded | not recorded | not recorded | HR@10 .1009 / NDCG@10 .0790 | plain R3 v1: .1226 / .0950 |
| Goodreads | .015 | .066 | not recorded | 367997: .0180/.0271/.0341/.0453, NDCG@10 .0298 | 368191 (base): HR@3/5/10 .0340/.0405/.0525, NDCG@3/5/10 .0289/.0316/.0354 | 367929 (Instruct): .0242/.0367/.0433/.0567, .0315/.0342/.0385 |
| Yelp | .009 | .038 | not recorded | 367997: HR@10 .0324 / NDCG@10 .0167 | not run | 367937 (Instruct): .0051/.0110/.0172/.0321, .0085/.0110/.0158 |
| MobileRec | .006 (all 2021 users .0125) | .003 (all users .0074) | not recorded | not run | not run | 367598 (Instruct): .000/.0004/.0007/.0020, NDCG@10 .0008 (82% same top-1) |

### Table 3. R3 vs no-latent SFT, and R3's own latent use (paired, all test rows; Δ = intervention − raw, HR@10)

| dataset | R3 − no-latent | R3 latent tests |
|---|---|---|
| Toys | R3_ori .1023 vs .1039 HR@10, .0659 vs .0667 NDCG@10: n.s. at every cutoff | R3_ori `r` is identical for all users (cos 1.0), so content is unused by construction |
| CDs | R3_ori significantly better at @1/@3/@5 and NDCG@10 (.0717 vs .0653); HR@10 .0989 vs .0945 (p .09) | not recorded in the sources read |
| Steam | 371447 HR@10 .0358 vs .0357 (p .95), NDCG@10 .0192 vs .0184 (p .34): all n.s. | mask −.0141*, mean +.0003 n.s., shuffle −.0007 (p_t .14). Slot used, content unused |
| MovieLens | 371446 .1335 vs .1332 (p .91): all n.s.; seed 1 only HR@3 significant (+.0040, p .03); 2-seed mean = no-latent | 371446: mask −.0923*, mean −.0022 (p_t .03), shuffle −.0027 (p_t .02 / p_b .01), i.e. ~2% of HR@10. Seed 1: shuffle n.s., mean −.002*, mask −.080 |
| Musical | R3 v1 .1226 vs SFT .1009 (recipes not verified comparable) | mean / shuffle change HR@10 by ≤ .0015 (n.s.) |

Collapse (distinct top-1 / max share):
- Steam R3: 447 / 13.2% (no-latent 385 / 25.2%).
- MovieLens R3: 415 / 6.6%.
- Goodreads R3: 874 / 5%.
- Yelp R3: 1,168 / 5.4%.
- MobileRec R3: 82% of users get one app.

### Table 4. Dataset characteristics (2026-11-22/23 cross-dataset analysis, `tmp/xds.py`; R3 = plain R3 (A) / R3_ori / 371447 / 371446)

| | Toys | CDs | Steam | MovieLens |
|---|---:|---:|---:|---:|
| train rows per item | 8.6 | 9.2 | 18.7 | 27.9 |
| train-target share of the top-10% items | .35 | .37 | .36 | .55 |
| lexical-target rows (target title Jaccard ≥ .5 with a history title) | 10.6% | 8.8% | 3.0% | 1.4% |
| target in history | .015 | .031 | 0 | 0 |
| last-item-successor table HR@10 | .089 | .120 | .023 | .104 |
| R3 − successor table HR@10 | +.016 | −.021 | +.013 | +.030 |
| R3 hits: lexical / transition-only / neither | .56/.22/.22 | .44/.38/.17 | .45/.10/.45 | .04/.48/.48 |
| R3 last-item successors per top-10 | .85 | .76 | .34 | 2.26 |
| R3 history items per top-10 | 1.01 | .65 | .58 | .48 |
| MW-R-H − R3 HR@10 (default seed) | +.0034 | +.0132 | +.0030 | −.0148 |

MovieLens specifics:
- 62–63% of test rows come less than 60 s after the previous rating. Bulk-rating sessions follow a near-canonical list.
- 34% of those fast rows have their exact (last item → target) pair in train. R3 HR@10 is .390 on them vs .053 elsewhere.
- 75% of copy_rows (ref-NLL Q1) are rating bursts.
- P(target in last-item bucket | copy_row) .378 vs .073 on other rows.

## Reading
- **Paper datasets.** Toys, CDs, Steam and MovieLens all have item-CF well above popularity (a sequential signal). R3 is at or above item-CF on all but Steam (R3 .0358 < item-CF .0401).
- **Dropped datasets:**
  - MobileRec: no sequential signal (item-CF .003 < popularity .006), and R3 collapsed (HR@10 .002).
  - Yelp: R3 .0321 ≈ untuned SASRec .0324 < item-CF .038. Never used for MW-R-H.
  - Goodreads: R3 is below item-CF (.0567 vs .066) and MW-R-H failed (E06). The user stopped this line on 2026-11-20.
- **R3's latent adds nothing over no-latent SFT** on Toys, Steam and MovieLens; only CDs shows a significant R3 gain. Plain R3 does not use `r`'s content anywhere (on MovieLens the effect is significant but ~2% of HR@10).
- **What separates the datasets is what the decoder can read from the history.**
  - Toys and CDs are title-readable: lexical hits are about half of R3's hits.
  - MovieLens is a dense item-transition dataset: 4% lexical hits, 27.9 rows/item, and R3 already lists 2.26 last-item successors per top-10.
  - This is the axis along which MW-R-H helps (most on CDs) or hurts (MovieLens); see E06.

## Paper use
- Table 1 (first four rows) + Table 2 (popularity, item-CF, no-latent, R3): **main** dataset/baseline table. SASRec needs protocol-matched runs for Toys, CDs, Steam and MovieLens first.
- Table 3: **motivation** (R3 ≈ no-latent; content unused).
- Table 4: **analysis / appendix**.
- Musical: **dev only**. Goodreads, Yelp, MobileRec: **appendix**, one line each on why dropped.

## Gaps and inconsistencies
- SASRec is not recorded for CDs, Steam or MovieLens. The Toys number uses a different protocol (history excluded) from Goodreads and Yelp (history not excluded).
- CDs popularity and item-CF are not recorded. Toys item-CF .094 appears only in a memory note.
- CONVENTIONS.md lists MovieLens n = 9,915; the test CSV has 10,164 rows and all comparisons use 10,164.
- Toys copy_row is 905 in the MW-R-H tables but 836 in the GRPO analyses.
- The MovieLens successor baseline has two definitions (.104 vs .1494).
- Goodreads, Yelp and MobileRec R3 use the Instruct backbone. The Steam no-latent prompt noun is "games" while R3 uses "video games".

## Sources
- Research log `analysis/document/research_notes/Latent utilization/research_log.md`:
  - Goodreads prep 2026-11-18 (~l. 2737);
  - Steam / MovieLens 2026-11-19..20 (l. 2755–2813);
  - cross-dataset diagnosis 2026-11-22 (l. 2886–2893);
  - MovieLens headroom 2026-11-23 (l. 3008–3030);
  - cross-dataset characteristics 2026-11-23 (l. 3031–3046);
  - Toys split (l. 619), Toys popularity (l. 848), Toys SASRec (l. 569).
- Memory: r3-steam, r3-movielens, r3-goodreads, r3-yelp, r3-mobilerec, r3-nolatent-sft-baseline, r3-sameday-variant-artifact.
- SASRec: scratch `R3/output_dir/sasrec_baseline/{Goodreads,Yelp_R3}/metrics_r3_protocol.json`.
- No-latent: `R3/output_dir/margin_wider_collab/no_latent_ref/<cat>/eval_no_latent.json`.
- Data: `R3/data/<dataset>/{train,valid,test}/*.csv`, `info/*.txt`.
