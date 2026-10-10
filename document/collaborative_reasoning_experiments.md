# Collaborative reasoning experiments: complementary CF for R3

> Follows [Decoder insensitivity to the latent](../analysis/document/decoder_insensitivity_to_latent.md) (Experiments 1–2) and [Collaborative reasoning](../analysis/document/collaborative_reasoning.md). Branch `worktree-R3CollaborativeReasoning`.

## Why

The decoder-insensitivity study found that the latent slot works as a channel. The decoder reads non-redundant content almost as well as text, and the SASRec vector was used as much as a history item. But SASRec's information **substitutes for** what R3 already gets from titles, rather than adding to it:
- 81% of SASRec hits are also R3 hits;
- even a perfect combination adds at most +1.7 HR@10;
- `sasrec_raw` ranks at 0.1196, versus 0.1226 for plain R3.

The question here is therefore which collaborative signal **covers R3's misses**, and how to deliver it.

A second issue showed up while building this. The baseline SASRec was trained on every training user, so on *training* rows it has already fit those rows' own targets. In the smoke run, the full model hit **58.6%** of training-row targets versus **7.3%** of validation targets. `sasrec_raw` was trained on that over-accurate signal, and so learned to trust SASRec far more than SASRec deserves at test time. Every source here avoids that (see [Leakage and calibration](#leakage-and-calibration)).

## Ideas → implementation

| # | idea | implementation | output |
|---|---|---|---|
| 1 | Screen CF sources for complementarity before training | `src/collab_reasoning/cf_sources.py`, `build_candidates.py`: 8 sources scored on every split; test-split HR@10, **HR@10 on R3 v1 misses**, union with R3, P(R3 hit given source hit) | `analysis/results/collab_reasoning/full/screen.{md,json}` |
| 2 | CF text control (ceiling) | `cf_text`: the fused scorer's top-10 titles written into the prompt ("Items bought by users with similar purchase histories: …") | `…/eval/full_cf_text/summary.json` |
| 3 | Candidate slots, one per item | `cf_slots`: 10 `<|Cand|>` tokens whose embeddings are `normalize(P(SASRec item vector)) · token_norm` (`src/latent/collab_latent.py`); present in both R3 passes | `…/eval/full_cf_slots/` |
| 4 | CF trained on R3's mistakes | residual scorer: same listwise model, trained only on validation rows R3 v1 misses, with R3's (wrong) top-10 added as hard negatives; `residual_text`, `residual_slots` | `…/eval/full_residual_{text,slots}/` |
| 5 | Predict the target's group; feed the prediction | group predictor: p(CF cluster, k=150) = cluster mass of the fused item distribution; `group_pred` adds it to the thought slot (`side_info_kind=row_table`). **Degraded-oracle sweep**: the trained `oracle_cfcluster150` model given the true cluster with probability p | `results/side_info/collab_full_group_pred/`, `…/degraded_oracle/full_cfcluster150/` |
| 6 | Signals the LLM cannot know | `trend` (other users' purchases in the 60 days before the row's history cutoff), `donor` (outcomes of other users' earlier rows with overlapping history), `longknn` (item-kNN over the user's whole past, beyond the 10-item prompt; 21.7% of test rows have a full 10-item history) | in the screen and as fused-scorer features |
| 7 | Multi-round retrieval inside the latent | not built separately: it overlaps with the outcome-memory pilot in another session. The `donor` source is a training-free version of its retrieval, and the screen reports whether donor outcomes cover R3's misses | screen row `donor` |

### Sources (`cf_sources.py`)

| source | signal |
|---|---|
| `sasrec` | SASRec next-item scores over the 10-item history |
| `itemknn` | item-item cosine co-occurrence, recency-weighted over the history |
| `longknn` | the same over the user's whole timeline |
| `markov` | first-order transitions from the last 3 items |
| `pop` | training popularity |
| `trend` | popularity among other users in the 60 days before the history cutoff |
| `donor` | session-kNN: targets of the 200 most similar earlier rows of other users (IDF-weighted history overlap) |
| `titlesim` | TF-IDF title similarity to the history; a proxy for what a title reader picks |

Items already in the 10-item history are excluded from every candidate list. Only 0.8% of test targets repeat a history item, so little is lost; R3 still predicts them itself.

### Fused and residual scorers (`build_candidates.py`)

- **Model.** A listwise MLP (24 features → 64 → 1, softmax over the pool). The pool is the union of each source's top-50. Features per source: reciprocal rank, top-10 indicator, and within-row z-scored score.
- **Fused.** Trained on the validation split. Validation is out-of-period for every statistic, exactly like test. 20% of rows are held out for early stopping.
- **Residual.** Trained only on validation rows R3 v1 misses, with R3's top-10 added to the pool as negatives. It cannot see R3 at inference. It learns which feature patterns (e.g. `titlesim`, the R3-like proxy) point to targets R3 does not find.

## Leakage and calibration

1. **Leave-user-out statistics.** For a training row, the row's own user is removed from co-occurrence, transitions and popularity. The test `tests/test_collab_reasoning.py::test_leave_user_out_knn_matches_brute_force` checks the closed form against a brute-force recompute.
2. **Fold SASRec.** Four SASRec models are trained (`train_sasrec_folds.py`, same Cornac configuration as the baseline). Each one leaves out a quarter of the training users (`sha256(user) % 4`), and a training row is scored by the model that never saw its user. Validation and test rows use the baseline SASRec, which never saw them.
3. **Time filters.** `trend` and `donor` read only other users' events strictly before the row's latest history timestamp.
4. **Calibration of training rows.** Even with 1–3, training-period statistics fit training rows better than test rows: training-row candidates were 2–3× more accurate in the smoke run, because popularity and co-occurrence drift over time. So training-row lists, and group vectors, are replaced by another random training row's with probability `q = 1 − valid_rate / train_rate`. `valid_rate` is the scorer's held-out validation hit rate. Training-row candidates then hit as often as test-row candidates will, and the LLM cannot learn to over-trust them. `q` is recorded in each table (`calibration`).

## Measurements

Each LLM variant is plain R3 SFT on Musical Instruments (Qwen2.5-1.5B-Instruct, 3 epochs, effective batch 128, lr 3e-4, best eval-loss checkpoint), as in the side-information runs. `collab_eval.py` reports:

- **Full-test ranking** (8,313 rows): HR/NDCG@10, plus:
  - **HR@10 on the rows R3 v1 misses** and the **union with v1**: the complementarity this work is about;
  - HR@10 when the target is / is not among the candidates;
  - copy rate (share of the top-10 that are candidates).
- **Teacher-forced shift** (2,000 rows), as a ratio to dropping one history item:
  - `swap_cands`: another user's candidates;
  - `no_cands`: empty list;
  - `swap_r`: own candidates, another user's thought;
  - `mask`.
- **Generation** (1,000 rows) with own, donor's and no candidates: HR@10, donor-target hits, copy rates.

`group_pred` is evaluated by `side_info_eval.py` (same measurements as Experiment 1).

### How to read the results

| outcome | reading |
|---|---|
| a source's *HR@10 on R3 misses* is ≈ 0, or its union with R3 ≈ R3 | it substitutes for title evidence; not worth delivering |
| `cf_text` improves full-test HR@10 over v1 (0.1226) and HR@10 on v1 misses | complementary CF exists and the decoder can use it; then compare slots with text |
| `cf_text` ≈ v1 | even with text the decoder gains nothing; CF on this dataset is substitutable at the candidate level |
| `cf_slots` ≈ `cf_text` | the slot channel carries item identity as well as text (cheaper prompts) |
| `residual_*` > `cf_*` on v1 misses and in union | training CF on R3's mistakes yields more complementary candidates |
| `group_pred` > v1 | a realistic group prediction helps; compare its accuracy with the degraded-oracle curve |
| degraded oracle: HR@10 at accuracy p crosses 0.1226 at p* | a group predictor needs about p* accuracy (an upper estimate of the requirement, since the oracle model fully trusts the slot) |
| `swap_r` ratio stays ≈ 0 in every variant | the thought itself remains redundant; the gain comes from the candidates, not from `r` |

## Jobs

Partition `researchshort`, constraint `3090|a40|l40|l40s|a100|h100`, logs in `R3/run_logs/<name>.<id>.out`.

| job | script | depends on | what |
|---|---|---|---|
| 315414 | `scripts/collab_sasrec_folds.sbatch` | — | 4 fold SASRec models → `scratchDirectory/R3/collab_reasoning/sasrec_folds/manifest.json` |
| 315425 | `scripts/collab_smoke.sbatch` | — | every GPU path with tiny limits on padded smoke tables (`BUILD_TAG=smoke_padded`); smoke numbers are not results |
| 315426 | `scripts/collab_v1_valid_preds.sbatch` | 315425 | R3 v1 top-10 on the validation split |
| 315427 | `scripts/collab_build.sbatch` | 315414, 315426 | screen + scorers + tables in `data/Musical_Instruments/info/collab_reasoning/*_full.pt` |
| 315428 | `scripts/collab_degraded_oracle.sbatch full` | 315427 | dose-response p ∈ {0, .1, .2, .3, .5, .7, 1} on 2,000 test rows + plug-in top-1 cluster |
| 315429 | `scripts/collab_train.sbatch cf_text full` | 315427 | idea 2 |
| 315430 | `scripts/collab_train.sbatch cf_slots full` | 315427 | idea 3 |
| 315431 | `scripts/collab_train.sbatch residual_text full` | 315427 | idea 4 (text) |
| 315432 | `scripts/collab_train.sbatch residual_slots full` | 315427 | idea 4 (slots) |
| 315433 | `scripts/collab_train.sbatch group_pred full` | 315427 | idea 5 |

Check progress with `squeue -u $USER`. The screen is readable as soon as 315427 finishes: `cat analysis/results/collab_reasoning/full/screen.md`.

## Validation before submission

- `tests/test_collab_reasoning.py` (8 tests) covers:
  - the leave-user-out kNN against brute force;
  - slot embedding scale and padding, slot order and beam expansion;
  - the slot-count check;
  - candidate prompt lines;
  - calibration;
  - duplicate-title canonicalization;
  - fold assignment.
- The full suite passes (56 tests).
- `build_candidates.py` ran end to end on 600 rows (tag `smoke`).
- My scorer reproduces the baseline SASRec: test HR@1 0.065 vs Cornac's 0.066; HR@10 0.092 vs 0.098, the difference being Cornac dropping the 1.8% of unknown targets.
- Fold training ran for one CPU epoch.
- The GPU paths are exercised by the smoke job before any full job starts. The first smoke (315415) caught a slot-model init bug (`cand_token_id` read before the tokenizer was extended), fixed in `e712502`. The rerun 315425 passed every step: v1 predictions, all four variant types (train + teacher-forced + generation), and the degraded-oracle sweep (p = 1 gives HR@10 0.50 and p = 0 gives 0, on 16 rows).

## Results

*Interim, 2026-10-08. `cf_text` and `residual_text` are still training; the slot runs are still generating.*

### Screen (job 315427; test split, R3 v1 HR@10 = 0.1226)

| source | HR@10 | HR@10 on R3 misses | union with R3 | P(R3 hit given source hit) | train HR@10 (leave-user-out) | valid HR@10 |
|---|---:|---:|---:|---:|---:|---:|
| sasrec | 0.0921 | 0.0197 | 0.1399 | 0.81 | 0.1277 | 0.0989 |
| itemknn | 0.0988 | 0.0256 | 0.1451 | 0.77 | 0.1307 | 0.0982 |
| longknn | 0.0989 | 0.0262 | 0.1456 | 0.77 | 0.1325 | 0.0976 |
| markov | 0.1019 | 0.0236 | 0.1433 | 0.80 | 0.1491 | 0.1085 |
| pop | 0.0694 | 0.0140 | 0.1348 | 0.82 | 0.0886 | 0.0751 |
| trend | 0.0725 | 0.0170 | 0.1375 | 0.79 | 0.0883 | 0.0758 |
| donor | 0.1010 | 0.0250 | 0.1445 | 0.78 | 0.1351 | 0.1076 |
| titlesim | 0.0801 | 0.0251 | 0.1446 | 0.73 | 0.0992 | 0.0858 |
| **fused** | **0.1292** | **0.0446** | **0.1617** | 0.70 | 0.1737 | — |
| residual | 0.0692 | 0.0418 | 0.1593 | **0.47** | 0.0971 | — |

- **Every single source overlaps heavily with R3.**
  - 73–82% of a source's hits are also R3 hits.
  - A source finds only 1.4–2.6% of R3's misses.
  - Time- and long-history signals (`trend`, `longknn`) are no more complementary than plain item-kNN.
- **The fused scorer is the first CF signal that beats R3 alone** (HR@10 0.1292 vs 0.1226). It doubles the coverage of R3's misses (4.5%), and its union with R3 is 0.1617, a ceiling of +3.9 points.
- **The residual scorer does what it was built for.**
  - Only 47% of its hits are R3 hits.
  - Its misses coverage (4.2%) is close to fused at half the overall HR.
- **In-sample SASRec confirms the leak.** The baseline SASRec hits 59.8% of training-row targets, versus 9.2% on test. The leave-user-out and fold sources are at 9–15% on training rows.
- **Calibration bug.** The held-out validation rate used for calibration was computed only on rows whose target was in the pool (0.36). So q = 0, and the training-row tables used by the five runs are **uncalibrated**: training candidates hit 0.174 (fused) and 0.097 (residual), versus 0.129 and 0.069 on test, about 1.35× over-accurate. Fixed in `build_candidates.py` (held-out rows now drawn from all rows; smoke: q ≈ 0.74). The current runs were not rebuilt.
- **Group predictor.** Top-1 cluster accuracy is 13.0% (top-5 25.7%), versus 8.1% for the last item's cluster and 4.4% for the majority cluster.

### Degraded CF-cluster oracle (job 315428; first 2,000 test rows, where v1 has HR@10 0.1575, NDCG@10 0.1248)

| p (true cluster) | cluster accuracy | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 0.007 | 0.035 | 0.022 |
| 0.1 | 0.111 | 0.100 | 0.066 |
| 0.2 | 0.215 | 0.163 | 0.103 |
| 0.3 | 0.294 | 0.215 | 0.139 |
| 0.5 | 0.496 | 0.321 | 0.210 |
| 0.7 | 0.699 | 0.455 | 0.295 |
| 1.0 | 1.0 | 0.627 | 0.408 |
| plug-in (group predictor top-1) | 0.149 | 0.1225 | 0.108 |

The oracle-trained model reaches v1's HR@10 at about 20% cluster accuracy on these rows. The realistic predictor (15%) falls short (0.1225 vs 0.1575). This model trusts the slot fully: a wrong cluster drops HR@10 to 0.035. So 20% is the requirement for a model that fully trusts the slot; a model trained on predicted groups could learn to discount them.

### LLM variants so far

| variant | best epoch (eval loss) | swap ratio (candidates / side info) | swap_r ratio | generation HR@10 (first 1,000; v1 = 0.158) |
|---|---|---:|---:|---|
| `cf_slots` | 2 (0.397) | 0.0020 | 0.0037 | running |
| `residual_slots` | 2 (0.400) | 0.0019 | 0.0020 | running |
| `group_pred` | 2 (0.399) | 0.0254 | 0.0111 | 0.158 (swap: 0.154) |

- **The slot models ignore their candidates.** Replacing the 10 candidate slots with another user's (or with none) moves the output 500× less than dropping one history item. This is despite the fused candidates being more accurate than R3 on test, and over-accurate on training rows (uncalibrated). So in this form the slot channel carries no item identity from weak, partly redundant CF candidates. The contrast is the single-slot oracle (ratio 13.6).
- **`group_pred` matches v1 exactly** (HR@10 0.158 vs 0.158 on the same rows), with a small swap effect (ratio 0.025, about the level of category-align).
### Final results (2026-10-08 23:45; all jobs complete)

Full test (8,313 rows; v1 HR@10 0.1226, NDCG@10 0.0950):

| variant | HR@10 | NDCG@10 | HR@10 on v1 misses | HR@10 on v1 hits | swap-candidates ratio | generation HR@10 without candidates (first 1,000; with = col. 2 on these rows) |
|---|---:|---:|---:|---:|---:|---|
| `cf_text` | 0.1241 | 0.0964 | 0.0206 | 0.866 | **4.93** | 0.063 (with: 0.156) |
| `residual_text` | 0.1222 | 0.0945 | 0.0175 | 0.871 | 0.31 | 0.127 (with: 0.157) |
| `cf_slots` | 0.1226 | 0.0950 | 0.0148 | 0.894 | 0.002 | 0.162 (with: 0.159) |
| `residual_slots` | 0.1255 | 0.0959 | 0.0166 | 0.905 | 0.002 | 0.156 (with: 0.161) |
| `group_pred` | 0.1211 | 0.0940 | — | — | 0.025 | — |
| *no training:* reciprocal-rank fusion of v1 and fused-CF top-10 lists (`rank_fusion.py`) | **0.1346** | — | — | — | — | — |

- **Delivering CF to the decoder does not improve ranking in any form.**
  - `cf_text` reads its candidates heavily: a swap moves the output 4.9× as much as dropping a history item, and without candidates HR@10 falls to 0.063. Yet it gains 150 v1 misses and loses 137 v1 hits (exact McNemar p = 0.48).
  - The decoder **substitutes** candidate evidence for title evidence, the same pattern as `sasrec_raw` one level up.
  - Its training candidates were uncalibrated (about 1.35× over-accurate; see the calibration bug), which plausibly pushed it toward this over-reliance.
- **Slots and the group vector are ignored** (swap ratios 0.002 and 0.025). Text carries the same candidates about 2,500× more strongly, so the slot design is the bottleneck for this weak signal, unlike the single-slot oracle.
- **Combining outputs works where combining inputs does not.** Untuned reciprocal-rank fusion of v1's top-10 with the fused-CF top-10:
  - reaches HR@10 **0.1346** (+1.2 points, +9.8% relative);
  - gains 189 rows and loses 89 versus v1 (exact McNemar p = 2×10⁻⁹);
  - beats both inputs (fused alone 0.1292) and is insensitive to the RRF constant (k = 10/30/120).

  The fused scorer was fit on the validation split only, so this is a clean test number. The union ceiling is 0.1617.
- **Implication.**
  - The complementary CF knowledge exists (screen) but gets traded off against title evidence when it enters the LLM's input.
  - Next levers: (1) rerun `cf_text` with the fixed calibration (less over-trust); (2) learn the fusion at the score level (e.g. tune RRF weights on validation predictions); (3) train with a loss that only rewards the candidate path on R3 misses, so the decoder keeps its title evidence.

**Update 2026-10-08 15:20.** Full-test ranking (8,313 rows; v1 HR@10 0.1226, NDCG@10 0.0950):

| variant | HR@10 | NDCG@10 | HR@10 on v1 misses | union with v1 | swap-candidates ratio |
|---|---:|---:|---:|---:|---:|
| `cf_slots` | 0.1226 | 0.0950 | 0.0148 | 0.1356 | 0.002 |
| `residual_slots` | 0.1255 | 0.0959 | 0.0166 | 0.1371 | 0.002 |
| `group_pred` | 0.1211 | 0.0940 | — | — | 0.025 |
| `residual_text` | eval running | | | | **0.306** (no candidates: 0.355) |
| `cf_text` | training (epoch 3 eval) | | | | |

- The slot models are statistically indistinguishable from v1: the +0.3 points of `residual_slots` comes with a decoder that ignores the slots.
- **Text candidates are read.** In `residual_text`, swapping the candidate titles moves the output 0.31× as much as dropping a history item, about 150× the slot models. So the slot design, not the information, is what fails to transmit candidates. Whether reading them improves ranking awaits its full ranking.

- **`cf_text` / `residual_text` decide the question.** If titles in text also leave the output unchanged, the CF candidates carry nothing the decoder can use. If text helps while slots do not, the slot design is the bottleneck.

## Follow-up: cluster as a latent variable (2026-10-09)

**Why.** Cluster correctness and item correctness go together for R3:
- R3 hits on 71% of the rows where its top item is in the target's cluster.
- On R3's misses, its implied cluster is right only 3.8% of the time.
- The fused group predictor is accurate where R3 is already right (top-1: 66.7% on R3 hits, 5.5% on R3 misses).
- New cluster information exists on only about 1% of rows.

The true cluster is in the predictor's top-5 on 17.8% of R3's misses, so instead of one cluster we keep several as parallel latent hypotheses.

### Idea 1: predict the cluster from the history's cluster ids (`src/collab_reasoning/cluster_seq_predictor.py`)

| predictor (test, k = 150) | top-1 | top-5 | top-1 on R3 hits | top-1 on R3 misses | top-5 on R3 misses |
|---|---:|---:|---:|---:|---:|
| cluster Markov | 0.1057 | 0.2321 | 0.5054 | 0.0492 | 0.1571 |
| GRU over 10 history clusters | 0.1183 | 0.2408 | 0.5937 | 0.0512 | 0.1648 |
| GRU over the full past's clusters | 0.1172 | 0.2394 | 0.5888 | 0.0506 | 0.1640 |
| fused item-level predictor | 0.1305 | 0.2569 | 0.6673 | 0.0546 | 0.1776 |
| mix (GRU-full + fused, averaged) | **0.1341** | **0.2608** | 0.6712 | **0.0582** | **0.1811** |
| R3's own top-1 item's cluster | 0.1160 | 0.1429 | 0.6683 | 0.0380 | 0.0681 |

Reading the history at cluster granularity does not decorrelate the predictor from R3. A separate model over cluster ids fails on the same rows (5.1% top-1 on R3 misses). The long history adds nothing. Averaging it with the item-level predictor gains half a point.

### Idea 2: parallel cluster paths with marginalization

```
p(item | x) = (1 - λ) · Σ_{c ∈ top-K} w_c · p(item | x, c)  +  λ · p(item | x, R3)
w_c ∝ p(c | x)^α,    p(item | x, path) = softmax(beam log-probs / T)
```

- **p(item | x, c)**: the `oracle_cfcluster150` model with hypothesis cluster c in its slot. It was trained to trust the slot, which is what conditioning on c means.
- **Paths.** For each of 3,000 test rows we decode the plain R3 path plus every cluster in the union of the top-5 of the fused, GRU-full and mix predictors (`src/prototype_grounding/cluster_paths_decode.py`).
- **Aggregation** (`src/collab_reasoning/cluster_paths_aggregate.py`). We grid-search predictor, K ∈ {1, 2, 3, 5}, α, λ and T on test rows 0–999 and report rows 1,000–2,999. Validation rows are not used for tuning because the fused predictor was trained on them.
- **Reported.** The R3 path alone, the top-1 plug-in, the best marginalization per predictor, marginalization without the R3 path, and two upper bounds: the best single cluster path, and any decoded path containing the target.

Jobs: smoke 317368 → full 317369 (`scripts/collab_cluster_paths.sbatch`). Results go to `analysis/results/collab_reasoning/cluster_paths/full/aggregate.md`.

### Results: parallel cluster paths (job 317369; tuned on test rows 0–999, reported on rows 1,000–2,999)

On average 8.0 cluster paths are decoded per row. The true cluster is among them on 32.7% of rows (23.2% of R3's misses), and some path contains the target on 24.4%.

| method | HR@10 | NDCG@10 | HR@10 on R3 misses | gained / lost vs R3 |
|---|---:|---:|---:|---:|
| R3 path | 0.1450 | 0.1138 | 0 | — |
| top-1 cluster plug-in (mix) | 0.1190 | 0.1006 | 0.022 | 38 / 90 |
| marginalization, selected on the tune rows (fused, K=5, λ=0.5) | 0.1445 | 0.1152 | 0.000 | 0 / 1 |
| marginalization without the R3 path (mix, K=5) | 0.1275 | 0.1048 | 0.023 | 40 / 75 |
| + decoder confidence in the path weights (best on the tune rows) | 0.1445 | — | — | 0 / 1 |
| **upper bound: best cluster path, chosen with the target known** | **0.2275** | 0.1479 | **0.116** | 198 / 33 |

- **The paths contain the answer, but no rule picks them.**
  - The best-path bound is +8.3 points over R3: on 11.6% of R3's misses, the target is in the right path's top-10.
  - Every tuned marginalization either collapses back to R3 (λ = 0.5 makes R3's peaked beam distribution dominate) or loses more R3 hits than it gains.
- **Verifiers: accuracy at picking the true-cluster path among the decoded ones** (`src/collab_reasoning/cluster_paths_verifiers.py`; R3-miss rows where it was decoded, n = 396, chance 0.129):

  | verifier | accuracy on R3 misses | accuracy on all rows |
  |---|---:|---:|
  | p(c \| x), mix predictor | **0.227** | 0.436 |
  | decoder top-beam log-prob (self-verification) | 0.073 (below chance) | 0.294 |
  | mean top-3 log-prob | 0.098 | 0.093 |
  | overlap with R3's top-10 | 0.179 | 0.234 |
  | overlap with the fused CF top-10 | 0.169 | 0.175 |
  | CF reciprocal-rank mass (4 sources, top-100) | 0.152 | 0.259 |
  | p(c \| x) × (1 + CF mass) | 0.199 | 0.418 |

  Self-verification is anti-informative on R3's misses: the decoder is most confident under the hypothesis R3 already believes. Independent CF evidence is only marginally above chance. The cluster predictor itself remains the best selector.
- **Conclusion.**
  - Parallel latent hypotheses turn "predict the cluster among 150" into "pick among about 8 decoded paths". The candidate set holds the answer often enough to be worth +8 points.
  - But picking is the same problem again: none of the available signals separates the right path on the rows R3 gets wrong.
  - The bottleneck is a verifier with evidence the predictor and the decoder do not already share. A learned verifier trained on decoded validation paths, with the features above plus path-item CF scores, is the remaining test; its target is better than 22.7% selection accuracy on R3 misses.

## Follow-up: trained evidence verifier (VRec-style, 2026-10-09)

**Reframing.** Instead of generating a reasoning latent and trusting it, the model retrieves evidence from a fixed corpus and a *trained* verifier decides how much of it enters the thought slot (after VRec, arXiv 2603.07725: a verifier over group-level classes, entropy as confidence, prototype guidance, trained jointly with the recommender).

**Why a trained verifier.** Statistical confidence is not enough. On the full test split (8,313 rows; 7,298 R3 misses) the existing cluster predictors are well calibrated (`mix`: 88% top-1 accuracy at max-prob ≥ 0.5) but confident almost only on rows R3 already gets right. On R3 misses, only 74 rows are confident (≥ 0.3) and right (138 at ≥ 0.2), worth at most about +0.5 pt. `group_pred` (the predicted distribution in the slot, from a fixed predictor) behaved accordingly: swap ratio 0.025, HR@10 0.1211.

**Model** (`src/latent/verified_evidence_latent.py`). Evidence corpus: the 150 SASRec cluster centroids, read through the side projection of `oracle_cfcluster150`, whose decoder already reads them (swap ratio ≈ 13).

```
p  = softmax(verifier(RMSnorm(r)))                 verifier: MLP 1536 → 512 → 150
c  = 1 − H(p) / log 150                            entropy gate
r' = r + 30 · c · Σ_j p_j · normalize(P(e_j))
L  = L_rec + w · CE(p, cluster(target))            label smoothing 0.1, w = 0.5
```

- Confident verifier → one evidence vector in the slot, exactly what the oracle decoder was trained on. Uncertain → a short blend the decoder learns to ignore.
- End to end: the recommendation loss reaches the verifier through p and c, so it learns where its evidence helps the decoder, not only which cluster the target is in.
- Curriculum: rows get the true cluster as one-hot p with probability 0.5, decaying linearly to 0 over the first epoch, so the decoder keeps reading the slot while the verifier is untrained.
- Starts from the `oracle_cfcluster150` checkpoint; the verifier is freshly initialised.

**Evaluation** (`src/prototype_grounding/evidence_eval.py`):
1. verifier top-1/top-5, confidence, and the selective table on all rows and on R3 misses (the number of misses where it is confident and right bounds the gain);
2. teacher-forced KL ratios: swap_evidence, zero_evidence, oracle_evidence, swap_r vs the one-item history drop;
3. full-test ranking vs R3 v1 on the same rows (gained/lost, exact McNemar), plus a zero-evidence ablation on the first 2,000 rows.

**Jobs.** `scripts/collab_evidence.sbatch [full|smoke] [weight] [gate]`: smoke 318241, full 318242 (after the smoke). Results: `analysis/results/collab_reasoning/verified_evidence/<mode>_evidence_w0.5_entropy/`.

**Risk.** The verifier is trained on the same rows as the backbone, so it can be more confident on training rows than on test rows and teach the decoder to over-trust it. Early stopping on validation loss is the only guard; the selective table on test shows whether it happened.

### Results: trained evidence verifier (job 318248; smoke 318247 after the loss-scaling fix)

Bug found in the first smoke (318241): under gradient accumulation the HF Trainer (4.48) passes `num_items_in_batch` and sums micro-batch LM losses, so a per-micro-batch auxiliary mean is counted 64× (train loss 168 vs ~5). Fixed by weighting it by the micro-batch's share of label tokens (11a676a). `CategoryAlignedLatentModel` adds its alignment loss the same way, so the earlier category-alignment run likely trained with that loss ~64× heavier than intended.

Validation loss 2.831 → 2.708 → 2.650 (epochs 1–3; includes 0.5 × verifier CE).

| verifier (test, 8,313 rows) | trained verifier | `mix` predictor (statistical) |
|---|---:|---:|
| top-1 cluster, all rows | 12.0% | 13.3% |
| top-1 on R3 hits / misses | 58.1% / **5.5%** | — / 5.8% |
| R3 misses confident (c ≥ 0.3) and right | **60** | 74 (max-prob ≥ 0.3) |
| accuracy at c ≥ 0.5 | 91% | 88% |

Training-row top-1 reached 15–20% in epoch 3 vs 12% on test (mild overfitting).

Teacher-forced KL ratios (to the one-item history drop): oracle evidence **0.137** (≈ 13 for the starting oracle model, 4.06 after the 2-step smoke), swap evidence 0.006, zero evidence 0.0045, swap base r 0.036. **The decoder unlearned the slot**: after the curriculum ended, the evidence was nearly empty on most rows (mean c = 0.14), and the trainable decoder learned to ignore it.

| ranking (full test) | HR@1 | HR@5 | HR@10 | NDCG@10 | gained / lost vs R3 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| R3 v1 | 0.0747 | 0.1020 | 0.1221 | 0.0950 | — | — |
| trained evidence verifier | 0.0670 | 0.0943 | **0.1160** | 0.0876 | 159 / 210 | 0.009 |

- On rows with c ≥ 0.3: 0.605 vs R3 0.616.
- Zero-evidence ablation (rows 0–1,999): 0.1470 vs 0.1475 with evidence (R3 0.1575). The evidence contributes nothing; the drop vs R3 comes from the extra fine-tuning.

**Reading.** The verifier is calibrated but, like every predictor so far, is confident almost only where R3 is already right; its uncertainty marks the rows that are hard for the recommender too. Gating by confidence cannot recover misses, and an evidence channel that is mostly empty is unlearned by a trainable decoder.

**Group granularity** (`mix` predictor aggregated to coarser spherical k-means groups of the 150 centroids, R3 misses = 7,298):

| groups | verifier right on misses | R3 top-1 in right group | verifier right, R3 wrong | …and max-prob ≥ 0.5 |
|---:|---:|---:|---:|---:|
| 5 | 45.8% | 39.0% | 1,082 | 236 |
| 10 | 30.5% | 24.9% | 985 | 56 |
| 20 | 20.0% | 16.1% | 657 | 14 |
| 50 | 10.3% | 6.9% | 477 | 7 |
| 150 | 5.8% | 3.8% | 292 | 2 |

## Follow-up: VRec-style coarse groups (2026-10-10)

`src/latent/vrec_group_latent.py`, `scripts/collab_vrec.sbatch [full|smoke] [k] [weight]`: k = 20 CF groups (`build_group_table.py`; sizes 105–390, test-target entropy 4.25 bits); linear verifier whose weight rows are the prototypes; `r* = (1 − c)·r + c·‖r‖·normalize(Σ p_j W_j)` keeps R3's own reasoning when uncertain; starts from R3 v1; L = L_rec + 0.5·CE. Not included: VRec's stage-1 pretraining on the recommender's successes/failures (needs R3 predictions on training rows), the multi-step monotonicity loss, extra verifier views. Smoke 318921 (train loss 2.08, eval 2.20), full 318922.

### Results: VRec-style coarse groups (job 318922)

Validation loss 1.772 → **1.723** → 1.882 (epochs 1–3; best checkpoint = epoch 2 is evaluated). Training-row verifier top-1 rose to 45–50% in epoch 3 vs 25% on test: the verifier memorized training rows, and validation loss turned up.

| verifier (test, k = 20, chance 5%) | value |
|---|---:|
| top-1 / top-5 group | 25.1% / 56.9% |
| top-1 on R3 hits / misses | 75.9% / **18.0%** (R3's own top-1 group on misses: 16.1%) |
| mean confidence c (right / wrong) | 0.35 / 0.17 |
| R3 misses with c ≥ 0.3 and right | 321 (34% accuracy among confident misses) |

Teacher-forced KL ratios: oracle group (r moved fully to the true prototype) **0.33**, swap verifier output 0.0046, zero nudge 0.0032, swap base r 0.0013. Joint training made the decoder respond to r more than R3 v1 does (v1 swap ratio 0.006; smoke 0.07), but even the true group moves the output only a third as much as dropping one history item, and the realistic nudge (mean c 0.21) barely moves it.

| ranking (full test) | HR@1 | HR@5 | HR@10 | NDCG@10 | gained / lost vs R3 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| R3 v1 | 0.0747 | 0.1020 | 0.1221 | 0.0950 | — | — |
| VRec-style, k = 20 | 0.0725 | 0.1033 | **0.1227** | 0.0944 | 144 / 139 | 0.81 |

Zero-nudge ablation (rows 0–1,999): 0.1575 without the nudge vs 0.1540 with it (R3 0.1575).

**Reading.** Coarse groups make the verifier right far more often (321 confident-and-right misses vs 60 at k = 150), and keeping r avoids the run-1 collapse, so the model no longer loses to R3. But the result is R3: a correct 250-item group rarely becomes a hit (R3's own top-1 is already in the target group on 16% of its misses), and the decoder responds only weakly to r even when it points at the true group. Consistent with the prediction made before the run.
