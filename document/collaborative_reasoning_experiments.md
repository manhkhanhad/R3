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

*Pending (jobs submitted 2026-10-07).*
