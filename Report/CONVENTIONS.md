# Conventions for the Report

Back to the [main plan](main.md). Every experiment report follows these rules, so results can be copied into the paper
without re-checking terminology.

## Terminology

| term | meaning |
|---|---|
| `r` / latent / thought | the latent reasoning vector R3 computes by attention-pooling the LLM's hidden states over the history; spliced into the prompt as the `<\|Thought\|>` token |
| R3 / LatentR³ | the base method (`src/latent/layers.py::LatentModel`); "R3_ori" = the original authors' code and checkpoints; "plain R3 (A)" = our Toys retrain in this codebase; R3 371447 (Steam), 371446 (MovieLens) = our runs on the new datasets |
| no-latent SFT | the same backbone and prompt with the thought slot removed, trained the same way (`no_latent_ref` jobs) |
| latent test / intervention | decode the test set with `r` replaced (`src/prototype_grounding/experiment.py generate --mode`): **mask** (`r` = 0), **mean** (train-mean `r`), **shuffle_user** (the `r` of a training row of another user). "Content used" = mean AND shuffle significantly hurt; "slot used" = only mask hurts |
| hinge / margin loss | history-swap hinge: relu(m + NLL(y \| x, r_own) − NLL(y \| x, r_other)), r_other = another in-batch user's latent, m = 0.2. margin_loss 0.2 = no gap, 0 = satisfied |
| WIDER | cross-user diversity penalty on `r` (`src/latent/wider_reg.py`): mean cos² between each normalised latent and an EMA shared direction; 1.0 = all latents identical (collapse) |
| MW | margin (hinge) + WIDER on R3 (Toys 356322, CDs 362561) |
| copy_row | Toys/CDs (`motive_contrastive.py::copy_row`): the target's day equals the **last** history timestamp, and some history title shares the target's brand (first word) with title-token Jaccard ≥ 0.3. Test: Toys 905, CDs 418. The 2026-11-11 `analysis/results/copy_rows_mw_vs_r3/compare.py` variant (any same-day history item) gives Toys 836 / CDs 368 and is used in the GRPO analyses; always say which. MovieLens: copy_row = lowest quarter of the history-only reference NLL (2,541 rows) |
| reference model | history-only (no-latent) LM; its NLL of the target given the history says how much the history alone explains the row |
| MW-R | hinge weighted per row by a reference NLL (base LM) |
| **MW-R-H** | **contribution 1**: hinge weighted per row by the history-only reference, rank weights w_i = rank(NLL_ref,i)/(N−1), sum(w·hinge)/sum(w); margin 0.2, WIDER 0.1, 3 epochs (last kept), base Qwen2.5-1.5B, lr 3e-4 (`METHOD=motive MOTIVE_W=0 MARGIN_MODE=refweight REF_WEIGHT_FN=rank`; `REF_TAU=0.5` is passed but only the sigmoid weight uses it) |
| MW-nc | hinge switched off on copy_rows (rule-based ceiling, development reference only) |
| MW-R-H-S / -D / -L | MW-R-H + swapped-pass CE / + KL to the no-latent reference / + reduced late hinge |
| MWC-* | MW + a collaborative term: **I** = original collaborative-reasoning InfoNCE (user + target heads, 10k queue), **I-B** target head + same-user hard negatives, **I-C** + item head, **G** donor grounding, **G3** CF-prior donor evidence, **M** same-motive InfoNCE, **M2** SASRec-neighbour positives |
| same-motive InfoNCE | positives = other users with the same next item + the same user's rows with a similar item (title cos ≥ 0.8); negatives from a bank of all train-row latents |
| MW-R-H-CR | MW-R-H + the original collaborative-reasoning InfoNCE (same as MWC-I) |
| MW-R-H + motive | MW-R-H + same-motive InfoNCE (`MOTIVE_W=0.1`) |

## Datasets in scope and per-dataset hyperparameters (author, 2026-11-27)
- **Development focuses on CDs, Toys and Steam.** MovieLens is a substitute dataset: keep its results, but run new work
  there only on request. Musical is development-only. Goodreads, Yelp and MobileRec are dropped.
- **Each dataset keeps its best MW-R-H hyperparameters, and every later model on that dataset uses the same values**
  (C2 variants, ablations, seeds). Example: any new model on Steam trains at lr 5e-5.

| dataset | lr | grad clip | margin | WIDER | hinge weights | epochs (kept) | batch / micro | base model | reference NLL | evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| CDs_and_Vinyl | **3e-4** | 1.0 | 0.2 | 0.1 | rank | 3 (last) | 128 / 4 | Qwen2.5-1.5B | `reference_nll/CDs_and_Vinyl_nolatent/train.pt` (365885) | 3 / 3 runs pass (366023, 368187, 368188) |
| Toys_and_Games | **3e-4** | 1.0 | 0.2 | 0.1 | rank | 3 (last) | 128 / 4 | Qwen2.5-1.5B | `reference_nll/Toys_and_Games_nolatent/train.pt` (365404) | 3 / 5 runs pass (365504, 365887, 368189); MW-R-H-L (late hinge 0.3) 2 / 2 |
| Steam | **5e-5** | 1.0 | 0.2 | 0.1 | rank | 3 (last) | 128 / 4 | Qwen2.5-1.5B | `reference_nll/Steam_nolatent/train.pt` (369661) | 1 / 1 (383067); replicate 385083 (`_lr5e-5_rep2`) submitted 2026-11-27. lr 3e-4: 1 / 2, lr 1e-4: 0 / 1 |
| MovieLens (substitute) | 3e-4 | 1.0 | 0.2 | 0.1 | rank | 3 (last) | 128 / 4 | Qwen2.5-1.5B | `reference_nll/Movielens_nolatent/train.pt` (369531) | plain MW-R-H 0 / 1; passes only with D (`DISTILL_W=1`, 375422) or S (375420, dropped) |

- Reference files live under `R3/output_dir/margin_wider_collab/`. In the sbatch, lr is set with `LR=` and clip with
  `MAX_GRAD_NORM=`. Code from 66b61b81 on accepts both.
- **Runs made before this rule at a non-best setting are not valid tests of that model on that dataset.** This applies to
  Steam MW-R-H-CR 384089 (lr 3e-4) and Steam MW-R-H 369662 / 383065 / 383066. Re-test such models at the table's values
  before drawing conclusions.

## Reporting rules
- Metrics as values (0.1086), never %. Report **HR and NDCG at @1, @3, @5, @10** (HR@1 = NDCG@1; @20 = @10 with 10 beams). Design goals focus on @5 and @10.
- Every comparison: paired t-test p AND paired bootstrap (2,000 resamples, seed 42; CI and/or p) over ALL test rows, written "diff (p_t / p_b)" or "diff [CI]". Mark significant with bold or `*`.
- Every model is named with its run ID, e.g. "MW-R-H (365504)".
- Success criterion of the line: latent **content** matters (mean and shuffle significantly hurt) AND ranking is at least plain R3 (≈ R3 is acceptable = not significantly below R3 at @5/@10 by both tests).
- **Seed rule (author, 2026-11-27):** a configuration is a success if the criterion holds on its single run, or, when several seeds were run, on **at least 2** of them. Reruns at identical settings count as seeds. Tables still list every seed, including failing ones.
- Numbers are copied from the research log / result files, never recomputed from memory. If a number was not recorded, write "not recorded".
- Test sizes: Toys n = 6,738 (copy_row 905), CDs n = 6,157 (copy_row 418), Steam n = 10,303, MovieLens n = 10,164 (copy_row 2,541), Musical n = 8,313.
- Sign of latent tests: say which convention a table uses. "raw − intervention" (positive = replacing `r` hurts) or "intervention − raw" (negative = hurts); reports E00, E01, E02 and E06 use the latter; E03, E04, E05, E07, E08 and E09 the former.

## Report template (`experiments/<NN_name>/report_<topic>.md`)
```
# E<NN> — <title>
Back to [main plan](../../main.md). Dates: <first–last>. Status: done / partial / dev only.

## Question
## Setup        (method, code commit / frozen code, scripts, datasets, run IDs)
## Results      (tables at all cutoffs; latent tests; significance)
## Reading      (what the result means; caveats: single seed, nondeterminism, GPU)
## Paper use    (main result / ablation / appendix / motivation / dev only)
## Sources      (research-log file + line ranges or section titles, result-file paths)
```
