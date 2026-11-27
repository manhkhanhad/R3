# Collaborative latent reasoning for LLM recommenders: research story and plan

*Make the latent thought count, then ground it in other users' reasoning.*
Status: research plan and evidence map, not a paper draft. Created 2026-11-27.
- All results live in [experiments/](experiments/); this file only links to them.
- Terminology and reporting rules: [CONVENTIONS.md](CONVENTIONS.md). Context for a new session: [CONTEXT.md](CONTEXT.md).

---

## Abstract (working)
**Latent reasoning.** Latent reasoning lets an LLM recommender "think" in a continuous vector before it names the next
item. LatentR³ (R3) pools the LLM's hidden states over the history into one latent thought and decodes the next item
from the history plus that thought.

**The two problems.** The thought is computed from a single sample, so nothing explains or checks it. Worse, the decoder
does not use it:
- replacing a user's thought with the mean thought, or with another user's, leaves the ranking unchanged;
- R3 ranks like the same model with no thought slot ([E01](experiments/01_r3_ignores_latent/report_r3_ignores_latent.md)).

**Contribution 2 needs contribution 1.** Our starting observation is that people often buy an item for the same reason.
This suggests **collaborative reasoning**: tie each thought to the thoughts of other users through a contrastive
objective, as collaborative filtering ties users through shared items. Applied to R3 directly, collaborative objectives
reshape the latent space but cannot change a prediction, because the decoder never reads the thought
([E02](experiments/02_collab_reasoning_on_r3/report_collab_on_r3.md)). So:
- **C1, MW-R-H.** A training objective that makes the decoder depend on the user's own thought without hiding the
  history. It is a history-swap hinge: the user's own thought must explain the next item better than another user's
  thought. The hinge is weighted per row by a history-only reference model, so rows the history already explains are
  not forced through the thought. A diversity regulariser (WIDER) is added.
- **C2, collaborative reasoning on top of MW-R-H.** Thoughts of users who buy the same item for similar reasons are pulled
  together. Groups of thoughts that lead to good recommendations become anchors that make a thought interpretable and
  checkable.

**Results (single default seed unless noted):**
- **CDs:** MW-R-H is significantly above R3 at every cutoff, over three seeds, and swapping thoughts costs accuracy. Adding
  same-motive collaborative reasoning doubles that dependence at equal accuracy.
- **Toys and Steam:** MW-R-H is comparable to or above R3 with the thought used.
- **MovieLens:** open, see §4.

## 1. Introduction (story order)
1. **Context.**
   - LLM sequential recommenders generate the next item's title from a textual history.
   - Explicit chain-of-thought is slow and noisy.
   - Latent reasoning writes one continuous thought instead: R3 builds it by attention pooling and refines it with RL.
2. **Problem 1: unverifiable, per-sample reasoning.** R3's thought is a function of one user's history only. We cannot
   read it, and nothing relates it to the reasoning of other users who made the same decision.
3. **Observation and idea.**
   - Purchases share reasons: many users buy a given item for the same motive.
   - **Collaborative reasoning** grounds each thought in other users' thoughts with InfoNCE. The latent space then
     groups users by shared reasons, much as collaborative filtering groups them by shared items.
   - A group of thoughts that leads to correct recommendations acts as an anchor that explains and checks the thoughts
     near it.
4. **Obstacle (pilot).** Before collaborative reasoning can work, the thought must matter. In R3 it does not
   ([E01](experiments/01_r3_ignores_latent/report_r3_ignores_latent.md)):
   - Mean and shuffle interventions leave ranking unchanged on every dataset.
   - R3 equals the no-thought model.
   - The decoder responds 140–590× more to dropping one history item than to swapping the thought.
   - The cause is redundancy: `r = f(history)` carries nothing the decoder cannot read from the visible history.
     - When the slot carries information the history lacks (an oracle category), the decoder uses it.
   - Collaborative InfoNCE on plain R3 improves the geometry of the latent space but not a single prediction
     ([E02](experiments/02_collab_reasoning_on_r3/report_collab_on_r3.md)).
5. **Why simple fixes fail** ([E03](experiments/03_routes_to_latent_use/report_routes.md)):
   - Bottlenecks that hide the history buy dependence but lose 11–25% HR@10, all of it on rows where the history
     already contains the answer.
   - Reshaping or enriching the thought keeps accuracy but leaves the thought unread.
6. **C1, MW-R-H** ([E04](experiments/04_margin_wider/report_margin_wider.md),
   [E05](experiments/05_mwrh_design/report_mwrh_design.md)):
   - The objective acts on the **effect** of the thought: with the user's own thought, the target must be likelier
     than with another user's.
   - A plain hinge over-applies on "copy" rows, where the target is a same-day variant of a history item, and costs
     accuracy there.
   - A **history-only reference model** gives each row a weight (how much the history alone explains it), so the hinge
     acts where reasoning is needed.
7. **C2, collaborative reasoning** ([E07](experiments/07_collab_on_mw/report_collab_on_mw.md),
   [E08](experiments/08_collab_on_mwrh/report_collab_on_mwrh.md)):
   - **Positives:** the same next item across users, plus the same user only for similar items.
   - With these positives the thought does not collapse into a user identifier, which is the failure of the original
     user/target objective.
8. **Contributions.**
   - (i) Diagnosis: the latent thought of a latent-reasoning recommender is ignored, and the cause is redundancy.
   - (ii) MW-R-H, a reference-weighted counterfactual hinge that makes the thought's content used at no accuracy cost.
   - (iii) Collaborative reasoning, which grounds thoughts in shared purchase reasons.
   - (iv) Experiments on four datasets, with intervention tests showing the thought is used.

## 2. Method

### 2.0 Backbone and latent slot
- **Input:** the history is written into R3's fixed prompt (never changed), followed by a `<|Thought|>` token.
- **Thought:** `r = Pool(h_1 … h_n)` attention-pools the LLM's hidden states over the history. `r` replaces the thought
  token's embedding.
- **Decoding:** the decoder generates the next title with constrained beam search.
- **Training loss:** next-token CE on the target title, L_CE = NLL(y | x, r).
- Code: `src/latent/layers.py`.

### 2.1 C1: MW-R-H
Exact formulas and the derivation are in [E05](experiments/05_mwrh_design/report_mwrh_design.md).
- **History-swap hinge:** h_i = relu(m + NLL(y_i | x_i, r_i) − NLL(y_i | x_i, sg r_π(i))).
  - r_π(i) is another user's thought in the micro-batch (a roll by one), put on the same history x_i.
  - m = 0.2. NLL is the per-token mean over the target title.
  - The negative thought is detached, but the decoder gets gradient through both passes (two-sided).
- **Reference weight:**
  - A history-only LM (the same R3 prompt without the thought token; no-latent SFT, trained the same way) gives
    NLL_ref,i for every training row (`reference_score.py`).
  - w_i = rank(NLL_ref,i) / (N − 1) ∈ [0, 1] over the N training rows. Rows the history already explains (low NLL_ref)
    get a small w_i: Toys copy rows 0.181 vs non-copy 0.569.
  - The hinge loss is Σ w_i h_i / max(Σ w_i, 1e-6).
- **WIDER:** mean (z_i · q)², where z_i = r_i / ‖r_i‖ and q is the normalised EMA (momentum 0.99) of the batch-mean z.
  Weight 0.1. It stops all thoughts converging to one vector.
- **Total loss:** L = mean NLL(y | x, r) + 1.0 · L_hinge + 0.1 · L_WIDER.
- **Training:** 3 epochs (last kept), lr 3e-4, batch 128 (micro-batch 4), base Qwen2.5-1.5B. Inference is plain R3.
- **Ablations along the way** (E04, E05):
  - plain hinge (MW);
  - epoch-1-only hinge (MW-3b);
  - rule-based no hinge on copy rows (MW-nc, ceiling);
  - base-LM reference (MW-R);
  - band, rank², Lagrangian and self-weighted hinges.

### 2.2 C2: collaborative reasoning
- **Projection:** the thought is projected into a contrastive space.
- **Positives:**
  - other users' training rows with the same next item;
  - the same user's rows whose items are similar (title cosine ≥ 0.8).
- **Negatives:** all other rows in a bank of every training row's thought.
- **Schedule:** multi-positive InfoNCE (τ 0.1, weight 0.1) from epoch 2, after the hinge has made the thought matter.
- **Original variant (Attempt 10, MW-R-H-CR):** a user head (all of the user's other rows are positives) and a target
  head (same next item), with a 10k queue. It is the ablation that shows why the positive definition matters: the user
  head turns the thought into a user ID and competes with the hinge.

## 3. Experiments

**Datasets (author, 2026-11-27):**
- **In scope:** Amazon Toys & Games, CDs & Vinyl, and Steam:
  - Toys & Games: n = 6,738;
  - CDs & Vinyl: n = 6,157;
  - Steam: n = 10,303.
- **Substitute:** MovieLens (n = 10,164). Results are kept, but no new development runs there.
- **Development only:** Musical Instruments.
- **Hyperparameters:** each dataset keeps its best MW-R-H hyperparameters, and every later model on it uses the same values
  ([CONVENTIONS.md](CONVENTIONS.md#datasets-in-scope-and-per-dataset-hyperparameters-author-2026-11-27)).
  - Toys and CDs: lr 3e-4.
  - Steam: lr 5e-5.
  - Everything else is shared: margin 0.2, WIDER 0.1, rank weights, 3 epochs, batch 128, clip 1.0.
- **Success rule:** a configuration succeeds if the criterion holds on its single run, or on at least 2 seeds when several
  were run.

**Baselines:**
- popularity, item-CF, SASRec;
- no-thought SFT;
- R3 (the original code on CDs, our retrain elsewhere).

**Metrics:** HR/NDCG @1/3/5/10, with paired t-test and bootstrap. **Latent tests:** mask / mean / shuffle_user.

| ID | Question | Status | Report |
|---|---|---|---|
| E00 | Datasets and baselines | done; SASRec missing on 3 datasets | [datasets + baselines](experiments/00_datasets_baselines/report_datasets_baselines.md) |
| E01 | Does R3 use its latent? (motivation) | done: mask hurts, mean and shuffle n.s. on Musical / Toys / CDs / Steam (MovieLens 2% effect); R3 = no-latent on Toys / Steam / MovieLens; swapping the thought moves the output 140–590× less than dropping one item | [R3 ignores the latent](experiments/01_r3_ignores_latent/report_r3_ignores_latent.md) |
| E02 | Does collaborative reasoning on plain R3 change predictions? | done: CDs A10 effective rank 21 → 111 but mean / shuffle n.s. | [collaborative on R3](experiments/02_collab_reasoning_on_r3/report_collab_on_r3.md) |
| E03 | Simpler routes to latent use (related / ablation) | done: bottlenecks lose 11–25%, everything else leaves `r` unused | [routes](experiments/03_routes_to_latent_use/report_routes.md) |
| E04 | Margin + WIDER, and the copy-row problem | done: MW Toys shuffle −.073 at ≈ R3; copy rows −.034 sig, non-copy +.0044 sig (pooled) | [MW](experiments/04_margin_wider/report_margin_wider.md) |
| E05 | From MW to MW-R-H (C1 design and ablations) | done: 13 variants; history-only reference best (Toys 365504 .1086 / .0698) | [MW-R-H design](experiments/05_mwrh_design/report_mwrh_design.md) |
| E06 | MW-R-H results on 4 datasets, seeds, Steam stability, MovieLens | done: CDs 3 seeds sig > R3; Toys seed-dependent; Steam lr 5e-5; MovieLens fails (S / D pass) | [MW-R-H results](experiments/06_mwrh_results/report_mwrh_results.md) |
| E07 | Collaborative terms on the MW base (C2 development) | done: same-motive MWC-M = R3, shuffle −.084; original CR −.0086 sig | [collaborative on MW](experiments/07_collab_on_mw/report_collab_on_mw.md) |
| E08 | Collaborative reasoning on MW-R-H (C2) | partial: CDs same-motive 374791 sig > R3_ori; CR passes only on Toys; same-motive on Toys / Steam not run | [collaborative on MW-R-H](experiments/08_collab_on_mwrh/report_collab_on_mwrh.md) |
| E09 | R3's RL stage (GRPO) on top of MW-R-H | done (negative); beam-NDCG runs submitted 11-27 | [GRPO](experiments/09_grpo_stage2/report_grpo.md) |

## 4. Open issues and risks
1. **MovieLens: C1 fails as is.**
   - MW-R-H re-recommends watched movies and is significantly below R3.
   - Only the swapped-pass (S) and distillation (D) variants meet the criterion there. The author dropped S on the
     other datasets as misaligned with the objective.
   - Decide whether MovieLens is in the paper, and with which recipe.
2. **Steam: a single recipe?** At lr 3e-4 the thought collapsed once and was unused twice; lr 5e-5 is the only setting
   meeting the criterion. Either justify a per-dataset lr or rerun every dataset at one lr.
3. **Seed robustness.**
   - Toys MW-R-H is seed-dependent: one seed is significantly below R3.
   - Steam, and every C2 run, is a single seed.
4. **C2's added value beyond C1.**
   - On CDs, same-motive collaborative reasoning equals MW-R-H in ranking and doubles latent dependence.
   - The paper needs a collaborative-specific measure: the cross-user same-item probe, a same-item donor swap, and
     gains on rows where the history alone is weak.
5. **C2 coverage.**
   - MW-R-H + same-motive exists on CDs (374791) and MovieLens (374774).
   - It is missing on Toys and Steam. Toys has same-motive only on the older MW base (MWC-M 362616).

## 4b. Evidence gaps found while compiling the Report (2026-11-27)
These were found when the experiment reports were compiled from the logs. Each report's "Source notes" or "Gaps"
section has the details.
- **Baselines:**
  - SASRec is missing for CDs, Steam and MovieLens. The Toys SASRec number uses a different protocol.
  - CDs popularity and item-CF are not recorded. Toys item-CF is in a memory note only.
  - These are needed for the main table ([E00](experiments/00_datasets_baselines/report_datasets_baselines.md)).
- **R3 latent tests:**
  - No latent test exists for R3_ori Toys stage 1 or for plain R3 (A). E01 uses the stage-2 replica 371452 instead.
  - The log's "R3_ori Toys" interventions were actually Toys v0 checkpoint-844
    ([E01](experiments/01_r3_ignores_latent/report_r3_ignores_latent.md)).
- **Paired t-tests:** none were recorded for the 2026-10 to 11-13 runs (E03, the MW-base E07 runs, early GRPO); they
  have bootstrap CIs only.
  - The E01, E02, E04 and E05 compilers recomputed p_t from the saved per-row files where those exist.
  - E03 and E07 still lack them.
- **Epoch selection differs:** R3, the no-latent SFT and MW 356322 keep the best epoch, while MW-R-H keeps epoch 3. This
  matters most for Steam, MovieLens and Goodreads ([E06](experiments/06_mwrh_results/report_mwrh_results.md)).
- **Determinism:** the same seed on the same GPU gives different training curves (Toys seed 2, Steam 369662 vs 383065).
  Seed tables should be read as run-to-run variance.
- **Results only in run logs:**
  - staged 2+1 interventions;
  - diffusion / regress / TRM;
  - the CDs A10 reproduction 335891, which was never logged.
- **Stale text:** the `margin_wider_collab.sbatch` header says the 356322 recipe uses Instruct, but all MW-family runs
  loaded base Qwen2.5-1.5B ([E04](experiments/04_margin_wider/report_margin_wider.md)).

## 4c. C1 status under the seed rule (2026-11-27)
| dataset | recipe (per-dataset hyperparameters) | runs passing | verdict |
|---|---|---|---|
| CDs | MW-R-H, lr 3e-4 | 3 / 3 | success |
| Toys | MW-R-H, lr 3e-4 | 3 / 5 | success |
| Steam | MW-R-H, lr 5e-5 | 1 / 1 (383067); replicate 385083 running | success so far; the replicate is the check |
| MovieLens (substitute) | MW-R-H, lr 3e-4 | 0 / 1 | fail (D 1 / 1) |

## 5. Proposed next runs (waiting for the author's go-ahead)
1. **MW-R-H + same-motive** on Toys (lr 3e-4) and Steam (lr 5e-5). This completes the C2 table on the MW-R-H base.
   - Steam MW-R-H-CR 384089 ran at lr 3e-4, not at Steam's lr. Rerun it at 5e-5 before concluding.
2. **C2 evidence beyond ranking:**
   - the motive probe on the final runs;
   - a new forward-only eval that swaps in the thought of a user with the same next item;
   - per-segment gains (non-copy rows, short histories, rare items).
3. **Seeds:** two more seeds of the final C1 and C2 recipes per dataset.
