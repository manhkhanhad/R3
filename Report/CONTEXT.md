# Collaborative latent reasoning for R3: project context for a new session

Read this first, then [main.md](main.md) (the research story and plan), [CONVENTIONS.md](CONVENTIONS.md) (terminology,
reporting rules) and the reports it links. Created 2026-11-27.

## What the project is
- **Base model:** LatentR³ (R3). An LLM recommender (Qwen2.5-1.5B) pools its hidden states over the user's history into a
  latent thought `r`, splices `r` into the prompt as a `<|Thought|>` token, and generates the next item title with
  catalogue-constrained beam search (10 beams). Code: `src/latent/layers.py::LatentModel`.
- **The paper has two contributions (author, 2026-11-27):**
  1. **C1: MW-R-H.** Make the decoder use the latent's content. It's a history-swap hinge weighted per row by a
     history-only reference model, plus the WIDER diversity regulariser.
  2. **C2: Collaborative reasoning.** Ground each latent in other users' latents through InfoNCE. People buy an
     item for shared reasons. C2 needs C1, because R3's decoder ignores `r`, so reshaping `r` alone cannot change
     predictions.
- **Author's framing of C2, in their words:** R3's latent is generated from one sample alone and cannot be understood or
  verified. Grounding a latent in other samples' latents works like collaborative filtering in latent space. A group of
  latents that lead to good recommendations is an anchor that "verifies" the latents near it.

## Where things are
- **Worktree:** `/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3MarginWiderCollab`, branch
  `worktree-R3MarginWiderCollab` (never push; never push to main).
- **Main checkout:** `/common/home/users/m/mk.ngohuu.2024/research/R3`. It holds `data/`, `output_dir/`, `run_logs/` and
  `analysis/` (gitignored). Its `src/` has another session's uncommitted edits, so never run jobs from it.
- **Running research log (source of truth for every number):**
  `R3/analysis/document/research_notes/Latent utilization/research_log.md`.
  - The older log (L1–L10, Attempts 1–11) is in `.../research_notes/Latent reasoning for recommendation/research_log.md`.
  - Earlier analysis docs: `R3/analysis/document/decoder_insensitivity_to_latent.md`,
    `avoid_reasoning_collapse_and_shortcut_learning.md`, `collaborative_reasoning.md` (the original C2 design),
    `research_notes/Latent utilization/MW_to_MWC_report_2026-11-12.md`.
- **Code (this worktree):**
  - **MW-R-H** = `src/latent/latent_attention_train_motive.py` (`METHOD=motive MOTIVE_W=0 MARGIN_MODE=refweight
    REF_WEIGHT_FN=rank REF_TAU=0.5`).
  - **Reference model:** the reference NLL comes from `src/latent/reference_score.py`, and the files are in
    `R3/output_dir/margin_wider_collab/reference_nll/<Category>_nolatent/train.pt`.
  - **Same-motive InfoNCE:** `src/latent/motive_contrastive.py` (`MOTIVE_W>0`).
  - **Original collaborative InfoNCE:** `src/latent/collaborative_margin_attention.py` +
    `latent_attention_train_collaborative_margin.py` (`METHOD=infonce`).
  - **WIDER:** `src/latent/wider_reg.py`.
  - **One sbatch for all of these:** `scripts/margin_wider_collab.sbatch`.
- **Frozen code per job:** `R3/output_dir/margin_wider_collab_code/code_<sha>`.
- **Outputs:**
  - Results: `R3/output_dir/margin_wider_collab/<run>/` (eval_raw.json, generation_test_*_{raw,mask,mean,shuffle_user}*.json).
  - Models: `scratchDirectory/R3/output_dir/<run>/`.
- **Analysis scripts:** `/common/home/users/m/mk.ngohuu.2024/.claude/jobs/e8c23a07/tmp/`. Examples: `report_tables.py`
  (Toys), `toys_cr.py`, `cds_cr.py`, `steam_lr_eval.py <suffix>`, `cr_progress.py <log>`.

## Rules (from the author; follow them)
- **Environment:** `module purge; module load CUDA/12.4.0; conda activate rrec`.
- **GPU jobs:** submit and cancel only through the **JobManager** peer session (SendMessage); never `sbatch` directly.
  Use `researchshort` with `a40|l40s|a100|l40`. H100 only if the JobManager's user allows it. The hady partition
  (`hadylauwresearch`, A40) is used when the author asks. Always use `--requeue`.
- **Code changes:** discuss the design with the author first; implement, test, commit (ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`), freeze, then submit. Keep new modules in fp32. Divide
  auxiliary losses by grad-accum.
- **Prompt and baselines:** never change the R3 prompt. Training-free tricks are baselines only. No out-of-fold ideas.
  No stop-gradient one-sided hinge.
- **Reporting:** report HR/NDCG @1/3/5/10, with a focus on @5/@10. Give the paired t-test AND the paired bootstrap
  (2,000 resamples, seed 42) on all test rows, and name each model with its run ID (see [CONVENTIONS.md](CONVENTIONS.md)).
- **Deleting outputs:** only with the author's OK, and only model files.
- **History-excluded evaluations:** postponed.
- **New experiments:** add a report under `Report/experiments/`, link it from the table in [main.md](main.md), and append to
  the research log.

## Scope and hyperparameters (author, 2026-11-27)
- **Develop on CDs, Toys and Steam only.** MovieLens is a substitute dataset; run there only on request.
- **Every model uses its dataset's best MW-R-H hyperparameters:**
  - Toys and CDs: lr 3e-4.
  - Steam: lr 5e-5.
  - Everything else is shared; the full table is in [CONVENTIONS.md](CONVENTIONS.md).
- **Seed rule:** a configuration succeeds if the criterion holds on its 1 run, or on ≥ 2 seeds when several were run.

## Current state (2026-11-27)
- **C1, MW-R-H:**
  - **CDs:** significantly above R3_ori with the latent used, over 3 seeds.
  - **Toys:** ≈ R3 on average, but seed-dependent.
  - **Steam:** lr 5e-5 (383067) uses the latent and is significantly above R3 at NDCG@5/@10.
  - **MovieLens:** significantly below R3; only the S/D variants meet the criterion there.
  - **Goodreads:** stopped.
  - Details: [E06](experiments/06_mwrh_results/report_mwrh_results.md).
- **C2, collaborative reasoning:**
  - **Same-motive InfoNCE:** works on CDs on the MW-R-H base (374791: significantly above R3_ori, twice MW-R-H's
    latent use) and on Toys on the MW base (MWC-M 362616).
  - **Original collaborative-reasoning InfoNCE (MW-R-H-CR):** meets the criterion only on Toys (383923).
  - **Not run yet:** MW-R-H + same-motive on Toys and Steam.
  - Details: [E07](experiments/07_collab_on_mw/report_collab_on_mw.md),
    [E08](experiments/08_collab_on_mwrh/report_collab_on_mwrh.md).
- **Running:**
  - No C1/C2 job is running in this session.
  - The beam-NDCG full-model GRPO on MW-R-H (Toys; code 071e729c; lr 3e-7 / 1e-6 / 3e-6) was submitted on 2026-11-27
    from another session. See [E09](experiments/09_grpo_stage2/report_grpo.md) and the end of the research log.
- **Proposed next runs:** in [main.md](main.md) §5. They wait for the author's go-ahead.
- **Evidence gaps found while compiling the Report:** in [main.md](main.md) §4b.
