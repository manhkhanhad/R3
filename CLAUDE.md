# R3 latent utilization: make the decoder use `r`'s content

Session brief written 2026-10-25 from the `R3LatentCollaborativeReasoning` session. Read it before doing anything.

## The problem

R3 (LatentR³) computes a latent thought `r = f(x)` by attention-pooling the LLM's hidden states over the history `x`, splices `r` into the input as a `<|Thought|>` token, and the decoder (Qwen2.5-1.5B-Instruct) generates the next item title from **history + `r`** (`src/latent/layers.py::LatentModel`). The decoder **ignores `r`'s content**: it needs *some* vector in the slot, but any sample's `r` (or the mean) works as well as the right one. Every method that only reshapes `r` (collaborative InfoNCE, queue tricks, category alignment, GRPO stage 2) therefore cannot change predictions.

**Goal of this line:** get a model whose predictions depend on the sample's own `r` (mean/shuffle interventions significantly hurt) **and** whose ranking is at least plain R3 (Musical HR@10 0.1226 / NDCG@10 0.0950).

## Evidence (all Musical Instruments unless noted; test n = 8,313)

- **Interventions** (`src/prototype_grounding/experiment.py generate --mode raw|mask|mean|shuffle_user|shuffle_target`): in plain R3 and every collaborative variant, replacing `r` with the mean or another sample's `r` changes HR@10 by ≤ 0.0015 (n.s.). `mask` (zero `r`) often hurts (activation dependence without content dependence).
- **Decoder-insensitivity study** (`analysis/document/decoder_insensitivity_to_latent.md` in the main checkout): swapping `r` moves the output 140–590× less than dropping one history item; recomputed vs stale `r` after a history change give the same output.
  - **H1 confirmed: redundancy.** `r = f(x)` carries nothing the decoder can't read from `x`. When the slot carries information the history lacks (target-category oracle), swapping it redirects 89% of the top-10 (text 92%). Attention to `r` is high (3–11× an average prompt token), scale mismatch is not the cause.
  - A SASRec user vector in the slot is used (swap costs 27% HR@10 on a subset) but overall HR@10 0.1196 < 0.1226: used *instead of* titles, not in addition.
  - A decoder-side margin loss raises content sensitivity 9× at a 4% HR@10 cost, but adds no information.
  - Caveat: plain SFT without a latent scores 0.1009 / 0.0790 and plain R3 with `r` rescaled to have no effect scores 0.1216; R3's gain over SFT may not come from latent content at all (recipes not verified comparable).
- **Attempt 1** (attention curriculum blocking history during part of training): the model learned a mode-conditional policy (use `r` only when history is blocked); at test time history is visible, so `r` is ignored again. Any bottleneck must be identical in training and inference.

## What has worked

| run | what | result |
|---|---|---|
| **L7** latent-only bottleneck (`--latent_only`, `src/latent/latent_bottleneck.py`, `scripts/latent_only_bottleneck.sbatch`, commit `1aedd7e`) | target tokens see only the fixed instruction prefix, `r`, and earlier target tokens; history hidden in training **and** generation (incl. KV-cache beam search); mask verified (history swap with `r` fixed: max \|Δlogit\| 4.8e-4) | **strong content dependence**: raw HR@10 0.1089, mean 0.0295, shuffle 0.021 (Δ ≈ −0.08, significant). **Cost**: standard HR@10 0.1085 / NDCG@10 0.0827 vs plain R3 0.1226 / 0.0950 (−11%). One latent token only. |
| **L5** behaviour-grounded reasoning on CDs (`src/latent/grounded_reasoning.py`, frozen `output_dir/grounded_code/code_4671046`) | 32 donors (other users' `r` + their observed next item) retrieved and added to `r` at inference | knn − shuffle_outcome HR@10 +0.0023 [+0.0010, +0.0037]; Musical (L4) no effect; Toys (L10) knn > off only. Grounding strength frozen at 0.5 by a bf16 bug (new modules should be fp32). |

## What has not worked (don't repeat without a new reason)

Attempt 10 / 11 collaborative InfoNCE (geometry improves, content still unused); L8a/L8b queue NLL gate / diversity caps (same); L9 GRPO stage 2 (`src/grpo_attention_tuning/`): no-op at lr 1e-5, collapse at 1e-4 / 5e-4. **Stage-2 KL penalty is structurally zero**: reference log-probs are computed on embeddings that already contain the policy's `r`, and the decoder is frozen, so there is no trust region (fix: compute reference log-probs from the reference attention module's `r`). Queue drift (D1, Toys): stored latents are stale in epoch 1, fine in epoch 2; not the bottleneck.

## Candidate directions (none agreed yet; discuss with the user first)

1. **Close L7's accuracy gap** while keeping content dependence: more latent tokens (RGLT uses 16; the codebase produces one vector per row), a partial bottleneck (some history tokens visible), better `r` producer (deeper pooling), longer training.
2. **Give `r` information the history lacks** (H1 says that is what gets used): other users' behaviour at inference (L5 donors), CF/SASRec signals, privileged-information distillation from target-conditioned teacher explanations (CDs: `RRec_coconut/Scoter_COT/results/description_reasoning_full_v3/candidates.jsonl`; embeddings `data/CDs_and_Vinyl/info/CDs_and_Vinyl_explanation_qwen3emb_1536.pt`, weak target signal: median target rank 2,257/5,841).
3. **Effect-based objectives instead of state alignment** (RGLT, arXiv 2608.14107: effect matching 34.2 vs state alignment 30.6 nDCG@10): supervise how `r` changes target vs hard-negative likelihood.
4. **Apply the collaborative ideas on L7**, where `r` matters (NLL-gated queue from L8a; a momentum encoder if needed).
5. **GRPO with a real KL anchor**, on L7.

## Where things are

- Main checkout `/common/home/users/m/mk.ngohuu.2024/research/R3` (branch `main`) holds `data/`, `output_dir/`, `run_logs/`, `analysis/` (gitignored). Its `src/` has another session's **uncommitted** edits: don't run jobs from it; run from frozen code copies.
- Previous research log (L1–L10, D1, full numbers): `analysis/document/research_notes/Latent reasoning for recommendation/research_log.md` (worktree copy in `.claude/worktrees/R3LatentCollaborativeReasoning/analysis/...` is kept identical). Start a **new** log for this line, e.g. `analysis/document/research_notes/Latent utilization/research_log.md`.
- Other docs: `analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md` (Attempts 1–11), `analysis/document/decoder_insensitivity_to_latent.md`.
- Baselines: plain R3 v1 Musical 0.1226 / 0.0950 (stage 1; the old "stage 2" result was a copy of stage 1). **Scratch `/common/scratch/users/m/mk.ngohuu.2024/R3/output_dir` was purged ~2026-10-23**: plain R3 v0/v1 checkpoints and original Attempt 10 runs are gone; result files copied to `R3/output_dir` / `analysis/` survive. Write all new outputs under `R3/output_dir/`.
- Metrics: `src/utils/ranking_metrics.calculate_ranking_metrics` on per-row result JSONs; paired bootstrap `scripts/paired_bootstrap_eval.py` (2,000 resamples, seed 42). Known quirk: `shuffle_user` and `shuffle_target` draw the same latent (`experiment.py:645`).

## How to run jobs

- `module purge; module load CUDA/12.4.0; conda activate rrec`. `racer` env only to load Qwen3-Embedding models (rrec has transformers 4.48.2; Qwen3 needs ≥ 4.51).
- Slurm: `--partition=researchshort --qos=research-1-qos`, `--constraint="l40s|a100|l40|a40|3090"`, micro-batch 2 on 3090 / 4 otherwise, batch 128; always `--requeue` (preemptions happen; some were not requeued, check `sacct`). Logs in `run_logs/%x.%j.out`, `--chdir` = main checkout.
- Freeze code for every job: `git archive HEAD src scripts | tar -x -C output_dir/<name>_code/code_<sha>` and pass `CODE_ROOT`; for scripts using relative `src/`, symlink `data/ output_dir/ run_logs/` into the frozen copy and submit with `--chdir`.
- Training is bf16; keep new trainable modules in fp32 (bf16 froze LayerNorm scales / scalar gates in L4–L6).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; don't push.

## Working with the user

- **Discuss the design before modifying code**; get agreement, then implement, test, commit, submit. Ask when a request is ambiguous (which dataset, which method, which settings).
- Report **HR and NDCG at @1, @3, @5, @10** (and note @20 = @10 with 10 beams), with paired-bootstrap CIs for comparisons.
- Log every launched experiment and every result in the research log (status board + section), and keep the memory index up to date.
- Default dataset: Musical Instruments (fast); CDs and Toys for replication.
