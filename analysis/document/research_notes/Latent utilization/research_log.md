# Research log: latent utilization (make the decoder use `r`'s content)

Started 2026-10-25 (worktree `R3LatentUtilization`; brief in its `CLAUDE.md`). Previous line: `../Latent reasoning for recommendation/research_log.md`.
**Default dataset: Toys_and_Games** (user decision, 2026-10-25). Test n = 6,738, train n = 53,898.

## Status board

| id | what | job(s) | status |
|---|---|---|---|
| U1 | Latent pair similarity on plain R3 Toys (same user / same target, history-overlap controlled), stage 2 `nomul_attn_lr_5e-5` and stage 1 `latent` | smoke 338551; 338553 (stage 2), 338554 (stage 1) | done 2026-10-26; **`r` part superseded by U2** (`r` is one constant; templates are batch noise). `h_mean` invalid (prefix bug), rerun 338615 (`toys_r3_stage1_v2`) done |
| U3 | Pooling-attention diagnosis across checkpoints (R3_ori CDs/Musical stage 1+2, Musical margin m0.2) | 338630–338634 | done 2026-10-26: every R3_ori pooling puts ≥99.5% attention on the shared template; `r` = mixture of a few fixed template value vectors |
| T0 | Continuous thoughts implementation + smoke (K=4, hidden, bottleneck; 256 train rows) | 338650 (val sample bug), 338652 (diagnostics OOM), 338653 | done: pipeline OK |
| T1 | S2: continuous thoughts + bottleneck, K=1 / 4 / 8 (Toys, R3_ori recipe) | 338655 / 338656 / 338657 | **failed: disk quota** (checkpoint save with optimizer state); epoch-1/2 diagnostics below |
| T2 | Baseline C: one learned constant thought, no bottleneck (R3 as a learned task bias) | 338658 | **failed: disk quota** after epoch 2 eval |
| T3 | Floor: one learned constant thought + bottleneck (decoder sees only the instruction) | 338659 | **failed: disk quota** after epoch 2 eval |
| T1–T3 rerun | same five runs from scratch, outputs on scratch `scratchDirectory/R3/output_dir/Toys_and_Games_thoughts_*` (user decision 2026-10-26), code `e5baa6a` | 338742 (K1) / 338743 (K4) / 338744 (K8) / 338745 (T2) / 338746 (T3) | all done 2026-10-27; K=1/K=8 content-dependent but HR@10 −23% vs T2; K=4 collapsed |
| B | Recipe-matched plain SFT on Toys | — | not started (plain_sft_train.py is Musical-only, no early stopping) |
| U2 | Collapse diagnosis of R3 Toys stage 1: pooling attention, prefix vs user contribution, fp32, fresh init | 338606 (failed: NaN in JSON), 338608 (prefix bug), 338614 (v2) | done 2026-10-26: pooling attends 100% to template token `" Input"`; collapse learned, not at init |
| V0 | Variational line smoke: history dropout C (item p=0.5) and C+A (CVAE d=64, aux 0.5, prior 0.2, free 2 nats), 256 rows, 1 epoch | 358504 (C), 358505 (C+A), researchshort, code_a4178cd2 (earlier 358373/358374 cancelled: a40 saturated; 358488/358489 valid-sample bug → 84ed02a0; 358490 same-file cp → a4178cd2) | **done 2026-11-08: both pipelines complete end to end** (train, probe, rankings raw/drop0.5/sample, extract, interventions, bootstrap); mu heads zero-initialised afterwards (9b68adc6, KL started at 131 nats) → full runs on code_9b68adc6 |
| V1 | C: history dropout full runs, Toys, item-level p = 0.3 / 0.5 / 0.7, ranked at eval_p = 0 and = p, interventions raw/mask/mean/shuffle_user (user decision 2026-11-08: C first, researchshort) | 358573 (0.3), 358574 (0.5), 358575 (0.7); code_9b68adc6, researchshort 48h l40s|a100|l40|a40, mbs 4 | **done 2026-11-09, negative**: HR@10 0.1042 / 0.0988 / 0.1011 (plain 0.1061), mean & shuffle n.s. at every p, mask −0.028/−0.043/−0.037; at eval_p = p HR@10 0.086 / 0.064 / 0.056; `r` eff. rank 1.7 / 1.3 / 3.8 — decoder reads the visible history, not `r` |
| V2 | A: variational latent full runs (Toys): d=64, beta 1 linear warm-up 30%, free 2 nats, aux 0.5 (Qwen3 item table), prior_prob 0.2; without / with dropout 0.5; ranked raw + CVAE_MODE=sample (+ eval_p 0.5), interventions, cvae_diagnostics.json | 359244 (no dropout, `Toys_and_Games_cvae64_b1_f2_aux0.5_pp0.2`), 359245 (dropout 0.5, `Toys_and_Games_dropitem0.5_cvae64_b1_f2_aux0.5_pp0.2`); code_9b68adc6, researchshort 48h | submitted 2026-11-09 after the V1 negative |

## U1: how similar are latents of the same user / same target item?

**Question (user, 2026-10-25):** on Toys, how similar is `r` between samples of the same user and samples with the same target item?

**Checkpoints (user-provided, original R3 pipeline):** `R3_ori/output_dir/Toys_and_Games/nomul_attn_lr_5e-5` (GRPO stage 2, own eval HR@1/3/5/10 0.0380/0.0628/0.0784/0.1055, NDCG 0.0380/0.0522/0.0586/0.0674) and its stage-1 parent `.../latent`.

**Data structure that confounds naive comparisons:** same-user rows are sliding-window prefixes (test: 82% of same-user pairs nested, median history Jaccard 0.6); same-target pairs mostly share no history item (median Jaccard 0, 18% share any).

**Design** (`src/prototype_grounding/latent_pair_similarity.py`, `scripts/latent_pair_similarity.sbatch`, commit `ae0a0ec`, frozen `output_dir/pair_similarity_code/code_ae0a0ec`):
- Vectors: `r`, plus history-only baselines from the same first pass: `h_last` (hidden state before `<|Thought|>`, the pooling query) and `h_mean` (mean over user-input tokens).
- Centered cosine (split mean removed) as main metric, raw cosine alongside.
- Pairs: random cross-user, same user (all; nested/not nested), same target across users, cross-user pairs sharing ≥1 history item (matched control for same-user); each binned by history Jaccard {0, (0,0.5], (0.5,1), 1}. Cluster-bootstrap 95% CIs (2,000, clusters = groups).
- Permutation-adjusted eta² for user and target; retrieval: same-target precision@10 among other users' nearest neighbours (within split and test→train bank) vs chance; same-user top-1.
- Outputs: vectors `output_dir/latent_pair_similarity/<tag>/<split>/`, report `analysis/results/latent_pair_similarity/<tag>/report.json`.

**Smoke (338551, 400 rows/split):** pipeline OK. **Finding: stage-2 `r` is near-constant**: 3 distinct vectors among 400 test rows (4 in train); ‖r‖ ≈ 300.7, ‖r − mean‖ ≈ 0.4 (0.13% of the norm), per-dim residual std 0.011 vs bf16 resolution ≈ 0.03 at |value| ≈ 6. Within-sample variation is below bf16 precision, so rows snap to a few templates. `h_last` retrieves same-target other-user rows at 9–22× chance on the subset; `r` at ≈ 0–2×.

**Full results (338553 stage 2, 338554 stage 1; test n = 6,738, train n = 53,898).**

`r` is effectively categorical. Distinct `r` vectors: test 4, train 9. Top-3 template shares: test 69.8 / 16.7 / 13.4%, train 51.9 / 33.6 / 14.3%. ‖r − mean‖/‖r‖ = 0.12–0.18%. Stage 2 keeps the stage-1 partition exactly (same row → template assignment), only rescaling ‖r‖ 272.6 → 300.7. All `r` numbers below are identical for both stages to 3 decimals.

Probability that two rows share a template:

| pairs | test | train |
|---|---:|---:|
| random cross-user | 0.53 | 0.40 |
| same user | 0.78 | 0.67 |
| same target, cross-user | 0.56 | 0.40 |

Centered cosine, history overlap 0 (test, 95% cluster-bootstrap CI): same-target − random: `r` +0.020 [−0.019, +0.057] (n.s.); `h_last` +0.114 [+0.100, +0.129]; `h_mean` +0.074 [+0.063, +0.086].

Retrieval (other users, precision@10 vs chance): test→train bank `r` 0.0002 vs 0.0003 (0.6×, ties dominate); `h_last` 0.0210 (66×); `h_mean` 0.0091 (29×). Within test: `r` 0.4×, `h_last` 27×, `h_mean` 13×. Same-user top-1: `r` 0.002, `h_last` 0.67, `h_mean` 0.65 (chance 0.0006). Permutation-adjusted eta² (test): `r` user 0.33 / target 0.010; `h_last` 0.37 / 0.106; `h_mean` 0.25 / 0.055.

Template depends weakly on history length (longer histories → more often the majority template).

**Interpretation.**
- Same-user `r` similarity is explained by shared history (sliding-window prefixes): `r` tracks the history only through a 3-way split.
- Same-target `r` similarity is at chance once history overlap is controlled. `r` carries no target information.
- The pooling *input* does: `h_last` ranks other users' same-target rows 66× above chance. The attention pooling + bf16 output discards it. The sample-specific part of `r` (≈0.1% of the norm, per-dim std ≈ 0.01) is below bf16 resolution at |value| ≈ 6 (≈ 0.03), so rows snap to a few templates.
- This explains the decoder-insensitivity on Toys more directly than redundancy: there is almost no per-sample content in `r` to use (≈ 1.3 bits).
- **Superseded (2026-10-26, U2):** the "templates" are a function of the batch (one template per batch of 8 in 100% of test / 99.96% of train batches): GPU numerical noise from padded batch shape. The same-user > random template sharing is an artifact of same-user rows being adjacent, hence co-batched. `r` itself is one constant vector.
- **Bug:** `shared prefix` was taken from rows 0 and 1, which belong to one user (116 tokens instead of 78), so `h_mean` averaged the wrong span and was zero for short prompts. Fixed in `a00c065`; rerun 338615.
- Open (old): whether an fp32 forward of the same attention module gives a diverse `r` (i.e., the collapse is bf16 rounding of a large shared bias) or the module itself is near-constant.


## U2: why is `r` constant? (collapse diagnosis, R3 Toys stage 1)

Script `src/prototype_grounding/latent_collapse_diagnosis.py` (`scripts/latent_collapse_diagnosis.sbatch`), test first 2,000 rows. Run 338608 (code `69be850`; prefix bug affects only the user-region numbers).

**The pooling attends to one fixed template token.** For all 2,000 rows the attention-pooling weights put mass 1.0 on position 60, the token `" Input"` of the header `### User Input:` (entropy 8e-12 nats vs 5.03 for uniform; mass on user tokens 4e-13). Hence `r = W_v h(" Input") + b_v`: the same vector for every sample. The value bias is negligible (‖b_v‖ = 0.12; ‖r‖ = 272.6).

- fp32 recomputation does not help: 7 distinct vectors, residual 0.11% of the norm. The collapse is not bf16 rounding of a signal; there is no signal.
- The residual variation is a function of the batch (U1 note above): numerical noise.
- **A freshly initialised pooling module on the same hidden states is diverse** (residual 50% of the norm, 1,984 distinct of 2,000). The collapse was learned in stage-1 training (full model, bf16, lr 3e-4, base `Qwen/Qwen2.5-1.5B`, early stop at epoch 2), not present at initialisation.
- Stage 2 (GRPO on the pooling module only) keeps it (U1: identical partition, norm 272.6 → 300.7).

**Implication.** On Toys, plain R3 is SFT plus one learned constant input vector (a one-token soft prompt). Its effect cannot depend on the user. Contrast Musical v1 (`decoder_insensitivity_to_latent.md` §4): `r` diverse (72% of its norm sample-specific) yet unused, so the collapse is run-/dataset-specific while the non-use is general.

**338614 (code `a00c065`, correct shared prefix = 77 tokens).** Same picture; the user-region numbers are now valid:
- Hidden-state norms: position 0 110.8, template 106.6, user tokens 85.5. The attended token is not an outlier in norm, so this is not the LLM's massive-activation sink: the trained query/key simply score `" Input"` ~28+ nats above every user token.
- **Trained module restricted to user tokens gives a diverse `r`**: residual 92% of the norm (‖r‖ 79.4), 1,936 distinct of 2,000. Fresh module restricted to user tokens: 72%. So K and V still carry sample-specific content; only the query–key routing collapsed onto the template.

**U1 rerun with fixed `h_mean` (338615, `toys_r3_stage1_v2`, test):** `h_mean` same-target − random at zero history overlap +0.101 [+0.087, +0.115]; test→train same-target precision@10 0.0133 (42× chance); same-user top-1 0.94. `h_last` unchanged (66×). Conclusion unchanged: the LLM states carry user and target structure; `r` carries none.

## U3: is template-token pooling the mechanism behind the earlier collapse reports too?

**Context.** Collapse was studied extensively before this line (`analysis/document/latent_reasoning_collapse_r3.md` (CDs SFT→RL: 2,337 → 348 FP16 templates), `latent_toys_and_games_v0_epoch_collapse_report.md`, `avoid_reasoning_collapse_and_shortcut_learning.md` Attempts 1–11 (r = μ + z framing), `decoder_insensitivity_to_latent.md` §6 ("attention concentrates on a few tokens")). Those documents describe the geometry; none identified *which* tokens the pooling attends to.

**Batch-artifact check on the earlier banks** (`analysis/evidence/*/train/thoughts.pt`): fraction of extraction batches (size 8) holding a single template: CDs RL 0.033, CDs SFT 0.005, Toys v0 / Musical banks 0.000 (shuffled null 0.000) vs 1.00 for R3_ori Toys. The earlier template counts are real, not batch noise.

**Diagnosis** (code `a00c065`, test first 2,000 rows; shared template = 73–77 tokens):

| checkpoint | attention mass on template | entropy (nats; uniform ≈ 5) | most-attended template tokens (rows of 2,000) | r residual / norm (bf16 = fp32) | share of r's variance from template part | user-only pooling, trained module |
|---|---:|---:|---|---:|---:|---:|
| R3_ori Toys stage 1 | 1.0000 | 0.000 | `" Input"` (2,000) | 0.0015 | 1.00 | 0.92 |
| R3_ori CDs stage 1 | 0.9954 | 0.92 | `" that"`×3 (pos 50/21/4), `" request"`, `" music"` | 0.240 | 1.00 | 0.61 |
| R3_ori CDs stage 2 (lr 5e-5) | 1.0000 | 0.24 | `" that"` pos 4 (1,946) | 0.030 | 1.00 | 0.74 |
| R3_ori Musical stage 1 | 0.9989 | 0.38 | `" a"` pos 32 (1,245), `" an"` pos 2 (523), `" a"` pos 19 | 0.123 | 0.97 | 0.86 |
| R3_ori Musical stage 2 (lr 1e-5) | 0.9981 | 0.38 | same | 0.125 | 0.95 | 0.86 |
| Musical margin m0.2 (this repo, Attempt 2) | 0.7321 | 0.13 | pos 29/42/10/38 | 0.378 | 0.57 (user part 1.82, negative cross term) | 0.65 |
| fresh pooling init (any) | — | — | — | 0.42–0.73 | — | — |

**Mechanism.** In every original R3 checkpoint the pooling query routes ≥ 99.5% of attention to a handful of tokens of the fixed instruction template, usually function words (`" that"`, `" a"`, `" an"`). Their hidden states are identical across samples (causal LM, shared prefix), so `r = Σ_k a_k(x) v_k` lies in the convex hull of K ≈ 1–5 fixed vectors. The sample enters only through the K mixing weights, chosen by the query `h_{t−1}`. This predicts the documented collapse signature exactly: few exact templates (near-one-hot attention → near-identical `r`), effective rank ≈ K − 1, `r` ≈ μ (the dominant token's value), and RL sharpening (CDs stage 2 entropy 0.92 → 0.24, residual 24% → 3%, matching 2,337 → 348 templates). R3_ori Toys is the K = 1 limit.

- Not numerical: fp32 equals bf16 to 4 decimals except Toys.
- Not present at initialisation: a fresh pooling module spreads attention (residual 42–73%).
- The keys/values of user tokens stay informative in every checkpoint (user-only pooling with the trained module: 61–92% residual).
- The margin loss (decoder-side content incentive) is the only checkpoint where user tokens get substantial mass (27%) and carry most of `r`'s variance: content incentive pulls attention off the template.
- All R3_ori runs use base `Qwen/Qwen2.5-1.5B` (launch scripts), trained full-model in bf16.

**Interpretation.** "Latent collapse" in R3 is, mechanistically, **template-token attention collapse in the pooling layer**: the cheapest stable `r` is a soft prompt built from constant prefix states, and next-item CE (with the history visible to the decoder) never penalises it.

## Agreed interpretation (discussion with the user, 2026-10-26)

- In original R3, the learned latent is **not reasoning**: it is a **learned task-specific bias** (≈ a one-token soft prompt). The pooling attends to constant instruction-template tokens, which is why `r` carries no sample content (U1–U3).
- It is not "additional computation": a constant (or K-template mixture) input vector adds no input-dependent computation.
- No experiment so far shows that `r`'s *content* improves ranking. Latent reasoning can help in general, but R3's measured gain over plain SFT has not been attributed to the latent: zeroing `r` at inference is harmless in several checkpoints (Musical v1, Toys v0 ckpt 844), and SFT vs R3 recipes were never matched.
- Proposed discriminating tests (not yet run): A) zero-`r` at inference on R3_ori Toys; B) recipe-matched plain SFT on Toys; C) same recipe with one learnable constant prompt token instead of pooled `r`.

## Redesign decisions (user, 2026-10-26): make `r` reasoning, not a learned bias

Requirements: R1 `r` sample-dependent; R2 decoder uses `r`'s content (mean/shuffle hurt significantly); R3 ranking beats a constant-prompt baseline (C) and R3_ori Toys (stage 1 HR@1/3/5/10 0.0377/0.0607/0.0778/0.1023, NDCG@3/5/10 0.0510/0.0580/0.0659; stage 2 0.0380/0.0628/0.0784/0.1055, 0.0522/0.0586/0.0674).

Chosen design:
- **Producer:** Coconut-style continuous thoughts, K serial steps (thought = LLM last hidden state fed back as next input via a small fp32 norm-matched projection). Replaces template-collapsing pooling.
- **Routing:** full bottleneck (L7 mask): target tokens see only instruction prefix + thoughts + earlier target tokens, identical in training and generation. K ∈ {1, 4, 8} to close L7's accuracy gap.
- **Supervision:** next-item CE only at first; auxiliaries (target-embedding InfoNCE, effect margin) deferred to a later stage.
- **Base model:** `Qwen/Qwen2.5-1.5B` (base), recipe matched to R3_ori Toys, so baselines B (plain SFT) and C (constant prompt token) are comparable.
- In-training diagnostics: thought attention entropy, per-sample share, shuffle ΔNLL.

## T0: implementation of continuous thoughts (2026-10-26)

Commit `722e37d` (+ `240382c` untracking pycache), frozen `output_dir/thoughts_code/code_722e37d`.
- `src/latent/continuous_thoughts.py`: `ContinuousThoughts` (fp32; `hidden` = Coconut K-step recurrence with KV cache over the left-aligned prompt prefix, identity-init projection rescaled to the mean token-embedding norm; `constant` = K learned vectors, baseline C), `first_thought_positions`, `load_fp32_thought_params` (from_pretrained otherwise rounds the module to bf16), `ThoughtDiagnostics` callback (residual share, shuffle ΔNLL, mean ΔNLL on 64 spread validation rows at every evaluation → `thought_diagnostics.jsonl`).
- `layers.py`: `config.latent_continuous_thoughts` builds the module; `generate_embs` splices the K thoughts; bottleneck keyed on the first thought (`latent_bottleneck.py`, K=1 path unchanged).
- `reasoning_dataset.py` / `LogitProcesser.py`: `n_thoughts` (prompt ends in K thought tokens; constrained-decoding hash keys shifted by K). Defaults keep old behaviour.
- `latent_attention_train.py`: `--thought_source hidden|constant --n_thoughts K`. `latent_attention_eval.py`: `--thought_mode raw|mask|mean|shuffle` (bank of raw thoughts `thought_bank.pt`; shuffle draws a different user's block). `latent_bottleneck_check.py` handles K thoughts.
- `continuous_thoughts_check.py`: padding side / batch / KV-cache-vs-naive / splicing equality checks.
- Launcher `scripts/continuous_thoughts.sbatch` (env SOURCE, K, LATENT_ONLY, …; recipe = R3_ori Toys: base Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs, early stopping; resumes from the last checkpoint on requeue).
- Note: the (unused) pooling module `model.attention` is still instantiated so scripts setting `end_k` keep working.

**Smoke 338653** (code `37089ea`, K=4, 256 train rows, 2 steps, 3090): self-check passed (padding/batch/KV-cache vs naive max diff ≤ 4.4e-5; fresh-projection per-sample share 0.088); bottleneck leak check passed (history swap with thoughts fixed: max |Δlogit| 6.9e-3, mean 1.2e-3; thought swap: 13.8); diagnostics after 2 steps: residual share 0.32, shuffle ΔNLL +0.35, mean ΔNLL +0.32; constrained decoding valid (0 unknown predictions); shuffle changes the top-10 in 31/32 rows. HR = 0 (untrained), as expected.

**Launched 2026-10-26 (code `37089ea`):** T1 338655–338657 (`output_dir/Toys_and_Games_thoughts_hidden_k{1,4,8}_bottleneck`), T2 338658 (`..._constant_k1`), T3 338659 (`..._constant_k1_bottleneck`). Each: train (base Qwen2.5-1.5B, lr 3e-4, batch 128, ≤ 3 epochs, early stopping), leak check (bottleneck runs), ranking raw / mask / mean / shuffle, paired bootstrap.

**T1–T3 status (2026-10-26): all five jobs failed with "Disk quota exceeded" while saving a checkpoint** (`save_only_model False` keeps ~5.6 GB optimizer state per checkpoint; 8.7 GB each). Complete checkpoint-422 (epoch 1) survives for K=1, T2, T3; K=4 / K=8 have none. No ranking results yet.

In-training diagnostics and eval loss (diagnostics on 64 spread validation rows; eval loss on the full validation split):

| run | epoch | eval loss | per-sample share of thoughts | shuffle ΔNLL | mean ΔNLL |
|---|---:|---:|---:|---:|---:|
| T1 K=1 + bottleneck | 1 | 0.759 | 0.894 | +0.118 | +0.085 |
| T1 K=1 + bottleneck | 2 | 0.597 | 0.929 | +0.121 | +0.090 |
| T1 K=4 + bottleneck | 1 | 0.815 | 0.431 | +0.019 | +0.004 |
| T1 K=8 + bottleneck | 1 | 0.919 | 0.022 | −0.00005 | −0.00009 |
| T2 constant, no bottleneck | 1 / 2 | 0.743 / 0.594 | 0 | 0 | 0 |
| T3 constant + bottleneck (floor) | 1 / 2 | 0.810 / 0.666 | 0 | 0 | 0 |

- K=1: thought diverse (93%) and used (shuffle +0.12 nats/token); eval loss 0.597 ≈ constant-without-bottleneck 0.594 and well below the no-information floor 0.666, i.e. one continuous thought transmits nearly all the history information that the decoder uses (in NLL; ranking pending).
- K>1 collapses: K=8 thoughts are near-constant (2% per-sample) and unused, with eval loss above the floor at epoch 1 (0.919 vs 0.810); K=4 in between. Likely optimisation of the K-step recurrence at lr 3e-4 (Coconut itself trains with a stage-wise curriculum); not yet diagnosed.

**Rerun (2026-10-26, user decision):** deleted the smoke run and all failed-run directories in `R3/output_dir` (~38 GB; diagnostics and epoch-1 trainer states kept in `analysis/results/continuous_thoughts/failed_quota_20261026/`). Home was at 997 GB of ~1 TB. All five runs resubmitted from scratch with outputs on scratch (`OUTPUT_ROOT`, commit `e5baa6a`): 338742–338746. K>1 rerun unchanged, as requested.

### T2 / T3 results (rerun, 2026-10-26; test n = 6,738; @20 = @10 with 10 beams; NDCG@1 = HR@1)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3_ori stage 1 | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |
| R3_ori stage 2 (lr 5e-5) | 0.0380 | 0.0628 | 0.0784 | 0.1055 | 0.0522 | 0.0586 | 0.0674 |
| **T2 constant token, no bottleneck** | 0.0375 | 0.0634 | 0.0790 | **0.1064** | 0.0525 | 0.0589 | **0.0678** |
| T2, token zeroed at inference (mask) | 0.0377 | 0.0650 | 0.0795 | 0.1075 | 0.0533 | 0.0593 | 0.0684 |
| T3 constant token + bottleneck (no history) | 0.0000 | 0.0006 | 0.0022 | 0.0033 | 0.0003 | 0.0010 | 0.0013 |

Paired bootstrap (2,000, seed 42): T2 − R3_ori stage 2: HR@10 +0.0009 [−0.0037, +0.0055], NDCG@10 +0.0004 [−0.0020, +0.0030]; T2 − stage 1: HR@10 +0.0042 [−0.0004, +0.0088], NDCG@10 +0.0019 [−0.0006, +0.0045] (n.s.). T2 raw − mask: HR@10 −0.0010 [−0.0030, +0.0009] (n.s.).
T2 eval loss by epoch 0.722 / 0.573 / 0.626 (best = epoch 2 loaded); T3 0.785 / 0.642 / 0.632.

- **A single learned constant token reproduces original R3 on Toys** (equal to stage 2, ≥ stage 1). Confirms quantitatively that R3's latent is a learned task bias.
- **The constant is not even needed at inference** (zeroing it: 0.1075). Whatever R3/T2 gain over plain SFT exists comes from training, not from the slot at test time. Baseline B (recipe-matched plain SFT) is still needed to measure that gain.
- **T3 (decoder sees no history) ranks at ≈ chance (HR@10 0.0033) although its eval loss (0.632) is only ~0.06 above T2's.** Teacher-forced NLL is dominated by within-title tokens and is a poor proxy for ranking; the earlier reading "K=1 thought transmits nearly all history information" (based on NLL 0.597 vs 0.594) is premature until ranking is measured.

Rerun diagnostics so far: K=1 epoch 1 residual share 0.14, shuffle ΔNLL ≈ 0 (collapsed), epoch 2 0.88 / +0.114 (recovered; first run had 0.89 at epoch 1) — early training is unstable across runs. K=4 epoch 1: 0.019 (collapsed; first run 0.43).

### T1 results: continuous thoughts + bottleneck (rerun 338742–338744, done 2026-10-27)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 constant token, no bottleneck (reference) | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |
| **K=1 raw** | 0.0180 | 0.0370 | 0.0497 | 0.0801 | 0.0288 | 0.0340 | 0.0439 |
| K=1 mask / mean / shuffle | 0.0000 / 0.0001 / 0.0012 | 0.0000 / 0.0018 / 0.0022 | 0.0009 / 0.0031 / 0.0031 | 0.0022 / 0.0043 / 0.0050 | 0.0000 / 0.0011 / 0.0018 | 0.0004 / 0.0017 / 0.0021 | 0.0008 / 0.0021 / 0.0028 |
| K=4 raw (collapsed) | 0.0003 | 0.0004 | 0.0018 | 0.0033 | 0.0004 | 0.0009 | 0.0014 |
| **K=8 raw** | 0.0175 | 0.0374 | 0.0527 | 0.0819 | 0.0288 | 0.0351 | 0.0445 |
| K=8 mask / mean / shuffle | 0.0001 / 0.0019 / 0.0010 | 0.0003 / 0.0022 / 0.0021 | 0.0007 / 0.0027 / 0.0037 | 0.0018 / 0.0033 / 0.0047 | 0.0002 / 0.0021 / 0.0016 | 0.0004 / 0.0023 / 0.0023 | 0.0008 / 0.0025 / 0.0026 |

Paired bootstrap: K=1 raw − shuffle HR@10 +0.0751 [+0.0686, +0.0816], NDCG@10 +0.0411 [+0.0371, +0.0449]; K=8 raw − shuffle +0.0772 [+0.0705, +0.0839] / +0.0418 [+0.0378, +0.0461] (mask and mean similar, all significant). K=8 − K=1: HR@10 +0.0018 [−0.0025, +0.0061], NDCG@10 +0.0006 [−0.0016, +0.0029] (n.s.). K=8 − T2: −0.0245 [−0.0309, −0.0180] / −0.0234 [−0.0275, −0.0191]; K=1 − T2: −0.0263 [−0.0327, −0.0202] / −0.0239 [−0.0282, −0.0200]; K=8 − R3_ori stage 2: −0.0236 / −0.0229 (all significant). Leak checks passed (history swap with thoughts fixed: max |Δlogit| 5e-5 to 8e-5).
Final diagnostics (epoch 3): K=1 share 0.91, shuffle ΔNLL +0.142; K=4 share 0.06, ΔNLL ≈ 0; K=8 share 0.59, ΔNLL +0.141. Eval loss (epochs 1/2/3): K=1 0.821/0.609/0.593; K=4 0.857/0.677/0.663; K=8 0.805/0.607/0.593.

- **R1 and R2 are met for the first time on Toys**: replacing the thought with the mean, zero or another user's thought drops HR@10 from 0.080 to ≤ 0.005 (≈ the no-history floor 0.0033). Ranking depends entirely on the sample's own thought.
- **R3 is not met**: −23% HR@10, −35% NDCG@10 vs T2 / R3_ori; HR@1 roughly halves (0.018 vs 0.0375). The gap is larger than L7 on Musical (−11%).
- **More thoughts do not help** (K=8 ≈ K=1), and **K=4 collapsed** (thoughts constant from epoch 1, never recovered → floor). Collapse vs recovery varies between runs (first K=4 attempt had share 0.43 at epoch 1; first K=1 attempt was healthy at epoch 1, the rerun collapsed and recovered): nothing anchors the thoughts early, consistent with the user's "no supervision on the latent" hypothesis.
- Results copied to `R3/output_dir/continuous_thoughts_results/`.

## 2026-10-28 — Does R3's collapsed latent produce similar lists across users? (Toys test, analysis only)

User hypothesis: since `r` collapsed to a constant, R3 may give every user a similar top-10. Script `analysis/results/list_similarity/list_similarity.py`, output `toys_test.csv` (20,000 random user pairs, seed 42).

| run | HR@10 | pair Jaccard@10 | pair same top-1 | distinct items (of 6,299) | top-20 items' slot share | most common top-1 share |
|---|---:|---:|---:|---:|---:|---:|
| R3_ori stage 1 | 0.1023 | 0.036 | 0.006 | 3,625 | 27.1% | 4.7% |
| R3_ori stage 2 lr 5e-5 | 0.1055 | 0.025 | 0.007 | 3,775 | 22.5% | 5.1% |
| R3_ori stage 2 lr 1e-4 | 0.1069 | 0.016 | 0.004 | 4,062 | 17.0% | 4.0% |
| T2 constant token | 0.1064 | 0.032 | 0.006 | 3,665 | 25.7% | 3.9% |
| K=1 thought + bottleneck | 0.0801 | 0.029 | 0.022 | 2,348 | 24.5% | 11.4% |
| K=1 thought = mean (positive control) | 0.0043 | 0.787 | 0.781 | 19 | 100% | 87.6% |
| T3 no history (positive control) | 0.0033 | 0.864 | 0.999 | 14 | 100% | 99.97% |

- **Hypothesis not supported.** R3's lists are as personalised as T2's: random pairs share ~2–4% of their top-10, coverage 58–64% of the catalogue. List collapse looks like the controls (Jaccard ~0.8, ≤ 19 distinct items). R3's ~2% identical lists come mostly from duplicate histories (1.4% of rows).
- Reason: the decoder reads the history directly, so a constant `r` acts as a fixed bias and personalisation comes from the history tokens. Latent collapse ≠ output collapse in R3; it does become output collapse only when the decoder has to go through the latent (bottleneck + mean thought).

## Idea-discovery run + plan v2 (2026-10-28)

Full pipeline run `latent-utilization-2026-10-28` (worktree `R3_LatentUtilize`): literature landscape, 29 deduped idea clusters, cross-model triage + novelty + ultra review (receipts in `R3_LatentUtilize/.aris/runs/latent-utilization-2026-10-28.json`; report `R3_LatentUtilize/idea-stage/IDEA_REPORT.md`). **User decision: score-combination ideas (anchored additive correction C10b, nine-plus-one splice C28) are trivial — demoted; training-free methods only as baselines.** Lead: **C1 recency-split bottleneck frontier** (last m items visible, older history only via r, masks identical train/inference; m∈{2,5,10}; matched equal-budget recent-only constant-slot baselines; noninferiority to plain R3 at δ_NI = 0.005 HR@10). Promotion path: **C11 CFG contrast training** gated on P1 below. Plan: `R3_LatentUtilize/refine-logs/EXPERIMENT_PLAN.md`.

## P1: zero-training CFG contrast probe on L7 Musical (job 341710, launched 2026-10-28)

**Dataset note**: Musical, not Toys — the only latent-only (L7) checkpoint lives there (`output_dir/Musical_Instruments_R3_latent_only/latent`); Toys bottleneck checkpoints (T1 rerun) are on scratch and use the continuous-thoughts code path.

Decode on s_null + w·(s_mode − s_null), null = train-mean latent in the slot (evidence symlinked tag `musical_instruments_latent_only_cfgprobe`); constrained 10-beam, seed 42, first 1,500 test rows, batch 8; w=0 subset baseline rerun for pairing. Runs: raw@{0,1.5,3,5}, mean@3 (identity check: slot≡null ⇒ guided≡null), shuffle_user@3. Three gates (v2 review `01a182e8`): (i) contrast magnitude (mean |δ|, KL(r‖null) in result JSONs); (ii) **promotion gate**: some w with guided raw > plain raw, paired CI > 0; (iii) interventions under guidance. Correction on record: guided scores are h + wδ (δ amplified, h retained — not cancellation); E3 training would use stop-gradient on the null branch.

Code: `latent_contrast.py` + `--cfg-w` in `experiment.py generate`, commit `e77b032`, frozen `output_dir/latent_cfg_probe_code/code_e77b032`; mock tests (beam-reorder sync, mask growth, w=1 identity) pass. Outputs → `analysis/results/intervention/musical_instruments_latent_only_cfgprobe/`; log `run_logs/R3_cfg_probe.341710.out`.

| run | job | status |
|---|---|---|
| P1 CFG probe (raw w0/1.5/3/5, mean@3, shuffle_user@3; 1,500 rows) | 341710 FAILED (beam-prefix bug) → **341711** (code 29495a3, --requeue) | PENDING at submit |

**341710 failure + fix (2026-10-28)**: smoke failed at the first early-finished batch item — `BeamSearchScorer.process` points a done item's `next_beam_indices` at absolute row 0 (cross-group copy), so the group-local prefix match in `latent_contrast.py` couldn't find the parent beam. Fixed by matching prefixes over the whole previous batch (the loop's actual invariant; identical prefixes ⇒ identical null caches). Mock test added (`.scratch/test_latent_contrast.py::test_cross_group_copy`). Resubmitted as **341711**, code `29495a3`, frozen `output_dir/latent_cfg_probe_code/code_29495a3`.

**P1 results (job 341711, COMPLETED 2026-10-29, 2h04m, node aloha).** Identity check exact: `mean_cfg3` has mean |δ| ≈ 1.4e-10 (slot = null ⇒ guided ≡ conditional), and its metrics match plain mean decoding (HR@10 0.0273 vs 0.0295 full-test) — the implementation is verified end to end. All on the first 1,500 test rows (this subset is easier than the full test: raw HR@10 0.1360 vs 0.1089; comparisons are paired within the subset).

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | mean\|δ\| | KL(r‖null) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (w=0) | 0.0873 | 0.1040 | 0.1200 | 0.1360 | 0.0873 | 0.0966 | 0.1032 | 0.1084 | — | — |
| raw_cfg1.5 | 0.0873 | 0.1067 | 0.1167 | **0.1400** | 0.0873 | 0.0983 | 0.1025 | 0.1098 | 0.579 | 0.0379 |
| raw_cfg3 | 0.0813 | 0.0987 | 0.1107 | 0.1260 | 0.0813 | 0.0917 | 0.0965 | 0.1015 | 0.602 | 0.0380 |
| raw_cfg5 | 0.0527 | 0.0787 | 0.0933 | 0.1127 | 0.0527 | 0.0670 | 0.0731 | 0.0793 | 0.607 | 0.0377 |
| mean_cfg3 | 0.0007 | 0.0093 | 0.0167 | 0.0273 | 0.0007 | 0.0050 | 0.0079 | 0.0114 | 1.4e-10 | −1.5e-10 |
| shuffle_user_cfg3 | 0.0047 | 0.0087 | 0.0107 | 0.0167 | 0.0047 | 0.0069 | 0.0078 | 0.0097 | 0.649 | 0.0441 |

(@20 = @10 with 10 beams.) Paired bootstrap (2,000 resamples, seed 42), raw − guided (positive = guidance hurts):
- w=1.5: HR@10 −0.0040 [−0.0113, +0.0033] n.s.; NDCG@10 −0.0014 [−0.0050, +0.0020] n.s. (guidance directionally helps, not significant)
- w=3: HR@10 +0.0100 [−0.0007, +0.0207] n.s.; NDCG@10 +0.0068 [+0.0011, +0.0127] **significantly worse**
- w=5: HR@10 +0.0233 [+0.0120, +0.0353], NDCG@10 +0.0290 [+0.0216, +0.0370] — **significantly worse**

Interventions under guidance (reference raw_cfg3 0.1260/0.1015): raw_cfg3 − mean_cfg3 HR@10 +0.0987 [+0.0800, +0.1167], NDCG@10 +0.0901; raw_cfg3 − shuffle_user_cfg3 HR@10 +0.1093 [+0.0920, +0.1260], NDCG@10 +0.0918 — all significant; content dependence under guidance exceeds unguided L7 (Δ ≈ 0.08).

**Gate verdicts**: (i) contrast magnitude PASS — the null branch genuinely diverges (|δ| ≈ 0.6, KL ≈ 0.038 nats/step), so there is signal to amplify; (ii) **promotion gate FAIL** — no w gives a significant decoding gain; best case w=1.5 is +0.004 HR@10 n.s., and the effect turns significantly negative by w=3–5 (amplification also amplifies noise/miscalibration in δ); (iii) interventions under guidance PASS (large, significant). **Consequence per plan: C11 is NOT promoted to method lead; E3 (CFG training with stop-grad null) stays un-triggered; C1 recency-split remains the lead.** The pattern (monotone degradation in w, dependence amplified) is consistent with the v2-review reading: inference-time amplification alone cannot add information — any C11 revival requires training against the contrast, which the prespecified gate does not license from this result.

Raw outputs + bootstrap JSONs: `analysis/results/intervention/musical_instruments_latent_only_cfgprobe/`; log `run_logs/R3_cfg_probe.341711.out`.

**Housekeeping 2026-10-29**: deleted the `latent/` checkpoints of the failed/non-informative Toys thought runs `Toys_and_Games_thoughts_hidden_k4_bottleneck` (collapsed) and `_k8_` (≈ K=1) on scratch (~42 GB freed; user request). Eval JSONs, `0.csv` and `thought_bank.pt` kept; the K=1 bottleneck checkpoint and both constant controls survive. Note: another session moved `R3/output_dir` to scratch (`output_dir/output_dir -> scratchDirectory/R3/output_dir`, 2026-10-29 10:37).

## 2026-10-29 — B1 step 0: history-subset probe (launched)

Brainstorm outcome (with user): A1 privileged hint dropped (hint as predictable as the item from history), A2 collaborative already tested (L5 CDs sig, L4 Musical none, L10 Toys knn vs shuffled outcomes n.s.; SASRec slot used but HR@10 lower), A3/B2/B3 not preferred, effect-based objectives to be resolved within SFT. **Chosen: B1, `r` steers which history items the decoder reads** (sparsemax item weights as an additive attention bias; exact zeros hide items).

Step 0 asks whether item selection has headroom at all. Fixed checkpoint T2 (`scratchDirectory/R3/output_dir/Toys_and_Games_thoughts_constant_k1/latent`, constant slot, full history in training); the test prompt keeps only k ∈ {1, 3} history items (order preserved) chosen by recent / random (seed 42) / oracle (most Qwen3-title-similar to the target) / anti-oracle. Items are deleted from the prompt (hidden from every position). Rows changed: k=1 6,297 / 6,738, k=3 4,820. Toys histories are short (test median 4, mean 5.1). Caveat: T2 never trained with missing items, so this underestimates headroom.

Code `scripts/history_subset_probe.py` / `.sbatch` / `_report.py` (commits bddcd16, 7bf87bb, frozen `output_dir/history_probe_code/code_7bf87bb`). Job **342900** (array of 2 × 4 variants, per-user submit limit 16). Outputs `output_dir/history_subset_probe/Toys_and_Games_T2/`.

Read-out: oracle-k ≫ full history → a selector has headroom, go to B1 training; oracle-k ≈ full → item-level B1 can at best match R3 on Toys (rethink: token level or other dataset).

### B1 step 0 results (job 342900, done 2026-10-29)

Full report: `output_dir/history_subset_probe/Toys_and_Games_T2/report.md` (also rows with history > k). All rows, n = 6,738:

| variant | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full history (T2) | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |
| recent k=1 | 0.0321 | 0.0565 | 0.0741 | 0.0965 | 0.0462 | 0.0534 | 0.0608 |
| random k=1 | 0.0181 | 0.0327 | 0.0450 | 0.0628 | 0.0263 | 0.0314 | 0.0371 |
| **oracle k=1** | 0.0365 | 0.0674 | 0.0908 | **0.1213** | 0.0540 | 0.0637 | 0.0736 |
| anti-oracle k=1 | 0.0088 | 0.0160 | 0.0221 | 0.0295 | 0.0129 | 0.0154 | 0.0178 |
| recent k=3 | 0.0374 | 0.0643 | 0.0809 | 0.1101 | 0.0529 | 0.0597 | 0.0692 |
| random k=3 | 0.0300 | 0.0512 | 0.0652 | 0.0898 | 0.0422 | 0.0479 | 0.0559 |
| **oracle k=3** | 0.0407 | 0.0690 | 0.0836 | **0.1184** | 0.0570 | 0.0630 | 0.0743 |
| anti-oracle k=3 | 0.0209 | 0.0350 | 0.0426 | 0.0617 | 0.0291 | 0.0322 | 0.0383 |

Paired bootstrap (ΔHR@10 / ΔNDCG@10): oracle_k1 − full +0.0148 [+0.0098, +0.0202] / +0.0058 [+0.0026, +0.0090]; oracle_k3 − full +0.0120 [+0.0082, +0.0157] / +0.0064 [+0.0044, +0.0084]; oracle_k1 − recent_k1 +0.0248 [+0.0199, +0.0297] / +0.0128; oracle_k3 − recent_k3 +0.0083 [+0.0049, +0.0119] / +0.0051; recent_k3 − full +0.0037 [+0.0006, +0.0067] / +0.0014 (n.s.). All others significant negative. The oracle item is the most recent one in only 31% of rows (k=1).

- **Headroom exists**: picking the right item beats the full history by +14% HR@10 with no retraining; picking the wrong one drops HR@10 to 0.03. Which items are visible controls the output strongly, which is what B1 needs `r` to exploit.
- Older items are partly noise for T2: recent-3 ≥ full.
- Caveat: the oracle uses the target (upper bound); part of its gain is the decoder's tendency to recommend items similar to what it sees. The realistic question is how much of +0.025 (oracle − recent, k=1) a selector predicted from history alone can recover.

## 2026-10-29 — B1 steered reading: implementation and launch (Toys)

Design (agreed with user): K=1 continuous thought `r` (hidden source, no bottleneck) scores history items, `s_i = <W_q r, W_k mean h_item>/√256` (fp32); top-k = 2 items kept, rows with ≤ 2 items keep all. Dropped items are masked from every position with positions recomputed (= deletion, the step-0 setting); identical masks in training, scoring and KV-cache beam search. Selector trained only by an auxiliary loss toward the oracle items (2 most Qwen3-title-similar to the target; training only), weight 0.2; next-item CE does not reach the scores (hard selection; no straight-through through the attention mask). Code `src/latent/steered_reading.py` (+ hooks in layers.py, continuous_thoughts.py, train/eval), commit `3790ece`, frozen `output_dir/steer_code/code_3790ece`, launcher `scripts/steered_reading.sbatch`.

Self-checks (smoke 343002): item assignment ok; masked vs deleted max |Δlogp| 9.9e-5 (vs full history 14.9); KV-cache decoding vs teacher forcing 9.0e-5. Pipeline end to end ok on 512 train / 64 test rows. Smoke 343001 failed only on a config issue (train sample 512 > 64-row validation file).

Full runs (outputs `scratchDirectory/R3/output_dir/Toys_and_Games_steer_{learned,recent}_k2`): learned k=2 and fixed-rule recent k=2 baseline (jobs 343006 learned, 343007 recent). Eval modes: raw, mean / shuffle `r`, selection forced to recent / all; paired bootstrap vs raw. Compare with T2 0.1064 / 0.0678 and step-0 probe (recent-3 0.1101, oracle-1 0.1213).

## 2026-10-29 — DivReason reimplementation (worktree DivReason, launched)

Source: An et al., "Beyond the Single Path: Divergent Reasoning for LLM-based Recommendation", SIGIR 2026 (doi 10.1145/3805712.3809732; PDF `R3/document/3805712.3809732.pdf`). Reports Toys HR@10 0.1273 vs LatentR3 0.1152 (their numbers; our R3_ori Toys 0.1055). Earlier DivReason-style aggregation on our L6a intents (L6d, Musical) did not beat the prior.

Worktree `.claude/worktrees/DivReason`, branch `worktree-DivReason` (from 3790ece); commit ea88d33, frozen `output_dir/divreason_code/code_ea88d33`.
- SFT stage only: N paths per prompt = first passes with MC-dropout on every FFN output + directional perturbation h_l + alpha·v_k (normalised k-means centroids of per-prompt mean layer-l states, base model, 4,096 train rows), pooled by R3's latent attention; router MLP − β·Σcos, softmax, eval-time pruning (w < 0.05), r_agg in the thought slot; decoder reads history + r_agg (as R3). Full model trained on CE.
- Unspecified in the paper (our defaults): N=4 (their Toys best), dropout 0.1, layer 14/28, alpha = 0.1 × median token norm (≈5.7), β=0.5, uncentred directions as in the paper (direction cosine 0.986–0.988, i.e. nearly parallel; `CENTER=True` variant available). GRPO stage (alternating accuracy/diversity rewards, backbone frozen) not implemented: rewards and schedule unspecified.
- Diagnostics: shuffle / mean / single-path ΔNLL on r_agg at every evaluation; ranking modes raw / single ("w/o Dual") / mean / shuffle with paired bootstrap.
- CPU test (real model, 12 rows) and GPU smoke (job 343049, 512 rows) passed.

Jobs: **343064** DivReason N=4 Toys (`scratchDirectory/R3/output_dir/Toys_and_Games_divreason_n4`); **343065** plain R3, identical recipe/code (`Toys_and_Games_r3_plain_matched`). Compare raw vs plain matched, raw vs single/mean/shuffle (does it use r_agg?), and vs T2 0.1064 / R3_ori 0.1055.

### DivReason: baseline done, method resubmitted (2026-10-29)

- 343064 (DivReason N=4) failed at second 0 on node toga before the first log line (environment/node; same script ran on other nodes). Resubmitted as **343404** with `--exclude=toga`.
- **343065 plain R3, matched recipe/code** (eval loss 0.748 / 0.595 / 0.645, best epoch 2), Toys test n = 6,738:

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain R3 matched (343065) | 0.0386 | 0.0616 | 0.0781 | 0.1066 | 0.0519 | 0.0587 | 0.0679 |
| T2 constant token | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |
| R3_ori stage 2 (lr 5e-5) | 0.0380 | 0.0628 | 0.0784 | 0.1055 | 0.0522 | 0.0586 | 0.0674 |
| R3_ori stage 1 | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |

Plain matched − T2: HR@10 +0.0001 [−0.0039, +0.0043], NDCG@10 +0.0001; − R3_ori stage 2: +0.0010 [−0.0034, +0.0055] / +0.0005 (both n.s.). The baseline reproduces R3 (≈ 0.106 on Toys, not the 0.1152 DivReason reports for LatentR3).

### B1 partial results (2026-10-29 22:00; sel_recent / sel_all and recent-2 shuffle still running)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |
| learned-2 raw | 0.0223 | 0.0429 | 0.0586 | 0.0824 | 0.0340 | 0.0405 | 0.0481 |
| learned-2 mean r | 0.0229 | 0.0423 | 0.0586 | 0.0840 | 0.0339 | 0.0406 | 0.0487 |
| learned-2 shuffle r | 0.0236 | 0.0433 | 0.0594 | 0.0844 | 0.0349 | 0.0415 | 0.0496 |
| recent-2 raw | 0.0283 | 0.0496 | 0.0677 | 0.0957 | 0.0405 | 0.0479 | 0.0569 |
| recent-2 mean r | 0.0280 | 0.0491 | 0.0657 | 0.0944 | 0.0401 | 0.0470 | 0.0562 |

Paired bootstrap: learned raw − shuffle HR@10 −0.0021 [−0.0045, +0.0000], NDCG@10 −0.0015* (shuffle slightly *better*); learned − recent −0.0134 [−0.0187, −0.0080]* / −0.0088*; learned − T2 −0.0240* / −0.0197*; recent − T2 −0.0107* / −0.0110*.
Selection (test, 5,710 rows with > 2 items): = most-recent-2 in 19.6%; shuffling `r` changes the selection in only 17.2% of rows. Training diagnostics (epoch 3, 64 probe rows): same-as-oracle 0.33, oracle recall 0.58 (recent rule: 0.25 / 0.52); shuffle ΔNLL +0.005.

- **B1 v1 fails all three criteria so far**: the decoder does not depend on `r` (shuffle/mean neutral), learned selection is worse than recency, and both are below T2.
- **The selector bypasses `r`**: selection is driven by the item keys (shuffling `r` changes only 17% of selections). The oracle target ("item most similar to the next item") is largely predictable from item properties alone, so `r` gets no role — the same shortcut as R3's template collapse, moved to the selector.
- Hiding history also costs accuracy: even fixed recent-2 is −10% HR@10 vs T2.

### B1 final (jobs 343006 / 343007 complete; results copied to `R3/output_dir/steer_results/`)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned-2 raw | 0.0223 | 0.0429 | 0.0586 | 0.0824 | 0.0340 | 0.0405 | 0.0481 |
| learned-2, selection forced recent | 0.0243 | 0.0470 | 0.0671 | 0.0929 | 0.0376 | 0.0459 | 0.0541 |
| learned-2, all items visible | 0.0197 | 0.0421 | 0.0560 | 0.0819 | 0.0328 | 0.0385 | 0.0468 |
| recent-2 raw | 0.0283 | 0.0496 | 0.0677 | 0.0957 | 0.0405 | 0.0479 | 0.0569 |
| recent-2, all items visible | 0.0236 | 0.0441 | 0.0585 | 0.0862 | 0.0354 | 0.0413 | 0.0502 |
| recent-2 shuffle r | 0.0283 | 0.0503 | 0.0662 | 0.0945 | 0.0409 | 0.0475 | 0.0566 |

Bootstrap: learned raw − forced recent HR@10 −0.0105 [−0.0147, −0.0064]* / NDCG@10 −0.0060*; learned raw − all +0.0004 (n.s.); recent raw − all +0.0095 [+0.0056, +0.0137]* / +0.0067*; recent raw − shuffle +0.0012 (n.s.).

- The learned selection is worse than recency **on the learned model's own decoder** (−0.0105): the oracle-imitating selector drops items the decoder needs.
- Showing all items does not help either trained model (train/test context mismatch: decoders trained on 2-item contexts), so the gap to T2 is partly the training-time restriction itself.
- **B1 v1 verdict: fail** (no `r` dependence, below recency, below T2). Root cause: oracle-similarity supervision is answerable from items alone, so the selector ignores `r`; hard 2-item restriction costs ~10% HR@10 by itself.

### DivReason N=4 (job 343405, code 47a2092): interim (2026-10-30, mean/shuffle ranking still running)

Training (3 epochs, 3090, ~12.5 h): eval loss 0.6xx / **0.625** / 0.684 (best epoch 2; plain matched 0.595). Diagnostics on 64 validation rows, epochs 1 / 2 / 3: shuffle ΔNLL −0.0008 / +0.0002 / −0.0010, mean ΔNLL −0.00001 / −0.0007 / +0.0006, single-path ΔNLL ≈ 0 / −0.0007 / +0.00004 — **the decoder ignores r_agg at every epoch**. Path cosine 0.94 → 0.90, path spread 0.22 → 0.25; router max weight 0.57 → 0.38, entropy 0.93 → 1.28 (of 1.39): weights become close to uniform.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DivReason N=4 raw | 0.0378 | 0.0620 | 0.0803 | 0.1079 | 0.0518 | 0.0592 | 0.0682 |
| DivReason single path ("w/o Dual", same checkpoint) | 0.0378 | 0.0620 | 0.0800 | 0.1088 | 0.0518 | 0.0591 | 0.0684 |
| plain R3 matched | 0.0386 | 0.0616 | 0.0781 | 0.1066 | 0.0519 | 0.0587 | 0.0679 |
| T2 constant token | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |

DivReason − plain matched: HR@10 +0.0013 [−0.0027, +0.0053], NDCG@10 +0.0003 (n.s.); raw − single path: −0.0009 [−0.0022, +0.0004] (n.s.); − T2: +0.0015 (n.s.). SFT-stage DivReason reproduces neither the paper's gain nor any benefit of multiple paths.

### DivReason + path-diversity test (launched 2026-10-30)

User hypothesis: with the history identical across paths, forcing the paths' predictions apart makes the decoder use the latent. Weights set by the user: per-path CE 1.0, diversity 0.2, margin 0.2 nats; paper (uncentred) directions.
- Loss = CE(r_agg) + 1.0 · mean_n NLL(target | x, r_n) + 0.2 · max(0, 0.2 − mean pairwise JS between the paths' target next-token distributions). Extras weighted by token share of the accumulated batch (num_items_in_batch), like the main CE. **Note:** with `**kwargs` in LatentModel.forward the HF Trainer does not divide by accumulation steps, so a per-micro-batch mean auxiliary term is over-weighted by the accumulation factor (32–64×); this applies to B1's `aux_weight` in the R3LatentUtilization worktree.
- Diagnostics separate **path identity** (same user, path 0 vs path 1: JS and ΔNLL) from **user content** (another user's path 0 in this user's prompt). Ranking modes raw / mean / shuffle / single / path0.
- Code 3a75c67 (frozen `output_dir/divreason_code/code_3a75c67`); CPU test passed (extra loss matches its expected token-weighted value). Jobs: smoke **343626**, full **343628** (afterok, 40+ GB GPU, `Toys_and_Games_divreason_n4_pathdiv`).

### DivReason N=4 final (job 343405, done 2026-10-30)

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw (method) | 0.0378 | 0.0620 | 0.0803 | 0.1079 | 0.0518 | 0.0592 | 0.0682 |
| single unperturbed path | 0.0378 | 0.0620 | 0.0800 | 0.1088 | 0.0518 | 0.0591 | 0.0684 |
| r_agg = mean | 0.0383 | 0.0629 | 0.0810 | 0.1089 | 0.0524 | 0.0597 | 0.0687 |
| r_agg shuffled across users | 0.0384 | 0.0619 | 0.0803 | 0.1083 | 0.0520 | 0.0595 | 0.0685 |
| plain R3 matched | 0.0386 | 0.0616 | 0.0781 | 0.1066 | 0.0519 | 0.0587 | 0.0679 |

raw − mean HR@10 −0.0010 [−0.0025, +0.0003], − shuffle −0.0004 [−0.0019, +0.0009], − single −0.0009 [−0.0022, +0.0004], − plain matched +0.0013 [−0.0027, +0.0053]; all n.s. **Paper DivReason (SFT stage) is plain R3: no gain, multiple paths unused, r_agg ignored (mean/shuffle cost nothing).** Results copied to `output_dir/divreason_results/Toys_and_Games_divreason_n4/`.

## 2026-10-30 — Literature: arXiv 2610.21182 (Qiu et al., "Does Latent Chain-of-Thought Actually Reason?")

Diagnostic/taxonomy paper (COCONUT, CODI, SIM-CoT-on-COCONUT; ProsQA, ProntoQA, GSM8K; GPT-2, Llama). Not a fix paper. Relevant points:
- Chain is ignored ("carrier"/inert) when the answer is already readable from the question's last hidden state before the chain (= our H1 redundancy). GPT-2 COCONUT ProsQA: deletion 0.973 → 0.973, mean/donor Δ 0.
- Content is used whenever the task needs it (GSM8K, all recipes: COCONUT deletion 0.333 → 0.103, donor −0.307 worse than deletion).
- **The one remedy: per-step supervision (SIM-CoT)** — an auxiliary decoder reconstructs written step k from slot k in training, discarded at inference. Installs dependence on logical tasks where COCONUT was inert (GPT-2 ProsQA 0.993 → 0.800 on deletion, ProntoQA 1.000 → 0.827, donor −0.507); a continued-training control with aux weight 0 stays inert. Not universal: Llama ProsQA becomes a *placeholder* (deletion hurts, no replacement changes any answer); Llama ProntoQA stays inert.
- Protocol worth adopting: deletion + noise/mean/donor replacement, helped/hurt counts, MDE per test.

## 2026-10-30 — Gist bottleneck (launched, Toys)

Rationale (no latent supervision; supervision is another session's line): routing is the only lever that reliably forces latent use (L7, K-thought bottleneck), and its accuracy cost likely comes from channel capacity (one input-embedding vector per slot). Gist design: M learned latent tokens after the history in the same pass read the history at every layer; a final readout token and the target tokens see only the instruction prefix, the latents and earlier target tokens, so the target reads the history only through the latents' per-layer KV (gisting, Mu et al. 2023). K thoughts = M latents + 1 readout (without the readout the first target token is predicted at a latent position that reads the history directly — found by the leak check). Positions unchanged (target positions reveal history length). Interventions: delete (latents hidden from the target side, eval `thought_mode=delete`); shuffle = another user's predictions (computed offline, output depends on the history only through the latents).

Code: `latent_bottleneck.py` gist mode, `--latent_gist`, `latent_gist_check.py`, `continuous_thoughts.sbatch GIST=1`; commits 61e641c, 9640b58, 8f72cc2; frozen `output_dir/gist_code/code_8f72cc2`. CPU check (K=5): history-token perturbation with latents deleted max |Δlogp| 0.0, latents visible 0.27.

Jobs: smoke 343805; **M=4 (K=5) 343806, M=16 (K=17) 343807**; outputs `scratchDirectory/R3/output_dir/Toys_and_Games_gist_m{4,16}`. Compare: T2 0.1064 / 0.0678, T3 (latents cannot read history) 0.0033, K=1 thought bottleneck 0.0801.

### Path-diversity run 343628: failed at the epoch-1 checkpoint (disk quota), epoch-1 diagnostics (2026-10-30)

Failed after 4.5 h with `OSError: Disk quota exceeded` while saving the epoch-1 checkpoint (scratch at 484 GB; nothing saved). Freed 24 GB by deleting my two smoke-run dirs. Epoch-1 lines saved to `output_dir/divreason_results/Toys_and_Games_divreason_n4_pathdiv_failed_epoch1/`.
- Training: mean pairwise path JS fell from 8.7e-4 (step 10) to 1–6e-5 (steps 110–410), **far below the 0.2 margin**: the hinge stays active but loses to the per-path CE (weight 1.0), which pulls every path's output to the same target.
- Epoch-1 diagnostics (64 validation rows): eval loss 0.753 (plain matched 0.748 at epoch 1); shuffle ΔNLL −0.0010, mean −0.0007, single −0.0003; JS same-user-other-path 2.1e-5, JS other-user-same-path 2.6e-5; ΔNLL 0.00000 / −0.0001. **At weight 0.2 the decoder still ignores the latent (path identity and user content alike).**

### Path-diversity rerun, option A (user choice, 2026-10-30)

Diversity weight 0.2 → **1.0** (per-path CE 1.0, margin 0.2 nats, paper directions), code 3a75c67, job **343950** (`Toys_and_Games_divreason_n4_pathdiv_w1`, 40+ GB GPU). If the path JS still collapses far below the margin, next are B (per-path CE 0.2) or C (winner-take-all).

## Category prediction vs item prediction (Toys, launched 2026-10-30)

**Question (user):** train a category-prediction model; are there rows where the category is predicted correctly but the item is not?

**Setup (user choices):** level-3 Amazon category paths (194 classes over the dataset's items; 257/6,738 test targets have none and are excluded from the analysis; 2,052 train / 252 valid rows dropped). Plain Qwen2.5-1.5B-Instruct SFT (`src/latent/plain_sft_*.py`, new `--instruction`) on the R3 prompt with history titles unchanged and the instruction asking only for the next item's category; lr 2e-5, batch 128, 3 epochs; constrained beam search (10 beams) over the category catalog. Data `output_dir/category_prediction/Toys_and_Games_level3/data/` (`scripts/category_prediction_data.py`). Analysis `scripts/category_vs_item_analysis.py`: category HR@k, 2x2 of category top-1 vs item hit@1/@10, and the item model's implied category (category of its top-1 item) as the baseline a category model must beat. Item models: R3_ori Toys (`R3_ori/output_dir/Toys_and_Games/latent/final_result.json`) and T2 (`output_dir/history_subset_probe/Toys_and_Games_T2/eval_full.json`; T2's scratch checkpoint dir no longer exists). Code `f50b50b`, frozen `output_dir/category_code/code_f50b50b`; smoke 343964 passed (3 min); full run **343971** submitted 2026-10-30.

**Implied-category baseline from R3 (no new model):** R3's top-1 item has the target's level-3 category in 0.1645 [0.1560, 0.1733] of rows; some top-10 item has it in 0.3788. Category-right / item-miss@10 already occurs in 636 rows by this measure.

## Does forcing the decoder to use r un-collapse the pooling? (L7 collapse diagnosis, 2026-10-30)

Job 343821, `src/prototype_grounding/latent_collapse_diagnosis.py` (code `8f72cc2`) on the Musical L7 latent-only checkpoint (`scratch/R3/output_dir/Musical_Instruments_R3_latent_only/latent`), 2,000 test rows; results `analysis/results/latent_collapse_diagnosis/musical_L7_latent_only/`.

| checkpoint | pooling mass on template prefix | on user tokens | share of r's across-row variance from user tokens | ‖r_user part‖ / ‖r_prefix part‖ |
|---|---:|---:|---:|---:|
| Musical plain R3 stage 1 | 0.9989 | 0.0011 | 0.042 | 0.12 / 89.0 |
| Musical L7 bottleneck | 0.0020 | 0.9980 | 0.998 | 129.3 / 0.23 |

**Reading:** same pooling module and data; the only change is that the decoder cannot read the history. The template collapse disappears completely. Collapse (H2) is a consequence of the decoder not needing r (H1), not an independent cause: with no gradient pressure, pooling settles on the user-invariant prefix.

## Gist M=16 failure and resubmission

343807 failed after 59 min writing its epoch-1 checkpoint (`PytorchStreamWriter failed writing file`, optimizer.pt truncated) to scratch; scratch was not full afterwards (filesystem 14% used), so it looks transient. Partial dir moved to `Toys_and_Games_gist_m16/failed_343807_write_error`; resubmitted from scratch as **343968** (same code and settings). M=4 (343806) unaffected.

### SASRec category model (launched 2026-10-30)

User request: a transformer (SASRec-like) category predictor alongside the LLM one. `src/latent/sasrec_category.py`: causal transformer over history item IDs (2 blocks, d=64, 2 heads, dropout 0.2, max_len 10), last position → softmax over the 194 level-3 categories, CE, Adam 1e-3, early stopping on valid top-1 (patience 10). Two variants: `ids` (IDs only) and `ids_cats` (+ history items' category embeddings). Same category CSVs. Bug fixed before launch: with left padding + causal mask, padded queries saw no key and produced NaNs that leaked into real positions (eval-mode top-1 0.013 < majority 0.059); mask now lets every position attend to itself. Analysis now also reports the trivial baseline "last history item's category": test top-1 0.1870 [0.1778, 0.1966]; the target category appears somewhere in the history for 0.3276 of rows. Code `a72b0d9` (frozen `output_dir/category_code/code_a72b0d9`), job **343973**, results `output_dir/category_prediction/Toys_and_Games_sasrec_category_level3/`.

**SASRec category results (343973, 3 min; n = 6,481 test rows with a category):**

| model | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|
| last history item's category (trivial) | 0.1870 | – | – | – |
| R3 implied (top-1 item's category) | 0.1645 | – | – | (any top-10 item: 0.3788) |
| SASRec IDs | 0.1805 | 0.2929 | 0.3499 | 0.4490 |
| SASRec IDs + history categories | 0.1879 | 0.3103 | 0.3697 | 0.4749 |

- vs last-item baseline (top-1): IDs −0.0065 [−0.0161, +0.0026], IDs+cats +0.0009 [−0.0080, +0.0097] (both n.s.): SASRec learns "repeat the last item's category" and nothing beyond.
- vs R3's implied category: +0.0160 [+0.0069, +0.0248] (IDs), +0.0235 [+0.0142, +0.0319] (IDs+cats), significant but small; R3's own top-1 item is less often in the right category than the last history item is.
- Category right / item wrong (IDs+cats vs R3): 1,046 rows at @1, 818 at @10. Rows the R3 ranking does not reflect (category right, R3 miss@10, R3 top-1 item's category wrong): 435 (6.7% of rows); no R3 top-10 item in the target category: 123 (1.9%). Converse (R3 implied right, SASRec wrong): 361. T2 nearly identical (430 / 108 / 350).

## R3 without the latent on Toys (launched 2026-10-30, session R3_Diffusion)

User request: compare R3 with the same model trained without the latent. No such Toys run existed (T2 has a constant token, not none). Plain SFT (`src/latent/plain_sft_train.py`) on the R3_ori recipe: base Qwen2.5-1.5B, same prompt with no `<|Thought|>` token, lr 3e-4, batch 128, ≤ 3 epochs, early stopping patience 1 on eval loss with the best epoch reloaded (added `--early_stopping`), catalog-constrained 10-beam eval. Code `24c71a5` (worktree R3_Diffusion), frozen `output_dir/no_latent_code/code_24c71a5`, job **344012**, results `output_dir/r3_no_latent/Toys_and_Games/eval_no_latent.json`. Compare with R3_ori Toys `final_result.json`: HR/NDCG @1/3/5/10, paired bootstrap, top-10 overlap.

## Gist bottleneck M=4 results (343806, done 2026-10-30, 4h01m)

Toys, n = 6,738; results copied to `output_dir/gist_results/m4/`. @20 = @10.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gist M=4 raw | 0.0230 | 0.0423 | 0.0603 | 0.0899 | 0.0341 | 0.0414 | 0.0510 |
| gist M=4, latents deleted | 0.0016 | 0.0022 | 0.0030 | 0.0039 | 0.0020 | 0.0023 | 0.0026 |
| T2 (no bottleneck) | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |
| K=8 thought bottleneck | 0.0175 | 0.0374 | 0.0527 | 0.0819 | 0.0288 | 0.0351 | 0.0445 |

- raw − delete: HR@10 +0.0861 [+0.0794, +0.0929], NDCG@10 +0.0484 [+0.0441, +0.0528]; helped 592 rows, hurt 12. Deleting the latents drops ranking to the T3 floor (0.0033): all history information reaches the target through the 4 latents.
- raw − T2: −0.0165 [−0.0220, −0.0113] / −0.0168 [−0.0202, −0.0136] (−16% HR@10). raw − K=8 thought bottleneck: +0.0080 [+0.0022, +0.0132] / +0.0065 [+0.0032, +0.0096]: reading the history at every layer closes about a third of the bottleneck gap.
- Not run: swapping in another user's latents (gist latents are per-layer KV; would need a new eval mode). The "thought diagnostics" (residual share / shuffle ΔNLL = 0) measure the constant input thought vectors, not the gist KV, so they are uninformative here.
- M=16 (343968) still training.

**LLM category results (343971, 3h20m; report regenerated with code a72b0d9 to include the last-item baseline; `output_dir/category_prediction/Toys_and_Games_category_level3/report.md`):** HR@1/3/5/10 0.1981 / 0.3331 / 0.3987 / 0.5098 (n = 6,481). − last-item baseline: +0.0111 [+0.0025, +0.0196]; − R3 implied category: +0.0336 [+0.0242, +0.0427]; − SASRec IDs+cats: +0.0102 [+0.0023, +0.0185] (both right 874, LLM only 410, SASRec only 344). Category right / R3 item miss@10: 904 rows; rows R3's ranking does not reflect (R3 top-1 category also wrong): 499 (7.7%); no R3 top-10 item in the target category: 126 (1.9%); converse (R3 implied right, LLM wrong): 365. Conclusion (user agreed before this result): predicted category adds little beyond the history; the LLM gains ~1 point top-1 over "repeat the last category".

## Gist bottleneck M=16 results (343968, done 2026-10-30, 4h12m)

Results `output_dir/gist_results/m16/`.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gist M=16 raw | 0.0236 | 0.0438 | 0.0595 | 0.0911 | 0.0351 | 0.0415 | 0.0517 |
| gist M=16, latents deleted | 0.0001 | 0.0007 | 0.0012 | 0.0022 | 0.0005 | 0.0007 | 0.0010 |

- raw − delete: HR@10 +0.0889 [+0.0822, +0.0957], NDCG@10 +0.0507 [+0.0464, +0.0549]; helped 605, hurt 6.
- M=16 − M=4: +0.0012 [−0.0025, +0.0053] / +0.0007 [−0.0013, +0.0028] (n.s.). M=16 − T2: −0.0153 [−0.0208, −0.0101] / −0.0161 [−0.0195, −0.0128].
- **Verdict:** gist bottleneck gives full reliance on the latents but plateaus ~15% below T2; more latent tokens do not help, so the capacity of the latent path is not the limit. Success criterion (HR@10 ≥ T2) not met.

**Result (344012, 3 h 42 min; best epoch 2, eval loss 0.593 vs R3_ori 0.595):** R3 vs no-latent on Toys test (n = 6,738), paired bootstrap 2,000 / seed 42:

| metric | R3 | no latent | R3 − no latent [95% CI] |
|---|---:|---:|---:|
| HR@1 | 0.0377 | 0.0365 | +0.0012 [−0.0024, +0.0049] |
| HR@3 | 0.0607 | 0.0594 | +0.0013 [−0.0030, +0.0056] |
| HR@5 | 0.0778 | 0.0761 | +0.0016 [−0.0033, +0.0064] |
| HR@10 | 0.1023 | 0.1045 | −0.0022 [−0.0077, +0.0034] |
| NDCG@3 | 0.0510 | 0.0498 | +0.0011 [−0.0023, +0.0046] |
| NDCG@5 | 0.0580 | 0.0567 | +0.0013 [−0.0021, +0.0046] |
| NDCG@10 | 0.0659 | 0.0658 | +0.0001 [−0.0032, +0.0033] |

No difference at any cutoff. Top-10 Jaccard 0.232, same top-1 0.301; hits: both 0.0764, only R3 0.0258, only no-latent 0.0280. The overlap equals that of two independently trained R3-type runs (R3 vs collaborative-InfoNCE R3: Jaccard 0.239, same top-1 0.309), so the lists differ by run-to-run variance only. On Toys the latent adds nothing to ranking.

### Where the gist gap comes from (offline, 2026-10-30)

Rows split by whether the target title's first word (≈ brand) matches a history title's first word (2,170 / 6,738 rows):

| rows | n | T2 HR@10 | gist M=16 | gap | share of gap |
|---|---:|---:|---:|---:|---:|
| same brand word in history | 2,170 | 0.2986 | 0.2571 | +0.0415 | 0.87 |
| other | 4,568 | 0.0151 | 0.0123 | +0.0028 | 0.13 |
| title word overlap ≥ 0.5 | 1,363 | 0.3903 | 0.3294 | +0.0609 | 0.81 |

90% of T2's hits (648/717) are in brand-repeat rows; in 68% of those the matching item is the last one. The bottleneck mainly loses exact lexical copying of brand / product-line tokens from the history, not "reasoning" capacity (consistent with M=16 ≈ M=4).

## Two-head R3: fast (bypass) + slow (latent) heads fused at the output (launched 2026-10-30, session R3_Diffusion)

Motivation (user): R3 and R3-without-latent differ only by run-to-run noise (shared top-10 items 3.56/10 vs 3.62/10 for two R3-type runs), so R3's latent path is effectively the plain LLM path. Build a model with an explicit bypass head and an explicit latent head, fused at the output (cf. PLR, arXiv 2601.03153; PLR itself has no fast-only / slow-only ablation and its slow path sees the full history).

Design (user choices 2026-10-30): one shared Qwen2.5-1.5B, gist prompt (history, 16 learned latent tokens, 1 readout token, target), two attention views (`latent_bottleneck.py`, `fast=`): **slow** = gist bottleneck (target sees prefix + latents + readout); **fast** = target sees the whole history + readout, not the latents (= R3 without latent). Fusion per target token: log p = log_softmax(log p_fast + w·log p_slow), w learned fp32 (init 1). Loss = NLL(fused) + NLL(fast) + NLL(slow). Ranking: catalog-constrained 10-beam search stepping both heads with one KV cache each (`two_head.py::beam_search`); modes fused / fast / slow from the same checkpoint, plus latent deletion (fused_delete, slow_delete). Recipe as the gist runs (base model, lr 3e-4, batch 128, ≤ 3 epochs, early stopping).

Checks (job 344525, on the gist M=4 checkpoint): custom beam search = HF generate (identical top-10 8/8 rows, score diff 0); fast view: replacing the latents changes fast logits by 0.0 (slow 2.53), one history token changes fast by 0.22; beam score = teacher forcing within bf16 error (fast −3.048 vs −3.072, slow −5.259 vs −5.267, fused −2.008 vs −2.055); loss + backward with grad on w. Smoke 344541 passed (train + 5 evals).

Code `e55dc7e` (worktree R3_Diffusion, branch `worktree-R3_Diffusion`), frozen `output_dir/two_head_code/code_e55dc7e`; full run **344555**; outputs `scratchDirectory/R3/output_dir/Toys_and_Games_two_head_k17`, results copied to `output_dir/two_head_results/Toys_and_Games_two_head_k17/`. Compare: R3 0.1023 / no-latent 0.1045 / T2 0.1064 HR@10; gist M=16 alone 0.0911. Success: fused > fast head and > no-latent R3 (paired bootstrap), and fused_delete < fused (the fused prediction depends on the latent).

### Option A (diversity weight 1.0, JS hinge, job 343950): training done 2026-10-31, ranking running

Path JS stayed 1e-5–1e-4 for all 1,263 steps (margin 0.2), hinge pinned at ≈ 0.2. Eval loss best 0.602 (epoch 2; plain matched 0.595). Diagnostics epochs 1/2/3: shuffle ΔNLL +0.0004 / −0.0007 / +0.0013, mean +0.00001 / −0.0002 / +0.0003; JS same-user-other-path 3e-5 / 2e-5 / 8e-5, other-user-same-path 6e-5 / 5e-5 / 2.2e-4. The decoder still ignores the latent.

**Diagnosis:** JS is quadratic where the paths agree, so the hinge has vanishing gradient at the collapsed point regardless of its weight (toy check: at JS 1e-5 the JS-hinge gradient is ~120× smaller than a sqrt(JS) hinge's; it shrinks with the difference, the sqrt one stays constant). Runs 343628/343950 therefore did not test the hypothesis.

### Path diversity with the Jensen-Shannon distance (launched 2026-10-31)

Hinge on mean per-token sqrt(JS) (first order at agreement), margin 0.45 (= JS 0.2), weight 1.0, per-path CE 1.0, paper directions. Code e3bc3f1 (frozen `output_dir/divreason_code/code_e3bc3f1`), CPU test passed (hinge 0.45 − 0.131 = 0.319). Job **345579** (`Toys_and_Games_divreason_n4_pathdiv_sqrtjs`, 40+ GB GPU).

**Two-head result (344555, done 2026-10-31; n = 6,738; @20 = @10):** training 3 epochs (best eval loss 2.014 = fused+fast+slow, epoch 2); w 1.0 → 0.83.

| eval | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused | 0.0347 | 0.0608 | 0.0748 | 0.1026 | 0.0499 | 0.0556 | 0.0646 |
| fast head | 0.0358 | 0.0619 | 0.0755 | 0.0994 | 0.0509 | 0.0565 | 0.0643 |
| slow head | 0.0193 | 0.0359 | 0.0487 | 0.0705 | 0.0290 | 0.0343 | 0.0413 |
| fused, latents deleted | 0.0298 | 0.0500 | 0.0600 | 0.0781 | 0.0417 | 0.0457 | 0.0516 |
| slow, latents deleted | 0.0000 | 0.0006 | 0.0007 | 0.0030 | 0.0004 | 0.0004 | 0.0011 |
| R3 without latent (344012) | 0.0365 | 0.0594 | 0.0761 | 0.1045 | 0.0498 | 0.0567 | 0.0658 |

Paired bootstraps: fused − fast HR@10 +0.0031 [−0.0022, +0.0082], NDCG@10 +0.0003 (n.s. at every cutoff); fused − no-latent R3 HR@10 −0.0019 [−0.0077, +0.0042] (n.s. everywhere); fused − fused_delete HR@10 +0.0245 [+0.0202, +0.0286] *; fast − fused_delete HR@10 +0.0214 *.

**Reading:** the slow head adds nothing to the fast head (fused ≈ fast ≈ no-latent R3). Deleting the latents hurts the fused model only because a broken slow head then corrupts the sum (fused_delete is below the fast head alone); that is dependence without benefit, not evidence that the latent helps. The slow head with shared weights (0.0705) is weaker than the standalone gist model (0.0911). Two-head fusion (PLR-style) does not make the latent useful on Toys. Results copied to `output_dir/two_head_results/Toys_and_Games_two_head_k17/`.

## CF complementarity: does SASRec carry information R3 misses? (offline, 2026-10-31)

Motivation: our results suggest the decoder uses the latent only when it carries information the visible history lacks; training-only signals (category head, supervision) leave r = f(x) at inference and cannot add any. The one inference-time external source that was used (SASRec vector in the slot, Musical) replaced the titles (0.1196 < 0.1226). Question: fusion failure, or nothing to add?

Script `scripts/cf_complementarity.py`; existing Toys SASRec (Verifiable-Rec Cornac export, `.../baselines/results/sasrec/Toys_and_Games_5_2017-1-2018-11/test_recommendations.csv`, rows aligned with the test CSV, history items excluded). Report `output_dir/cf_complementarity/Toys_report.md`. Note: SASRec scores HR@10 0.0883 on all 6,738 rows (its metrics.json 0.1159 is Cornac's own protocol, not comparable). Copy rows = target first word in a history title (2,144); other rows 4,594. @20 = @10.

| rows | model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| all | SASRec | 0.0328 | 0.0545 | 0.0668 | 0.0883 | 0.0576 |
| all | R3 | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0659 |
| all | T2 | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0678 |
| copy | SASRec | 0.0914 | 0.1446 | 0.1758 | 0.2267 | 0.1525 |
| copy | R3 | 0.1119 | 0.1772 | 0.2267 | 0.2948 | 0.1919 |
| other | SASRec | 0.0054 | 0.0124 | 0.0159 | **0.0237** | 0.0133 |
| other | R3 | 0.0030 | 0.0063 | 0.0083 | 0.0124 | 0.0071 |
| other | T2 | 0.0041 | 0.0067 | 0.0094 | 0.0152 | 0.0086 |

- Other rows: SASRec − R3 HR@10 +0.0113 [+0.0067, +0.0163] (SASRec ≈ 2× R3); hits SASRec-only 88, R3-only 36, both 21.
- All rows: SASRec-only 233, R3-only 327, both 362 (overlap only 39% of the union). Oracle best-of-both HR@10 0.1368 (+0.0346 [+0.0304, +0.0390] over R3); even on copy rows SASRec-only 145.
- Training-free RRF baseline (1/(rank) each list, equal weight, not tuned): R3+SASRec HR@10 0.1112 (+0.0089 [+0.0045, +0.0135]), NDCG@10 0.0727 (+0.0068 [+0.0045, +0.0091]); T2+SASRec 0.1107 (+0.0043 [−0.0003, +0.0089]), NDCG +0.0043 [+0.0021, +0.0067].

**Conclusion:** the CF signal is genuinely complementary (it is the better model exactly where the LLM fails, and naive late fusion already gains). So the Musical SASRec-slot result (used but substitutive) was a fusion failure, not absence of information. A latent carrying SASRec information, added so it complements the text path, can be both relied on and improve ranking; the RRF number is the baseline it must beat.

### Candidate latents as extra prompt tokens: rejected before launch (2026-10-31)

Design drafted (SASRec top-10 candidates, out of fold, as 10 placeholder tokens after the history, fine-tuned from the R3 Toys checkpoint); SASRec export job 345649 cancelled, code reverted, nothing committed. **User constraint: the prompt must not change; only the latent reasoning (the thought) may be updated/refined.** External information must therefore enter through the R3 latent itself.

### Candidate–target cosine similarity (offline, 2026-10-31)

Before designing "r attends to SASRec candidates": how close are SASRec's top-10 to the target when they miss? `scripts/candidate_target_similarity.py`, report `output_dir/cf_complementarity/Toys_candidate_target_similarity.md`. Max cosine over the list (Qwen3 title space / SASRec item space), rows where the target is not in the list:

| list | Qwen3 max, near-miss copy | Qwen3 max, near-miss other | Qwen3 max ≥ 0.8, other | SASRec-space max, other |
|---|---:|---:|---:|---:|
| SASRec top-10 | 0.806 | 0.735 | 0.165 | 0.188 |
| R3 top-10 | 0.829 | 0.728 | 0.146 | 0.196 |
| last history item | 0.794 | 0.659 | 0.053 | 0.059 |
| random 10 items | 0.728 | 0.721 | 0.086 | 0.176 |

SASRec − R3 max-cos (Qwen3, other rows) +0.0099 [+0.0078, +0.0118]; copy rows −0.0292. **When SASRec misses, its candidates are barely closer to the target than 10 random items** (other rows 0.735 vs 0.721 title space; 0.188 vs 0.176 SASRec space). SASRec's complementarity is binary (exact hits: 233 SASRec-only rows = 3.5%), not graded closeness; a latent built from its candidates helps only if r can pick out the right one.

### Target rank in SASRec's full dot-product ranking (offline, 2026-10-31)

SASRec (Cornac) scores items by h_u · e_i (shared input/output item table, no MLP, no normalisation, no item bias in our checkpoint; item-embedding norms 0.43–1.07). `scripts/sasrec_target_rank.py`, report `output_dir/cf_complementarity/Toys_sasrec_target_rank.md`; history items excluded as in the export (rank ≤ 10 rows 595 = export hits 595). Catalog 6,269; 312 targets outside it, 66 in history.

| rows | n | rank 1 | 2–10 | 11–50 | 51–100 | 101–500 | >1000 | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 2,004 | 0.098 | 0.145 | 0.121 | 0.076 | 0.170 | 0.293 | 188 |
| other | 4,356 | 0.006 | 0.019 | 0.033 | 0.037 | 0.139 | 0.653 | 1,840 |
| miss@10, other | 4,247 | — | — | 0.034 | 0.038 | 0.143 | 0.670 | 1,924 |

Random ranking median ≈ 3,134. On other rows SASRec's misses are not near misses: two thirds of targets rank beyond 1,000, only 3.4% in 11–50. Consistent with the cosine check: SASRec's signal on the hard rows is a few exact hits, not a neighbourhood around the target.

### Interaction-only signal screen (offline, 2026-10-31)

User constraint: external information must come from user-item interactions only (no item text/metadata: not fair, not a contribution). User reports LightGCN and BPR rank worse than SASRec. Split is global-temporal (train targets ≤ 2018-02-08 ≤ valid ≤ 2018-04-15 ≤ test ≤ 2018-09-28), so train+valid interactions predate every test target; R3/SASRec knowledge is 2–7 months stale at test time. `scripts/interaction_signal_screen.py`, report `output_dir/cf_complementarity/Toys_interaction_signal_screen.md`. History items excluded; recent popularity counts interactions strictly before the row timestamp (past + earlier test-row histories; day-granular timestamps, so same-day excluded).

| signal | HR@10 all | copy | **other** | HR@50 other | only-this hits vs R3 (other) | best-of-both other |
|---|---:|---:|---:|---:|---:|---:|
| R3 | 0.1014 | 0.2924 | 0.0122 | – | – | – |
| pop_past | 0.0144 | 0.0065 | 0.0181 | 0.0644 | 78 | 0.0292 |
| pop_recent_7d | 0.0237 | 0.0159 | 0.0274 | 0.0498 | 123 | 0.0390 |
| **pop_recent_30d** | 0.0310 | 0.0182 | **0.0370** | 0.0842 | **167** | **0.0485** |
| itemknn (sequential) | 0.0816 | 0.2243 | 0.0150 | 0.0492 | 57 | 0.0246 |
| userknn | 0.1008 | 0.2598 | 0.0266 | 0.0614 | 96 | 0.0331 |
| SASRec | 0.0883 | 0.2267 | 0.0237 | – | 88 | 0.0313 |
| LightGCN | 0.0242 | 0.0364 | 0.0185 | – | 79 | 0.0294 |
| BPR | 0.0128 | 0.0079 | 0.0150 | – | 65 | 0.0263 |

Other rows − R3 HR@10: pop_recent_30d +0.0248 [+0.0187, +0.0311]; pop_recent_7d +0.0152; userknn +0.0144 [+0.0096, +0.0192]; SASRec +0.0115; LightGCN +0.0063; pop_past +0.0059; itemknn / BPR n.s. (R3 here 0.1014 vs 0.1023 elsewhere: duplicate titles mapped to the first item id.) **Strongest complementary signal on the hard rows is time: what was popular in the 30 days before the purchase (3× R3, 1.6× SASRec), which neither the LLM nor any model trained on the train split can know.** userknn ≈ R3 overall and 2× R3 on other rows.

**Can a trained model capture the recent-popularity signal? (2026-10-31)** Of the 209 test rows hit by pop_recent_30d, 54.5% have a target with **zero training-split interactions** (first seen after training ended; median 50 days since first appearance vs 487 for all targets); 34.9% are items with ≥ 20 training interactions. Over all test targets only 4.6% are new. So most of the popularity gain is temporal drift / new items that no model trained on the train split (SASRec, LightGCN, R3) can score; it needs test-time interaction data, not a better model. User decision: popularity-awareness is a separate, engineering-type direction; **not pursued.**

**HR@10 by the target's training-split interaction count (2026-10-31).** New targets (0 count) are only 312 rows (4.6%): every model ≈ 0 (R3 1 hit). Other (non-copy) rows: count 1–4 (n = 880) R3 0.0011 / T2 0.0023 / SASRec 0.0136; 5–19 (n = 2,316) 0.0112 / 0.0142 / 0.0263; 20+ (n = 1,160) 0.0259 / 0.0302 / 0.0310. SASRec's advantage on the hard rows is on **long-tail items seen in training** (rare: 12×, mid: 2.3×, popular: ≈ equal), i.e. learnable collaborative signal, not temporal drift; R3 almost never recommends rarely-seen items without a lexical cue.

## SASRec-informed pooling of the R3 latent (launched 2026-10-31)

Constraints (user): prompt unchanged, only the latent changes; external information from interactions only. Motivation: SASRec beats R3 on non-copy rows mostly on long-tail items seen in training (above). Design (`src/latent/candidate_pooling.py`): R3's pooling reads the row's SASRec top-10 candidates (out of fold for training rows, 5 folds, last-position CE, history excluded; `sasrec_candidates.py`) in a separate branch, `r = r_R3 + g · softmax_j(τ·standardise_j(q·Wk c_j/√d) + α·logp_j) · Wv c_j`, with R3's own Wq/Wk/Wv (unrotated); g, τ, α fp32 init 1. Two candidate representations (user choice: run both): **title** = R3's last-layer hidden states averaged over the quoted title (`candidate_title_vectors.py`), **sasrec** = frozen SASRec item embedding → fp32 Linear(64→1536), rescaled to the mean title-vector norm. Fine-tuned from the R3 Toys checkpoint (lr 1e-4, ≤ 2 epochs, early stopping, batch 128). Evaluation: raw / shuffle (another user's candidates) / drop (branch removed = exactly R3), per-row candidate weights saved.

Design fix before launch: a first version put candidates in R3's pooling softmax; CPU check showed zero mass and zero gradient because **R3's pooling puts a logit of ~100–360 on one template token (' Input', position 60; next ~30)**, so the separate branch is required. CPU check (code 7fec718, real R3 checkpoint, fake candidate files): prompt token-identical to R3; drop loss = plain R3 loss exactly (0.56657); initial raw loss 0.5680 (title) / 0.5674 (sasrec); swap changes loss; gradients reach all fp32 params; fp32 save/load round trip OK.

Jobs: prep 345685 (code df5911c; prep scripts unchanged since) → smoke 345687 (title, 512 train / 64 valid / 64 test rows) → full title / sasrec (code 7fec718, auto-submitted when queue slots free). Outputs: prep `output_dir/candidate_pooling/prep/`, runs scratch `Toys_and_Games_cand_pool_{title,sasrec}_c10`, results `output_dir/candidate_pooling/<name>/`. Compare: R3 0.1023 / 0.0659, rank fusion R3+SASRec 0.1112 / 0.0727. Note: no shuffled-candidate training control yet (user asked for the two variants).

**Prep 345685 done (17 min).** Our SASRec (d64, 2 layers, last-position full-softmax CE, history excluded): recall@1/5/10/20 train (out of fold) 0.047 / 0.092 / 0.113 / 0.140, valid 0.035 / 0.072 / 0.092 / 0.114, **test 0.038 / 0.080 / 0.102 / 0.126** (Cornac export 0.088 @10; R3 0.1023). Out-of-fold train candidates are only slightly better than test (0.113 vs 0.102 @10), so the trust signal is roughly calibrated. Fold valid recall@10 0.081–0.086 vs full model 0.092. Title vectors: 6,299 titles, mean norm 64.9. Smoke 345687 running; full runs title 345702 / sasrec 345703 queued after it.

**Smoke 345687 failed** (config bug, not code): VALID_SAMPLE=64 truncated the valid file below the 512-row sample the train script also applies to the eval set ("Cannot take a larger sample than population"); full runs 345702/345703 stuck on the dependency, cancelled. Resubmitted: smoke **345718** (no VALID_SAMPLE) → title **345719**, sasrec **345720** (same code 7fec718). Reported by the forked session, which is running a from-base-model study A–D in worktree R3GuidanceScaffold (plain R3 retrain / + real SASRec candidates / + random candidates / gate annealed to 0) reusing our prep files read-only; our fine-tune-from-R3 runs overlap with its B (user to decide). Note: OOF train recall@10 0.113 > test 0.102 may partly be same-user information across row-level folds (a user's other rows in the training folds), besides train rows being in-period.

## Guidance scaffold study A–D (forked session, launched 2026-10-31)

Context: IntuRec (arXiv 2606.27684) cross-attends the initial latent to candidate items (≈ our candidate branch). Its CDs ablation (IntuRec-B): no intuition R@10 0.1015 / N@10 0.0731, random intuition 0.1156 / 0.0835, real 0.1221 / 0.0871 → random candidates keep 68% (R@10) / 74% (N@10) of the gain; real − random +0.0065 R@10 (single runs, no CIs). Toys main table: IntuRec-B 0.1077 vs LatentR3-B 0.1060 R@10. IntuRec stage 2 trains from base Qwen2.5-1.5B (lr 3e-4, batch 128). User hypothesis: candidates act mainly as **guidance** that helps the latent escape R3's degenerate optimum (pooling max weight 1.0 on ' Input', position 60), not as content.

Design (user-approved; from base Qwen2.5-1.5B, R3 recipe lr 3e-4, batch 128, ≤ 3 epochs, early stopping patience 1; Toys; candidate kind = projected SASRec embedding rescaled to the batch's mean hidden-state norm, so nothing depends on a trained R3):

| run | candidates in training | eval modes |
|---|---|---|
| A | none (plain R3 retrain) | raw |
| B | SASRec top-10 (out of fold, prep 345685) | raw, shuffle, drop |
| C | 10 random non-history items per row (fixed, constant logp) | raw, real, drop |
| D | as B, branch scale 1 → 0 linearly over training fraction [1/3, 2/3], 0 after; validation loss and raw eval at s = 0 (plain R3 inference) | raw, guided (s = 1) |

Predictions: guidance hypothesis → D ≈ B > A, C ≈ B, drop/shuffle barely hurt B; content hypothesis → B > C ≈ A, shuffle/drop hurt B. Diagnostics per evaluation (`pool_diagnostics.jsonl`): R3 pooling max weight / entropy / argmax positions, candidate-weight max / entropy, gate, τ, α, branch scale.

Code: worktree `.claude/worktrees/R3GuidanceScaffold` (branch from 7fec718), commit 64e2040, frozen `output_dir/guidance_scaffold_code/code_64e2040`; `scripts/guidance_scaffold.sbatch` (RUN=A|B|C|D). CPU check (R3 checkpoint, real prep candidates): random candidates non-history / deterministic / prompt unchanged; anneal schedule values correct; annealed model at eval (s = 0) loss = plain R3 exactly (0.56657); guided s = 1 0.56666; gradients reach branch at s = 0.5; diagnostics run for candidate model and plain R3 (R3 pooling max weight 1.0 at position 60). Prep caveat: OOF train candidate recall@10 0.1125 > test 0.1021 (valid 0.092), likely same-user leakage across row-level folds. Jobs: smokes C/D (512 train / 64 valid / 64 test rows, 1 epoch) → A, B, C, D (submitted by a retry loop while the 16-job limit is full). Outputs: scratch `Toys_and_Games_guidance_{A,B,C,D}`, results `output_dir/guidance_scaffold/<name>/`. References: R3 0.1023 / 0.0659 (original checkpoint), T2 0.1064 / 0.0678.

### Option A final (job 343950, done 2026-10-31) and early sqrt-JS training

**Data loss:** the scratch dirs `Toys_and_Games_r3_plain_matched` and `Toys_and_Games_thoughts_constant_k1` were deleted by something outside this session (scratch near quota); the plain-matched per-row JSON had not been copied (its metrics survive above and in `run_logs/R3_plain_matched.343065.out`). T2 per-row results copied from `scratch/.../continuous_thoughts_results/constant_k1` to `output_dir/divreason_results/references_T2_constant_k1/`. Option-A results copied to `output_dir/divreason_results/Toys_and_Games_divreason_n4_pathdiv_w1/`.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| option A raw (JS hinge, weight 1.0) | 0.0347 | 0.0586 | 0.0738 | 0.1024 | 0.0484 | 0.0547 | 0.0640 |
| r_agg = mean | 0.0346 | 0.0582 | 0.0745 | 0.1017 | 0.0480 | 0.0548 | 0.0635 |
| r_agg shuffled | 0.0344 | 0.0589 | 0.0742 | 0.1028 | 0.0484 | 0.0547 | 0.0640 |
| single unperturbed path | 0.0349 | 0.0586 | 0.0745 | 0.1031 | 0.0485 | 0.0550 | 0.0643 |
| path 0 alone | 0.0347 | 0.0588 | 0.0739 | 0.1015 | 0.0484 | 0.0547 | 0.0636 |
| paper DivReason | 0.0378 | 0.0620 | 0.0803 | 0.1079 | 0.0518 | 0.0592 | 0.0682 |
| T2 constant token | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 |

Interventions all n.s. (raw − mean +0.0007, − shuffle −0.0004, − single −0.0007, − path0 +0.0009 HR@10). raw − paper DivReason HR@10 **−0.0055 [−0.0098, −0.0012]**, NDCG@10 **−0.0042 [−0.0066, −0.0018]**; − T2 −0.0040 [−0.0083, +0.0001] / −0.0038 [−0.0063, −0.0016]. The JS-hinge run costs accuracy and leaves the latent unused.

**sqrt-JS run 345579 (started on lagoon), steps 10–110:** path JS 0.003 → 0.0001 (step 40) → **0.013–0.027 (steps 50–110)**, JS distance 0.08–0.10 (margin 0.45), hinge 0.35–0.44. 100–1000× the JS level of the JS-hinge runs (1e-5): the paths' outputs now separate.

**Guidance A–D resubmission (2026-10-31).** Smoke D 345721 failed (fire parses `--candidate_anneal "0.3333,0.6667"` as a tuple); A 345782 / B 345784 cancelled (DependencyNeverSatisfied). Smoke C 345781 passed (train, diagnostics, raw/real/drop evals; base-model pooling at the first evaluation is not collapsed: max weight 0.86, varied argmax positions). Fix in commit 52cf195 (frozen `output_dir/guidance_scaffold_code/code_52cf195`). Resubmitted: smoke D 345856, A 345857, B/C next (no dependency), D after smoke D (IDs filled by the retry loop).

### sqrt-JS run 345579: preempted at step 258 (2026-10-31 18:49), requeued

Preempted on lagoon after 2 h 43 min (step 258/1263 of epoch 1, before the first checkpoint, so it restarts from scratch). Requeued automatically; Slurm start estimate 2026-11-01 23:25. Steps 10–250 saved to `output_dir/divreason_results/Toys_and_Games_divreason_n4_pathdiv_sqrtjs/attempt1_preempted_train_steps.jsonl`: path JS kept rising, 0.013 (step 100) → 0.03–0.044 (steps 140–250), JS distance 0.13–0.17 (margin 0.45); per-path NLL stays normal (0.4–2.8, noisy per batch).

### sqrt-JS run 345579, attempt 2 (analog, from 2026-10-31 23:47): epoch-1 diagnostics

Eval loss 0.794 (plain matched epoch 1: 0.748). On 64 validation rows:
- **Output now depends on the latent**: JS same user / other path **0.058**, JS other user / same path **0.054** (JS-hinge runs: 2e-5 / 3e-5; ~2,000×).
- **But not in a target-informative way**: ΔNLL other user / same path +0.003, same user / other path +0.006; r_agg shuffle ΔNLL −0.002, **mean −0.029 (the mean latent is better)**, single unperturbed path +0.075 (decoder adapted to perturbed paths). Path cosine 0.78, router max weight 0.46.
- Reading: the decoder's distribution moves with the latent, about equally for path identity and user identity, while the target's likelihood does not — the extra output variation sits on non-target mass.

### sqrt-JS run 345579 final (done 2026-11-01 16:52) — verdict on the path-diversity hypothesis

Epochs 1/2/3: eval loss 0.794 / **0.610** / 0.628 (best epoch 2; plain matched 0.595). Diagnostics (64 val rows): JS same-user-other-path 0.058 / 0.044 / 0.054, JS other-user-same-path 0.054 / 0.038 / 0.059; ΔNLL other-user-same-path +0.003 / +0.027 / −0.020; r_agg shuffle ΔNLL −0.002 / +0.0007 / −0.004, mean −0.029 / +0.0007 / −0.005.

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | top-10 overlap with raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sqrt-JS raw (r_agg) | 0.0347 | 0.0577 | 0.0720 | 0.1012 | 0.0480 | 0.0539 | 0.0633 | — |
| r_agg = mean | 0.0343 | 0.0579 | 0.0724 | 0.1020 | 0.0479 | 0.0538 | 0.0634 | 0.939 |
| r_agg shuffled | 0.0343 | 0.0577 | 0.0727 | 0.1024 | 0.0478 | 0.0540 | 0.0635 | 0.934 |
| single unperturbed path | 0.0350 | 0.0582 | 0.0717 | 0.1014 | 0.0483 | 0.0538 | 0.0634 | 0.930 |
| path 0 alone | 0.0337 | 0.0546 | 0.0683 | 0.0963 | 0.0457 | 0.0513 | 0.0603 | **0.705** (same top-1 0.61) |
| paper DivReason (ref.) | 0.0378 | 0.0620 | 0.0803 | 0.1079 | 0.0518 | 0.0592 | 0.0682 | (its mean/shuffle/single: 0.939 / 0.934 / 0.944) |
| T2 constant token (ref.) | 0.0375 | 0.0634 | 0.0790 | 0.1064 | 0.0525 | 0.0589 | 0.0678 | |

raw − mean −0.0007, − shuffle −0.0012 [−0.0027, +0.0001], − single −0.0001 (all n.s.); raw − path0 **+0.0049 [+0.0016, +0.0083]**; raw − paper DivReason **−0.0067 [−0.0113, −0.0022]** / NDCG −0.0049; − T2 **−0.0052 [−0.0098, −0.0009]** / −0.0045; − option A −0.0012 (n.s.).

**Verdict.** Forcing the paths' outputs apart (with a loss that has gradient at agreement) makes the decoder **sensitive to individual paths** (one path alone changes 30% of the top-10 and 39% of the top-1, and hurts), but **not to the user content of the latent it is actually given**: with r_agg, replacing it by the mean or another user's changes the lists no more than in paper DivReason (93–94% overlap) and does not hurt. Aggregation averages the path-level differences away, and the differences are perturbation-driven, not user-informative. Ranking pays for it (−0.005 HR@10 vs T2). The hypothesis holds only in the weak sense (sensitivity), not in the sense of content use; consistent with H1.

### SASRec-informed pooling results (345719 title / 345720 sasrec, done 2026-10-31 22:25)

Both stopped at epoch 1 (eval loss 0.6405 → 0.8226 title, 0.6426 → 0.8268 sasrec at epoch 2; best = checkpoint-422): fine-tuning the trained R3 overfits after one epoch. g, τ, α stayed at 1.00 (moved < 0.002). n = 6,738, @20 = @10.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3 (checkpoint fine-tuned from) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |
| title raw | 0.0359 | 0.0592 | 0.0742 | 0.1030 | 0.0495 | 0.0557 | 0.0650 |
| title shuffle | 0.0355 | 0.0592 | 0.0736 | 0.1021 | 0.0493 | 0.0552 | 0.0644 |
| title drop | 0.0356 | 0.0604 | 0.0745 | 0.1051 | 0.0501 | 0.0558 | 0.0656 |
| sasrec raw | 0.0347 | 0.0597 | 0.0769 | 0.1042 | 0.0492 | 0.0562 | 0.0649 |
| sasrec shuffle | 0.0352 | 0.0592 | 0.0754 | 0.1026 | 0.0491 | 0.0557 | 0.0644 |
| sasrec drop | 0.0356 | 0.0595 | 0.0758 | 0.1023 | 0.0494 | 0.0561 | 0.0645 |

- vs R3 HR@10: title +0.0007 [−0.0046, +0.0061], sasrec +0.0019 [−0.0033, +0.0076] (n.s.); both below rank fusion 0.1112.
- Reliance: sasrec raw − drop +0.0019 [+0.0006, +0.0034], raw − shuffle +0.0016 [+0.0001, +0.0031] (significant but tiny; NDCG n.s.); title raw − drop **−0.0021** [−0.0042, −0.0001] (removing candidates helps), raw − shuffle +0.0009 n.s.
- Subsets (HR@10): rows whose SASRec top-10 contains the target (n = 688): R3 0.570, title raw 0.606 / drop 0.622, sasrec raw 0.621 / drop 0.615; other rows R3 0.012, all fine-tuned 0.015–0.017. The gains over R3 appear **with candidates dropped too**, so they come from the extra fine-tuning epoch, not from candidate content at inference.
- Pooling did select: candidate weights entropy 1.74 (uniform 2.30); weight on the target when present 0.26 vs SASRec softmax share 0.17; argmax = target 32% (SASRec top-1 37%). So r_cand carries the candidate, but **the decoder still does not use it** (same insensitivity as before; with the history visible, the decoder keeps reading titles).

Conclusion: routing complementary SASRec candidates into r of an already-trained R3 does not create reliance or gain; the information reaches r (selection works) but the decoder ignores r. Results copied to `output_dir/candidate_pooling/Toys_and_Games_cand_pool_{title,sasrec}_c10/`.

### Guidance scaffold A–C results (2026-11-01); D resubmitted

A 345857, B 345883, C 346494 done (code 52cf195; 3 epochs, best = epoch 2 for all). D was never submitted (retry loop died with the previous session); submitted 2026-11-01 as **348523**. Test n = 6,738, @20 = @10.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A raw (plain R3 retrain) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |
| B raw (SASRec candidates) | 0.0374 | 0.0616 | 0.0769 | 0.1033 | 0.0513 | 0.0576 | 0.0662 |
| B shuffle | 0.0361 | 0.0598 | 0.0770 | 0.1023 | 0.0499 | 0.0570 | 0.0652 |
| B drop | 0.0375 | 0.0622 | 0.0766 | 0.1046 | 0.0517 | 0.0576 | 0.0667 |
| C raw (random candidates) | 0.0356 | 0.0595 | 0.0767 | 0.1043 | 0.0495 | 0.0565 | 0.0655 |
| C real (SASRec at test) | 0.0355 | 0.0591 | 0.0760 | 0.1026 | 0.0491 | 0.0560 | 0.0646 |
| C drop | 0.0358 | 0.0591 | 0.0781 | 0.1042 | 0.0493 | 0.0571 | 0.0655 |
| R3 checkpoint (original) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |

HR@10 / NDCG@10 differences: B − A −0.0019 [−0.0059, +0.0022] / −0.0006; C − A −0.0009 / −0.0013; B − C −0.0010 / +0.0007; B raw − shuffle +0.0010 [−0.0010, +0.0031] / +0.0010 [−0.0001, +0.0022]; B raw − drop −0.0013 / −0.0005; C raw − drop +0.0001; C real − C raw −0.0018 [−0.0036, −0.0001] (sig, small); A − original R3 +0.0030 (n.s., run-to-run noise). Eval loss (best, epoch 2): A 0.5935, B 0.5907, C 0.5943.

Pooling diagnostics (64 valid rows, per epoch): **A collapses** (R3 pooling max weight 0.995 → 0.999, entropy 0.02 → 0.007, argmax positions {0, 3}); **B and C do not** (B max weight 0.56 / 0.59 / 0.78, entropy 1.85 / 1.33 / 0.57, argmax spread over many prompt positions; C 0.74 / 0.72 / 0.79). Candidate branch stays on (gate ≈ 1.0–1.05; B candidate max weight ≈ 0.40–0.43, entropy ≈ 1.7; C ≈ 0.31–0.33, entropy ≈ 1.92 ≈ uniform).

**Conclusion:** the candidate branch, real or random, prevents the template collapse of R3's pooling (the user's "escape the degenerate optimum" mechanism is real at the latent level), but it changes nothing downstream: B ≈ C ≈ A on every metric, and shuffling or dropping B's candidates at test time has no significant effect. The decoder ignores r's content whether or not the latent escapes collapse, so the bottleneck is the decoder's redundancy with the history (H1), not the latent's optimum. IntuRec's real-vs-random gain (+0.0065 R@10 on CDs, single runs) does not replicate here on Toys (their own Toys gain over LatentR3 was +0.0017 R@10). Same conclusion as the original session's fine-tune-from-R3 candidate-pooling runs (no gain; pooling selects the target but the decoder ignores r). D (anneal) pending.

**Decision (user, 2026-11-01): the SASRec-candidate direction is closed** (not promising). Code (7fec718), prep files and results are kept for reference; no further candidate runs from this session. The forked session's from-base study A–D (R3GuidanceScaffold) is separate and unaffected.

## Does the oracle model still use the history? (2026-11-02, analysis only)

Question (user): in the side-info oracle training (Musical, `oracle_raw`: target-category embedding added to `r`; decoder_insensitivity_to_latent.md Exp. 1), does the model rely on the oracle only, or still read the history? The checkpoints were in the purged scratch, so no new interventions; only saved summaries plus one new baseline (`scripts/oracle_category_popularity.py`, commit after 7fec718).

| | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oracle_raw` model (category in `r` + history) | 0.2187 | 0.3550 | 0.4469 | 0.5922 | 0.2970 | 0.3348 | 0.3816 |
| `text_control` (category in prompt + history) | — | — | — | 0.62 | — | — | — |
| **category popularity, no history** (top-10 training targets in the true category) | 0.1765 | 0.3541 | 0.4693 | **0.6124** | 0.2780 | 0.3254 | 0.3715 |
| same, history items excluded | 0.1866 | 0.3580 | 0.4678 | 0.6086 | 0.2851 | 0.3303 | 0.3757 |

(@20 = @10 for the models, 10 beams. No per-row oracle predictions survive, so no paired bootstrap.)

Evidence:
- A history-free rule given the same oracle category matches the oracle model at HR@5/10 (0.612 vs 0.592); the model is better only at the very top (HR@1 +0.042, NDCG@10 +0.010).
- Swap test (first 1,000 rows): own history + donor's category → the donor's own target is in the top-10 for 46.9% of rows, the row's own target 2.6%. The history cannot pull the list back toward the user's own item.
- Teacher-forced KL: swapping the category moves the output 13.6× as much as dropping the last history item; dropping a history item still moves it (KL 0.061 vs plain R3 0.073), so the history is not literally ignored at the token level.

Reading: the oracle model relies mainly on the oracle, using history only for small within-category reordering (top-1). Mirror image of plain R3: whichever source suffices takes over and the other becomes nearly redundant. A decisive check (delete / shuffle the history with the oracle fixed) needs retraining `oracle_raw` (checkpoint purged).

### Category prediction: R3 coverage by cutoff, and closeness of missed predictions (2026-11-02)

Target category among the categories of R3's top-k items (n = 6,481): @1 0.1657, @3 0.2638, @5 0.3169, @10 0.3790 (T2 0.1629 / 0.2705 / 0.3208 / 0.3811) vs LLM category model HR@1/3/5/10 0.1981 / 0.3331 / 0.3987 / 0.5098.

User hypothesis: a wrong predicted category may still be semantically close to the target. Script `$CLAUDE_JOB_DIR/tmp/category_closeness.py` (gte-large-en-v1.5 CLS embeddings of the 194 level-3 paths without the root; mean pairwise cosine 0.599; uniform-random category cosine 0.610 / same level-2 parent 0.059). On miss rows, with a popularity-weighted random baseline drawn for the same rows:

| misses of | n | cosine to target | pop-random | same level-2 parent | pop-random |
|---|---:|---:|---:|---:|---:|
| LLM top-1 | 5,197 | 0.644 | 0.630 | 0.105 | 0.072 |
| LLM top-5 (best of 5) | 3,897 | 0.716 | 0.738 | 0.234 | 0.306 |
| last history item's category | 5,040 | 0.657 | 0.631 | 0.142 | 0.072 |

**Wrong predictions are barely closer than a random popular category** (top-1 +0.014 cosine, +3.3 pt same parent); on top-5 misses the 5 predictions are *less* close than 5 popularity-random categories (they cluster together; the miss rows are the off-pattern rows). The last item's category, on its own misses, is slightly closer than the LLM's. Hypothesis not supported: a missed category prediction carries little semantic signal about the target.

**All category predictors (2026-11-02, `$CLAUDE_JOB_DIR/tmp/category_closeness_all.py`).** Miss-row closeness, Δ vs k popularity-random categories on the same rows (cosine / same level-2 parent): LLM SFT top-1 +0.014 / +0.033, top-5 −0.019 / −0.060; SASRec IDs top-1 +0.023 / +0.033, top-5 −0.017 / −0.058; SASRec IDs+cats top-1 +0.025 / +0.036, top-5 −0.018 / −0.059; last item's category +0.027 / +0.073. Same picture for every model: missed top-1 barely closer than chance, missed top-5 less close than random; the trivial last-item category is the closest when wrong.

### Guidance scaffold D result (2026-11-02) — study closed

D 348523 (code 52cf195; candidate branch annealed 1 → 0 over [1/3, 2/3] of training; validation and raw eval at s = 0). n = 6,738, @20 = @10.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D raw (plain-R3 inference) | 0.0364 | 0.0600 | 0.0761 | 0.1023 | 0.0500 | 0.0566 | 0.0651 |
| D guided (branch on, s = 1) | 0.0380 | 0.0616 | 0.0778 | 0.1036 | 0.0515 | 0.0581 | 0.0665 |

D raw − A HR@10 −0.0030 [−0.0070, +0.0013], NDCG@10 −0.0016 [−0.0038, +0.0007]; D guided − A −0.0016 (n.s.); D raw − B −0.0010 (n.s.); raw − guided NDCG@10 −0.0014 [−0.0025, −0.0002] (sig, small). Pooling stays un-collapsed even after the branch is removed (max weight 0.77 / 0.80 / 0.86 by epoch). **Guidance-scaffold hypothesis not supported**: the scaffold changes the latent's optimum (no collapse, persists after removal) but gives no ranking gain; A ≈ B ≈ C ≈ D.

## Category-conditioned latents (launched 2026-11-02)

User idea: the category need not be predicted perfectly; a ~20–25%-correct category in the latent may still lift item prediction ~5% (≈ +0.005 HR@10 ≈ 35 hits; 904 rows have the right LLM top-1 category but an R3 item miss@10, 295 R3 hits sit on wrong-category rows). The oracle category on Musical was strongly used (HR@10 0.65 vs 0.03 swapped), so trust is the issue, not readability. Step 1 (user-approved): train with the **oracle category**, evaluate with predicted categories. Two injections (prompt unchanged; from base Qwen2.5-1.5B, R3 recipe):
- **additive**: SideInfoLatentModel kind oracle_category with a category-indexed table (gte-large-en-v1.5 embeddings of the 194 paths), r' = r + g·normalize(P(e_c)), fp32 branch; eval category_mode oracle / shuffle / pred1 / mix5 (5 beam searches, one per LLM-predicted category, merged by category score + item score).
- **inject (IntuRec-style)** (`src/latent/category_latent.py`, InjectNet ported from IntuRec/model/IntuRecF2.py): K = 5 category path token sequences, per-category self-attention pooling, cross-attention with the last-layer state at the thought position as query; result replaces R3's r. Training list = target + 4 out-of-fold predicted (user choice), random order; next-token loss only (user choice; no L_intu). Eval: pred (LLM top-5) / oracle (target inserted) / shuffle.

Prep 349106 (`src/latent/category_prep.py`, `output_dir/category_latent/prep/Toys_category_prep.pt`): SASRec category (IDs + history categories) 5-fold OOF top-5 for train (top-1 0.2353 / top-5 0.4267, above valid 0.1898 / 0.361: same row-level-fold optimism as before), valid lists from the all-rows model, test lists = LLM SFT top-5 (0.1981 / 0.3987). CPU check (code 5025df9; R3 checkpoint, fake lists): prompt unchanged; additive oracle vs other category changes loss, fp32 gain survives save/load; inject order-invariant, uniform initial weights, gradients reach both attention stages, fp32 reload exact; all 7 eval modes run end to end (one dedup bug in the oracle list fixed). Jobs: smoke (inject, 512 rows) → additive / inject full (code 5025df9, auto-submitted). Results → `output_dir/category_latent/<name>/`.

**Category-latent jobs (submitted by JobManager, 2026-11-02):** smoke 349155 (inject, 512/64/64 rows, 1 epoch, modes pred oracle) → additive **349156**, inject **349157** (afterok:349155), code 5025df9. Scratch was at 502.9 / 536.9 GB (~34 GB free); each full run needs ≈ 9 GB (2 model-only checkpoints + final model).

**Category additive 349156, interim (2026-11-02):** training done (eval loss with oracle category 0.561 / **0.364** / 0.393 by epoch; plain R3 run A 0.741 / 0.594 / 0.643). Oracle-category test ranking (ceiling, not a deployable result): HR@1/3/5/10 0.1115 / 0.2050 / 0.2634 / **0.3776**, NDCG@3/5/10 0.1651 / 0.1892 / 0.2260 (R3 0.1023): the decoder uses the latent category strongly on Toys too. Job preempted during the pred eval and requeued (restarts at eval; train skipped). Inject 349157 epoch-1 eval loss 0.741 = run A (not yet using the oracle).
Additive 349156 preempted twice (aloha after training + oracle eval; toga during the eval-only restart). Cancelled; remaining evals split into one job per mode so a preemption costs one mode: pred1 **350064**, shuffle **350065**, mix5 **350066** (same frozen code 5025df9, training skipped since latent/config.json exists). eval_oracle.json already done.

**Category inject 349157 result (2026-11-02; preempted after pred + oracle evals, requeue cancelled, shuffle resubmitted as 350070).** HR@1/3/5/10, NDCG@3/5/10:
- inject pred (LLM top-5 list): 0.0359 / 0.0592 / 0.0760 / 0.1049, 0.0495 / 0.0564 / 0.0658
- inject oracle (target inserted): 0.0356 / 0.0597 / 0.0770 / 0.1061, 0.0497 / 0.0568 / 0.0662
pred − A HR@10 −0.0003 [−0.0046, +0.0039]; oracle − A +0.0009 [−0.0034, +0.0050]; HR@10 on top-1-right rows identical (0.2983) in both modes. **The IntuRec-style injector ignores the categories even when the true one is in the list** (epoch-1 eval loss = run A). With the target hidden among 4 plausible predicted categories, the cross-attention never learned to single it out, so the decoder learned to ignore the injected vector: the same redundancy trap. Additive with the oracle category: HR@10 0.3776 (+0.272 over A, uniform across top-1-right / wrong rows 0.385 / 0.383), i.e. a single unambiguous category is read directly. Pending: additive pred1 350064, shuffle 350065, mix5 350066; inject shuffle 350070.

**Category additive pred1 350064 (2026-11-02):** HR@1/3/5/10 0.0276 / 0.0512 / 0.0650 / **0.0914**, NDCG@3/5/10 0.0412 / 0.0469 / 0.0554. pred1 − R3 HR@10 −0.0108 [−0.0169, −0.0049], NDCG@10 −0.0105 (sig); − A −0.0138 / −0.0113 (sig). Split by LLM top-1 correctness (HR@10): right rows (1,284) 0.3847 vs R3 0.2960 (+0.089 ≈ +114 hits); wrong rows (5,197) **0.0227 vs R3 0.0568** (−0.034 ≈ −177 hits). **Over-trust**: the oracle-trained decoder follows the latent category whatever it is, so the gain on the 20% right rows is outweighed by the loss on the 80% wrong rows. Pending: mix5 350066 (marginalises over top-5), shuffle 350065, inject shuffle 350070. Next (step 2): train with predicted (out-of-fold) categories / soft distributions so the decoder learns calibrated trust.

**How the additive model relies on the latent category (2026-11-02, `$CLAUDE_JOB_DIR/tmp/category_reliance.py`).** Share of top-10 items in the latent's category: true-category eval 0.871, pred1 eval 0.938 (R3: 13.4% of its items fall in the target category); distinct categories per top-10: 1.87 / 1.48 vs R3 5.50. HR@10 copy / other rows: R3 0.2948 / 0.0124; true category 0.5275 / **0.3076**; pred1 0.2556 / 0.0148. On pred1 wrong-category rows (5,197): 93.9% of items in the injected wrong category, 1.0% in the target category (R3 5.4%). **Mechanism: the latent category acts as a near-hard filter (category picks the region, history still picks the item within it; brand copying survives inside the category).** Consequence: the model is a category-conditioned generator p(item | c, x), so the principled inference is the mixture Σ_c p(c|x) p(item|c,x) (mix5 pending) or training for soft trust.

**Does history still matter with the oracle category? (offline, 2026-11-02).** Level-3 categories are small (mean 147 items per target's category), so the true category alone is very informative. HR@10 (rows with a category, n = 6,481; all / copy / other): random 10 items of the true category 0.2221 / 0.1568 / 0.2531; 10 most popular items of the true category (past interactions, history excluded, **no history used**) 0.3402 / 0.2615 / 0.3775; additive model with the true category 0.3834 / **0.5240** / 0.3168. → History adds only +0.043 overall: large on copy rows (+0.26 over category popularity, brand copying), but on other rows the model is **below** the history-free category-popularity ranking (0.317 vs 0.378). Most of the 0.38 oracle ceiling is the category's own information. Direct test (oracle category + another user's history) not run yet.

**History-swap / latent-swap interventions (submitted 2026-11-02, code d187b04, `scripts/swap_history.py`, `scripts/eval_swap_history.sbatch`).** History swap: every test row gets another user's history (own target → own oracle category kept). Jobs: additive checkpoint, oracle category + swapped history **350270** (`eval_oracle_swaphist.json` in the additive scratch dir); plain R3 (original checkpoint; guidance run A's scratch dir has been deleted) with swapped history **350271** (`output_dir/category_latent/R3_swaphist/`). Latent swap (own history, another user's category) = additive shuffle mode, job 350065 (pending).

**Additive latent swap (shuffle) 350065:** own history + another user's true category: HR@1/3/5/10 0.0071 / 0.0114 / 0.0147 / **0.0218**, NDCG@3/5/10 0.0096 / 0.0110 / 0.0133. − oracle −0.3557 [−0.3669, −0.3443]; − R3 −0.0804 [−0.0876, −0.0738]; − pred1 −0.0696 (all sig). A random user's category is far worse than the predicted one (a plausible wrong category still overlaps the target region sometimes) and destroys ~80% of R3's own hits: the decoder's reliance on the latent is near-total.

**Inject shuffle 350070:** another user's predicted list: HR@10 0.1036 / NDCG@10 0.0653; − oracle −0.0025 [−0.0050, +0.0001], − pred −0.0013 (n.s.). The IntuRec-style injector is insensitive to its category list (oracle ≈ pred ≈ shuffle ≈ R3): confirmed it ignores the injected vector.

**Additive history swap 350270 (2026-11-03):** own true category + another user's history: HR@1/3/5/10 0.0555 / 0.1126 / 0.1558 / **0.2424**, NDCG@3/5/10 0.0882 / 0.1059 / 0.1338. − own-history oracle: HR@10 −0.1352 [−0.1456, −0.1253], NDCG@10 −0.0922 [−0.0990, −0.0859] (sig). Split (rows with a category; all / copy / other): own history 0.3834 / 0.5240 / 0.3168 → swapped 0.2495 / 0.1713 / 0.2866; history-free category popularity 0.3402 / 0.2615 / 0.3775. **Both inputs matter**: the history carries brand copying (copy rows 0.52 → 0.17) and some within-category ranking (other rows −0.03); a wrong history is worse than no history (below category popularity).

**Plain R3 history swap 350271:** original R3 checkpoint with another user's history: HR@10 **0.0022**, NDCG@10 0.0012 (R3 own history 0.1023). Reliance summary (HR@10): plain R3 — swap latent ≈ no effect (earlier interventions), swap history 0.1023 → 0.0022; additive category model — swap latent 0.3776 → 0.0218, swap history 0.3776 → 0.2424. Plain R3 relies only on history; the additive model relies on both, latent dominant.

**Additive mix5 350066 (2026-11-03) — step 1 complete.** Mixture over the LLM's top-5 categories (5 beam searches merged by category score + item score): HR@1/3/5/10 0.0294 / 0.0555 / 0.0693 / **0.0910**, NDCG@3/5/10 0.0446 / 0.0503 / 0.0573. − R3 HR@10 −0.0113 [−0.0169, −0.0059], NDCG@10 −0.0085 (sig); − pred1 −0.0004 / +0.0019 (n.s.). Split (HR@10): top-1-right rows 0.3045 (pred1 0.3847, R3 0.2960), top-1-wrong 0.0425 (pred1 0.0227, R3 0.0568); top-5-contains 0.2233 (R3 0.2187). Mixing trades the gain on right rows for less damage on wrong rows; net = pred1 < R3. The category scores are nearly flat (beam log-probs −0.17 to −0.23), so the mixture is close to uniform over 5 categories.

**Step 1 summary (Toys, additive category latent trained with the true category):** HR@10 oracle 0.3776, oracle + another user's history 0.2424, pred1 0.0914, mix5 0.0910, another user's category 0.0218; plain R3 0.1023, plain R3 + another user's history 0.0022. Inject (IntuRec-style): oracle 0.1061 ≈ pred 0.1049 ≈ shuffle 0.1036 (ignores categories). **Conclusion:** a category latent is used strongly and cooperatively with the history (category picks the region, history picks the brand/item), but a decoder trained on true categories over-trusts predicted ones: 80% wrong rows cost more than the 20% right rows gain, with top-1 or a 5-way mixture. Next (pending user): step 2, train with out-of-fold predicted categories (hard top-1 or soft top-5 distribution) for calibrated trust.

## Hypothesis check: does the latent bias generation toward popular items? (2026-11-03, analysis only, `scripts/latent_popularity_bias.py`, d991260)

Popularity = training interaction count (targets + histories), Toys test n = 6,738; head = top-10% items. Targets: mean log(1+pop) 3.557, head share 0.239. Popularity top-10 baseline HR@10 0.0110.

| run | HR@10 | mean log-pop of top-10 | head share | top-1% share | catalog coverage |
|---|---:|---:|---:|---:|---:|
| R3 (latent) | 0.1023 | 4.090 | 0.338 | 0.076 | 0.575 |
| no-latent SFT (same recipe) | 0.1045 | 4.143 | 0.367 | 0.077 | 0.548 |
| T2 constant token raw / mask | 0.1064 / 0.1075 | 4.116 / 4.116 | 0.340 / 0.340 | 0.083 / 0.083 | 0.582 / 0.584 |
| cand-pool sasrec raw / drop | 0.1042 / 0.1023 | 4.050 / 4.050 | 0.334 / 0.333 | 0.059 / 0.059 | 0.717 / 0.717 |
| T3 constant token, no history (latent only) | 0.0033 | 4.227 | 0.205 | 0.100 | 0.002 (14 items) |
| L7 K=1 raw / mean r | 0.0801 / 0.0043 | 4.346 / 4.217 | 0.431 / 0.312 | 0.099 / 0.001 | 0.373 / 0.003 (19 items) |

Paired bootstrap Δ mean log-pop: R3 − no-latent −0.053 [−0.062, −0.046]; T2 raw − mask +0.000 [−0.003, +0.003]; cand raw − drop −0.000; L7 raw − mean +0.129 [+0.118, +0.140].

Reading: **not supported.** (1) Removing or zeroing the latent does not reduce popularity (R3 is slightly *less* popular than no-latent; mask/drop Δ = 0). (2) A latent with no user content (T3 constant, L7 mean r) produces one fixed list of mid-popularity items (popularity ranks 22–5,246, mostly 300–1,300), not the popular head, and scores below the popularity baseline. (3) All decoders over-recommend popular items relative to targets (head share 0.33–0.43 vs 0.24), but that bias is equal with and without the latent: it comes from SFT/the decoder, not from r.

## Re-check of L5 (CDs grounded) controls: random donors vs off (2026-11-04, analysis only)

Paired bootstrap (`scripts/paired_bootstrap_eval.py`, 2,000, seed 42) on the saved per-row outputs `scratch/R3/output_dir/CDs_and_Vinyl_R3_grounded/eval_*.json` (n = 6,157), prompted by the user finding random > off odd.
- random − off: HR@10 +0.0015 [−0.0007, +0.0036], NDCG@10 +0.0006 [−0.0004, +0.0015] (n.s.).
- random − shuffle_outcome: HR@10 +0.0015 [+0.0002, +0.0028] (sig), NDCG@10 +0.0002 (n.s.); random − latent_only +0.0005 (n.s.); shuffle_outcome − off +0.0000 (n.s.). Earlier: knn − random +0.0008 (n.s.).
- Every donor mode changes 89% of top-10 lists vs off but keeps 90% of the items: reordering only.

Reading: random donors with their own outcomes do as well as retrieved ones; what separates knn/random from shuffle_outcome is whether each donor's (latent, next item) pair is intact. So the significant knn − shuffle_outcome result does not show that *similar users'* behaviour helps; it is consistent with shuffled pairs being off-distribution for the value network (trained only on intact pairs), or with a generic prior from real (latent, outcome) pairs. `off` is also off-distribution (training always added a 0.5‖r0‖ donor component). L5's behaviour-grounding claim is weaker than recorded in the previous log.
- Donor bank of the saved L5 checkpoint is diverse (49,251 rows, 47,076 distinct, residual share 0.785, ‖r‖ 122), so knn ≈ random is not caused by collapsed donor keys.
- Collapse check of the grounded donor banks (user hypothesis: on Toys the latent collapses into a few clusters, making latent-based donor retrieval meaningless). Final-model banks are not collapsed: distinct / rows, residual share — Toys 48,815 / 53,898, 0.809; Musical 57,701 / 66,500, 0.773; CDs 0.785. Clustering: k-means k = 4 / 16 / 64 explains 9% / 21% / 36% of the variance on Toys (CDs 11% / 26% / 39%), largest cluster 35% / 14% / 4%; top-5 PCs 20% (CDs 28%). Unlike plain R3 Toys (one constant r, U2), the grounded model's r0 stays spread, plausibly because the item-CE on r1 sends a target-dependent gradient into the pooling. Early-epoch and queue-time collapse not measured (queue not saved). So collapse does not explain knn ≈ random in L5/L10 final models; it would break retrieval in any configuration where r collapses (plain R3 Toys).

## Recency-split bottleneck m = 1 (option 2, Toys; requested 2026-11-04 via JobManager)

Context: m = 2 had effectively already run as B1's fixed `recent-2` baseline (343007: K=1 hidden thought from the full history, the last 2 items visible, older items deleted from every position): HR@10 0.0957 vs T2 0.1064, raw − shuffle +0.0012 (n.s.), so `r` was ignored. User chose m = 1 plus a matched baseline (probe gap recent-1 0.0965 vs full 0.1064, the most room for `r`).
- Method `Toys_and_Games_split_recent_k1`: K=1 hidden continuous thought (first pass sees the full history), recommendation pass sees only the last history item (older items masked as deleted, positions recomputed; same masks in training / scoring / KV-cache beam search). AUX_WEIGHT = 0 (in 343007 the unused oracle aux loss, 0.2, also sent gradient into `r` through the selector query). Modes raw / mean / shuffle / sel_all.
- Matched baseline `Toys_and_Games_split_recent_k1_constant`: one learned constant thought, same mask, same recipe and budget. New path `SteeredReading.select_by_rule` (fixed recent / all from `item_index`, no latent or first pass); CPU check: identical selection and block to `forward` with rule recent / all for k = 1, 2.
- Recipe as T1–T3 / B1 (base Qwen2.5-1.5B, lr 3e-4, batch 128, ≤ 3 epochs, early stopping). Code b6cc1fb, frozen `output_dir/steer_code/code_b6cc1fb`, launcher `scripts/steered_reading.sbatch` (new THOUGHT_SOURCE). Smoke (constant, 512 train rows, 1 epoch) then both full runs afterok.
- Success: method − constant baseline significantly > 0 (the thought carries older-history information that gets used); raw − mean / shuffle significant; noninferior to T2 / R3 (margin 0.005 HR@10). Jobs (submitted by JobManager 2026-11-04): smoke 353003 failed on a launch setting (TRAIN_SAMPLE 512 also samples the validation file, cut to 64 rows by VALID_SAMPLE; self-checks had passed: mask = deletion max |Δlogp| 2.4e-4, KV-cache 7.3e-5); 353004 / 353005 cancelled. Resubmitted: smoke **353010** (VALID_SAMPLE 512) → method **353011**, constant baseline **353012** (afterok:353010). Logs `run_logs/R3_split_*.<id>.out`.
- Smoke 353010 completed 2026-11-04 (3m29s, aloha): constant thought + fixed recent-1 path runs end to end (64 test rows, 0 unknown predictions; numbers not meaningful). Full runs 353011 / 353012 started on voodoo.

## Soft bottleneck (fixed attention bias on the history; requested 2026-11-04 via JobManager)

User-approved after discussion: instead of hiding the history from target-side queries (L7/T1, b = −∞) or leaving it open (plain R3/T2, b = 0), add a fixed bias b to the attention logits from the thought and target-token queries to history keys, in every layer and head, identical in training and generation; positions unchanged. Intent: the decoder can still copy a strongly wanted history token (brand words) at a cost, while `r` and the instruction are the cheap route for everything else. No loss on `r`.
- Code 9bee817 (frozen `output_dir/thoughts_code/code_9bee817`): `bottleneck_mask_and_positions(history_bias=b)`, train `--history_bias`, launcher `HISTORY_BIAS`, leak check reports instead of failing. CPU check on a tiny random Qwen2: hard mode unchanged, bias only on (target query, history key) pairs, history queries unpenalised, positions unchanged, KV-cache vs full scoring max |Δlogit| 1.3e-7, b = −30 ≡ hard mask.
- Choice of b: heuristic, not calibrated (equal-score estimate of the remaining history attention: b = −2 ≈ 0.16, −4 ≈ 0.025, −6 ≈ 0.003, vs ≈ 0.58 at b = 0). Started with b = −4 only (user agreed to implement; defaults: b = −4 + baseline first, K = 1 continuous thought for the Toys anchors T1/T2/T3, not waiting for 353011/353012). Calibrating b from measured attention on a full-history checkpoint is still an option.
- Runs (Toys, T1–T3 recipe, scratch outputs): smoke (hidden, 512 rows, 1 epoch) → `Toys_and_Games_thoughts_hidden_k1_bottleneck_hb4` (raw / mask / mean / shuffle) and matched baseline `..._constant_k1_bottleneck_hb4` (raw / mask). Anchors: b = −∞ T1 K=1 0.0801, T3 0.0033; b = 0 T2 0.1064.
- Success: raw − mean/shuffle significant; method − constant baseline (same b) significant; noninferior to T2 (margin 0.005 HR@10). Risks: the model learns to cancel the bias (check history attention mass after training); smooth interpolation with no winning b.
- Jobs (JobManager, 2026-11-04): smoke **353054** → method **353055**, constant baseline **353056** (afterok:353054). Logs `run_logs/R3_softbn_*.<id>.out`.
- Smoke 353054 done (4m32s): pipeline OK; history swap with r fixed max |Δlogit| 8.5 (mean 3.6), r swap 6.6 (untrained, 4 optimizer steps), so both routes reach the output under b = −4. HR@10 0/64 for raw and shuffle (recency-split smoke 0.0625 had a fully visible last item; with b = −4 the base model's copying is suppressed at the start: not informative). Full runs 353055 / 353056 running on voodoo.

## Category-prediction uncertainty (offline, 2026-11-04, `scripts/category_uncertainty.py`)

Question (user): how uncertain are the category predictions, and does the confidence tell when to trust the category latent? Report `output_dir/category_latent/category_uncertainty_report.md`. Rows without a level-3 category excluded.

| predictor / split | top-1 acc | AUROC of top-1 confidence for "top-1 correct" | ECE | top-1 acc at 5% / 10% / 20% coverage |
|---|---:|---|---:|---|
| SASRec train (5-fold OOF) | 0.235 | 0.787 [0.782, 0.792] | 0.103 | 0.735 / 0.661 / 0.566 |
| SASRec valid (all-rows model) | 0.190 | 0.770 [0.754, 0.785] | 0.134 | 0.670 / 0.588 / 0.482 |
| LLM SFT test (beam score) | 0.198 | 0.781 [0.766, 0.798] | – | 0.661 / 0.616 / 0.528 |

Margin top1−top2 is a weaker feature (0.70–0.74). SASRec is overconfident (each bin's accuracy ≈ 0.6–0.75 × its mean probability). **Confidence ranks correctness well**: the top 10% most confident rows are right ~60% of the time vs 20% overall.

**Gate (diagnostic, τ swept on test: optimistic):** the additive model's pred1 list where the LLM confidence ≥ τ, R3 otherwise. Best: 5% of rows gated, HR@10 +0.0007 [−0.0009, +0.0025]; 20%: −0.0027 (n.s.), NDCG@10 −0.0023 (sig); 48%: −0.0050 (sig). Oracle gate (exactly the top-1-right rows, 19%): +0.0169 HR@10 / +0.0090 NDCG@10.

**Why confidence does not convert into gains:** confident rows are rows R3 already solves. Top 10% confident: R3 HR@10 0.313 vs 0.080 on the rest (top 5%: 0.330 vs 0.091). Inside the top 10%: right category (399 rows) R3 0.398 → pred1 0.436 (+0.038); wrong (249) 0.177 → 0.112 (−0.065). The large pred1 gain on right rows (+0.089 overall: 1,284 rows, 0.296 → 0.385) sits mostly on low-confidence right rows, which the confidence cannot single out. Confidence tracks "easy / repeat-pattern row" (redundant with the history), not extra information.

**Conclusion:** a history-based category predictor knows when it is likely right, but those are the rows where the decoder does not need the category; calibrated trust (the two-slot design with soft categories + confidence) has no headroom on Toys from this signal (≤ +0.001 HR@10 even with τ tuned on test). The +0.017 oracle-gate gain needs a correctness signal that is not available from the history.

## 2026-11-04 — Staged latent training: latent-only first, then full attention (launched, Toys)

User idea: R3 ignores r because the history is more useful from the first epoch; train with the latent first so the decoder learns to use r, then open the history so it fixes hard samples. Closest prior test: Attempt 1 follow-up B (CDs): 2 epochs of mixed reasoning_only / history_only batches, then 1 epoch with 1/3 full-attention batches → content dependence erased (raw − mean HR@10 −0.0002 n.s.), HR@10 0.1013. Difference here: a pure L7 latent-only stage (never history_only batches) followed by pure full attention. Prediction stated to the user: dependence decays within the full stage, ranking ≈ R3.

Design: plain R3 (attention-pooled r), R3_ori Toys recipe (base Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs); `--latent_only True --latent_only_epochs N`: `BottleneckSchedule` callback sets `model.config.latent_bottleneck` for epochs < N and removes it from epoch N (training, epoch-end eval and saved config switch to plain R3). Early stopping and best-model reload disabled; final epoch kept (R3_ori's best epoch was 2 of 3). Eval with history visible: constrained 10-beam ranking, interventions raw / mask / mean / shuffle_user (experiment.py), paired bootstrap. Baseline: R3_ori Toys 0.1023 / 0.0659 (interventions `toys_and_games_v0_checkpoint_844`).

Code `72d7de0`, frozen `output_dir/staged_code/code_72d7de0`, launcher `scripts/staged_latent.sbatch`; outputs `scratchDirectory/R3/output_dir/Toys_and_Games_staged_latent{1_full2,2_full1}`, interventions `analysis/results/intervention/toys_and_games_staged_latent*`. Jobs: smoke 353227 (512 rows, 1+1 epochs) → **353228 (1 latent + 2 full)**, **353229 (2 latent + 1 full)** (afterok).

Staged smoke 353227 passed (5 min): schedule printed "epoch 0: latent-only bottleneck", "epoch 1: full attention"; final config has no bottleneck; ranking + interventions ran. 353228 / 353229 released.

## 2026-11-04 — Recency split m=1: constant-thought baseline result (353012)

Constant thought, last item visible to the target, older items deleted (code b6cc1fb). n = 6,738.

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (recent-1) | 0.0380 | 0.0638 | 0.0781 | 0.0981 | 0.0380 | 0.0529 | 0.0587 | 0.0652 |
| sel_all (all history visible at eval) | 0.0297 | 0.0491 | 0.0652 | 0.0904 | 0.0297 | 0.0408 | 0.0473 | 0.0554 |

Paired bootstrap: raw − sel_all HR@10 +0.0077 [+0.0028, +0.0128], NDCG@10 +0.0098 [+0.0066, +0.0132]; raw − T2 HR@10 −0.0083 [−0.0138, −0.0030] (sig), NDCG@10 −0.0026 [−0.0057, +0.0005]; raw − R3 HR@10 −0.0042 [−0.0095, +0.0013], NDCG@10 −0.0006 (n.s.). The last item alone gives near-R3 ranking without any latent: the bar the method run 353011 (latent reads the older items) must clear.

## 2026-11-04 — History-only pooling (launched, Toys)

User design (choices: token-level, title tokens only, plain R3 with full attention, check pooling weight on non-history = 0). R3's pooling puts ~0.999 of its mass on the template; here the pooling keys are restricted to history title tokens (between the fixed "### User Input: …before:" header and the last "### Response:", minus item-boundary tokens made only of quote/comma/space/newline characters and containing a quote). Mask computed from input_ids in `LatentModel.pooling_mask` (used by generate_embs and experiment.py `thought_embeds`); prompt and both passes unchanged. Offline check (3,000 Toys train+test rows): vs exact character-offset title mask, 0 template tokens included, 436 / 180,018 title tokens dropped (quote-only tokens inside titles), agreement 0.9976; pooling weight on non-title tokens 0.0. Prediction stated to the user: r becomes user-specific but stays redundant with the history (B1 / guidance-scaffold runs had non-collapsed pooling and r was still ignored) → mean/shuffle n.s., HR@10 ≈ R3.

Recipe R3_ori Toys (base Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs, early stopping). Code `e111e6d`, frozen `output_dir/histpool_code/code_e111e6d`, `scripts/history_pooling.sbatch` (+ `scripts/history_pooling_check.py`: trained-checkpoint pooling weight on non-title tokens, must be < 1e-6). Jobs: smoke 353241 → **353242** (afterok); output `scratchDirectory/R3/output_dir/Toys_and_Games_history_pooling`. Compare: R3 0.1023 / 0.0659 and its interventions (`toys_and_games_v0_checkpoint_844`).

## 2026-11-04 — Soft bottleneck b=−4: constant-thought baseline result (353056)

Constant thought, target-side attention to the history penalised by −4 (code 9bee817). Leak check: history swap with r fixed max |Δlogit| 7.0 (history readable by design); r swap 0 (constant). n = 6,738.

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0338 | 0.0594 | 0.0761 | 0.1023 | 0.0338 | 0.0488 | 0.0557 | 0.0641 |
| mask (zero thought) | 0.0156 | 0.0370 | 0.0515 | 0.0785 | 0.0156 | 0.0278 | 0.0338 | 0.0425 |

Paired bootstrap: raw − mask +0.0237 [+0.0186, +0.0292] / +0.0216 (sig; activation dependence on a constant vector, no content); raw − T2 −0.0042 [−0.0083, −0.0001] / −0.0037 [−0.0062, −0.0014] (sig); raw − R3 +0.0000 / −0.0018 (n.s.); raw − recency-split constant +0.0042 [−0.0012, +0.0092] (n.s.). The b = −4 penalty costs ~4% vs unpenalised T2 without any latent information; the method run 353055 must beat 0.1023 with mean/shuffle hurting.

History-pooling smoke 353241: training, trained-checkpoint pooling check (max weight on non-title tokens 0.0; the same checkpoint with the plain mask would put 0.600 there), ranking and the raw intervention passed; the job then failed because the smoke output directory was deleted externally between 21:27:14 and 21:27:47 (the staged smoke dir is gone too; asked JobManager). Code not at fault; 353242 (stuck on the dependency) cancelled and the full run resubmitted as **353334**.

## Does the additive oracle-category model just return the category's popular items? (2026-11-05, analysis, `scripts/oracle_category_vs_popularity.py`, e0176d2)

Rows with a level-3 category (n = 6,481; median category size 112). Category popularity = top-10 of the true category by training interaction count (targets + histories), history items excluded, no history otherwise (HR@10 0.3135 all / 0.2481 copy / 0.3446 other; the earlier 0.3402 used a different count).

| | HR@10 all / copy / other | overlap with category-popularity top-10 (random-in-category) | within-category popularity percentile of in-category recs (random 0.5) |
|---|---|---:|---:|
| oracle model, own history | 0.3782 / 0.5182 / 0.3118 | 0.310 (0.223) | 0.306 |
| oracle model, another user's history | 0.2449 / 0.1670 / 0.2818 | 0.344 (0.223) | 0.284 |

Hits (own history): both 1,430, model only 1,021, popularity only 602. Reading: **partly true.** The category embedding conditions the region (87% of items in the category) and the model leans to popular items inside it (percentile 0.31), but its list is not the popularity list (3 of 10 items shared, vs 2.2 at random); on other rows it is close to, and slightly below, category popularity (0.312 vs 0.345), on copy rows it is far above it (0.518 vs 0.248, brand copying from the history). With a wrong history it does not fall back to popularity: it follows the wrong history and drops below the popularity rule (0.245 vs 0.314).

## Next phase (user, 2026-11-05): find additional information for the latent first, then balance

Agreed: the decoder relies on whichever source lowers the loss most (modality imbalance); balancing comes after the latent carries information the prompt lacks. Constraints: prompt fixed, interaction data only (no metadata/text), no popularity direction, SASRec candidates closed.

**What interaction information the prompt omits (Toys test, quick screen):** the prompt holds only history titles. Missing: timestamps, ratings (11.3% of history ratings < 4, yet all items are presented as liked), history beyond 10 items (12.2% of rows at the cap), interactions before 2017 (outside the split window, present in the raw file), other users' behaviour.

Timing is strongly informative about the copy-vs-explore decision:

| time from last history item to target | share | copy-row rate | R3 HR@10 (copy / other) |
|---|---:|---:|---|
| same day | 0.497 | 0.447 | 0.1723 (0.366 / 0.0151) |
| 1–30 days | 0.120 | 0.280 | 0.0483 (0.146 / 0.0103) |
| > 30 days | 0.383 | 0.163 | 0.0283 (0.119 / 0.0106) |

History-only timing (no target timestamp): last two history items on the same day in 53% of rows; copy rate 0.379 vs 0.253 otherwise. Open question for the user: is the request time (target timestamp) admissible at inference?

## Target-item oracle latents (agreed with user 2026-11-05)

Question (user): if the latent explicitly encodes the target item, can the decoder copy it as the answer? If yes, these oracles could serve as ground truth for a diffusion refinement latent. Prior evidence: Musical `oracle_cfitem` (SASRec item embedding) 0.985 HR@10 by lookup only (0.000 on targets never seen as training targets).

Design: category additive latent reused (`r' = r + g·normalize(P(v))`, trained from base Qwen2.5-1.5B, R3 recipe, prompt unchanged) with v = the target item's vector. Kinds (`src/latent/item_oracle_prep.py`): **r3title** (R3's own last-layer title representation), **qwen3** (Qwen3-Embedding title), **sasrec** (SASRec item embedding; untrained for items never seen in training), **charac** (user addition: item characteristics without the title, i.e. brand, category path, features, description from metadata, encoded by the R3 backbone, ≤ 256 tokens; coverage 99.98%; oracle/upper-bound use only). Test targets seen as training targets: 94.98% (≈ 338 unseen rows for the read-vs-lookup split).

Eval modes (`latent_attention_eval.py perturb_oracle`, `scripts/item_oracle.sbatch`): oracle, shuffle (another user's target), pred1 (SASRec top-1 item's vector), nn:1/5/20 (k-th nearest item), noise:0.25/0.5/0.75/0.9 (variance-preserving noise on the mean-centred vector; centred cos 0.97/0.87/0.66/≈0.44), swaphist (own vector + another user's history). Centred cos of the true vector to nn:1/5/20: r3title 0.78/0.65/0.53, qwen3 0.76/0.62/0.49, sasrec 0.56/0.45/0.38; random ≈ 0.

Code a0ed966 (merge of the category-latent branch + item oracle), frozen `output_dir/item_oracle_code/code_a0ed966`; prep tables r3title/qwen3/sasrec built (CPU), charac needs GPU. Submission requested via JobManager: prep_charac, smoke (r3title, 512 rows, 1 epoch, 64 test rows, all 11 modes), then 4 full runs after the smoke. Results → `output_dir/item_oracle/Toys_and_Games_item_oracle_<kind>/`.
Jobs (JobManager, 2026-11-05): prep_charac **353426**, smoke **353427** (r3title, Toys_item_oracle_smoke), full r3title **353428**, qwen3 **353429**, sasrec **353430** (afterok:353427), charac **353431** (afterok:353427:353426).

## 2026-11-05 — Recency split m=1: method result (353011)

Hidden (Coconut-style) thought computed from the full history; the target sees the last item + thought, older items deleted (code b6cc1fb). n = 6,738, 8h36m.

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0419 | 0.0678 | 0.0830 | 0.1043 | 0.0419 | 0.0570 | 0.0632 | 0.0701 |
| mean | 0.0419 | 0.0669 | 0.0824 | 0.1043 | 0.0419 | 0.0565 | 0.0628 | 0.0699 |
| shuffle | 0.0414 | 0.0677 | 0.0821 | 0.1042 | 0.0414 | 0.0567 | 0.0626 | 0.0698 |
| sel_all | 0.0300 | 0.0524 | 0.0675 | 0.0944 | 0.0300 | 0.0429 | 0.0491 | 0.0578 |

Paired bootstrap: raw − mean HR@10 +0.0000 [−0.0016, +0.0018] / NDCG@10 +0.0003 (n.s.); raw − shuffle +0.0001 / +0.0003 (n.s.); raw − sel_all +0.0099 / +0.0123 (sig, train/test mismatch). raw − constant baseline (353012) +0.0062 [+0.0015, +0.0110] / +0.0049 [+0.0021, +0.0078] (sig); raw − T2 −0.0021 [−0.0082, +0.0039] / +0.0023 (n.s.); raw − R3 +0.0021 [−0.0040, +0.0080] / +0.0043 [+0.0006, +0.0081] (NDCG sig).

**Reading:** the thought's content is not used even though it is the only route to the older items (mean/shuffle exactly neutral). The gain over the constant baseline therefore does not come from r's content; candidates: training-run variance (single seeds; bootstrap CIs cover test sampling only) or a training-side effect of the hidden first pass on the shared weights. Ranking ≈ T2 / R3. Success criterion (content dependence) **not met**: with the last item visible, the decoder does not need the older history, consistent with the last-item baseline ≈ R3.
Smoke 353427 passed 2026-11-05 (10 min; all 11 modes written; modes change predictions vs oracle on 24-64/64 rows; 1-epoch/512-row model not meaningful). charac prep 353426 done (6,298/6,299 items, all test targets covered). Full runs 353428-353431 pending.

## 2026-11-05 — Soft bottleneck b=−4: method result (353055)

Hidden thought, target-side history attention biased by −4 (code 9bee817), 8h26m. Leak check: history swap with r fixed max |Δlogit| 7.9; r swap with history fixed mean max |Δlogit| 0.077 (small).

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0341 | 0.0561 | 0.0744 | 0.1002 | 0.0341 | 0.0467 | 0.0542 | 0.0626 |
| mask | 0.0178 | 0.0392 | 0.0557 | 0.0822 | 0.0178 | 0.0298 | 0.0365 | 0.0451 |
| mean | 0.0340 | 0.0567 | 0.0742 | 0.0996 | 0.0340 | 0.0470 | 0.0542 | 0.0624 |
| shuffle | 0.0338 | 0.0567 | 0.0747 | 0.1000 | 0.0338 | 0.0469 | 0.0543 | 0.0625 |

Paired bootstrap: raw − mask +0.0180 / +0.0175 (sig, activation dependence); raw − mean +0.0006 [−0.0007, +0.0019] / +0.0001 (n.s.); raw − shuffle +0.0001 / +0.0001 (n.s.).
raw − constant baseline 353056 −0.0021 [−0.0061, +0.0022] / −0.0015 (n.s.); raw − T2 −0.0062 [−0.0105, −0.0016] / −0.0052 (sig); raw − R3 −0.0021 (n.s.) / −0.0033 [−0.0056, −0.0009] (sig).

**Reading:** b = −4 is too weak: the decoder still reads the history (Δlogit 7.9 vs 0.08 for r), r's content is unused, and the latent adds nothing over a constant vector. Success criterion not met. A stronger bias would move toward L7 (content used, −23%); between b = −4 (history fully used) and b = −∞ the trade-off is untested.

### Staged 1+2 (353228): ranking done, interventions OOM, eval rerun 353734 (2026-11-05)
- 353228 trained on an RTX 3090 (voodoo) and finished standard ranking: HR@10 0.1008, NDCG@10 0.0600 (n = 6,738).
- It then crashed with CUDA OOM during the first intervention (raw, beam search in latent_grpo_processor) at batch 16 on 24 GB.
- Rerun of evaluation only: job 353734, SKIP_TRAIN=1, same code copy (code_72d7de0) and output dir, constraint l40s|a100|l40|a40 (no 3090).
- 353229 (2+1) is also on a 3090 and may hit the same OOM; if so, rerun its eval the same way.
- 353229 (2+1): same failure. Training and ranking finished: HR@10 0.0870, NDCG@10 0.0477. OOM at the raw intervention on the 3090. Eval-only rerun: job 353741 (no 3090).

### Standard ranking so far vs R3 (paired bootstrap, 2,000 resamples, seed 42; Toys n = 6,738), 2026-11-05
| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | ΔHR@10 vs R3 | ΔNDCG@10 vs R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| history pooling 353334 | 0.0341 | 0.0592 | 0.0757 | 0.1036 | 0.0341 | 0.0489 | 0.0556 | 0.0646 | +0.0013 [−0.0028, +0.0056] n.s. | −0.0012 [−0.0036, +0.0011] n.s. |
| staged 1+2 353228 | 0.0288 | 0.0543 | 0.0738 | 0.1008 | 0.0288 | 0.0434 | 0.0513 | 0.0600 | −0.0015 [−0.0073, +0.0042] n.s. | −0.0059 [−0.0092, −0.0027] sig |
| staged 2+1 353229 | 0.0191 | 0.0405 | 0.0571 | 0.0870 | 0.0191 | 0.0313 | 0.0381 | 0.0477 | −0.0153 [−0.0211, −0.0092] sig | −0.0182 [−0.0219, −0.0144] sig |
History pooling check (trained checkpoint): max weight on non-title tokens 0.0; the plain mask puts 0.136 there. Interventions pending (353334 running; staged evals 353734 / 353741 pending).

### Goal update (user, 2026-11-05)
Beating R3 is not required: the target is a latent whose content matters for the recommendation (mean / shuffle interventions significantly hurt). Ranking is still reported but is not a pass/fail criterion.
Clarified by the user: "no drop in performance". Criterion = content dependence AND ranking not significantly below R3 (matching R3 is enough).

### Latent dependency test v2 (intervention v2), launched 2026-11-05 (code 9f8a582)
- Design (agreed with the user): per-sample, forward only.
  - y_hat = the checkpoint's own top-1 (from its ranking file).
  - Teacher-force y_hat with the own r, with the zero vector, and with K = 16 latents drawn at random from the same checkpoint's train-latent bank (same user excluded).
  - Per sample: D = log p(y_hat | own r) − log p(y_hat | bank r), summed over title tokens (mean / std / min / max over K); KL(own ‖ swapped) summed over title tokens; share of argmax changes; own_rank among K + 1.
  - Under the null (content unused), own_rank = 1 with probability 1/17. y_hat is selected with the own r, so D is biased upward a little: plain R3 is the null reference.
  - Script `src/prototype_grounding/latent_dependency_v2.py`, launcher `scripts/latent_dependency_v2.sbatch` (extracts the bank if missing; verifies its checkpoint sha).
- Bug fixed in the same commit: `experiment.py generate` computed the raw r with the plain mask, so for history-only pooling (353334) the `raw` intervention used an untrained pooling; mask / mean / shuffle_user and standard ranking were unaffected. Rerun `raw` for 353334 with code 9f8a582 after it finishes.
- No L7 checkpoint survives (scratch purge), so there is no positive reference; the existing toys v0_checkpoint_844 bank comes from a different R3 run (sha 94b8… vs R3_ori a29b…).
- Smoke 353757 (staged 1+2, 24 rows, A40, 4 s): runs. On those rows D_mean 1.99 nats (median 1.71), KL 1.31, argmax change 4%, own_rank = 1 in 67%. Not interpretable before the R3 null.
- Runs: 353830 plain R3 (R3_ori Toys; new bank tag toys_and_games_r3_ori, extraction ~1.5 h); 353831 staged 1+2; 353832 staged 2+1. History pooling to follow after 353334 finishes (bank + raw rerun).

### History pooling 353334 finished (2026-11-05 13:09); raw intervention invalid, rerun 353953
- Standard ranking (correct mask): HR@10 0.1036 / NDCG@10 0.0646 (table above).
- Interventions from the job (code e111e6d), with `raw` computed by the buggy all-token pooling:
  - raw HR@10 0.1045 / NDCG@10 0.0652;
  - raw − mask +0.0093 sig / +0.0111 sig;
  - raw − mean +0.0012 [−0.0004, +0.0030] n.s.;
  - raw − shuffle_user +0.0010 n.s.
- These raw-based deltas are not valid for the trained model. The buggy raw file was moved to `intervention/toys_and_games_history_pooling/buggy_pool_raw/`.
- Rerun 353953 (code 9f8a582): raw generation with the history-only mask → paired bootstrap vs mask / mean / shuffle_user → dependency test v2 (predictions = eval_raw.json; shuffle file = shuffle_user generation).

### v2 result: plain R3 null reference (353830, R3_ori Toys, n = 6,738, K = 16)
| quantity | swap to bank latent (mean over K) | swap to zero vector |
|---|---|---|
| D = logp drop of own top-1, nats (mean / median / p90) | −0.0003 / −0.00004 / 0.044 | 3.61 / 3.57 / 6.00 |
| KL(own ‖ swapped), summed over title tokens (mean / median / p90) | 0.0010 / 0.0008 / 0.0019 | 3.55 / 3.52 / 5.58 |
| share of title positions whose argmax changes | 0.0009 | 0.049 |
| share of rows with D > 1 nat | 0.0 | 0.92 |
- Reading: per sample, plain R3 is insensitive to which real latent fills the slot (no row with D > 1; p90 0.04 nats) but needs a vector there (zero costs 3.6 nats). Activation dependence without content dependence, now at the level of every sample.
- own_rank is not usable as a per-sample p-value: own_rank = 1 in 49.6% of rows while the mean own percentile is 0.499. When the differences are at bf16 resolution, ties (counted against own) and the selection of y_hat with own r make it bimodal. Use D / KL magnitudes against this null.
- Smoke comparison (staged 1+2, 24 rows): D 1.99, KL 1.31, about 1,000× this null; full runs 353831 / 353832 pending.

## 2026-11-05 — Item oracle r3title result (353428)

7h27m on lunar, best epoch 2 (eval loss 0.178 / 0.063 / 0.063). n = 6,738 (@20 = @10).

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| oracle | 0.9316 | 0.9488 | 0.9495 | 0.9617 | 0.9462 |
| swaphist | 0.9267 | 0.9469 | 0.9478 | 0.9586 | 0.9427 |
| noise 0.25 / 0.5 (centred cos 0.97 / 0.87) | 0.9304 / 0.9222 | 0.9484 / 0.9472 | 0.9494 / 0.9486 | 0.9616 / 0.9598 | 0.9454 / 0.9414 |
| noise 0.75 / 0.9 (cos 0.66 / 0.44) | 0.8544 / 0.5223 | 0.9246 / 0.6940 | 0.9383 / 0.7511 | 0.9531 / 0.8139 | 0.9071 / 0.6669 |
| nn:1 / nn:5 / nn:20 | 0.0337 / 0.0009 / 0.0001 | 0.4038 / 0.0853 / 0.0162 | 0.5318 / 0.1598 / 0.0289 | 0.6937 / 0.3198 / 0.0687 | 0.3579 / 0.1304 / 0.0272 |
| pred1 (SASRec top-1 item) | 0.0380 | 0.0494 | 0.0570 | 0.0684 | 0.0514 |
| shuffle | 0.0006 | 0.0013 | 0.0021 | 0.0036 | 0.0017 |

Bootstrap oracle − mode HR@10: swaphist +0.0031 [+0.0015, +0.0049]; noise 0.25 +0.0001 (n.s.); noise 0.5 +0.0019; noise 0.75 +0.0086; noise 0.9 +0.1478; nn:1 +0.2680; pred1 +0.8933; shuffle +0.9581 (all others sig).

Analysis (`scripts/item_oracle_analysis.py`): HR@10 seen / unseen-as-training-target (338 rows): oracle 0.9930 / **0.2456** (R3 0.1075 / 0.0030); swaphist 0.9912 / 0.2160; noise 0.9 0.8470 / 0.0740. The item encoded by the given vector is the top-1 output in 92.6% (oracle), 92.6% (nn:1), 96.7% (nn:5), 96.5% (nn:20), 97.9% (pred1), 92.0% (shuffle donor's target).

Reading: (1) the decoder copies the latent: whatever item the vector encodes becomes the top-1 (≥ 92% in every mode), and the history is essentially ignored (swaphist −0.003). (2) Mostly lookup, partly reading: 0.99 on seen targets, 0.25 on never-seen targets (vs R3 0.003), so the R3-title space lets the decoder spell some unseen titles. (3) It is a nearest-item decoder: isotropic noise is tolerated down to centred cos ≈ 0.66 (still 0.95), but a vector that is closer to another item (nn:1, cos 0.78) yields that item. For diffusion: the predicted latent only has to land in the target's cell, but a predictor's errors are plausible other items (pred1 → HR@10 0.068 < R3 0.102), and the decoder then concentrates the whole list around the wrong item. An oracle-trained decoder turns ranking quality into the top-1 accuracy of the latent predictor.

## 2026-11-05 — Item oracle qwen3 result (353429)

7h29m on lunar, best epoch 2 (eval loss 0.230 / 0.074 / 0.074). HR@1/3/5/10, NDCG@10: oracle 0.9205 / 0.9446 / 0.9469 / 0.9512, 0.9374; swaphist 0.9491 HR@10 (−0.0021 sig); noise 0.25/0.5/0.75/0.9 HR@10 0.9510 / 0.9503 / 0.9424 / 0.7974; nn:1/5/20 0.6293 / 0.2668 / 0.0798; pred1 0.0671; shuffle 0.0027. Seen / unseen HR@10: oracle 0.9919 / **0.0562** (r3title 0.2456); given item in top-1 91–97% in every mode.

Reading: same copy / nearest-item behaviour as r3title, slightly lower overall, and much weaker on never-seen targets: an external text embedding is learned as a lookup, while R3's own title space lets the decoder spell 4× more unseen titles. For a diffusion target, r3title is the better space.

## 2026-11-05 — Diffusion-refined thought (stage B), session "DiffSBR"

User request: implement diffusion refinement of R3's latent toward the target title embedding, with the thought **r' = diffuse(r)** (replaces r; prompt unchanged). Worktree `.claude/worktrees/R3DiffusionRefine` (branch `worktree-R3DiffusionRefine`, from `49641cc`), code `83ccdb4`, frozen copy `output_dir/diffusion_refine_code/code_83ccdb4`, `scripts/diffusion_refine.sbatch`.

Design (`src/latent/diffusion_refine.py`):
- x0 = sqrt(d)·normalize(v_target − μ), v from `output_dir/item_oracle/prep/Toys_item_r3title.pt` (centred R3 title space).
- Denoiser f(x_t, t, c(r)): fp32 MLP (width 2048, 3 residual blocks), x0-prediction, T = 32 (DDPM linear schedule rescaled to T). Condition c = Linear(LayerNorm(r)).
- r' = g·normalize(P(x̂0)) (P: 1536→hidden, g init 30) replaces R3's pooled r in the thought slot.
- Loss = NLL + 1.0·(MSE(x̂0, x0)/d + 1.0·catalogue CE(cos(x̂0, items)/0.05)). The NLL backpropagates through r' into f and R3's pooling.
- Decoder input in training: one-step x̂0 at random t, or (p = 0.5) the test-time DDIM chain from noise (gradient through the last step). Test: deterministic DDIM chain from a fixed seeded noise vector shared by all rows (beam copies get the same r').
- Control `MODE=regress`: x̂0 = f(0, 0, c(r)), same losses (no diffusion).
- Eval modes: raw, oracle (target vector in place of x̂0: this decoder's ceiling), shuffle (another user's x̂0), mean, swaphist. Each eval writes `diffusion_x0.json`: nearest-item retrieval HR/NDCG@1–20 of x̂0 over the 6,299 items + centred cosine, split seen / unseen targets.

Success (user goal): r' content matters (shuffle/mean ≪ raw) and raw HR@10 ≈ R3 (0.1023) or better. Bar for x̂0 alone: nearest-item HR must approach R3's HR@1 0.038 / HR@10 0.102 for the copy behaviour to help.

Jobs (2026-11-05, `--requeue`): smoke 354134 (code `83ccdb4`) ran end to end (train, raw/oracle/shuffle ranking, x̂0 readout) but exposed a bug: the aux loss was added per micro-batch while the Trainer normalises the NLL over the whole accumulated batch, so it counted 32× (64× on 3090; logged loss ~650 vs ~4.8 for the R3 recipe). Full jobs 354135/354136 cancelled before starting. Fixed in `f2b6c8b` (aux scaled by micro-batch label tokens / num_items_in_batch; loss parts now logged), frozen `output_dir/diffusion_refine_code/code_f2b6c8b`. Resubmitted: smoke2 354572 → diffusion 354573 (afterok). Regress submitted 2026-11-06 as 354956 (same code; first attempt hit QOSMaxSubmitJobPerUserLimit). Outputs: scratch `…/output_dir/Toys_and_Games_diffusion_refine_{diffusion,regress}/`, copied to `output_dir/diffusion_refine/`.

| run | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffusion raw (354573) | 0.0242 | 0.0462 | 0.0657 | 0.0978 | 0.0978 | 0.0242 | 0.0368 | 0.0448 | 0.0551 | 0.0551 |
| diffusion oracle / shuffle / mean / swaphist | | | | | | | | | | |
| regress raw | | | | | | | | | | |
| x̂0 nearest-item, diffusion (centred cos 0.056; unseen HR@10 0.036) | 0.0062 | 0.0119 | 0.0162 | 0.0261 | 0.0374 | 0.0062 | 0.0094 | 0.0112 | 0.0145 | 0.0173 |
| x̂0 nearest-item, regress | | | | | | | | | | |

## 2026-11-05 — Operator trace: which operator removes r's user-specific information? (CDs R3_ori)

User question: in y = softmax(QKᵀ/√d)V, which operator makes the decoder ignore r? On Toys the pooling softmax makes r constant (U2), so the trace uses CDs R3_ori, whose r varies across users. `src/prototype_grounding/latent_operator_trace.py` (code 1ce53d8, frozen `output_dir/op_trace_code/code_1ce53d8`, `scripts/latent_operator_trace.sbatch`): per test row A and donor B (other user), decoder run on A's input with r_A vs r_B; the difference followed through pooling (prefix/user mass, ||Δr||/||r||), per layer at the thought position (relative difference after RMSNorm, W_V gain on Δ vs on r, per-head attention from target positions to r and its change, W_O gain, share of the head output from Δ vs from all of r), residual at the target positions after attention / after the layer, logits, KL; control: r_A + random direction with the same ||Δr||; reference: drop the last history item.

CPU check (2 rows only, not conclusive): pooling mass prefix 0.86 / user 0.14; ||Δr||/||r|| 0.40; W_V and W_O do not suppress Δ (gain ≈ 0.7–1.7); per-head attention from target tokens to r 1–7% (max head 7–24%); Δ is 2–4% of head outputs at r-attending heads; the relative difference at the thought position decays 0.36 → 0.08 over layers and at the target positions 0.05 → 0.01; final KL 3e-5 (random direction 1e-5, drop last item 0.12); relative logit change 0.27% (random 0.22%). Tentative: no single operator zeroes Δ; it is diluted (small attention share × accumulation of the residual stream), and the decoder treats the user-specific change like a random change of r. Full run (n = 300, GPU) requested via JobManager.
Full run: job **354280** (JobManager, 2026-11-05), n = 300, output analysis/results/latent_operator_trace/cds_r3_ori.json.

### Operator trace result (354280, CDs R3_ori, n = 300; `analysis/results/latent_operator_trace/cds_r3_ori.json`)

- Pooling: mass prefix 0.776 / user 0.224; ||r_A − r_B|| / ||r_A|| = 0.327 (r is user-specific).
- Output: KL swap r 0.00015 (ΔNLL −0.0003) vs random-direction r of the same ||Δr|| 0.00012 vs drop last history item 0.107 (≈ 700×); relative logit change swap 0.57% vs random 0.54%.
- W_V gain on Δ relative to r: 0.65–1.67 (mean ≥ 1 in 22/28 layers); W_O 0.95–1.19: **no suppression by the value/output projections**.
- Attention from target tokens to r: mean per head 0.9–6.7% (max head 2.4–22%), barely changed by the swap (|Δa| ≤ 0.019).
- Share of a head's output at the target positions: from Δ 0.25–2.9% (mean), from all of r 2.4–12.7%.
- Relative difference at the thought position after RMSNorm: 0.29 (layer 0) → 0.07 (layer 27) (r's own position fills with prompt context).
- Residual at target positions: 4.1–5.5% (layer 0) → 1.1–1.6% (layer 27); MLPs do not remove it (after attention ≈ after layer); random direction gives the same profile.

Reading: no operator selectively removes the user-specific part (it propagates exactly like a random change of r). The limiting operator is the decoder's attention softmax: r is one position among hundreds, and target tokens give it only a few percent of each head (≈ 10× an average token, but small in absolute terms), so Δ contributes ~1–3% of head outputs; the residual stream then dilutes it further (r's own position and the targets accumulate history-driven content), leaving a 0.6% logit change. The history differs in many positions that receive ~95% of the attention, hence 700× the effect. The decoder has not learned to read r at all (neither amplifying nor suppressing user directions); it treats r as a small, generic input perturbation.

## 2026-11-05 — Guaranteed attention share for r (B2 budget / A2 fixed, CDs)

Motivation: the operator trace showed the limiting operator is the decoder's attention softmax (target tokens give r ~3% per head). User design: give r a guaranteed *ratio* of attention (not a fixed logit bias), adaptive, and not learnable (a learnable share would shrink to ~0). Latent collapse is deferred.

Mechanism (`src/latent/attention_share.py`, code 15cb6980, frozen `output_dir/attention_share_code/code_15cb6980`): in every decoder layer and head, query positions after the thought position p use w'_tj = (1 − ρ_t)·a_tj/(1 − a_tp) (j ≠ p), w'_tp = ρ_t (a = usual causal softmax); positions up to p unchanged. **A2 fixed**: ρ = 0.3 everywhere. **B2 budget**: ρ_t^(h) = min(1, 0.3·a_tp^(h)/mean_h' a_tp^(h')), i.e. a per-layer, per-position budget of 0.3 distributed over heads by their own attention to r (per query position, so teacher forcing = KV-cache decoding). Eager attention registered as `latent_share`; thought positions set at prefill.

CPU checks (R3_ori CDs weights): rows sum to 1, inactive rows unchanged, other weights rescaled uniformly; fixed: realised share 0.3000 in every layer; budget: mean 0.281 (clipping at 1 in a few heads; per-layer 0.22–0.30); KV-cache vs teacher forcing max |Δlogit| 2.7e-5 / 2.3e-5; hidden states up to r unchanged (2.7e-5). Plain-R3 reference: R3_ori CDs HR@1/3/5/10 0.0495 / 0.0707 / 0.0814 / 0.0996, NDCG@3/5/10 0.0616 / 0.0660 / 0.0719 (different codebase; no matched plain run in this codebase on CDs yet).

Runs (`scripts/attention_share.sbatch`, CDs, base Qwen2.5-1.5B, R3 recipe): smoke (budget, 512 rows) → B2 budget 0.3, A2 fixed 0.3. Each: ranking, interventions raw/mask/mean/shuffle_user (experiment.py), bootstrap, operator trace (n = 300). Submission requested via JobManager.
Jobs (JobManager): smoke **354389**, B2 **354390** (afterok:354389); A2 pending submission (QOS 16-job cap; JobManager retry loop).

**C2 attention-floor regulariser (user: "submit C2", 2026-11-05).** Attention unchanged (inference = plain R3); training adds λ·mean_{l,t} relu(ρ̄ − mean_h a_tr)² over target-predicting positions after r, ρ̄ = 0.3, λ = 10 (mode `regularize` in attention_share.py, code 642a09fa, frozen `output_dir/attention_share_code/code_642a09fa`). CPU check (R3_ori CDs weights): initial head-mean share to r 0.023, penalty 0.077 (×10 = 0.77, exactly the loss increase over plain CE 0.507); penalty gradients reach q_proj/k_proj; eval logits = plain R3 (max |Δ| 2.3e-5). Submission requested via JobManager (afterok smoke 354389); same evaluation as B2/A2.

**Correction (2026-11-05): operator-trace pooling split.** Code 1ce53d8 split pooling mass at the end of the instruction, so the headers after it ("### User Input:", the category phrase, "### Response:") counted as *user* tokens. Fixed in 2459a348 (history title tokens via history_title_mask vs all template). Recomputed on 6 CDs test rows: R3_ori CDs template **0.9982** / history titles **0.0018** (not 0.776 / 0.224 as reported for 354280; consistent with U3's 0.9954); smoke 354389 (1 epoch, budget) template 0.23 / titles 0.77 (not meaningful). All decoder-side numbers of 354280 are unaffected. Since template tokens *after* the history (e.g. "### Response:") have history-dependent hidden states, r's 33% across-user variation on CDs comes from pooling those template states, not the title tokens. B2/A2/C2 run the old trace code (frozen); their pooling split will be recomputed with the fixed code.
Jobs: A2 **354497** (JobManager); smoke 354389 passed (4.5 min, all steps); B2 354390 pending.
C2 job **354567** (JobManager). Attention-share runs: B2 354390, A2 354497, C2 354567.

## Toys `sasrec_raw` replication (2026-11-06)

User request: repeat the Musical `sasrec_raw` side-information run on Toys (hadylauwresearch partition, a40).

**Design:** r' = r + g·normalize(P(v)), where v is the frozen Cornac SASRec sequence embedding of the row's history (128-d; checkpoint `Verifiable-Rec/.../sasrec/Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl`, trained on the train split). It runs through the additive category-latent path, so side modules are fp32 (unlike the Musical run, where `g` was frozen at 30 in bf16). Prompt unchanged. Trained from Qwen2.5-1.5B with R3's recipe (3 epochs, lr 3e-4, batch 128).

**Evals:** oracle (own vector), shuffle (another user's), pred1 (zero side vector = drop); paired bootstrap.

**Code:** `src/latent/sasrec_sequence_prep.py`, `scripts/sasrec_latent.sbatch`, commit cfe8b1f7, frozen `output_dir/sasrec_latent_code/code_cfe8b1f7`. Prep table: `output_dir/sasrec_latent/prep/Toys_sasrec_sequence.pt` (67,373 × 128).

**Prep finding:** the SASRec vector nearly encodes the answer on training rows but not on test rows. Cosine retrieval over all items, history not excluded:

| split | top-1 | top-10 |
|---|---:|---:|
| train | 0.245 | **0.835** |
| valid | 0.043 | 0.104 |
| test | 0.032 | 0.100 |

Because the vectors are in-sample on training rows, the decoder sees a far more reliable signal in training than at test. This is the likely over-trust mechanism behind "used but no gain" (the Musical run was in-sample too). Expected: strong reliance (shuffle hurts), ranking ≤ R3 (0.1023). If confirmed, the next step is out-of-fold SASRec vectors for training rows.

Submitted via JobManager as job **354714** (hadylauwresearch, a40, running since 2026-11-06).

### v2 result: staged 1 latent-only + 2 full (353831, n = 6,738, K = 16)
| quantity (mean / median / p90) | staged 1+2, bank swap | plain R3 null, bank swap | staged 1+2, zero |
|---|---|---|---|
| D, nats | 2.20 / 1.98 / 3.94 | −0.0003 / −0.00004 / 0.044 | 1.55 / 1.28 / 3.81 |
| KL, summed over title tokens | 1.70 / 1.41 / 3.01 | 0.0010 / 0.0008 / 0.0019 | 1.42 / 1.11 / 2.79 |
| argmax change share | 0.050 | 0.0009 | 0.035 |
| rows with D > 1 nat | 83% | 0% | 57% |
- Further numbers:
  - own percentile among donors 0.93 (R3 0.50);
  - logp(y_hat): own −3.96 vs donor mean −6.16;
  - D per title token 0.11;
  - rows with D_mean > 0.5: 95%;
  - D higher on rows whose top-1 is correct (2.91 vs 2.18).
- Reading: per sample, the staged model's output distribution depends strongly on which latent fills the slot. This is about 2,000× the R3 null, and a donor latent costs more than the zero vector.
- Caveat: D measures sensitivity to r, not whether r carries correct information, because y_hat is the own-r argmax. Whether the own r helps the true target is measured by the mean / shuffle_user HR tests (353734 rerun) or by a target-forced D (not implemented).
- The predictions file eval_raw.json was written by 353734's first attempt; that ranking is deterministic and is rewritten unchanged on rerun.

### v2 result: staged 2 latent-only + 1 full (353832), and latent bank diversity
- Staged 2+1, bank swap:
  - D 0.0069 / median 0.0047 / p90 0.077; KL 0.0069; no row with D > 1;
  - own percentile 0.53.
- Staged 2+1, zero vector: D 5.50 (all rows > 1).
- Reading: 2+1 needs a vector in the slot but ignores which one, like R3, despite 2 latent-only epochs.
- Bank diversity (train latents; 5,000 random pairs):

| model | norm mean ± sd | mean pairwise cosine | ‖r − mean‖ / ‖r‖ | v2 D |
|---|---|---|---|---|
| plain R3 (R3_ori) | 272.7 ± 0.6 | 1.0000 | 0.0000 | −0.0003 |
| staged 1+2 | 128.9 ± 11.4 | 0.20 | 0.89 | 2.20 |
| staged 2+1 | 158.3 ± 29.5 | 0.88 | 0.33 | 0.007 |
| history pooling | 82.3 ± 19.6 | 0.38 | 0.80 | pending (353953) |

- Caveat on the R3 null: R3_ori's r is the same vector for every user (template collapse), so its v2 null is degenerate; any swap is a no-op. Staged 2+1 is the informative null: its latents vary (‖r − mean‖ is 33% of ‖r‖), yet D ≈ 0.007, so its decoder ignores the variation.
- Staged 1+2 is the only model so far with both diverse latents and strong per-sample decoder sensitivity.

### History pooling: corrected interventions + v2 (353953, code 9f8a582) — content unused
- Paired bootstrap, raw computed with the history-only mask (raw HR@10 0.1046 / NDCG@10 0.0654):
  - raw − mask: +0.0095 [+0.0052, +0.0138] sig / +0.0114 sig;
  - raw − mean: +0.0013 [−0.0003, +0.0030] n.s. / +0.0007 [−0.0001, +0.0014] n.s.;
  - raw − shuffle_user: +0.0012 n.s. / +0.0002 n.s.
- v2, bank swap:
  - D 0.0095 (median 0.0053, p90 0.076); KL 0.0049; argmax change 0.0017; no row with D > 1;
  - own percentile 0.54;
  - only 8.9% of shuffle_user top-1s differ; AUROC of D for that change 0.58.
- v2, zero vector: D 2.02, KL 1.94.
- Verdict: pooling over history titles makes r user-specific (pair cos 0.38), but the decoder ignores its content (like staged 2+1). Success criterion not met.

### Item oracle: charac done (353431, 2026-11-06)

Preempted once; finished on a 3090. Best epoch 3; eval loss per epoch 0.307 / 0.090 / 0.089. trainer_state.json was not copied because the checkpoints were already gone; the values come from the log. n = 6,738, @20 = @10.

**Ranking (HR@1/3/5/10, NDCG@10):**
- oracle 0.9026 / 0.9354 / 0.9435 / **0.9482**, 0.9272
- swaphist 0.9461 HR@10 (−0.0021, significant)
- noise 0.25 / 0.5 / 0.75 / 0.9, HR@10: 0.9486 / 0.9475 / 0.9360 / 0.7591
- nn:1 / 5 / 20, HR@10: 0.6150 / 0.3019 / 0.0831
- pred1 0.0755
- shuffle 0.0030

**Seen / unseen-target HR@10 (unseen = 338 rows):** oracle 0.9911 / **0.0118**. For comparison: r3title 0.2456, qwen3 0.0562, R3 0.0030. The item the vector encodes is the top-1 output in 88–96% of rows across modes.

nn:1 still reaches 0.61 HR@10. In characteristics space the nearest item is often a variant of the same product (same description), and the decoder then lists the target among its neighbours.

**Reading:** pure lookup. The item's characteristics, without its title, let the decoder spell almost no unseen title. Ranking of the three kinds as a diffusion target: r3title > qwen3 > charac.

### Item oracle: sasrec done (353430, 2026-11-06); all four kinds done

Preempted once; the resume trained 0 steps. Final model = checkpoint-844 (epoch 2, eval loss 0.0708). Epoch 1 eval loss 0.3324; epoch 3 was never evaluated. n = 6,738, @20 = @10.

**Ranking (HR@1/3/5/10, NDCG@10):**
- oracle 0.9240 / 0.9350 / 0.9392 / **0.9433**, 0.9335
- swaphist 0.9396 HR@10 (−0.0037, significant)
- noise 0.25 / 0.5 / 0.75 / 0.9, HR@10: 0.9432 / 0.8966 / **0.3973** / 0.0800
- nn:1 / 5 / 20, HR@10: 0.2014 / 0.0698 / 0.0230
- pred1 0.0662
- shuffle 0.0043

**Seen / unseen-target HR@10:** oracle 0.9855 / 0.0207. Unseen items have untrained SASRec embeddings, so a near-zero unseen score is expected. The item the vector encodes is the top-1 output in 91–98% of rows (except under noise).

**Summary of all four kinds (oracle HR@10 / unseen HR@10 / HR@10 at noise 0.75 / HR@10 at nn:1):**

| kind | oracle | unseen | noise 0.75 | nn:1 |
|---|---:|---:|---:|---:|
| r3title | 0.9617 | 0.246 | ≈0.94 | ≈0.69 |
| qwen3 | 0.9512 | 0.056 | 0.9424 | 0.6293 |
| charac | 0.9482 | 0.012 | 0.9360 | 0.6150 |
| sasrec | 0.9433 | 0.021 | 0.3973 | 0.2014 |

**Reading:** every kind is copied as a nearest-item lookup. Only the R3 title space generalises somewhat to unseen titles. SASRec space is the least tolerant of noise or neighbour substitution: its neighbours are not title variants, and its 128 dimensions are densely used. As a diffusion target: r3title > qwen3 > charac > sasrec.

**Data note:** the `latent/checkpoint-*` dirs of all four item-oracle runs disappeared around 2026-11-06 00:12–00:13 (final model files remain). Not caused by this session. trainer_state values come from logs or earlier reads.

### Toys `sasrec_raw` results (354714, done 2026-11-06)

6h09m on an A40, no preemption. Best epoch 2; eval loss per epoch 0.7250 / **0.5769** / 0.6244.

**Ranking** (n = 6,738, @20 = @10):

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| own SASRec vector | 0.0367 | 0.0608 | 0.0779 | **0.1049** | 0.0504 | 0.0574 | 0.0661 |
| zero vector (drop) | 0.0362 | 0.0607 | 0.0785 | 0.1028 | 0.0504 | 0.0577 | 0.0655 |
| another user's vector | 0.0327 | 0.0557 | 0.0709 | 0.0948 | 0.0459 | 0.0521 | 0.0598 |
| plain R3 retrain (guidance A) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

**Paired bootstrap:**
- own − drop: HR@10 +0.0021 [−0.0007, +0.0049], NDCG@10 +0.0006; both n.s.
- own − shuffle: HR@10 **+0.0101** [+0.0065, +0.0138], NDCG@10 +0.0063 [+0.0043, +0.0082]; both significant.
- plain R3 − own: HR@10 +0.0003 [−0.0039, +0.0045], NDCG@10 +0.0006; both n.s.

**Split by whether SASRec has the target in its top-10** (cosine retrieval over the test vectors, history excluded; 602 / 6,738 rows):

| rows | own | drop | shuffle | plain R3 |
|---|---:|---:|---:|---:|
| SASRec hit (602) | 0.606 | 0.603 | 0.560 | 0.611 |
| SASRec miss (6,136) | 0.0557 | 0.0538 | 0.0492 | 0.0556 |

own − plain R3 on hit rows: −0.005 [−0.028, +0.022]. own − shuffle on hit rows: +0.047 [+0.027, +0.068].

**Reading:**
1. The ranking equals plain R3. The model gains nothing even on the 602 rows where SASRec has the target in its top-10; R3 still misses 39% of those.
2. The model is sensitive to a wrong vector (shuffle −0.010 overall, −0.047 on hit rows) but not to a missing one (drop n.s.). So it reads the SASRec vector as a consistency check against the titles, not as a source of candidates.
3. The over-trust prediction (own < plain R3) is not supported, even though the training vectors are in-sample (top-10 hit rate 0.835 on train vs 0.100 on test).
4. Reliance is weaker than on Musical (−27% on a subset there, −10% relative here).
5. Contrast with the item oracle: the target item's own SASRec embedding is copied (0.943). A user sequence vector would need a retrieval step (argmax over items of h_u·e_i) that the decoder does not learn.

Results are in `output_dir/sasrec_latent/Toys_and_Games_sasrec_raw/`.

**B2 354390 update (2026-11-06):** training done on an A100. Eval loss per epoch 1.2214 / **1.0331** / 1.2125, so the best is epoch 2 (checkpoint-768). Preempted at 08:38 during ranking (425/770); requeued, and training will be skipped on restart. C2 354567 and A2 354497 are running.

## R3_ori CDs latent-utilization check (requested 2026-11-06)

User request: check latent utilization of `R3_ori/output_dir/CDs_and_Vinyl/latent`. This is an eval-only run of `attention_share.sbatch` (SKIP_TRAIN=1, code cfe8b1f7), with a scratch task dir whose `latent` is a symlink to the checkpoint.

It runs ranking in this codebase (matched with the A2/B2/C2 evals), the `r` interventions mask / mean / shuffle_user with paired bootstrap, and the operator trace with the corrected pooling split.

Submitted via JobManager as job **355050** (moved to hadylauwresearch, a40, 2026-11-06).

## WIDER regulariser on R3, K = 1 (2026-11-06)

User request: apply WIDER (arXiv 2609.07406) to R3, starting with K = 1. With one latent per example, WIDER's within-example penalty is a constant (no gradient). The cross-user form is used instead:

- z_b = r_b / ‖r_b‖ on the pooled r;
- q = EMA (momentum 0.99) of the batch-mean unit direction, stop-grad. The EMA is used because the micro-batch is only 4;
- L = CE + 0.1 · mean_b (z_b · q)², with λ as in the paper.

Training only; eval loss and inference are plain R3.

**Code:** `src/latent/wider_reg.py`, hooks in `layers.py` (generate_embs stores r, forward adds the penalty), `--wider_weight` / `--wider_momentum`, `scripts/latent_rank.py`, `scripts/wider.sbatch`. Commit b8ce8f79, frozen `output_dir/wider_code/code_b8ce8f79`.

**CPU check** (Qwen2.5-0.5B, 4 rows): the penalty adds 0.1 × 0.737 to the training loss and changes the pooling gradients (query-weight grad norm 2.70 → 2.50). Eval loss is identical to no-reg.

**Geometry baselines** (`latent_rank.py`, train banks, unit-normalised r):

| bank | effective rank | top-1 energy | mean cos to mean direction | residual share |
|---|---:|---:|---:|---:|
| R3_ori Toys | 1.00 | 1.000 | 1.000 | 0.00004 |
| Toys v0 (ckpt 844) | 15.1 | 0.565 | 0.734 | 0.671 |
| history-only pooling | 64.2 | 0.391 | 0.614 | 0.795 |

**Run:** Toys, from base, R3 recipe, hadylauwresearch a40. Ranking, interventions (mask/mean/shuffle_user) with bootstrap, geometry, trace. Controls: plain R3 retrain guidance A (HR@10 0.1052) and R3_ori Toys (0.1023).

**Expectation:** `r` becomes diverse (rank up). Whether it gets used is the open question; earlier diversity-only methods were not used.

Submitted via JobManager as job **355111** (hadylauwresearch a40, pending 2026-11-06).

### A2 (fixed share 0.3, CDs) ranking (354497, eval_raw done before the preemption)

HR@1/3/5/10 0.0473 / 0.0687 / 0.0786 / **0.0968**; NDCG@3/5/10 0.0598 / 0.0639 / 0.0697 (n = 6,157; @20 = @10).

Paired bootstrap against R3_ori CDs stage 1 (`latent/final_result.json`; HR@10 0.0989 under the bootstrap metric, 0.0996 in its metric.txt): HR@10 −0.0021 [−0.0065, +0.0023], NDCG@10 −0.0020 [−0.0045, +0.0007], both n.s.

**Caveat:** `R3_ori/output_dir/CDs_and_Vinyl/{0,1}.json` (written 17:18, after the stage-1 final_result at 15:49) score HR@10 0.1095 with 60 unknown predictions. They are probably the stage-2 (GRPO) eval, not this checkpoint. Use final_result.json as the stage-1 reference.

Interventions are pending; the requeue reruns them.

**355111 started (10:07 on alarm, A40).** [wider] mean cos to the shared direction: 0.910 at call 1, 0.238 at call 101, 0.250 at call 201; penalty 0.827 → 0.064. The penalty is satisfied within about 6 optimiser steps. Pace about 10.5 s/step, roughly 3h40m for 1,263 steps.

**`sasrec_raw` vs R3: hit overlap (2026-11-06).**

| pair | Jaccard of hit rows | best-of-both HR@10 |
|---|---:|---:|
| `sasrec_raw` vs plain retrain A | 0.75 | 0.120 |
| two R3 trainings (A vs R3_ori) | 0.72 | 0.121 |
| any LLM vs Cornac SASRec | 0.41 | 0.137 |

Top-10 list overlap between `sasrec_raw` and A is 0.55, against 0.57 between the two R3 runs. SASRec-only hits: 217 rows missed by `sasrec_raw`, 214 by plain R3.

**Reading:** `sasrec_raw` differs from R3 only by retraining noise. The SASRec vector in r does not transfer SASRec's distinct hits.

### R3_ori CDs latent-utilization results (355050, done 2026-11-06 12:17)

2h41m on an A40, no preemption; R3_ori untouched. n = 6,157, @20 = @10.

**Ranking (HR@1/3/5/10; NDCG@3/5/10):**

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ranking in this codebase | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0617 | 0.0662 | 0.0717 |
| raw `r` | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0617 | 0.0662 | 0.0717 |
| mask (zero) | 0.0169 | 0.0383 | 0.0539 | 0.0731 | 0.0294 | 0.0358 | 0.0420 |
| mean | 0.0490 | 0.0703 | 0.0819 | 0.0991 | 0.0612 | 0.0659 | 0.0715 |
| shuffle_user | 0.0492 | 0.0702 | 0.0819 | 0.0991 | 0.0613 | 0.0661 | 0.0717 |

The ranking reproduces R3_ori's own eval (0.0996) within 0.0007.

**Paired bootstrap, raw minus mode:**
- mask: HR@10 +0.0258 [+0.0210, +0.0313], NDCG@10 +0.0297; both significant.
- mean: HR@10 −0.0002 [−0.0013, +0.0010]; n.s.
- shuffle_user: HR@10 −0.0002 [−0.0013, +0.0010]; n.s.

**Trace (fixed labelling):**
- pooling mass: template 0.996 / history titles 0.004;
- ‖r_A − r_B‖/‖r_A‖ 0.33;
- KL: swap 1.5e-4, random direction 1.2e-4, drop last item 0.107;
- target→r attention per layer: mean 0.9–6.7%, max head up to 22%.

**Reading:** the CDs R3 decoder needs something in the slot (mask −26% relative) but ignores which user's r it is. Content utilisation is zero, the same as on Musical and Toys. This is the matched reference for A2/B2/C2.

### A2 (fixed share 0.3, CDs) results (354497, done 2026-11-06 13:05)

Best epoch 2 (eval loss 1.0310); the final model equals checkpoint-768. n = 6,157, @20 = @10.

**Ranking (HR@1/3/5/10; NDCG@3/5/10):**
- eval_raw: 0.0473 / 0.0687 / 0.0786 / 0.0968; NDCG 0.0598 / 0.0639 / 0.0697
- intervention raw: 0.0471 / 0.0681 / 0.0793 / 0.0975; NDCG 0.0593 / 0.0639 / 0.0698
- mask: HR@10 0.0640, NDCG@10 0.0385
- mean: HR@10 0.0976, NDCG@10 0.0703
- shuffle_user: HR@10 0.0973, NDCG@10 0.0699

**Paired bootstrap, raw minus mode:**
- mask: HR@10 +0.0335 [+0.0276, +0.0393]; significant.
- mean: HR@10 −0.0002 [−0.0013, +0.0008]; n.s.
- shuffle_user: HR@10 +0.0002 [−0.0010, +0.0013]; n.s.

Against R3_ori CDs stage 1: HR@10 −0.0021 [−0.0065, +0.0023], n.s.

**Trace** (old prefix/user pooling labels, code 15cb6980):
- target→r attention about 0.27 in every layer (R3_ori 0.01–0.07);
- whole-r share of head output about 0.3;
- **user-specific share only 0.01–0.07**;
- **‖r_A − r_B‖/‖r_A‖ = 0.11** (R3_ori 0.33);
- KL: swap 1.3e-4 = random 1.1e-4, drop last item 0.097;
- gain_V 0.8–2.1, gain_O ≈ 1 (no suppression in the projections).

**Reading:** forced attention does not create content use. The model answered the guaranteed 27% share by making r more constant across users: the per-user residual fell from 33% to 11% of ‖r‖. r became a sink: heavily read, with nothing user-specific in it. This is the failure mode predicted before launch. Attention is not the bottleneck; redundancy is.

### C2 (attention-floor regulariser ρ̄ 0.3, λ 10, CDs) results (354567, done 2026-11-06 13:16)

Best epoch 2 (eval loss 1.0330); the final model equals checkpoint-768. n = 6,157.

**Ranking:** HR@1/3/5/10 0.0482 / 0.0697 / 0.0819 / **0.1004**; NDCG@3/5/10 0.0606 / 0.0656 / 0.0715.

**Interventions** (HR@10 / NDCG@10): mask 0.0754 / 0.0398; mean 0.1000 / 0.0715; shuffle_user 0.1000 / 0.0714.

**Paired bootstrap, raw minus mode:**
- mask: +0.0250 [+0.0198, +0.0304]; significant.
- mean: +0.0003 [−0.0010, +0.0016]; n.s.
- shuffle_user: +0.0003 [−0.0008, +0.0015]; n.s.

**Trace:**
- realised test-time attention to r 0.40–0.61 per layer (max head up to 0.98), well above the floor;
- **pooling 100% on prefix; ‖r_A − r_B‖/‖r_A‖ = 0.005** (R3_ori 0.33, A2 0.11);
- KL: swap 9e-5 = random 8e-5;
- user-specific share of head output below 1%.

**Reading:** the learned attention floor makes r a pure sink. r is near-identical for every user and attended at 40–60%, and ranking is unchanged. Both forced (A2) and learned (C2) attention lead the model to remove user content from r instead of reading it. Attention-share line: A2 and C2 negative; B2 pending.
C2 vs R3_ori CDs stage 1: HR@10 +0.0015 [−0.0031, +0.0058], NDCG@10 −0.0002; both n.s.

## Attention-share study (CDs): consolidated summary (written 2026-11-06)

### Question

Does forcing the decoder to attend to the latent `r` make predictions depend on `r`'s content? In plain R3 the decoder gives `r` about 1–7% of each head's attention, and swapping `r` for another user's changes nothing. The operator trace (354280) found the limiting step to be the decoder attention softmax plus residual dilution. These runs raise that attention without changing the prompt.

### Setup (all runs)

- **Data:** CDs_and_Vinyl. Test n = 6,157; constrained 10-beam ranking, so @20 = @10, and NDCG@1 = HR@1.
- **Training:** from base Qwen2.5-1.5B, R3 recipe (lr 3e-4, batch 128, 3 epochs, early stopping). Prompt unchanged.
- **Interventions:** `experiment.py generate` on test, with `r` replaced by zero (**mask**), the mean training `r` over 10k rows (**mean**), or another user's `r` (**shuffle_user**). Paired bootstrap, 2,000 resamples, seed 42.
- **How to read the interventions:**
  - mean/shuffle n.s. means `r`'s content is unused;
  - mask significant only means the slot must hold some vector.
- **Operator trace** (n = 300): attention from target positions to `r`; ‖r_A − r_B‖/‖r_A‖ (how much `r` differs between users); KL when swapping `r` vs a random-direction control.
- **Attention share** (`src/latent/attention_share.py`): at every position after `r`, in every layer and head,
  w'_tr = ρ, w'_tj = (1 − ρ)·a_tj/(1 − a_tr).

| run | job | mode | code | checkpoint |
|---|---|---|---|---|
| **R3** (reference, original pipeline) | 355050 (eval only) | none | R3_ori (eval with cfe8b1f7) | `/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/CDs_and_Vinyl/latent` |
| **A2** | 354497 | **fixed**: ρ = 0.3 for every head, train and test | 15cb6980 | `/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/CDs_and_Vinyl_attn_share_fixed_0.3/latent` (= checkpoint-768, epoch 2) |
| **B2** | 354390 | **budget**: a layer budget of 0.3 split across heads in proportion to their own attention to `r` (mean realised ≈ 0.28 after clipping), train and test | 15cb6980 | `/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/CDs_and_Vinyl_attn_share_budget_0.3/latent` (= checkpoint-768, epoch 2) |
| **C2** | 354567 | **regularize**: attention unchanged; training adds 10 · mean relu(0.3 − head-mean attention to `r`)² at target-predicting positions; inference is plain attention | 642a09fa | `/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/CDs_and_Vinyl_attn_share_regularize_0.3/latent` (= checkpoint-768, epoch 2) |

Result files:
- ranking: `R3/output_dir/attention_share/<run>/eval_raw.json`;
- interventions: `R3/analysis/results/intervention/<tag>/`;
- traces: `R3/analysis/results/latent_operator_trace/<tag>.json`.

Eval loss per epoch:
- A2: 1.2148 / **1.0310** / 1.2461
- B2: 1.2214 / **1.0331** / 1.2125
- C2: 1.2121 / **1.0330** / 1.1855

All three overfit in epoch 3, so epoch 2 is kept.

### Ranking (standard eval, constrained 10-beam)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0494 | 0.0617 | 0.0662 | 0.0717 |
| A2 fixed | 0.0473 | 0.0687 | 0.0786 | 0.0968 | 0.0473 | 0.0598 | 0.0639 | 0.0697 |
| B2 budget | 0.0471 | 0.0674 | 0.0797 | 0.0976 | 0.0471 | 0.0587 | 0.0637 | 0.0695 |
| C2 regularize | 0.0482 | 0.0697 | 0.0819 | 0.1004 | 0.0482 | 0.0606 | 0.0656 | 0.0715 |

Paired bootstrap, run minus R3 (R3 = `latent/final_result.json`, HR@10 0.0989):

| run | HR@10 [95% CI] | NDCG@10 [95% CI] | significant? |
|---|---|---|---|
| A2 | −0.0021 [−0.0065, +0.0023] | −0.0020 [−0.0045, +0.0007] | no |
| B2 | −0.0013 [−0.0057, +0.0031] | −0.0023 [−0.0049, +0.0004] | no |
| C2 | +0.0015 [−0.0031, +0.0058] | −0.0002 [−0.0030, +0.0027] | no |

*Comment:* none of the attention changes costs or gains ranking. They all sit at R3's level.

### Interventions on `r` (does its content matter?)

| run | condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 | raw | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0494 | 0.0617 | 0.0662 | 0.0717 |
| R3 | mask | 0.0169 | 0.0383 | 0.0539 | 0.0731 | 0.0169 | 0.0294 | 0.0358 | 0.0420 |
| R3 | mean | 0.0490 | 0.0703 | 0.0819 | 0.0991 | 0.0490 | 0.0612 | 0.0659 | 0.0715 |
| R3 | shuffle_user | 0.0492 | 0.0702 | 0.0819 | 0.0991 | 0.0492 | 0.0613 | 0.0661 | 0.0717 |
| A2 | raw | 0.0471 | 0.0681 | 0.0793 | 0.0975 | 0.0471 | 0.0593 | 0.0639 | 0.0698 |
| A2 | mask | 0.0195 | 0.0343 | 0.0443 | 0.0640 | 0.0195 | 0.0280 | 0.0321 | 0.0385 |
| A2 | mean | 0.0478 | 0.0695 | 0.0793 | 0.0976 | 0.0478 | 0.0604 | 0.0644 | 0.0703 |
| A2 | shuffle_user | 0.0476 | 0.0687 | 0.0786 | 0.0973 | 0.0476 | 0.0598 | 0.0639 | 0.0699 |
| C2 | raw | 0.0482 | 0.0697 | 0.0819 | 0.1004 | 0.0482 | 0.0606 | 0.0656 | 0.0715 |
| C2 | mask | 0.0094 | 0.0388 | 0.0528 | 0.0754 | 0.0094 | 0.0266 | 0.0324 | 0.0398 |
| C2 | mean | 0.0484 | 0.0694 | 0.0819 | 0.1000 | 0.0484 | 0.0605 | 0.0656 | 0.0715 |
| C2 | shuffle_user | 0.0481 | 0.0695 | 0.0825 | 0.1000 | 0.0481 | 0.0604 | 0.0657 | 0.0714 |
| B2 | raw | 0.0471 | 0.0677 | 0.0791 | 0.0978 | 0.0471 | 0.0589 | 0.0635 | 0.0696 |
| B2 | mask | 0.0198 | 0.0346 | 0.0474 | 0.0633 | 0.0198 | 0.0283 | 0.0335 | 0.0388 |
| B2 | mean | 0.0471 | 0.0681 | 0.0804 | 0.0978 | 0.0471 | 0.0592 | 0.0642 | 0.0698 |
| B2 | shuffle_user | 0.0474 | 0.0679 | 0.0806 | 0.0979 | 0.0474 | 0.0591 | 0.0643 | 0.0699 |

Paired bootstrap, raw minus condition, HR@10 / NDCG@10:

| run | mask | mean | shuffle_user |
|---|---|---|---|
| R3 | **+0.0258** [+0.0210, +0.0313] / **+0.0297** | −0.0002 [−0.0013, +0.0010] / +0.0002, n.s. | −0.0002 [−0.0013, +0.0010] / +0.0000, n.s. |
| A2 | **+0.0335** [+0.0276, +0.0393] / **+0.0313** | −0.0002 [−0.0013, +0.0008] / −0.0005, n.s. | +0.0002 [−0.0010, +0.0013] / −0.0001, n.s. |
| C2 | **+0.0250** [+0.0198, +0.0304] / **+0.0318** | +0.0003 [−0.0010, +0.0016] / +0.0000, n.s. | +0.0003 [−0.0008, +0.0015] / +0.0001, n.s. |
| B2 | **+0.0344** [+0.0287, +0.0401] / **+0.0308** [+0.0269, +0.0349] | +0.0000 [−0.0015, +0.0015] / −0.0002, n.s. | −0.0002 [−0.0016, +0.0013] / −0.0004, n.s. |

*Comment:* in every model, zeroing `r` hurts a lot, so the slot must hold a vector. But the mean `r`, or a different user's `r`, does exactly as well as the user's own. **`r`'s content is unused in every model**, including the two that attend to it heavily.

### Operator trace (n = 300)

| run | attention to `r` at target positions, per layer (mean over heads) | max head | ‖r_A − r_B‖/‖r_A‖ | share of head output from `r` | user-specific share of head output | KL swap `r` / random direction / drop last item |
|---|---|---|---:|---:|---:|---|
| R3 | 0.009–0.067 | 0.22 | **0.33** | small | 0.001–0.019 | 1.5e-4 / 1.2e-4 / 0.107 |
| A2 | 0.265–0.288 (forced) | 0.40 | **0.11** | ≈0.30 | 0.012–0.069 | 1.3e-4 / 1.1e-4 / 0.097 |
| C2 | 0.40–0.61 (learned) | 0.98 | **0.005** | ≈0.35–0.67 | 0.004–0.010 | 9e-5 / 8e-5 / 0.096 |
| B2 | budget ≈0.28 (forced) | — | **0.22** | 0.15–0.45 | 0.03–0.07 | 1.5e-4 / 1.2e-4 / 0.096 |

Pooling split:
- R3 (fixed labels): template 0.996 / history titles 0.004;
- A2 and C2 (old labels, code 15cb6980 / 642a09fa): prefix 0.881 / user 0.119 and 1.000 / 0.000. They are not directly comparable to R3's split, but C2's pooling is entirely on fixed tokens.

The value/output projections do not suppress `r` in any model (gain_V 0.8–2.9, gain_O ≈ 1–1.8).

*Comment:* forcing attention works mechanically: the decoder reads `r` heavily, up to 30–60% of each head's output. But training answers by **removing the user-specific part of `r`**: user variation falls from 33% of ‖r‖ (R3) to 11% (A2) to 0.5% (C2). The more attention is pushed onto `r`, the more constant `r` becomes. It ends as an attention sink: heavily attended, the same for everyone, and swapping it does no more than a random perturbation.

### Conclusion

1. **Attention is not the bottleneck.** Raising the decoder's attention to `r` from about 3% to 27% (A2, forced) or 40–60% (C2, learned) changes neither ranking (all n.s. vs R3) nor content use (mean/shuffle n.s. in every run).
2. **Forcing attention onto a redundant latent makes it more constant, not more read.** The model meets the attention constraint at no cost by turning `r` into a fixed vector (a sink). This is the failure mode predicted before launch, and it gets stronger the harder attention is pushed (A2 0.11 → C2 0.005 user variation).
3. **The root cause is redundancy.** `r` is computed from the same history the decoder reads directly, so the loss never rewards putting user content in `r`. Any method that only changes how `r` is read (attention share, floor penalties) leaves this in place. Use appears only when `r` carries information the history lacks (oracle category or item, or the L7 bottleneck).
4. **Status:** A2, B2 and C2 are all negative and the direction is closed for routing alone. B2 (354390, done 2026-11-06 after 3 preemptions; best epoch 2, eval loss 1.0331) matches A2: ranking n.s. vs R3, mask +0.034 HR@10 significant, mean/shuffle exactly neutral; trace (old labels) prefix 0.875 / user 0.125, ‖r_A − r_B‖/‖r_A‖ 0.22, swap KL = random-direction KL. The remaining combination is WIDER (keeps `r` diverse across users, job 355111) with forced attention (keeps it read). Even that leaves `r` redundant, so it is a diagnostic, not a likely method.

## 2026-11-06 — WIDER K = 1 result (355111, Toys)

Run: 7h44m on alarm (A40), no preemption. Eval loss by epoch 1.7516 / 0.9982 / **0.6687** (best = checkpoint-1263, epoch 3). Last [wider] line (call 40301): penalty 0.039, mean cos to the shared direction −0.19, |ema| 0.037 (the batch-mean direction has almost vanished). Checkpoint `scratchDirectory/R3/output_dir/Toys_and_Games_wider_0.1/latent/checkpoint-1263`; results `R3/output_dir/wider/Toys_and_Games_wider_0.1/`.

**Ranking and interventions** (n = 6,738, @20 = @10, NDCG@1 = HR@1):

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WIDER raw | 0.0234 | 0.0444 | 0.0571 | **0.0870** | 0.0354 | 0.0407 | 0.0503 |
| WIDER mask | 0.0212 | 0.0445 | 0.0582 | 0.0853 | 0.0347 | 0.0403 | 0.0491 |
| WIDER mean | 0.0233 | 0.0436 | 0.0591 | 0.0859 | 0.0351 | 0.0414 | 0.0501 |
| WIDER shuffle_user | 0.0236 | 0.0456 | 0.0576 | 0.0874 | 0.0362 | 0.0411 | 0.0507 |
| plain R3 retrain (guidance A) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap (2,000, seed 42):
- plain A − WIDER: HR@10 **+0.0183** [+0.0123, +0.0243], NDCG@10 **+0.0164** [+0.0126, +0.0203], significant (−17% HR@10).
- raw − mask: +0.0016 [−0.0019, +0.0052] / +0.0012; raw − mean: +0.0010 [−0.0021, +0.0042] / +0.0002; raw − shuffle: −0.0004 [−0.0025, +0.0016] / −0.0004. All n.s.

**Geometry** (`latent_rank.json`, train, n = 10,000) vs baselines: effective rank **106.9** (R3_ori 1.0, v0 15.1, history pooling 64.2), top-1 energy 0.096, mean cos to mean direction 0.030, mean pairwise cos 0.002, residual share 1.00, mean norm 123.

**Operator trace** (n = 300): pooling template 0.000 / history titles 1.000; ‖r_A − r_B‖/‖r_A‖ **1.47** (R3 0.33); KL swap r 0.0022 (ΔNLL −0.0016), KL random-direction r **0.112**, KL drop last item 0.049; relative logit change at targets swap 0.026, random 0.116.

**Reading:**
1. WIDER does what it promises on geometry: `r` goes from a near-constant to a nearly isotropic, user-specific vector (rank 107, pairwise cos 0.002), and pooling moves entirely onto history titles.
2. The decoder answers by **ignoring `r` altogether**: even mask is n.s. (the first run where zeroing `r` doesn't hurt). Swapping in another user's `r` moves the output 50× less than a random direction of the same norm, so the decoder has learned to be insensitive to the subspace where user `r`'s live, while staying sensitive off it.
3. Ranking drops 17%, the largest loss in this line. Plausible cause: the constant part of `r` that R3 uses as an anchor (mask hurts in plain R3) is removed by the penalty, and the decoder has to work around a high-norm, diverse, uninformative token.
4. Together with the attention-share study, this brackets the problem: forcing *reading* makes `r` constant (C2); forcing *diversity* makes `r` unread (WIDER). Neither creates content use, because `r` = f(history) adds no information. **WIDER K = 1 closed.** WIDER + forced attention is not recommended (expected: the decoder would read a diverse but redundant vector and lose ranking).

### WIDER K = 1: implementation check (2026-11-06, user asked) — **bug: penalty 32× too strong**

- `LatentModel.forward` takes `**kwargs`, so HF 4.48 Trainer sets `model_accepts_loss_kwargs = True`: the CE is normalised by the token count of the whole accumulated batch (`num_items_in_batch`) and the Trainer does **not** divide the returned loss by `gradient_accumulation_steps` (trainer.py `training_step`). The WIDER penalty is a per-micro-batch mean added at full weight, so over 32 micro-batches (128 / 4) the effective weight is 32 × 0.1 = **3.2**, not 0.1.
- Confirmed in the log: step-1 train loss 7.13 vs 4.70–4.72 for plain Toys runs (sasrec_raw 354714, guidance A 345857); the gap 2.4 ≈ 32 × 0.1 × 0.76 (penalty at the start).
- The same scaling applies to the C2 attention floor (`attention_floor_penalty`, weight 10 → effective 320). MarginLatentModel already corrects for this (`grad_accum_steps`).
- Other checks: penalty on the pooled r before observe/project (identity in plain R3, so = the spliced r); q stop-gradient; EMA updated only in training forwards (40,301 calls ≈ 1,263 steps × 32); eval loss and inference untouched. Design deviation (known, by construction): the paper's q(x) is per example over m = 6 latent steps; at K = 1 it degenerates (L = 1), so the cross-user EMA form was used. Cross-user zero-mean is a stronger constraint than within-trajectory diversity: it also removes the shared component of r that R3 uses as an anchor.
- **Consequence:** the 355111 result (r ignored, −17% HR@10) is for λ_eff = 3.2, above the paper's over-regularisation point (λ = 0.20). The "WIDER closed" conclusion above is withdrawn until a corrected run (λ_eff = 0.1) is done.

**Fix and rerun (2026-11-06, user: "fix it and rerun, delete the old ckpt").** Commit 001a5a2a: WIDER penalty and attention floor multiplied by `loss_scale` = 1/gradient_accumulation_steps (config key; old checkpoints without it = 1). Frozen `output_dir/wider_code/code_001a5a2a`. Old scratch checkpoint `scratchDirectory/R3/output_dir/Toys_and_Games_wider_0.1` deleted (8.8 GB; results stay in `output_dir/wider/Toys_and_Games_wider_0.1`). Rerun `Toys_and_Games_wider_0.1_fixed` (λ_eff = 0.1) = job **355749** (JobManager, pending); expected step-1 train loss ≈ 4.8. C2 not rerun (not requested).

**355749 started** (alarm, A40). Config shows loss_scale 0.03125. Step-1 train loss 4.836 ≈ plain 4.72 + 0.1 × 0.92 (first penalty) → fix verified (buggy run: 7.13). [wider] call 1: penalty 0.922, mean cos 0.961.

## 2026-11-07 — WIDER K = 1, corrected (355749, λ_eff = 0.1, code 001a5a2a)

Run 7h41m on alarm (A40), no preemption. Eval loss 0.7283 / **0.5762** / 0.6258 (best checkpoint-844, epoch 2; buggy run 1.7516 / 0.9982 / 0.6687). Last [wider] line: penalty 0.0043, mean cos to shared direction 0.031. Checkpoint `scratchDirectory/R3/output_dir/Toys_and_Games_wider_0.1_fixed/latent`; results `R3/output_dir/wider/Toys_and_Games_wider_0.1_fixed/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WIDER fixed raw | 0.0350 | 0.0628 | 0.0790 | **0.1070** | 0.0512 | 0.0578 | 0.0669 |
| mask | 0.0344 | 0.0628 | 0.0798 | 0.1061 | 0.0508 | 0.0578 | 0.0663 |
| mean | 0.0346 | 0.0632 | 0.0791 | 0.1077 | 0.0512 | 0.0577 | 0.0669 |
| shuffle_user | 0.0355 | 0.0634 | 0.0791 | 0.1079 | 0.0515 | 0.0579 | 0.0672 |
| plain R3 retrain (guidance A) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |
| buggy WIDER 355111 (λ_eff 3.2) | 0.0234 | 0.0444 | 0.0571 | 0.0870 | 0.0354 | 0.0407 | 0.0503 |

(@20 = @10, NDCG@1 = HR@1.) Paired bootstrap: WIDER fixed − plain A HR@10 +0.0018 [−0.0025, +0.0061], NDCG@10 +0.0002 [−0.0021, +0.0025], n.s. raw − mask +0.0009 [−0.0009, +0.0027]; raw − mean −0.0007 [−0.0024, +0.0009]; raw − shuffle −0.0009 [−0.0028, +0.0009]; all n.s.


Geometry: effective rank **232.6** (buggy 106.9, R3_ori 1.0), top-1 energy 0.051, mean cos to mean direction 0.046, pairwise cos 0.002, mean norm 89.8. Trace: pooling 100% on history titles; ‖r_A − r_B‖/‖r_A‖ 1.46; KL swap 0.0014 vs random direction 0.0021 vs drop last item 0.067.

**Reading:** with the correct weight the ranking cost disappears (≈ plain R3, n.s.), so the −17% of 355111 was the bug. `r` is now fully user-specific (rank 233, no shared direction), but the decoder still ignores it: mask, mean and shuffle all n.s., and a swap moves the output less than dropping one history item (0.0014 vs 0.067). Diversity of `r` does not create use of `r`; consistent with redundancy (r = f(history)). **WIDER K = 1 closed (corrected).**

## 2026-11-07 — Why the L7 bottleneck loses ranking: row-level comparison with plain R3 (Musical, analysis only)

Files: `analysis/results/bottleneck_vs_r3/{compare.py,output.txt,rows.csv}`. L7 latent-only (325712, `analysis/results/intervention/musical_instruments_latent_only`, raw HR@10 0.1089) vs plain R3 v0 checkpoint-1038 (same experiment.py pipeline, raw HR@10 0.1219); test n = 8,313.

- Hits: both 0.0909, R3-only 0.0309, L7-only 0.0179, neither 0.860; Jaccard 0.65. Top-10 overlap 0.35, same top-1 0.36.
- **The whole gap (−0.013 HR@10) is on "copy" rows.** Target bought the same day as the last history item: R3 0.200 vs L7 0.173 (−0.027, contributes −0.013); other rows 0.0500 vs 0.0500 (exactly 0). Target shares brand with the last item: −0.074 on 12.7% of rows (−0.009 of the gap). Same day or same brand (50% of rows): −0.014; the other 50%: L7 +0.002.
- R3-only hits are sibling variants of a history item (e.g. history "Fred Kelly D2-L-8 … Light Pick" → target "Fred Kelly D3-L-8 …"; "Ernie Ball 2223 Super Slinky" → "Ernie Ball 2221 Regular Slinky"): mean max title-token Jaccard to history 0.24 (all rows 0.13), 40% same brand as the last item, 60% same day. L7-only hits are popular, non-copy targets (mean train count 191 vs 81).
- Top-10 behaviour R3 vs L7: share sharing a brand with the history 0.48 vs 0.35; items from the history 0.11 vs 0.06; title similarity to history 0.25 vs 0.18; popularity of predicted items 132 vs 193 (top-1 227 vs 314).

**Reading:** the bottleneck doesn't lose general preference prediction; it loses **copying of exact item identities** (brand + model number variants, same-session purchases). A single pooled vector carries "what kind of item" but not the exact title tokens of history items, so L7 falls back to popular items of the right kind. R3's advantage over L7 is the same-day variant artifact (cf. memory: 10% same-day same-brand rows give 57% of R3's hits). Implication: to close L7's gap, the latent route needs a token-level copy channel (more latent tokens, or `r` selecting a history item whose title the decoder may read), not a better summary vector.

## 2026-11-07 — Margin loss + WIDER (Musical, user side task)

Counterfactual margin loss of Attempt 2 (−log P(y|x, r) + 0.2 ≤ −log P(y|x, r_other), weight 1.0; `margin_loss_attention.py`) plus the cross-user WIDER penalty (λ = 0.1) on the pooled r. MarginLatentModel's training branch bypasses `LatentModel.forward`, so the penalty is added inside it, before its own division by grad_accum_steps (so no loss_scale; the 32× bug does not apply). CPU check (Qwen2.5-0.5B, 4 rows, accum 4): loss +0.025 = 0.1 × 1.0 / 4 vs no-WIDER. Commit 6fac3d4e, frozen `output_dir/margin_wider_code/code_6fac3d4e`, `scripts/margin_wider.sbatch` (Qwen2.5-1.5B-Instruct as in the Musical margin run, hadylauwresearch a40, ranking + interventions + geometry + trace). References: plain R3 v1 Musical 0.1226 / 0.0950; margin-only (old pipeline, 304312) HR@10 0.1176. Margin-only control with this pipeline = same script with WEIGHT=0 (not submitted). Job **356320** (JobManager, pending 2026-11-07).

**Changed to Toys (user, 2026-11-07):** 356320 (Musical) cancelled before start; resubmitted with CATEGORY=Toys_and_Games, BASE_MODEL=Qwen/Qwen2.5-1.5B (matches Toys controls: plain retrain A 0.1052, WIDER-fixed 355749 0.1070). Results → output_dir/margin_wider/Toys_and_Games_margin_0.2_wider_0.1/. Job **356322** (pending).

**356322 started 2026-11-07 16:01 (album, A40).** WIDER config OK; ce/margin/wider logged. Step-1 'loss' 4.975 vs ce+margin+0.1·wider 4.469 was flagged by JobManager, but the components are from the last micro-batch (4 rows) and 'loss' is the 32-micro-batch mean: gap over steps 1–5 = +0.51, +0.37, +0.58, −0.63, −0.43 (mean +0.08, sd 0.51) → sampling noise, no scaling bug (a 32× bug would add ≈ +2.9). Kept running.

## 2026-11-07 — Item-transition reading latent (idea 2, agreed with user; Toys)

**Idea:** give r information the history lacks: what other users bought right after the row's recent items. Transition graph from the train rows (a → b when b was bought right after a; one count per user/position), **time-causal** (a row only sees transitions before its target time; Toys is time-split, so training and test rows follow the same rule and no row sees its own answer). Neighbour set: top-10 successors of each of the last 5 history items, minus history items, ≤ 50.

**Pre-check** (`transition_prep.py`, `output_dir/transition_latent/prep/Toys_transitions.pt`): set size train 14.7 / valid 25.4 / test 26.2; target covered train 0.0995, valid 0.1049, test 0.1205. Among plain-R3 (guidance A) misses, test coverage 0.057; among hits 0.639. Ceiling for a perfect reader ≈ +0.05 HR@10.

**Model** (`transition_latent.py`, fp32 reader): q = W_q r (d = 256); s_n = q·W_k e_n/√d + β·log(1+c_n) + b[src_n]; a = masked softmax; m = Σ a_n e_n (e = R3 title vectors, `output_dir/candidate_pooling/prep/Toys_r3_title_vectors.pt`); r' = r + g·normalize(P m) (g init 30; zero if the set is empty). Prompt unchanged. Interventions: none (r' = r), shuffle (another user's neighbour set), shuffle_r (r' of another prompt in the batch). CPU check (Qwen2.5-0.5B, 4 rows): gradients reach the reader, weights sum to 1, none differs from raw by g, shuffle_r = rolled raw.

Commit e5ce8f85, frozen `output_dir/transition_latent_code/code_e5ce8f85`, `scripts/transition_latent.sbatch` (base Qwen2.5-1.5B, R3 recipe, hadylauwresearch a40). Smoke **356387** (512 rows, 1 epoch) → full **356388** (afterok), via JobManager. Controls: plain R3 retrain A 0.1052; success = HR@10 significantly above it and shuffle / shuffle_r significantly worse than raw. Diagnostics after the run: HR split by coverage, reader weight by source position and concentration (stored per row as `candidate_weights` in eval JSONs).

**Smoke 356387 failed** at the epoch-end eval (KeyError: no eval_loss): HF infers label_names from forward's signature, and TransitionLatentModel.forward(*args, trans_*, **kwargs) hides 'labels'. Fix: label_names=['labels'] for transition runs (commit 8ec78f7c, frozen code_8ec78f7c). 356388 cancelled; resubmitted as smoke **356390** → full **356391** (afterok). Training itself ran (loss 4.75 → 3.23 over 4 steps).

**Smoke 356390 passed** (4m42s): eval_loss logged (3.366), all four evals written (64 rows), bootstrap runs. Full **356391** running on album since the smoke finished.

### Transition latent 356391: partial result (raw / none done; shuffle, shuffle_r running) — 2026-11-07

Eval loss 0.7285 / **0.5720** / 0.6176 (best epoch 2).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| transition raw | 0.0356 | 0.0614 | 0.0772 | 0.1052 | 0.0506 | 0.0572 | 0.0662 |
| transition none (no neighbours) | 0.0352 | 0.0622 | 0.0767 | 0.1042 | 0.0509 | 0.0569 | 0.0658 |
| plain R3 retrain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Bootstrap: raw − none HR@10 +0.0010 [−0.0004, +0.0027] n.s.; raw − plain A +0.0000 [−0.0042, +0.0043] n.s. Covered rows (812): raw 0.560, none 0.560, plain A 0.567; uncovered 0.043 / 0.042 / 0.042.

**Diagnosis: the reader never learned.** Learned scalars are at their init (gain 30.02 vs 30, β 0.997 vs 1, source biases ≈ 0). The reader's softmax is saturated from the start (mean max weight 0.97, normalised entropy 0.02): title vectors have norm ≈ 65, so random W_q/W_k give large logits and an arbitrary near-one-hot pick with vanishing gradient. The top-weighted neighbour is the target on 4.2% of covered rows (chance with ~26 neighbours ≈ 4%), and is the highest-count neighbour only 4.6% of the time. So the decoder received a random neighbour's vector and learned to ignore it. This is an implementation flaw, not a test of the idea. Fix (proposed): normalise e before the key projection and standardise the scores over the set with a learned temperature (as candidate_pooling.py does), so the reader starts near uniform and the count prior can act.

**Reader v2** (commit ecf38f1a, frozen code_ecf38f1a): keys from unit-normalised title vectors; content score standardised over the row's neighbours × learned temperature τ (init 1); std with ε inside the sqrt (rows with one neighbour gave inf gradients). CPU check on 64 train rows with real title vectors and r at pooled scale: at init mean max weight 0.41 (v1 0.97), normalised entropy 0.76 (v1 0.02); gradients reach τ, β, key, query, source biases. v1 run 356391 left to finish its shuffle evals. Smoke **357274** → full **357275** (Toys_and_Games_transition_latent_v2, afterok), via JobManager.

## 2026-11-07 — CARE-style input choice for the transition latent (user request)

Borrowed from CARE (Conflict-Aware Soft Prompting for RAG, arXiv 2508.15253): give the injected signal where the model fails without it, and a misleading one where it already succeeds, so the decoder learns selective trust. Closest earlier run: Attempt 4 (CASE-inspired, CDs) poisoned r on every row, but r carried no information; mean/shuffle n.s., poison loss ≈ main loss.

**Implementation** (`transition_latent.py`, training only; eval unchanged): before each training micro-batch, a no-grad probe of the current model **without** neighbours (r' = r) gives per-token target NLL per row. Rows below a running median (EMA 0.99 of batch medians) count as "solved"; each solved row gets, with probability 0.5, another row's neighbour set (roll within the micro-batch); other rows keep their real neighbours. The model's own current competence defines "solved", so no separate R3 scoring of the training rows is needed. Cost: one extra no-grad forward (+pooling pass) per micro-batch. CPU check (Qwen2.5-0.5B, 4 rows, neg_prob 1): solved row swapped, others kept, reader gets gradient.

Commit 557abaff (on top of reader v2), frozen `output_dir/transition_latent_code/code_557abaff`, `CARE_NEG_PROB=0.5`. Smoke **357376** → full **357377** (`Toys_and_Games_transition_care`, afterok); runs alongside v2 (357274/357275), which is its direct control.

**v1 356391 complete** (6h54m, best epoch 2). shuffle (another user's neighbours): HR@1/3/5/10 0.0352/0.0617/0.0764/0.1048, NDCG@3/5/10 0.0507/0.0568/0.0660; shuffle_r (another prompt's r'): 0.0358/0.0617/0.0769/0.1052, 0.0508/0.0570/0.0662. raw − shuffle HR@10 +0.0004 [−0.0010, +0.0019]; raw − shuffle_r +0.0000 [−0.0016, +0.0016]; all n.s. → v1 = plain R3, neighbours and r' content unused (reader saturated, see diagnosis). Results `output_dir/transition_latent/Toys_and_Games_transition_latent/`.

**v2 smoke 357274 passed** (5 min; eval_loss 3.331, all four evals written). Full 357275 queued for an A40. CARE smoke 357376 started.

**CARE smoke 357376 passed** (4m54s): [care] call 1 solved 0.25 / neg 0.25 / threshold 4.96; eval_loss 3.905 (v2 smoke 3.331; eval itself never uses CARE negatives, so the gap is a 512-row 1-epoch training effect, not an eval artefact); all four evals written. Full 357377 queued.

## 2026-11-08 — Margin + WIDER result (356322, Toys)

9h23m on album, no preemption. Eval loss 0.8882 / 0.6180 / **0.6054** (best checkpoint-1263, epoch 3). Last [wider]: penalty 0.069, mean cos to shared direction 0.246. Results `output_dir/margin_wider/Toys_and_Games_margin_0.2_wider_0.1/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0373 | 0.0634 | 0.0803 | **0.1030** | 0.0525 | 0.0595 | 0.0668 |
| mask | 0.0123 | 0.0278 | 0.0356 | 0.0522 | 0.0212 | 0.0244 | 0.0298 |
| mean | 0.0215 | 0.0402 | 0.0499 | 0.0721 | 0.0324 | 0.0363 | 0.0435 |
| shuffle_user | 0.0068 | 0.0138 | 0.0184 | 0.0297 | 0.0108 | 0.0127 | 0.0163 |
| plain R3 retrain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |
| WIDER-fixed 355749 | 0.0350 | 0.0628 | 0.0790 | 0.1070 | 0.0512 | 0.0578 | 0.0669 |

Bootstrap: raw − mask HR@10 **+0.0508** [+0.0450, +0.0567]; raw − mean **+0.0309** [+0.0260, +0.0358]; raw − shuffle **+0.0733** [+0.0668, +0.0801] (NDCG@10 +0.037 / +0.023 / +0.051, all significant). vs plain A: HR@10 −0.0022 [−0.0076, +0.0036], NDCG@10 +0.0001, n.s.; vs WIDER-fixed −0.0040 n.s.

Geometry: effective rank 3.69, top-1 energy 0.78, mean cos to mean direction 0.30, pairwise cos 0.09, mean norm 274. Trace: pooling 93% on history titles; ‖r_A − r_B‖/‖r_A‖ 1.33; KL swap r **1.79** (ΔNLL +1.74) vs random direction 0.17 vs drop last item 0.089; relative logit change swap 0.126 vs random 0.060.

**Reading:** first run in this line that meets the stated success criterion literally: r's content matters strongly (another user's r costs −0.073 HR@10, −71%; swapping r moves the output 20× more than dropping the last history item) and ranking is not significantly below plain R3 (−0.002 n.s.; on Musical the margin loss alone cost −4%). WIDER plausibly removed the margin's cost by keeping r user-specific. Caveat to test: shuffle hurts more than mask, i.e. a foreign r actively misleads. That is what a "mismatch detector" also produces (decoder checks r against the history), and ranking does not improve, so whether r carries preference information or acts as a user-specific key is open. Proposed diagnostic: swap in the r of the most similar other user (nearest history); a preference-carrying r should cost little, a key-like r as much as a random user's.

**2026-11-08: scratch quota exceeded** (≈504 GB). v2 full 357275 failed at the final save after training (eval_loss 0.7221 / 0.5771 for epochs 1–2; no usable checkpoint). CARE full 357377 failed at step 844 (eval_loss 0.7228 / 0.5725; last [care] call 26801: solved 0.5, neg 0.5, threshold 0.420, so the mechanism stayed active). User freed the quota and asked to rerun all quota-failed runs: partial dirs moved to *_failed_quota; v2 and CARE resubmitted fresh via JobManager (other sessions' quota failures relayed to JobManager).
Reruns: v2 **357936**, CARE **357937** (fresh). Other sessions: R3-continue 357351/357352 → 357934/357935 (resumed from full checkpoints); CollabRec/LatentDiag quota failures left to their owner session.

## 2026-11-08 — Transition latent v2 result (357936, fixed reader)

6h55m, best epoch 2 (eval loss 0.7239 / **0.5739** / 0.6257). Results `output_dir/transition_latent/Toys_and_Games_transition_latent_v2/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw (own neighbours) | 0.0373 | 0.0611 | 0.0800 | **0.1070** | 0.0511 | 0.0589 | 0.0676 |
| none | 0.0374 | 0.0617 | 0.0806 | 0.1054 | 0.0515 | 0.0592 | 0.0672 |
| shuffle (another user's neighbours) | 0.0374 | 0.0607 | 0.0790 | 0.1058 | 0.0509 | 0.0585 | 0.0672 |
| shuffle_r (another prompt's r') | 0.0371 | 0.0606 | 0.0800 | 0.1051 | 0.0507 | 0.0587 | 0.0668 |
| plain R3 retrain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Bootstrap: raw − none HR@10 **+0.0016** [+0.0001, +0.0031]; raw − shuffle +0.0012 [−0.0001, +0.0025] n.s.; raw − shuffle_r **+0.0019** [+0.0007, +0.0033] (NDCG@10 +0.0008 sig); raw − plain A +0.0018 [−0.0024, +0.0058] n.s.

Coverage split (HR@10): covered rows (812) raw 0.567 / none 0.557 / shuffle 0.558 / shuffle_r 0.559 / plain A 0.567; uncovered raw 0.044 / none 0.044 / plain A 0.042.

Reader: mean max weight 0.25, normalised entropy 0.82 (v1 0.97 / 0.02); top neighbour = highest-count 26%; on covered rows the target is the reader's top-1 in 8.9% (random ≈ 5.3%), top-3 24.6%, median rank 7. Learned scalars barely moved (τ 0.978, β 1.04, gain 30.01, source biases ≤ 0.03).

**Reading:** the fix works mechanically: the reader is no longer saturated, picks the target above chance, and the decoder uses r' a little (small but significant none / shuffle_r drops, the first significant content effect from a non-margin method in this line). The effect is ~0.002 HR@10, far from the ~0.05 coverage ceiling: the reader learns slowly (scalars at init after 1,263 steps with lr 3e-4) and its selection is only modestly better than chance. Next levers: a larger learning rate for the reader (separate parameter group), or a reader warm-up with the decoder frozen (the user's staged idea, applied here). CARE 357937 still running.

## 2026-11-08 — Transition latent v3: reader lr ×10 + phase schedule (user: "implement and use both")

Two levers for the slow reader in v2: (1) reader learning rate ×10 (separate AdamW group; rest at 3e-4); (2) per-epoch phases r3 → reader → joint: epoch 1 plain R3 (neighbours off in training), epoch 2 only `trans.*` trains (decoder and pooling frozen, 1.26 M params), epoch 3 joint. Evaluation always uses the neighbours; early stopping off with a schedule (best epoch by eval loss is still loaded). CPU check (Qwen2.5-0.5B): r3 phase → reader unused and no reader gradient; reader phase → gradients only on trans.* (10 tensors); joint → all. Commit 4984a998, frozen `output_dir/transition_latent_code/code_4984a998`, env READER_LR_MULT=10 SCHEDULE=r3,reader,joint. Smoke **358207** (256 rows × 3 epochs) → full **358208** (`Toys_and_Games_transition_sched`, afterok). Controls: v2 357936 (0.1070), CARE 357937 (running), plain A 0.1052.

## 2026-11-08 — CARE-style transition latent result (357937)

7h56m, best epoch 2 (eval loss 0.7235 / **0.5743** / 0.6249). Last [care] (call 40201): solved 0.5, neg 0.25, threshold 0.299. Results `output_dir/transition_latent/Toys_and_Games_transition_care/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CARE raw | 0.0373 | 0.0626 | 0.0772 | 0.1058 | 0.0518 | 0.0578 | 0.0670 |
| none | 0.0371 | 0.0623 | 0.0770 | 0.1054 | 0.0517 | 0.0577 | 0.0668 |
| shuffle | 0.0367 | 0.0623 | 0.0769 | 0.1063 | 0.0515 | 0.0574 | 0.0668 |
| shuffle_r | 0.0371 | 0.0631 | 0.0775 | 0.1054 | 0.0521 | 0.0579 | 0.0669 |

Bootstrap: raw − none +0.0004, raw − shuffle −0.0004, raw − shuffle_r +0.0004 HR@10, all n.s. CARE − plain A +0.0006 n.s.; CARE − v2 −0.0012 n.s. Reader: max weight 0.24, target top-1 on covered rows 9.0% (v2 8.9%), top-3 25.4%. Covered rows HR@10 raw 0.557 / none 0.555 / shuffle 0.562.

**Reading:** CARE did not strengthen use; it removed v2's small significant effect. With a weak reader (target top-1 only ~9%), the real neighbours are themselves rarely decisive, so adding 25% misleading sets mostly teaches the decoder that the neighbour signal is unreliable → ignore it. Selective trust needs a signal that is reliably useful when it is real; the reader must get better first (v3: lr ×10 + reader-only phase, 358207/358208).

## 2026-11-08 — Margin + WIDER + collaborative reasoning (user: "apply Collaborative Reasoning on top of margin+WIDER, Toys first")

**Motivation.** 356322 (margin + WIDER) is the first run whose `r` content is used (shuffle −0.073 HR@10) at R3-level ranking. Collaborative methods failed before because the decoder ignored `r`; now that it reads `r`, test whether collaborative signal through `r` changes predictions. User chose **both** variants, **from base, joint** training, same recipe as 356322 (Qwen2.5-1.5B base, lr 3e-4, batch 128, 3 epochs, margin 0.2 × 1.0, WIDER 0.1).

- **MWC-I (InfoNCE):** L1 two-phase model (`collaborative_margin_attention.py`) + WIDER: epoch 1 CE + margin + WIDER (queue fills), epochs 2–3 + user/target multi-positive InfoNCE (τ 0.1, weights 0.1/0.1, 10k FIFO, token-accuracy gate 0.5 as in the Toys A10 runs). Early stopping / load-best by eval loss as in the margin trainer. Eval = 356322 pipeline (ranking; experiment.py raw/mask/mean/shuffle_user; bootstrap; geometry; trace). Expectation from H1: r still = f(x), so likely ≈ 356322.
- **MWC-G (donor grounding):** `grounded_margin.py`: r1 = r0 + s·‖r0‖·dir(Σ a_j V(r0_j, e(y_j))) over 32 time-causal donors (other users' r0 + observed next item, Qwen3 item embeddings), item-grounding CE 0.2 + retrieval KL 0.1 (L5 settings); margin hinge on r1 (negative = another row's r1, detached); WIDER on r1. Grounding modules + evidence scale in **fp32** (L4–L6 bf16 freeze). Eval: donor controls knn / off / shuffle_outcome, and slot interventions on r1 via new `SLOT_MODE` (mask / mean / shuffle_user from a 10k train-row r1 bank saved after training); bootstrap vs knn.
- Success: knn > off and knn > shuffle_outcome (donor content used) and slot mean/shuffle significant, ranking ≥ 356322 / plain A (0.1052).

Commit e5306d4e (branch worktree-R3MarginWiderCollab), frozen `output_dir/margin_wider_collab_code/code_e5306d4e`, `scripts/margin_wider_collab.sbatch` (METHOD=infonce|grounded). Results → `output_dir/margin_wider_collab/<name>/`. Smoke (256 rows, 2 epochs): infonce **358241**, grounded **358242** (researchshort).

## 2026-11-08 — GRPO stage 2 on margin + WIDER (Toys, user request)

Stock R3 stage 2 (noise-GRPO; only the pooling attention trains, LLM frozen; lr 1e-5, β 0.01, 8 generations, ε 0.2/0.28, batch 256, 1 epoch) starting from the margin + WIDER checkpoint (`scratchDirectory/R3/output_dir/Toys_and_Games_margin_0.2_wider_0.1/latent`, HR@10 0.1030). Motivation: on plain R3 (L9C) stage 2 at lr 1e-5 was a no-op because the decoder ignored r; here r strongly drives the output (shuffle −0.073), so tuning the r producer with a ranking reward can now change predictions. Known issue kept as in stock R3: reference log-probs use the policy's r, so the KL term is ≈ 0 (no trust region). New `scripts/grpo_stage2.sbatch` (any category, hadylauwresearch a40, scratch outputs, ranking + interventions + bootstrap, results to `output_dir/grpo_stage2/<tag>`), commit 15d5523e, frozen `output_dir/grpo_stage2_code/code_15d5523e`. Smoke **358298** → full **358299** (`toys_margin_wider_grpo`, afterok).

## 2026-11-08 — Variational view of the ignored latent; options C (history dropout) and A (CVAE latent) implemented (user: "implement C first then A")

**Framing.** Plain R3 is a conditional VAE whose "posterior" never sees the target and whose KL is identically zero (the stage-2 KL = 0 observation is the same fact): `r = f(x)` is a deterministic function of the history the decoder reads, so `I(y; r | x) = 0` by construction and the latent is non-identifiable — collapse iff non-identifiable (Wang, Blei & Cunningham, NeurIPS 2021). This is the log's "root cause is redundancy" with a theorem. The line's results map onto Alemi et al.'s rate–distortion curve: CE-only = rate 0 (latent unused, 0.106); L7 / gist = forced rate with a lossy encoder (0.08–0.09, brand copying lost); every soft variant slides back to rate 0 (modality competition); margin + WIDER = latent-regression term (InfoGAN / BicycleGAN family), hence the "mismatch detector" reading. Cross-domain matches: VLAE (strong AR decoder ignores z → restrict the decoder's local channel), Bowman 2016 word dropout, Zhao 2017 CVAE dialogue (KL annealing + bag-of-words loss), McCarthy 2020 variational NMT (MI-promoting ELBO + BoW), multimodal "modality competition", MLLM visual neglect, Coconut analyses (arXiv 2604.06374: fine-tuned Coconut reaches 96.6% with zero latent tokens = our T2).

**Fixes implemented** (branch `worktree-R3Variational`, commit 6a475911, frozen `output_dir/variational_code/code_6a475911`; `latent_attention_train.py` untouched):

- **C — stochastic history dropout** (`src/latent/history_dropout.py`, mask built in `layers.py::_latent_bottleneck_inputs`, `config.latent_bottleneck = {"dropout": spec}`): per row, each history *item* (or title token) is hidden with prob p from **every** query of the recommendation pass (target-side-only hiding leaked the hidden items through the separators / `### Response:` scaffold's deeper-layer K/V — caught by the unit test); template, headers, separators visible; positions unchanged; the pooling pass sees the full history, so `r` is the only route to a dropped item. Unlike Attempt 1's curriculum the decoder always sees a partial history (no detectable mode switch). Inference `eval_p` (default 0 = full history; the sbatch also ranks at the training rate, `EVAL_DROPOUT_RATES`). Segmentation on real Toys prompts: item count correct on 5/6 sampled rows (a title with inch-mark quotes over-segments, same quirk as history_pooling).
- **A — variational latent** (`src/latent/cvae_latent.py`): prior `p(z|x) = N(mu_p, sigma_p)` from the existing pooled r (`PriorLatentLayer` wraps the pooling; slot = `up(mu_p)` at inference, so `experiment.py` extract / mean / shuffle act on the slot as before); posterior `q(z|x,y)` from the first-pass hidden states with the last target token as query (training rows already contain the target: no extra pass); z ∈ R^64 reparameterised; loss `CE + beta_t·max(KL(q‖p), free_nats) + w_aux·aux(z_q) + w_aux_p·aux(mu_p)`, aux = cosine softmax over the Qwen3 item table (6,299 Toys items, rows = item ids) predicting the target from the latent alone (Zhao 2017 / McCarthy 2020 bag-of-words analogue); linear or cyclical beta warm-up; `prior_prob` rows decoded from a prior sample (train/test mismatch control, cf. the category "over-trust" result); new modules fp32. `cvae_diagnostics.json` per evaluation on 64 validation rows: KL (nats = the latent's rate), ‖mu_q − mu_p‖, teacher-forced NLL with z = mu_q (posterior oracle) / mu_p (inference) / 0 (mask) / shuffled, aux hit@10. Eval `CVAE_MODE=sample` ranks with a seeded prior sample. C and A combine.
- Tests `tests/test_history_dropout.py`, `tests/test_cvae_latent.py` (27 pass incl. `test_latent_bottleneck.py`). Pipeline `scripts/variational_latent.sbatch` (Toys, Qwen2.5-1.5B, R3 recipe, early stopping on the inference-path validation CE; ranking, interventions raw/mask/mean/shuffle_user, bootstrap, latent rank; outputs `output_dir/variational/<name>` since scratch is full).

**Smoke** (256 rows, 1 epoch, 64 test rows): **358504** C (`smoke_dropitem0.5`), **358505** C+A (researchshort, frozen code_a4178cd2; earlier attempts: 358373/358374 cancelled, 358488/358489 valid-sample bug, 358490 cp bug after a successful train + rank) (`smoke_dropitem0.5_cvae`: dropout 0.5, d=64, beta 1 linear warm-up 30%, free 2 nats, aux 0.5, prior_prob 0.2).

**Proposed full runs (not launched; to confirm):** V1 C: p ∈ {0.3, 0.5, 0.7} item-level, rank at eval_p = 0 and = p; V2 A: d=64, beta 1, free 2 nats, aux 0.5, prior_prob 0.2, with/without dropout 0.5; controls: A with aux 0 (does KL alone collapse?), A with beta 0 (posterior copy, expected to over-trust). Success = mean/shuffle significantly hurt **and** HR@10 ≥ plain R3 0.1061 / T2 0.1064; for A also report KL nats and the posterior–prior NLL gap.

**Smokes:** grounded 358242 passed (3 min; slot bank saved, evidence scale fp32 and trainable, all six ranking modes + bootstrap run). infonce 358241 failed (validation reused the train sample size on the 64-row VALID_SAMPLE file; smoke-only) → `eval_sample` fix, commit **b3740008**, frozen `output_dir/margin_wider_collab_code/code_b3740008`; re-smoke **358545** passed (phase switch at epoch 1, WIDER logged, interventions/trace run). CPU check (Qwen2.5-0.5B, 4 rows): loss = CE + margin + 0.2·item + 0.1·retrieval + 0.1·WIDER exactly; evidence_scale grad 0.30; slot override exact.
**Full (Toys, BASE_MODEL=Qwen/Qwen2.5-1.5B, hadylauwresearch a40, code b3740008):** MWC-G `Toys_and_Games_margin_0.2_wider_0.1_grounded` = **358546**; MWC-I `Toys_and_Games_margin_0.2_wider_0.1_infonce` = **358547** (afterok 358545). Both pending.

**v3 smoke 358207 passed** (5 min): three [schedule] lines r3 → reader (3,147,784 trainable = reader only at hidden 1536) → joint; eval_loss 4.453 / 4.045 / 3.479; four evals written. Full 358208 queued.

**GRPO smoke 358298 passed** (3 min, 32 steps, 64-row HR@10 0.125, not meaningful). Logged kl = 0.0 (the known stock issue); the stock trainer logs no reward. Full 358299 queued.

**2026-11-09:** 358546/358547 still pending after ~18 h on hadylauwresearch (A40s held by multi-day jobs) → moved with scontrol to researchshort (research-1-qos, a40|l40s|a100|h100|l40, 48 h; resume from checkpoint on preemption). Same micro-batch 4.

## 2026-11-09 — V1 result: history dropout (option C) does not make the decoder read `r` (Toys, 358573 / 358574 / 358575)

Item-level history dropout in the recommendation pass, p ∈ {0.3, 0.5, 0.7}, R3 recipe (Qwen2.5-1.5B, lr 3e-4, batch 128, early stopping; p=0.3/0.5 stopped after epoch 2, p=0.7 ran 3 epochs), code_9b68adc6, researchshort (lucas / avenue, 4.6–8.5 h). Test n = 6,738, interventions with the 10k-row train bank, paired bootstrap 2,000 × seed 42.

| p | HR@10 / NDCG@10 (history visible) | HR@10 at eval_p = p | raw − mask HR@10 | raw − mean | raw − shuffle_user | `r` effective rank (top-1 energy) |
|---|---|---|---|---|---|---|
| 0.3 | 0.1042 / 0.0629 | 0.0861 | **+0.0275** [+0.0227, +0.0322] | −0.0010 [−0.0024, +0.0001] n.s. | −0.0003 n.s. | 1.72 (0.88) |
| 0.5 | 0.0988 / 0.0592 | 0.0644 | **+0.0430** [+0.0380, +0.0484] | −0.0006 n.s. | −0.0004 n.s. | 1.28 (0.95) |
| 0.7 | 0.1011 / 0.0598 | 0.0562 | **+0.0368** [+0.0313, +0.0422] | +0.0004 n.s. | +0.0003 n.s. | 3.75 (0.58) |

Reference: plain R3 v0 0.1061 / 0.0678, T2 0.1064. Full metrics in `output_dir/variational/Toys_and_Games_dropitem{0.3,0.5,0.7}/summary/metrics_{raw,drop<p>}.txt`.

**Reading.** Same signature as plain R3: `mask` hurts (activation dependence), `mean` / `shuffle_user` are exactly neutral (no content dependence), ranking 2–7% *below* plain R3. Ranking with the training dropout rate collapses (−17% to −44%), so the decoder learned to predict from whatever history is visible plus a constant slot, not to read the dropped items out of `r` — even though `r` was the only route to them. The pooled `r` stays near-constant (effective rank 1.3–1.7 at p ≤ 0.5; 3.75 at p = 0.7 but still unused). Dropout therefore changes the gradient *preference* far less than hoped: the visible-item shortcut remains cheaper than learning to decode a pooled vector, and the pooling never receives a gradient that rewards user content (the same "nothing anchors the latent" failure as the continuous thoughts). Consistent with the variational reading: a deterministic `r = f(x)` still has no target information the history lacks; dropout only makes part of x *harder to read*, it does not give `r` information the decoder cannot get otherwise. Word-dropout works in text VAEs because the latent is trained with a target-aware posterior — which is option A.

**Decision:** C closed as a negative result (do not combine further without a target-aware latent). Proceed to V2 (A), with and without dropout 0.5, as agreed.

## 2026-11-09 — Transition latent v3 result (358208: reader lr ×10, schedule r3 → reader → joint)

6h24m on album, 0 tracebacks. [schedule]: epoch 1 r3 (1,553,529,864 trainable), epoch 2 reader (3,147,784), epoch 3 joint. Eval loss per epoch 0.7244 / 0.7239 / **0.5868** (epoch 3 loaded). Results `output_dir/transition_latent/Toys_and_Games_transition_sched/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 raw | 0.0340 | 0.0585 | 0.0755 | 0.1034 | 0.0481 | 0.0552 | 0.0642 |
| none | 0.0340 | 0.0580 | 0.0748 | 0.1023 | 0.0478 | 0.0548 | 0.0636 |
| shuffle | 0.0349 | 0.0588 | 0.0747 | 0.1037 | 0.0485 | 0.0551 | 0.0645 |
| shuffle_r | 0.0340 | 0.0579 | 0.0754 | 0.1031 | 0.0477 | 0.0550 | 0.0639 |

(@20 = @10 with 10 beams.) Bootstrap: raw − none HR@10 +0.0012 [−0.0004, +0.0030], raw − shuffle −0.0003, raw − shuffle_r +0.0003, all n.s. (NDCG@10 within ±0.0005). v3 − plain A HR@10 −0.0018 [−0.0070, +0.0034] n.s., NDCG@10 −0.0026 n.s.; v3 − v2 HR@10 −0.0036 n.s., **NDCG@10 −0.0034 [−0.0063, −0.0004] sig**.

Reader: max weight 0.233, entropy 0.838; covered-row target top-1 **14.0%** (v2 8.9%, chance ≈5.3%), top-3 29.6%, median rank 6; weight by source position 0.34/0.23/0.18/0.15/0.11. Covered rows HR@10 raw 0.536 / none 0.538 / shuffle 0.536; uncovered 0.044 / 0.043.

Load-warning check: every transition eval prints "trans.bias unused / trans.beta newly initialized". This is transformers 4.48's legacy key rename (beta → bias); `load_fp32_trans_params` then copies `trans.beta` from the safetensors file. Verified on this checkpoint: loaded β = 1.2145 = file value. All transition results stand.

**Reading:** the levers made the reader better (target top-1 8.9% → 14.0%) but the decoder does not use it: the reader-only epoch left eval loss unchanged (0.7244 → 0.7239), so with a frozen decoder the read adds nothing to the likelihood, and in the joint epoch the decoder again ignores r' (none/shuffle n.s.). Ranking dropped slightly vs v2 (NDCG@10 sig). Neighbour content is redundant with what the decoder already reads from the history (H1 again): on covered rows the target is already found 54% of the time without the read. Transition line: v1/v2/CARE/v3 all ≈ R3 with ≤ 0.002 content effect.

## 2026-11-09 — MWC-G result (358546, margin + WIDER + donor grounding, Toys)

Started 12:16 on lunar (L40S) after one preemption (resumed from checkpoint-422), done 17:09. Eval loss epoch 2 **0.5952** / epoch 3 0.5993 (final model = epoch 3; save_strategy epoch, no load-best in this trainer). Evidence scale 0.50 → 0.46 (now trainable, fp32). Last logged: retrieval KL 0.03, item NLL r0 6.24 → r1 6.22 (grounding barely helps the item head), attention entropy 3.66 (≈ uniform over ~40 donors). Results `output_dir/margin_wider_collab/Toys_and_Games_margin_0.2_wider_0.1_grounded/`.

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| knn (method) | 0.0358 | 0.0638 | 0.0798 | **0.1064** | 0.0521 | 0.0587 | 0.0672 |
| off (no donors) | 0.0355 | 0.0632 | 0.0787 | 0.1049 | 0.0514 | 0.0578 | 0.0662 |
| shuffle_outcome | 0.0361 | 0.0638 | 0.0790 | 0.1064 | 0.0520 | 0.0583 | 0.0672 |
| slot mask | 0.0089 | 0.0224 | 0.0341 | 0.0502 | 0.0167 | 0.0214 | 0.0266 |
| slot mean | 0.0046 | 0.0162 | 0.0249 | 0.0389 | 0.0113 | 0.0149 | 0.0193 |
| slot shuffle_user | 0.0050 | 0.0147 | 0.0237 | 0.0389 | 0.0105 | 0.0142 | 0.0191 |
| 356322 margin+WIDER | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Bootstrap (HR@10 / NDCG@10): knn − off +0.0015 [−0.0007, +0.0037] / +0.0010 n.s.; knn − shuffle_outcome 0.0000 / +0.0001 n.s.; knn − slot mask **+0.0562** / +0.0407; − slot mean **+0.0675** / +0.0479; − slot shuffle_user **+0.0675** / +0.0481 (all sig). knn − 356322 +0.0034 [−0.0019, +0.0086] n.s.; knn − plain A +0.0012 [−0.0043, +0.0064] n.s. Slot geometry (10k train r1): effective rank 77 (356322: 3.7), top-1 energy 0.15, pairwise cos 0.013.

**Reading:** meets the success criterion again, more strongly than 356322: the slot vector's content matters (mean/shuffle −0.068 HR@10, −63%; 356322 mean −0.031) at R3-level ranking (+0.001 vs plain A, +0.003 vs 356322, n.s.), and r1 is far less low-rank (77 vs 3.7), which argues against the pure "key" reading. But the **collaborative part is unused**: shuffling the donors' outcomes changes nothing and removing donors costs 0.0015 n.s., although the evidence term is 0.46·‖r0‖. The item head barely benefits from grounding (NLL 6.24 → 6.22) and donor attention is near-uniform, so the evidence vector is close to a constant offset and the decoder reads only the r0 part. Next lever (to discuss): make donor evidence informative before asking the decoder to use it (e.g. a stronger retriever / item head, the per-row "outcome" margin: hinge against r1 with shuffled outcomes instead of another row's r).

## 2026-11-09 — MWC-I partial result (358547, margin + WIDER + collaborative InfoNCE, Toys)

Preempted 3× on researchshort (analog, A100-40GB; first attempt 09:03–14:24 finished training; log overwritten by each requeue). Eval loss 0.9206 / 0.6400 / **0.5974** (best = epoch 3). Results `output_dir/margin_wider_collab/Toys_and_Games_margin_0.2_wider_0.1_infonce/`. Geometry (10k train r): effective rank 76, top-1 energy 0.21, pairwise cos 0.046.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MWC-I raw | 0.0322 | 0.0561 | 0.0718 | **0.0966** | 0.0455 | 0.0520 | 0.0600 |
| 356322 margin+WIDER | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

MWC-I − 356322 HR@10 **−0.0064** [−0.0117, −0.0013], NDCG@10 −0.0068 (sig); − plain A **−0.0086** [−0.0137, −0.0036], NDCG@10 −0.0067 (sig). InfoNCE on top of margin+WIDER costs ranking (−8% vs plain A), failing criterion (2) regardless of the interventions (running).

### 2026-11-09 — Implementation check of MWC-G / MWC-I (user: "is the model implemented correctly?")

**MWC-G mechanics verified on the saved checkpoint** (`tmp/grounded_probe.py`, 8 test rows, CPU): bank 53,898 donors, evidence scale 0.4596 fp32; ‖r1 − r0‖ = 82.1 = scale·‖r0‖ exactly; `off` returns r0 exactly; donor attention entropy 3.46 = ln 32 (uniform); component losses from the stdout log sum to the logged loss (step-1 6.98 = CE + margin + 0.2·item + 0.1·retrieval + 0.1·WIDER → no grad-accum scaling bug); margin hinge satisfied by the end (0.001), WIDER 0.016, item NLL 7.66 → 6.95, retrieval KL 0.03. Interventions did change outputs (shuffle_outcome alters the top-10 in 4,553/6,738 rows and top-1 in 341; off 6,104 / 1,036), so the modes were applied. Decoder target NLL: knn 0.589, shuffle_outcome 0.589, off 0.593, another row's r1 2.31, zeros 1.71.

**Finding: the grounding module collapsed to a constant direction.** Evidence vector e = r1 − r0: pairwise cosine across different users 0.9995; cos(e_knn, e_random-donors) 0.9998; cos(e_knn, e_items-zeroed) 0.993. Value-MLP outputs over 512 random donors: mean vector norm 406 vs per-donor deviation 36, pairwise cos 0.9965 (not from the output bias, norm 0.13 — the weights learned a shared output). So r1 = r0 + 0.46‖r0‖·c with c the same unit vector for everyone; donors, their outcomes and the retriever (uniform attention; teacher softmax(utility/0.5) with utility std 0.2 is itself near-uniform) have no effect. The decoder's content dependence (slot mean/shuffle −0.068) is therefore dependence on **r0** (margin + WIDER, as in 356322), not on collaborative evidence. "knn = shuffle_outcome = off" is the correct output of a degenerate solution, not a code error. Cause is the L5 design: `dir(Σ_j a_j V(r_j, e_j))` is free to make V nearly constant, which the item-grounding CE rewards (a fixed shift the item head likes) while nothing rewards donor-specific variation (L5 CDs showed only +0.0023, probably the same collapse at a smaller scale).

**MWC-I components** (W&B `output_training-358547…log`, 1,263 steps): epoch 1 CE 3.31 → 0.89, margin 0.22 → 0.18, WIDER 0.45 → 0.06, queue accept 0.41 → 0.96 (gate 0.5), queue full (10k) by end of epoch 1; InfoNCE on at step 422 (loss +0.85; user-positive rate 0.61, target 0.81); end: CE 0.44, margin 0.02, WIDER 0.06, total 0.91 (InfoNCE ≈ 0.44 of it), accept 1.0. Phase switch and WIDER behave as designed; the ranking cost (−0.0086 vs plain A) is a genuine effect of the InfoNCE terms.
InfoNCE terms (keys `user_infonce_loss` / `target_infonce_loss`): user 4.32 → 0.40 (step 800) → 0.28; target 5.16 → 3.96 → 4.02 (flat, as in Attempt 10); so the queue objective is learned on the user side (same-user rows pulled together) and the ranking cost coincides with it.

**Why WIDER did not prevent the collapse (user question, probe extended):** the constant evidence direction c is anti-aligned with r0's shared direction: cos(c, q_r0) = **−0.97**. WIDER penalty mean (z·q)² on r0 0.245 → on r1 0.020; pairwise cos r0 0.25 → r1 0.03; ‖mean r0‖ 103 → ‖mean r1‖ 60 with ‖evidence‖ 82. So the cheapest way to satisfy WIDER on r1 was for the donor module to output a fixed vector that subtracts r0's common component — WIDER *selected* the constant solution (a per-user evidence would have had to do the same job with more variance). The margin is indifferent to c (both r1 and the other row's r1 contain it). Diversity in the slot therefore comes entirely from r0; margin + WIDER make the slot diverse but give the donor path no reason to be. Fix must make the evidence itself non-constant (centering/variance term on V or on e, outcome-swap margin negative).

### 2026-11-09 — MWC-I analysis (user: "analyse MWC-I and try to improve it")

**Where the ranking is lost** (per-row HR@10, test n = 6,738; copy row = target bought the same day as a history item of the same brand with title Jaccard ≥ 0.3, 13.4% of rows):

| rows | n | MWC-I | 356322 | plain A | MWC-G | MWC-I − 356322 | share of gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 6,738 | 0.0966 | 0.1030 | 0.1052 | 0.1064 | −0.0064 | 1.00 |
| copy rows | 905 | 0.4818 | 0.4785 | 0.5127 | 0.5028 | +0.0033 | −0.07 |
| non-copy | 5,833 | 0.0369 | 0.0447 | 0.0420 | 0.0449 | −0.0079 | 1.07 |
| same-brand target | 2,144 | 0.2822 | 0.2957 | 0.3027 | 0.3008 | −0.0135 | 0.67 |
| other brand | 4,594 | 0.0100 | 0.0131 | 0.0131 | 0.0157 | −0.0030 | 0.33 |

Top-10 composition: share of predictions with a history brand MWC-I 0.631 / 356322 0.583 / plain A 0.623; exact history items 0.127 / 0.104 / 0.101. MWC-I copies **more** from the history, not less; the whole loss is on non-copy rows (−18% there), i.e. on genuine next-item prediction.

**Geometry** (10k train r, centred cosines): same-user consecutive rows 356322 0.804 → MWC-I **0.865**; random cross-user 0.019 → 0.000; same-target cross-user 0.152 → 0.159 (unchanged); ‖r‖ 274 → 134. Training: user InfoNCE 4.3 → 0.28 (learned), target InfoNCE 5.2 → 4.0 (flat, as in Attempt 10 / L1).

**Reading.** The user term is what was learned, and it is a user-identity objective: a user's rows are sliding-window prefixes, so pulling them together makes r ignore the newest items and behave like a user ID; the decoder then leans harder on repeating history (brand/item copy up), which keeps copy rows but loses non-copy rows. The collaborative (same-target, cross-user) structure did not change at all. So MWC-I's cost comes from the user positives, not from the collaborative idea. Improvement candidates: (A) user weight 0, target weight 0.1–0.2 (same code); (B) invert the user term: same-user, different-target rows as **hard negatives** for the target head (forces r to track the latest items), positives = same target across users; (C) B + NLL-gated queue (L8a). Decision pending with the user. MWC-I interventions still running.

## 2026-11-09 — MWC-I-B: same-user hard negatives (user chose B)

Change (`collaborative_contrastive_attention.py::_relation_loss`): for the target head, in addition to the usual 256 random negatives (other user, other target), up to `hard_negatives` = 16 rows of the **same user with a different next item** are added as negatives (previously all same-user rows were excluded from negatives). Positives unchanged (same target, other users). User InfoNCE weight 0, target 0.1, everything else = MWC-I (margin 0.2, WIDER 0.1, two-phase, gate 0.5, early stopping). Rationale: the user term made r a user identity; the hard negatives require r to separate a user's consecutive rows, i.e. to encode what is new in the history, while the positives keep the collaborative pull. CPU check (Qwen2.5-0.5B, synthetic bank): hard-negative rate 1.0 when the queue holds same-user other-target rows; single anchor equal to its own user's other-target row: target loss 2.80 → 9.63 with hard negatives (deterministic). Prediction: non-copy rows recover (≥ 356322's 0.0447), same-user cos falls below 0.80, target InfoNCE no longer flat.

## 2026-11-09 — MWC-I-C: item-prediction head on r (user: "the problem is the same-target cross-user pairs")

Diagnosis of the flat target InfoNCE (≈ 4.0, about chance for 16 positives / 256 negatives): exact-ID positives are sparse and noisy on long-tail Toys targets; aligning two users' r is the full prediction task asked of a 256-d head at weight 0.1; random negatives saturate. Options offered: C1 dense item-prediction head (chosen), C2 soft title-similar positives, C3 nearest-but-different hard target negatives, C4 stronger weight / bigger queue.

**C1 (`collaborative_margin_attention.py::_item_loss`)**: head LN + Linear(1536 → 1536) in fp32 maps r to the Qwen3 title-embedding space; CE over all 6,299 items at τ 0.07, weight 0.2, applied in both phases (dense, no queue needed); logged `item_loss`, `item_hit10`. Same-next-item users are pulled to the same point through the shared item table; similar items share partially. On top of B (same-user hard negatives, user weight 0, target 0.1), margin 0.2, WIDER 0.1. Inference unchanged (head unused). CPU check (Qwen2.5-0.5B, 4 rows): total = CE + margin + 0.2·item exactly; item CE 8.81 ≈ ln 6299 at init; item loss alone gives gradient on the pooling attention (1.7e4) and head; ids verified 0..6298 = table rows. B smoke 359985 passed (hard-negative rate 0 there is expected with 256 random rows: no same-user pairs in the queue); full B 359986 running since 18:29 (A100).

## 2026-11-09 — MWC-I final (358547): interventions

Done 19:36 (2h24m after the last requeue). Interventions (experiment.py, n = 6,738):

| condition | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0322 | 0.0561 | 0.0718 | **0.0966** | 0.0455 | 0.0520 | 0.0600 |
| mask | 0.0015 | 0.0039 | 0.0068 | 0.0159 | 0.0028 | 0.0040 | 0.0068 |
| mean | 0.0027 | 0.0053 | 0.0111 | 0.0194 | 0.0042 | 0.0066 | 0.0093 |
| shuffle_user | 0.0052 | 0.0093 | 0.0137 | 0.0217 | 0.0076 | 0.0093 | 0.0119 |

Bootstrap: raw − mask **+0.0807** [+0.0741, +0.0876]; raw − mean **+0.0772** [+0.0705, +0.0840]; raw − shuffle **+0.0749** [+0.0684, +0.0815] HR@10 (NDCG@10 +0.053 / +0.051 / +0.048), all significant; stronger than 356322 (mean −0.031, shuffle −0.073). Trace (n = 300): pooling 71% on user tokens; KL swap r **1.95** vs random-direction 0.48 vs drop-last-item 0.053; rel. logit change swap 0.135 vs random 0.055. Geometry: eff. rank 76, pairwise cos 0.046, ‖r‖ 134.

**Reading:** MWC-I has the strongest content dependence so far (−78% HR@10 under mean, −80% under a foreign r) but fails criterion (2): −0.0086 HR@10 vs plain A, all of it on non-copy rows (analysis above). Content dependence and ranking moved in opposite directions: the user-InfoNCE made r a stronger per-user signal that the decoder leans on, at the cost of next-item prediction. B (359986) and C (360047) test whether replacing the user term keeps the dependence without the cost.

**2026-11-09 19:40 — scratch quota exceeded again.** C smoke 360045 trained (item loss 8.70 at step 1 ≈ ln 6299; item head fp32) but failed at the epoch-1 checkpoint save (`Disk quota exceeded`); dependent full C 360047 cancelled (DependencyNeverSatisfied). B's checkpoint-422 was written 2 min earlier (intact; B at epoch 1.04: hard-negative rate **0.75**, target-positive rate 1.0 — the same-user negatives exist in the real queue). Freed 38 GB from this line only: smoke_mwcB / smoke_mwcC and the optimizer checkpoints (checkpoint-1263) of the finished MWC-G and MWC-I runs (final model.safetensors, slot bank and all evals kept). Scratch now ≈ 465 GB of ~504 GB. C smoke resubmitted.
C smoke **360055** passed (19:45; item loss 8.69 at init, pipeline through trace OK); full C **360056** started 19:46 on avenue (GPU per log). Smoke dir removed (scratch 225 → ~216 GB).

## 2026-11-09 — GRPO stage 2 on margin + WIDER result (358299)

11h03m on album (training 6.69 h, 1,684 steps), 0 tracebacks. Last step loss 0.0755, kl 0.0 (the known stock issue: no trust region), grad_norm 0.90; train_loss 0.0216; the stock trainer logs no reward. Results `output_dir/grpo_stage2/toys_margin_wider_grpo/` (per-row `0.json`).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO stage 2 | 0.0223 | 0.0433 | 0.0565 | **0.0781** | 0.0345 | 0.0400 | 0.0469 |
| margin + WIDER stage 1 (356322) | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain R3 retrain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

(@20 = @10.) Bootstrap: GRPO − stage 1 HR@10 **−0.0249** [−0.0295, −0.0202], NDCG@10 −0.0199 (both sig); GRPO − plain A HR@10 **−0.0272** [−0.0335, −0.0208] sig.

Interventions (experiment.py, raw − mode, HR@10): mask +0.0255 sig (≈0.0526), mean +0.0039 [−0.0009, +0.0086] **n.s.** (NDCG +0.0030 sig), shuffle_user +0.0416 sig (≈0.0365). Stage 1 for comparison: mask −0.051, mean −0.031, shuffle −0.073.

**Reading:** stage 2 is harmful here. Unlike plain R3 (L9C lr 1e-5 no-op, because r was ignored), r now drives the output, so the r producer gets real gradient. Without a KL anchor (kl ≡ 0) it drifts and ranking drops 0.025 HR@10. The mean intervention is now nearly harmless while shuffle_user still hurts: the post-GRPO r carries less per-user content (it moved toward a common vector that works about as well as each user's own), consistent with partial collapse, as in L9B at higher lr. If stage 2 is revisited, it needs the real KL anchor (reference log-probs from the frozen stage-1 attention's r) and/or a lower lr.

## 2026-11-09 — GRPO stage 2 with a real KL anchor (user: "yes, please implement the KL anchor")

**Why.** Stock R3 stage 2 computes the reference log-probs from `original_embeds`, which already contain the policy's r (+ noise). The decoder is frozen, so π_ref = π_θ exactly and KL ≡ 0 with zero gradient (no trust region). 358299 then drifted: −0.025 HR@10 and partial collapse of r.

**Fix (`--kl_anchor` / `KL_ANCHOR=1`).** The reference model is TRL's `create_reference_model` deep copy of the starting checkpoint, so it includes the frozen stage-1 attention. The reference now builds its own r (`ref.generate_embs`), adds the same Gaussian noise at the thought slot, and scores the same completion. KL is TRL's per-token k3 estimator, exp(ref−logp) − (ref−logp) − 1, plus β·KL in the loss, unchanged. CPU check (tiny Qwen2, policy attention perturbed): stock KL 0.0, anchor KL 7.1e-5 > 0.

**Match to standard GRPO.** Same as standard GRPO: the reference policy is frozen and is the starting model, and it sees the same input as the policy (prompt + noise); the KL estimator, its placement and the masking are TRL's. Different (inherent to R3's noise-GRPO, kept): the "completion" is the teacher-forced target rather than a sample from π_θ, so k3 is a per-token log-ratio penalty at the target tokens, not an unbiased estimate of KL(π_θ‖π_ref) over samples. Also, the exploration is the Gaussian noise on r, and the advantage is R3's custom baseline/normalisation.

Commit 710e2f81, frozen `output_dir/grpo_stage2_code/code_710e2f81`. Same settings as 358299 (lr 1e-5, β 0.01, 8 generations, batch 256, 1 epoch, from the margin + WIDER checkpoint). Tags `toys_margin_wider_grpo_kl_smoke` → `toys_margin_wider_grpo_kl`, sent to JobManager (job ids to follow).

Jobs: smoke **360236** → full **360237** (afterok), submitted by JobManager 2026-11-09.

**KL-anchor smoke 360236 passed** (3.6 min). The log shows the anchor line. kl > 0 from step 1 (1.0e-4 at step 1; range 3.3e-5 to 1.7e-4 over 32 steps, flat; 358298 logged exactly 0.0). Since the policy equals the reference at step 1, the ~1e-4 level is a bf16 numerical floor: the reference r is now computed separately by the reference copy. Real drift will show as growth above this floor in the full run. Full run **360237** running on alarm.

## 2026-11-09 — Comparison: margin + WIDER vs earlier latent-utilization attempts (trade-off view, user request)

Content dependence = HR@10 drop when r is replaced by another user's r (shuffle_user) or the train mean; ranking = HR@10 vs the matched plain baseline (Musical: plain R3 v1 0.1226; Toys: plain retrain A 0.1052, T2 constant token 0.1064). Criterion: dependence significant AND ranking not significantly below baseline.

| family | run (dataset) | HR@10 | Δ vs baseline | shuffle / mean ΔHR@10 | verdict |
|---|---|---:|---:|---|---|
| structural bottleneck | L7 latent-only (Musical) | 0.1085 | −0.014 (−11%) sig | −0.088 / −0.079 sig | used, ranking fails |
| structural bottleneck | K=1 thought + bottleneck (Toys) | 0.0801 | −0.026 (−25%) sig | −0.075 sig | used, ranking fails |
| structural bottleneck | K=8 thoughts + bottleneck (Toys) | 0.0819 | −0.025 (−23%) sig | −0.077 sig | used, ranking fails |
| gist bottleneck | M=4 / M=16 (Toys) | 0.0899 / 0.0911 | −0.016 (−15%) sig | delete → 0.004 | used, ranking fails |
| soft bottleneck | history bias b=−4 (Toys) | 0.1002 | −0.005 | n.s. | unused |
| partial bottleneck | recency split m=1 (Toys) | 0.1043 | −0.001 n.s. | exactly 0 | unused |
| staged attention | latent-only → full (1+2) (Toys) | 0.1008 | −0.002 n.s. (NDCG −0.006 sig) | per-sample D 2.20 nats (HR interventions not completed) | sensitive, ranking borderline |
| staged attention | full → latent-only (2+1) (Toys) | 0.0870 | −0.015 sig | D 0.007 | unused, ranking fails |
| latent reshaping | Attempt 10 / L1 / L8a / L8b InfoNCE (Musical) | 0.1208–0.1223 | ≈ 0 | ≤ 0.0015 n.s. | unused |
| latent reshaping | WIDER K=1 corrected (Toys) | 0.1070 | +0.002 n.s. | n.s. | unused (diverse but unread) |
| latent reshaping | history-only pooling (Toys) | 0.1036 | −0.002 n.s. | n.s. | unused |
| latent reshaping | DivReason path diversity (Toys) | −0.005 | n.s. | path-sensitive, content n.s. | unused |
| extra signal in r | SASRec history vector (Toys) | 0.1049 | ≈ 0 | shuffle −0.010 sig, drop n.s. | weakly used, no gain |
| extra signal in r | transition-neighbour latent v2 (Toys) | 0.1070 | +0.002 n.s. | none −0.0016 / shuffle_r −0.0019 sig | weakly used, no gain |
| extra signal in r | two-head fast + slow gist (Toys) | 0.1026 | −0.003 n.s. | slow head alone 0.0705 | unused |
| decoder objective | margin only (Attempt 2, Musical) | 0.1176 | −0.005 (−4%) | sensitivity ×9 (HR interventions not run) | used, small ranking cost |
| **decoder objective + reshaping** | **margin + WIDER 356322 (Toys)** | **0.1030** | **−0.002 n.s.** | **−0.073 / −0.031 sig** | **both criteria met** |
| + collaborative | MWC-G donor grounding (Toys) | 0.1064 | +0.001 n.s. | −0.068 / −0.068 sig (via r0; donors unused) | both met, no collaborative effect |
| + collaborative | MWC-I user+target InfoNCE (Toys) | 0.0966 | −0.009 sig | −0.075 / −0.077 sig | used, ranking fails |

**Trade-off reading.** Earlier attempts sit on two corners: bottlenecks buy dependence by taking the history away (−11% to −25% HR@10, lost on copy rows), while reshaping r (InfoNCE, WIDER, pooling masks, path diversity) keeps ranking but leaves r unread, because r = f(history) is redundant for the decoder. The margin loss is the only mechanism that makes the decoder *prefer the correct r* without hiding the history; alone it cost 4% (Musical), and WIDER removed that cost (Toys: −0.002 n.s.) by keeping r user-specific (effective rank 3.7 → collaborative variants 76–77). Dependence still buys no ranking gain anywhere: the best dependent runs (356322, MWC-G) equal plain R3, and pushing dependence further (MWC-I: −0.077 under mean) costs ranking. Open question: whether r carries preference content or acts as a user-specific key (nearest-user swap diagnostic proposed 2026-11-08).

**Toys margin-only control:** never run (planned as WEIGHT=0 of margin_wider.sbatch when 356322 was set up). User decided 2026-11-09 to skip it for now (GPU/quota kept for B and C); the "WIDER removed the margin's cost" reading therefore rests on Musical margin-only (0.1176, −4%) vs Toys margin+WIDER (−0.002 n.s.) — a cross-dataset comparison, flagged as a caveat.

## 2026-11-09 — Latent-bank quality for InfoNCE (user: "how to check the bank; how to ensure every row has a good positive")

**Coverage (Toys train, 53,898 rows / 10,167 users / 6,255 targets):** 99.8% of rows have ≥ 1 other user with the same target (≥ 3: 94%, ≥ 10: 51%; singleton targets 0.2%). Expected share of anchors with a same-target row in a random **10k queue: 0.80** (= logged target_positive_rate 0.80–0.85); 20k queue 0.93; full-train bank **0.998** (165 MB fp16). The 20% zero-loss anchors are a queue-window effect, not a data limit.

**Diagnostic** `src/prototype_grounding/bank_quality.py` (+ `scripts/bank_quality.sbatch`, commit 02ffaf86, frozen `code_02ffaf86`): on the first 10k train rows (same as `extract --limit 10000`) it reports coverage inside the bank, geometry (effective rank; centred cosines of same-target/other-user, same-user/other-target, random pairs), and **usefulness** = swap-in utility u_ij = NLL(y_i | x_i, r_i) − NLL(y_i | x_i, r_j) with the decoder (teacher-forced, same path as the interventions) for 800 anchors × up to 4 same-target positives, 1 same-user hard negative, 1 random, bank mean; share "good" (u > −0.1), "better" (u > 0), and best-of-k for positives (does some good positive exist). Jobs: MWC-I-B, MWC-I, 356322 (ids below).
Jobs: bank quality MWC-I-B **360284**, MWC-I **360285**, 356322 **360286** (researchshort).

**Chance level of the target InfoNCE (user question):** multi-positive InfoNCE at equal scores = log((P+N)/P). In a 10k Toys queue (5 random windows) anchors with ≥1 same-target other-user positive 0.80, median P 2.6, mean P 4.0 (cap 16) → chance **4.56** with 256 negatives (B: 4.62 with +16 hard). Logged end values MWC-I **4.02**, B **4.10**: ~0.5 nats below chance = positives get ≈ 1.6× the probability mass of chance. Start of phase 2 was 5.3 (random head, above chance), so the head learned ≈ 1.2 nats then plateaued; the residual structure matches the same-target centred cosine 0.15–0.16 that 356322 already has without any InfoNCE. Conclusion: not pure chance, but close to it and essentially nothing beyond what CE+margin already induce.

### Bank-quality results (360284–360286, 2026-11-09, 3.5 min each; 10k train rows, 800 anchors, k = 4 positives)

Coverage inside the 10k bank (identical for all three, same rows): anchors with a same-target other-user row 0.825 (≥ 3: 0.481); with a same-user other-target row 0.947.

| checkpoint | eff. rank | centred cos: same-target / same-user / random | own NLL | positives (mean over k): mean u / good / better | positives best-of-4: good / better | same-user other-target: median u / good / better | random: mean u / good | bank mean: mean u / good |
|---|---:|---|---:|---|---|---|---|---|
| 356322 margin+WIDER | 3.7 | 0.157 / 0.547 / 0.010 | 0.384 | −1.26 / 0.12 / 0.04 | 0.45 / 0.20 | −0.006 / 0.66 / 0.26 | −1.72 / 0.15 | −0.57 / 0.23 |
| MWC-I | 76.0 | 0.156 / 0.734 / −0.003 | 0.444 | −1.39 / 0.15 / 0.07 | 0.45 / 0.22 | −0.011 / 0.75 / 0.30 | −1.86 / 0.11 | −1.58 / 0.12 |
| MWC-I-B | 20.9 | 0.208 / 0.596 / 0.006 | 0.438 | −1.17 / 0.10 / 0.05 | 0.36 / 0.17 | −0.009 / 0.68 / 0.26 | −1.55 / 0.06 | −1.33 / 0.06 |

(u = NLL(y_i | x_i, r_i) − NLL(y_i | x_i, r_j); good = u > −0.1; better = u > 0.)

**Reading.**
1. **Same-target other-user latents are bad positives in effect terms.** A typical positive costs ≈ 1 nat on the anchor's own target (median −0.9 to −1.2), not far from a random user's r (−1.0 to −1.7); only 10–15% are good, and even the best of 4 is good for just 36–45% of anchors. Pulling r toward such vectors cannot improve the anchor's prediction; this is why the target InfoNCE sits near chance and why fixing coverage would not help ranking.
2. **The user's own other rows are the best substitutes** (median u ≈ −0.01, 66–75% good, 26–30% *better* than the anchor's own r). The decoder-useful content of r is user-level, nearly interchangeable across a user's rows — i.e. r carries the user, not the next item. B's hard negatives push apart exactly the vectors the decoder finds interchangeable (effective rank 76 → 21, same-user cos 0.73 → 0.60, random/mean "good" 0.11 → 0.06): more own-r dependence, no reason to expect a ranking gain.
3. Implication for collaborative reasoning: positives should be defined by **effect** (u_ij > −τ: another user's reasoning that works for this anchor), not by shared target ID. Such rows exist (15% of random other-user rows are good for 356322) but are not the same-target rows. Follow-up diagnostic proposed: utilities over 64 random other-user rows per anchor → share of anchors with ≥ 1 good other-user r, and what the good donors share with the anchor (target, brand, history overlap).

**Random-donor diagnostic** (`src/prototype_grounding/bank_random_donors.py`, commit fabc7e81, frozen `code_fabc7e81`): 400 anchors × 64 random other-user latents each, swap-in utility u, good = u > −0.1; per donor: same target, same brand, target-title cosine (Qwen3), history Jaccard, donor target in anchor history (and vice versa), latent cosine. Jobs: 356322 **360311**, MWC-I **360312**, MWC-I-B **360313**.

### Random-donor results (360311 356322, 360312 MWC-I; 360313 B pending) — 400 anchors × 64 random other-user latents

| checkpoint | anchors with ≥1 good donor / ≥3 | mean good of 64 | best donor better than own r | donor good rate | good rate: same target / same brand / history overlap > 0 | good rate: latent-cos top decile / bottom decile | corr(u, latent cos) | corr(u, target-title cos) / corr(u, history Jaccard) |
|---|---|---:|---:|---:|---|---|---:|---|
| 356322 | 0.995 / 0.963 | 9.5 | 0.805 | 0.149 | 0.25 / 0.22 / 0.29 | **0.56 / 0.00** | **0.46** | 0.03 / 0.02 |
| MWC-I | 0.990 / 0.922 | 7.5 | 0.700 | 0.116 | 0.33 / 0.23 / 0.30 | 0.38 / 0.005 | 0.44 | 0.10 / 0.03 |

Good vs bad donors (356322): latent cos **0.86 vs −0.06**; same target 0.001 vs 0.000; same brand 0.03 vs 0.02; history Jaccard 0.002 vs 0.001; donor-target title cos to the anchor's target 0.631 vs 0.626 (good rate flat across title-cos bins 0.14–0.18). MWC-I the same pattern (latent cos 0.29 vs 0.01).

**Reading.** Effect-defined positives are plentiful (≈ 15% of random other users' r are good for an anchor; 99% of anchors have one among 64; the best one beats the own r for 70–80% of anchors), but they have **no collaborative signature**: sharing the target, the brand, history items, or having a similar next item does not predict usefulness at all. The only predictor is closeness in latent space (corr 0.46; top decile 56% good, bottom decile 0%). A donor is good because its r is already near the anchor's r, i.e. the decoder's response to r is smooth; "effect-mined positives" would be latent nearest neighbours = self-smoothing, carrying nothing the anchor's own r lacks. Consequence: no InfoNCE positive definition (target ID, title-similar targets, CF neighbours, or effect-mined) can inject decoder-usable collaborative information on Toys; the InfoNCE route to a ranking gain is closed by this evidence. Collaborative information, if it is to help, must enter the slot as explicit evidence (items / outcomes) with effect-based training (the G3 design), not via state alignment of r.
Best-donor gain magnitude: mean best u +0.021 (356322) / +0.022 (MWC-I) nats, median +0.007 / +0.010, p90 +0.05 / +0.07 — the "better than own r" donors improve the target by a few hundredths of a nat (noise-level smoothing), while the average random donor costs −1.65 / −1.87 nats.

## 2026-11-09 — MWC-G3: CF-prior donor evidence + evidence-swap margin (approved by user, implemented)

**Pre-check (Toys, 3,000 sampled rows, `tmp/cf_coverage.py`)**: CF donors = up to 32 train rows of *other* users whose history shares ≥ 1 item with the query history and whose next item happened before the query's latest history event (ranked by shared items, then recency). Test: 0.4% rows with no donor, 1.3% < 8, 94.5% full 32 (train 6.8% / 15% / 64%). Target among the 32 donor outcomes: **test 10.9% (non-copy rows 10.1%) vs 0.8% for 32 random causal donors** (train 12.9% vs 1.1%). Donor-outcome frequency alone ranks the target in the top 10 for 6.3% of non-copy test rows. Target-in-history copy rows are only 1.1% of test.

**Design** (`src/latent/cf_grounded_margin.py`, trainer `latent_attention_train_cf_grounded_margin.py`, sbatch `METHOD=cfg3`, commit ba0ccfc9, frozen `output_dir/margin_wider_collab_code/code_ba0ccfc9`):
- r1 = r0 + s·‖r0‖·dir(e − μ), e = Σ_j a_j V([e(y_j), log1p(shared_j), log1p(gap_days_j)/5]); μ = EMA of e (seeded by first batch), saved as a buffer.
- Donor set fixed by the CF prior (precomputed per row); learned attention (query r0, key from donor outcome + features) only re-weights it.
- Deviation from the sketch: donor values use outcomes + overlap features only, **no donor r0_j** (random-donor diagnostic: other users' r carry no collaborative signal; also removes the stale-bank issue). No latent-similarity fallback (rows without donors keep r1 = r0).
- Losses: CE(r1) + history-swap margin 0.2 (other row's r0 ⊕ own evidence, detached) + **evidence-swap margin 0.2** (own r0 ⊕ another in-batch row's donor set, gradients through the evidence path) + item-grounding CE 0.2 + retrieval KL (τ_util 0.1) + WIDER 0.1 on **r0**; summed then ÷ grad-accum.
- Eval controls: knn (CF donors), off, random (32 random causal other-user train rows, seed 42), swap (CF donor set of the next test row of another user; may include donors later than this row's history cutoff, which can only favour the control); slot mask / mean / shuffle_user; paired bootstrap vs knn.
- Gates: (i) logged `evidence_cos_centred` across rows < 0.5; (ii) knn − swap and knn − random significant; (iii) knn − plain A ≥ +0.005 HR@10 or NDCG@10. Fail (i)/(ii) → donor direction closed on Toys.
- Unit test (CPU, dummy pooling): causality holds, grads reach value/query/scale, beam expansion and `off` correct; at init raw evidence cos across rows 0.97 (motivates centring).
- Smoke 360344 (256 rows, 2 epochs, 16 test rows) submitted.

**Random donors on MWC-I-B (360313, done)**: good rate 0.073 (lower than 356322 0.149 / MWC-I 0.116; B's latents are more spread), anchors with ≥1 good 0.965, best donor beats own r 0.60; corr(u, r_cos) 0.46, corr(u, title cos) 0.07, corr(u, hist Jaccard) 0.04; good vs bad r_cos 0.55 vs 0.02, same target 0.002 vs 0.000, title cos 0.637 vs 0.626 (slight rise only in the top bin, 0.138 at cos ≥ 0.8, n = 348). Same conclusion: usefulness tracks latent closeness, not collaborative overlap.
- Smoke 360344 COMPLETED (6 min): train rows with donor 0.91 (256-row sample), valid 1.00; centred evidence cross-row cos −0.09…0.05 vs raw 0.89–0.95 (centring works); evidence_gap (neg − pos NLL) −0.08 → +0.03 over 4 steps; ~13–18 s per optimizer step (≈ 5–6 h for 3 epochs); fp32 restore at eval (20 tensors); every control changes the top-10 on 11–14 of 16 rows. **Full run 360502** (`Toys_and_Games_margin_0.2_wider_0.1_cfg3`, BASE_MODEL Qwen/Qwen2.5-1.5B as 356322) submitted 2026-11-10.

### MWC-I-B results (359986, COMPLETED 2026-11-10, after two preemptions)

Toys test n = 6,738 (@20 = @10).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MWC-I-B raw | 0.0331 | 0.0591 | 0.0726 | 0.1017 | 0.0481 | 0.0536 | 0.0629 |
| mask | 0.0033 | 0.0064 | 0.0096 | 0.0141 | 0.0049 | 0.0063 | 0.0077 |
| mean | 0.0045 | 0.0089 | 0.0125 | 0.0180 | 0.0070 | 0.0084 | 0.0102 |
| shuffle_user | 0.0030 | 0.0076 | 0.0126 | 0.0186 | 0.0056 | 0.0077 | 0.0096 |
| 356322 margin+WIDER | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap: B − 356322 HR@10 −0.0013 [−0.0068, +0.0036] n.s., NDCG@10 −0.0039 [−0.0068, −0.0009] **sig**; B − plain A HR@10 −0.0036 n.s., NDCG@10 −0.0038 [−0.0067, −0.0008] **sig**. Interventions: raw − mask/mean/shuffle HR@10 +0.088/+0.084/+0.083 (all sig; strongest content dependence of the line). Slot effective rank 20.9 (vs MWC-I 76), mean pairwise cos 0.06.

**Reading**: dropping the user-InfoNCE and using same-user hard negatives recovers most of MWC-I's HR@10 loss (0.0966 → 0.1017, now n.s. vs R3), but NDCG and the top cutoffs stay below R3 (HR@1 0.0331 vs 0.0367–0.0373), so the success criterion (no ranking drop) is not met. Content dependence is very strong, but per the random-donor diagnostic the InfoNCE reshaping cannot carry collaborative information; B is a control, not a lead. C (360056) pending.

### MWC-I-C results (360056, COMPLETED 2026-11-10)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MWC-I-C raw | 0.0263 | 0.0511 | 0.0677 | 0.0977 | 0.0405 | 0.0474 | 0.0570 |
| mask | 0.0076 | 0.0196 | 0.0291 | 0.0595 | 0.0145 | 0.0184 | 0.0283 |
| mean | 0.0074 | 0.0188 | 0.0279 | 0.0552 | 0.0139 | 0.0176 | 0.0264 |
| shuffle_user | 0.0062 | 0.0172 | 0.0260 | 0.0482 | 0.0126 | 0.0161 | 0.0233 |

Paired bootstrap: C − plain A HR@10 −0.0076 [−0.0132, −0.0021] **sig**, NDCG@10 −0.0097 [−0.0130, −0.0064] **sig**; C − 356322 HR@10 −0.0053 n.s., NDCG@10 −0.0098 **sig**. Interventions raw − mask/mean/shuffle HR@10 +0.038/+0.042/+0.049 (sig). Effective rank 66.

**Reading**: the item-prediction head on r *hurts* (C < B at every cutoff: HR@1 0.0263 vs 0.0331, NDCG@10 0.0570 vs 0.0629) and weakens content dependence (shuffle −0.049 vs −0.083). Pulling r toward the target's title embedding competes with what the decoder reads from the slot. InfoNCE line (MWC-I, B, C) closed: none matches R3 ranking; B is the best control.

## 2026-11-10 — GRPO + KL anchor result (360237; partial: shuffle_user intervention still running)

Training 7h39m, 1,684 steps, train_loss 0.0466. kl (every 200 steps): 8.9e-5 → 0.018 → 0.025 → 0.139 → 0.186 → 0.183 → 0.103 → 0.041 → 0.031; last step 0.31. The anchor is active: the policy drifts and is pulled back.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO + KL anchor | 0.0362 | 0.0603 | 0.0766 | **0.0962** | 0.0502 | 0.0569 | 0.0632 |
| GRPO stock (358299) | 0.0223 | 0.0433 | 0.0565 | 0.0781 | 0.0345 | 0.0400 | 0.0469 |
| margin + WIDER stage 1 | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |

Bootstrap HR@10: vs stock GRPO **+0.0181** sig; vs stage 1 **−0.0068** [−0.0095, −0.0043] sig (NDCG −0.0036 sig); vs plain A −0.0091 sig. Interventions (experiment.py, raw 0.0959): mask −0.0433 sig, **mean −0.0190 sig** (stock GRPO: −0.004 n.s.; stage 1: −0.031). shuffle_user pending.

**Reading:** the anchor removes most of the damage (73% of the stock drop) and keeps r's per-user content (the mean swap hurts again). But stage 2 still lowers ranking below its stage-1 start, so the reward (−PPL of the target under noisy r) does not improve ranking on this checkpoint even with a trust region. Possible next steps: larger β (0.04) or a lower lr. The more likely conclusion is that noise-GRPO adds nothing once r is used.

**360237 complete (11h58m).** shuffle_user: raw − shuffle_user HR@10 **+0.0641** [+0.0580, +0.0708], NDCG@10 +0.0457, sig. Interventions vs stage 1 / stock GRPO (HR@10 drop): mask 0.043 vs 0.051 / 0.026; mean 0.019 vs 0.031 / 0.004; shuffle_user 0.064 vs 0.073 / 0.042. So r's use is close to stage 1, not above it. kl means: first 10% 1.6e-4, middle 10% 0.17 (max 1.8), last 10% 0.15 (max 2.1); grad_norm 0.05–0.15. Results copied to `output_dir/grpo_stage2/toys_margin_wider_grpo_kl/`.

### MWC-G3 results (360502, COMPLETED 2026-11-10, 8h15m)

Toys test n = 6,738 (@20 = @10).

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **G3 knn (CF donors)** | 0.0321 | 0.0516 | 0.0665 | 0.0901 | 0.0435 | 0.0496 | 0.0573 |
| off (no evidence) | 0.0270 | 0.0463 | 0.0549 | 0.0739 | 0.0380 | 0.0416 | 0.0477 |
| random donors | 0.0089 | 0.0154 | 0.0184 | 0.0233 | 0.0128 | 0.0140 | 0.0156 |
| swap (other row's donors) | 0.0114 | 0.0206 | 0.0285 | 0.0401 | 0.0167 | 0.0198 | 0.0236 |
| slot mask | 0.0064 | 0.0151 | 0.0223 | 0.0313 | 0.0115 | 0.0144 | 0.0173 |
| slot mean | 0.0240 | 0.0413 | 0.0500 | 0.0663 | 0.0339 | 0.0375 | 0.0428 |
| slot shuffle_user | 0.0110 | 0.0223 | 0.0297 | 0.0393 | 0.0175 | 0.0206 | 0.0237 |
| 356322 margin+WIDER | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap (HR@10 / NDCG@10): knn − off +0.0162 / +0.0095; knn − random +0.0668 / +0.0417; knn − swap +0.0500 / +0.0337; knn − slot mask/mean/shuffle +0.059/+0.024/+0.051 HR@10 — all significant. knn − plain A **−0.0151 [−0.0211, −0.0093]** / −0.0095 [−0.0131, −0.0058] sig; knn − 356322 −0.0129 / −0.0096 sig.

**Gates**: (i) PASS — centred evidence cross-row cos 0.30 → 0.008 (raw evidence = huge constant, ‖μ‖ 25 → 1038, plus a small varying part); (ii) PASS — donor outcomes are used specifically (swap −0.050, random −0.067 HR@10); (iii) **FAIL** — −0.015 HR@10 vs plain R3.

**Training diagnostics** (run log): history-swap margin loss = 0.200 = m throughout (swapping r0 never changes NLL) and WIDER penalty on r0 saturated at 1.000 → **r0 collapsed to one direction**; the slot content is donor evidence only. Evidence-swap gap 0.00 → 0.40 nats; item hit@10 from r1 0.056 vs r0 0.022.

**Where the loss is** (`tmp/g3_split.py`): target among the 32 donor outcomes for 712 rows (10.6%) — there R3 already hits 0.583 (G3 0.518); other 6,026 rows R3 0.049, G3 0.040. Copy rows R3 0.513 / G3 0.448; non-copy 0.042 / 0.035. G3 is lower on every subset. Top-10 share with a history brand 0.44 (G3) vs 0.62 (R3); only 13% of G3's top-10 are donor-outcome titles.

**Reading**: the evidence is genuinely read (first run in the line where collaborative content changes predictions by a large margin), but it **substitutes for history-title reading instead of adding to it**: r0 collapsed, history-brand share fell, and the rows where donors contain the target are rows R3 already gets (0.58 hit rate) — the CF information on Toys is largely redundant with what the decoder reads from the titles. Requirement 6 of the G3 design (evidence must add to, not replace, title reading) failed; the history-swap margin could not prevent the r0 collapse because the decoder reads history directly from the prompt.

## 2026-11-10 — GRPO stage 2 with a ranking reward (user: "Let try GRPO with ranking reward")

**Why.** Noise-GRPO's reward is the target's perplexity under noise on r, with teacher-forced completions. Once r is used, noise lowers the target likelihood, the advantages turn mostly negative, and stage 2 unlearns (358299 −0.025; anchored 360237 −0.007). The new design rewards the ranked list directly.

**Design (user choices: hit + rank penalty; r producer only; margin + WIDER and plain R3).**
- **Group:** for each training prompt, the model's own top-8 titles from constrained beam search (the evaluation decoder: same prefix tree, num_beams = num_return_sequences = 8, length penalty 1).
- **Reward:** the beam at rank k gets +1 if it is the target, otherwise −0.5/log2(k+1) (MiniOneRec/ReRe-style ranking reward; groups without a hit still push confident wrong titles down).
- **Advantage:** GRPO's (r − group mean)/(group std + 1e-4).
- **Loss:** TRL's token-level clipped surrogate + β·k3 KL. The KL reference is anchored to the frozen stage-1 model with its own r. The completions are the policy's own outputs, so k3 now estimates the real KL.
- **What trains:** no noise on r; only the pooling attention trains, LLM frozen.
- **Settings:** lr 1e-5, β 0.04, batch 256 (32 prompts × 8), 1 epoch.

**bf16 finding.** In 358299 / 360237 only 13–24% of the attention weights ever changed (bf16 rounds most lr-1e-5 Adam updates to zero), so stock stage 2 trains a partly frozen module. New `--attention_fp32` (ATTN_FP32=1) keeps the attention in fp32; `generate_embs` casts the hidden states to the attention's dtype.

**CPU test** (tiny random Qwen2 with the real tokenizer and Toys data): beams are valid catalogue titles; the prompt ends in "### Response: \n<|Thought|>"; advantages increase with rank; grad > 0; after 4 steps all 6 attention tensors changed (fp32) and nothing else; kl rises from 0 to 4e-4.

Commit b3f54bd3, frozen `output_dir/grpo_stage2_code/code_b3f54bd3`; env REWARD=rank RANK_PENALTY=0.5 KL_ANCHOR=1 BETA=0.04 LR=1e-5 ATTN_FP32=1.

Runs:
- smoke `toys_mw_rankgrpo_smoke` → full `toys_mw_rankgrpo` (margin + WIDER, 0.1030)
- smoke `toys_mw_rankgrpo_smoke` → full `toys_r3ori_rankgrpo` (R3_ori Toys stage 1 0.1023, r constant across users = "r ignored" control)

Both fulls afterok on the smoke; sent to JobManager.

Jobs: smoke **361058** → full margin + WIDER **361059** and full R3_ori control **361060** (both afterok), submitted by JobManager 2026-11-10.

**Rank-GRPO smoke 361058 passed** (6m15s). All four header lines present; kl ≈ 1e-4 from step 3; clip 0; grad_norm 0.04–0.10. Group hit rate on training prompts 0.13–0.50, hit rank 1.3–5.7. These are training rows already fitted in stage 1, so this is higher than the test HR@10 ≈ 0.10; the signal comes partly from rows the model already fits. 9.0 s/step in the smoke; estimated full run ≈ 35 s/step → ~16–17 h of training + ~3.5 h eval. 361059 started on album; 361060 pending.

## 2026-11-10 — Rank GRPO v1 cancelled (361059 / 361060), fix: no-hit groups get advantage 0

Both full runs lowered their own training hit rate (group has the target among its 8 beams). Means over the first / middle / last 10% of logged steps:

| run | hit rate | hit rank | kl |
|---|---|---|---|
| 361059 margin + WIDER | 0.275 → 0.239 → 0.212 | 2.61 → 2.28 | 2.7e-3 → 4.6e-2 |
| 361060 R3_ori | 0.223 → 0.168 → 0.175 | 2.72 → 2.08 | 3.5e-4 → 6.1e-2 |

Each 100-step window covers ≈3,200 prompts (SE ≈ 0.008), so the drop is real.

**Cause.** In the 75–80% of groups without a hit, rewards depend only on rank position (rank 1 always −0.5, rank 8 always −0.16). The policy cannot raise them, and the gradient just pushes down whatever is ranked first, flattening every list, including items that are correct for other users. Hit groups still learned (hit rank improved), but the flattening dominated.

Cancelled with user approval at 795 / 750 steps. Outputs and logs kept (`Toys_and_Games_R3_grpo_toys_{mw,r3ori}_rankgrpo`, run_logs 361059/361060).

**Fix (commit bffc90af, RANK_SKIP_NOHIT=True default):** advantage × 1[group has a hit]. Inside hit groups the target gets +1 and wrong beams are penalised by rank. Synthetic check: target at rank 3 gets advantage +2.39, ranks 1–2 get −0.90 / −0.49, a no-hit group gets 0. About 20–25% of prompts carry signal. KL still applies to all rows.

Resubmitted via JobManager: smoke `toys_mw_rankgrpo2_smoke` → `toys_mw_rankgrpo2` + `toys_r3ori_rankgrpo2`.

Jobs (v2): smoke **361690** → margin + WIDER **361691** and R3_ori **361692** (afterok).

## 2026-11-11 — Rank GRPO v2: early verdict, scratch quota failure, resubmission

**Early verdict (fix works).** Mean group hit rate / kl per 100-step window:

| run | 1–100 | 101–200 | 201–300 | 301–400 |
|---|---|---|---|---|
| v2 361691 margin + WIDER | 0.277 / 4.5e-4 | 0.270 / 1.2e-3 | 0.296 / 1.7e-3 | 0.294 / 2.1e-3 |
| v1 361059 | 0.273 / 5.8e-3 | 0.253 / 2.4e-2 | 0.245 / 3.3e-2 | 0.237 / 3.8e-2 |
| v2 361692 control | 0.228 / 2.1e-4 | 0.233 / 2.1e-3 | | |
| v1 361060 control | 0.221 | 0.197 | | |

In v2 361691, reward_mean went from −0.200 to −0.197.

So the hit rate now holds or rises (+0.017 over 400 steps vs −0.036 in v1), with kl ~20× lower.

**Quota failure.** Scratch reached 498/500 G (other projects ≈ 240 G). 361692 died saving checkpoint-200; 361691 died saving checkpoint-400 (checkpoint-200 complete).

With user approval I deleted 8 of my scratch dirs:
- the rankgrpo / rankgrpo2 smoke dirs
- the v1 mw / r3ori dirs
- transition_care_failed_quota
- the incomplete r3ori_rankgrpo2 dir
- the stock GRPO and KL-anchor GRPO model dirs (results in output_dir/grpo_stage2)
- the incomplete mw checkpoint-400

Scratch is now 466 G.

**Resubmission.** New sbatch option RESUME_FROM (commit c9550c22, frozen code_c9550c22). margin + WIDER resumes from checkpoint-200; the R3_ori control restarts fresh. Sent to JobManager.

Jobs (v2 resubmission): margin + WIDER resumed **362509**, R3_ori control fresh **362510**.

### 2026-11-11 — Why the warm-up InfoNCE runs lose ranking (row split, `tmp/rowsplit_bc.py`)

| run | all HR@10 | copy rows (n=905) | non-copy (n=5,833) | top-10 share: exact history item | top-10 share: history brand |
|---|---:|---:|---:|---:|---:|
| plain A | 0.1052 | 0.5127 | 0.0420 | 0.101 | 0.623 |
| 356322 | 0.1030 | 0.4785 | 0.0447 | 0.104 | 0.583 |
| MWC-I | 0.0966 | 0.4818 | 0.0369 | 0.127 | 0.631 |
| MWC-I-B | 0.1017 | 0.4873 | 0.0418 | 0.127 | 0.646 |
| MWC-I-C | 0.0977 | 0.4773 | 0.0387 | 0.139 | 0.685 |

All InfoNCE variants hold or gain on copy rows but lose on non-copy rows (−0.003 to −0.008 vs 356322), and all re-recommend history items more (exact-item share 0.127–0.139 vs 0.10). Reading: InfoNCE positives defined by user (and, weakly, by target) teach r what is stable across a user's rows — their history items and brands — so the slot the decoder now reads strongly becomes a history/identity summary and steers generation back toward already-bought items. That helps on the 13% copy rows and hurts the 87% where the next item is new. Supporting: user-InfoNCE fell to 0.28 (near-perfect user ID), same-user cos 0.80 → 0.87; removing the user term (B) recovers most of the loss; the item head (C) pulls r toward title embeddings, which on Toys sit close to history titles, and copies most (0.139) with the second-worst non-copy score. Target InfoNCE stayed ~0.5 nats below chance, so it adds mostly noise rather than counteracting signal. Secondary: total loss jumps at the phase switch (MWC-I 1.41 → 1.55, B 1.07 → 1.16). Not yet tested directly: cosine of r to the mean history-item embedding vs to the target embedding.

## 2026-11-11 — margin + WIDER vs original R3 on copy / non-copy rows (user request)

Script `analysis/results/copy_rows_mw_vs_r3/compare.py` (output.txt, rows.csv). Toys test n = 6,738.
- **Runs:** margin + WIDER 356322 (`output_dir/margin_wider/.../eval_raw.json`), R3_ori Toys stage 1 (`R3_ori/output_dir/Toys_and_Games/latent/final_result.json`), plain R3 retrain A.
- **Copy row:** the target was bought on the same day as a history item of the same brand (first title word) with title Jaccard ≥ 0.3. That gives 836 rows (12.4%). Timestamps are day-granular. The 2026-11-09 MWC-I analysis reported 905 rows with a nominally identical rule; its exact code was not found, so the numbers there are not directly comparable.
- **Bootstrap:** paired, 2,000 resamples, seed 42, within each subset.

| rows | n | run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| all | 6,738 | margin + WIDER | .0373 | .0634 | .0803 | .1030 | .0668 |
| | | R3_ori | .0377 | .0607 | .0778 | .1023 | .0659 |
| | | plain A | .0367 | .0625 | .0795 | .1052 | .0667 |
| copy | 836 | margin + WIDER | .2368 | .3517 | .4115 | .5072 | .3596 |
| | | R3_ori | .2261 | .3493 | .4330 | .5335 | .3652 |
| | | plain A | .2201 | .3589 | .4354 | .5443 | .3675 |
| non-copy | 5,902 | margin + WIDER | .0090 | .0225 | .0334 | .0457 | .0253 |
| | | R3_ori | .0110 | .0198 | .0274 | .0412 | .0235 |
| | | plain A | .0107 | .0205 | .0291 | .0430 | .0241 |
| non-copy & same-brand | 1,334 | margin + WIDER | .0322 | .0780 | .1169 | .1574 | .0881 |
| | | R3_ori | .0382 | .0660 | .0937 | .1402 | .0799 |
| other brand | 4,568 | margin + WIDER | .0022 | .0063 | .0090 | .0131 | .0070 |
| | | R3_ori | .0031 | .0063 | .0081 | .0123 | .0070 |

Bootstrap HR@10, margin + WIDER − R3_ori:
- all +0.0007 [−0.0050, +0.0065]
- copy −0.0263 [−0.0562, +0.0012]
- non-copy +0.0046 [−0.0003, +0.0095]
- non-copy & same-brand +0.0172 [−0.0008, +0.0367]

All n.s., two of them borderline. Against plain A, copy rows −0.0371 [−0.0658, −0.0108] **sig**. Decomposition of the +0.0007 total gap vs R3_ori: copy rows −0.0033, non-copy +0.0040.

Hit overlap with R3_ori:

| rows | both | margin + WIDER only | R3_ori only |
|---|---:|---:|---:|
| copy | 354 | 70 | 92 |
| non-copy | 143 | 127 | 100 |

**Reading:** the equal totals hide a trade. margin + WIDER copies less well (copy rows −0.026 vs R3_ori, −0.037 vs plain A, sig) and does better on non-copy rows (+0.0046, borderline; +0.017 on same-brand non-copy rows), i.e. on genuine next-item prediction. Except at HR@1 (non-copy 0.0090 vs 0.0110), its non-copy gains are at ranks 2–10. The direction matches L7 (whose whole gap was on copy rows), but here the non-copy gain pays for it. Where r carries content, the model leans less on copying the history.

## 2026-11-11 — margin + WIDER on CDs (replicate the copy / non-copy split; user: "how about CDs dataset")

No CDs margin + WIDER run existed. Baseline: R3_ori CDs stage 1 (`R3_ori/output_dir/CDs_and_Vinyl/latent`, Qwen2.5-1.5B base). Its metric.txt gives HR@1/3/5/10 0.0495 / 0.0707 / 0.0814 / 0.0996 and NDCG 0.0616 / 0.0660 / 0.0719. With the same copy rule: n = 6,157, copy rows 368 (6.0%; Toys 12.4%); R3_ori HR@10 all 0.0989 / copy 0.660 / non-copy 0.0632. With 94% of rows non-copy and a higher non-copy hit rate than Toys (0.063 vs 0.041), CDs gives more power on the non-copy question.

Submitted via JobManager: margin + WIDER CDs, code 6fac3d4e (= 356322), CATEGORY=CDs_and_Vinyl, BASE_MODEL=Qwen/Qwen2.5-1.5B, margin 0.2 × 1.0, WIDER 0.1. Output `CDs_and_Vinyl_margin_0.2_wider_0.1`. Then rerun `analysis/results/copy_rows_mw_vs_r3/compare.py` with CDs paths.
Job **362561** (pending).

## 2026-11-11 — MWC-M: same-motive InfoNCE (user's definition; approved)

User's goal: latent reasoning carries collaborative structure — two users buying the same item → similar r (same motivation); one user's two purchases → similar r only when the items are similar. Neither MWC-I (all same-user rows positive → identity code) nor B (same-user different-target rows as hard negatives) implemented the within-user rule.

Pair statistics (Toys train, `tmp/motive_pairs.py`): 53,898 rows, 10,167 users (median 4 rows); same-item other-user positive for 99.8% of rows (median 10); Qwen3 title cosine of random item pairs median 0.63 / p90 0.73; within-user pairs ≥ 0.8: 16% of pairs, rows with ≥ 1 such positive 40% (44% incl. repeat purchases).

Design (approved: threshold 0.8, same-user dissimilar neutral, cross-user exact item only): epoch 1 margin 0.2 + WIDER 0.1 (356322 recipe) while writing every row's r into a persistent bank indexed by train-row id (all 53.9k rows, survives preemption); epochs 2–3 + multi-positive InfoNCE (weight 0.1, τ 0.1, LayerNorm+Linear 256-d projection), positives ≤ 16 cross-user same item + ≤ 16 within-user similar item, 256 bank negatives (other users, different item); same-user dissimilar rows never enter the loss. Inference = plain R3. Property probe `src/prototype_grounding/motive_probe.py` (raw r, first 20k train rows): cos same-item other-user vs random other-user, precision@10 of same-item among nearest other-user latents, within-user cos similar vs dissimilar items — to run on MWC-M, 356322, plain A, MWC-I-B.

Code `src/latent/motive_contrastive.py`, `latent_attention_train_motive.py`, sbatch `METHOD=motive` (eval = infonce branch: ranking, interventions, bootstrap, rank, trace); commit 0d58bf29, frozen `code_0d58bf29`. CPU test: definitions verified on 200 rows; loss 3.95 random → 0.03 with aligned positives; grads reach r and projection. Smoke 362604 submitted.
- Smoke 362604 COMPLETED (3 min): both phases ran (InfoNCE ON at epoch 1), contrastive path executed on GPU with real positives (256-row sample → bank 0.5% full, so most steps have none), eval/interventions/bootstrap complete. Full run handed to the JobManager session for submission (2026-11-11; job submission now goes through JobManager per user).
- **Full run 362616** (`R3_mwcM`, `Toys_and_Games_margin_0.2_wider_0.1_motive`) submitted by JobManager 2026-11-11, PENDING; constraint "a40|l40s|a100|l40" (JobManager keeps jobs off H100 per user rule). Log `run_logs/R3_mwcM.362616.out`.

## 2026-11-11 — MWC-M2: SASRec-sequence motive positives (user-approved, parallel to MWC-M)

User's relaxation: replace "same user + similar item" by "similar sequence (SASRec) + similar item"; K = 20; no same-user rule; retrieved sequences must not all come from one user. Implementation: SASRec sequence vectors per train row (`output_dir/sasrec_latent/prep/Toys_sasrec_sequence.pt`, 128-d, Cornac SASRec, in-sample on train), top-20 cosine neighbours excluding the anchor's own user and **at most one row per neighbour user** (20 distinct users); within positives = those with next-item title cos ≥ 0.8 or identical; all same-user rows neutral; cross-user same-item positives and everything else as MWC-M. Pre-check (4,000 rows, top-20 any user): 11% of neighbours same user, 15.8% have a similar next item vs 1.3% random. Built index: rows with ≥ 1 SASRec positive 53.9% (mean 2.8), cross 99.8%, either 99.9%; build 9 s. Unit test: 300 rows verified (20 distinct other users, positives ⊂ neighbours, similar items). Code commit 87278c64 (`--within_mode sasrec`, sbatch WITHIN_MODE/SASREC_K/SASREC_FILE), frozen `code_87278c64`; smoke → full handed to JobManager (`tmp/submit_m2.sh`, output `Toys_and_Games_margin_0.2_wider_0.1_motive_sasrec`).
- Smoke 362617 (R3_mwcM2_smoke) submitted by JobManager, PENDING; full run follows on pass. MWC-M 362616 RUNNING (lunar).
- Smoke 362617 COMPLETED (3:58, lagoon; SASRec index built: within 0.539, cross 0.998). **Full MWC-M2 run 362632** (R3_mwcM2) submitted by JobManager 2026-11-11, PENDING. Log run_logs/R3_mwcM2.362632.out.

### MWC-M results (362616, COMPLETED 2026-11-11, 5h21m, lunar) — **first run meeting the criterion with collaborative training**

Toys test n = 6,738 (@20 = @10).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MWC-M raw** | 0.0355 | 0.0640 | 0.0816 | **0.1057** | 0.0521 | 0.0593 | **0.0670** |
| mask | 0.0030 | 0.0062 | 0.0102 | 0.0157 | 0.0048 | 0.0064 | 0.0082 |
| mean | 0.0030 | 0.0071 | 0.0117 | 0.0175 | 0.0053 | 0.0072 | 0.0091 |
| shuffle_user | 0.0049 | 0.0110 | 0.0153 | 0.0217 | 0.0083 | 0.0100 | 0.0121 |
| 356322 margin+WIDER | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap: MWC-M − plain A HR@10 +0.0004 [−0.0052, +0.0061] n.s., NDCG@10 +0.0003 n.s.; − 356322 HR@10 +0.0027 n.s., NDCG@10 +0.0002 n.s. Interventions raw − mask/mean/shuffle HR@10 +0.090/+0.088/+0.084, NDCG@10 +0.059/+0.058/+0.055 (all sig). Slot effective rank 16.6, mean pairwise cos 0.026.

Training (run log): history-swap margin 0.20 → 0.02 (learned), WIDER 0.70 → 0.04; motive InfoNCE 3.43 → 2.23; projection-space cos at end: same-item other-user 0.36, same-user similar-item 0.75, negatives 0.18; eval loss 0.803 / 0.605 / 0.593 (no epoch-3 overfit); positives available: cross 100%, within 44%.

Row split (`tmp/rowsplit_m.py`, `tmp/subset_boot.py`): non-copy rows MWC-M **0.0466** (best of the line) vs plain A 0.0420 (+0.0046 [−0.0003, +0.0094], borderline n.s.) and 356322 0.0447 (+0.0019 n.s.); copy rows 0.4862 vs A 0.5127 (−0.0265 sig) / 356322 0.4785 (n.s.). Top-10 exact-history-item share 0.111 (A 0.101, 356322 0.104, MWC-I/B 0.127) — no history-copying drift.

**Reading**: success criterion met — content dependence strongest of the line (Δ ≈ −0.09 HR@10) with ranking equal to plain R3 at every cutoff. Versus MWC-I/B the difference is the positive definition: similar-item gating within a user removes the identity pressure (copying 0.111 vs 0.127) and the full-train bank gives every row its same-item positives; the non-copy rows, where MWC-I lost, are now the best of the line. Whether the raw latent shows the collaborative structure is measured by the motive probe (363014, pending: MWC-M, 356322, MWC-I-B; plain A checkpoint no longer exists).
- Motive probe 363014 FAILED (code bug: mean helper on numpy array, crashed after computing the first tag); fixed in f33e7cd6 (frozen code_f33e7cd6), resubmission requested from JobManager.
- Probe resubmitted as 363016 (code_f33e7cd6), PENDING.

### Motive probe (363016, raw slot latent r, first 20k train rows; same-item anchors 18,565; within-user anchors 8,838 similar / 18,110 dissimilar)

| model | cos same item, other user | cos random other user | gap | P@10 same item among nearest other-user latents (chance 0.0005) | within-user cos similar items | dissimilar items | gap | mean pairwise cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 356322 margin+WIDER | 0.228 | 0.084 | +0.145 | 0.0265 | 0.744 | 0.686 | +0.058 | 0.086 |
| MWC-I-B | 0.264 | 0.053 | +0.211 | 0.0278 | 0.742 | 0.686 | +0.056 | 0.058 |
| **MWC-M** | **0.294** | **0.021** | **+0.273** | **0.0341** | **0.794** | 0.724 | **+0.071** | 0.024 |

Reading: the cross-user property is clearly stronger in the raw latent — same-item gap nearly doubles vs the control (+0.273 vs +0.145) and same-item nearest-neighbour precision rises 29% (0.0341 vs 0.0265; 70× chance), while unrelated users are pushed apart (random cos 0.021). The within-user property moves less: similar-item pairs gain (0.794 vs 0.744) but dissimilar pairs also rise (0.724 vs 0.686; they are neutral in the loss and share history prefixes), so the gap grows only +0.058 → +0.071. Caveat: measured on training rows (in-sample for the positives); a held-out check (test rows vs train rows with the same target) is still to do. MWC-I-B sits between control and MWC-M on the cross-user gap and does not improve the within-user gap.

### 2026-11-11 — Slice analysis: where MWC-M wins / loses vs plain R3 (`tmp/slices.py`, paired bootstrap per slice)

HR@10 (A / 356322 / MWC-M, M − A [95% CI]):
- copy rows (13%): 0.513 / 0.479 / 0.486, **−0.0265 [−0.052, −0.001] sig**; HR@1 0.203 → 0.211 (M better at rank 1, worse at listing variants in 2–10). The deficit is already in 356322 (0.479) → it comes from the margin+WIDER base, not the motive loss.
- non-copy (87%): 0.0420 / 0.0447 / 0.0466, +0.0046 [−0.0003, +0.0094]; **non-copy & target not among CF-donor outcomes (5,505 rows): 0.0258 → 0.0307, +0.0049 [+0.0007, +0.0093] sig** (+19% relative). Non-copy HR@1 lower (0.0108 → 0.0082).
- Gains concentrate on harder / richer-context rows: history 8+ +0.0066 (0.077 → 0.084), heavy users 11+ train rows +0.020 [0.000, +0.044] (0.038 → 0.058), popular targets (30+) +0.013, 1–60-day gaps +0.007–0.011; losses on 3–4-item histories (−0.005) and rare targets (1–2, −0.007). Unseen targets (5% of test) 0 for every model.
- Per-row agreement: both hit 7.8%, only A 2.7%, only M 2.75%, union 13.3% (vs 10.5%); copy rows only-A 9.2% vs only-M 6.5%; non-copy only-A 1.7% vs only-M 2.2%.
- Failure types on misses similar for all (≈ 10–11% history items, 60% history brand in top-10).
Decomposition: copy deficit costs −0.0036 overall, non-copy gain gives +0.0040 → net ≈ 0. Matching A on copy rows while keeping the non-copy gain would give ≈ 0.109 HR@10.
- Follow-up (`tmp/slices2.py`): MWC-M is worse than A in two ways. (1) copy rows: HR@10 −0.0265 sig but NDCG@10 −0.0012 n.s. and HR@1 +0.0077 → only tail positions 2–10 lose variants. (2) **non-copy HR@1**: −0.0026 [−0.0051, 0.0000] overall, significant in sub-slices: target among CF-donor outcomes 0.107 → 0.064 (−0.043 sig), targets with 10–29 train occurrences −0.0062 sig, history 3–4 −0.0048 sig — on non-copy rows M recalls more targets into the top-10 (+0.0046) but ranks the right one first less often; non-copy NDCG@10 equal (+0.0005). Other HR@10 dips (users 6–10 rows −0.009, rare targets −0.003) n.s.
- Rank-1 diagnostic (`tmp/top1_diag.py`): non-copy rank-1 disagreements are few and two-way (A@1 only 36 rows, M@1 only 21); the wrong top-1 is a sibling of the target in both directions (same brand 44–52%, title cos ≥ 0.8 for 50–57% vs 11% same brand on ordinary misses). User redirected focus to @5/@10 (sibling hinge dropped).
- @5 slices (`tmp/slices5.py`), M − A: HR@5 all +0.0021, copy +0.0088, non-copy +0.0010; NDCG@5 all +0.0008, copy +0.0104, non-copy −0.0006 (all n.s.). So both effects sit at ranks 6–10: on copy rows M keeps the top-5 (better than A) but drops variants from 6–10 (HR@10 −0.027), on non-copy rows M's extra hits land in 6–10 (HR@5 +0.001 vs HR@10 +0.0046). The prompt has no timestamps, so the model cannot tell copy rows apart at inference; the tail slots are a shared budget between history variants and novel items.

## 2026-11-11 — MWC-M3a / M3b: fix the copy-row tail without a hand-coded row rule (user-approved, parallel)

User objected that switching the hinge off on rule-defined copy rows is hard-coded. Two principled variants of MWC-M (everything else identical, @5/@10 are the target metrics):
- **M3a, self-weighted hinge**: each row's history-swap hinge × stopgrad(1 − exp(−NLL(y | x, r_other))) — demand latent dependence only where the history alone (another row's latent in the slot) does not already predict the target; no thresholds, no extra pass.
- **M3b, hinge as warm-up only**: hinge in epoch 1 (teaches the decoder to read the slot), off in epochs 2–3 (motive InfoNCE + WIDER remain); also skips the negative decode → faster.
Copy flag (same-day near-copy; test 13.43% = analysis definition, train 17.8%) is computed per row for logging only (`hinge_weight_copy` / `hinge_weight_noncopy`). Code commit 364d6927 (`--margin_mode self`, `--margin_epochs 1`; sbatch MARGIN_MODE / MARGIN_EPOCHS), frozen `code_364d6927`; outputs `..._motive_selfmargin` / `..._motive_marginwarm`; smoke → full via JobManager (`tmp/submit_m3a.sh`, `tmp/submit_m3b.sh`). Gate: HR/NDCG@5 or @10 significantly above plain A, content dependence significant, probe structure kept.
- Smokes submitted by JobManager: M3a 363137, M3b 363138 (PENDING); fulls follow on pass.
- M3a smoke 363137 passed (4:15); **M3a full 363141** submitted by JobManager. Smoke hinge weights: ~1.0 at the start (untrained model, every target is hard), by the last smoke steps copy rows 0.77 vs non-copy 0.98 — the self-weight already separates copy rows without the rule. M3b smoke 363138 running.

### MWC-M2 results (362632, COMPLETED 2026-11-11, 7h19m, lagoon)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MWC-M2** (SASRec top-20 distinct-user neighbours) | 0.0343 | 0.0598 | 0.0769 | 0.1045 | 0.0489 | 0.0559 | 0.0648 |
| mask | 0.0092 | 0.0184 | 0.0278 | 0.0416 | 0.0145 | 0.0184 | 0.0228 |
| mean | 0.0091 | 0.0220 | 0.0359 | 0.0530 | 0.0165 | 0.0223 | 0.0278 |
| shuffle_user | 0.0076 | 0.0172 | 0.0263 | 0.0390 | 0.0131 | 0.0168 | 0.0209 |
| MWC-M | 0.0355 | 0.0640 | 0.0816 | 0.1057 | 0.0521 | 0.0593 | 0.0670 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Paired bootstrap (`tmp/m2_vs_m.py`): M2 − A all n.s. (HR@5 −0.0027, NDCG@5 −0.0025, HR@10 −0.0007, NDCG@10 −0.0019); M2 − M HR@5 −0.0047 (CI touches 0), **NDCG@5 −0.0034 sig**, HR@10 −0.0012, NDCG@10 −0.0022 n.s. Interventions raw − mask/mean/shuffle HR@10 +0.063/+0.052/+0.065 (sig; weaker than M's ≈ 0.09). Effective rank 50 (M 16.6). Training: margin 0.17 → 0.02, WIDER 0.07, motive 3.06 → 2.14, projection cos cross 0.35 / within(SASRec) 0.62 / neg 0.18; within positives for 49–54% of rows; eval loss 0.785 / 0.633 / 0.612 (slightly above M's 0.593).
**Reading**: meets the "≈ R3 + content used" criterion but is weaker than MWC-M at every cutoff (significant at NDCG@5) with weaker content dependence. Replacing same-user similar-item positives by other users' SASRec-sequence neighbours does not help; the same-user rule (M) stays the better within-user definition. Probe for M2 to run together with the held-out probe.
- M3b smoke 363138 passed (4:02; hinge off at epoch 1.000); **M3b full 363145** submitted by JobManager. Full runs pending: M3a 363141, M3b 363145.
- Copy-row decline in margin+WIDER (`tmp/slices_mw.py`): 356322 − A on copy rows HR@10 **−0.0343 [−0.062, −0.008] sig** (MWC-M −0.0265 sig), HR@5 −0.0199 n.s. (M +0.0088), NDCG@5 −0.0015 (M +0.0104), NDCG@10 −0.0060 (M −0.0012); HR@1 +0.017 (M +0.008). Share of copy-row slots that carry a history brand: A 0.825 (1–5) / 0.756 (6–10); 356322 0.794 / 0.681; M 0.824 / 0.724. → the decline originates in margin+WIDER (larger there, also visible at @5); the motive loss partly repairs it (top-5 fully, tail partly). Supports the hinge as the cause, which M3a/M3b test.

### 2026-11-11 — Root cause of the copy-row decline: across-run comparison (`tmp/copy_across_runs.py`)

Copy-row HR@10 (905 rows) vs whether the latent's content is used:
- latent unused: plain A 0.513, guidance B/C/D 0.511/0.519/0.501, DivReason SFT 0.511, **WIDER only 0.517**, transition CARE 0.514 → 0.50–0.52.
- latent used (margin hinge): 356322 0.479, MWC-I 0.482, B 0.487, C 0.477, MWC-M 0.486, M2 0.494, MWC-G 0.503, G3 0.448 → 0.45–0.50.
WIDER alone does not cause it; every run trained with the history-swap hinge loses copy rows (only exception near baseline: MWC-G 0.503). Copy-row top-10 history-brand share: unused 0.785–0.792, 356322 0.738, G3 0.606 (G3's evidence replaced title reading).
Hypothesised mechanism: on a copy row the history alone predicts the target; the hinge NLL(y|x,r_own) + m ≤ NLL(y|x,r_other) can then only be met by making copying *conditional on r* — the decoder learns that a history variant may be emitted only when the slot "licenses" it. One latent vector encodes one direction (the actual target), so the other variants of the history items lose their licence and drop out of beam slots 6–10, while the top-5 survive. Planned forward-only test on WIDER-only (no hinge, checkpoint exists), 356322, MWC-M: on copy rows, log-prob of the target and of the other history-brand variants under own r vs another row's r, and target-token attention to history titles vs the slot.

## 2026-11-11 — MWC-M4: per-token "novel" hinge (root-level fix of the copy-row decline; user-approved option 1)

Mechanism (discussion with user): on rows where the history alone predicts the target, NLL(y|x,r_own) ≈ NLL(y|x,r_other); the hinge can only be met by raising NLL under a wrong latent, i.e. the decoder learns "slot overrides history when they disagree". r is a target-peaked one-vector summary (CE + WIDER), so it disagrees with the history about sibling variants → they lose support and leave beam slots 6–10 (copy-row HR@10 0.45–0.50 for every hinge-trained run vs 0.50–0.52 for all runs without it, WIDER-only 0.517). Changing the negative cannot remove this incentive.
Fix: charge the latent only for what the history cannot supply — the hinge is computed on the NLL of **novel target tokens** (normalised token form not among the tokens of any history title; special/punctuation tokens count as copyable); rows without novel tokens contribute 0. No row rule, no threshold, no extra pass; inference unchanged. Everything else = MWC-M.
Check (`tmp/test_novel.py`, 3,000 train rows): novel-token share copy rows 0.16 (32% have none → hinge off), non-copy 0.60 (0.1% none); collator keeps hinge_labels aligned (right padding). Code commit 5ee286e8 (`--margin_mode novel`, `HingeLabelCollator`, logs novel_token_share copy / non-copy), frozen `code_5ee286e8`; smoke → full via JobManager (`tmp/submit_m4.sh`, output `..._motive_novelmargin`).
- M4 smoke 363195 submitted by JobManager (PENDING); M3a 363141 / M3b 363145 still PENDING.
- User proposal (dissimilar SASRec negatives for copy rows) checked (`tmp/neg_sim.py`): the current in-batch negative is already an unrelated sequence — SASRec cos to the random partner median 0.012 (copy anchors) / 0.022 (non-copy), only 2.8–3.4% above 0.3, 44–46% below 0; the most dissimilar 5% of rows sit at ≈ −0.19. Choosing dissimilar negatives would shift cos from ≈ 0.01 to ≈ −0.2, a small change, and the hinge is already satisfied (MWC-M margin 0.02) — the problem is how it is satisfied (copying made conditional on r), not that it is hard. Not run; reply to user pending.

## 2026-11-11 — MWC-M with a smaller margin: m = 0.05 and m = 0.1 (user request)

Same frozen code as MWC-M (code_0d58bf29, 362616), only `MARGIN` changes (0.2 → 0.05 / 0.1); outputs `Toys_and_Games_margin_{0.05,0.1}_wider_0.1_motive`; submitted directly as full runs via JobManager (`tmp/submit_m_margin.sh`). Rationale (user): a smaller required gap on rows where the history already determines the target (NLL(y|x,r_own) ≈ NLL(y|x,r_other)) gives the hinge less reason to make copying conditional on r. Expectation: copy-row tail recovers partly; risk: weaker content dependence. Compare copy-row HR@10, non-copy HR@10, HR/NDCG@5/@10 vs A and MWC-M (m = 0.2), interventions.
- Margin runs submitted by JobManager: m=0.05 job 363201, m=0.1 job 363202 (PENDING). Scratch 405 G used.

## 2026-11-11 — MWC-M6: adaptive one-sided margin (user-approved)

h_i = [ m_i + NLL(y|x, r_own) − sg(NLL(y|x, r_other)) ]₊,  m_i = min(m, α · sg(NLL(y|x, r_other))), m = 0.2, α = 0.5.
(1) Adaptive margin: the demanded gap is capped at a fraction of the loss the history leaves unexplained → ≈ 0 where the history already predicts the target (copy rows), full m elsewhere. (2) One-sided gradient: the negative decode runs under no_grad, so the gap can only be created by improving the own latent — the "make another latent hurt / slot overrides history" route is removed at its source (new vs M3a/M4, which only reduce pressure). Risk: weaker content dependence (dependence must be earned); the epoch-1 warm-up still runs the same hinge. Toy check: copy-like row (neg 0.06) → m_i 0.03, hinge 0.02; grads only on the own-r term. Code commit 032f5a2c (`--margin_mode adaptive --adaptive_alpha 0.5`, logs adaptive_margin / hinge_active by copy / non-copy), frozen `code_032f5a2c`; smoke → full via JobManager (`tmp/submit_m6.sh`, output `..._motive_adaptmargin`).
- M6 smoke 363212 submitted by JobManager (PENDING).

**Rank GRPO v2 mid-run (2026-11-11 ~22:30).** In 100-step windows, the margin + WIDER run (362509) group hit rate is 0.296 → 0.328 (flat to rising), with kl mostly 0.002–0.012. The R3_ori control (362510) has hit rate / kl 0.228 / 2e-4 → 0.234 → 0.248 → 0.252 → 0.244 / 4.2e-3 over steps 1–500, with no v1-style decline. CDs margin + WIDER 362561 eval loss: epoch 1 1.285, epoch 2 1.053.
- JobManager restarted 2026-11-11 (new session, same tracking). M3a full 363141 RUNNING (album, ~22:28); pending: M3b 363145, M4 smoke 363195, m0.05 363201, m0.1 363202, M6 smoke 363212.
- M3a 363141 preempted on album after 1:06:38 (before the first epoch checkpoint, so it restarts training from scratch), requeued automatically; a second preemption moves it to hadylauwresearch a40 (JobManager rule).

## 2026-11-12 — Rank GRPO v2 from margin + WIDER (362509): ranking result

Training completed (1,684 steps, 14h50m, train_loss 0.0085; last step kl 0.0028, hit rate 0.281). Group hit rate per 200-step window: 0.294 → 0.295 → 0.274 → 0.280 → 0.306 → 0.314 → 0.326.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **rank GRPO v2** | 0.0381 | 0.0637 | 0.0809 | **0.1072** | 0.0530 | 0.0601 | **0.0686** |
| margin + WIDER stage 1 (start) | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| R3_ori | 0.0377 | 0.0607 | 0.0778 | 0.1023 | | | 0.0659 |
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

Bootstrap:

| comparison | HR@10 | NDCG@10 |
|---|---|---|
| − start | **+0.0042 [+0.0019, +0.0064] sig** | **+0.0018 [+0.0008, +0.0029] sig** |
| − R3_ori | +0.0049 [−0.0009, +0.0107] n.s. | +0.0027 n.s. |
| − plain A | +0.0019 n.s. | +0.0019 n.s. |

**First stage 2 in this line that improves on its start** (noise-GRPO −0.025, anchored −0.007).

Post-processing (extract → interventions) crashed: experiment.py calls model.attention directly with bf16 states while the GRPO attention is fp32 (code c9550c22). Fix 3b1f2091: SelfAttentionLayer casts its input to its own dtype. Eval-only rerun (SKIP_TRAIN=1) requested for 362509. 362510 will hit the same crash after its ranking; JobManager will submit its eval-only rerun automatically.
Eval-only rerun for 362509: job **363469** (code 3b1f2091, SKIP_TRAIN=1).

## 2026-11-12 — margin + WIDER on CDs (362561) + copy / non-copy split

6h44m, best epoch 2 (eval loss 1.285 / 1.053). Results `output_dir/margin_wider/CDs_and_Vinyl_margin_0.2_wider_0.1/`. Interventions (raw − mode, HR@10): mask +0.0125, mean +0.0068, shuffle_user +0.0039 [+0.0002, +0.0076], all sig. Content is used, but much less than on Toys (shuffle −0.073 there).

Split (`compare.py CDs_and_Vinyl`, output_CDs.txt; copy rule as for Toys; n = 6,157, copy 368 = 6.0%):

| rows | n | run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| all | 6,157 | margin + WIDER | .0525 | .0726 | .0851 | **.1049** | **.0754** |
| | | R3_ori | .0495 | .0707 | .0819 | .0989 | .0717 |
| copy | 368 | margin + WIDER | .4946 | .6005 | .6413 | .6929 | .5890 |
| | | R3_ori | .4755 | .5571 | .6033 | .6603 | .5588 |
| non-copy | 5,789 | margin + WIDER | .0244 | .0390 | .0497 | .0675 | .0427 |
| | | R3_ori | .0225 | .0397 | .0487 | .0632 | .0408 |

(@20 = @10; R3_ori all-row HR@10 here 0.0989 vs its metric.txt 0.0996, string-normalisation difference.)

Bootstrap, margin + WIDER − R3_ori:

| rows | HR@10 | NDCG@10 |
|---|---|---|
| all | **+0.0060 [+0.0006, +0.0114] sig** | **+0.0037 sig** |
| copy | **+0.0326 sig** | **+0.0302 sig** |
| non-copy | +0.0043 [−0.0009, +0.0093] n.s. | |
| same-brand target | +0.0247 sig | |

Hits (both / margin + WIDER only / R3_ori only): copy 230 / 25 / 13; non-copy 254 / 137 / 112.

**Pooled Toys + CDs non-copy rows** (n = 11,691, bootstrap stratified by dataset): HR@10 **+0.0044 [+0.0008, +0.0080] sig**.

**Reading:**
- The non-copy gain replicates with almost the same size (Toys +0.0046, CDs +0.0043); n.s. in each dataset alone, significant pooled.
- The copy-row loss does **not** replicate: on CDs margin + WIDER copies better (+0.033 sig), unlike Toys (−0.026 n.s.).
- On CDs margin + WIDER beats R3_ori overall (HR@10 +0.0060, NDCG@10 +0.0037, both sig) while r's content is used (mean / shuffle hurt).
- M4 smoke 363195 passed (3:06); **M4 full 363511** submitted by JobManager. M6 smoke 363212 running.
- M6 smoke 363212 passed (2:29); **M6 full 363513** submitted by JobManager. Queue: M3a 363141 (requeued), M3b 363145, m0.05 363201, m0.1 363202, M4 363511, M6 363513.
- M3a 363141 restarted from scratch (no checkpoint before preemption); RUNNING on lunar (L40S) since 05:04 2026-11-12, ETA ~10:00-12:00.

**363469 (eval-only rerun, fix 3b1f2091) done:** the ranking reproduces (HR@10 0.1072). Interventions on rank GRPO v2 (raw HR@10 0.1070), raw − mode, all sig:

| mode | HR@10 | NDCG@10 |
|---|---|---|
| mask | +0.0545 [+0.0488, +0.0604] | +0.0385 |
| mean | +0.0282 [+0.0232, +0.0329] | +0.0209 |
| shuffle_user | +0.0712 [+0.0653, +0.0779] | +0.0479 |

The start model (356322) had mask 0.051, mean 0.031, shuffle 0.073, so content use is preserved. Rank GRPO v2 therefore meets the criterion with the best Toys ranking in this line: r used + HR@10 0.1072 (> start, sig; ≥ R3_ori / plain A, n.s.). Results in `output_dir/grpo_stage2/toys_mw_rankgrpo2/`.
- 2026-11-12 scratch quota incident: m=0.1 363202 and M6 363513 FAILED writing checkpoints (scratch ~493–501 G / 500 G). Per user, JobManager deleted their partial checkpoints (~22 G freed, scratch ~432 G) and resubmitted as 363591 / 363592. User asked for single-checkpoint storage: commit 62872468 (frozen `code_62872468`, helper `tmp/submit_v2.sh`) — SingleCheckpointCallback deletes the previous checkpoint before each epoch save; after training trainer_state.json is written next to the final model and the resume checkpoint is deleted (finished run ≈ 3 G). Optimizer state kept (resume correctness) and final-epoch selection kept (comparability). Asked JobManager to cancel 363591/363592 (2 min in) and resubmit on code_62872468; M3a 363141, M3b 363145, m0.05 363201, M4 363511 keep their code (2–6 h in); their resume checkpoints are to be removed after completion.
- Resubmitted on code_62872468: m=0.1 363593 (RUNNING), M6 363594 (PENDING). Old-code runs kept: M3a 363141, M3b 363145, m0.05 363201, M4 363511 (delete their resume checkpoints after completion).

### 2026-11-12 — First results of the copy-row fixes (ranking only; interventions still running) (`tmp/variants_eval.py`)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain A | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 | 0.5127 | 0.0420 |
| MWC-M (m 0.2) | 0.0355 | 0.0640 | 0.0816 | 0.1057 | 0.0521 | 0.0593 | 0.0670 | 0.4862 | 0.0466 |
| **M3a** self-weighted hinge (363141) | 0.0371 | 0.0650 | 0.0809 | **0.1080** | 0.0531 | 0.0596 | 0.0684 | 0.5061 | **0.0463** |
| **M3b** hinge epoch 1 only (363145) | 0.0380 | 0.0649 | **0.0827** | **0.1080** | 0.0535 | **0.0608** | **0.0690** | **0.5160** | 0.0447 |
| m = 0.05 (363201) | 0.0373 | 0.0649 | 0.0776 | 0.1060 | 0.0534 | 0.0586 | 0.0677 | 0.5039 | 0.0442 |

Paired bootstrap vs plain A (HR@5 / NDCG@5 / HR@10 / NDCG@10): M3a +0.0013 / +0.0012 / +0.0028 / +0.0017; M3b +0.0031 / +0.0024 / +0.0028 / +0.0022; m 0.05 −0.0019 / +0.0001 / +0.0007 / +0.0010 — all n.s. Copy rows vs MWC-M: M3b +0.0298 [+0.0055, +0.0541] **sig**, M3a +0.0199 (n.s.), m 0.05 +0.0177 (n.s.); non-copy vs MWC-M: M3a −0.0003, M3b −0.0019, m 0.05 −0.0024.
Reading: reducing the hinge's pressure on history-determined rows recovers the copy rows as the mechanism predicted (M3b back to plain-A level, significantly above MWC-M); M3a keeps nearly all of MWC-M's non-copy gain while recovering most of the copy loss (best balance); m 0.05 recovers less and loses @5. All three now lead plain A at @5/@10 by +0.001–0.003 but not significantly. Content dependence (interventions) pending — decisive for M3b, whose hinge is off in epochs 2–3.

## 2026-11-12 — M3a / M3b on margin + WIDER only (no motive InfoNCE; user request)

Same trainer with MOTIVE_W = 0 (commit a21e067c skips the InfoNCE computation when its weight is 0; frozen `code_a21e067c`, includes the single-checkpoint storage fix): **MW-3a** = margin + WIDER with the self-weighted hinge; **MW-3b** = margin + WIDER with the hinge in epoch 1 only (epochs 2–3: CE + WIDER). Outputs `Toys_and_Games_margin_0.2_wider_0.1_selfmargin` / `_marginwarm`; smoke → full via JobManager (`tmp/submit_mw.sh`). Reference for the fixed-hinge base = 356322 (HR@10 0.1030, copy 0.479, non-copy 0.0447; trained with margin_loss_attention, a different implementation of the same recipe). Question: does the hinge fix alone (without the motive loss) recover copy rows and beat R3, and how much of M3a/M3b's non-copy gain comes from the motive term.
- mw3a smoke 363600, mw3b smoke 363601 submitted by JobManager (PENDING). Smoke dirs to be deleted after pass only with user OK.
- mw3a smoke 363600 passed (4:23); **mw3a full 363602** submitted by JobManager. mw3b smoke 363601 pending.
- mw3b smoke 363601 passed (3:59); **mw3b full 363603** submitted by JobManager.

## 2026-11-12 — Scratch: one checkpoint per run (user)

Scratch hit ~501 G (other users' jobs died). With user approval I deleted the failed-run checkpoints: 362509's `model/checkpoint-1684` (final model kept) and `Toys_and_Games_transition_latent_v2_failed_quota`; scratch is now 444 G. Commit 1c3b4c8c sets save_total_limit=1 in every trainer default and sbatch (17 files; latent_collab_drift stays 0). Frozen copies `output_dir/{grpo_stage2_code,margin_wider_code}/code_1c3b4c8c` are for new submissions.

### M3a full results (363141, COMPLETED 2026-11-12, 5h22m) — ranking ≈ R3+ but **latent unused**

Ranking: HR@1/3/5/10 0.0371/0.0650/0.0809/0.1080, NDCG@3/5/10 0.0531/0.0596/0.0684 (vs A all n.s., see above). Interventions raw − mask / mean / shuffle_user HR@10 +0.0003 / +0.0009 / −0.0001 (all n.s.; MWC-M ≈ +0.084–0.090) → **fails the content-dependence criterion**. eval loss 0.809 / 0.606 / 0.604.
Why (run log): the self-weight 1 − exp(−NLL(y|x,r_other)) uses the per-token average NLL, which falls for every row as CE drops, so the hinge weakened everywhere, not only on copy rows: mean weight 0.87 → 0.46 (end epoch 1) → 0.32 (end), copy 0.67 → 0.14, non-copy 0.90 → 0.37. The hinge was never satisfied (margin loss 0.17 → 0.06, MWC-M reached 0.02), so the decoder never came to depend on r; projection-space motive structure is also weaker (same-item cos 0.21 vs MWC-M 0.36). Copy rows "recover" (0.506) because r is ignored, as in plain R3 — not the intended fix. Lesson: a gate must be relative (e.g. normalised within the batch / against a running mean), not an absolute function of the token NLL. M6's adaptive margin (min(0.2, 0.5·NLL)) stays near full strength for NLL ≥ 0.4, so it should not decay the same way; M3b / M4 / M6 / MW-3a pending (MW-3a uses the same weight → expect the same failure).
Storage: M3a's leftover checkpoint-1263 removed after completion (trainer_state.json kept in results); final model kept.
- M3a-n (normalised self-weight min(2, u/EMA(u)), with motive) commit 253fba16, frozen code_253fba16, helper tmp/submit_m3an.sh (m3an smoke|full); MW-3a 363602 cancelled (partial dir left in scratch).
- M3a-n smoke 363636 passed; **M3a-n full 363638** submitted. M3b 363145 preempted after training finished (restarted on lunar → ranking/interventions); M4 363511 preempted, resumed from checkpoint-422 on analog. Running: m0.05 363201, m0.1 363593, M6 363594, MW-3b 363603.
- M6 363594 preempted at 1:54 after a complete checkpoint-422 (10:54; optimizer 6.2 G) → resumes from epoch 1. Epoch-1 log: adaptive margin separates copy 0.134 vs non-copy 0.194 (no decay, unlike M3a), BUT margin loss 0.181 ≈ mean m_i 0.180 with the hinge active on 100% of rows → own-r NLL ≈ other-r NLL, no gap created (MWC-M had 0.161 at this point, 0.02 at the end). Early warning: with the one-sided gradient the latent may never become used (like G3's pinned 0.200).

## 2026-11-12 — Rank GRPO v2 R3_ori control (362510) result: the gain does not need a used r

Training done (crashed afterwards in extraction like 362509; eval-only rerun 363654 for the interventions).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **rank GRPO on R3_ori** | 0.0389 | 0.0634 | 0.0793 | **0.1098** | 0.0530 | 0.0595 | **0.0694** |
| rank GRPO on margin + WIDER | 0.0381 | 0.0637 | 0.0809 | 0.1072 | 0.0530 | 0.0601 | 0.0686 |
| R3_ori (start) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | | | 0.0659 |

Bootstrap HR@10:

| comparison | HR@10 | NDCG@10 |
|---|---|---|
| control − R3_ori | **+0.0076 [+0.0046, +0.0104] sig** | +0.0035 sig |
| control − rank-GRPO margin + WIDER | +0.0027 n.s. | |
| control − margin + WIDER | +0.0068 sig | |
| control − plain A | +0.0046 [+0.0000, +0.0091] borderline | NDCG +0.0027 sig |

**Reading:** the ranking-reward gain is at least as large when r is constant across users (R3_ori Toys r has cos 1.0 between users). Tuning the pooling attention then moves one shared vector, i.e. a learned soft prompt/bias trained with a ranking objective. So the gain on margin + WIDER is not evidence that the reward exploits per-user r content; it is a generic ranking-objective gain. The margin + WIDER arm keeps its content use (shuffle −0.071), the control has none by construction (interventions pending, 363654).

Next (user: "try with CDs"): both arms on CDs with code 1c3b4c8c. Smoke `cds_mw_rankgrpo2_smoke` → `cds_mw_rankgrpo2` + `cds_r3ori_rankgrpo2`.
CDs rank-GRPO jobs: smoke **363659** → margin + WIDER **363660** and R3_ori **363661** (afterok). Checkpoint deletion after each run is done by this session (JobManager deletes only with explicit user OK).

### m = 0.05 full results (363201, COMPLETED 2026-11-12, 7h24m)
Ranking HR@1/3/5/10 0.0373/0.0649/0.0776/0.1060, NDCG@3/5/10 0.0534/0.0586/0.0677 (vs A all n.s.; copy 0.504, non-copy 0.0442). Interventions raw − mask / mean / shuffle HR@10 +0.056 / +0.010 / +0.025, NDCG@10 +0.035 / +0.007 / +0.016 (all sig) — latent used, but 3–8× weaker than MWC-M (m 0.2: +0.090 / +0.088 / +0.084). eval loss 0.752 / 0.603 / 0.615. Reading: a smaller margin trades content dependence for copy recovery roughly proportionally; it does not separate the two. Leftover checkpoint-1263 removed (trainer_state kept).

### M3b full results (363145, COMPLETED 2026-11-12) — best ranking of the line but **latent unused**
Ranking HR@1/3/5/10 0.0380/0.0649/0.0827/0.1080, NDCG@3/5/10 0.0535/0.0608/0.0690 (vs A +0.0031 / +0.0024 / +0.0028 / +0.0022 at HR@5 / NDCG@5 / HR@10 / NDCG@10, all n.s.; copy 0.516, non-copy 0.0447). Interventions raw − mask / mean / shuffle HR@10 +0.0013 / +0.0006 / +0.0010 (all n.s.) → **fails the content-dependence criterion**. eval loss 0.815 / 0.617 / 0.603.
Reading: the dependence the hinge builds in epoch 1 is undone within two epochs once the hinge is off, even with the motive InfoNCE on (the motive term shapes r but does not make the decoder read it). With M3a, m = 0.05 and M3b: **every variant that recovered the copy rows did so by (partly or fully) dropping latent use** — copy recovery and content dependence have moved together so far. MW-3b (363603, no motive, same schedule) will almost surely be unused too. Remaining real tests of decoupling them: M4 (token-level), M3a-n (relative weight), M6 (one-sided; epoch-1 warning). Leftover checkpoint-1263 removed.

**363654 (control eval-only rerun) failed later, at latent-distribution.** Its ranking rerun matched (HR@10 0.1098) and extraction completed: R3_ori + rank GRPO r has only **12 distinct vectors** over 53,898 train rows, constant as expected. The template-frequency plot assumed ≥ 20 templates. Fix ed0d0635: plot min(20, #templates), and the sbatch skips an existing extraction. The control's post-processing is resubmitted with code_ed0d0635. The CDs fulls 363660 / 363661 are cancelled and resubmitted with ed0d0635 (smoke 363659 unaffected).
Resubmitted on code_ed0d0635: control post-processing **363709**; CDs fulls **363710** (margin + WIDER) / **363711** (R3_ori), afterok on smoke 363659 (old 363660/363661 cancelled while pending).
- M4 363511 preempted twice (album 3:34, analog 2:17) → moved to hadylauwresearch a40, resumes from checkpoint-842 (epoch 2). Epoch 1.4–2.0 of the resumed log: novel-token hinge 0.141 → 0.046 (gap forming, unlike M6's pinned 0.181 and M3a's 0.06 plateau); novel-token share copy 0.04–0.14 vs non-copy 0.60; eval loss epoch 2 0.678. Early sign that M4 keeps the latent in use.

### M3a-n ranking (363638; training + ranking done before preemption 1; interventions rerunning)
HR@1/3/5/10 0.0331/0.0589/0.0693/0.0960, NDCG@3/5/10 0.0480/0.0523/0.0609; copy 0.472, non-copy 0.0377. vs plain A: HR@5 −0.0102, NDCG@5 −0.0061, HR@10 −0.0092, NDCG@10 −0.0059 — **all significantly worse**; vs MWC-M HR@10 −0.0096 sig, non-copy −0.0089 sig.
Training: the normalised weight behaved as designed (mean ≈ 0.94–1.0 throughout, copy 0.60–0.63 vs non-copy 1.03–1.07, EMA of u 0.89 → 0.50 → 0.59) and the hinge was learned (0.195 → 0.042), but the WIDER penalty jumped in epoch 2 (0.054 → 0.477, end 0.173; MWC-M stayed ≤ 0.04) — the latents partly collapsed toward a shared direction while the hinge pressure was concentrated on non-copy rows. Reading (tentative): shifting hinge pressure onto the hard rows (weights up to the cap 2) makes the decoder over-rely on a less diverse r exactly where the history is weak; it hurts both row types. Fails the ranking criterion regardless of the pending interventions.

### MW-3b results (363603, margin + WIDER only, hinge in epoch 1 only; COMPLETED 2026-11-12, 5h34m) — **significantly above plain R3, but latent unused**
Ranking HR@1/3/5/10 0.0398/0.0687/0.0856/0.1125, NDCG@3/5/10 0.0562/0.0632/0.0718; copy 0.515, non-copy 0.0501. vs plain A: **HR@5 +0.0061 [+0.0010, +0.0113], NDCG@5 +0.0047 [+0.0014, +0.0082], HR@10 +0.0073 [+0.0016, +0.0128], NDCG@10 +0.0051 [+0.0019, +0.0082] — all significant**; non-copy +0.0081 sig, copy +0.0022 n.s. vs MWC-M: HR@10 +0.0068, NDCG@10 +0.0048, NDCG@5 +0.0039 sig. Interventions raw − mask / mean / shuffle HR@10 −0.0010 / −0.0016 / −0.0006 (all n.s.) → latent unused at test time.
Reading: the epoch-1 history-swap hinge works as a training-time regulariser of the decoder — after it is switched off the model ignores r but ranks better than any plain-R3 retrain we have (A 0.1052; B/C/D 0.1033/0.1043/0.1023; DivReason SFT 0.1079; WIDER-only 0.1070). Adding the motive loss (M3b: 0.1080) removes about 60% of the gain. Meets "beats R3" but not "r's content matters". Caveats: single seed; retrain spread of plain R3 on Toys is ≈ 0.102–0.108, MW-3b is above all of it; replicate (second seed and CDs) before claiming. Its resume checkpoint was already removed by the single-checkpoint code. trainer_state in results lacks eval_loss entries (the end-of-run save_to_json may not include eval logs; check).

### m = 0.1 results (363593, single-checkpoint code, COMPLETED 2026-11-12)
Ranking HR@1/3/5/10 0.0353/0.0614/0.0803/0.1031, NDCG@3/5/10 0.0504/0.0582/0.0655; copy 0.498, non-copy 0.0418. vs A all n.s. (HR@10 −0.0021); vs MWC-M non-copy −0.0048 sig. Interventions raw − mask / mean / shuffle HR@10 +0.085 / +0.078 / +0.080 (sig) — latent strongly used (≈ MWC-M).
Margin sweep (with motive): m 0.2 (MWC-M) — dependence ≈ 0.09, copy 0.486, non-copy 0.0466; m 0.1 — 0.08, copy 0.498, non-copy 0.0418; m 0.05 — 0.01–0.06, copy 0.504, non-copy 0.0442. No margin value gives both strong dependence and recovered copy rows; m 0.2 remains the best of the sweep. Single-checkpoint code left no checkpoint behind (storage fix works). Note: trainer_state.json written by the new end-of-run save lacks eval_loss entries — to check.
- 2026-11-12 report written: MW_to_MWC_report_2026-11-12.md (+ report_tables.py).
