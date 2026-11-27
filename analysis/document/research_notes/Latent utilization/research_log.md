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
| ML-S | MovieLens MW-R-H + swapped-pass CE (COPY_SWAP_W=1, row weight 1 − w_i), code 1186ca19 | 375420 | **done 2026-11-24: MEETS CRITERION** — HR@10 .1380 / NDCG@10 .0737 ≥ R3 (.1335 / .0698; seed 1 .1372 / .0720) at all cutoffs, sig at @5 vs both R3 seeds; shuffle −.0381 sig; copy_row .4026 = R3; single seed (section 2026-11-24) |
| ML-D | MovieLens MW-R-H + distillation to the no-latent reference (DISTILL_W=1), code 1186ca19 | 375422 | **done 2026-11-24: meets criterion** — HR@10 .1350 / NDCG@10 .0718 = R3 (n.s. vs seed 1; HR@5, NDCG@5 sig vs seed 0); shuffle −.0729 sig (copy −.21, non-copy −.026); = ML-S overall (n.s.), below ML-S on non-copy HR@10 |
| ML-CR | MovieLens MW-R-H + original collaborative reasoning InfoNCE (user + target heads, 0.1 / 0.1), code c3b42206 | 376544 | **done 2026-11-24: fails at @10** — HR@10 .1265 (−.0070 sig vs R3), NDCG@10 .0677 n.s., = R3 at @1–@5; shuffle −.0507; copy .3617 (not recovered); > same-motive at HR@5/@10; sig below ML-S |
| ML-M-HN | MovieLens MW-R-H + same-motive + history-matched InfoNCE negatives (MOTIVE_HARD_NEG=16), code bd018410 | 376838 (376835 failed at start: sbatch missing space) | **done 2026-11-24: negative** — HR@10 .1155 / NDCG@10 .0623 sig below R3, n.s. vs same-motive 374774; InfoNCE could not separate the history-matched negatives (cos_hard ≥ cos_cross) |
| ML-M-W3 | MovieLens MW-R-H + same-motive, InfoNCE weight 0.3, code bd018410 | 376839 (376836 failed at start) | **done 2026-11-24: negative** — HR@10 .1127 / NDCG@10 .0603 sig below R3 and below 374774; latent weakly used (shuffle −.0096); hinge never satisfied, already before the InfoNCE starts (confounded) |
| ML-P3 | Motive probe on ML-M-HN and ML-M-W3 | 376840 (376837 cancelled) | done 2026-11-24: next/history ratio .45 (HN-neg) and .39 (w0.3) vs .48 for 374774 — neither moved the latent toward the next item |
| ML-prev | MovieLens earlier MW variants: MW-R-H 369532, MW-nc 374722, same-motive 374774, MW-3b 374225, MW-HN 374776, probes 374778 / 376414 | — | done: none meets the criterion (latent-using runs .119–.125 HR@10 < R3; MW-3b / MW-HN ≈ R3 but latent unused) |

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

**363709 done (control interventions).** The ranking reproduces (HR@10 0.1098). R3_ori + rank GRPO, raw − mode (HR@10):

| mode | HR@10 | NDCG@10 |
|---|---|---|
| mask | +0.0166 [+0.0126, +0.0208] sig | +0.0171 |
| mean | +0.0006 n.s. | |
| shuffle_user | +0.0001 n.s. | |

So the control's r is a shared soft prompt: needed as a vector, no per-user content (12 distinct vectors). Toys summary of rank GRPO v2:

| arm | HR@10 | gain vs start | shuffle effect |
|---|---|---|---|
| margin + WIDER | 0.1072 | +0.0042 sig | −0.071 (content used) |
| R3_ori | 0.1098 | +0.0076 sig | 0.000 (content unused) |

The best Toys ranking comes from the arm whose r is not used. The ranking reward's gain is generic and does not require per-user content. Results in `output_dir/grpo_stage2/toys_r3ori_rankgrpo2/`. CDs both arms pending (363710 / 363711).

## MW-3b-L: MW-3b + copy-safe late hinge (2026-11-12, code 2cb3645b)

Motivation: MW-3b (363603) is significantly above plain R3 at @5/@10 but r is unused (epochs 2-3 are CE + WIDER only, so the decoder drifts back to ignoring r). MWC-M keeps r used but loses copy rows (hinge forces r to override a sufficient history).
Design (user approved): epoch 1 = fixed history-swap hinge (m 0.2) + WIDER, no motive (MOTIVE_W=0); epochs 2-3 = hinge kept but switched to a copy-safe form via --late_margin_mode (no copy flag):
- MW-3b-L4: novel-token hinge (M4 form; copy targets have almost no novel tokens).
- MW-3b-L6: adaptive one-sided margin min(0.2, 0.5*NLL(y|x,r_other)), negative under no_grad (M6 form).
Gate: shuffle/mean drop >= ~0.02 HR@10 sig; HR/NDCG@5,@10 sig above plain R3 (like MW-3b); copy HR@10 >= 0.50.
Submit helper: $CLAUDE_JOB_DIR/tmp/submit_mw3bl.sh <l4|l6> <smoke|full>; smokes requested via JobManager.
- MW-3b-L smokes 364057 (l4) / 364058 (l6) PASSED (switch verified). L4 full = 364065 (Toys_and_Games_margin_0.2_wider_0.1_marginwarm_latenovel). L6 full ON HOLD: one-sided hinge == CE (M6 log: margin_loss 0.181->0.156, hinge_active 1.0, no gap); proposed L6b (two-sided adaptive) pending user.
- 2026-11-12: one-sided (stop-gradient negative) hinge DROPPED by user: grad of [m + NLL_pos - sg(NLL_neg)]+ = CE grad, so no contrast (M6 gap never formed). M6 363594 cancel requested; MW-3b-L L6 full never submitted; L6b not pursued.

## MW-3b-lambda: late hinge at reduced weight (2026-11-12, code 000b81fe)
Dial between MW-3b (lambda 0: sig gain, latent unused) and MW (lambda 1: latent used, copy rows hurt). Epoch 1: fixed two-sided hinge m 0.2 x1 + WIDER; epochs 2-3: same hinge x lambda (--late_margin_weight); no motive. lambda 0.1 and 0.3 (user approved), full runs requested via JobManager (no smoke: same path as the L4 smoke, switch unit-tested). Gate: shuffle/mean sig drop; @5/@10 >= plain R3; copy HR@10 >= 0.50. Helper tmp/submit_mw3blam.sh.

**CDs rank GRPO mid-run.** 363710 (margin + WIDER) steps 1–500, 100-step windows: hit rate 0.229 → 0.245 → 0.260 → 0.271 → 0.275, reward −0.209 → −0.201, kl 9e-5 → 1.2e-3. Rising steadily, like on Toys.
- M3a-n 363638 DONE (interventions): raw HR@10 .0956; mask +.0635, mean +.0414, shuffle +.0469 HR@10 (all sig) -> latent USED, but ranking HR@10 .0960 / NDCG@10 .0609 sig below plain R3 -> fails criterion; closed.
363711 (R3_ori control) steps 1–500: hit rate 0.250 → 0.262 → 0.261 → 0.298 → 0.304, kl 8e-5 → 7.2e-3 (moves faster than the margin + WIDER arm, as on Toys).
- M4 363511 DONE (old code 5ee286e8): HR@5 .0736 / NDCG@5 .0521 / HR@10 .1005 / NDCG@10 .0608; vs plain R3 HR@5 -0.0059*, NDCG@5 -0.0063*, HR@10 -0.0047 (n.s.), NDCG@10 -0.0059*. copy HR@10 .482 (NOT recovered), non-copy .0413 (gain lost). Latent used: mask/mean/shuffle -0.033/-0.033/-0.032 (sig, ~1/3 of MWC-M). Why copy not spared: copy rows still have 16.5% novel target tokens (vs 60% non-copy) — these are the variant tokens (size/colour/model no.) that separate siblings, so the hinge still sharpens r exactly where sibling slots are decided. Leftover checkpoint-1263 (8.9G) deleted.
- MW-3b-L4 364065 DONE: HR@5 .0785 / NDCG@5 .0568 / HR@10 .1034 / NDCG@10 .0648; vs plain R3 all n.s. (-0.0010/-0.0016/-0.0018/-0.0019). copy HR@10 .499 (partly recovered vs MWC-M .486), non-copy .0420 (= R3; MW-3b's .0501 gain LOST). Latent used: mask/mean/shuffle -0.028/-0.032/-0.036 (sig). Reading: the late novel hinge restores dependence but erases MW-3b's non-copy gain; ~R3 with latent used (meets the weak criterion, not 'beat R3').

## 2026-11-13 — CDs rank GRPO v2 from margin + WIDER (363710)

14h20m, 1,539 steps, full pipeline OK. Group hit rate (first / mid / last 10%) 0.233 / 0.307 / 0.343; kl 1.2e-4 → 5.6e-3. checkpoint-1539 deleted (final model kept). Results `output_dir/grpo_stage2/cds_mw_rankgrpo2/`.

| run (CDs) | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank GRPO on margin + WIDER | .0536 | .0724 | .0854 | **.1069** | .0644 | .0698 | **.0766** |
| margin + WIDER (start) | .0525 | .0726 | .0851 | .1049 | .0640 | .0691 | .0754 |
| R3_ori | .0495 | .0707 | .0819 | .0996 | | | .0719 |

Bootstrap:

| comparison | HR@10 | NDCG@10 |
|---|---|---|
| − start | +0.0019 [−0.0005, +0.0044] n.s. | +0.0012 n.s. |
| − R3_ori | **+0.0080 [+0.0028, +0.0133] sig** | **+0.0049 sig** |

Interventions raw − mode HR@10, all sig: mask +0.0135, mean +0.0050, shuffle_user +0.0081 (stage 1: +0.0125 / +0.0068 / +0.0039). Content use is kept, with shuffle slightly larger. On CDs the GRPO gain over its start is small and n.s. (Toys +0.0042 sig). The R3_ori control (363711) is in post-processing.

## 2026-11-13 — CDs rank GRPO v2 R3_ori control (363711) + two-dataset summary

16h19m. Group hit rate 0.248 / 0.328 / 0.394; kl up to 2.1e-2. checkpoint-1539 deleted.

| run (CDs) | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank GRPO on R3_ori | .0546 | .0758 | .0880 | **.1095** | .0666 | .0717 | **.0786** |
| rank GRPO on margin + WIDER | .0536 | .0724 | .0854 | .1069 | .0644 | .0698 | .0766 |
| margin + WIDER | .0525 | .0726 | .0851 | .1049 | .0640 | .0691 | .0754 |
| R3_ori | .0495 | .0707 | .0819 | .0996 | | | .0719 |

Bootstrap HR@10:

| comparison | HR@10 | NDCG@10 |
|---|---|---|
| control − R3_ori | **+0.0106 [+0.0070, +0.0140] sig** | +0.0068 sig |
| control − rank-GRPO margin + WIDER | +0.0026 n.s. | |
| control − margin + WIDER | +0.0045 n.s. | |

Control interventions: mask +0.0365 sig, mean +0.0011 n.s., shuffle +0.0008 n.s. (no content use).

**Two-dataset summary of rank GRPO v2:**

| | Toys margin + WIDER arm | Toys R3_ori arm | CDs margin + WIDER arm | CDs R3_ori arm |
|---|---|---|---|---|
| HR@10 | 0.1072 | 0.1098 | 0.1069 | 0.1095 |
| gain vs own start | +0.0042 sig | +0.0076 sig | +0.0019 n.s. | +0.0106 sig |
| r content used (shuffle) | −0.071 | 0 | −0.008 | 0 |

On both datasets the ranking reward gains more on R3_ori, whose r is a shared soft prompt, and ends numerically (n.s.) above the margin + WIDER arm. The ranking-reward gain is a generic ranking-objective gain through one shared vector. It does not use, and is not helped by, per-user r content. For the latent-utilization goal: margin + WIDER (+ rank GRPO) gives a used r at ≥ R3-level ranking on both datasets, but the best ranking does not need r's content.

## Rule-based no-hinge-on-copy-rows (2026-11-13, code d0b10de8, user decision)
--margin_mode skipcopy: hinge weight 0 on copy rows (copy_row: same-day, same brand, title Jaccard >= 0.3 vs a history item; 17.8% of train), averaged over non-copy rows. Runs: MWC-M-nc (MWC-M settings) and MW-nc (MOTIVE_W=0, hinge all epochs). Helper tmp/submit_nc.sh <mwcm|mw> <smoke|full>; smoke requested via JobManager.

## MWC-M7: Lagrangian latent-dependence constraint (2026-11-13, code a89b8680, user approved)
Motivation: no principled fix balanced copy rows and latent use; skipcopy (d0b10de8, smoke 364852) is only the ceiling. Diagnostics: (1) on the 83 copy rows R3 hits but MWC-M misses, MWC-M fills top-10 with latent-pushed items (target-brand share 0.73 -> 0.59; e.g. LEGO Deep Sea siblings replaced by Star Wars / Nerf / Exploding Kittens): over-trust of r. (2) M3a row weight does discriminate copy rows (copy 0.22/0.14 vs non-copy 0.46/0.37 in epochs 1-2); M3a failed on scale (decay), M3a-n on forced scale.
Design: --margin_mode lagrange: hinge = mu * mean(w_i * hinge_i), w_i = sg(1-exp(-NLL(y|x,r_other))); constraint E[NLL(y|x,r_other) - NLL(y|x,r_own)] >= delta via dual ascent mu <- clip(mu + 0.01*(delta - EMA_0.99(gap)), 0, 5) per micro-batch; mu fp32 persistent buffer. MWC-M settings otherwise. delta 0.05 / 0.10. Logs lagrange_mu, swap_gap(_copy/_noncopy). Helper tmp/submit_m7.sh; smoke (0.10) requested.
- skipcopy smoke 364852 PASSED (hinge_weight_copy 0, noncopy 1). Full: 364875 MWC-M-nc (..._motive_skipcopy), 364876 MW-nc (..._skipcopy). M7 smoke 364860 pending.
- M7 smoke 364860 PASSED (mu 1.0->1.13 while gap < delta; eval loads). Full: 364879 (delta 0.05, ..._motive_lagrange0.05), 364880 (delta 0.10, ..._motive_lagrange0.10).

## 2026-11-13 — MW-3b vs margin + WIDER / R3_ori (user question; run by the margin_wider_collab session)

MW-3b (363603; hinge only in epoch 1, then CE + WIDER) eval_raw.json bootstrapped directly (HR@10 / NDCG@10):

| comparison | HR@10 | NDCG@10 |
|---|---|---|
| − margin + WIDER 356322 | +0.0095 [+0.0042, +0.0150] sig | +0.0050 sig |
| − R3_ori | +0.0102 sig | +0.0059 sig |
| − plain A | +0.0073 sig | +0.0051 sig |
| − rank GRPO margin + WIDER (0.1072) | +0.0053 n.s. | +0.0032 sig |
| − rank GRPO R3_ori (0.1098) | +0.0027 n.s. | +0.0024 n.s. |

Copy split (my 836-row definition):

| rows | MW-3b | margin + WIDER | R3_ori | MW-3b − margin + WIDER | MW-3b − R3_ori |
|---|---:|---:|---:|---|---|
| copy | 0.538 | 0.507 | 0.534 | +0.031 [+0.005, +0.055] sig | +0.005 n.s. |
| non-copy | 0.0522 | 0.0457 | 0.0412 | +0.0064 [+0.0017, +0.0110] sig | +0.0110 [+0.0058, +0.0164] sig |

But its interventions are null (mask / mean / shuffle −0.001 / −0.002 / −0.001 n.s.): r is unused at test time.

**Reading:** a large ranking improvement on margin + WIDER (both copy and non-copy rows, the latter sig vs R3_ori), but it gives up the content dependence, which is the other half of the criterion. Best Toys ranking in the line (single seed, no GRPO). The epoch-1 hinge acts as a training regulariser. The follow-ups that restore a late hinge (MW-3b-L4, M4, λ variants) bring dependence back but lose the gain so far.
- M7 d0.05 364879: training+ranking DONE before 1st preemption (interventions to rerun). HR@5 .0807 / NDCG@5 .0599 / HR@10 .1077 / NDCG@10 .0686; vs R3 +0.0012/+0.0014/+0.0025/+0.0019 (all n.s., all positive; > MWC-M). copy .496, non-copy .0475. Dual: mu 1.68->1.84 (ep0-0.5) then decays to ~0 by ep1.75; train swap gap stays ~0.05-0.06 (= delta) with mu~0 -> dependence self-sustaining after epoch 1 (motive loss?); copy gap < non-copy gap mostly; weights w_copy 0.16-0.57 vs w_non 0.39-0.83. Latent use pending interventions.
- MWC-M-nc 364875 DONE (skip-copy ceiling): HR@5 .0787 / NDCG@5 .0585 / HR@10 .1043 / NDCG@10 .0669; vs R3 all n.s. (-0.0009/+0.0001/-0.0009/+0.0002). copy .497 (only +0.011 vs MWC-M .486; R3 .513), non-copy .0434 (gain mostly lost vs MWC-M .0466). Latent used: mask/mean/shuffle -0.034/-0.054/-0.063 sig. => removing the hinge on copy rows does NOT restore copy rows: the shared decoder still over-trusts r at test time (DRM condition met). M7 d0.05 (copy .496, non-copy .0475) matches/beats this rule ceiling adaptively.
- M7 d0.10 364880 DONE: HR@5 .0815 / NDCG@5 .0604 / HR@10 .1070 / NDCG@10 .0686; vs R3 +0.0019/+0.0020/+0.0018/+0.0019 (all n.s., all positive). copy .502, non-copy .0458. Latent WEAKLY used: mask/mean/shuffle -0.005/-0.008/-0.009 HR@10 (sig but ~1/10 of MWC-M). Dual: mu spiked 2.5->3.8 in ep0-0.25 (gap jumped to .23), then ~0.07-0.13 till ep1.5 and ~0 after; TRAIN swap gap held .10-.12 (>= delta) with mu~0, but TEST dependence small -> train-set gap overstates test-time dependence (gap_copy > gap_noncopy on train). M7 effectively = early hinge burst then off (MW-3b-like).
- M7 d0.05 364879 interventions DONE: mask/mean/shuffle -0.017/-0.010/-0.015 HR@10 (sig, weak; ~1/6 of MWC-M). Same failure as d0.10: train-set gap constraint met with mu~0, test dependence weak.
- MW-nc 364876 DONE (skip-copy rule, no motive): HR@5 .0813 / NDCG@5 .0602 / HR@10 .1109 / NDCG@10 .0697; vs R3 +0.0018/+0.0017/+0.0056* (CI [+0.0001,+0.0111])/+0.0030 (CI [-0.0003,+0.0062]). copy .515 (FULLY recovered = R3 .513), non-copy .0482. Latent USED: mask/mean/shuffle -0.029/-0.016/-0.037 sig. => FIRST model sig above R3 (HR@10) with latent used -- but rule-based (ceiling). Contrast MWC-M-nc (with motive) copy .497: the motive InfoNCE, not the hinge on copy rows, makes the decoder over-trust r on copy rows when the rule is used. Adaptive target: approximate this ceiling on the MW (no motive) base.

## MW-R: reference-model-weighted hinge (2026-11-14, user approved; step 1 code d15f6b02)
Motivation: -nc references (dev only) show per-row allocation on the MW base (no motive) is what's needed: MW-nc copy .515, HR@10 .1109 sig, latent used. Adaptive replacement for the rule: weight each row's hinge by how much the history LACKS about the target, measured by a frozen reference LM never trained on these rows (no memorisation confound; cf. RHO-LOSS, Rho-1).
Step 1 (diagnostic, no training): src/latent/reference_score.py scores NLL_ref(y|x) with Qwen/Qwen2.5-1.5B on the R3 prompt minus <|Thought|>, train/valid/test -> output_dir/margin_wider_collab/reference_nll/Toys_and_Games/*.pt. Go: AUROC(-NLL_ref -> copy rule) >= 0.80 (rule used only for this check).
Step 2 (if go): MW base, hinge sum(w_i hinge_i)/sum(w_i), w = rank percentile of NLL_ref (soft) and sigmoid((NLL_ref - median)/tau) (sharp). Target: approach MW-nc without the rule.
- MW-R step 1 365378 DONE: AUROC(-NLL_ref -> copy) train .914 / valid .900 / test .892 (gate .80 PASSED); copy NLL_mean 2.97 vs non-copy 5.22 (train); lowest 20% NLL captures 73% of copy rows. Weights: rank (copy .159 vs non .574, ratio .28) and sigmoid((NLL-median)/0.5) (copy .101 vs non .586, ratio .17). Step 2 code efc11712 (--margin_mode refweight, MW base MOTIVE_W=0); smoke (rank) requested; helper tmp/submit_mwr.sh.
- MW-R smoke 365384 PASSED (copy hinge weight 0.04 vs non-copy 0.56 per batch). Full: 365389 rank (..._refweight_rank), 365390 sigmoid (..._refweight_sigmoid).
- 2026-11-14 scratch cleanup (user OK: remove failed-result ckpts): deleted final models of G3 cfg3, MWC-I/I-B/I-C, M3a-n, M4, M2, M3a, M3b, MWC-M m0.1, m0.05, plus smokes smoke_m7_0.10/mw3a/mw3b/mwr_rank/nc_mwcm and empty MW-3a dir (~50 GB). Result JSONs kept in output_dir/margin_wider_collab. Kept models: MW, MWC-G, MWC-M, MW-3b, L4, M7 0.05/0.10, MWC-M-nc, MW-nc, running MW-R.

## MW-R-T: MW-R with a TRAINED reference (2026-11-14, user request; code 4e9c30f0)
Reference = MW-3b (363603) final model: in-domain, fine-tuned on these Toys rows, best ranker, latent content unused (= prediction without latent). Score NLL_ref(y|x, r_MW3b) per row (reference_score --latent_model), out reference_nll/Toys_and_Games_mw3b/. Diagnostic first: compare AUROC vs base-model reference (.914/.900/.892); expected risk = memorisation compresses train-row NLL (train AUROC < valid/test). If separation holds (train AUROC >= .80), run MW-R-T rank (MW base, same as MW-R).
- MW-R-H (user clarification 11-14): trained reference must be HISTORY-ONLY (no latent). No surviving Toys no-latent checkpoint (344012 weights purged; only eval json). Code 8a62b675: scripts/no_latent_reference.sbatch retrains R3-without-latent (344012 recipe, plain_sft_train synced to 24c71a5 + save_total_limit=1; prompt verified identical to R3 prompt minus <|Thought|> on 5 rows) -> scratch Toys_and_Games_R3_no_latent_ref, eval -> margin_wider_collab/no_latent_ref/, scores -> reference_nll/Toys_and_Games_nolatent/. Then gate (train AUROC >= .80) -> MW-R-H rank run. 365403 (MW-3b latent reference) kept as extra comparison.
- 365403 MW-3b (trained, latent-unused) reference scores: AUROC train .878 / valid .831 / test .795 (base-model ref .914/.900/.892); train NLL compressed (copy .11 vs non .33, valid .30 vs .73) = memorisation; lowest-20% captures 66% of copy rows (base 73%). Not used for training (user wants history-only reference) -- comparison only. 365404 (history-only ref) running on sodag.
- 365404 DONE: history-only reference (no-latent SFT retrain) HR@1 .0377 / HR@5 .0785 / HR@10 .1039 / NDCG@5 .0585 / NDCG@10 .0667 (= 344012's .1045). Copy AUROC train .888 / valid .829 / test .785 (base ref .914/.900/.892; MW-3b ref .878/.831/.795): trained references separate less and generalise worse (memorisation). Gate passed on train. Weights: rank copy .181 vs non .569 (ratio .32); sigmoid tau .5 useless here (NLL scale .1-.5; ratio .72) -> only MW-R-H rank submitted (helper tmp/submit_mwrh.sh).
- MW-R-H rank full = 365504 (..._refweight_nolatent_rank), pending on QOS cap.
- Swap-gap analysis (user request 11-14, code 9b1ebf33): per-row gap NLL(y|x,r_other)-NLL(y|x,r_own) (training-identical per-token NLL; partner = fixed random derangement), copy vs non-copy, test (all) + train sample 5000, for WIDER-only, MW, MWC-M, MW-3b, MW-nc, MWC-M-nc, M7 .05/.10, MW-3b-L4 -> output_dir/margin_wider_collab/swap_gap/Toys_and_Games/ (npz per row + summary.jsonl). Helper tmp/submit_swapgap.sh.

### 2026-11-14 — MW-R results (refweight hinge, base-LM reference, code efc11712)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| plain R3 | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .513 | .0420 |
| MW-R sigmoid 365390 | .0371 | .0629 | .0791 | .1036 | .0522 | .0588 | .0666 | .493 | .0432 |
| MW-R rank 365389 (ranking only; interventions running) | .0402 | .0660 | .0804 | .1046 | .0552 | .0611 | .0689 | .501 | .0432 |

- Bootstrap vs R3: all n.s. (sigmoid ΔHR@10 −.0016, ΔNDCG@10 −.0001; rank ΔHR@10 −.0006, ΔNDCG@10 +.0022 [−.0011,+.0054]).
- Copy Δ vs R3: sigmoid −.020 [−.044,+.004], rank −.012 [−.039,+.013] (MW −.034 sig) → partial copy recovery; the non-copy gain is lost (+.0012 n.s. vs MW-nc +.0062 sig).
- Sigmoid interventions (raw − x, HR@10): mask +.024*, mean +.035*, shuffle +.031* → latent used.
- Read: down-weighting the hinge on copy-like rows moves the copy rows halfway back but does not reach the MW-nc ceiling; consistent with the swap-gap diagnosis (MW copy gap +2.11 ≫ m: CE keeps the over-reliance). Band-hinge proposal pending.

### 2026-11-14 — Swap-gap analysis 365511 DONE (code 9b1ebf33; gap = NLL(y|x,r_other) − NLL(y|x,r_own), per-token mean, test all + train 5k)

Test gap mean (median) [q90], copy vs non-copy; copy HR@10 in brackets:
- wider_only +0.003 vs −0.000 (control, OK); mw3b +0.001 vs −0.000 (latent unused) [.515]
- mw +2.11 (1.24) [5.61] vs +1.40 (0.89) [3.31] [.478]
- mw_nc +1.13 (1.08) [2.15] vs +1.08 (0.95) [2.11] [.515]
- mwcm +2.27 vs +2.13; mwcm_nc +2.31 vs +2.59 [.497]; mw3b_l4 +2.75 vs +3.00 [.499]
- m7_005 +0.061 (0.005) vs +0.063; m7_010 +0.210 (0.019) vs +0.108 (median ≈ 0: few rows carry all the dependence)
- Train ≈ test for every model (mw train copy +1.93 vs non-copy +1.50).
Read: among non-motive models, copy-row loss tracks the size of the copy-row gap (mw 2.11 → .478; mw_nc 1.13 → .515). MW-nc never applies the hinge on copy rows yet still has copy gap 1.13 (CE spreads latent use), and full copy recovery is compatible with that; the copy hinge in MW adds the excess (mean 2.11, q90 5.6). MW-nc's distribution is balanced across groups and its upper tail is cut (q90 2.15 vs 5.61). → An upper bound on the gap (band hinge, U ≈ 1–2) is the supported adaptive replacement for the skip-copy rule.
- 2026-11-14 MW-R rank 365389 DONE: interventions (raw − x, HR@10) mask +.042*, mean +.046*, shuffle +.045* → latent clearly used; ranking .1046/.0689 n.s. vs R3, copy .501. Best MW-R so far; still below the MW-nc ceiling (copy .515, non-copy .0482).
- 2026-11-14 MW-R-H rank 365504 ranking DONE (interventions running): HR@1 .0383 HR@3 .0681 HR@5 .0828 HR@10 .1086 NDCG@3 .0554 NDCG@5 .0615 NDCG@10 .0698; vs R3 all four @5/@10 positive n.s. (ΔHR@10 +.0034, ΔNDCG@10 +.0031 [−.0002,+.0062], ΔNDCG@5 +.0030 [−.0003,+.0065]); copy .504 (−.009 n.s.), non-copy .0473 (+.0053 SIG). Best adaptive variant so far — history-only reference beats base-LM reference.
- 2026-11-15 MW-R-H rank 365504 DONE: interventions (raw − x, HR@10) mask +.030*, mean +.030*, shuffle +.035* (NDCG@10 +.026/+.026/+.029) → latent USED. First adaptive variant: all @5/@10 above R3 (n.s., NDCG borderline), non-copy +.0053 sig, copy −.009 n.s., latent used — nearly matches the MW-nc ceiling (.1109/.0697, shuffle −.037).

### 2026-11-15 — MW-R-H rank replication (code f3292730: new TRAIN_SEED / --train_seed; unset = old default seeding)
User choice: CDs only for the new dataset; extra seeds for MW-R-H only (vs the single plain R3 run).
- CDs: history-only reference retrain + scoring (no_latent_reference.sbatch, CATEGORY=CDs_and_Vinyl) → MW-R-H rank (afterok); compare with CDs R3_ori (attention_share/CDs_and_Vinyl_R3_ori_utilization) and CDs MW 362561.
- Toys: MW-R-H rank seeds 1 and 2 (365504 = default seeding, third seed).
- Submitted via JobManager (submit_mwrh_cds_seeds.sh): CDs ref 365885 → CDs MW-R-H rank 365886 (afterok); Toys seed1 365887, seed2 365888.

### 2026-11-15 — MW-R-H variants #1 band hinge, #2 sharper weights (code 6b7c02b6; user chose 1 and 2)
- MW-R-H-B: MW-R-H rank + band term β·mean[gap − U]+ (U = 1.5, β = 1, all rows, two-sided, no stop-gradient). Motivation: swap gap shows MW copy over-reliance (copy gap 2.11, q90 5.6) while MW-nc recovers copy rows at a bounded gap (~1.1, q90 2.15).
- MW-R-H2: w = rank^2 (copy .067 vs non-copy .391; rank^1 .181 vs .569), toward the MW-nc zero-weight limit without a rule.
- CPU checks: weights as above; band term active only for gap > U, gradient lowers neg NLL / raises pos NLL. Smoke (both options, 256 rows) → both fulls afterok. Single seed each; judge against the seed spread from 365887/365888.
- Submitted via JobManager (submit_mwrh_variants.sh): smoke 365896 → band 365897, pow2 365898 (afterok).
- 2026-11-15 CDs history-only reference 365885 DONE (resumed after 1 preemption): no-latent model HR@1/3/5/10 .0417/.0637/.0734/.0945, NDCG@3/5/10 .0545/.0585/.0653 (vs CDs R3_ori HR@10 .0989 / NDCG@10 .0717; MW .1049/.0754). Copy-detection AUROC train/valid/test .875/.858/.866 (copy rate 7.6% train, 6.8% test; Toys .888/.829/.785). CDs MW-R-H rank 365886 released.
- 2026-11-15 MW-R-H rank seed 2 (365888) DONE — FAILED SEED: HR@1/3/5/10 .0374/.0635/.0763/.0980, NDCG@3/5/10 .0525/.0577/.0648; ΔHR@10 vs R3 −.0073 SIG, ΔHR@5 −.0033, ΔNDCG@10 −.0020 (n.s.); copy .474 (−.039 sig), non-copy .0396; latent used (mask +.061*, mean +.037*, shuffle +.043*).
  Training curves (job logs, per-epoch means) seed 0 vs seed 2: margin loss ep2 .134 vs .192, ep3 .024 vs .125 (hinge satisfied late / not at all in seed 2); WIDER ep3 .076 vs .443 and grad norm 1.4 vs 5.0 → late-epoch instability (WIDER jump, as in M3a-n). Seed 0 formed the gap in epoch 2; seed 2 was still forming it in epoch 3. MW-R-H is NOT seed-robust so far; wait for seed 1.
- 2026-11-15 GPU confound: 365504 (good), 365389, 356322 ran on A40; seed 1 365887 and seed 2 365888 on L40S; pow2 365898 on L40S, band 365897 on A40. User: rerun seed 2 on hady-lab A40 (submit_mwrh_s2_a40.sh, output _seed2_a40): 366001 on album (A40, hady-priority).
- 2026-11-15 CDs MW-R-H: 365886 cancelled 38 s into an L40S run (no checkpoint) → resubmitted on hady A40 as 366023 (same output name; CDs MW 362561 was A40). Band 365897 now on hady A40 (alarm) after its preemption.
- 2026-11-15 MW-R-H rank seed 1 (365887, trained on L40S, 2 preemptions) DONE: HR@1/3/5/10 .0404/.0663/.0837/.1101, NDCG@3/5/10 .0553/.0625/.0710; vs R3 ΔHR@5 +.0042, ΔNDCG@5 +.0040 SIG, ΔHR@10 +.0049 [−.0006,+.0104], ΔNDCG@10 +.0043 SIG; copy .514 (+.001, fully recovered), non-copy +.0055 SIG; latent strongly used (mask +.054*, mean +.080*, shuffle +.080*). Training curve lost (log overwritten by the eval-only requeue).
  3 seeds (default/A40, s1/L40S, s2/L40S): HR@10 .1086/.1101/.0980 (mean .1056), NDCG@10 .0698/.0710/.0648 (mean .0685). Seed 1 on L40S is good → GPU is not sufficient to explain seed 2; A40 rerun 366001 decides.
- 2026-11-15 MW-R-H2 rank^2 (365898, L40S) ranking DONE (interventions running): HR@10 .1027, NDCG@10 .0660 (n.s. vs R3, below MW-R-H), copy .482 (−.031 SIG), non-copy +.0019. Margin loss still .140 in epoch 3 (hinge not satisfied late, like failed seed 2) → sharper weights hurt; not promoted.
- 2026-11-15 MW-R-H2 rank^2 365898 DONE: interventions mask +.040*, mean +.008*, shuffle +.022* (weaker latent use than MW-R-H: .030/.030/.035 and seed 1 .054/.080/.080). rank^2 CLOSED: worse ranking, copy −.031 sig, weaker latent use.
- 2026-11-16 MW-R-H rank seed 2 on A40 (366001) DONE: HR@1/3/5/10 .0334/.0574/.0717/.0944, NDCG@3/5/10 .0471/.0530/.0603 — SIG BELOW R3 at HR@5, NDCG@5, HR@10, NDCG@10; copy .463 (−.050 sig); latent used (mask +.050*, mean +.051*, shuffle +.047*). → GPU is NOT the cause: seed 2 fails on both L40S and A40 (the A40 run is not bit-identical to the L40S one; it had a grad-norm spike ~78 and CE 2.46 in late epoch 1).
  Failure signature (bad: seed 2 L40S/A40, rank^2; good: 365504): epoch-3 margin loss .118-.140 vs .024 (hinge never satisfied), WIDER rises in epoch 3 (.28-.44 vs .076), grad norm 4-5 vs 1.4, copy rows collapse. 4 MW-R-H rank runs: HR@10 .1086/.1101/.0980/.0944 (mean .1028), NDCG@10 .0698/.0710/.0648/.0603 (mean .0665) → MW-R-H is NOT robust (2/4 seeds good). Need a stabiliser, tested on seed 2.
- 2026-11-16 MW-R-H-B band U=1.5 (365897, A40, default seed) DONE: HR@1/3/5/10 .0389/.0643/.0791/.1079, NDCG@3/5/10 .0534/.0595/.0689; n.s. vs R3 (ΔHR@10 +.0027, ΔNDCG@10 +.0021); copy .526 (+.013 n.s.), non-copy .0430; latent used (mask +.011*, mean +.021*, shuffle +.038*).
  BAND NEVER ACTIVE (band_active 0.000 in every epoch): the in-training swap gap (per-token mean, in-batch partner) is only −.002 / .060 / .36 / .44 (ep1 / ep2 / ep2.5 / end) — U = 1.5 was calibrated on MW's post-hoc test gap (1.4-2.1) and is far above MW-R-H's training gap. MY MISCALIBRATION. So this run = a replicate of 365504 (same default seed, A40, band term zero): .1079 vs .1086 HR@10 — default seed reproduces (not bit-identical: copy .526 vs .504).
  Tally MW-R-H rank (5 runs): default .1086 / .1079, seed1 .1101, seed2 .0980 (L40S) / .0944 (A40) → the seed decides; seed 2 fails reproducibly.
- 2026-11-16 CDs MW-R-H rank (366023, A40, default seed) DONE: HR@1/3/5/10 .0555/.0801/.0906/.1121, NDCG@3/5/10 .0696/.0739/.0808 (R3_ori .0494/.0710/.0819/.0989, .0617/.0662/.0717; MW 362561 .1049/.0754).
  vs R3_ori ALL FOUR SIG: ΔHR@5 +.0088, ΔNDCG@5 +.0078, ΔHR@10 +.0132 [+.0070,+.0188], ΔNDCG@10 +.0091 [+.0053,+.0128]. vs MW ALL FOUR SIG: ΔHR@5 +.0055, ΔNDCG@5 +.0049, ΔHR@10 +.0071, ΔNDCG@10 +.0054.
  Copy (418 rows) +.048 sig, non-copy +.0106 sig (MW: +.043 sig / +.0033 n.s.). Latent used: mask +.024*, mean +.015*, shuffle +.017* (MW .013/.007/.004). Single seed (default; the seed that worked on Toys).

### 2026-11-16 — MW-R-H-L: fix for the seed-dependent failure (user: "let try to fix it")
Diagnosis: failing runs (seed 2 L40S/A40, rank^2) still have hinge loss ~.12-.14 in epoch 3 (good runs .02-.03) → the gap forms in the last epoch; the two-sided hinge then teaches the decoder not to copy without the right latent, and with the decaying lr nothing repairs the copy rows (seed 2 copy −.039/−.050 sig, non-copy ≈ 0).
Fix: hinge weight 1.0 for epochs 1-2, 0.3 in epoch 3 (MARGIN_EPOCHS=2, LATE_MARGIN_WEIGHT=0.3; refweight kept; code f3292730, no code change). Good runs barely change (their epoch-3 hinge is ~.02).
Runs (hady A40): seed 2 (must rescue) + default seed (control, must stay ≈ .108). Submitted via JobManager (submit_mwrh_late.sh): seed 2 367253, default seed 367254 (hady A40).
- 2026-11-17 MW-R-H-L seed 2 (367253, A40) DONE: HR@1/3/5/10 .0381/.0669/.0806/.1070, NDCG@3/5/10 .0550/.0606/.0692; vs R3 all four @5/@10 positive n.s. (ΔHR@10 +.0018, ΔNDCG@10 +.0024 [−.0007,+.0058]); copy .498 (−.014 n.s.), non-copy .0463 (+.0043 borderline); latent used (mask +.023*, mean +.022*, shuffle +.039*). RESCUED vs MW-R-H seed 2 (.0980 / .0944, sig below R3).
  Curve: margin loss ep1 .170 / ep2 .110 / ep3 .052 (×0.3 from epoch 2.000 confirmed), WIDER ep3 .039, grad 1.4 → stable.
  CAVEAT: epochs 1-2 use the identical recipe as failed seed-2 A40 (366001), yet the margin loss already differs (ep2 .110 vs .195) → training is NOT deterministic on the same seed/GPU; part of the rescue may be run-to-run luck. Need more seeds to measure the failure rate.
- MW-R-H-L default seed (367254) ranking DONE (interventions running): .0386/.0652/.0809/.1072, NDCG .0538/.0602/.0686, copy .510; n.s. vs R3 (all positive); slightly below MW-R-H default (.1086/.1079), n.s.
- 2026-11-17 MW-R-H-L default seed (367254) DONE: interventions mask +.042*, mean +.037*, shuffle +.042* (latent used). Curve: margin ep2 .073 / ep3 .026, WIDER ep3 .034, grad 1.0 (stable; early grad spike 62 in first quarter of epoch 1, CE 3.25 — again differs from 365504's start → nondeterminism). Both MW-R-H-L runs: n.s. above R3 at all @5/@10, latent used, stable epoch 3.
- 2026-11-18 Statistics (user): every comparison now reports paired t-test p (p_t) and paired-bootstrap 95% CI + two-sided p (p_b) over ALL test rows (Toys 6,738, CDs 6,157). Recomputed tables: conclusions unchanged (p_t and p_b agree within ~0.01 everywhere).

## 2026-11-18 — Rank GRPO v2 on MW-R-H (Toys 365504) (user request)

Start model: MW-R-H rank, default seed 365504 (margin_wider_collab session): HR@10 .1086 / NDCG@10 .0698, r used (shuffle −.035). Path `scratchDirectory/R3/output_dir/Toys_and_Games_margin_0.2_wider_0.1_refweight_nolatent_rank/latent` (read-only).

Per user: merged worktree-R3MarginWiderCollab (up to 1fa432de) into this worktree, merge commit **1a2053ff**. The only real conflict was plain_sft_train.py (took theirs: early stopping + save_total_limit=1); tracked pyc kept ours. Compile check passed, and my GRPO fixes and the one-checkpoint rule are intact.

Loading check (grpo LatentModel, CPU): missing [], unexpected = 8 training-only keys (bank_*, lagrange_state, projection.*); attention weights equal the file. MW-R-H inference = plain R3 path (confirmed by the collab session).

Frozen `output_dir/grpo_stage2_code/code_1a2053ff`; same GRPO settings as 362509. Smoke `toys_mwrh_rankgrpo_smoke` → full `toys_mwrh_rankgrpo` via JobManager. Comparisons: start 365504; rank GRPO on margin + WIDER (.1072) and on R3_ori (.1098). Submitted 2026-11-18 on hady A40: smoke 368197 → full 368198 (afterok, 72 h limit). Smoke 368197 DONE 2026-11-18 (6 min, exit 0; header correct; group hit rate mean 0.29 over 32 steps) → full 368198 waiting for an A40. Full 368198 RUNNING 2026-11-18 (hady A40, ~16 h train + ~4 h eval). Trend at step 907/1684 (2026-11-19): group hit rate 0.285 (1–100) → 0.318 (801–900), hit rank 2.60 → 2.52, KL ≈ 0.003 — healthy.

## 2026-11-18 — MW-R-H rank: 3 seeds Toys/CDs + Goodreads (submitted)

All hady A40, MW-R-H rank recipe (refweight, history-only ref, rank, tau 0.5, base Qwen2.5-1.5B), via JobManager:
- CDs seed 1 **368187**, seed 2 **368188** (code_f3292730; default seed = 366023) → CDs 3 seeds.
- Toys seed 1 rerun on A40 **368189** (`..._rank_seed1_a40`, code_f3292730; 365887 was L40S). Toys 3 seeds on A40 = 365504 (default) + 368189 (s1) + 366001 (s2, failed: HR@10 .0944).
- Goodreads (new; code **1fa432de** = Goodreads "books" category entry only): Qwen3 item-title table 368190 + history-only reference (R3-without-latent, Qwen2.5-1.5B) 368191 → MW-R-H default seed **368192** (afterok).
  Caveat: the only Goodreads R3 baseline (367929, HR@10 .0567) was trained from Qwen2.5-1.5B-**Instruct**; MW-R-H uses base Qwen2.5-1.5B (as the Toys baseline guidance_A did) → a base-model plain R3 on Goodreads is needed for a clean comparison. The no-latent reference eval (368191) gives a base-model no-latent number.

- 2026-11-18: deleted (user-approved) scratch model dirs of failed runs 365888 (rank_seed2), 366001 (rank_seed2_a40), 365898 (rank_pow2); per-row results kept in output_dir/margin_wider_collab/.

- 2026-11-18: Goodreads prep DONE: item table 368190 (7,949×1536; CDs recipe check cos .9999); no-latent reference 368191 (base Qwen2.5-1.5B, no latent) test HR@3/5/10 .0340/.0405/.0525, NDCG@3/5/10 .0289/.0316/.0354 (vs R3-Instruct 367929 HR@10 .0567 / NDCG@10 .0385); reference NLL train/valid/test written. MW-R-H 368192 released (pending A40).

- 2026-11-18: CDs MW-R-H seed 1 368187 DONE: HR@1/3/5/10 .0554/.0768/.0903/.1116, NDCG@3/5/10 .0677/.0733/.0801; all @5/@10 sig > R3_ori (HR@10 +.0127 p_t 5e-5) and > MW (HR@10 +.0067 p_t .026, NDCG@10 p_t .012; HR@5 p_t .052 / p_b .05 borderline); copy .653, non-copy .0721; latent used: mask −.031, mean −.009, shuffle −.022 (all sig). Replicates default seed 366023.

- 2026-11-18: CDs MW-R-H seed 2 368188 DONE: HR@1/3/5/10 .0499/.0775/.0911/.1121, NDCG@3/5/10 .0660/.0716/.0784; all @5/@10 sig > R3_ori; vs MW: HR@5 +.0060 (p_t .028) and HR@10 +.0071 (p_t .019) sig, NDCG@5/@10 n.s. (p_t .21/.13); copy .636, non-copy .0739; latent used (mask −.022, mean −.029, shuffle −.032, all sig).
  **CDs 3 seeds (366023/368187/368188), mean ± sd:** HR@1/3/5/10 .0536±.0032 / .0781±.0017 / .0907±.0004 / .1119±.0003; NDCG@3/5/10 .0678±.0018 / .0729±.0012 / .0797±.0012; copy .638±.014, non-copy .0736±.0013. vs R3_ori .0989/.0717: every seed sig at all @5/@10; vs MW .1049/.0754: HR@10 sig in 3/3 seeds, NDCG@10 sig in 2/3. Latent used in 3/3 (shuffle −.017/−.022/−.032). MW-R-H is seed-robust on CDs (unlike Toys).

- 2026-11-19: Toys MW-R-H seed 1 on A40 368189 DONE: HR@1/3/5/10 .0404/.0647/.0804/.1088, NDCG@3/5/10 .0544/.0608/.0700; vs R3: NDCG@10 +.0032 sig (p_t .04 / p_b .03), HR@1 sig, others n.s.; copy .510, non-copy .0465 (p_t .06); latent strongly used (mask −.052, mean −.027, shuffle −.051, all sig). Reproduces 365887 (L40S, .1101/.0710).
  **Toys 3 A40 seeds (365504 / 368189 / 366001):** HR@1/3/5/10 .0374±.0036 / .0634±.0055 / .0783±.0059 / .1039±.0083; NDCG@3/5/10 .0523±.0045 / .0584±.0047 / .0667±.0055; copy .492±.026. 2/3 seeds ≥ R3 (.1086/.1088), seed 2 fails (.0944 sig below) → mean ≈ R3 (.1052 / .0667) with large spread. All 5 runs: HR@10 .1040±.0073, NDCG@10 .0672±.0045. Seed-2 failure is seed-specific (fails on both GPU types); MW-R-H-L (late hinge 0.3) rescues it (367253 .1070).

- 2026-11-19: **Goodreads MW-R-H 368192 DONE (default seed, base Qwen2.5-1.5B, code 1fa432de): FAILS.** HR@1/3/5/10 .0188/.0290/.0353/.0454, NDCG@3/5/10 .0247/.0273/.0306. Sig below plain R3-Instruct 367929 (.0567/.0385, all cutoffs p_t ≤ 4e-5) AND below the base-model no-latent reference 368191 (.0525/.0354; HR@10 p_t 2.5e-4, NDCG@10 p_t 5.5e-5). Latent used (mask −.0125, mean −.0094, shuffle −.0061 HR@10, all sig). Training stable (margin .198→.046, WIDER .43→.16, grad 3.0 in ep 3; no Toys-seed-2 signature). Likely confound: MW-R-H trains a fixed 3 epochs and keeps the last (load_best_model_at_end=False, no eval loss), while Goodreads R3 overfits in epoch 3 (eval loss 1.152/1.019/1.165, best = epoch 2) and both baselines used early stopping. Proposed next (not submitted): MW-R-H NUM_EPOCHS=2 on Goodreads + base-model plain R3 baseline.

- 2026-11-19 **Goodreads MW-R-H failure analysis** (scripts job tmp goodreads_diag{,2,3}.py): the whole gap is on SERIES-CONTINUATION rows (target '(Series, #k)' with the same series in history; 778/9,670 test rows = 8%, but ~64% of baseline hits). Continuation HR@10: R3-Instruct .450, no-latent .429, MW-R-H .308 (−.121 vs no-latent, p_t 1.5e-18). Other 8,892 rows: MW-R-H .0224 = R3 (+.0001, p .95), +.0028 vs no-latent (p_t .10). MW-R-H puts fewer same-series titles in its top-10 (1.05 vs 1.34) and fewer next volumes at top-1 (.212 vs .262); history copying is NOT the cause (already-read volume at top-1 .041 vs .039; removing history items changes no HR). Hinge weighting worked as designed: continuation train rows rank weight .106 vs .534 (1.7% of hinge mass; Toys copy 6.4%, CDs 2.3%). Latent is used and helps even on continuation rows (raw − mask +.050, − mean +.053, − shuffle +.023, sig) → the decoder routes prediction through r learned on the hinge rows and lost history-title reading for series continuation (= MWC-M-nc 'shared decoder over-trusts r'); unlike Amazon copy rows (identical title), continuation needs composing series + next number from history. Overfit hypothesis (no epoch selection) not excluded, but the loss is confined to one row type, so it is not a general overfit signature.

- 2026-11-19: Goodreads follow-ups submitted (code_1fa432de, via JobManager, tmp/submit_goodreads_fix.sh): (1) swap-gap calibration on MW-R-H 368192 (test + 5k train; split by tmp/goodreads_gap_split.py) → sets U for (2) band hinge run (after 1); in parallel MW-R-H-L (MARGIN_EPOCHS=2, LATE 0.3, `Goodreads_..._rank_late0.3`) and MW-3b (fixed hinge epoch 1 only, `Goodreads_..._marginwarm`), hady A40.

- 2026-11-19 Goodreads swap gap (369385 crashed on empty copy group → fix f0eb6879 → 369389 DONE 3.6 min): MW-R-H 368192 history-swap gap test all +.745 / continuation +.528 / other +.764; train (5k) +.751 / +.497 / +.774 → r is used LESS on continuation rows, not more. Teacher-forced per-token NLL with own r is worse than the no-latent reference on continuation rows: test .562 vs .454 (79% of rows worse; other rows 1.467 vs 1.389, 59%), train .346 vs ref .235 → continuation rows are under-fitted already in training (not only a generalisation failure); band cap cannot target this (gap smaller there). Band run still submitted per user: U=0.5 (train median; active 51% continuation / 59% other), beta 1, `Goodreads_..._rank_band0.5`, code_1fa432de.

- 2026-11-19: **Steam** [jobs: R3 369521, emb 369522, ref 369523, MW-R-H 369524] (data/Steam, Steam_5_2016-1-2017-12: 83,064 train / 10,383 valid / 10,383 test rows, 4,683 games, 10.5k users; target never in history). Code 513bae11 (Steam category 'games' + new scripts/r3_plain.sbatch = plain R3 for any category, early stopping/best epoch, same eval incl. interventions). Planned via JobManager (tmp/submit_steam.sh; queued behind QOS 20-job limit): plain R3 base Qwen2.5-1.5B (Steam_R3_base) ‖ item emb + no-latent reference → MW-R-H rank default seed (afterok). Note recipe difference: R3 keeps best epoch, MW-R-H keeps epoch 3.

- 2026-11-19: **MovieLens** [jobs: R3 369529, emb 369530, ref 369531, MW-R-H 369532] (data/Movielens, Movielens_5_2014-1-2015-3: 81,308 train / 10,163 valid / 10,164 test rows, 2,915 movies). Code 7e603e45: user asked to replace the 'Movies' category key by 'Movielens' (noun 'movie') in all 22 maps + sbatch maps. Same 4-job chain as Steam via JobManager (tmp/submit_movielens.sh; waits for QOS slots after Steam): plain R3 base (Movielens_R3_base) ‖ item emb + no-latent ref → MW-R-H rank default seed.

- 2026-11-19: Steam STOPPED by user (dataset being updated): plain R3 369521 and MW-R-H 369524 cancelled before start. Stale artifacts from the old Steam data (not deleted; need removal/regeneration after the update): data/Steam/info/Steam_qwen3emb_1536.pt (369522), scratch Steam_R3_no_latent_ref + output_dir/margin_wider_collab/{no_latent_ref,reference_nll}/Steam* (369523: no-latent HR@10 .0212, top-1 collapse 30% on one title; co-occurrence .0308, popularity .0077; 20.7% test targets unseen in train).

- 2026-11-19 15:xx: **Steam rerun on re-preprocessed data** (82,419 / 10,302 / 10,303 rows, 4,419 games; unseen test targets 1.6% (was 20.7%); popularity HR@10 .0094, last-3 co-occurrence .0401). Old-data artifacts renamed *_old20261119 (not deleted). Chain via JobManager, code_513bae11, all hady A40: plain R3 369659, item emb 369660, no-latent ref 369661 → MW-R-H 369662.

- 2026-11-19: MovieLens no-latent reference 369531 DONE (1 preemption): HR@1/3/5/10 .0253/.0541/.0797/.1332, NDCG@3/5/10 .0419/.0524/.0695; no collapse (418 distinct top-1, max share 5.2%); baselines: popularity .0495, last-3 co-occurrence .1282 (reference slightly above); unseen test targets 0.2%. MW-R-H 369532 released.

### Result: rank GRPO on MW-R-H Toys (368198, DONE 2026-11-19)
Script/output: `analysis/results/mwrh_rankgrpo/compare.py`, `output.txt` (bootstrap 2,000 seed 42 + paired t-test, n = 6,738).

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MW-R-H + rank GRPO 368198 | .0381 | .0681 | .0827 | .1089 | .0555 | .0615 | .0700 |
| MW-R-H 365504 (start) | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 |
| R3_ori | .0377 | .0607 | .0778 | .1023 | .0510 | .0580 | .0659 |
| MW + rank GRPO 362509 | .0381 | .0637 | .0809 | .1072 | .0530 | .0601 | .0686 |
| R3_ori + rank GRPO 362510 | .0389 | .0634 | .0793 | .1098 | .0530 | .0595 | .0694 |

- vs start 365504: no change at any cutoff (HR@10 +.0003 [−.0018, +.0022], p .80). GRPO did reorder lists (only 9.5% of top-10 lists identical, top-1 same 87%, overlap 8.7/10) but net zero. Train group hit rate rose .285 → .32; KL ~.003.
- vs R3_ori: sig at HR@3, HR@10, NDCG@3/5/10 (HR@10 +.0067 [+.0012, +.0122]); HR@5 p .05. Gain is on non-copy rows (+.0080 sig); copy rows = R3_ori.
- vs R3_ori+GRPO 362510: n.s. at @5/@10 (HR@10 −.0009), HR@3 +.0047 sig.
- Interventions unchanged from start: mask −.031, mean −.030, shuffle −.033 HR@10 (all sig) → r still used.
- Reading: rank GRPO adds nothing on top of MW-R-H (unlike MW +.0042); MW-R-H alone already reaches the GRPO'd R3_ori level while using r. checkpoint-1684 deleted (final model kept).

### Audit 2026-11-19: our GRPO vs original R3 stage 2 (R3_ori git HEAD c558b44)
- grpo_trainer.py: identical noise group (G=8, N(0,1.5²) on r, first row noise-free), reward −PPL of the target, advantage = (reward − mean of noise-free rewards) / ‖·‖ over the gathered batch; ours adds only the optional KL anchor + a no-op swap guard. Training script: extra options default off; dataset, epsilon 0.2/0.28, num_iterations 1, batch 256, micro 8, 1 epoch, bf16 attention all match. noise_eval: only wandb logging added.
- **Deviation: GPU count.** R3 runs 2 GPUs (README: "strongly recommend two GPUs … related to the calculation of the advantage function"); all our stock runs (L9, 358299, 360237) used 1 GPU. With 2 GPUs the gathered batch holds 2 groups, so the baseline is the mean of TWO users' noise-free rewards and the norm runs over 16 values; on 1 GPU the baseline is the group's own noise-free reward and the norm runs over 8 (advantages √2 larger). So no run here is an exact R3 stage-2 replica. Exact 1-GPU emulation: micro-batch 16 (2 groups) + zero noise on the first row of every group.
- beta 0.01 (ours) vs 0.0 (R3): irrelevant in stock mode (KL ≡ 0). Rank-GRPO runs (362509/362510/CDs pair/368198) are a different method: beam group, no noise, fp32 attention.

### Exact R3 stage-2 replica (user request 2026-11-19: "every setting match exactly what R3 do")
Base R3_ori Toys stage 1 (read-only), lr 1e-5, 2 GPUs (torchrun), code = R3_ori git HEAD c558b44 frozen at `output_dir/r3_exact_grpo_code/code_c558b44`, unmodified except the dead `noisy_layer` import (file never committed). attention_grpo.sh settings: G 8, sample -1, 1 epoch, beta 0.00, batch 256, micro 8, seed 42, num_iterations 1, eps 0.2/0.28; code defaults warmup 200, cosine, save_steps 200 (limit 2). Eval = R3 noise_eval on 2 shards; interventions afterwards (analysis only, code_1a2053ff). Script `scripts/r3_exact_grpo.sbatch` (b63bb4be). Smoke 370692 (toys_r3ori_exact_smoke) → full 370693 (toys_r3ori_exact, afterok), hady 2×A40, submitted 2026-11-19.

- 2026-11-20: scratch quota full (500/500 G, mostly other lines): Goodreads MW-3b 369387 and band0.5 369396 FAILED at checkpoint save (Disk quota exceeded); partial dirs left. Asked JobManager to hold not-started 369532/369661/369662. Deletion of ~25 G of closed-run model dirs proposed to user (pending OK).

- 2026-11-20: more scratch-quota failures: Goodreads MW-R-H-L 369386 (finished ep 3, final save failed; intact checkpoint-1208 = end ep 2, broken checkpoint-1812 → resume possible), MovieLens plain R3 369529 (failed at first checkpoint save, broken checkpoint-636, nothing usable). Steam emb 369660 DONE (4419x1536). Steam R3 369659 still running (will fail at its first save if scratch stays full). Held: 369532, 369661, 369662. Waiting for user OK to delete ~25 G + broken partial checkpoints.

- 2026-11-20: Steam plain R3 369659 FAILED (quota, first checkpoint save). Nothing of this line is running; all blocked on scratch space + user OK to delete.

- 2026-11-20: scratch freed by user (433/500 G). Broken partial checkpoints moved aside (*_broken20261120, broken-checkpoint-1812; not deleted). Resubmitted staggered via JobManager (tmp/resubmit_after_quota.sh): wave 1 Goodreads MW-R-H-L 371445 (resume ckpt-1208), MovieLens R3 371446, Steam R3 371447, Steam ref 369661 released; wave 2 Goodreads MW-3b 371448 (afterany 371445), band0.5 371449 (afterany 369661), MovieLens MW-R-H 369532 (afterany 371446), Steam MW-R-H 369662 (afterok 369660:369661).
- 2026-11-20: smoke 370692 FAILED (album, 30 min): both ranks loaded model + data, then the first NCCL all-gather (DDP param check) hung until the 30-min watchdog — NCCL P2P transport hang, not a code bug. Fix in launch env only (commit 2c33ae98): NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, NCCL_DEBUG=INFO. Asked JobManager to cancel 370693 and resubmit smoke → full.
- Resubmitted 2026-11-20 with NCCL fix: smoke 371451 → full 371452 (afterok); 370693 cancelled.

- 2026-11-20: Steam no-latent reference rerun 369661 DONE on new data: HR@1/3/5/10 .0065/.0139/.0208/.0357, NDCG@3/5/10 .0108/.0136/.0184 (old data .0212/.0117); co-occurrence .0401, popularity .0094. MW-R-H Steam 369662 + Goodreads band0.5 371449 released.

- 2026-11-20 **Goodreads MW-R-H-L 371445 DONE (resumed from ep-2 ckpt): WORSE.** HR@1/3/5/10 .0143/.0231/.0297/.0397, NDCG@3/5/10 .0194/.0221/.0254; vs no-latent base all cutoffs sig worse (HR@10 p_t 1.1e-10); continuation HR@10 .271 (MW-R-H .308, base .429, p_t 7e-27), other .0195 = base (p .95); latent UNUSED (mask/mean/shuffle HR@10 n.s.); 1,078 distinct top-1. → Weaker late hinge drops latent use AND worsens continuation rows: continuation loss is NOT caused by r over-reliance; points to epoch-3 overfitting (both keep last epoch; Goodreads R3 best = epoch 2; weaker hinge = epoch 3 closer to plain CE fit). Epoch-2 checkpoint-1208 was removed by the single-checkpoint cleanup at the end of the resume, so the epoch-2 test needs a NUM_EPOCHS=2 run (proposed). Table script tmp/goodreads_variants.py.

- 2026-11-20: user STOPPED the Goodreads line (focus on Steam + MovieLens): band0.5 371449 (ran 2 min) and MW-3b 371448 (pending) cancelled; NUM_EPOCHS=2 test not run. Running: Steam R3 371447, Steam MW-R-H 369662, MovieLens R3 371446; MovieLens MW-R-H 369532 after 371446.

- 2026-11-20 19:35: plain R3 ranking done (interventions running): Steam_R3_base 371447 HR@1/3/5/10 .0077/.0151/.0226/.0358, NDCG@3/5/10 .0119/.0150/.0192 (no-latent ref .0357/.0184); Movielens_R3_base 371446 HR@1/3/5/10 .0249/.0556/.0806/.1335, NDCG@3/5/10 .0425/.0527/.0698 (no-latent ref .1332/.0695).

- 2026-11-20 20:45 plain R3 (stage 1) vs no-latent base (paired, all rows): Steam 371447 HR@10 .0358 vs .0357 (p .95), NDCG@10 .0192 vs .0184 (p .34), all cutoffs n.s.; MovieLens 371446 .1335 vs .1332 (p .91), .0698 vs .0695, all n.s. Interventions (shuffle pending): Steam mask −.0141 sig, mean +.0003 n.s.; MovieLens mask −.0923 sig (activation dependence), mean −.0022 (p .03). Collapse: Steam R3 447 distinct top-1, max 13.2% (base 25.2%); MovieLens 415 / 6.6%. Script tmp/newds_tables.py.

- 2026-11-20: Steam plain R3 371447 DONE: shuffle_user −.0007 (p_t .14, n.s.) → content unused (mask −.0141 sig = activation dependence only), as plain R3 elsewhere.
- Smoke 371451 DONE 2026-11-20 (NCCL over sockets, 32 steps clean); full 371452 waiting for 2×A40.

- 2026-11-20: MovieLens plain R3 371446 DONE: interventions mask −.0923 (sig), mean −.0022 (p_t .03), shuffle −.0027 (p_t .02, p_b .01) → small but significant content use (first plain-R3 run with sig shuffle; effect ~2% of HR@10). MovieLens MW-R-H 369532 started.

- 2026-11-21 **Steam MW-R-H rank 369662 DONE**: HR@1/3/5/10 .0100/.0193/.0262/.0388, NDCG@3/5/10 .0154/.0182/.0223. vs plain R3 371447: sig better at HR@1/3/5 and NDCG@3/5/10 (NDCG@10 +.0031, +16%, p_t .0016 / p_b .001; HR@5 p_t .01), HR@10 +.0030 n.s. (p_t .09); vs no-latent same pattern (HR@10 p_t .08). Collapse removed: 1,226 distinct top-1, max share 3.2% (R3 447 / 13.2%, base 385 / 25.2%). BUT latent content UNUSED: mask −.0117 sig, mean −.0009 (p .15), shuffle −.0007 (p .35) → gain = hinge/WIDER as training regulariser (de-collapse), like MW-3b on Toys; content-dependence criterion NOT met on Steam.

- 2026-11-21: deleted (user-approved) model.safetensors only (~26 GB) of: Steam_R3_no_latent_ref_old20261119, Toys band1.5, lagrange0.05, lagrange0.10, grounded, marginwarm_latenovel (MW-3b-L4), refweight_sigmoid (MW-R sigmoid), refweight_nolatent_rank_seed1 (L40S); configs/tokenizers and all results kept. Scratch 425 G after.

- 2026-11-21 **MovieLens MW-R-H rank 369532 DONE**: HR@1/3/5/10 .0193/.0492/.0701/.1188, NDCG@3/5/10 .0363/.0449/.0605 — SIG BELOW plain R3 at every cutoff (HR@10 −.0147, p_t 2.9e-7; NDCG@10 −.0093, p_t 4e-9). Latent strongly USED: mask −.0464, mean −.0159, shuffle −.0267 (10x plain R3's −.0027; all p_t ≤ 3e-13). No collapse (397 distinct top-1, max 4.7%). Training: hinge mostly satisfied (margin .131→.068) but grad norm jumps 2.7→10.7 in epoch 3 (Toys-seed-2-like). Confound on BOTH new datasets: plain R3 eval loss rises in epoch 3 (MovieLens .584/.548/.599, Steam 1.293/1.226/1.379) and R3 keeps the best epoch (2), MW-R-H keeps epoch 3. Steam MW-R-H curve: margin stays ≈.20 all 3 epochs (hinge never satisfied) and WIDER saturates at 1.0 → explains unused latent on Steam. Next proposal: MW-R-H NUM_EPOCHS=2 on MovieLens (+ Steam).

- 2026-11-21 **MovieLens MW-R-H analysis** (tmp/ml_diag.py, ml_diag2.py): (1) whole net loss on history-explained rows: ref-NLL Q1 HR@10 R3 .406 → MW-R-H .332 (−.074, p_t 6e-19, −189 hits); MW-R-H GAINS on Q2–Q4 (Q3 +.0094 p_t .009; +39 hits) — same pattern as Goodreads continuation rows; also co-occurrence-hit rows −.044, franchise/sequel rows (311) .714 → .608. Note R3 already trades Q1 (.471 no-latent → .406) for Q2/Q3; MW-R-H amplifies it. (2) Mechanism: MW-R-H re-recommends ALREADY-WATCHED movies (targets are never in history): 1.09 history items per top-10 vs R3 0.48 (64% of rows vs 35%), top-1 = a history item 14.5% vs 3.6% (Q1 rows 16.6%); the lost rows are mostly R3 rank 4–10 (67%) pushed out of the list. (3) Latent is used most exactly where history explains the target (shuffle −.133 on co-occurrence-hit rows, −.076 Q1, ~0 on Q4) → r encodes the user's history-item identity (makes the hinge easy: own r ≈ 'items this user has'), and the decoder regenerates history titles. On Amazon this helps copy rows (same-day variants); on dedup datasets (MovieLens/Goodreads) it wastes slots. 63% of MovieLens rows have tied/same-minute history timestamps (batch rating, order noisy). Epoch-3 confound still open.

- 2026-11-21: **exclude-history decoding** (commit 3ae4990f, frozen code_3ae4990f): latent_attention_eval --exclude_history prunes, per test row, the trie edges used only by that user's history titles (get_title_edges/excluded_edges in LogitProcesser.py) so already-watched titles cannot be generated; verified exact on 300 MovieLens + 300 Steam rows (reachable = all − history, 0 mismatches; 100% history titles matched). scripts/eval_exclude_history.sbatch. Submitted via JobManager: MovieLens R3 + MW-R-H eval-only → output_dir/exclude_history/Movielens/{r3,mwrh}.json.
- 2026-11-21: full 371452 pending (hady A40s full, est. 11-23); twin 373416 submitted on researchshort (a40, 2 GPUs, 48 h); JobManager guard cancels whichever starts second.

### 2026-11-21 MovieLens exclude-history decoding (code 3ae4990f) — R3 part
R3 excl 373414 DONE (history match 100%): HR@1/3/5/10 .0254/.0569/.0832/.1362, NDCG .0254/.0433/.0541/.0712.
vs R3 normal (paired, all 10,164 rows): HR@5 +.0027 (p_t 3.8e-5, p_b <5e-4), HR@10 +.0027 (p_t 1.2e-6), NDCG@5 +.0014, NDCG@10 +.0015 (both p <5e-4); @1 n.s.
MW-R-H excl 373415 running (album, A40).
MW-R-H excl 373415 DONE: HR@1/3/5/10 .0241/.0579/.0823/.1277, NDCG .0241/.0433/.0533/.0680.
vs MW-R-H normal: all cutoffs sig up (HR@5 +.0121, HR@10 +.0090, NDCG@10 +.0075; p <5e-4).
MW-R-H excl − R3 excl: @1–@5 n.s. (HR@5 −.0010 p .68, NDCG@5 −.0008 p .60); HR@10 −.0085 (p_t .004) and NDCG@10 −.0033 (p_t .04) still sig.
→ watched-movie re-recommendation explains the whole @1–@5 gap and ~40% of the HR@10 gap (−.0147 → −.0085); the remaining @10 gap is in ranks 6–10.
- 2026-11-21: full 371452 RUNNING on album since 21:05 (hady won); step 896/1684 at 1h41m, loss ~0, grad norm ~0.012, cosine lr 5.5e-6; asked JobManager to cancel twin 373416.

### 2026-11-21 MovieLens: where MW-R-H loses to R3 (history-excluded results; tmp ml_diag3/4/5.py)
- Net −86 hits @10 (484 R3-only, 398 MW-only). Gap is entirely on "easy" rows: ref-NLL Q1 HR@10 .4116 → .3534 (−.0582, p<5e-4; −148 hits), no-latent-hit rows −.0805.
  MW better on Q2 (HR@5 +.0201 sig, NDCG@10 +.0082 sig) and Q3 (HR@10 +.0126 sig); Q4 ≈ 0.
- Other cuts: same-session rows (gap to last rating <1 min, 62% of test) −.0155 sig; gap ≥1 min ≥ R3. Old targets (<2000) −.0194 sig; rare/mid-pop targets (pop Q1/Q2) sig worse.
- Q1 rows: 75% rated <1 min after the previous movie (rating bursts); 31% target in last item's top-10 train successors (4% elsewhere) → co-rating pattern.
- MW lists differ (top-10 overlap with R3 .38): fewer last-item train successors (.197 vs .220; Q1-lost .270 vs .329), more franchise/sequel titles of the history (.226 vs .202). Lost hits are spread over all R3 ranks (rank 1: 62, 2–3: 89, 4–5: 96, 6–10: 237).
- Reading: the latent carries history-item identity → decoder shifts toward content-similar items of the history and away from the collaborative burst pattern, losing Q1 and gaining on Q2–Q3.

### 2026-11-21 MW-nc on MovieLens with copy_row = reference-NLL Q1 (user naming)
- Rationale: MovieLens has no rule-based copy rows; the ref-NLL Q1 rows ("easy", 75% rating bursts) are where MW-R-H loses (−.0582 HR@10). On Toys, hard skip (MW-nc .1109, copy .515) beat soft rank weights (MW-R-H .1086 / .504) and rank^2 (.1027 / .482).
- Code c420d800: REF_WEIGHT_FN=cut, REF_CUT=0.25 → hinge weight 0 on train rows with ref-NLL percentile < 0.25 (copy rows), 1 elsewhere; loss = skipcopy's. Test: 20,327 / 81,308 train rows (25.00%), loss == skipcopy.
- Run Movielens_margin_0.2_wider_0.1_nc_refq1 (MW-R-H recipe otherwise, 3 epochs, hady A40) submitted via JobManager: job 374203.
- 2026-11-21 (user request) MovieLens MW-3b: MARGIN_MODE=fixed, MARGIN_EPOCHS=1 (hinge epoch 1 only), MOTIVE_W=0, code c420d800, run Movielens_margin_0.2_wider_0.1_marginwarm, hady A40, submitted via JobManager: job 374225.

### Result: exact R3 stage-2 GRPO on R3_ori Toys (371452, DONE 2026-11-22)
Training 1684/1684 in 3h09 (2×A40, NCCL over sockets); R3 noise_eval on 2 shards, 6,738 rows merged. Script/output `analysis/results/r3_exact_grpo/compare.py`, `output.txt`.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3 exact GRPO 371452 | .0375 | .0619 | .0779 | .1012 | .0515 | .0581 | .0656 |
| R3_ori stage 1 | .0377 | .0607 | .0778 | .1023 | .0510 | .0580 | .0659 |
| R3_ori + rank GRPO 362510 | .0389 | .0634 | .0793 | .1098 | .0530 | .0595 | .0694 |

- vs stage 1: no change at any cutoff (HR@10 −.0010 [−.0027, +.0006], p .25; all CIs span 0) → R3's own stage 2, run exactly as released (2 GPUs, lr 1e-5), is a no-op on Toys.
- vs rank GRPO 362510: sig worse at HR@10 (−.0086), NDCG@3/5/10.
- Interventions failed (experiment.py needs R3_ROOT; fixed bbe55f7e) → interventions-only rerun requested.
- Interventions-only rerun: 374331 (hady a40) / 374332 (researchshort a40), JobManager guard cancels the later one.
- 2026-11-22 MW-nc refQ1 374203 FAILED after 5h55m at the epoch-2 checkpoint save (scratch 501/500 GB). checkpoint-1272 (epoch 2, eval ce .6936): model.safetensors verified intact; optimizer.pt truncated, no trainer_state → not resumable. MW-3b 374225 held by JobManager for quota. Plan pending user: eval-only the epoch-2 model in a separate dir + fresh 3-epoch rerun.
- Interventions 374331 DONE 2026-11-22 (job exit 1 only from final cp: $results unbound under SKIP_EVAL; fixed in sbatch, JSONs copied by hand). raw HR@10 .1020. mask HR@10 −.0086 / NDCG@10 −.0136 sig; mean +.0001 n.s.; shuffle −.0007 n.s. → content of r unused (only presence matters), same as R3_ori.
- 2026-11-22 scratch freed (~424 GB, not by this session). MW-nc ep2 (checkpoint-1272 moved to Movielens_margin_0.2_wider_0.1_nc_refq1_ep2) eval-only 374720 + exclude-history 374721 RUNNING; fresh 3-epoch MW-nc rerun 374722 + exclude-history 374723 (afterok); MW-3b 374225 released.

### Correction 2026-11-22: where rank-GRPO gains come from (copy vs non-copy, Toys HR@10)
Earlier claim "GRPO signal mostly from copy rows, MW-R-H ties R3 there" was WRONG (user spotted the contradiction).
- R3_ori+GRPO: copy .534→.551 (+.018 sig), non-copy .041→.047 (+.006 sig); non-copy gives +.0053 of the +.0076 total. Gains at ranks 6–10 (HR@5 n.s. in every run).
- MW+GRPO: copy +.014 sig, non-copy +.003 sig. MW-R-H+GRPO: copy +.005 n.s., non-copy −.000.
- Supported reading: on non-copy rows GRPO brings models to ~.047–.049 and MW-R-H already starts at .050 → no headroom. Copy rows: MW-R-H ties R3_ori before (.526 vs .534) but gains only +.005 vs +.018 — unexplained; hypothesis (untested): R3_ori r is a shared constant (cos 1.0) so GRPO tunes one soft prompt coherently, per-user r in MW-R-H spreads/cancels the sparse reward.

### Test A: gradient alignment of rank GRPO (user OK 2026-11-22)
Hypothesis: R3_ori gains on copy rows because its r is shared (one soft prompt, per-row updates add up); MW-R-H's per-user r spreads/cancels updates. Diagnostic, no training: 300 copy + 300 other Toys TRAIN rows (same for all models), rank-GRPO step exactly as rank_grpo.py (G 8 beams, +1 / −0.5/log2(k+1), skip no-hit), per-row gradient on the pooling attention (count sketch 16,384) and dL/dr. Metrics: mean pairwise cosine (bootstrap CI), |Σg|/Σ|g|. Models R3_ori, MW 356322 (per-user r, still gained +.014 on copy rows — key check), MW-R-H 365504. Prediction if hypothesis holds: R3_ori ≫ MW-R-H in alignment, with MW in between/near R3_ori. Code 1e7ab332 (`src/grpo_attention_tuning/grad_coherence.py`, `scripts/grad_coherence.sbatch`), smoke toys_smoke → full toys via JobManager.
- Test A submitted 2026-11-22: smoke 374736 → full 374737 (hady a40). MW 356322 model.safetensors had been removed by the 11-22 scratch cleanup (Drive backup); JobManager restored it (rclone check identical).
- 2026-11-22 MW-nc ep2 exclude-history 374721 DONE: HR@1/3/5/10 .0238/.0520/.0751/.1143, NDCG .0238/.0399/.0493/.0620 — sig below R3 excl (HR@10 −.0218, NDCG@10 −.0092) and MW-R-H excl (HR@10 −.0134) at @3–@10.
  copy_row (Q1) HR@10 .3156 < MW-R-H .3534 < R3 .4116 although Q1 rows get NO hinge → Q1 loss is not caused by the hinge on Q1 rows; shared-decoder over-trust (as MWC-M-nc on Toys). Non-copy: HR@5 +.0081 sig vs R3, HR@10 +.0029 n.s.
  Caveat: epoch 2 of a 3-epoch LR schedule (not a converged model); final rerun 374722 pending.
- **Test A DONE 2026-11-22 (374737, 27 min).** Hit rows R3_ori 292 / MW 304 / MW-R-H 316 (copy hit rates .83/.81/.84, other .14/.20/.21). Mean pairwise cosine of per-row gradients, attention (sketch) | dL/dr with corrected bootstrap CI (`corrected_ci.txt`; script CI had a duplicate-pair bias, fixed in grad_coherence.py):
  - R3_ori: all +.111 | +.110 [+.083,+.143]; copy +.121 | +.120 [+.090,+.157]; other +.076 | +.075 [−.002,+.193]
  - MW: all +.0065 | +.019 [+.011,+.031]; copy +.007 | +.020 [+.012,+.032]
  - MW-R-H: all +.004 | +.020 [+.013,+.030]; copy +.006 | +.027 [+.018,+.039]
  - per-row |g|: R3_ori .035, MW .16, MW-R-H .12.
  - Reading: R3_ori's updates are ~5–20× more aligned than either per-user model (shared r confirmed as a difference), BUT MW and MW-R-H are equally unaligned while MW still gained +.014 on copy rows → alignment does not explain why MW-R-H didn't gain. MW vs MW-R-H copy gains (+.014 [+.005,+.025] vs +.005 [−.004,+.013]) overlap → the "shortfall" may be noise. Remaining supported reason: no headroom on non-copy rows. Test B (shared δ) weakly motivated.

### 2026-11-22 Cross-dataset diagnosis: why MW-R-H loses on MovieLens (tmp/xds.py; R3 vs MW-R-H default seed, normal decoding)

- Per ref-NLL quartile, MW−R3 HR@10: Toys Q1 −.0178* / Q2 +.0190* / Q3 +.0107*; CDs Q1 +.0071 n.s. / Q2 +.0266* / Q3 +.0149*; Steam Q1 −.0101 (p .09) / Q2 +.0151* / Q3 +.0050*; MovieLens Q1 −.0744* / Q2 +.0039 n.s. / Q3 +.0094*. Same pattern everywhere (gain Q2–Q3, cost Q1); MovieLens differs in SIZE: Q1 cost 4–7x larger, Q2 gain missing.
- Row types (lexical = target title Jaccard ≥ .5 with a history title; transition-only = target in last item's top-10 train successors): share of R3 hits lexical / transition-only / neither: Toys .56/.22/.22, CDs .44/.38/.17, Steam .45/.10/.45, MovieLens .04/.48/.48. MW−R3 HR@10 on transition-only rows: Toys +.034 n.s., CDs +.095*, Steam +.074*, MovieLens −.059*; on "neither": Toys +.0051*, CDs +.0025, Steam +.0017, MovieLens −.0089*.
- Lists: history items per top-10 R3→MW: Toys 1.01→1.02, CDs .65→.83, Steam .58→.67, MovieLens .48→1.09; last-item successors per top-10: Toys .85→.89, CDs .76→1.20, Steam .34→.47, MovieLens 2.26→2.04 (only dataset where it falls).
- Dataset: rows/item Toys 8.6, CDs 9.2, Steam 18.7, MovieLens 27.9; top-10% items hold .35/.37/.36/.55 of train targets; last-item-successor baseline HR@10 .089/.120/.023/.104; target-in-history .015/.031/0/0.
- Training: Steam hinge NEVER satisfied (margin_loss ≈ .20 = margin all 3 epochs, wider_loss ≈ 1.0) → explains unused latent there; Toys/CDs/MovieLens margin_loss falls to .02/.06/.055 in epoch 3.
- Reading: MovieLens is a dense item-transition dataset with no title-readable rows; R3's decoder already lists 2.26 successors per top-10; the latent (history identity) displaces that with history items instead of adding successors as on CDs/Steam.

### 2026-11-22 MovieLens: make the latent encode "next is B" (user: latent should be the reasoning result, not "watched A"; try both)

- Diagnosis (user discussion): the swap hinge is satisfied by a history summary ("watched A"), so the latent does not need to carry the conclusion ("next is B"); on MovieLens the decoder already reads the history, so the latent only re-surfaces history items.
- Code 013bd0ac (frozen output_dir/margin_wider_collab_code/code_013bd0ac):
  - **MW-HN** `HINGE_NEG=history`: swapped latent = bank latent (detached) of another user's train row with the same last history item and a different next item (top-32 by history overlap; in-batch fallback until the bank row is filled). MovieLens: 99.98% rows have candidates, mean 26.6, mean overlap .30 of own history. Tests: tmp/test_hardneg.py (index constraints; tiny-model forward; batch mode unchanged).
  - motive_probe: + history_vs_next (cos same-last/different-next vs same-next/different-last vs random).
- Submitted via JobManager (tmp/submit_movielens_next_is_b.sh, hady A40):
  1. MW-R-H + same-motive InfoNCE (MOTIVE_W=0.1) → Movielens_margin_0.2_wider_0.1_refweight_nolatent_rank_motive (+ exclude-history TAG=mwrh_motive)
  2. MW-HN (MOTIVE_W=0, HINGE_NEG=history) → ..._rank_hardneg (+ exclude-history TAG=mwrh_hardneg)
  3. motive probe on R3 base and MW-R-H (tags movielens_r3_base, movielens_mwrh)
- Job ids: motive 374774 (exhist 374775), MW-HN 374776 (exhist 374777), probe 374778.
- 2026-11-22 (user): also CDs MW-R-H + same-motive InfoNCE (MOTIVE_W=0.1, code 013bd0ac, 366023 recipe otherwise) → CDs_and_Vinyl_margin_0.2_wider_0.1_refweight_nolatent_rank_motive, submitted via JobManager (tmp/submit_cds_mwrh_motive.sh): job 374791.

## 2026-11-22 — Why rank GRPO does not help MW-R-H; new arms A / C / A+C (user request)

**Prediction re-analysis (Toys test 6,738; MW-R-H 365504 vs +GRPO 368198; script `$CLAUDE_JOB_DIR/tmp/pa.py` logic, numbers here).**
- GRPO barely moves the model: same top-1 on 87% of rows; hit@10 sets 710 shared, 22 only before / 24 only after.
- Headroom is ordering inside the list: MW-R-H hits 732, of which 258 at rank 1 (35%); non-copy hits 22% at rank 1. Perfect reorder of the top-10 → HR@1 .109 (now .038).
- Reward starvation on train rows (Test A records, G=8): non-copy rows have a hit 21%, hits at rank ≥2 only 42/300 (14%); train non-copy hit rate 21% vs test 4.9% (memorised targets).
- Top-10 waste: 10.2% of slots are history items (target in history 1.5%); 44% share top-1 brand; 3.6 distinct brands; target brand in top-10 only 32%.
- Popularity bias: top-1 mean train freq 26.9 vs target 16.4; HR@10 .157 (freq>10), .09 (3–10), .03 (1–2), 0 (unseen, 5% of rows).
- R3_ori and MW-R-H top-10 lists share nothing; hits both 519 / only MW-R-H 213 / only R3 170 (union .134 vs .109).

**New arms (code f3409399, frozen `output_dir/grpo_stage2_code/code_f3409399`; user chose A+C, 3 arms, no diagnostic, attention-only):**
- A: G=16 beams; if no beam hits, the target (label ids) replaces the lowest-ranked beam, reward +1 → every row has a signal.
- C: extra −0.5 for a beam that is a history item, −0.25 for a beam redundant with a higher-ranked beam (same first word and title Jaccard ≥ 0.5); no-hit groups: only these penalties (no rank penalty).
- A+C: both. All: rank reward −0.5/log2(k+1), KL anchor β 0.04, fp32 attention, lr 3e-5 (was 1e-5), micro-batch 16, start model MW-R-H 365504. Tags toys_mwrh_rankA / rankC / rankAC, smoke toys_mwrh_rankAC_smoke. Logged metrics: rank/label_ids_match, injected_frac, hist_slot_frac, dup_slot_frac, distinct_brands.
- Success: ≥ R3 at @5/@10 (paired bootstrap + t-test), interventions mask/mean/shuffle still significantly hurt; compare vs 365504 and 368198. Submitted via JobManager (ids pending).
- Submitted 2026-11-22 via JobManager (hady A40, 72 h, requeue; all PENDING): smoke 374797 (toys_mwrh_rankAC_smoke) → afterok: A 374798 (toys_mwrh_rankA), C 374799 (toys_mwrh_rankC), A+C 374800 (toys_mwrh_rankAC). Full runs do not start if the smoke fails. Smoke checks: rank/label_ids_match ≈ 1, injected_frac, hist_slot_frac.

### 2026-11-22 MW-nc ep2 (374720) normal ranking + interventions DONE — CORRECTS the "over-trust" reading of the ep2 model

- Normal decoding HR@1/3/5/10 .0226/.0474/.0695/.1102, NDCG .0226/.0368/.0458/.0590. vs R3: HR@10 −.0233, NDCG@10 −.0108 (sig @3–@10). vs MW-R-H: HR@10 −.0086 sig (p .003), HR@1 +.0033 sig, rest n.s.
- Latent NOT used at all: mask − raw HR@10 +.0020 n.s.; mean +.0030 (p .04/.03); shuffle +.0027 (p .06/.05), NDCG@10 +.0012 (p .05/.04). Even masking is harmless.
- Training log 374203: hinge never satisfied (margin_loss ≈ .20 through epoch 2.0–2.1) → like Steam. So this model's Q1 loss cannot be decoder over-trust in r; it is a worse decoder (epoch-2 state, hinge unsatisfied). The ep2 model is NOT a valid test of MW-nc; wait for rerun 374722 (its margin_loss already falls in epoch 1.0–1.5: .125) — runs are nondeterministic.
- MW-3b 374225: hinge satisfied late in epoch 1 (.173 avg in 0.5–1.0), then off.

### 2026-11-22 Motive probe MovieLens 374778 DONE (first 20,000 train rows; cos of normalised r; other users only)

| model | same last item, diff next | same next item, diff last | random | gap history | gap next | within-user similar − dissimilar | global mean cos |
|---|---|---|---|---|---|---|---|
| R3 base | .9810 | .7550 | .7193 | +.2617 | +.0357 | +.0074 | .7205 |
| MW-R-H | .1069 | .0703 | .0456 | +.0613 | +.0247 | +.1067 | .0540 |

- Both latents follow the history side more than the next item (R3 7x, MW-R-H 2.5x); MW-R-H same-next retrieval P@10 .0071 (chance .0018; R3 .0284). MW-R-H's largest structure is same-user (+.107). Supports "r = watched A / who the user is, weakly next is B". Baselines for the motive (374774) and MW-HN (374776) models: gap_next should rise above gap_history.

### 2026-11-22 MovieLens MW-R-H: latent interventions split by copy_row (tmp/ml_copy_interv.py) — the copy-row loss is in the DECODER, not in r

| rows | R3 | no-latent | MW-R-H raw | mean r | shuffle r | mask |
|---|---|---|---|---|---|---|
| copy_row HR@10 (n 2,541) | .4061 | .4707 | .3318 | .2857 (−.0460*) | .2562 (−.0756*) | .1909 (−.1409*) |
| non-copy HR@10 (n 7,623) | .0426 | .0207 | .0478 | .0418 (−.0059*) | .0374 (−.0104*) | .0328 (−.0150*) |

- On copy rows the own latent HELPS MW-R-H (removing it costs .046–.141); without it the decoder is far below R3/no-latent. So the decoder lost its history-only reading of copy rows and now needs r as a (lossy) substitute. Withdraws "latent misleads / over-trust" as the mechanism; test-time gating is dead (oracle gate mean-on-copy: HR@10 .1072, sig below R3).
- Note copy_row is defined by the no-latent reference, hence no-latent .4707 > R3 .4061 there (and .0207 < .0426 on non-copy).
- Design candidates discussed with user: (a) swapped-latent CE on copy rows (weight 1 − w_i), (b) KL distillation from the frozen no-latent reference on copy rows.

### 2026-11-22 MW-R-H-S / MW-R-H-D on MovieLens (user: try both) — keep the decoder's history-only reading on copy rows

- Code 1186ca19 (frozen code_1186ca19); refweight only; row weight (1 − w_i)^COPY_POWER (default 1; rank weights → mean .5, copy rows .75–1):
  - **MW-R-H-S** `COPY_SWAP_W=1`: + mean((1−w_i)·NLL(y | x, r_swapped)) — CE on the swapped-latent pass (already computed for the hinge; no extra cost).
  - **MW-R-H-D** `DISTILL_W=1 DISTILL_MODEL=<no-latent ref>`: + mean((1−w_i)·KL(ref ‖ model)) on target tokens; teacher = frozen Movielens_R3_no_latent_ref/model (thought slot removed, token-aligned, vocab slice 151,666), outside the module tree.
- Tests tmp/test_copyloss.py: values = manual per-row computation (padded row, varying thought positions, smaller teacher vocab), student-as-teacher KL 0, teacher frozen / not saved, off by default.
- Note: with rank weights the new terms also act on Q2–Q3 rows (weight .5–.75), where the hinge gives its gain — possible dilution; COPY_POWER=2 is the follow-up if non-copy gain drops.
- Success check: copy_row HR@10 → ~.41 with mean-r ≈ raw on copy rows; non-copy gain kept and shuffle still hurts there.
- Submitted via JobManager (tmp/submit_movielens_copyloss.sh): ..._rank_swapce, ..._rank_distill (+ exclude-history TAG=mwrh_swapce / mwrh_distill): swapce 375420 (exhist 375421), distill 375422 (exhist 375423).
- Smoke 374797 DONE 2026-11-22 (8m50s, exit 0): label_ids_match 1.0, injected_frac 0.25–1.0, hist_slot_frac 0.05–0.125, KL ~1e-4, clip_ratio 0. A 374798 / C 374799 / AC 374800 now pending (Priority).

### 2026-11-22 DONE: MovieLens MW-3b 374225 and CDs MW-R-H + same-motive 374791 (tmp/done_1122.py)

- **MovieLens MW-3b**: HR@1/3/5/10 .0269/.0617/.0830/.1263, NDCG .0269/.0469/.0557/.0696. vs R3: HR@3 +.0061* (.0045/.008), HR@5 +.0025 n.s., HR@10 −.0072* (.014/.01), NDCG@3 +.0045*, NDCG@5 +.0029 (.07), NDCG@10 −.0002 n.s. vs MW-R-H: all cutoffs sig better (HR@10 +.0076, NDCG@10 +.0091). Latent UNUSED (mask +.0019 [−.0006,+.0044], mean +.0013, shuffle +.0011, n.s.). History items per top-10 .39 (R3 .48). Hinge satisfied by end of epoch 1, then off.
  - copy_row: HR@10 .3676 vs R3 .4061 (−.0386*), @1–@5 n.s.; non-copy: HR@3 +.0062*, HR@5 +.0071*, HR@10 +.0033 n.s., NDCG@10 +.0033*.
  - → the copy(−@10)/non-copy(+) trade appears WITHOUT latent use: partly a regulariser/redistribution effect, not only latent content.
- **CDs MW-R-H + motive**: HR .0547/.0801/.0947/.1145, NDCG .0547/.0692/.0752/.0816. vs R3_ori: all cutoffs sig (HR@10 +.0156, NDCG@10 +.0099). vs MW-R-H 366023: n.s. everywhere (HR@5 +.0041 p .09, HR@10 +.0024). Latent used MORE: mask −.0369 [.0313,.0426], mean −.0352, shuffle −.0401 (MW-R-H shuffle −.017). motive_loss 3.04→1.96, cos_cross .067→.181 vs cos_neg .014. (JobManager's ".0744" line was the shuffled-latent result, not the ranking.)

### 2026-11-23 MovieLens rankings DONE (interventions + exclude-history pending): MW-nc rerun 374722, MW-R-H+motive 374774, MW-HN 374776 (tmp/ml_runs.py)

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | hist/top-10 | copy HR@10 | non-copy HR@5 / @10 |
|---|---|---|---|---|---|---|---|---|---|---|
| R3 | .0249 | .0556 | .0806 | .1335 | .0425 | .0527 | .0698 | .48 | .4061 | .0101 / .0426 |
| MW-R-H | .0193 | .0492 | .0701 | .1188 | .0363 | .0449 | .0605 | 1.09 | .3318 | .0143 / .0478 |
| MW-3b | .0269 | .0617 | .0830 | .1263 | .0469 | .0557 | .0696 | .39 | .3676 | .0172 / .0459 |
| MW-nc rerun | .0228 | .0557 | .0793 | .1250 | .0417 | .0514 | .0660 | .62 | .3597 | .0163 / .0467 |
| MW-R-H+motive | .0261 | .0531 | .0761 | .1207 | .0416 | .0509 | .0652 | .90 | .3424 | .0143 / .0468 |
| MW-HN | .0263 | .0588 | .0841 | .1279 | .0449 | .0552 | .0694 | .39 | .3628 | .0173 / .0496 |

- vs R3: MW-nc HR@10 −.0086* NDCG@10 −.0037*, @1–@5 n.s.; motive HR@10 −.0128* NDCG@10 −.0045*, HR@5 −.0045 (.05); MW-HN HR@10 −.0056 (.05/.048) rest n.s. (NDCG@10 −.0004).
- vs MW-R-H: MW-nc and MW-HN sig better at all cutoffs; motive sig at @1, @5, NDCG.
- All variants: non-copy gain at @3/@5 sig (+.003 to +.007), copy_row HR@10 loss −.043 to −.064 sig.
- CAUTION MW-HN: margin_loss .08/.02 already in epoch 0–1 (in-batch runs: ~.19) and WIDER loss ≈ 1.0 in epoch 0.5–1.5 → the hinge is trivially satisfied against STALE bank latents (old-model r is separable for the wrong reason); profile ≈ MW-3b (hist/top-10 .39) → latent probably unused; wait for interventions.
- motive: cos_cross .36 vs cos_neg .23 (global similarity also rose → partial collapse of spread).
- 2026-11-22 (user): history-excluded evals postponed — "fixing it, MW-R-H is still lower than R3; do it later". Asked JobManager to cancel pending 374723, 374775, 374777, 375421, 375423 (trainings untouched; resubmit later via eval_exclude_history.sbatch with TAG mwnc / mwrh_motive / mwrh_hardneg / mwrh_swapce / mwrh_distill).

### 2026-11-22 MW-nc rerun 374722 DONE (3 epochs; 20,327 copy rows without hinge) — latent STRONGLY used at .1250 / .0660

- Ranking (normal): HR .0228/.0557/.0793/.1250, NDCG .0228/.0417/.0514/.0660; vs R3 @1–@5 n.s., HR@10 −.0086*, NDCG@10 −.0037*; vs MW-R-H sig better at all cutoffs. (JobManager's ".0786" line is the shuffled-latent result.)
- Interventions (x − raw), all p < 5e-4: mask HR@10 −.0414 (copy −.1338 / non-copy −.0106); mean −.0335 (copy −.0992 / non-copy −.0117); shuffle −.0463. NDCG@10 −.0270 / −.0250 / −.0309.
- Latent use is LARGER than MW-R-H (shuffle −.0267) with better ranking; and the latent matters most on copy rows although they get no hinge → decoder dependence on r generalises to all rows (shared decoder), consistent with the copy-row intervention split of MW-R-H.

### 2026-11-22 MovieLens MW-R-H + same-motive 374774 DONE — latent used most of all runs, ranking between MW-R-H and MW-nc

- Ranking: HR .0261/.0531/.0761/.1207, NDCG .0261/.0416/.0509/.0652; vs R3 HR@10 −.0128*, NDCG@10 −.0045*, @1–@5 n.s. (HR@5 −.0045 p .05); vs MW-R-H better at @1, @5, NDCG (HR@10 +.0020 n.s.).
- Interventions (x − raw), all p < 5e-4: mask HR@10 −.0727 (copy −.2011 / non-copy −.0299); mean −.0545 (copy −.1527 / non-copy −.0218); shuffle −.0627 (copy −.1696 / non-copy −.0270); NDCG@10 −.0407 / −.0324 / −.0366.
- Latent use on MovieLens (shuffle HR@10): R3 −.0027, MW-3b −.0011 n.s., MW-R-H −.0267, MW-nc −.0463, motive −.0627.
- Probe on MW-nc + motive models requested via JobManager (tmp/submit_movielens_probe2.sh).

### 2026-11-22 MovieLens MW-HN 374776 DONE — per-user latent content UNUSED (as predicted from the trivially satisfied hinge)

- Ranking .0263/.0588/.0841/.1279, NDCG@10 .0694 (vs R3 HR@10 −.0056 p .05, rest n.s.).
- Interventions (JobManager CIs): shuffle_user HR@10 −.0006 [−.0030,+.0018] n.s.; mean −.0103 [.0061,.0144] sig; mask −.0715 sig. → another user's latent works as well as the own one; only the mean / zero vector hurt = the decoder needs a "current-model-like" latent but not this user's.
- Cause: hinge satisfied against STALE bank latents (margin_loss .02–.08 from epoch 1). Fix if pursued: recompute the hard negative's r with the current encoder (extra history-encoder pass) instead of reading the bank.


## 2026-11-23 MovieLens headroom map (analysis only, tmp/ml_headroom.py; no new runs)

Test rows bucketed by where train co-occurrence counts put the target (history items excluded from candidates):
last-item successor count top-10 ("last item"), else earlier-history-item count top-10 ("earlier history only"), else "neither".
Last-item count table alone: HR@10 .1494 (> R3 .1335); earlier-history table .0942; either list .2081.

| bucket | n | share | R3 | no-latent | MW-R-H | MW-nc | motive | MW-3b | MW-HN |
|---|---|---|---|---|---|---|---|---|---|
| last item | 1,518 | .149 | .5264 | .5283 | .4704 | .4875 | .4611 | .5165 | .5250 |
| earlier history only | 597 | .059 | .3551 | .3769 | .3417 | .3886 | .3635 | .3233 | .3166 |
| neither | 8,049 | .792 | .0430 | .0406 | .0359 | .0370 | .0385 | .0381 | .0390 |

- R3 − no-latent n.s. in every bucket. Last item: MW-R-H − R3 −.0560, MW-nc −.0389, motive −.0652 (all p ≤ .002 both tests).
  Earlier history only: MW-R-H −.0134 (.54/.58), MW-nc +.0335 (.10/.11), motive +.0084 (.68/.70). Neither: MW-R-H −.0071 (.004/.004), MW-nc −.0060 (.012/.013), motive −.0045 (.07/.08).
- Share of R3 hits: .59 / .16 / .25. Top-50 buckets give the same picture (shares .255 / .132 / .613 of rows).
- History length: MW-R-H below R3 in every length group; MW-nc = R3 for length ≤ 4, below for 10+.
- Complementarity of hits@10 with R3: no-latent union .1706 (each-only .037); MW-R-H .1682; MW-nc .1692; motive .1691; MW-3b .1732; MW-HN .1722.
  Latent-using models are no more complementary to R3 than the no-latent model is. Union of all 7: .2481.
- copy_row vs buckets: P(last item | copy_row) .378 vs .073 non-copy.
Reading: the headroom on MovieLens is in last-item transitions (R3 finds 53% of targets a count table lists in its top-10), not in
summarising the user's history; forced latent use costs exactly there and gains nowhere significantly.


## 2026-11-23 MovieLens: per-bucket gap with history excluded; cross-dataset characteristics; MW-R-H-CR submitted (376544)

- MW-R-H (369532) − R3 (371446) per evidence bucket, existing exclude-history files (tmp/ml_headroom_ex.py):
  HR@5 normal −.0104 → excluded −.0010 (n.s.) overall; last item −.0336 → −.0033; earlier-only −.0536 → −.0034; neither −.0029 → −.0004.
  HR@10 normal −.0148 → excluded −.0085 (sig); last item −.0560 → −.0389 (sig); earlier-only −.0134 → +.0134 (n.s.); neither −.0071 → −.0043 (.08/.09).
  Reading: the across-the-board gap = watched movies taking list slots (1.09 vs .48 per top-10); residual = last-item successors at ranks 6–10.
- Cross-dataset (tmp/xds.py): R3 − last-item-successor table HR@10: Toys +.016, CDs −.021, Steam +.013, MovieLens +.030;
  MW-R-H − R3: +.0034, +.0132, +.0030, −.0148. Rows/item 8.6 / 9.2 / 18.7 / 27.9; lexical-target rows 10.6% / 8.8% / 3.0% / 1.4%.
- User 11-23: try MW-R-H + ORIGINAL collaborative reasoning InfoNCE (user head = same user's other rows, target head = same next item)
  on MovieLens. Code c3b42206 (METHOD=infonce + MARGIN_MODE=refweight, EARLY_STOP=False; tiny-model check tmp/test_mwrh_cr.py: with
  InfoNCE weights 0, CE / hinge / total loss = MW-R-H's exactly). Run Movielens_margin_0.2_wider_0.1_refweight_nolatent_rank_cr,
  USER_W=TARGET_W=0.1 from epoch 2, GATE=0.5, queue 10k, 16 positives per head, 256 negatives; job **376544** (R3_mwrh_cr_movielens,
  hady A40, via JobManager, tmp/submit_movielens_mwrh_cr.sh). No exclude-history eval.
  Differences from the same-motive run 374774 besides the positives: FIFO queue of 10k gated latents (not the full row bank), two heads.


## 2026-11-24 probe 376414 DONE (MovieLens MW-nc rerun 374722, MW-R-H + same-motive 374774; first 20,000 train rows)

| model | gap_history (same last item, different next) | gap_next (same next, different last) | next / history | cross-user same-item gap | within-user gap |
|---|---|---|---|---|---|
| R3 371446 | +.262 | +.036 | .14 | | |
| MW-R-H 369532 | +.061 | +.025 | .41 | | +.107 |
| MW-nc 374722 | +.186 | +.068 | .37 | +.075 | +.054 |
| MW-R-H + same-motive 374774 | +.318 | +.151 | .48 | +.159 | +.064 |

gap_next does not exceed gap_history in any model. Same-motive raises gap_next 6x over MW-R-H (.025 → .151) but gap_history rises too
(.061 → .318). Train rows, and same-next rows are the trained positives of 374774, so its gap_next is partly fitted.
MW-R-H-CR 376544 started 11-24 ~04:20 on album.


## 2026-11-24 MovieLens: two same-motive follow-ups after probe 376414 (user: "implement 1 and 2")

Base for both = 374774 recipe (MW-R-H rank weights, margin 0.2, WIDER 0.1, same-motive InfoNCE from epoch 2, 3 epochs, Qwen2.5-1.5B).
1. **History-matched InfoNCE negatives** (code 1caa41c6, MOTIVE_HARD_NEG=16, MOTIVE_W=0.1): per anchor up to 16 bank rows of other
   users with the same last history item and a different next item are added to the 256 random negatives. Positives and negatives
   are all bank latents, so the fresh-vs-stale shortcut of MW-HN (374776) does not apply. Tiny-model test tmp/test_motive_hardneg.py:
   loss = manual logsumexp, off = unchanged, hinge negative mode untouched. Output ..._rank_motive_hn.
2. **Stronger InfoNCE** (MOTIVE_W=0.3, settings only). Output ..._rank_motive_w03.
3. Probe on both (afterok). Submitted via JobManager, tmp/submit_movielens_motive_next.sh; job IDs: see JobManager reply / run_logs
   (R3_mwrh_motive_hn_movielens, R3_mwrh_motive_w03_movielens, R3_motive_probe_movielens3). No exclude-history evals.
Question: does the next gap exceed the history gap, and does ranking at @5/@10 move toward R3 (.0806 / .1335; NDCG .0527 / .0698)?

Job IDs (2026-11-24): 376835 same-motive + history-matched InfoNCE negatives, 376836 InfoNCE weight 0.3, 376837 probe (afterok on both).

2026-11-24: 376835 / 376836 FAILED at start-up (TypeError str > int): margin_wider_collab.sbatch had no space after --motive_hard_neg "..." so the value was glued to --copy_swap_weight. Fixed in bd018410 (frozen code_bd018410); tmp/test_sbatch_args.sh now renders the motive command and checks argument types through fire against train(). Probe 376837 cancelled (dependency never satisfied); resubmission of the same three jobs requested from JobManager.

Resubmitted from code_bd018410 (2026-11-24): 376838 same-motive + history-matched InfoNCE negatives, 376839 InfoNCE weight 0.3, 376840 probe (afterok). Start-up verified: both log training steps; 376838 prints the hard-negative index (99.98% of rows have candidates, mean 26.6) with hinge: False, InfoNCE: 16 per anchor.

## 2026-11-22/23 — HiLaR GRPO on MW-R-H: step 1 = signal probe (user request)

HiLaR (arXiv 2607.27760, "Hierarchical Latent Reasoning for LLM-based Recommendation"): G=6 rollouts of K=4 Gaussian latents (σ .12/.08/.05/.025), reward = hit + prefix + F1 + SASRec z-score + target logp + process (align, gain), KL .05, lr 5e-6, backbone frozen; its repo is a placeholder (README "Coming Soon"), so weights are unknown. User chose: their GRPO recipe on MW-R-H, SASRec term included. Prior reimplementation (worktree HiLaR, Musical, doc `document/hilar.md`): stage 2 HR@10 .1205 / NDCG@10 .0713 vs R3 .1226 / .0950, latents unused (swap .1550→.1540), probe at the paper's σ: 1.0–1.2 distinct titles per 6 rollouts, reward std 0–0.015 → GRPO never run. Caveat for the SASRec term: the Cornac SASRec is in-sample on train rows (train top-10 hit .835 vs test .100).
Single-token design (user point: only 1 thinking token): no hierarchy/process terms; policy N(r, σ²I); σ relative to median inter-user ‖r_i−r_j‖; antithetic ± rollouts, G=12; dLL vs the noise-free r instead of the empty-slot gain.
**Step 1 (probe, no update):** `src/grpo_attention_tuning/hilar_probe.py` + `scripts/hilar_probe.sbatch`, code 6af2c9c8 (frozen `output_dir/hilar_probe_code/code_6af2c9c8`); σ_rel 0.1/0.25/0.5/1.0; models MW-R-H 365504 and MW; 128 copy + 128 non-copy train rows; reports distinct titles per group and within-group std of hit/prefix/F1/dLL/combined. Submitted via JobManager (smoke → full; ids pending). Gate: train only if some σ gives ≥2 distinct titles per group with a usable reward std.
- Probe jobs submitted via JobManager: smoke 376904 (1 h limit) → full 376905 (afterok). Output analysis/results/hilar_probe/toys_probe/.


## 2026-11-24 MovieLens MW-R-H-S (swapped-pass CE, COPY_SWAP_W=1) 375420 DONE — FIRST MovieLens run meeting the criterion

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | hist/top-10 |
|---|---|---|---|---|---|---|---|---|
| R3 seed 0 (371446) | .0249 | .0556 | .0806 | .1335 | .0425 | .0527 | .0698 | .48 |
| R3 seed 1 | .0261 | .0581 | .0819 | .1372 | .0445 | .0542 | .0720 | |
| MW-R-H (369532) | .0193 | .0492 | .0701 | .1188 | .0363 | .0449 | .0605 | 1.09 |
| MW-R-H-S (375420) | .0267 | .0616 | .0887 | .1380 | .0467 | .0578 | .0737 | .41 |

- vs R3 seed 0 (p_t / p_boot): HR@1 +.0018 (.25/.26), HR@3 +.0060 (.005/.006), HR@5 +.0082 (.001/.001), HR@10 +.0045 (.12/.12);
  NDCG@3 +.0042 (.008/.009), NDCG@5 +.0051 (.001/.003), NDCG@10 +.0039 (.012/.016).
- vs R3 seed 1: HR@5 +.0069 (.006/.009), NDCG@5 +.0037 (.021/.028) sig; HR@1 +.0006, HR@3 +.0034 (.11/.12), HR@10 +.0008 (.78/.82), NDCG@3 +.0022, NDCG@10 +.0017 n.s.
- vs MW-R-H: all cutoffs sig (HR@10 +.0193, NDCG@10 +.0132, p < 5e-4).
- copy_row HR@10 .4026 (R3 .4061, MW-R-H .3318): −.0035 vs R3 n.s.; HR@5 +.0185 (.037/.039). non-copy HR@10 .0498 (R3 .0426): +.0072 (.008/.006), sig at all cutoffs.
- Latent tests (p < 5e-4): mask HR@5 −.0283 / HR@10 −.0346 / NDCG@10 −.0191; mean −.0258 / −.0356 / −.0190; shuffle −.0284 / −.0381 / −.0206.
  By copy split (HR@10): shuffle copy −.0307 (.0007), non-copy −.0405; mean copy −.0240, non-copy −.0395; mask copy −.0201 (.022), non-copy −.0395.
  → on non-copy rows about 80% of the hits need the user's own latent; on copy rows the decoder keeps its history-only reading.
- Training: copy_swap_ce .39 → .22; margin_loss .20 → .06 (hinge satisfied by epoch 3); WIDER stays low (≤ .16).
- Latent rank: effective rank 26.5, mean pairwise cos .017.
Caveats: single seed for MW-R-H-S; MW-R-H was not seed-robust on Toys.
- 2026-11-23 probe 376905 (code 6af2c9c8): decode columns valid (MW-R-H sigma_rel 1.0: 1.56 distinct/12, P(>=2) .40, hit unchanged .30; MW 2.52 distinct, hit .205), BUT dLL column INVALID: transformers 4.48 ignores logits_to_keep so the teacher-forced slice used prompt positions (clean mean logp -8.85, constant in sigma). Fixed in code 0bbb59bc (CPU check: clean -0.18, noisy -0.94); rerun pending.


## 2026-11-24 MovieLens MW-R-H-D (KL to the no-latent reference, DISTILL_W=1) 375422 DONE — also meets the criterion

MW-R-H-D: HR@1/3/5/10 .0276/.0576/.0851/.1350, NDCG@3/5/10 .0446/.0558/.0718; hist/top-10 .45.
- vs R3 seed 0: HR@1 +.0028 (.044/.051), HR@3 +.0020 n.s., HR@5 +.0045 (.045/.042), HR@10 +.0015 (.60/.63); NDCG@3 +.0021 n.s., NDCG@5 +.0031 (.025/.023), NDCG@10 +.0020 (.15/.16).
- vs R3 seed 1 (.1372 / .0720): n.s. at every cutoff (HR@5 +.0032, HR@10 −.0023, NDCG@10 −.0002).
- vs MW-R-H: all cutoffs sig (HR@10 +.0162, NDCG@10 +.0113).
- copy_row HR@10 .4093 (R3 .4061; +.0031 n.s.); non-copy .0436 (R3 .0426; +.0009 n.s.; NDCG@3/@5 sig +.0013/+.0016).
- Latent tests (p < 5e-4): mask HR@5 −.0413 / HR@10 −.0593 / NDCG@10 −.0334; mean −.0447 / −.0694 / −.0380; shuffle −.0472 / −.0729 / −.0395.
  By copy split (HR@10): shuffle copy −.2125, non-copy −.0264; mean copy −.2019, non-copy −.0252; mask copy −.1708, non-copy −.0222.
- Training: distill_kl .18 → .016; margin_loss .20 → .014; WIDER ≤ .27. Latent rank: effective rank 9.5 (S: 26.5), top-1 energy .50.
- S − D (paired): all cutoffs n.s. overall (HR@10 +.0030, NDCG@10 +.0019); non-copy HR@10 +.0063 (.019/.016), NDCG@10 +.0026 (.011/.010) for S; copy_row n.s.
Reading: both recover copy_rows to R3 level. S does it by making the decoder predict them WITHOUT the right latent (shuffle on copy rows
−.031) and keeps the non-copy gain; D matches the reference WITH the own latent, so copy rows become strongly latent-dependent (−.21)
and the non-copy gain over R3 is not significant. S is the stronger of the two on ranking; D shows the stronger latent dependence.


## 2026-11-24 Method descriptions for the report: MW-R-H-S (swapped-pass loss) and MW-R-H-D (reference distillation)

**Common setup.** Reference model = the same base (Qwen2.5-1.5B) fine-tuned on the same data and prompt with the latent slot
removed; trained once, frozen. Row weight `w_i` = rank percentile (0–1) of the reference loss on row i's target; low `w_i` = the
history alone explains the target (copy_row = lowest quarter). Per row, the decoder runs an own pass (history + own latent, loss
`own_i`) and a swapped pass (history + another row's latent from the batch, detached; loss `swapped_i`).

```
L = mean_i( own_i )                                              prediction
  + sum_i( w_i · max(0, m + own_i − swapped_i) ) / sum_i( w_i )   hinge (MW-R-H), m = 0.2
  + [S]  mean_i( (1 − w_i) · swapped_i )                          swapped-pass loss, weight 1   (375420)
  + [D]  mean_i( (1 − w_i) · KL_i )                               reference distillation, weight 1   (375422)
  + 0.1 · WIDER                                                   latent diversity
```

`KL_i` = KL(reference ‖ model own pass) on the next-token distribution, averaged over the target title tokens of row i; the
reference reads the same prompt without the latent slot, target tokens aligned one by one; reference under no_grad.

**Why MW-R-H needed it.** In MW-R-H low-`w` rows only get a small hinge. The decoder is shared, so the habit learned on high-`w`
rows ("trust the latent") spreads to all rows; on MovieLens that cost −.074 HR@10 on copy_rows (decoder lost its history-only
reading). A small hinge weight says "don't push here"; S and D say something active about those rows.

**How the weighted terms divide the work.**
- High-`w` rows: the hinge dominates → the own latent must beat another user's latent by the margin.
- Low-`w` rows, S: the decoder must still predict the target with the WRONG latent → it must be able to read the history alone.
  S reuses the swapped pass already run for the hinge (no extra compute). On the swapped pass the hinge and S pull in opposite
  directions; `w_i` decides which wins per row, giving the decoder a rule for when to rely on the latent.
- Low-`w` rows, D: the model, with its OWN latent in place, must predict what the history-only model predicts. One extra forward
  pass of a frozen model per step.
- Inference, prompt and latent module are unchanged from R3 in both.

**What the results say about the mechanism (MovieLens, HR@10; shuffle = another user's latent).**

| rows | R3 | MW-R-H | S | S shuffle | D | D shuffle |
|---|---|---|---|---|---|---|
| copy_rows (2,541) | .4061 | .3318 | .4026 | −.0307 | .4093 | −.2125 |
| non-copy (7,623) | .0426 | .0478 | .0498 | −.0405 | .0436 | −.0264 |
| all | .1335 | .1188 | .1380 | −.0381 | .1350 | −.0729 |

- Both restore copy_rows to R3's level (the whole MW-R-H deficit).
- S: copy_rows stay largely independent of the latent; the latent carries the non-copy rows (about 80% of those hits need the own
  latent) and S is sig above R3 there (+.0072).
- D: copy_rows become strongly latent-dependent (half of their hits are lost under shuffle) although the latent adds no accuracy
  over R3 there; non-copy gain over R3 n.s. (+.0009). Larger latent-test numbers, but part of the dependence is on rows where the
  latent does not help.
- S − D: n.s. overall; S sig better on non-copy HR@10 (+.0063). S is the lead recipe. Both single seed; replication pending.


## 2026-11-24 MovieLens MW-R-H-CR 376544 and same-motive + history-matched InfoNCE negatives 376838 DONE — neither meets the criterion

| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | hist/top-10 | copy HR@10 | non-copy HR@10 | shuffle HR@10 (copy / non-copy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R3 371446 | .0249 | .0556 | .0806 | .1335 | .0425 | .0527 | .0698 | .48 | .4061 | .0426 | −.0027 |
| same-motive 374774 | .0261 | .0531 | .0761 | .1207 | .0416 | .0509 | .0652 | .90 | .3424 | .0468 | −.0627 |
| MW-R-H-CR 376544 | .0249 | .0552 | .0818 | .1265 | .0423 | .0532 | .0677 | .57 | .3617 | .0481 | −.0507 (−.1452 / −.0192) |
| motive + HN-neg 376838 | .0241 | .0532 | .0729 | .1155 | .0406 | .0486 | .0623 | .91 | .3310 | .0437 | −.0413 (−.1149 / −.0168) |
| MW-R-H-S 375420 | .0267 | .0616 | .0887 | .1380 | .0467 | .0578 | .0737 | .41 | .4026 | .0498 | −.0381 |

- CR vs R3: = at @1–@5 (n.s.), HR@10 −.0070 (.015/.006), NDCG@10 −.0021 (.17/.15). copy −.0445 sig, non-copy +.0055 (.034/.041), HR@5 +.0060 sig.
  CR vs MW-R-H: all cutoffs sig better. CR vs same-motive: HR@5 +.0057 (.017/.025), HR@10 +.0058 (.037/.038), NDCG n.s.
  S − CR: HR@3 +.0064, HR@5 +.0070, HR@10 +.0115, NDCG@10 +.0060, all sig. Latent tests CR (p < 5e-4): mask −.0382, mean −.0437, shuffle −.0507.
  Hinge satisfied early (margin_loss .003 from epoch 1.5). The same-user term did not make CR worse than same-motive here (feared from Toys 358547).
- HN-neg vs R3: HR@5 −.0077, HR@10 −.0180, NDCG@5 −.0041, NDCG@10 −.0075, all sig. vs same-motive 374774: n.s. (HR@10 −.0052, .06/.078; NDCG@10 −.0030, .049/.06).
  Latent tests (p < 5e-4): mask −.0333, mean −.0405, shuffle −.0413.
  Training: InfoNCE did NOT separate the history-matched negatives: cos_hard ≥ cos_cross throughout (ep 2.5: positives .619, hard .668,
  random .579; 374774 at ep 2.5: positives .356, random .229), motive_loss 3.03 → 2.78 (374774: 2.85 → 2.24); all similarities rose
  together (projection space contracting). ~14.8 hard negatives per anchor were used.
  Reading: a latent computed from the history cannot tell "same last movie, different next" rows apart well enough; adding those
  negatives degraded the InfoNCE rather than sharpening it. Option 1 is negative. (Probe 376840 pending on 376839.)
- 376839 (weight 0.3) still running; its hinge is NOT satisfied in epochs 1.5–3 (margin_loss ≈ .195) → expect weak latent use.


## 2026-11-24 MovieLens same-motive with InfoNCE weight 0.3 (376839) DONE — worse, and not a clean test of the weight

HR@1/3/5/10 .0226/.0503/.0710/.1127, NDCG@3/5/10 .0384/.0469/.0603; hist/top-10 .80.
- vs R3: HR@3 −.0053, HR@5 −.0095, HR@10 −.0209, NDCG@5 −.0058, NDCG@10 −.0094, all sig. copy_row .3117 (−.0945), non-copy .0463 (+.0037 n.s.; HR@5 +.0043 sig).
- vs same-motive weight 0.1 (374774): HR@1 −.0034, HR@5 −.0050, HR@10 −.0081, NDCG@10 −.0049, all sig.
- Latent tests: mask HR@10 −.0059 (.004/.002), mean −.0082, shuffle −.0096 (p < 5e-4) — significant but 6x smaller than 374774 (−.0627).
- Training: hinge NEVER satisfied (margin_loss .202 / .206 / .199 / .197 / .195 / .194 per half epoch); it was already unsatisfied in
  epoch 0–1, BEFORE the InfoNCE turns on and where the recipe is identical to 374774 (there .200 / .197 then .109 in epoch 1–1.5).
  ce_loss spiked to 1.64 in epoch 0.5–1.0 (374774: .655). So the run diverged from 374774 before the weight could matter
  (runs are not deterministic) → the weak latent use cannot be attributed to the weight alone. InfoNCE itself separated
  positives from negatives (cos .400 vs .289 at the end).
Conclusion for the pair (user's options 1 and 2): neither history-matched negatives (376838) nor a stronger InfoNCE (376839) improves
on same-motive 374774; 376839 is confounded by an unsatisfied hinge from epoch 1.

## 2026-11-24 probe 376840 DONE (same-motive + HN-neg 376838, same-motive weight 0.3 376839; first 20,000 train rows)

| model | gap_history | gap_next | next / history | cross-user same-item gap |
|---|---|---|---|---|
| MW-R-H 369532 | +.061 | +.025 | .41 | |
| same-motive w0.1 374774 | +.318 | +.151 | .48 | +.159 |
| + HN-neg 376838 | +.225 | +.102 | .45 | +.108 |
| weight 0.3 376839 | +.432 | +.168 | .39 | +.176 |

Neither follow-up raised the next / history ratio above 374774 (.48). HN-neg lowered both gaps (history −29%, next −33%); weight 0.3 raised both, history more. gap_next < gap_history in every model so far. All MovieLens jobs of this batch are finished.
- 2026-11-23 corrected probe 377036 (toys_probe2, code 0bbb59bc): MW-R-H sigma_rel 1.0: noise/‖r‖ 1.35, distinct 1.52/12, P(≥2) .36, hit .303 (clean .301), mean dLL −.137 (std .071), combined reward std .143, 80% of rows > .01 (non-copy: distinct 1.62, sd .107); sigma_rel ≤0.5: sd ≤ .046. MW sigma_rel 1.0: sd .287 but hit .289→.209. Concerns: signal mostly noise cost (mean dLL −.137), perverse incentive (robustness to noise = ignoring r), SASRec in-sample. User said go ahead.
- **Trainer built** `src/grpo_attention_tuning/hilar_grpo.py` + `scripts/hilar_grpo.sbatch`, commit 55908a7e (frozen `output_dir/hilar_grpo_code/code_55908a7e`): policy N(mu(H_u), σ²I), σ = 1.0·D/√d (D = median inter-user ‖mu_i−mu_j‖ on the start model), antithetic G=12, batch 8 users, greedy constrained decode per rollout; R = hit + .5 prefix + .5 F1 + .1·SASRec z(clip ±3) + dLL (clip ±1, vs noise-free mu); group-normalised advantage; loss −mean A log N(r_g; mu_θ, σ²) + β‖mu_θ−mu_ref‖²/(2σ²), β .05, lr 5e-6, attention fp32 only, one update per batch; logs antisym/sym reward std of the ± pairs (first-order signal); resume.pt every 500 steps. SASRec = Cornac Toys 2026-11-19 pkl (in-sample on train). Submitted via JobManager: smoke toys_mwrh_hilar_smoke → full toys_mwrh_hilar (ids pending).
- Submitted 2026-11-23 via JobManager: HiLaR GRPO smoke 378363 (toys_mwrh_hilar_smoke) → full 378364 (toys_mwrh_hilar, afterok). JobManager declines to delete the smoke checkpoint (only with its own user OK); I delete it myself after checking the smoke (user asked 11-23 to remove smoke ckpts from scratch).
- Smoke 378363 DONE (2m07s): 8 steps OK, antisym/sym reward std ratio ~1.4-1.7, sigma 5.37, SASRec 6241/6269 titles mapped; smoke ranking (64 rows) hook check only. Smoke checkpoint dir deleted 2026-11-24 (user OK). Full 378364 (toys_mwrh_hilar) RUNNING on alarm, ~15-18 h train + eval.

### Result: rank GRPO arm A (target injected into no-hit groups), Toys, 374798 DONE 2026-11-24 (29 h 34 m)
Compare script `analysis/results/rank_grpo_AC/compare.py` (+ output.txt; paired bootstrap + t-test, n = 6,738).
| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| MW-R-H 365504 (start) | .0383 | .0681 | .0828 | .1086 | .0615 | .0698 |
| MW-R-H + beam GRPO 368198 | .0381 | .0681 | .0827 | .1089 | .0615 | .0700 |
| R3_ori | .0377 | .0607 | .0778 | .1023 | .0580 | .0659 |
| **A 374798** | .0370 | .0652 | .0795 | .1043 | .0594 | .0674 |
- A vs start: SIG WORSE at @3/@5/@10 (HR@10 −.0043 [−.0071,−.0013] p .003; NDCG@10 −.0024, p .002; HR@1 n.s.). vs 368198 same. vs R3_ori: n.s. (HR@10 +.0021, p .49).
- Split: copy HR@10 −.0132 sig vs start; non-copy −.0030 sig vs start (still +.0053 vs R3_ori, borderline p .05).
- Interventions: raw HR@10 .1037; mask −.0252, mean −.0252, shuffle −.0260 (all sig) → latent still used (start ≈ −.030 to −.035).
- List quality: A DOES move the model (top-1 changed on 28% of rows vs 13% for 368198) and improves list diversity slightly (history slots .102→.088, distinct brands 3.61→3.88, top-1 brand share .44→.40) but hits fall (732→703), so accuracy is lost faster than waste is removed. Training end: group hit .25, injected .75, KL .012.
- Reading: hindsight injection gives a real gradient (unlike beam-only GRPO) but acts like SFT on memorised train targets with lr 3e-5; it overfits. Confound: lr 3e-5 / G=16 differ from 368198. C (374799) and A+C (374800) still running.

### Result: rank GRPO arm C (history −0.5 / redundant-beam −0.25 penalties, no target injection), Toys, 374799 DONE 2026-11-24 (29 h 48 m)
HR@1/3/5/10 .0315 / .0586 / .0723 / .0959; NDCG@3/5/10 .0471 / .0527 / .0603 (raw interventions run: .0956 / .0600).
- vs start MW-R-H: SIG WORSE at every cutoff (HR@10 −.0128 [−.0169,−.0086], NDCG@10 −.0095 [−.0118,−.0071], HR@1 −.0068). vs R3_ori: SIG WORSE (HR@10 −.0064 [−.0125,−.0004] p .04; NDCG@10 −.0056 p .003).
- Split: copy HR@10 .4581 (−.0682 vs start, −.0754 vs R3_ori, sig); non-copy .0446 (−.0049 vs start sig; +.0034 vs R3_ori n.s.).
- Interventions: mask −.0171, mean −.0295, shuffle −.0251 (sig) → latent used.
- List quality: history slots .102→.065, distinct brands 3.61→4.10, but hits 732→646, top-1 changed on 51% of rows. The redundancy penalty (same brand + title Jaccard ≥ .5) penalises exactly the same-product variants that copy-row targets are → copy recall collapses. Design flaw of the dup term, not only over-training. Training end: KL .048, hist_slot_frac .031, dup_slot_frac .086.
- 2026-11-24 scratch cleanup (user OK): deleted checkpoints Toys_and_Games_R3_grpo_toys_mwrh_rankA, ..._rankC (6 GB each) and Toys_and_Games_R3exact_toys_r3ori_exact (3 GB) = 15 GB; results/predictions kept in output_dir/grpo_stage2/<tag> and output_dir/r3_exact_grpo/<tag>. rankAC (running) and hilar (running) untouched.

### Result: rank GRPO arm A+C, Toys, 374800 DONE 2026-11-24 (29 h 51 m) — all three arms done
HR@1/3/5/10 .0356 / .0631 / .0803 / .1033; NDCG@3/5/10 .0516 / .0587 / .0660.
- vs start MW-R-H: SIG WORSE except HR@5 (HR@10 −.0053 [−.0086,−.0022] p .001; NDCG@10 −.0037 [−.0054,−.0021]; HR@5 −.0025 p .065 / pt .047). vs R3_ori: n.s. everywhere (HR@10 +.0010 p .74; NDCG@10 +.0002).
- Split: copy HR@10 .4952 (−.0311 vs start sig; −.0383 vs R3_ori sig); non-copy .0478 (−.0017 vs start n.s.; +.0066 vs R3_ori sig).
- Interventions: mask −.0242, mean −.0255, shuffle −.0261 (sig) → latent used. List quality: history slots .077, brands 3.99, hits 696, top-1 changed on 37% of rows. KL .019.
- Summary of arms (HR@10 / NDCG@10): start .1086/.0698; A .1043/.0674; A+C .1033/.0660; C .0959/.0603; beam-only 368198 .1089/.0700; R3_ori .1023/.0659. Ordering by damage tracks the redundancy penalty (copy-row loss: A −.013, A+C −.031, C −.068); the history penalty and hindsight injection alone do not help ranking. None meets "≥ R3 AND beats start".

### Result (ranking): HiLaR-style GRPO on MW-R-H, Toys, 378364 (training DONE 2026-11-24, 11.6 h, 6,737 steps; interventions still running)
HR@1/3/5/10 .0365 / .0650 / .0812 / .1077; NDCG@3/5/10 .0531 / .0597 / .0683 (compare script `analysis/results/rank_grpo_AC/compare.py`, scratch final_result.json).
- vs start MW-R-H: no gain. HR@10 −.0009 [−.0033,+.0013] p .50 n.s.; HR@3 −.0031 sig; NDCG@3/5/10 −.0023/−.0018/−.0015 sig (small). vs R3_ori: n.s. at all cutoffs (HR@10 +.0055 [+.0000,+.0113] p .053; NDCG@10 +.0024 p .17).
- Split: copy HR@10 .5239 (−.0024 n.s. vs start; NDCG@10 −.0073 sig); non-copy .0488 (−.0007 n.s. vs start; +.0076 sig vs R3_ori). List quality unchanged from start (history slots .108, brands 3.68, same top-1 as start 82%).
- Training curve: reward .450 → .554; dLL −.115 → −.014 (accounts for the whole gain), noisy hit .141 → .151, SASRec z flat 2.10 → 2.06, reward std .140 → .102, antisym/sym ≈ 1.4 stable, drift ‖μ−μ_ref‖ 0.4 → 16. Interventions (mask/mean/shuffle) pending: they decide whether r got ignored.

## 2026-11-24 — Rank GRPO removed (user instruction)
User: ignore the 368198 recipe; "R3's GRPO" = the original R3 code. Removed rank_grpo.py, the rank/A/C options in train_noise_grpo.py + grpo_stage2.sbatch, and the Test A script (grad_coherence.py/.sbatch) — commit 6284ef8b; all recoverable from git (55908a7e and earlier). HiLaR probe/trainer made self-contained (running job 378364 uses frozen code_55908a7e, unaffected). Results/logs of 368198, A, C, A+C and Test A stay in this log and analysis/results as history, but are not to be used as a method. The exact R3 recipe on MW-R-H 365504 is still to be run.

### Check: does R3's own stage 2 help? R3_ori runs (user: "we've trained R3 on Toys and CDs for a few seeds, use that"), 2026-11-24
`analysis/results/r3_ori_stage2_check/compare.py` (+ output.txt). R3_ori `output_dir/{Toys,CDs}/{.,run2,run3}/`: stage 1 = `latent`, stage 2 = `nomul_attn_lr_5e-5` (attention_grpo.sh recipe; Toys run 1 also lr 1e-5, 1e-4). Paired bootstrap + t-test over all test rows, stage 2 − stage 1, HR@10 (NDCG@10):
- Toys run1 lr 5e-5: +.0033 [+.0007,+.0058] sig (+.0015 sig); lr 1e-5: −.0004 n.s. (+.0001); lr 1e-4: +.0046 sig (+.0012 n.s.). Toys run2 lr 5e-5: COLLAPSED (HR@10 .0024; −.1027). Toys run3: +.0009 n.s. (+.0013 n.s.; NDCG@5 +.0017 sig).
- CDs run1: +.0106 [+.0067,+.0146] sig (+.0087 sig); run2: +.0002 n.s. (−.0000); run3: −.0016 n.s. (−.0010 n.s.; HR@5 −.0019 sig). Mean of the 3 CDs runs: HR@10 +.0030 [+.0016,+.0045] sig, NDCG@10 +.0026 sig, driven by run1.
- Stage-1 run-to-run spread on Toys HR@10 .1023 / .1051 / .1082 (~±.003, as large as the stage-2 gains).
- Reading: R3's own stage 2 gives a small, non-reproducible gain (one big CDs run, otherwise ≈0), one collapse in 6 lr-5e-5 runs; at lr 1e-5 it is a no-op, which matches our exact replica 371452 (−.0010 n.s.) — replication confirmed. The paper's RL ablation (Toys +~4%, CDs +~15% HR@10) is one run per cell.

## 2026-11-24 — R3's exact stage-2 GRPO on MW-R-H (user request)
Question: can R3's own GRPO (original code, unmodified) help MW-R-H? R3_ori's own runs: small non-reproducible gain at lr 5e-5 (CDs +.0106 / +.0002 / −.0016; Toys +.0033 / collapse / +.0009), no-op at lr 1e-5.
Setup: code `output_dir/r3_exact_grpo_code/code_c558b44` (R3_ori HEAD), script copy `output_dir/r3_exact_grpo_code/r3_exact_grpo.sbatch` (09716616), ANALYSIS_CODE `grpo_stage2_code/code_1a2053ff`, base MW-R-H 365504 (read-only), 2×A40, G 8, noise N(0,1.5²) on r with the first copy noise-free, −PPL reward, beta 0, batch 256, micro 8, seed 42, 1 epoch. Arms: lr 5e-5 (R3's launch-script lr, tag toys_mwrh_r3exact_lr5e-5) and lr 1e-5 (tag toys_mwrh_r3exact_lr1e-5); smoke toys_mwrh_r3exact_smoke first (weights load check). Submitted via JobManager (ids pending). Compare against MW-R-H 365504 and R3_ori (+ R3_ori's own stage 2) with paired bootstrap + t-test, copy split, mask/mean/shuffle interventions.
- Submitted via JobManager 2026-11-24: smoke 379966 (toys_mwrh_r3exact_smoke, lr 5e-5) → full A 379967 (lr 5e-5) and full B 379968 (lr 1e-5), both afterok:379966; 2×A40 each, 72 h. Smoke pending on Priority.

### Result (final): HiLaR-style GRPO on MW-R-H, Toys, 378364 DONE 2026-11-24 (15 h 59 m incl. interventions)
Ranking as above (HR@10 .1077 / NDCG@10 .0683: = start n.s. at HR@10, NDCG@3/5/10 slightly lower sig; = R3_ori n.s.). Interventions (raw HR@10 .1082 / NDCG@10 .0686): mask −.0297 [−.0352,−.0240], mean −.0294 [−.0349,−.0240], shuffle_user −.0343 [−.0396,−.0286] (all sig) → latent content STILL USED, same size as the start (−.030 to −.035): the feared "robust to noise = ignores r" did not happen. Net: stable, latent kept, but no ranking gain. Checkpoint (3.0 GB) on scratch Toys_and_Games_R3_grpo_toys_mwrh_hilar (deletion pending user OK).
- 2026-11-24 20:2x: hady A40 pool unavailable (album drained "SlurmdSpoolDir is full", alarm full) → user: move to researchshort. Asked JobManager to cancel 379966/379967/379968 and resubmit the same chain on researchshort (--qos research-1-qos, 48 h, a40, 2 GPUs). New ids pending.
- Moved to researchshort 2026-11-24: old 379966/7/8 cancelled; new chain smoke 380282 → full A 380283 (lr 5e-5) and full B 380284 (lr 1e-5), afterok:380282; 48 h, research-1-qos, a40 ×2. Note: researchshort jobs on alarm/album can be preempted by hady jobs (JobManager moves a job to hady after 2 preemptions).
- Smoke 380282 DONE 2026-11-25 (2m06s, exit 0): original R3 code loads and trains on MW-R-H 365504; smoke checkpoint dir (5.9 GB) deleted (standing user instruction to remove smoke ckpts). Full A 380283 (lr 5e-5) and B 380284 (lr 1e-5) pending on Priority (researchshort).
- 2026-11-25: JobManager (its user request) moved full runs 380283 (lr 5e-5) and 380284 (lr 1e-5) from researchshort to hadylauwresearch (a40, 48 h; no requeue risk). Hady a40 full; estimated starts: 380283 today 19:40, 380284 30 Nov 13:06.

### Result: R3's exact stage-2 GRPO (original code) on MW-R-H 365504, Toys — lr 5e-5 (380283, DONE 11-25, 6 h 59 m) and lr 1e-5 (380284, ranking done)
Script `analysis/results/r3_exact_mwrh/compare.py` (+ output.txt); paired bootstrap + t-test, n = 6,738.
| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| MW-R-H 365504 (start) | .0383 | .0681 | .0828 | .1086 | .0615 | .0698 |
| R3_ori stage 1 / stage 2 lr 5e-5 | .0377 / .0380 | .0607 / .0628 | .0778 / .0784 | .1023 / .1055 | .0580 / .0586 | .0659 / .0674 |
| **R3 exact, lr 5e-5** | .0197 | .0383 | .0484 | **.0701** | .0345 | .0415 |
| **R3 exact, lr 1e-5** | .0380 | .0689 | .0828 | **.1095** | .0614 | .0701 |
- lr 5e-5: COLLAPSE. Vs start HR@10 −.0386 [−.0438,−.0335], NDCG@10 −.0283, all cutoffs sig; copy HR@10 .346 (−.181), non-copy .0310 (−.0185). Latent use destroyed: raw .0699; mask +.0086 (masking HELPS), mean −.0101, shuffle −.0065. Top-1 changed on 55% of rows, distinct brands 2.79, history slots .067. Training: grad_norm grew from .03–.05 to 4.8 (epoch .36) → unstable, then loss ≈ −.03; clip_ratio 0. Same failure mode as R3_ori Toys run 2 (collapse at 5e-5), here on every seed-42 run.
- lr 1e-5: NO-OP. HR@10 +.0009 [−.0003,+.0022] p .16; all cutoffs n.s. vs start; same top-1 on 93% of rows; copy +.0024 n.s., non-copy +.0007 n.s. Vs R3_ori stage 1: sig better (HR@10 +.0073) — the start already was. Interventions for lr 1e-5 still running.
- Conclusion: R3's own GRPO does not help MW-R-H: harmless but inert at lr 1e-5, destructive at lr 5e-5 (the lr at which R3_ori sometimes gains).

## 2026-11-25 Which rows benefit from R3's GRPO (R3_ori) vs MW-R-H
Script/output: analysis/results/grpo_row_types/{analyze.py,output.txt} (Toys test n=6,738, copy 836).
- R3_ori stage1->stage2 (lr 5e-5): run1 +.0032 HR@10 (49 gained / 27 lost rows), run3 +.0009 (49/43), run2 collapsed (-.103). Top-1 changes on 16-24% of rows (MW-R-H lr1e-5: 7%).
- Run1 gain: 41% copy rows (+.0108), 59% non-copy, concentrated in high-train-frequency targets (+.0063; low/mid ~0). Run3: copy -.0072, non-copy +.0020 spread over tf low/mid. No consistent row type across seeds.
- R3_ori s2 gains are mostly rows MW-R-H start already hits: 39-49% of gained rows (slack rows recovered 10-11% vs 0.5% for others). GRPO moves R3_ori toward MW-R-H (.1055/.1091 ~ .1086), not past it.
- MW-R-H: lr1e-5 only 18 rows change status (12/6); lr5e-5 loses 301 (copy 165, tf-high non-copy 82), history slots .102->.067, brands 3.61->2.79.


## 2026-11-25 MW-R-H-S (swapped-pass CE) on CDs / Toys / Steam SUBMITTED

Code bd018410, recipe = MovieLens 375420 (COPY_SWAP_W=1, rank weights, margin 0.2, WIDER 0.1, 3 epochs, base Qwen2.5-1.5B, hady A40, default seed). Jobs: CDs 382418, Toys 382419, Steam 382420 (pending). Outputs <Dataset>_margin_0.2_wider_0.1_refweight_nolatent_rank_swapce. Exclude-history evals postponed.

UPDATE 2026-11-25: moved to researchshort at user request. hady jobs 382418/382419/382420 cancelled (pending, nothing written); resubmitted as CDs 382429, Toys 382430, Steam 382431 (research-1-qos, a40|l40s|a100|l40, 2 days, requeue).
UPDATE 2026-11-25: GPU list of 382429/382430/382431 widened in place to a40|l40s|a100|l40|h100|h100nvl (scontrol update by JobManager; same job IDs). Estimated starts 26 Nov 03:00 / 03:29 / 16:07 (CDs / Toys / Steam).


## 2026-11-26 CDs MW-R-H-S (swapped-pass CE, 382429) DONE — best ranking on CDs, but latent NOT used

Code bd018410, default seed, analog A100, 4h54m. n = 6,157 (418 copy rows). Normal decoding:
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| R3_ori | .0494 | .0710 | .0819 | .0989 | .0617 | .0662 | .0717 | .5766 | .0641 |
| MW-R-H 366023 | .0555 | .0801 | .0906 | .1121 | .0696 | .0739 | .0808 | .6244 | .0748 |
| MW-R-H-S 382429 | .0607 | .0866 | .1002 | .1215 | .0757 | .0813 | .0882 | .6196 | .0852 |
- S − R3_ori: all cutoffs sig (t-test and bootstrap): HR@1 +.0114, HR@3 +.0156, HR@5 +.0184, HR@10 +.0226, NDCG@10 +.0165 (p_t < 1e-6). copy +.0431, non-copy +.0211.
- S − MW-R-H: all cutoffs sig: HR@5 +.0096 (p_t 1.3e-4), HR@10 +.0094 (8e-4), NDCG@10 +.0074 (6e-5); copy −.0048, non-copy +.0105.
- Latent tests (raw − intervention, HR@10): mask +.0000, mean −.0010, shuffle −.0021, ALL n.s. (p .23–1) → latent content UNUSED. MW-R-H 366023 had shuffle −.017 (used). On CDs the ranking gain comes without latent use, so the goal (latent used AND ranking ≥ R3) is not met on this seed. Single seed.

## 2026-11-26 Exact R3 stage-2 GRPO on MW-R-H 365504, 5 seeds at lr 1e-5 (42 + 1..4; jobs 380284, 382413-382416, ranking only)
Script: analysis/results/grpo_row_types/seeds.py -> seeds_output.txt. HR@10 .1095/.1095/.1083/.1091/.1089 vs start .1086; NDCG@10 .0701/.0700/.0697/.0700/.0698 vs .0698.
No seed differs from the start at any cutoff (each p>.13); mean of 5 seeds HR@10 +.0004 [-.0005,+.0014], NDCG@10 +.0001 (n.s.). Seed spread .0012 HR@10. Conclusion: R3's GRPO at lr 1e-5 is a seed-independent no-op on MW-R-H; lr 5e-5 collapses. Middle lr not tried.

## 2026-11-26 Is lr the factor? weight drift of the pooling attention under R3 stage-2 GRPO (R3_ori vs MW-R-H)
Scripts: analysis/results/grpo_row_types/{weight_drift,drift_cos,drift_r3ori}.py (+ *_output.txt). Trained tensors = the 6 pooling-attention tensors (query/key/value W,b), ||W||≈54 in both models; decoder unchanged bit-for-bit. (First weight_drift.py used all 34 attention.* keys incl. frozen ones, so its "rel" values .1%/2% are understated; correct: below.)
- MW-R-H 365504: lr1e-5 ||Δ|| .20-.25 (0.4%), 5 runs mutually aligned cos .53-.63; lr5e-5 ||Δ|| 4.44 (8.2%, collapsed), cos with lr1e-5 update .06.
- R3_ori Toys run1 (same stage 1): lr1e-5 .55 (1.0%), 5e-5 2.15 (4.0%), 1e-4 4.41 (8.3%); cos(1e-5,5e-5) .16, (5e-5,1e-4) .52. lr5e-5: Toys run2 5.99 (collapsed), run3 3.57; CDs 4.41/1.86/3.95 (all fine, CDs run1 +.0106).
- Drift size alone does not separate gain from collapse (CDs run1 4.41 fine vs MW-R-H 4.44 collapsed): the decoder's sensitivity to r does.

CLEANUP 2026-11-26 (user OK): deleted latent/model.safetensors (and the cr checkpoint-1905) of MovieLens runs rank (369532), rank_cr, rank_motive, rank_motive_hn, rank_motive_w03, rank_hardneg, marginwarm, nc_refq1, nc_refq1_ep2 on scratch (~38 GB); results/configs/trainer_state kept. Kept: swapce, distill, R3 bases, Movielens_R3_no_latent_ref.


## 2026-11-26 Toys MW-R-H-S (swapped-pass CE, 382430) DONE — ranking above R3, latent essentially unused

Code bd018410, default seed, sodag A40, 9h31m. n = 6,738 (905 copy rows). Normal decoding:
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| plain R3 (A) | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .5127 | .0420 |
| MW-R-H 365504 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | .5039 | .0473 |
| MW-R-H s1 A40 368189 | .0404 | .0647 | .0804 | .1088 | .0544 | .0608 | .0700 | .5105 | .0465 |
| MW-R-H s2 A40 366001 | .0334 | .0574 | .0717 | .0944 | .0471 | .0530 | .0603 | .4630 | .0372 |
| MW-R-H-S 382430 | .0416 | .0706 | .0856 | .1095 | .0585 | .0648 | .0723 | .5050 | .0482 |
- S − R3 (A): HR@1 +.0049 (p_t .011), HR@3 +.0082 (4e-4), HR@5 +.0061 (.015), NDCG@3 +.0071, NDCG@5 +.0063, NDCG@10 +.0056 (7e-4) all sig (t and bootstrap); HR@10 +.0043 n.s. (.12/.11). copy −.0077, non-copy +.0062.
- S − MW-R-H 365504: all n.s. except NDCG@5 +.0033 (.041/.050); HR@10 +.0009. vs seed 1 (368189): HR@3 +.0059, HR@5 +.0052, NDCG@3/@5 sig, HR@10 n.s. vs failed seed 2 (366001): all cutoffs sig (HR@10 +.0151, NDCG@10 +.0120), copy +.0420.
- Latent tests (raw − intervention, HR@10): mask +.0006, mean +.0016, shuffle +.0027 all n.s.; NDCG@10 shuffle +.0032 sig (p .005/.006), mask/mean n.s. → latent barely used (MW-R-H 365504 shuffle −.035, seed 1 −.051). So like CDs, S ≥ R3 but the latent-use goal is not met on this seed. Single seed.


## 2026-11-26 Why MW-R-H-S ranks well on Toys / CDs without using the latent (training-log check, no new runs)

Per-epoch means from the run logs (margin_loss = weighted hinge, 0.2 = no gap, 0 = satisfied; copy_swap_ce = CE on the swapped-latent pass):
| run | hinge ep 0–1 | ep 1.5–2 | ep 2.5–3 | copy_swap_ce ep 2.5–3 | shuffle HR@10 |
|---|---|---|---|---|---|
| Toys MW-R-H 365504 | .200 | .088 | .020 | – | −.035 (used) |
| Toys S 382430 | .175 | .139 | .093 | .172 | +.003 n.s. |
| CDs S 382429 | .200 | .191 | .175 | .185 | −.002 n.s. |
| MovieLens S 375420 | .199 | .094 | .060 | .219 | −.038 (used) |
- In S the hinge (own latent must beat another user's latent by margin 0.2) and the swapped-pass CE (predict the target well with another user's latent, weight 1 − w) pull the same swapped-pass loss in opposite directions. On CDs and Toys the swapped CE falls to .17–.19 (the decoder predicts well with the WRONG latent) and the hinge stays largely unsatisfied (CDs .175, Toys .093 vs .020 in MW-R-H) → decoder learns to read the history and ignore the latent. On MovieLens the hinge is satisfied to .060 and the latent is used.
- Reading: the ranking gain on Toys/CDs comes from the decoder keeping/strengthening its history-only reading (S acts as a regulariser: swapped pass = history-only training signal), not from latent content. Correlational (logs); not tested causally.

DECISION 2026-11-26 (user): swapped-pass (MW-R-H-S) design does not align with the objective (on CDs/Toys it ranks well but the latent is unused, because the swapped-pass CE counteracts the hinge). Steam swapce 382431 cancelled via JobManager; CDs 382429 and Toys 382430 results kept; no further S runs.


## 2026-11-26 WIDER / hinge trajectories: Steam vs Toys vs CDs vs MovieLens (MW-R-H, training logs)

WIDER penalty = mean cos² between each user's pooled latent and the stop-gradient EMA shared direction across users (src/latent/wider_reg.py; 1 = all users' latents identical). Hinge: 0.2 = no own-vs-other gap.
| epochs | Steam 369662 | Toys 365504 | CDs 366023 | MovieLens 369532 |
|---|---|---|---|---|
| WIDER 0–.25 | .724 | .444 | .575 | .771 |
| .25–.5 | .957 | .067 | .236 | .490 |
| .5–1 | 1.000 | .051 | .074 | .096 |
| 1–1.5 | .995 | .068 | .108 | .171 |
| 1.5–2 | .991 | .149 | .352 | .250 |
| 2–2.5 | 1.000 | .087 | .230 | .259 |
| 2.5–3 | 1.000 | .069 | .162 | .132 |
| hinge 2.5–3 | .198 | .020 | .062 | .055 |
- Steam: latents collapse to one direction by epoch 0.5 and stay there (final mean cos to shared direction 1.0000, |ema| .9999); hinge stays .20 → no gap; shuffle n.s. Toys/CDs/MovieLens: WIDER drops to .05–.10 by epoch 1 (diverse latents), hinge falls to .02–.06 by epoch 3, latent used.
- Likely stuck fixed point (inference from the code): at z = q the WIDER gradient (I − zzᵀ)q/|r| is exactly zero, so once the Steam latents are identical the regulariser (weight 0.1) cannot pull them apart; the hinge cannot either, since swapping identical latents changes nothing.
- Steam is the only dataset whose latents collapse in the first half epoch; not yet checked why (Steam CE at start 2.8 vs ~1.8 Toys; items 4,419; text/titles).


## 2026-11-26 Steam MW-R-H first steps (log check, no new run): the latent collapses at steps ~150–200, not in the first 50

Source: run logs 369662 (Steam) and 365504 (Toys); 643 / 421 steps per epoch; lr warm-up to 3e-4 by step ~20.
- Steps 1–50: no collapse on Steam. WIDER .64 (Toys .59), hinge .24 → .20, grad_norm 203 → 5–10 (Toys 87 → 3–5). Both start with latents near one direction (mean cos to shared direction .91 Steam, .95 Toys at call 1).
- Divergence from step ~50: Toys WIDER .32 (51–100) → .065 (101–150) and stays .04–.09 until the hinge starts to be satisfied (steps 600+, hinge .14 → .03). Steam WIDER .73 → .76 → **.998 at steps 151–200** and ≈1.0 for most of the run; hinge ≈ .20 throughout.
- The collapse coincides with an instability on Steam: grad_norm rises 21 → 39 → 80 → 58 (steps 51–200; Toys 21 → ~6 in the same window), CE spikes 1.9 → 2.6 → **4.4** at steps 151–200 (Toys 1.9 → 1.2), then recovers to 1.7 by step 300 while WIDER stays ≈1.0.
- Reading (inference from logs): at peak lr 3e-4 a loss/gradient spike on Steam pushes the pooled latents onto one direction; the WIDER gradient is zero at z = q (see wider_reg.py) so it cannot undo it, and the hinge cannot either. Not yet shown which comes first (spike or collapse) or that lr is the cause.

## 2026-11-26 Steam collapse test SUBMITTED (code 66b61b81: LR and MAX_GRAD_NORM env)

Steam MW-R-H recipe, default seed, researchshort a40|l40s|a100|l40, exclude alarm,album. 383065 = clip1.0 (explicit MAX_GRAD_NORM=1.0, lr 3e-4; reproduction check of 369662), 383066 = lr 1e-4, 383067 = lr 5e-5. Question: does lowering the peak lr stop the WIDER collapse at steps ~117-200 (369662: WIDER 1.0, hinge .20, shuffle n.s.)? Scratch at 445G of the 485G guard; JobManager may hold unstarted jobs.


## 2026-11-26 Steam lr / clip runs: first ~410 of 1,929 steps (log check)

383065 (clip1.0, lr 3e-4), 383066 (lr 1e-4), 383067 (lr 5e-5) vs 369662 (default). Mean per step block:
| steps | WIDER 369662 / 383065 / 383066 / 383067 | CE (same order) | hinge (same order) |
|---|---|---|---|
| 101–150 | .761 / .072 / .508 / .432 | 2.55 / 1.85 / 1.65 / 1.69 | .201 / .197 / .198 / .191 |
| 151–200 | **.998** / .062 / .774 / .360 | **4.36** / 1.61 / 1.44 / 1.45 | .198 / .200 / .209 / .197 |
| 251–300 | .928 / .061 / .148 / .252 | 1.83 / 1.52 / 1.38 / 1.37 | .201 / .199 / .198 / .173 |
| 301–400 | .967 / .058 / .043 / .607 | 1.71 / 1.40 / 1.28 / 1.29 | .203 / .201 / .202 / .204 |
- The explicit-clip-1.0 run at the SAME lr 3e-4 does NOT collapse (WIDER .06 from step ~100, no CE spike, grad norm 4 → 2): the collapse of 369662 is therefore not reproduced by the same settings → likely a run-to-run (nondeterministic) instability, not the lr by itself (consistent with Toys seed 2 / seed-to-seed variance). Caveat: one run each, 410/1929 steps; a clip default other than 1.0 in 369662 is not excluded (not verified in the installed transformers).
- lr 1e-4: a transient WIDER bump (.77 at 151–200, grad norm 48) that recovers to .04; lr 5e-5: WIDER stays higher (.25–.61), slower to separate.
- Hinge still ≈ .20 in all (Toys' hinge only started falling after ~1.4 epochs = step ~600 of 1,263); the hinge/latent-use outcome needs the full runs.

## 2026-11-26 SUMMARY: attempts to improve MW-R-H (Toys, start 365504: HR@10 .1086 / NDCG@10 .0698, latent used: shuffle −.035) with GRPO
Goal: ranking ≥ start while keeping latent use. Test n=6,738 (copy rows 836 = 12.4%). Paired bootstrap + paired t over all rows. "R3's GRPO" = original R3_ori stage-2 code (noise N(0,1.5²) on r, G=8, reward −PPL, only the pooling attention trains).

| # | attempt | HR@10 / NDCG@10 | vs start | latent | verdict |
|---|---|---|---|---|---|
| 1 | beam-only rank GRPO 368198 (own design, no noise; dropped as not R3's GRPO, code removed 2026-11-24) | .1089 / – | no-op | used | 13% top-1 changed, reward starved on memorised train rows |
| 2 | rank GRPO + A (target injection into no-hit groups, G=16) | .1043 | −.0043 sig (p .003) | used | worse |
| 3 | rank GRPO + C (history −.5, redundancy −.25 penalties) | .0959 | −.0127; copy rows −.068 | – | much worse |
| 4 | rank GRPO + A+C | .1033 | worse (copy −.031) | – | worse |
| 5 | HiLaR-style (arXiv 2607.27760) single-latent port: Gaussian policy on r (σ rel. to inter-user dist.), antithetic rollouts G=12, hit/prefix/F1/SASRec/dLL reward, latent KL β .05 | .1077 / .0683 | HR@10 = start; NDCG@3/5/10 slightly lower sig | used (shuffle −.034) | stable but no gain |
| 6 | R3 exact stage 2, lr 5e-5 (380283) | .0701 / .0415 | −.0386 sig, all cutoffs | destroyed (mask +.009) | collapse, grad-norm .03→4.8 |
| 7 | R3 exact, lr 1e-5, 5 seeds (42,1,2,3,4) | .1095/.1095/.1083/.1091/.1089; NDCG@10 .0697–.0701 | mean +.0004 [−.0005,+.0014], p .33; no seed sig | unchanged | seed-independent no-op (93% top-1 unchanged) |
| 8 | R3 exact, lr 2e-5 (383021) | .0950 / .0544 | −.0137 [−.0175,−.0099], NDCG@10 −.0153 | – | already damaged |
| 9 | R3 exact, lr 3e-5 (383022) | .0473 / .0240 | −.0613 | – | collapse (worse than 5e-5) |
| 10 | R3 exact, lr 4e-5 (383023) | pending (restarted after preemption) | | | |

Diagnostics:
- Why rank GRPO fails: MW-R-H already sits at the non-copy level rank GRPO gives other models; beam-only GRPO barely moves the model; reward starved on memorised train rows (SASRec in-sample: train hit .835 vs test .100). Test A (gradient alignment) did not explain it.
- Does R3's GRPO help R3_ori? Small, inconsistent: Toys +.0033 sig / collapse / +.0009 n.s.; CDs +.0106 sig / +.0002 / −.0016. Its gain is below R3_ori's own stage-1 seed spread (HR@10 .1023/.1051/.1082) and 39–49% of gained rows are rows MW-R-H already hits (moves R3_ori toward MW-R-H level, not past it). Row types (analysis/results/grpo_row_types/): no consistent row type across runs; run1 gain 41% copy + high-frequency non-copy targets, run3 non-copy low/mid. User's reading: gain is run noise; improvement of GRPO on R3_ori is not established.
- Weight drift of the 6 pooling-attention tensors (decoder bit-identical): MW-R-H lr1e-5 ||Δ|| .20–.25 (0.4%), updates mutually aligned (cos .53–.63) but too small to change outputs; lr5e-5 4.44 (8.2%, cos .06 with the 1e-5 direction). R3_ori Toys run1 1e-5 .55 / 5e-5 2.15 / 1e-4 4.41; CDs 5e-5 1.9–4.4. Drift size does not separate gain from collapse (CDs run1 4.41 fine vs MW-R-H 4.44 collapsed): MW-R-H's decoder reads r, so noise on r + shifted attention changes its outputs; R3_ori's decoder mostly ignores r.
- Grad norm (mean over run): lr1e-5 .07; 2e-5 .38; 3e-5 .24; 5e-5 .93 → not monotone with outcome.
- Interventions lr1e-5 (seed 42): mask −.030, mean −.030, shuffle −.033 = start (−.0297/−.0294/−.0343); lr5e-5: mean −.010, shuffle −.007, mask +.009 (identical mask rows across runs are expected: mask zeroes r, decoder frozen).

Conclusion: nothing met the goal. R3's GRPO on MW-R-H has a no-op regime (lr ≤1e-5, any seed) and a destructive regime (lr ≥2e-5); no useful lr in between (4e-5 pending). Own rank-GRPO variants and the HiLaR-style port are no-ops or worse. The only no-loss outcomes are runs where the model barely moves.
Open (not run): extrapolating along the consistent lr1e-5 update direction (start + α·Δ), seeds at lr 2e-5.
Files: analysis/results/{rank_grpo_AC,r3_exact_mwrh,r3_ori_stage2_check,grpo_row_types}/; checkpoints in scratch Toys_and_Games_R3exact_*, *_R3_grpo_* (deletion pending user approval).

### 2026-11-26 Approach descriptions and full results (replaces the short table in the summary entry above)
Start model: MW-R-H 365504 (Toys, single latent token, margin+WIDER+reference-weighted hinge). Test n=6,738. All approaches train only the pooling attention that produces r; decoder frozen.
1. **Beam-only rank GRPO (368198, own design; dropped, code removed 2026-11-24).** One unperturbed r per row; the group is G beams from beam search; reward by rank of the target among the beams (hit/rank penalty); KL anchor to the start attention. No noise on r (so rollouts differ only in the beam), which is not what R3's GRPO does.
2. **Rank GRPO + A (target injection).** Same as 1, but when no beam hits the target, the target is injected into the group as a hindsight sample (G=16), to avoid reward-starved groups.
3. **Rank GRPO + C (penalties).** Same as 1 plus penalties: a beam that equals a history item gets -0.5, redundant beams (near-duplicates of another beam) get -0.25.
4. **Rank GRPO + A+C.** Both.
5. **HiLaR-style (arXiv 2607.27760) single-latent port.** Policy N(mu(history), sigma^2 I) on the single r (sigma = sigma_rel*D/sqrt(d), D = median inter-user distance of r, sigma_rel 1); antithetic +-eps rollouts, G=12, 8 users/update, lr 5e-6; dense reward = hit + 0.5 prefix + 0.5 F1 + 0.1 SASRec z-score (clip +-3) + target-likelihood gain (clip +-1); loss = -mean A*log N(r_g; mu_theta, sigma^2) + beta*||mu_theta-mu_ref||^2/(2 sigma^2), beta .05 latent KL to the start attention.
6. **R3 exact stage 2 (original R3_ori code c558b44, lr 1e-5), 5 seeds (42,1,2,3,4).** Noise N(0,1.5^2) on r at the thought slot, 8 copies per row (first copy of the micro-batch noise-free), reward = -PPL of the target, advantage normalised by the noise-free mean, clipped surrogate (inactive, 1 iteration), beta 0; batch 256, micro 8, 1 epoch, 2 GPUs. Seed changes row order and noise only; start model identical.
7. **R3 exact, lr search 2e-5 / 3e-5 / 4e-5 (seed 42).** Same recipe, higher lr.
8. **R3 exact, lr 5e-5 (seed 42).** R3's own stage-2 lr.
Reference rows: R3_ori stage 1 and its own stage 2 (lr 5e-5), Toys run 1.

Full results (paired bootstrap p / paired t-test p vs start in brackets; script analysis/results/grpo_row_types/full_table.py):

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | ΔHR@10 vs start (p bootstrap / p t) | ΔNDCG@10 vs start (p b / p t) |
|---|---|---|---|---|---|---|---|---|---|---|
| start MW-R-H 365504 | 0.0383 | 0.0681 | 0.0828 | 0.1086 | 0.0383 | 0.0554 | 0.0615 | 0.0698 | — | — |
| 1 beam-only rank GRPO 368198 | 0.0381 | 0.0681 | 0.0827 | 0.1089 | 0.0381 | 0.0555 | 0.0615 | 0.0700 | +0.0003 (0.800 / 0.768) | +0.0002 (0.647 / 0.656) |
| 2 rank GRPO + A | 0.0370 | 0.0652 | 0.0795 | 0.1043 | 0.0370 | 0.0534 | 0.0594 | 0.0674 | -0.0043 (0.002 / 0.003) | -0.0024 (0.001 / 0.002) |
| 3 rank GRPO + C | 0.0315 | 0.0586 | 0.0723 | 0.0959 | 0.0315 | 0.0471 | 0.0527 | 0.0603 | -0.0128 (0.000 / 0.000) | -0.0095 (0.000 / 0.000) |
| 4 rank GRPO + A+C | 0.0356 | 0.0631 | 0.0803 | 0.1033 | 0.0356 | 0.0516 | 0.0587 | 0.0660 | -0.0053 (0.000 / 0.001) | -0.0037 (0.000 / 0.000) |
| 5 HiLaR-style | 0.0365 | 0.0650 | 0.0812 | 0.1077 | 0.0365 | 0.0531 | 0.0597 | 0.0683 | -0.0009 (0.500 / 0.453) | -0.0015 (0.020 / 0.020) |
| 6 R3-exact lr1e-5 seed42 | 0.0380 | 0.0689 | 0.0828 | 0.1095 | 0.0380 | 0.0557 | 0.0614 | 0.0701 | +0.0009 (0.170 / 0.157) | +0.0003 (0.380 / 0.412) |
| 6 R3-exact lr1e-5 seed1 | 0.0380 | 0.0678 | 0.0828 | 0.1095 | 0.0380 | 0.0552 | 0.0614 | 0.0700 | +0.0009 (0.147 / 0.134) | +0.0002 (0.494 / 0.527) |
| 6 R3-exact lr1e-5 seed2 | 0.0378 | 0.0681 | 0.0831 | 0.1083 | 0.0378 | 0.0553 | 0.0615 | 0.0697 | -0.0003 (0.782 / 0.655) | -0.0001 (0.733 / 0.710) |
| 6 R3-exact lr1e-5 seed3 | 0.0381 | 0.0680 | 0.0831 | 0.1091 | 0.0381 | 0.0554 | 0.0616 | 0.0700 | +0.0004 (0.543 / 0.491) | +0.0002 (0.456 / 0.492) |
| 6 R3-exact lr1e-5 seed4 | 0.0380 | 0.0677 | 0.0830 | 0.1089 | 0.0380 | 0.0551 | 0.0614 | 0.0698 | +0.0003 (0.711 / 0.655) | -0.0000 (0.953 / 0.927) |
| 7 R3-exact lr2e-5 seed42 | 0.0230 | 0.0493 | 0.0668 | 0.0950 | 0.0230 | 0.0382 | 0.0454 | 0.0544 | -0.0137 (0.000 / 0.000) | -0.0153 (0.000 / 0.000) |
| 7 R3-exact lr3e-5 seed42 | 0.0071 | 0.0191 | 0.0301 | 0.0473 | 0.0071 | 0.0139 | 0.0184 | 0.0240 | -0.0613 (0.000 / 0.000) | -0.0458 (0.000 / 0.000) |
| 7 R3-exact lr4e-5 seed42 | (pending) |
| 8 R3-exact lr5e-5 seed42 | 0.0197 | 0.0383 | 0.0484 | 0.0701 | 0.0197 | 0.0303 | 0.0345 | 0.0415 | -0.0386 (0.000 / 0.000) | -0.0283 (0.000 / 0.000) |
| ref R3_ori stage 1 (run1) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0377 | 0.0510 | 0.0580 | 0.0659 | -0.0064 (0.027 / 0.028) | -0.0039 (0.019 / 0.025) |
| ref R3_ori stage 2 lr5e-5 (run1) | 0.0380 | 0.0628 | 0.0784 | 0.1055 | 0.0380 | 0.0522 | 0.0586 | 0.0674 | -0.0031 (0.278 / 0.276) | -0.0024 (0.161 / 0.169) |

Notes: 4 is pending (lr 4e-5, 383023, restarted after a preemption). HiLaR-style: HR@10 equal to the start (p .50) but NDCG@3/5/10 slightly lower, NDCG@10 significant (p .02).

### 2026-11-26 lr 4e-5 (383023) finished — lr search complete
See row "7 R3-exact lr4e-5" in analysis/results/grpo_row_types/full_table.md: HR@1/3/5/10 .0255/.0481/.0614/.0841, NDCG@1/3/5/10 .0255/.0385/.0440/.0513. Search (seed 42, one run per lr): HR@10 2e-5 .0950, 3e-5 .0473, 4e-5 .0841, 5e-5 .0701 — all far below start .1086; non-monotone (3e-5 lowest), no trend readable without seeds. No lr between 1e-5 (no-op) and 5e-5 helps.


## 2026-11-26 15:30 Steam lr / clip runs: progress at epoch ~2.45 of 3 (after preemption + resume)

Preempted on lagoon at step 1242 (13:03); resumed from checkpoint-643 on lexicon (13:05, resume line present in all three logs; requeue overwrote the log files, so epochs 0–1 detail is only in the earlier log section). Means per quarter epoch:
| epochs | WIDER 369662 / 383065 clip1.0 / 383066 lr1e-4 / 383067 lr5e-5 | hinge (same order) |
|---|---|---|
| 1.0–1.25 | 1.000 / .035 / .023 / .129 | .201 / .200 / .199 / .161 |
| 1.5–1.75 | .982 / .023 / .017 / .120 | .199 / .200 / .194 / .044 |
| 2.0–2.25 | 1.000 / .017 / .059 / .107 | .200 / .197 / .175 / .026 |
| 2.25–2.5 | 1.000 / .109 / .106 / .086 | .201 / .177 / .162 / .021 |
- No collapse in any new run (WIDER ≤ .16 throughout epochs 1–2.5, the resume did not disturb it).
- lr 5e-5: the hinge is satisfied (.161 → .021 from epoch 1.25) — the first Steam MW-R-H run where the own latent beats another user's. clip1.0 (3e-4) and lr 1e-4: hinge only starting to fall at epoch 2+ (.177 / .162).
- CE at epoch 2.25–2.5: 369662 .944, clip1.0 .892, lr1e-4 1.000, lr5e-5 1.077 (lower lr → slower CE fit).


## 2026-11-26 Steam MW-R-H clip1.0 (383065, lr 3e-4, MAX_GRAD_NORM=1.0 explicit, code 66b61b81) DONE

n = 10,303. Paired t-test p / bootstrap p (2,000 resamples, seed 42), script tmp/steam_lr_eval.py.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| R3 (371447) | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 |
| MW-R-H (369662, collapsed) | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 |
| MW-R-H clip1.0 (383065) | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 |
- vs R3: HR@1 +.0022 (.025/.027), NDCG@3 +.0021 (.040/.038), NDCG@5 +.0020 (.047/.042) sig; HR@3 +.0018 (.14), HR@5 +.0016 (.28), HR@10 +.0005 (.78), NDCG@10 +.0017 (.096/.090) n.s.
- vs 369662: all n.s. (HR@10 −.0025, p .15/.17; NDCG@10 −.0014, p .18/.20).
- Latent tests (raw − intervention), all sig p < 5e-4: mask HR@10 +.0292 / NDCG@10 +.0172; mean +.0116 / +.0068; shuffle_user +.0094 / +.0050 (@5: mask +.0201/+.0143, mean +.0074/+.0055, shuffle +.0048/+.0035).
- **First Steam run whose latent CONTENT is used** (mean and shuffle hurt sig; 369662: mean/shuffle n.s.), ranking ≥ R3 at all cutoffs (sig at HR@1, NDCG@3/5) and n.s. below the collapsed 369662. Hinge in training only reached ~.17 by epoch 2.5, yet content use transfers to test. Meets the criterion (content matters AND ≈/≥ R3). Single run; the collapse in 369662 vs none here shows run-to-run variance at identical settings.


## 2026-11-26 Steam MW-R-H lr 5e-5 (383067, clip 1.0, code 66b61b81) DONE

n = 10,303; paired t p / bootstrap p (2,000, seed 42); tmp/steam_lr_eval.py _lr5e-5.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| R3 (371447) | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 |
| MW-R-H (369662, collapsed) | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 |
| MW-R-H clip1.0 3e-4 (383065) | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 |
| MW-R-H lr 5e-5 (383067) | .0102 | .0193 | .0248 | .0360 | .0154 | .0177 | .0213 |
- vs R3: sig at HR@1 +.0025 (.0086/.012), HR@3 +.0042 (.00055/.001), NDCG@3 +.0035, NDCG@5 +.0027 (.0047/.008), NDCG@10 +.0021 (.024/.032); HR@5 +.0022 (.10) and HR@10 +.0002 (.91) n.s.
- vs 369662: all n.s. (HR@10 −.0028, p .098/.11; NDCG@10 −.0010, p .34/.36).
- Latent tests (raw − intervention), all sig: @10 mask +.0066 / +.0067, mean +.0034 (p .007) / +.0043, shuffle_user +.0070 / +.0062; @5 mask +.0081/+.0071, mean +.0056/+.0050, shuffle +.0077/+.0064.
- Content used (shuffle hurts sig) and ranking ≥ R3 with more sig cutoffs than 383065 (NDCG@5 and NDCG@10 sig). Different profile from 383065: much smaller mask drop (HR@10 −.0066 vs −.0292) → the decoder depends less on having some vector in the slot; shuffle drop similar in size at @10 (−.0070 vs −.0094) and larger at @5 (−.0077 vs −.0048); hinge was satisfied in training (.021).


## 2026-11-26 MW-R-H-CR on Toys, CDs, Steam — submitted via JobManager (tmp/submit_mwrh_cr_3ds.sh)

User: run MW-R-H + original collaborative-reasoning InfoNCE on Toys, CDs, Steam; the InfoNCE must be the same as margin+WIDER+InfoNCE (MWC-I 358547).
Implementation check (git diff b3740008 = MWC-I code → HEAD 66b61b81):
- InfoNCE path unchanged: same trainer `latent_attention_train_collaborative_margin.py`, same `_relation_loss`; two-phase (epoch 1 margin+WIDER only, InfoNCE epochs 2–3), user head (same user's other rows) 0.1 + target head (same next item) 0.1, projection 256, temp 0.1, FIFO queue 10k, 16 positives per head, 256 negatives, token-accuracy gate 0.5. Code added since (same-user hard negatives, item head, CF grounding) is off by default (HARD_NEG=0, ITEM_W=0).
- Intended differences (MW-R-H base, commit c3b42206): hinge weighted per row by the history-only reference (rank, tau 0.5, sum(w·hinge)/sum(w)); CE = mean per-row NLL; EARLY_STOP=False (MW-R-H has no best-checkpoint reload; MWC-I used early stopping). Optimiser/schedule identical (lr 3e-4, warm-up 20, HF clip 1.0, save per epoch, 1 checkpoint).
Runs (code_66b61b81, base Qwen2.5-1.5B, margin 0.2, WIDER 0.1, 3 epochs, lr 3e-4 on all three incl. Steam; researchshort a40|l40s|a100|l40, excl. alarm,album): Toys_and_Games / CDs_and_Vinyl / Steam `_margin_0.2_wider_0.1_refweight_nolatent_rank_cr`, references `reference_nll/<cat>_nolatent/train.pt`. Comparisons: R3 (Toys A / CDs R3_ori / Steam 371447) and MW-R-H (365504 / 366023 / 383065).
- Job IDs (JobManager, 2026-11-26): Toys **383923**, CDs **383924**, Steam **383925**. Scratch 429G / 485G at submission; one job may queue (JobManager's researchshort cap 8).


## 2026-11-26 Steam MW-R-H lr 1e-4 (383066, clip 1.0, code 66b61b81) DONE — Steam collapse test complete

n = 10,303; paired t p / bootstrap p (2,000, seed 42); tmp/steam_lr_eval.py _lr1e-4.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | shuffle HR@10 drop | mean HR@10 drop |
|---|---|---|---|---|---|---|---|---|---|
| R3 (371447) | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 | | |
| MW-R-H 3e-4 (369662, collapsed) | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 | n.s. | n.s. |
| MW-R-H clip1.0 3e-4 (383065) | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 | .0094 sig | .0116 sig |
| MW-R-H lr 1e-4 (383066) | .0116 | .0231 | .0302 | .0425 | .0182 | .0211 | .0251 | .0026 n.s. | −.0009 n.s. |
| MW-R-H lr 5e-5 (383067) | .0102 | .0193 | .0248 | .0360 | .0154 | .0177 | .0213 | .0070 sig | .0034 sig |
- lr 1e-4 vs R3: ALL cutoffs sig (p < 5e-4): HR@5 +.0076, HR@10 +.0067, NDCG@5 +.0061, NDCG@10 +.0059. vs 369662: sig at @3/@5/@10 (HR@10 +.0037, .039/.047; NDCG@10 +.0028, .0068/.008).
- Latent tests: mask @10 HR +.0079 / NDCG +.0036 sig; mean @10 −.0009 / −.0004 n.s. (p .50/.54); shuffle_user @10 +.0026 (.068/.084) / +.0008 n.s. → content NOT used (only slot presence).
- Training: hinge only fell to ~.16 by epoch 2.5 (not satisfied), WIDER ≤ .11 (no collapse).
- Reading: the best Steam ranking so far comes WITHOUT content use (regulariser effect, as in 369662); the two runs that use content (383065, 383067) rank ≈ R3 at HR@10. On Steam ranking and content use still trade off; lr 5e-5 is the only run with both content use and NDCG@10 sig > R3.
- 2026-11-26: Steam 383925 FAILED at start (KeyError 'Steam': collab-margin trainer's category_dict only had lowercase 'steam'; earlier Steam MW-R-H runs used the motive trainer, whose CATEGORY_TEXT has 'Steam'). Fix commit 42e0da0a adds 'Steam': 'games' (+ 'Goodreads'), prompt text identical to the motive trainer for all shared keys (tmp/test_category_text.py PASS; old code fails). Frozen code_42e0da0a (diff -r vs 66b61b81: that line only). Resubmit via JobManager: tmp/submit_mwrh_cr_steam_fix.sh. CDs 383924 preempted at ~1h (no checkpoint yet), requeued from step 0.
- Steam CR resubmitted to hady (user): **384089** (hadylauwresearch, hady-priority, a40, code_42e0da0a; tmp/submit_mwrh_cr_steam_fix_hady.sh); researchshort retry stopped before any submission.


## 2026-11-27 MW-R-H-CR training progress (tmp/cr_progress.py; means per half epoch)

CDs 383924 restarted on avenue from step 0 (no checkpoint before the preemption). Toys 383923 at epoch 2.65 (sodag), Steam 384089 at epoch 1.65 (alarm).
| run | epochs | CE | hinge | WIDER | user InfoNCE | target InfoNCE | user / target positive rate |
|---|---|---|---|---|---|---|---|
| Toys 383923 | 0.5–1.0 | .878 | .146 | .287 | – | – | – |
| | 1.0–1.5 | .672 | .073 | .154 | 1.874 | 4.428 | .63 / .80 |
| | 2.0–2.5 | .474 | .023 | .078 | .466 | 3.938 | .61 / .83 |
| | 2.5–3.0 | .456 | .024 | .047 | .391 | 3.815 | .65 / .84 |
| Steam 384089 | 0.0–0.5 | 1.933 | .211 | .567 | – | – | – |
| | 0.5–1.0 | 1.351 | .200 | **.724** | – | – | – |
| | 1.0–1.5 | 1.256 | .199 | .157 | 3.571 | 4.164 | .66 / .88 |
| | 1.5–2.0 | 1.210 | .198 | .093 | 2.430 | 4.226 | .68 / .87 |
- Toys: hinge satisfied (.023 from epoch 2, as MW-R-H 365504). As in MWC-I 358547, the user head is what is learned (1.87 → .39) while the target head stays near chance (~4.5 → 3.8).
- Steam: partial collapse in epoch 1 (WIDER .72, grad norm 60 early; margin+WIDER only, same recipe as 369662), recovered to .09–.16 once the InfoNCE switched on. Hinge NOT satisfied (.198) at 1.65 — as in the MW-R-H lr 3e-4 runs.


## 2026-11-27 MW-R-H-CR Toys (383923, code 66b61b81) DONE

n = 6,738 (copy_row 905, same-brand rule); paired bootstrap 95% CI + t-test p / bootstrap p; tmp/toys_cr.py.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| plain R3 (A) | .0367 | .0625 | .0795 | .1052 | .0514 | .0584 | .0667 | .5127 | .0420 |
| MW-R-H 365504 | .0383 | .0681 | .0828 | .1086 | .0554 | .0615 | .0698 | .5039 | .0473 |
| MW-R-H s1 A40 368189 | .0404 | .0647 | .0804 | .1088 | .0544 | .0608 | .0700 | .5105 | .0465 |
| MW-R-H s2 A40 366001 | .0334 | .0574 | .0717 | .0944 | .0471 | .0530 | .0603 | .4630 | .0372 |
| MWC-I 358547 (margin+WIDER+InfoNCE) | | | | .0966 | | | .0600 | | |
| MW-R-H-CR 383923 | .0337 | .0637 | .0806 | .1076 | .0509 | .0578 | .0665 | .5061 | .0458 |
- vs R3 (A): all n.s. (HR@1 −.0030 p .12; HR@5 +.0010 p .68/.70; HR@10 +.0024 p .39/.40; NDCG@5 −.0006 p .72; NDCG@10 −.0002 p .90/.94); copy −.0066, non-copy +.0038.
- vs MW-R-H 365504: sig worse at HR@1 −.0046 (.008/.006), HR@3 −.0045 (.032/.036), NDCG@3 −.0045, NDCG@5 −.0036 (.018/.024), NDCG@10 −.0033 (.032/.038); HR@5 −.0022 (.33) and HR@10 −.0010 (.70) n.s. vs seed 1 368189: NDCG@10 −.0034 sig, HR@10 n.s. vs the failed seed 2 366001: better at all @3–@10 (sig).
- Latent tests (raw − intervention), all p < 5e-4: mask HR@10 +.0793 / NDCG@10 +.0505; mean +.0706 / +.0469; shuffle_user +.0793 / +.0516 (−74% of raw HR@10; copy rows +.35, non-copy +.037). MW-R-H 365504 shuffle was −.035 → CR more than doubles the content dependence (= MWC-I's level, −.075).
- Reading: meets the criterion (content strongly used AND ranking = R3 at every cutoff). On the MW-R-H base the original InfoNCE no longer costs ranking vs R3 (MWC-I on the margin+WIDER base: HR@10 .0966, −.0086 sig). But it is below MW-R-H 365504 at the top of the list and NDCG (−.003 to −.005 sig), so the InfoNCE buys dependence at a small accuracy cost vs MW-R-H. Training: hinge satisfied (.023), user InfoNCE learned (1.87 → .39), target InfoNCE near chance (4.4 → 3.8), as in MWC-I.


## 2026-11-27 MW-R-H-CR Steam (384089, code 42e0da0a, hady A40 alarm, lr 3e-4) DONE — content NOT used

n = 10,303; t p / bootstrap p; tmp/steam_lr_eval.py _cr, tmp/steam_cr_vs_clip.py _cr.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| R3 (371447) | .0077 | .0151 | .0226 | .0358 | .0119 | .0150 | .0192 |
| MW-R-H 369662 (collapsed) | .0100 | .0193 | .0262 | .0388 | .0154 | .0182 | .0223 |
| MW-R-H clip1.0 383065 (same settings) | .0099 | .0170 | .0242 | .0363 | .0140 | .0170 | .0209 |
| MW-R-H-CR 384089 | .0071 | .0165 | .0235 | .0361 | .0125 | .0154 | .0195 |
- vs R3: all n.s. (HR@5 +.0009 p .50/.52; HR@10 +.0003 p .86/.90; NDCG@10 +.0003 p .76/.74).
- vs 383065: HR@1 −.0028 sig (.0055/.006); rest n.s. (NDCG@10 −.0014, .17/.18). vs 369662: sig worse at HR@1/3, NDCG@3/5/10 (NDCG@10 −.0028, .0025/.005).
- Latent tests: none sig, all within ±.0007 (mask HR@10 +.0003 p .77; mean +.0000; shuffle_user +.0007 p .35/.41) → content AND slot unused (even mask is neutral; 383065: mask −.029, shuffle −.009).
- Training: hinge NEVER satisfied (.20 all six half-epochs); WIDER partial collapse in epoch 1 (.72), down to .05 by the end; user InfoNCE learned (3.57 → 1.44), target InfoNCE at chance (4.16 → 4.20); grad norm 60 early, 13 at epoch 1.5–2.
- Reading: fails the criterion (= R3, latent ignored). On Steam the InfoNCE neither rescued the hinge nor made r used; the slot became fully ignorable (mask neutral), unlike every MW-R-H Steam run. lr 3e-4 on Steam remains unstable (epoch-1 WIDER rise); a CR run at lr 5e-5 (where the MW-R-H hinge was satisfied) is the untested option.


## 2026-11-27 MW-R-H-CR CDs (383924, code 66b61b81, avenue after one preemption) DONE — FAILS; 3-dataset summary

n = 6,157 (copy_row 418); bootstrap CI + t p / bootstrap p; tmp/cds_cr.py.
| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 | non-copy HR@10 |
|---|---|---|---|---|---|---|---|---|---|
| R3_ori | .0494 | .0710 | .0819 | .0989 | .0617 | .0662 | .0717 | .5766 | .0641 |
| MW-R-H 366023 | .0555 | .0801 | .0906 | .1121 | .0696 | .0739 | .0808 | .6244 | .0748 |
| MW-R-H-CR 383924 | .0422 | .0594 | .0695 | .0884 | .0522 | .0564 | .0625 | .5574 | .0542 |
- vs R3_ori: sig WORSE at every cutoff (HR@5 −.0123, HR@10 −.0106 (.0006/.003), NDCG@5 −.0098, NDCG@10 −.0092 (<5e-4)); copy −.019, non-copy −.010.
- vs MW-R-H 366023: sig worse at every cutoff (HR@10 −.0237, NDCG@10 −.0183, p < 5e-4).
- Latent tests (raw − intervention) @10: mask −.0068 HR / −.0058 NDCG (mask significantly BETTER than raw; copy rows −.062); mean +.0122 / +.0106 sig; shuffle_user +.0060 / +.0052 sig — the content effect is almost all on copy rows (mean +.148, shuffle +.086; non-copy +.002 / +.000). The latent hurts on average (zero slot better than own r).
- Training: hinge NEVER satisfied (.194–.203 all epochs) vs MW-R-H 366023 falling to .077/.062 in epoch 2–3; CE ends .874 vs .662; user InfoNCE learned (3.54 → .70), target at chance (4.4 → 4.0). The InfoNCE (user head) displaces the hinge and slows CE fitting.

**3-dataset summary, MW-R-H-CR (original user+target InfoNCE on MW-R-H, lr 3e-4):**
| dataset | ranking vs R3 | vs MW-R-H | latent content | hinge in training | criterion |
|---|---|---|---|---|---|
| MovieLens 376544 | = @1–@5, HR@10 −.0070 sig | better | used (shuffle −.051) | satisfied (.003) | no (HR@10) |
| Toys 383923 | = all cutoffs | NDCG −.003 sig | strongly used (shuffle −.079) | satisfied (.023) | YES |
| CDs 383924 | WORSE all cutoffs (HR@10 −.011) | −.024 sig | mixed (mask > raw) | NOT satisfied (.20) | no |
| Steam 384089 | = all cutoffs | HR@1 −.003 sig (vs 383065) | unused (mask neutral) | NOT satisfied (.20) | no |
Reading: CR works only where the hinge is satisfied (Toys, MovieLens). On CDs and Steam the user-InfoNCE head (the only one that learns; the target head stays at chance on all three) competes with the hinge, which never forms; on CDs that costs ranking as in MWC-I on Toys. MW-R-H alone remains better or equal on every dataset at ranking. Not seed-replicated.

### 2026-11-27: Beam-NDCG reward, full-model GRPO on MW-R-H (Toys): smoke passed, 3 lr runs submitted

- **Smoke 383569** (code 40087bf9, lagoon 2× L40): completed in 6 min with no OOM.
  - Speed: 15.3 s/step at 4 rows/step, about 3.8 s per row. `BATCH` counts generations (8 per row), not rows, so the earlier estimate of 16k rows ≈ 7 h was wrong; 16k rows would take about 17 h.
  - Reward: 6 of the 16 steps had no hit in any group.
- **Bug in the stock R3 advantage under a hit/miss reward.** The baseline is the mean noise-free reward over all groups in the step (one group per GPU). A row whose 8 copies all miss therefore gets a negative advantage whenever the other row hits, and GRPO lowers that row's target log-prob. This explains the non-zero grad_norm on steps 8, 12 and 15, where the logged reward was 0.
- **Fix (user-approved).** `adv_baseline=group`: advantage = reward − the same row's noise-free reward. Ties, including all-miss groups, give 0, so only the effect of the noise on the beam list is rewarded.
- **Checkpoint fix.** The smoke checkpoint was 24 GB, including a 12 GB optimizer.pt. Now `save_only_model` is set and `checkpoint-*` is removed after training, leaving one ~6 GB final model. The smoke model was deleted.
- **Code:** commits b0e806ec and 071e729c; frozen copy at `output_dir/r3_beam_grpo_code/code_071e729c`, sbatch `r3_beam_grpo_071e729c.sbatch`.
- **Runs:**
  - Setup: 8,000 train rows, batch 64 (8 rows/step, 1,000 steps), warmup 25, β 0, 1 iteration, full model in fp32 with bf16 autocast, seed 42.
  - lr 3e-7 / 1e-6 / 3e-6, plus an fp32 eval of the start model (`toys_mwrh_start_fp32`) as the comparison baseline.
  - Submitted via JobManager: start eval 385004; lr3e-7 385005, lr1e-6 385006, lr3e-6 385007. The three lr runs were held for scratch space and released after the user approved deleting ~47 GB of finished exact-R3 and HiLaR checkpoints (scratch 473G → 426G).


## 2026-11-27 Paper Report folder created (no runs)

Author: the line is now a paper with C1 = MW-R-H and C2 = collaborative reasoning. Writing material lives in worktree `R3MarginWiderCollab/Report/`: main.md (story, method, experiment table E00–E09, open issues, evidence gaps, proposed runs), CONTEXT.md, CONVENTIONS.md, experiments/00–09/report_*.md (compiled from this log, the older log and the analysis docs; numbers spot-checked against this log). Corrections found while compiling: MovieLens test n = 10,164; Toys copy_row 905 (trainer rule) vs 836 (compare.py); REF_TAU has no effect with rank weights; the "R3_ori Toys" interventions at line ~915 are Toys v0 ckpt-844.


## 2026-11-27 Scope, per-dataset hyperparameters, Steam lr 5e-5 replicate SUBMITTED

- Author: develop only on CDs, Toys, Steam; MovieLens = substitute dataset (results kept, no new development runs).
- Author: each dataset keeps its best MW-R-H hyperparameters and every later model uses them: Toys / CDs lr 3e-4, Steam lr 5e-5 (clip 1.0, margin 0.2, WIDER 0.1, rank weights, 3 epochs, batch 128 / micro 4, base Qwen2.5-1.5B, shared). Table in Report/CONVENTIONS.md. Steam MW-R-H-CR 384089 (lr 3e-4) is therefore not a valid Steam test of CR.
- Author seed rule: a configuration succeeds if the criterion holds on its single run, or on >= 2 seeds when several were run (reruns at identical settings count). Under it: CDs MW-R-H 3/3, Toys 3/5, Steam lr 5e-5 1/1, MovieLens 0/1 (D 1/1).
- Steam MW-R-H lr 5e-5 replicate **385083** (R3_mwrh_lr5e5_rep2_steam; identical to 383067: code_66b61b81, LR=5e-5, MAX_GRAD_NORM=1.0, default seed; output Steam_margin_0.2_wider_0.1_refweight_nolatent_rank_lr5e-5_rep2; researchshort a40|l40s|a100|l40 excl. alarm,album; tmp/submit_steam_lr5e5_rep2.sh), via JobManager.

## 2026-11-27 — C2 phase starts: MW-R-H + same-motive on Toys and Steam (submitted via JobManager)
- Author: "based on MW-R-H let move to collaborative reasoning contribution".
- Lead C2 variant is **same-motive InfoNCE** (`MOTIVE_W=0.1`, from epoch 2).
  - It is the only collaborative term that meets the criterion on the MW-R-H base: CDs 374791 is significantly above
    R3_ori at all cutoffs, equals MW-R-H, and its shuffle drop is −.040 vs −.017.
  - The original CR (user + target heads) passes only on Toys 383923. It fails on CDs, and Steam was run at a non-best lr.
- What's missing is the C2 table on Toys and Steam.
  - Script: `.claude/jobs/e8c23a07/tmp/submit_mwrh_motive_toys_steam.sh`, code_42e0da0a, researchshort.
  - The motive path equals 013bd0ac (374791): all later additions are off by default and `max_grad_norm` 1.0 is the HF default
    (checked by `git diff 013bd0ac 42e0da0a`).
  - Toys_and_Games_margin_0.2_wider_0.1_refweight_nolatent_rank_motive (lr 3e-4); Steam_margin_0.2_wider_0.1_refweight_nolatent_rank_lr5e-5_motive (lr 5e-5).
  - Job IDs: pending JobManager.
