# E01 — R3's decoder ignores the latent's content
Back to [main plan](../../main.md). Dates: 2026-09 (decoder-insensitivity study, Musical) to 2026-11-22 (Steam / MovieLens R3, Toys stage-2 replica). Status: done (motivation).

## Question
R3 pools the LLM's hidden states over the history into `r` and splices it into the prompt. Does the decoder's prediction depend on **which** `r` it gets (content), or only on there being **some** vector in the slot (presence)? If the content is unused, why?

## Setup
- **Latent test** (`src/prototype_grounding/experiment.py generate --mode raw|mask|mean|shuffle_user`, constrained 10-beam search over the catalogue, full test split). "Content used" = mean AND shuffle significantly hurt. "Slot used" = only mask hurts.
- **Runs** (plain R3, no extra loss):

| dataset | model (run) | checkpoint | test n |
|---|---|---|---:|
| Musical | plain R3 v0, final epoch (job ID not recorded) | `output_dir/Musical_Instruments_v0/latent/checkpoint-1557` | 8,313 |
| Toys | R3_ori stage 1 + exact R3 stage-2 GRPO replica (371452; latent test 374331) | `scratch/.../Toys_and_Games_R3exact_toys_r3ori_exact/model` | 6,738 |
| Toys | plain R3 v0, epoch 2 (this codebase; job ID not recorded) | `output_dir/Toys_and_Games_v0/latent/checkpoint-844` | 6,738 |
| CDs | R3_ori stage 1 (eval-only 355050) | `R3_ori/output_dir/CDs_and_Vinyl/latent` | 6,157 |
| Steam | R3 (371447) | `scratch/.../Steam_R3_base/latent` | 10,303 |
| MovieLens | R3 seed 0 (371446), seed 1 (375356) | `scratch/.../Movielens_R3_base{,_seed1}/latent` | 10,164 |

- **No-latent SFT** (same backbone and prompt without `<|Thought|>`; `no_latent_reference.sbatch`): Toys 365404, CDs 365885, Steam 369661, MovieLens 369531.
- **Decoder-insensitivity study** (`src/prototype_grounding/decoder_sensitivity.py`, Musical, first 2,000 test rows, teacher forcing): KL(p_raw ‖ p_variant) over the target title, compared with dropping the last history item. **Side-information injection** (Experiment 1, `src/latent/side_info_latent.py`, Musical; jobs 304903/304904/304717–304720): `r' = r + 30·normalize(P(v))`.
- **Collapse diagnosis** U1–U3 (jobs 338553/338554, 338614, 338630–338634). **Dependency test v2** (`latent_dependency_v2.py`, code 9f8a582): per sample, teacher-force the model's own top-1 with its own `r` and with K = 16 bank latents from other users; D = log-prob drop in nats.
- Tables 1–2 were **computed for this report** (read-only) from the saved per-row files: per-row HR/NDCG from the stored rank, paired t-test (`scipy.stats.ttest_rel`) and paired bootstrap (2,000 resamples, seed 42, two-sided p). Every value matches the logged numbers where those exist.

## Results

### 1. Latent test on plain R3 (all cutoffs)

| dataset (run) | mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Musical v0 ckpt 1557 | raw | 0.0749 | 0.0914 | 0.1036 | 0.1231 | 0.0845 | 0.0895 | 0.0957 |
| | mask | 0.0443 | 0.0611 | 0.0739 | 0.0919 | 0.0538 | 0.0591 | 0.0649 |
| | mean | 0.0748 | 0.0911 | 0.1035 | 0.1240 | 0.0841 | 0.0892 | 0.0958 |
| | shuffle | 0.0753 | 0.0925 | 0.1043 | 0.1243 | 0.0851 | 0.0899 | 0.0963 |
| Toys stage-2 replica (371452) | raw | 0.0377 | 0.0619 | 0.0779 | 0.1020 | 0.0516 | 0.0582 | 0.0660 |
| | mask | 0.0209 | 0.0465 | 0.0644 | 0.0934 | 0.0357 | 0.0430 | 0.0524 |
| | mean | 0.0380 | 0.0622 | 0.0781 | 0.1021 | 0.0518 | 0.0583 | 0.0661 |
| | shuffle | 0.0377 | 0.0619 | 0.0787 | 0.1012 | 0.0516 | 0.0585 | 0.0658 |
| Toys v0 ckpt 844 | raw | 0.0341 | 0.0614 | 0.0779 | 0.1083 | 0.0498 | 0.0566 | 0.0664 |
| | mask | 0.0343 | 0.0598 | 0.0785 | 0.1083 | 0.0490 | 0.0567 | 0.0664 |
| | mean | 0.0346 | 0.0611 | 0.0782 | 0.1083 | 0.0500 | 0.0570 | 0.0666 |
| | shuffle | 0.0347 | 0.0613 | 0.0784 | 0.1083 | 0.0501 | 0.0571 | 0.0668 |
| CDs R3_ori (355050) | raw | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0617 | 0.0662 | 0.0717 |
| | mask | 0.0169 | 0.0383 | 0.0539 | 0.0731 | 0.0294 | 0.0358 | 0.0420 |
| | mean | 0.0490 | 0.0703 | 0.0819 | 0.0991 | 0.0612 | 0.0659 | 0.0715 |
| | shuffle | 0.0492 | 0.0702 | 0.0819 | 0.0991 | 0.0613 | 0.0661 | 0.0717 |
| Steam R3 (371447) | raw | 0.0077 | 0.0151 | 0.0226 | 0.0358 | 0.0119 | 0.0150 | 0.0192 |
| | mask | 0.0019 | 0.0069 | 0.0126 | 0.0217 | 0.0047 | 0.0071 | 0.0100 |
| | mean | 0.0075 | 0.0151 | 0.0223 | 0.0355 | 0.0119 | 0.0148 | 0.0190 |
| | shuffle | 0.0077 | 0.0155 | 0.0226 | 0.0351 | 0.0122 | 0.0151 | 0.0191 |
| MovieLens R3 (371446) | raw | 0.0249 | 0.0556 | 0.0806 | 0.1335 | 0.0425 | 0.0527 | 0.0698 |
| | mask | 0.0049 | 0.0194 | 0.0269 | 0.0412 | 0.0130 | 0.0161 | 0.0207 |
| | mean | 0.0242 | 0.0547 | 0.0811 | 0.1313 | 0.0416 | 0.0524 | 0.0685 |
| | shuffle | 0.0243 | 0.0548 | 0.0802 | 0.1309 | 0.0417 | 0.0522 | 0.0685 |

HR@1 = NDCG@1; @20 = @10 (10 beams).

**Significance.** Δ = mode − raw (negative = hurts), written as "Δ (p_t / p_b)". Bold = significant. "<5e-4" means no bootstrap resample crossed 0.

| run | mask HR@10 | mask NDCG@10 | mean HR@5 | mean HR@10 | mean NDCG@10 | shuffle HR@5 | shuffle HR@10 | shuffle NDCG@10 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Musical v0 | **−0.0312** (3e-33/<5e-4) | **−0.0309** | −0.0001 (.84/.92) | +0.0010 (.19/.24) | +0.0001 (.87/.90) | +0.0007 (.29/.33) | +0.0012 (.14/.17) | +0.0006 (.15/.14) | slot only |
| Toys 371452 | **−0.0086** (3e-5/<5e-4) | **−0.0136** | +0.0001 (.78/.91) | +0.0001 (.81/.92) | +0.0002 (.52/.54) | +0.0007 (.17/.18) | −0.0007 (.20/.24) | −0.0002 (.43/.42) | slot only |
| Toys v0 844 | +0.0000 (1/1) | −0.0000 | +0.0003 (.64/.69) | +0.0000 (1/1) | +0.0002 (.51/.51) | +0.0004 (.47/.51) | +0.0000 (1/1) | +0.0004 (.28/.27) | neither |
| CDs 355050 | **−0.0258** (6e-22/<5e-4) | **−0.0297** | +0.0000 (1/1) | +0.0002 (.76/.90) | −0.0002 (.40/.39) | +0.0000 (1/1) | +0.0002 (.78/.92) | −0.0000 (1/.97) | slot only |
| Steam 371447 | **−0.0141** (6e-21/<5e-4) | **−0.0092** | −0.0003 (.37/.43) | −0.0003 (.47/.56) | −0.0002 (.41/.42) | +0.0000 (1/1) | −0.0007 (.14/.16) | −0.0001 (.50/.50) | slot only |
| MovieLens 371446 | **−0.0923** (2e-180/<5e-4) | **−0.0491** | +0.0005 (.59/.63) | **−0.0022** (.033/.028) | **−0.0012** (.006/.004) | −0.0004 (.69/.70) | **−0.0027** (.020/.015) | **−0.0013** (.017/.018) | slot + tiny content (≈2% of HR@10) |
| MovieLens seed 1 (375356) | **−0.0796** | **−0.0404** | **−0.0018** (.013/.016) | **−0.0020** (.011/.006) | −0.0006 (.10/.10) | −0.0005 (.53/.59) | −0.0003 (.74/.78) | −0.0001 (.86/.85) | slot + mean only |

Content is unused on Musical, Toys, CDs and Steam. MovieLens seed 0 is the only plain-R3 run with a significant shuffle effect: 0.0027 HR@10 (2% relative). Seed 1 does not replicate it.

### 2. R3 ≈ no-latent SFT (Δ = R3 − no-latent, p_t / p_b)

| dataset | model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | R3_ori stage 1 | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |
| | no-latent 365404 | 0.0377 | 0.0628 | 0.0785 | 0.1039 | 0.0520 | 0.0585 | 0.0667 |
| | Δ | +0.0000 (1/1) | −0.0021 (.36/.39) | −0.0007 (.77/.79) | −0.0016 (.56/.56) | −0.0011 (.55/.54) | −0.0005 (.77/.80) | −0.0008 (.63/.63) |
| CDs | R3_ori stage 1 | 0.0494 | 0.0710 | 0.0819 | 0.0989 | 0.0617 | 0.0662 | 0.0717 |
| | no-latent 365885 | 0.0417 | 0.0637 | 0.0734 | 0.0945 | 0.0545 | 0.0585 | 0.0653 |
| | Δ | **+0.0076** (4e-5/<5e-4) | **+0.0073** (9e-4/.004) | **+0.0084** (2e-4/<5e-4) | +0.0044 (.092/.10) | **+0.0072** (3e-5/<5e-4) | **+0.0076** (4e-6/<5e-4) | **+0.0064** (7e-5/<5e-4) |
| Steam | R3 371447 | 0.0077 | 0.0151 | 0.0226 | 0.0358 | 0.0119 | 0.0150 | 0.0192 |
| | no-latent 369661 | 0.0065 | 0.0139 | 0.0208 | 0.0357 | 0.0108 | 0.0136 | 0.0184 |
| | Δ | +0.0012 (.17/.18) | +0.0013 (.24/.27) | +0.0018 (.14/.14) | +0.0001 (.95/.96) | +0.0011 (.16/.16) | +0.0014 (.077/.084) | +0.0008 (.34/.33) |
| MovieLens | R3 371446 | 0.0249 | 0.0556 | 0.0806 | 0.1335 | 0.0425 | 0.0527 | 0.0698 |
| | no-latent 369531 | 0.0253 | 0.0541 | 0.0797 | 0.1332 | 0.0419 | 0.0524 | 0.0695 |
| | Δ | −0.0004 (.78/.80) | +0.0015 (.44/.44) | +0.0009 (.69/.67) | +0.0003 (.91/.88) | +0.0005 (.70/.69) | +0.0004 (.80/.78) | +0.0003 (.85/.85) |

An older Toys no-latent run (344012) gives the same picture: HR@10 0.1045 vs R3 0.1023, Δ −0.0022 [−0.0077, +0.0034], n.s. at every cutoff. On Musical, plain SFT scores 0.1009 / 0.0790 vs R3 v1 0.1226 / 0.0950, but the recipes were not matched. An `r` rescaled to have no effect (`plain_token`) still scores 0.1216.

### 3. Decoder-insensitivity study (Musical, 2,000 rows; mean KL over target tokens, 95% CI)

| intervention | v0 | v1 | collaborative InfoNCE |
|---|---:|---:|---:|
| shuffle `r` | 0.00019 [0.00016, 0.00024] | 0.00043 [0.00033, 0.00059] | 0.00053 [0.00048, 0.00059] |
| mean `r` | 0.00016 | 0.00029 | 0.00036 |
| drop last history item, `r` recomputed / fixed | 0.109 / 0.109 | 0.073 / 0.073 | 0.073 / 0.073 |
| mask | 0.562 | 0.0039 | 0.043 |
| **shuffle / drop-last** | **0.0017** (≈590× smaller) | **0.0059** (≈170×) | **0.0073** (≈140×) |
| attention on `r` / avg prompt token | 2.83× | 10.64× | 9.80× |

Swapping `r` changes the argmax on ≈0.3% of target tokens. Recomputing `r` after a history change gives the same output as keeping the old one: `r` does not carry the history change.

**Side-information injection (H1 test).** HR@10 on the full test set (n = 8,313). Steering on the first 1,000 rows; KL on 2,000 rows.

| slot content | HR@10 full | HR@10 true → swapped | top-10 in donor's category (true → swapped) | swap KL / drop-last |
|---|---:|---|---|---:|
| target category as text in prompt (`text_control`, leaky) | 0.6160 | 0.665 → 0.015 | 0.0% → 92.4% | 28.2 |
| target-category oracle in `r`, raw scale (`oracle_raw`, leaky) | 0.5922 | 0.647 → 0.026 | 0.1% → 89.3% | 13.6 |
| same, token scale (`oracle_token`, leaky) | 0.5965 | 0.634 → 0.026 | 0.1% → 89.1% | 12.4 |
| SASRec history vector (`sasrec_raw`) | 0.1196 | 0.157 → 0.115 (−27%) | 2.4% → 4.7% | 1.10 |
| random category (`null_raw`) | 0.1231 | 0.153 → 0.156 | 2.2% → 2.3% | 0.001 |
| plain R3 at token scale (`plain_token`) | 0.1216 | — | — | 0.001 |
| *plain R3 v1 (reference)* | *0.1226* | — | — | *0.006* |

Verdicts:
- H1 (redundancy) is confirmed.
- No attention (H2), norm mismatch (H3), collapse (H4) and off-manifold input (H7) are rejected as causes.
- The attention sink (H5) is a consequence of redundancy, not a cause.
- A decoder-side margin loss raises the swap ratio 9× (0.053) at HR@10 0.1176 (−4%), but adds no information.

### 4. Why R3's `r` has no content to use: template-token pooling (U1–U3)

| checkpoint | pooling mass on fixed template | most-attended token(s) | r residual / norm | user-only pooling with trained module |
|---|---:|---|---:|---:|
| R3_ori Toys stage 1 | 1.0000 | `" Input"` (pos 60, all 2,000 rows) | 0.0015 | 0.92 |
| R3_ori CDs stage 1 | 0.9954 | `" that"` ×3, `" request"`, `" music"` | 0.240 | 0.61 |
| R3_ori CDs stage 2 | 1.0000 | `" that"` pos 4 (1,946 rows) | 0.030 | 0.74 |
| R3_ori Musical stage 1 / 2 | 0.9989 / 0.9981 | `" a"`, `" an"` | 0.123 / 0.125 | 0.86 |
| fresh pooling init | — | — | 0.42–0.73 | — |

- R3_ori Toys train latents: pairwise cosine **1.0000**, ‖r − mean‖/‖r‖ 0.0000, effective rank 1.00, norm 272.7 ± 0.6. `r` is one constant vector, effectively a one-token soft prompt.
- The collapse is learned, not numerical. fp32 recomputation gives the same result. A fresh pooling module on the same hidden states is diverse (1,984 / 2,000 distinct).
- The LLM's states do carry target structure: `h_last` retrieves other users' same-target rows at 66× chance, while `r` is at 0.6× chance.

### 5. Dependency test v2 (Toys, per sample, K = 16 bank swaps)

| model | D (nats) mean / median / p90 | KL summed over title | rows with D > 1 | mean pairwise cos of r |
|---|---|---|---:|---:|
| plain R3 (R3_ori Toys, 353830) | −0.0003 / −0.00004 / 0.044 | 0.0010 | 0% | 1.0000 |
| same, zero vector | 3.61 / 3.57 / 6.00 | 3.55 | 92% | — |
| staged 2 latent-only + 1 full (353832) | 0.0069 / 0.0047 / 0.077 | 0.0069 | 0% | 0.88 |
| history-only pooling (353953) | 0.0095 / 0.0053 / 0.076 | 0.0049 | 0% | 0.38 |
| *positive reference: staged 1 latent-only + 2 full (353831)* | *2.20 / 1.98 / 3.94* | *1.70* | *83%* | *0.20* |

## Reading
- **Presence, not content.** Across five datasets, plain R3 needs a vector in the slot (mask hurts on Musical, Toys 371452, CDs, Steam and MovieLens) but not the right one. Mean and shuffle are n.s. on four datasets; on MovieLens the effect is tiny and not seed-robust. Toys v0 ckpt 844 is unaffected even by mask, so even the presence dependence is incidental.
- **The latent adds no ranking.** R3 equals the no-latent model at every cutoff on Toys, Steam and MovieLens. Only CDs gains (@1–@5 and NDCG@10). Its content is still unused there, so the gain comes from a shared learned vector, not from per-user reasoning.
- **Mechanism.** The pooling attends to constant template tokens, so `r` is close to one shared soft prompt. Even when `r` is diverse (Musical v1, history pooling, staged 2+1), the decoder ignores it, because `r = f(x)` repeats what the decoder already reads from `x` (H1).
- **The channel itself works.** Information the history lacks (oracle category, SASRec vector) is read through the same slot.
- **Caveats:**
  - single seeds, except MovieLens;
  - the dependency-v2 null on R3_ori Toys is degenerate (`r` is constant), so staged 2+1 and history pooling are the informative nulls;
  - the decoder-insensitivity study uses 2,000 rows and Instruct-backbone checkpoints (v0/v1), while the new-dataset runs use base Qwen2.5-1.5B.

## Paper use
**Motivation (main figure/table):**
1. Table 1, compressed to HR@10 / NDCG@10 for raw / mask / mean / shuffle with significance, for CDs 355050, Toys 371452, Steam 371447 and MovieLens 371446 (+ Musical v0). Message: "mask hurts, mean and shuffle don't".
2. Table 2, HR@10 / NDCG@10 per dataset. Message: the latent adds nothing on 3 of 4 datasets.
3. One-line numbers:
   - swapping `r` moves the output 140–590× less than dropping one history item;
   - R3_ori Toys `r` has pairwise cosine 1.0000, and every R3_ori checkpoint puts ≥ 99.5% of pooling mass on template tokens;
   - per-sample D ≈ 0 (−0.0003 nats).
4. The oracle contrast (Section 3): the swap redirects 89% of the top-10 (text 92%), while the random-category null scores 0.001. This argues that the channel works and redundancy is the problem.

**Appendix:** hypothesis verdicts, U3 table, dependency-v2 table.

## Sources
- `analysis/document/research_notes/Latent utilization/research_log.md`:
  - U1–U3 and "Agreed interpretation" (lines 25–127);
  - no-latent Toys 344012 (lines 471–516);
  - dependency test v2 (lines 1040–1071, 1182–1225);
  - R3_ori CDs check 355050 (lines 1312–1318, 1372–1400);
  - stage-2 replica 371452 (lines 2850–2866);
  - Steam / MovieLens R3 and no-latent (lines 2755–2813).
- `analysis/document/decoder_insensitivity_to_latent.md`: Conclusion, §1–§4, §7, Experiment 1 Results.
- `analysis/document/latent_reasoning_collapse_r3.md` (CDs SFT/RL signature) and `analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md` (Problem statement).
- Result files:
  - `analysis/results/intervention/{musical_instruments_v0_checkpoint_1557, R3exact_toys_r3ori_exact, toys_and_games_v0_checkpoint_844, cds_and_vinyl_r3_ori_utilization, steam_r3_base, movielens_r3_base, movielens_r3_base_seed1}/`;
  - `output_dir/margin_wider_collab/no_latent_ref/<cat>/eval_no_latent.json`;
  - `R3_ori/output_dir/Toys_and_Games/latent/final_result.json`;
  - `analysis/results/side_info/full_<variant>/summary.json`.
