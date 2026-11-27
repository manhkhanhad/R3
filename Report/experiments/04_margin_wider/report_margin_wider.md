# E04 — Margin + WIDER (MW): the first latent that is used, and the copy-row problem
Back to [main plan](../../main.md). Dates: 2026-11-07 – 2026-11-13. Status: done.

## Question
Can a decoder-side objective make R3's decoder read the *content* of `r` (mean and shuffle significantly hurt)
without the ranking cost of the structural bottlenecks (E03: −11% to −25% HR@10)? Where does the remaining
ranking cost come from?

## Setup

**Method (MW).** For a micro-batch of B = 4 rows with own latents r_b, the training loss is

```
L = (1/B) Σ_b NLL_b(r_b)                                         # R3 CE (per-token mean NLL of the target title)
  + λ_h · (1/B) Σ_b relu( m + NLL_b(r_b) − NLL_b(sg(r_π(b))) )   # history-swap hinge
  + λ_W · (1/B) Σ_b (z_b · q)²                                    # WIDER
```

- NLL_b(r) = mean over target tokens of −log p(y_t | x_b, r, y_<t), r in the `<|Thought|>` slot, full history x_b
  visible (deployed attention). π = roll by one inside the micro-batch, so r_π(b) is another in-batch user's latent on
  the *same* history x_b. sg = detach: the other row's pooling gets no gradient, but the decoder receives gradient through
  both passes (two-sided hinge).
- WIDER (`src/latent/wider_reg.py::WiderState.penalty`): z_b = r_b/‖r_b‖, q = normalize(e), e ← 0.99·e + 0.01·mean_b z_b
  (EMA, stop-gradient, updated after each call). 1.0 = all latents on one shared direction (collapse).
- Hyperparameters: m = 0.2, λ_h = 1.0, λ_W = 0.1 (EMA momentum 0.99), base Qwen/Qwen2.5-1.5B (run logs), lr 3e-4,
  batch 128 (micro 4), 3 epochs, warm-up 20 steps, bf16. Inference, prompt and ranking = plain R3 (constrained 10-beam).
- Code: `src/latent/margin_loss_attention.py::MarginLatentModel.forward` (hinge + WIDER added before its own
  / grad_accum_steps, so the grad-accum 32× aux-loss bug does not apply), trainer `src/latent/latent_attention_train_margin.py`
  (load_best_model_at_end=True, early stopping patience 1), `scripts/margin_wider.sbatch`; commit 6fac3d4e, frozen
  `output_dir/margin_wider_code/code_6fac3d4e` (Toys 356322, CDs 362561).
- Later MW-family runs (MW-3b, MW-3b-L4, MW-3a) use the re-implementation `src/latent/motive_contrastive.py::MotiveMarginLatentModel`
  with `MOTIVE_W=0` (same loss; trainer `latent_attention_train_motive.py`, keeps the **last** epoch), via
  `scripts/margin_wider_collab.sbatch METHOD=motive`; codes a21e067c (MW-3b), 2cb3645b (MW-3b-L4), 000b81fe (MW-3b-λ).

**Runs.** Musical 356320 (cancelled before start; the user moved the side task to Toys), Toys 356322, CDs 362561,
MW-3b 363603, MW-3a 363602 (cancelled), MW-3b-L4 364065, MW-3b-λ (λ 0.1 / 0.3; job IDs not recorded, no results).
Baselines: plain R3 (A) 345857 (Toys retrain), R3_ori stage 1 (Toys, CDs), WIDER-only 355749.

**copy_row (Toys/CDs).** `motive_contrastive.py::copy_row`: target timestamp = last history timestamp (day granularity)
and some history title shares the target's brand (first word) with title-token Jaccard ≥ 0.3. Test: Toys 905 rows
(13.4%), CDs 418. The 2026-11-11 `compare.py` split uses a near-identical rule (any same-day history item) and gives
836 (Toys) / 368 (CDs); both are shown where the source used them.

## Results

### Ranking, all cutoffs (HR@1 = NDCG@1; @20 = @10)

| model (run) | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Toys** plain R3 (A) 345857 | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |
| R3_ori stage 1 | 0.0377 | 0.0607 | 0.0778 | 0.1023 | not recorded | not recorded | 0.0659 |
| WIDER only 355749 | 0.0350 | 0.0628 | 0.0790 | 0.1070 | 0.0512 | 0.0578 | 0.0669 |
| **MW 356322** | 0.0373 | 0.0634 | 0.0803 | 0.1030 | 0.0525 | 0.0595 | 0.0668 |
| MW-3b 363603 (hinge epoch 1 only) | 0.0398 | 0.0687 | 0.0856 | **0.1125** | 0.0562 | 0.0632 | **0.0718** |
| MW-3b-L4 364065 (late novel-token hinge) | 0.0341 | 0.0613 | 0.0785 | 0.1034 | 0.0498 | 0.0568 | 0.0648 |
| **CDs** R3_ori stage 1 | 0.0495 | 0.0707 | 0.0819 | 0.0989 | 0.0616 | 0.0662 | 0.0717 |
| **MW 362561** | 0.0525 | 0.0726 | 0.0851 | **0.1049** | 0.0640 | 0.0691 | **0.0754** |

CDs R3_ori: metric.txt HR@10 0.0996; 0.0989 is the per-row recomputation (string normalisation). Its NDCG@3/5 were
recomputed read-only for this report from `R3_ori/output_dir/CDs_and_Vinyl/latent/final_result.json`.

### Significance vs the matched baseline (diff (p_t / p_b); * = significant)

| comparison | ΔHR@5 | ΔNDCG@5 | ΔHR@10 | ΔNDCG@10 |
|---|---|---|---|---|
| Toys MW 356322 − plain A | +0.0007 (.77/.76) | +0.0010 (.56/.53) | −0.0022 (.43/.47) | +0.0001 (.96/.94) |
| Toys MW 356322 − R3_ori | not recorded | not recorded | +0.0007 [−0.0050, +0.0065] | not recorded |
| Toys MW-3b 363603 − plain A | **+0.0061 (.018/.019)*** | **+0.0047 (.006/.008)*** | **+0.0073 (.010/.011)*** | **+0.0051 (.002/.003)*** |
| Toys MW-3b − MW 356322 | not recorded | not recorded | **+0.0095 [+0.0042, +0.0150]*** | **+0.0050*** |
| Toys MW-3b-L4 364065 − plain A | −0.0010 (.68/.73) | −0.0016 (.33/.35) | −0.0018 (.52/.56) | −0.0019 (.24/.26) |
| CDs MW 362561 − R3_ori | +0.0032 (.19/.21) | +0.0029 (.11/.11) | **+0.0060 (.029/.031)*** | **+0.0037 (.036/.040)*** |

p_t (paired t-test) and p_b (bootstrap two-sided p) in the "− plain A" and CDs rows were computed read-only from the
existing `eval_raw.json` files for this report; their bootstrap CIs equal the log's. Other rows: bootstrap CIs from the log.

### Latent tests (raw − intervention, HR@10 [95% CI] / NDCG@10; positive = replacing `r` hurts)

| model | mask | mean | shuffle_user | verdict |
|---|---|---|---|---|
| WIDER only 355749 | +0.0009 n.s. | −0.0007 n.s. | −0.0009 n.s. | unused |
| **Toys MW 356322** | **+0.0508** [+0.0450, +0.0567] / +0.037 | **+0.0309** [+0.0260, +0.0358] / +0.023 | **+0.0733** [+0.0668, +0.0801] / +0.051 | **content used** |
| CDs MW 362561 | **+0.0125** | **+0.0068** | **+0.0039** [+0.0002, +0.0076] | content used (weaker) |
| MW-3b 363603 | −0.0010 n.s. | −0.0016 n.s. | −0.0006 n.s. | **unused** |
| MW-3b-L4 364065 | **+0.028** | **+0.032** | **+0.036** | content used |

Toys MW operator trace: KL(swap r) 1.79 (ΔNLL +1.74) vs random direction 0.17 vs drop-last-item 0.089 (swap moves the
output 20× more than dropping the last history item); effective rank 3.69, top-1 energy 0.78, pooling 93% on history titles.

### Copy rows vs non-copy rows (HR@10)

| model | copy (Toys 905) | non-copy (5,833) | copy (Toys 836, compare.py) | non-copy (5,902) |
|---|---:|---:|---:|---:|
| plain A | 0.513 | 0.0420 | 0.5443 | 0.0430 |
| R3_ori | not recorded | not recorded | 0.5335 | 0.0412 |
| WIDER only | 0.517 | 0.0434 | not recorded | not recorded |
| **MW 356322** | **0.479** | **0.0447** | 0.5072 | 0.0457 |
| MW-3b 363603 | 0.515 | 0.0501 | 0.538 | 0.0522 |
| MW-3b-L4 364065 | 0.499 | 0.0420 | not recorded | not recorded |

- Toys (836 split), MW − R3_ori: copy −0.0263 [−0.0562, +0.0012], non-copy +0.0046 [−0.0003, +0.0095], non-copy
  same-brand +0.0172 [−0.0008, +0.0367] (all n.s., borderline). MW − plain A on copy rows −0.0371 [−0.0658, −0.0108] **sig**
  (905 split: −0.0343 [−0.062, −0.008] **sig**). Total gap vs R3_ori decomposes into copy −0.0033, non-copy +0.0040.
- CDs (368 split), MW − R3_ori: copy **+0.0326 sig**, non-copy +0.0043 [−0.0009, +0.0093] n.s., same-brand target
  +0.0247 sig. (418 split, recomputed read-only: MW 0.620 / 0.0674 vs R3_ori 0.581 / 0.0638.)
- Pooled Toys + CDs non-copy rows (n = 11,691, stratified bootstrap): **+0.0044 [+0.0008, +0.0080] sig**.
- MW-3b − MW (836 split): copy +0.031 [+0.005, +0.055] sig, non-copy +0.0064 [+0.0017, +0.0110] sig.
- Across runs (905 split): every run trained with the history-swap hinge has copy HR@10 0.45–0.50 (MW 0.479, MWC-I
  0.482, MWC-M 0.486, G3 0.448; exception MWC-G 0.503); every run without it 0.50–0.52 (plain A 0.513, WIDER only 0.517).

## Reading
- **MW is the first run of the line that meets the criterion** (Toys): another user's `r` costs −0.073 HR@10 (−71%),
  the mean −0.031, and ranking is ≈ plain R3 (−0.0022, p_t .43). On CDs MW beats R3_ori (HR@10 +0.0060, NDCG@10 +0.0037,
  both sig) with content used, though about 20× more weakly (shuffle −0.004).
- **Why ranking does not improve on Toys: the copy-row trade.** On a copy row the history already contains the target
  (a same-day sibling variant), so NLL(y | x, r_own) ≈ NLL(y | x, r_other). The hinge can then only be met by making
  copying *conditional on r*: the decoder learns that the slot overrides the history when they disagree. `r` is one
  target-peaked vector, so the other sibling variants lose support and leave beam slots 6–10 (history-brand share of
  slots 6–10: plain A 0.756 → MW 0.681; slots 1–5 0.825 → 0.794). MW loses copy rows (−0.034 vs plain A, sig) and gains on
  genuine next-item rows (non-copy +0.0044 pooled, sig); on Toys (13% copy rows) the two cancel. WIDER alone does not
  cause it (copy 0.517). The copy loss does not replicate on CDs (+0.033), the non-copy gain does (+0.0046 / +0.0043).
- **This motivates E05:** the hinge must act where the history does **not** explain the target and stay weak where it
  does; this needs a per-row weight, a global schedule is not enough (next point).
- **MW-3b (hinge only in epoch 1): best ranking, latent unused.** All @5/@10 significantly above plain A and HR@10
  +0.0095 above MW (sig), on copy and non-copy rows, but mask/mean/shuffle are n.s.: the dependence built in epoch 1 is
  undone in epochs 2–3 by CE + WIDER. The early hinge is a training regulariser of the decoder, not a content route.
  Meets "beats R3", fails "content matters"; single seed.
- **MW-3a** (self-weighted hinge on MW) was cancelled (363602) after its MWC-M twin M3a showed the absolute weight
  1 − exp(−NLL) decays everywhere and leaves the latent unused (E05). **MW-3b-L4** restores latent use but loses MW-3b's
  non-copy gain (0.0420 = plain A); copy only partly recovers (0.499). **MW-3b-λ** (late hinge × 0.1 / 0.3, code 000b81fe)
  was requested but no job ID, log or result exists: not recorded.
- Caveats: one seed per run; the claim "WIDER removes the margin's cost" rests on a cross-dataset comparison (Musical
  margin-only 304312 HR@10 0.1176, −4% vs plain R3 v1 0.1226; no Toys margin-only control, user decision 2026-11-09).
  MW 356322 selects the best epoch (= epoch 3), the motive-trainer runs keep the last. Plain-R3 retrains on Toys span
  ≈ 0.102–0.108 HR@10.

## Paper use
- **Method section (motivation of contribution 1):** the MW loss (hinge + WIDER), the Toys/CDs latent tests (first used
  latent at R3-level ranking), and the copy-row decomposition (copy loses, non-copy gains, pooled non-copy +0.0044 sig)
  as the argument for reference weighting.
- **Ablation table:** "MW (fixed hinge)" and "MW-3b (hinge epoch 1 only)" as the two failure corners (used but copy
  loss / better ranking but unused). MW-3b-L4 as an appendix row.
- **Appendix:** across-run copy-row table (hinge vs no hinge), operator trace and geometry, Musical margin-only history.
- **Dev only:** MW-3a, MW-3b-λ (no results).

## Sources
- `analysis/document/research_notes/Latent utilization/research_log.md`: 1634–1641 (Musical → Toys), 1686–1707 (356322),
  1937–1966 (trade-off table, margin-only caveat), 2188–2234 (Toys copy split, CDs setup), 2333–2335 (across-run copy
  root cause), 2379–2412 (CDs 362561 split, pooled non-copy), 2439–2447 (MW-3a/MW-3b setup), 2501–2504 (MW-3b),
  2527–2539, 2545 (MW-3b-L4, MW-3b-λ), 2606–2630 (MW-3b vs MW / R3_ori).
- `analysis/document/research_notes/Latent utilization/MW_to_MWC_report_2026-11-12.md` (905-row tables, bootstrap vs A).
- `analysis/results/copy_rows_mw_vs_r3/{compare.py,output.txt,output_CDs.txt}`.
- Results: `output_dir/margin_wider/{Toys_and_Games,CDs_and_Vinyl}_margin_0.2_wider_0.1/eval_raw.json`;
  `output_dir/margin_wider_collab/Toys_and_Games_margin_0.2_wider_0.1_{marginwarm,marginwarm_latenovel}/eval_raw.json`;
  `output_dir/guidance_scaffold/Toys_and_Games_guidance_A/eval_raw.json`; logs `run_logs/R3_margin_wider.{356322,362561}.out`,
  `R3_mw3b.363603.out`, `R3_mw3bL4.364065.out`.
- Memory notes `r3-margin-wider.md`, `r3-margin-wider-collab.md`.
