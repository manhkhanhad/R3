# Decoder Insensitivity to the Latent Reasoning Vector in R3

> Related: [Avoiding Latent Reasoning Collapse and Shortcut Learning in R3](avoid_reasoning_collapse_and_shortcut_learning.md), [Latent Reasoning Collapse in R3](latent_reasoning_collapse_r3.md), [Musical Instruments v0 epoch collapse report](latent_musical_instruments_v0_epoch_collapse_report.md).

## Problem statement

Across the attempts in [the collapse investigation](avoid_reasoning_collapse_and_shortcut_learning.md), improving the latent thought vector `r` (its geometry, its user/target structure, its diversity) has not reliably improved recommendation. The working hypothesis here is that a large part of the problem sits on the **decoder side**: the second-pass LLM produces its recommendation list while being largely unaffected by which `r` it receives. If that is true, better latents cannot translate into better recommendations, because the decoder does not read their content.

The E1 ranking interventions (`mask` / `mean` / `shuffle_*` / `noise_*`) are only a coarse test of this: they report whether top-10 lists change. This document measures the decoder's response to `r` directly, at the level of output distributions, attention and gradients. It then tests a set of candidate explanations for the insensitivity ([Hypotheses](#hypotheses)).

## Conclusion

**R3's decoder ignores the latent thought `r` because `r` is redundant with the history. The decoder is able to read the latent slot; plain R3 never puts anything in it that the decoder does not already get from the history text.**

- **The insensitivity is real and specific to content.**
  - Replacing `r` with another user's `r` moves the output 140–590× less than removing a single history item (§1).
  - Recomputing `r` after a history change adds nothing on top of the history tokens (§1).
  - This holds for plain R3 and for every `r`-side objective tried: collaborative InfoNCE, category alignment (§1, §4–5).
- **It is not a reading failure.**
  - The decoder attends to `r` 3–11× more than to an average prompt token (§2), and diverse latents are ignored as much as collapsed ones (§4).
  - When the slot carries information the history lacks, the same architecture uses it almost as well as text ([Experiment 1](#experiment-1)): a fully trained target-category oracle redirects 89% of the top-10 when swapped (text: 92%).
  - The same slot filled with an uninformative vector is ignored exactly like plain R3.
- **The scale/manifold mismatch is not the cause.**
  - `r` is ~80–240× larger than a token embedding by construction (§6), but the oracle is read equally well at raw and at token scale (89% / 89% steering).
  - Rescaling plain R3 to token scale leaves its content use at zero.
- **Incentives change sensitivity, not information.**
  - A decoder-side margin loss raises content sensitivity 9× (§7) but still adds nothing beyond the history, at a 4% HR@10 cost.
- **Non-redundant information is read, but so far does not improve ranking.**
  - A SASRec collaborative vector in the slot is used as much as a history item (swapping it costs 27% relative HR@10 on the probed subset).
  - Overall HR@10 (0.1196) does not exceed plain R3 (0.1226): the decoder uses it *instead of* part of the title evidence, not in addition to it.

**The open problem is therefore no longer "why does the decoder ignore the latent", but "what information can the latent carry that is absent from the history text and improves the ranking"** (see [Next steps](#next-steps)).

**Caveat on R3's gain.** A plain-SFT model without any latent scores HR@10 0.1009 / NDCG@10 0.0790 on Musical Instruments (`scratchDirectory/R3/output_dir/Musical_Instruments_plain_sft`), versus 0.1226 / 0.0950 for R3 v1. Since `r`'s content is unused, and an `r` rescaled to have no effect at all (`plain_token`) still scores 0.1216, R3's advantage over plain SFT would not come from latent reasoning content. It would come from something else, such as the extra slot or differences in the training recipe. *The two training recipes have not yet been verified to be comparable*, so this is a lead to check, not a result.

## Summary of hypotheses

| # | hypothesis | verdict | key evidence |
|---|---|---|---|
| H1 | `r` carries no information beyond the history `x`, so the decoder has no incentive to read it | **Confirmed** ([Experiment 1](#experiment-1)) | with information the history lacks in the slot (target-category oracle, SASRec CF vector), the decoder reads it heavily (swap moves the output 12–14× resp. 1.1× as much as dropping a history item; oracle steers 89% of the top-10, text 92%); an uninformative vector in the same slot is ignored like plain R3. Earlier: recomputing `r` after a history change adds nothing (§1); a decoder-side margin loss raises sensitivity 9× without changing that (§7) |
| H2 | The decoder does not attend to `r` | **Rejected** | `r` gets 3–11× the attention of an average prompt token in plain checkpoints, up to 765× in category-align (§2) |
| H3 | `r`'s norm is far too large (scale mismatch with token embeddings) | **Rejected as the cause** ([Experiment 1](#experiment-1)) | the oracle is read equally well at raw and at token scale (steering 89% / 89%); rescaling plain R3 to token scale makes `r` fully ignored (swap ratio 0.001, mask KL 0.0001). The mismatch is real and architectural (§6) but not what blocks use |
| H4 | `r` is collapsed: too little sample-specific variation to matter | **Rejected as the main cause** | v1 has the most diverse `r` (72% of norm is sample-specific) and is still insensitive (§4) |
| H5 | The decoder attends to `r` but reads it as a constant anchor (sink): the value/output path discards the directions in which `r` varies | **Consequence, not a blocker** | anchor/sink behaviour appears when `r` is redundant; with informative content the same slot transmits it (oracle steering 89% vs 92% for text) |
| H6 | The history shortcut is learned first and locks in before `r` becomes useful (training dynamics) | **Supported indirectly (CDs only); applies to redundant content** | CDs curricula that block `x` during training produce content-dependence; any training exposure to the unrestricted path removes it. When the slot carries needed information, sensitivity *grows* with training (oracle steering 68–76% at epoch 1 → 89% at epoch 3; [Experiment 1](#experiment-1)) |
| H7 | `r` lies off the input-embedding manifold in *direction*, not only in scale, so pretrained circuits cannot interpret it | **Rejected as a blocker** | an unconstrained learned projection into the slot carries the oracle almost as well as text (89% vs 92% steering) |
| H8 | The insensitivity is a measurement artefact of top-10 ranking | **Rejected** | logit-level KL confirms it (§1) |

Details and the evidence for each are in [Hypotheses](#hypotheses).

## Method

Script: `src/prototype_grounding/decoder_sensitivity.py` (launcher `scripts/decoder_sensitivity_musical.sbatch <checkpoint> <tag> [limit]`). It reuses the prompt construction, thought extraction and splicing from `src/prototype_grounding/experiment.py`, so the thought vector and prompts are identical to the E1 pipeline. All measurements use teacher forcing on the held-out target title; nothing is trained or modified.

Three probes:

1. **Output shift.** For every target-title position, `KL(p_raw || p_variant)` over the full vocabulary, averaged per example; also the KL at the first target token alone, the target NLL change, and top-1 (argmax) agreement with the raw prediction.
   - `r` interventions: `mask` (zero), `mean` (mean `r` over the evaluated subset), `shuffle` (the `r` of a different user from the same subset).
   - History reference: `drop_last` removes the single most recent history item from the prompt, either with `r` recomputed from the shortened prompt (`drop_last_recomputed_r`) or with the original `r` kept (`drop_last_fixed_r`). This gives a natural scale: how much does a small, real change to the history move the output compared with replacing `r` entirely?
2. **Attention.** Eager attention on the raw input, averaged over heads. For queries at target-token positions after the thought, the attention mass on the thought position versus the average mass per prompt token (excluding the first token), per layer.
3. **Gradient.** Gradient of the target NLL with respect to the input embeddings: `‖∇‖` at the thought position versus the average prompt token, and the linearized content effect `|<∇_r, r − mean_r>|` (how much NLL would change, to first order, by removing `r`'s sample-specific component).

The script also reports the spread of `r` over the evaluated subset (norm, distance to the subset mean, cosine to the mean). Two auxiliary scripts are used in §5–6: `src/prototype_grounding/category_probe.py` and `src/prototype_grounding/latent_norm_trace.py`.

Setup: `Musical_Instruments`, test split, first 2,000 examples, Qwen2.5-1.5B-Instruct backbone. Confidence intervals are bootstrap 95% (2,000 resamples, seed 42) over examples. Re-running a checkpoint changes the numbers only in the fourth decimal place (GPU nondeterminism).

Checkpoints (all paths under `scratchDirectory/R3/output_dir/`):

| tag | checkpoint | what it is | standard HR@10 / NDCG@10 | E1 HR@10 raw / mask / mean / shuffle_user |
|---|---|---|---|---|
| `musical_v0_ckpt1557` | `Musical_Instruments_v0/latent/checkpoint-1557` | plain R3 SFT (run v0), final epoch | — | 0.1231 / 0.0919 / 0.1240 / 0.1243 |
| `musical_v1` | `Musical_Instruments_v1/latent` | plain R3 SFT (run v1, same architecture/config as v0) | 0.1226 / 0.0950 | not run |
| `musical_collaborative_infonce` | `Musical_Instruments_R3_collaborative_infonce/latent` | R3 + collaborative multi-positive InfoNCE ([Attempt 10](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10)) | — | 0.1217 / 0.1087 / 0.1204 / 0.1208 |
| `musical_category_align` | `Musical_Instruments_R3_category_align_gte_v1/latent` (= checkpoint-1038, best eval loss) | R3 + target-category alignment loss (weight 0.1; `category_projection(r)` → gte-large embedding of the target's category path) | 0.1188 / 0.0912 | not run |

v0 and collaborative InfoNCE show the familiar E1 signature: zeroing `r` hurts, swapping in a different valid `r` does not.

## Results

### 1. Output shift: replacing `r` entirely moves the output far less than dropping one history item

Mean KL over target tokens (95% CI):

| intervention | v0 | v1 | collaborative InfoNCE | category-align |
|---|---:|---:|---:|---:|
| shuffle `r` | **0.00019** [0.00016, 0.00024] | **0.00043** [0.00033, 0.00059] | **0.00053** [0.00048, 0.00059] | **0.00198** [0.00183, 0.00214] |
| mean `r` | 0.00016 [0.00013, 0.00021] | 0.00029 [0.00023, 0.00038] | 0.00036 [0.00033, 0.00039] | 0.00108 [0.00098, 0.00117] |
| drop last history item, `r` recomputed | **0.109** [0.101, 0.118] | **0.073** [0.066, 0.081] | **0.073** [0.066, 0.080] | **0.067** [0.061, 0.074] |
| drop last history item, `r` fixed | 0.109 [0.101, 0.118] | 0.073 [0.065, 0.080] | 0.073 [0.066, 0.080] | 0.064 [0.058, 0.071] |
| mask (zero `r`) | 0.562 [0.551, 0.575] | 0.0039 [0.0035, 0.0042] | 0.043 [0.039, 0.047] | 0.422 [0.406, 0.437] |
| **shuffle / drop-last ratio** | **0.0017** | **0.0059** | **0.0073** | **0.029** |

KL at the first target token, ΔNLL (variant − raw) and top-1 agreement with raw:

| intervention | v0 KL₁ / ΔNLL / agree | v1 KL₁ / ΔNLL / agree | collab KL₁ / ΔNLL / agree | category-align KL₁ / ΔNLL / agree |
|---|---|---|---|---|
| shuffle `r` | 0.0013 / 0.00009 (n.s.) / 99.7% | 0.0044 / 0.00027 (n.s.) / 99.7% | 0.0039 / 0.00037 (n.s.) / 99.6% | 0.032 / **0.0020** [0.0013, 0.0028] / 99.6% |
| mean `r` | 0.0009 / −0.00010 (n.s.) / 99.8% | 0.0029 / 0.00015 (n.s.) / 99.8% | 0.0024 / 0.00041 / 99.7% | 0.019 / 0.0006 / 99.7% |
| drop last, `r` recomputed | 0.332 / 0.072 / 94.7% | 0.271 / 0.059 / 95.9% | 0.249 / 0.055 / 95.6% | **0.192** / 0.046 / 96.3% |
| drop last, `r` fixed | 0.331 / 0.073 / 94.7% | 0.263 / 0.059 / 96.0% | 0.249 / 0.055 / 95.6% | **0.128** / 0.043 / 96.4% |
| mask | **10.00** / 0.550 / 91.0% | 0.023 / 0.0027 / 99.0% | 0.328 / 0.038 / 97.3% | 2.38 / 0.396 / 87.3% |

Observations:

- **In the plain and collaborative checkpoints, replacing `r` with another user's `r` shifts the output 140–590× less than removing one history item** (v0: 0.00019 vs 0.109; v1: 0.00043 vs 0.073; collaborative InfoNCE: 0.00053 vs 0.073). The swap changes the argmax on only ~0.3% of target tokens, and the target NLL change is not significant. This is the decoder-level explanation for the null `mean`/`shuffle` E1 results.
- **In those three checkpoints, `r` does not transmit history changes.** When the history changes, recomputing `r` from the new history and keeping the old `r` give practically the same output (e.g. v0 0.1091 vs 0.1090; v1 0.0731 vs 0.0725). The effect of the history change reaches the output through the history tokens directly.
- **Category-align is the first checkpoint where this starts to break**, though still weakly:
  - swapping `r` significantly hurts target likelihood (ΔNLL +0.0020);
  - the swap/drop-last ratio is 4–17× that of the other checkpoints;
  - recomputing `r` after the history change moves the first target token noticeably more than keeping the old `r` (KL₁ 0.192 vs 0.128), so roughly a third of that first-token shift now flows through `r`.

  A swap still moves the output ~34× less than dropping one history item, and ranking did not improve (HR@10 0.1188).
- **Dependence on `r`'s presence varies widely between runs; dependence on its content does not.**
  - v0: zeroing `r` destroys the first-token distribution (KL₁ = 10.0). The decoder needs a vector of the right kind in the slot, but any valid vector works.
  - v1: zeroing `r` barely matters (KL 0.0039, 99.0% argmax agreement), yet it ranks as well as v0 (HR@10 0.1226 vs 0.1231).
  - collaborative InfoNCE sits in between (mask KL 0.043).

  Two plain R3 runs with identical configuration reach the same ranking quality with a decoder that respectively depends heavily on and ignores `r`'s presence, which shows the `mask` effect is incidental to how the model makes recommendations.

### 2. Attention: the decoder does attend to `r`

Attention from target-token queries to the thought position, relative to the average prompt token (mean over layers; 95% CI):

| checkpoint | mass on `r` | avg mass per prompt token | ratio `r` / avg prompt token |
|---|---:|---:|---:|
| v0 | 0.0079 | 0.0035 | **2.83** [2.79, 2.88] |
| v1 | 0.0102 | 0.0011 | **10.64** [10.47, 10.81] |
| collaborative InfoNCE | 0.0196 | 0.0023 | **9.80** [9.63, 9.98] |
| category-align | **0.440** | 0.0011 | **765** [754, 775] |

Per-layer ratio (layers 0–27):

- v0: 3.3, 10.9, 2.6, 3.2, 0.8, 0.4, 0.8, 7.9, 0.9, 2.1, 1.3, 4.1, 1.0, 5.8, 1.6, 2.5, 1.2, 2.3, 6.7, 1.0, 1.4, 1.6, 4.1, 0.5, 0.5, 0.5, 1.3, 8.9
- v1: 6.8, 12.2, 5.7, 19.8, 9.3, 5.3, 13.2, 25.1, 4.9, 16.7, 5.6, 12.1, 4.4, 19.1, 4.3, 9.0, 5.8, 7.3, 15.2, 6.4, 3.8, 9.7, 27.9, 3.5, 7.8, 17.2, 13.2, 6.5
- collaborative InfoNCE: 3.3, 13.5, 5.9, 11.3, 7.7, 2.1, 9.7, 18.6, 1.1, 9.7, 3.2, 14.4, 1.8, 15.1, 5.9, 10.1, 5.2, 10.4, 19.3, 7.8, 7.8, 7.0, 19.8, 9.5, 5.1, 11.9, 8.0, 29.1
- category-align: 5, 14, 126, 1801, 1363, 283, 505, 181, 612, 378, 637, 407, 780, 398, 879, 530, 475, 508, 388, 830, 661, 960, 1038, 1409, 2325, 2657, 1248, 9. The absolute mass on `r` is 0.24–0.71 in every layer from 2 to 26 (0.70 in layer 3, 0.71 in layer 25).

`r` receives more attention than an average history token in most layers of every checkpoint. **Attention routing is not the bottleneck.** Two checkpoints make the point sharply:

- v1 pays `r` 10.6× the attention of an average token, yet zeroing `r` hardly changes its output.
- In category-align, `r` has become a **massive attention sink**, absorbing 44% of all attention from target tokens, yet swapping `r` still moves the output ~34× less than dropping one history item.

Attention mass on `r` is therefore not information flow from `r`.

### 3. Gradient: small per unit change, but confounded by `r`'s norm

| quantity (mean, 95% CI) | v0 | v1 | collaborative InfoNCE | category-align |
|---|---:|---:|---:|---:|
| `‖r‖` | 100.0 | 78.8 | 205.3 | 243.2 |
| avg prompt-token embedding norm | 1.01 | 1.01 | 1.02 | 1.01 |
| `‖∇‖` at `r` / avg prompt token `‖∇‖` | 0.034 [0.032, 0.035] | 0.056 [0.054, 0.058] | 0.044 [0.043, 0.045] | 0.087 [0.084, 0.090] |
| linearized content effect `\|<∇_r, r − mean_r>\|` (NLL units) | 0.0013 [0.0011, 0.0014] | 0.0025 [0.0023, 0.0027] | 0.0042 [0.0039, 0.0046] | 0.0068 [0.0062, 0.0075] |

Per unit of input change, the NLL is 11–30× less sensitive at the thought position than at an ordinary token embedding. **This ratio should not be read as "the decoder learned to be flat along `r`".** Every attention and MLP block reads its input through RMSNorm, whose gradient scales as ~1/‖input‖. An input with norm 80–240 therefore receives a per-unit gradient roughly two orders of magnitude smaller than a norm-1 token even if the decoder used it fully. The measured ratio is roughly what the norm alone predicts.

The linearized content effect (0.001–0.007 NLL) is norm-independent in the relevant sense, because it uses the actual displacement `r − mean_r`. It agrees in order of magnitude with the measured `mean`/`shuffle` ΔNLL, confirming the output-shift results in §1; those remain the primary evidence.

### 4. The insensitivity is not explained by `r` being collapsed

Spread of `r` over the same 2,000 test examples:

| | v0 | v1 | collaborative InfoNCE | category-align |
|---|---:|---:|---:|---:|
| mean `‖r‖` | 100.0 | 78.8 | 205.3 | 243.2 |
| `‖mean_r‖` | 95.5 | 54.8 | 173.6 | 229.7 |
| mean `‖r − mean_r‖` | 24.7 | 58.7 | 110.1 | 77.1 |
| mean `‖r − mean_r‖ / ‖r‖` | 0.25 | **0.72** | 0.54 | 0.31 |
| mean cosine to `mean_r` | 0.96 | **0.72** | 0.84 | 0.95 |
| shuffle KL (from §1) | 0.00019 | 0.00043 | 0.00053 | 0.00198 |

In v0, `r` is dominated by a shared component (cosine 0.96 to the mean), so part of its insensitivity could be attributed to little sample-specific variation. The other checkpoints rule this out:

- In v1, the sample-specific component is ~72% of `r`'s norm, the least collapsed of the four, yet swap KL is only 0.00043.
- Category-align is nearly as collapsed as v0 (cosine 0.95) yet has the highest content sensitivity.

Content sensitivity does not track `r`'s diversity. **Making `r` more diverse does not make the decoder use it.**

### 5. Category-alignment pilot: does a target-oriented `r` get used?

The category-alignment run adds `0.1 · ‖normalize(category_projection(r)) − e_cat(target)‖²` to the CE loss, where `e_cat` is the gte-large-en-v1.5 embedding of the target item's category path (412 distinct categories). It was intended as a pilot for H1: give `r` information the decoder cannot read directly from the history.

**Design limitation.** At inference `r` is still computed from the history alone (`src/latent/category_alignment_attention.py`). The target category is used only as a training label for an auxiliary head, so the model learns to *predict* the target category from `x`. `r` cannot contain information beyond `x`; at best it becomes a precomputed, target-oriented summary of `x`. This tests a weaker version of H1: does a more useful summary get used?

**Did `r` learn the category?** Category probe on 1,973 test examples with a valid category (`analysis/results/decoder_sensitivity/musical_category_align/category_probe.json`), using the checkpoint's own trained head, cosine-matched against the 412 category embeddings:

| predictor of the target's category | top-1 accuracy |
|---|---:|
| category-aligned `r` through its trained head | 9.5% (top-5: 16.8%) |
| most frequent category in the training set | 8.5% |
| most frequent category in the user's history | 10.8% |
| **category of the last history item** | **14.4%** |

The alignment barely beat the frequency prior and lost to a trivial history heuristic. `r` learned little category information. The target category is also partly predictable from the history (last-item heuristic 14.4%), so "not extractable from the history" does not hold for this signal.

**Effect on the decoder** (§1–2): the highest content sensitivity measured (swap/drop-last 0.029, significant swap ΔNLL, `r` carrying ~⅓ of the first-token shift after a history change), together with `r` turning into a 44%-attention sink and ‖r‖ rising to 243.

**Verdict.** An `r`-side auxiliary objective measurably increased the decoder's use of `r`, from a tiny base and without a ranking gain. It cannot test H1 properly, because `r` gained no information beyond `x`. A clean test needs information truly absent from `x` at inference (see [Next steps](#next-steps)).

### 6. Where `r`'s large norm comes from

`r = W_V · (Σᵢ αᵢ hᵢ) + b_V`, where `hᵢ` are the backbone's final-layer hidden states *after* the final RMSNorm (`hidden_states[-1]` in HF Qwen2; verified in `Qwen2Model.forward`), and αᵢ are the pooling attention weights. Norm at each stage, mean over 200 test prompts (`analysis/results/decoder_sensitivity/latent_norm_trace.json`; script `src/prototype_grounding/latent_norm_trace.py`; the recomputed `r` matches the model's own within 5% in bf16):

| stage | base + fresh pooling init | v0 epoch 1 | v0 | v1 | collab InfoNCE | category-align |
|---|---:|---:|---:|---:|---:|---:|
| input token embedding ‖e‖ | 1.01 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| final hidden states ‖h‖ (after final RMSNorm) | **175** | 74 | 106 | 91 | 75 | 61 |
| mean pairwise cosine between hidden states | 0.53 | 0.25 | 0.17 | 0.21 | 0.45 | 0.59 |
| pooled ‖Σαh‖ | 156 | 94 | 104 | 77 | 114 | 80 |
| value gain ‖r‖ / ‖pooled‖ | 0.57 | 0.94 | 0.96 | 0.92 | **1.77** | **2.99** |
| value weight spectral norm | 1.15 | 1.95 | 1.97 | 2.02 | 4.21 | 4.58 |
| **‖r‖** | **89** | 89 | 100 | 71 | 202 | 239 |

"Base + fresh pooling init" is the pretrained Qwen backbone with the pooling layer freshly initialized, i.e. the state R3 SFT starts from.

- **The mismatch is architectural, present from step 0.** RMSNorm fixes each final hidden state's norm at ~√1536 ≈ 39 times its gain (mean 3.46, max 8.9, unchanged by fine-tuning in every checkpoint). That output-side scale suits the tied unembedding (sharp logits). Input embeddings from the same tied matrix have norm ≈ 1. `r` is spliced from the output side into the input side with no rescaling.
- **Pooling barely reduces it**, because the hidden states are fairly aligned (cosine 0.17–0.59) and the attention concentrates on a few tokens.
- **The default `nn.Linear` initialization of the value projection (gain 0.57) leaves ‖r‖ ≈ 89 at the first step.** Plain SFT keeps it at 70–100.
- **Objectives acting on `r` inflate it** by raising the value gain: collaborative InfoNCE ×1.8 (‖r‖ ≈ 200) and category alignment ×3.0 (‖r‖ ≈ 240), as did CDs contrastive V0 (‖r‖ ≈ 343). The category head normalizes its output, so the norm itself is irrelevant to that loss. The growth plausibly comes from the decoder's response (the same model turned `r` into a 44% sink), but that is an inference, not a measurement.

### 7. Decoder-side incentive: the counterfactual margin loss

The margin loss of [Attempt 2](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-2) (margin 0.2, weight 1.0) was retrained on Musical Instruments (job 304312; `output_dir/Musical_Instruments_R3_margin_m0.2/latent`; best checkpoint by eval loss). During training it penalizes the decoder whenever the target is not at least 0.2 nats more likely under the true `r` than under another example's `r`. So unlike the `r`-side objectives in §4–5, it acts directly on the decoder's use of `r`. Diagnostic: job 304754 (`analysis/results/decoder_sensitivity/musical_margin_m0.2/`).

| | v1 (plain R3) | category-align | **margin=0.2** |
|---|---:|---:|---:|
| standard HR@10 / NDCG@10 | 0.1226 / 0.0950 | 0.1188 / 0.0912 | 0.1176 / 0.0940 |
| swap KL | 0.00043 | 0.00198 | **0.00378** [0.00294, 0.00471] |
| mean-replacement KL | 0.00029 | 0.00108 | 0.00306 [0.00228, 0.00391] |
| drop-last KL (`r` recomputed / `r` fixed) | 0.073 / 0.073 | 0.067 / 0.064 | 0.0708 / 0.0707 |
| **swap / drop-last ratio** | 0.0059 | 0.029 | **0.053** |
| swap ΔNLL | n.s. | +0.0020 | **+0.0038** [0.0027, 0.0051] |
| swap KL at the first target token | 0.0044 | 0.032 | **0.110** |
| mask KL (first token) | 0.0039 (0.023) | 0.42 (2.38) | 0.61 (11.6) |
| attention on `r` / avg prompt token | 10.6× | 765× | 10.3× |
| ‖r‖; cosine to mean | 79; 0.72 | 243; 0.95 | 104; 0.93 |

- **The decoder-side incentive produces the largest content sensitivity measured on Musical Instruments.** The swap/drop-last ratio is 9× v1 and 31× v0; swapping `r` costs +0.0038 NLL; the first target token shifts by KL 0.11 (25× v1). It does so without inflating ‖r‖ or turning `r` into a sink (unlike category-align), consistent with H5 being a symptom rather than a requirement.
- **The effect is still small in absolute terms.** A swap still moves the output ~19× less than dropping one history item.
- **`r` still does not transmit history changes.** Recomputed and fixed `r` give the same drop-last shift (0.0708 vs 0.0707). The decoder became sensitive to *which* `r` it receives, but `r` does not track the history better than before. That is expected: the margin loss rewards distinguishing thoughts, not encoding information absent from `x` (H1).
- **Cost:** HR@10 −4% relative to v1 (0.1176 vs 0.1226), NDCG@10 −1%. This replicates the CDs pattern (margin 0.2: real but modest content-dependence at a small ranking cost).

This fills the raw-`r` / margin cell of the H3 2×2 (see [Next steps](#next-steps)).

## Hypotheses

### H1. `r` carries no information beyond the history (redundancy → no incentive)

*Claim.* `r = f(x)` is computed from the same history the decoder reads directly. It cannot contain information that `x` lacks, so under next-token CE the decoder's best policy is to predict from `x`. Any use of `r` could only be computational (a precomputed summary), and the pretrained decoder already reads item titles well.

*Evidence for.*
- In v0, v1 and collaborative InfoNCE, recomputing `r` after a history change gives the same output as keeping the old `r` (§1). `r` adds nothing on top of `x`.
- On CDs, the only interventions that produced content-dependence either blocked the history path (the attention curricula) or directly penalized ignoring `r` (the margin loss).

*Evidence partly against / open.*
- Category-align shows `r` can start to carry part of a history change, and be read, when it has a target-oriented objective, even though it contains no information beyond `x` (§5). So redundancy does not fully prevent use; it removes the incentive.
- A decisive test with information genuinely absent from `x` has not been run.

- The margin loss (§7) makes the decoder distinguish thoughts (swap/drop-last 0.053, the highest measured), yet recomputed and fixed `r` still give the same output after a history change. Incentive alone changes *how much* the decoder reacts to `r`, not *what* `r` contributes.

- **Decisive test ([Experiment 1](#experiment-1)).** When the slot carries information absent from the history, the decoder reads it heavily:
  - the target-category oracle: swap ratio 12–14, 89% of the top-10 steered to the injected category (text: 92%);
  - a SASRec CF vector: swap ratio 1.10, HR@10 −27% when swapped.

  An uninformative vector in the same slot is ignored (ratio 0.001).

*Verdict: confirmed.* The latent slot is ignored because its content is redundant with the history, not because the decoder cannot read it.

### H2. The decoder does not attend to `r`

*Verdict: rejected.* `r` receives 2.8–10.6× the attention of an average prompt token in plain and collaborative checkpoints, 765× in category-align (§2). Insensitivity coexists with high attention in every checkpoint. More latent tokens or more bandwidth, which address attention routing, are therefore low priority.

### H3. `r`'s norm is too large

*Claim.* `r` is 70–240× larger than an input token embedding. At that scale its residual stream is effectively frozen (attention/MLP updates of normal size cannot change it), and it behaves like a "massive activation" token. LLMs tend to use such tokens as fixed anchors rather than as information.

*Evidence.*
- The mismatch is built in from step 0 (‖r‖ ≈ 89 at initialization) and never corrected by training (§6). The COCONUT paper reports the same off-manifold scale for fed-back hidden states.
- The largest-norm checkpoint (category-align, 243) is the one whose `r` became a 44% attention sink.
- Within the observed range, larger norm does not mean less use; if anything the opposite. Swap/drop-last rises with norm from v1 (‖r‖ 79, 0.0059) to collaborative InfoNCE (205, 0.0073) to category-align (243, 0.029), with v0 (100, 0.0017) the exception. The likely confound is that the higher-norm checkpoints are exactly those with an `r`-side objective. On CDs, large-norm `r` was still used when the loss demanded it (contrastive V0 at ‖r‖ ≈ 343 showed the strongest content-dependence there).

*Open.* Every checkpoint is deep in the ≫1 regime, where the proposed mechanisms would already be saturated. So the within-range comparison cannot rule this hypothesis out. Only a model trained with `r` at token scale can.

*Experiment 1.*
- The target-category oracle is read equally well at raw scale (swap ratio 13.6, 89% steering) and at token scale (12.4, 89%). The scale mismatch does not prevent use of informative content.
- Rescaling plain R3 to token scale (`plain_token`) leaves content use at zero (ratio 0.001) and removes even the presence dependence (mask KL 0.0001).

*Verdict: rejected as the cause.* The scale mismatch is real and architectural (§6) but is not what blocks use of `r`.

### H4. `r` is collapsed

*Verdict: rejected as the main cause.* v1's `r` is the most diverse (72% sample-specific, cosine 0.72 to the mean) and is still insensitive. Category-align is nearly as collapsed as v0 yet the most sensitive (§4). Diversity of `r` does not track its use.

### H5. The decoder attends to `r` but treats it as a constant anchor

*Claim.* What a target position receives from `r` in one layer is `α · W_O W_V · RMSNorm(r)`. High attention weight α says nothing about whether `W_O W_V` preserves the directions in which `r` differs between users. The decoder can use `r` as a fixed reference point (sink/anchor) whose shared direction matters and whose sample-specific part is projected away.

*Evidence.*
- Category-align puts 44% of attention on `r` yet responds to swaps only weakly.
- v0 collapses when `r` is zeroed (KL₁ = 10) but ignores which valid `r` it gets: the signature of a needed anchor rather than an information source.
- Mean replacement is as harmless as the true `r` in every checkpoint.

*Open.* No direct measurement yet of the per-layer value contribution `α · W_O W_V · RMSNorm(r)` and of how much it changes under a swap.

*Verdict: a consequence, not a blocker.* In Experiment 1 the same slot transmits informative content almost as well as text (oracle steering 89% vs 92%). The anchor/sink behaviour is how the decoder handles a redundant latent (H1).

### H6. The history shortcut is learned first and locks in

*Claim.* At initialization `r` is an arbitrary pooled vector, while the pretrained decoder already reads item titles well. Early training therefore optimizes the history path, and once CE is low from `x` there is little gradient left to make `r` useful.

*Evidence (CDs, from [the collapse investigation](avoid_reasoning_collapse_and_shortcut_learning.md)).*
- Training schedules that never expose the decoder to the unrestricted path (alpha=0.75, 7b) produce content-dependence.
- Every schedule that trains on the unrestricted path, even for a third of one epoch (three-way curriculum) or mostly (alpha=0.9), loses it.

*Experiment 1 refinement.* Lock-in concerns redundant content. The oracle models became *more* dependent on the slot with training (steering 68–76% at epoch 1 → 89% at epoch 3), so training does not erode use of information the decoder needs.

*Verdict: supported indirectly for redundant `r`; not measured on Musical Instruments for plain R3.* The CDs curriculum mechanism (mode-conditional policy) is explained in the [Discussion](#discussion). The diagnostic could be run on the intermediate SFT checkpoints (519/1038/1557) to see whether content sensitivity is ever higher early in training.

### H7. `r` is off the embedding manifold in direction, not only in scale

*Claim.* Even rescaled to token norm, `r` (a pooled output-space vector through a free linear map) may point in directions the pretrained input circuits never see, so they cannot interpret it.

*Verdict: rejected as a blocker ([Experiment 1](#experiment-1)).* An unconstrained learned projection into the slot, off the embedding manifold, carries the oracle almost as effectively as the same information in text (89% vs 92% steering), so there is little room for on-manifold constraints to improve channel efficiency. Rescaling alone (H3's test) does not address direction. A direction test would constrain `r` to the input-embedding manifold, e.g. as a mixture of vocabulary embeddings (the repository's `latent_soft_vocabulary` option; the CDs soft-vocabulary run did not complete).

### H8. The insensitivity is a measurement artefact

*Verdict: rejected.* The E1 null results could in principle reflect the coarseness of top-10 ranking. Logit-level KL over the full vocabulary, per target token, confirms the insensitivity with tight confidence intervals (§1).

<a id="experiments"></a>
## Experiments

Experiments designed to discriminate between the [hypotheses](#hypotheses). Each has a design, the expected results and how to read them, a progress tracker, and results.

<a id="experiment-1"></a>
### Experiment 1: side-information injection (H1 vs H3/H5/H7)

**Question.** If the latent slot carried information the decoder cannot get from the history text, would the decoder use it? Every checkpoint so far has `r = f(x)`, so it cannot separate:

- **redundancy (H1):** the decoder ignores `r` because `r` adds nothing;
- **architecture (H3/H5/H7):** the decoder cannot use a spliced vector, because of its norm, its sink/anchor role, or its position off the embedding manifold.

These tests put information into the slot that `x` lacks.

#### Design

All variants are plain R3 SFT on Musical Instruments (Qwen2.5-1.5B-Instruct, 3 epochs, effective batch 128, lr 3e-4, early stopping and best-checkpoint selection as in v0/v1). Only the thought vector changes:

```
r' = r + g · normalize(P(v))          (optionally rescaled: r' ← normalize(r') · ‖e‖_token)
```

- `v` is a per-example side vector;
- `P` is a learned linear map into the hidden space;
- `g` is a scalar initialized at 30, comparable to the sample-specific part of `r` (‖r − mean_r‖ ≈ 25–60 in plain R3).
  - It was declared learnable but **stayed at exactly 30.0 in every run**. Weights are stored in bfloat16, whose resolution near 30 is 0.125, while each AdamW update at lr 3e-4 is ~10⁻³, so every update rounds away.
  - In effect the side component is a fixed-norm vector `30 · normalize(P(v))`: its direction is learned through `P`, its size is not.
  - The models balanced the slot through the base `r` instead. The oracle and SASRec runs shrank ‖r‖ from ≈175 to 18–25, letting the side component dominate; `null_raw` kept ‖r‖ ≈ 181, keeping its random component relatively small.
- `‖e‖_token` = 1.014 is the mean input-embedding norm.

The same component is added at training and at test time. Code: `src/latent/side_info_latent.py`, `side_info_dataset.py`, options in `src/latent/latent_attention_train.py`; launcher `scripts/side_info_musical.sbatch <variant>`.

| variant | side vector `v` | scale | role |
|---|---|---|---|
| `oracle_raw` | gte-large embedding of the **target item's** category path (412 categories) | raw (‖r‖ ≈ 10²) | **H1 test**: privileged information through the slot. Deliberately leaky: ranking is an upper bound, only sensitivity/steering count |
| `oracle_token` | same | token scale (‖r'‖ = 1.014) | **H3 test**: the same information at input-embedding scale |
| `text_control` | none; the prompt gets the line "The category of the item to recommend is: {path}" | — | **positive control**: the same information through text, which the decoder can certainly read. Gives the ceiling |
| `null_raw` | category of a **random** item (resampled every forward pass) | raw | **null control**: same architecture, no information. Shows what "ignored" looks like |
| `sasrec_raw` | frozen SASRec sequence embedding of the row's history (128-d, Cornac SASRec trained on the train split only, validation NDCG@10 0.0953) | raw | **realistic test**: collaborative knowledge learned from other users' item-ID co-occurrences, absent from title text. No leak: ranking is valid |
| `plain_token` | none | token scale | **H3 cell of the 2×2**: plain R3 with `r` rescaled to token norm |

**Why category for the oracle.** It narrows the catalog without identifying the item (412 categories). It is only weakly predictable from the history (category of the last history item matches 14.4% of the time; §5), so it is genuinely new to the decoder. The target item's own embedding would be too strong: the decoder could copy it.

**What the CF vector adds.** It is still computed from the history, but from item IDs combined with co-occurrence statistics learned across all users. It adds population-level collaborative knowledge the LLM does not have from titles, not new facts about this user. It is weaker than the oracle but legitimate.

#### Measurements

Evaluation script `src/prototype_grounding/side_info_eval.py`, test split:

1. **Teacher-forced output shift** (first 2,000 examples), KL(raw ‖ variant) on the target title as in §1. Conditions:
   - `swap_side`: side component from a donor example with a different user and a different target category (or, for the text control, the donor's category written in the prompt);
   - `swap_r`: donor's base `r`, own side component;
   - `swap_all`: donor's full `r'`;
   - `mask`;
   - the one-item history drop (`drop_last_recomputed_r`) as the reference.

   Reported as the ratio to the history drop.
2. **Steering by generation** (first 1,000 examples, constrained 10-beam search), with the true and the donor side information:
   - HR@10 / NDCG@10;
   - fraction of the top-10 in the **own** vs the **donor's** target category;
   - hit rate on the donor's target item.
3. **Full-test ranking** (8,313 examples) with the true side information.

#### Expected results and how to read them

Reference values from existing checkpoints:
- swap/drop-last ratio: 0.0017–0.0073 (plain), 0.029 (category-align), 0.053 (margin 0.2);
- HR@10: 0.1231 (v0), 0.1226 (v1).

| outcome | reading |
|---|---|
| `text_control` steers strongly (top-10 follows the donor category; large swap_side ratio), **and** `oracle_raw` does too, far above `null_raw` | **H1 confirmed.** The slot works when it carries non-redundant information; redundancy was the problem. Go to realistic information sources (CF). |
| `text_control` steers, `oracle_raw` does not, `oracle_token` does | **H3 confirmed.** The scale mismatch blocks use even of valuable information; fix the scale first. |
| `text_control` steers, neither oracle variant does | **Architecture (H5/H7).** A single spliced vector is a poor channel regardless of scale; consider token-space latents (soft vocabulary) or a structural bottleneck. |
| `text_control` barely steers either | Category is too weak a probe on this data; use a stronger oracle (e.g. an item cluster). |
| `sasrec_raw` swap_side ratio ≫ 0.007 and HR@10 above v0/v1 | The decoder reads collaborative information through `r`: a realistic path to making the latent useful. |
| `sasrec_raw` ≈ plain while the oracle works | The CF knowledge is redundant with what the LLM already infers from titles, or too weak to change the target distribution. |
| `plain_token` swap ratio ≫ plain raw (0.002–0.006) | Rescaling alone increases use of `r`'s content (H3), without any information change. |

Quantitative expectations under H1: `oracle_raw` swap_side ratio of the same order as the text control (≥ 0.1, i.e. comparable to dropping a history item), `null_raw` ≈ 0. Top-10 in the donor category ≫ the true-condition rate of top-10 in the donor category.

#### Progress

Check status and the latest numbers with `python scripts/side_info_progress.py` (SLURM state, epochs done and eval loss per variant, key results once written). Results go to `analysis/results/side_info/full_<variant>/summary.json`; checkpoints to `scratchDirectory/R3/output_dir/Musical_Instruments_R3_side_<variant>/latent`.

| variant | job | checkpoint location | status |
|---|---|---|---|
| `oracle_raw` | 304903 (rerun, 3 epochs from scratch) | scratch `…/Musical_Instruments_R3_side_oracle_raw` | completed (best: epoch 3); epoch-1 model and results kept as `…_oracle_raw_epoch1` (job 304780) |
| `oracle_token` | 304904 (rerun, 3 epochs from scratch) | scratch `…/Musical_Instruments_R3_side_oracle_token` | completed (best: epoch 3); epoch-1 model and results kept as `…_oracle_token_epoch1` (job 304781) |
| `plain_token` | 304717 | scratch, `…_side_plain_token` | completed (best: epoch 2) |
| `null_raw` | 304718 | repo `output_dir/Musical_Instruments_R3_side_null_raw` | completed (best: epoch 2) |
| `text_control` | 304719 | repo `output_dir/Musical_Instruments_R3_side_text_control` | completed (best: epoch 2) |
| `sasrec_raw` | 304720 | repo `output_dir/Musical_Instruments_R3_side_sasrec_raw` | completed (best: epoch 2) |

*Run history.*
- First submission: 304550/304551 and 304552–304555. The four pending jobs were resubmitted as 304576–304579 to allow RTX 3090s.
- All six then failed while saving the epoch-2 checkpoint: the scratch directory hit its 500 GB quota ("Disk quota exceeded"), since each full checkpoint is ~9 GB with optimizer state.
- Epoch-1 eval loss before the failure: `oracle_raw` 0.284, `oracle_token` 0.303, `plain_token` 0.467.
- Fix: checkpoints are now weights-only with `save_total_limit=1` (best + latest, ~3 GB each); the three runs with a complete epoch-1 checkpoint resume from it; the other three run in the repo `output_dir` (home quota).
- The two oracle resumes (304715/304716) then ran out of GPU memory on RTX 3090s (24 GB) in the CE loss at micro-batch 2. They were resubmitted with micro-batch 1 on 3090s (`MICRO_BATCH_3090=1`; effective batch unchanged at 128).
- The resubmitted oracle resumes (304780/304781) completed but **trained no further steps** (train runtime 2 s, loss 0.0). The checkpoint was saved with micro-batch 4; resuming with micro-batch 1 made HF Trainer 4.48 treat training as finished. Their final models are the epoch-1 checkpoints, and their results are reported as epoch-1 results. The epoch-1 directories were renamed to `…_epoch1`, and clean runs without resuming (`RESUME=0`) were submitted as 304903/304904.

Each job trains for 3 epochs (≈3–4 h on one L40S/A40, longer on a 3090, where micro-batch 2 keeps the effective batch at 128), then runs `side_info_eval.py` with `--full_ranking` (≈1 h).

Validation before submission: every variant passed a smoke test (256 training examples or 0.2% of an epoch, 16 evaluation examples), with 100% of generated titles matching the catalog. The smoke runs caught and fixed three evaluation bugs:
- a relative category-file path;
- catalog titles containing tab characters;
- a prompt fragment leaking into decoded predictions under left padding.

Smoke numbers are not results.

#### Results

All six variants completed (results in `analysis/results/side_info/full_<variant>/summary.json`).

**Training length.** All variants trained 3 epochs with best-eval-loss checkpoint selection:
- `text_control`, `null_raw`, `sasrec_raw` and `plain_token` are evaluated at their best epoch, 2;
- the two oracle runs (clean reruns 304903/304904; eval loss 0.313 → 0.215 → 0.214 raw, 0.323 → 0.217 → 0.213 token) are evaluated at epoch 3.

The first oracle attempt only produced epoch-1 models (see run history). Their results are kept in `full_oracle_{raw,token}_epoch1` and shown in italics for comparison. More training made the oracles *more* effective, not less: steering 68–76% → 89%, and the gap between raw and token scale closed.

**1. Ranking results.** Standard HR@K and NDCG@K on the full test set (8,313 examples), each example with its own (true) side information:

| variant | what the latent slot carries | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_control` (leaky) | nothing extra; target category written in the prompt | 0.2152 | 0.3687 | 0.4634 | 0.6160 | 0.2152 | 0.3040 | 0.3431 | 0.3923 |
| `oracle_raw` (leaky) | target category, raw scale | 0.2187 | 0.3550 | 0.4469 | 0.5922 | 0.2187 | 0.2970 | 0.3348 | 0.3816 |
| `oracle_token` (leaky) | target category, token scale | 0.2179 | 0.3544 | 0.4475 | 0.5965 | 0.2179 | 0.2964 | 0.3346 | 0.3827 |
| `oracle_cfcluster150` (leaky) | target's SASRec cluster, k = 150 (6.90 bits, entropy-matched to category; Exp. 2) | 0.2041 | 0.3344 | 0.4245 | 0.5926 | 0.2041 | 0.2786 | 0.3156 | 0.3695 |
| `oracle_cfcluster` (leaky) | target's SASRec cluster, k = 412 (8.13 bits; Exp. 2) | 0.3023 | 0.5125 | 0.6424 | 0.8215 | 0.3023 | 0.4235 | 0.4767 | 0.5348 |
| `oracle_cfitem` (leaky) | target's SASRec item embedding (≈ identity; Exp. 2) | 0.9767 | 0.9820 | 0.9840 | 0.9854 | 0.9767 | 0.9799 | 0.9808 | 0.9812 |
| *`oracle_raw`, epoch-1 model* | *same* | *0.1696* | *0.2869* | *0.3610* | *0.4766* | *0.1696* | *0.2370* | *0.2676* | *0.3049* |
| *`oracle_token`, epoch-1 model* | *same* | *0.1635* | *0.2698* | *0.3313* | *0.4299* | *0.1635* | *0.2247* | *0.2500* | *0.2817* |
| `sasrec_raw` | SASRec vector of the user's history | 0.0713 | 0.0895 | 0.1001 | 0.1196 | 0.0713 | 0.0817 | 0.0860 | 0.0923 |
| `null_raw` | category of a random item (no information) | 0.0739 | 0.0917 | 0.1026 | 0.1231 | 0.0739 | 0.0841 | 0.0885 | 0.0951 |
| `plain_token` | plain R3 thought, rescaled to token norm | 0.0740 | 0.0907 | 0.1009 | 0.1216 | 0.0740 | 0.0835 | 0.0877 | 0.0944 |
| *reference: plain R3 v0 (E1 pipeline)* | *plain R3 thought* | *0.0749* | *0.0914* | *0.1036* | *0.1231* | *0.0749* | *0.0845* | *0.0895* | *0.0957* |
| *reference: plain R3 v1 (standard eval)* | *plain R3 thought* | *0.0747* | *0.0901* | *0.1020* | *0.1226* | *0.0747* | *0.0836* | *0.0884* | *0.0950* |
| *reference: margin 0.2 (§7, standard eval)* | *plain R3 thought, margin loss* | *0.0755* | *0.0908* | *0.1010* | *0.1176* | *0.0755* | *0.0845* | *0.0887* | *0.0940* |

*How to read it.* HR@K is the fraction of test examples whose target item appears in the generated top-K list (constrained 10-beam generation over catalog titles). NDCG@K additionally rewards placing the target higher (discount 1/log2(rank+1)); with a single target, HR@1 = NDCG@1. "Leaky" marks runs whose side information describes the target item (its category or its SASRec representation): their scores show how much that information helps, not a legitimate recommendation result.

What the table shows:
- **Category information in the slot is used almost as fully as in the prompt.** HR@10 rises from ≈0.12 (plain R3) to 0.59–0.60 with the category in the latent slot, versus 0.62 with it written as text. Raw and token scale give the same result.
- **CF information about the target is as useful as category information at equal information content** ([Experiment 2](#experiment-2)). The entropy-matched CF clustering (`oracle_cfcluster150`) matches the category oracle (HR@10 0.5926 vs 0.5922). The finer CF clustering (k = 412, more information) reaches 0.82. The target's own SASRec embedding reaches 0.985, but by memorizing a vector→title lookup for items seen as training targets: HR@10 is 0.994 on those and 0.000 on the 92 test examples whose target never appeared as a training target.
- **The realistic CF vector does not help ranking.** `sasrec_raw` (0.1196) is at or slightly below plain R3 (0.1226–0.1231) at every cutoff.
- **Uninformative or redundant slot content changes nothing.** `null_raw` and `plain_token` match plain R3.

Whether the decoder actually *reads* the slot is shown by swapping in another user's side information (table 2): HR@10 collapses for the category variants (0.65 → 0.03) and drops 27% for SASRec, while the random-category model is unaffected.

**2. Where the recommendations go when the information is swapped (steering)** (first 1,000 test examples, constrained 10-beam search; top-10 category shares use the 412 category paths):

| variant | HR@10 true → swapped | top-10 in own category (true) | top-10 in **donor** category (true → swapped) | donor's target in top-10 (true → swapped) |
|---|---|---:|---|---|
| `text_control` | 0.665 → 0.015 | 92.3% | 0.0% → **92.4%** | 0.0% → **51.4%** |
| `oracle_raw` | 0.647 → 0.026 | 88.3% | 0.1% → **89.3%** | 0.2% → **46.9%** |
| `oracle_token` | 0.634 → 0.026 | 88.6% | 0.1% → **89.1%** | 0.0% → **46.0%** |
| *`oracle_raw`, epoch-1 model* | *0.544 → 0.026* | *74.1%* | *0.3% → 75.5%* | *0.0% → 39.9%* |
| *`oracle_token`, epoch-1 model* | *0.502 → 0.049* | *65.6%* | *0.2% → 68.1%* | *0.1% → 35.0%* |
| `sasrec_raw` | 0.157 → **0.115** | 7.2% | 2.4% → 4.7% | 1.6% → 3.6% |
| `null_raw` | 0.153 → 0.156 | 7.2% | 2.2% → 2.3% | 1.9% → 1.9% |
| `plain_token` | 0.154 | 7.1% | — | — |

*How to read table 2.* Each example gets a real top-10 recommendation list by constrained 10-beam generation over catalog titles, as in the standard evaluation. This is run twice: with its **true** side information, and with the **donor's** side information swapped in (**swapped**).

| column | definition | what it shows |
|---|---|---|
| **HR@10 true → swapped** | fraction of examples whose own target item is in the top-10, with true vs donor side information | how much ranking quality depends on the side information being the right one |
| **top-10 in own category (true)** | average fraction of the 10 recommended items whose category path equals the example's own target category, with the true side information | how strongly the list concentrates on the target's category. Without side information it is ≈ 7% (`null_raw`, `plain_token`) |
| **top-10 in donor category (true → swapped)** | the same fraction, but for the donor's target category: incidental overlap with true information, and after swapping in the donor's information | **steering**: whether the recommendation list follows the injected information. Near 0% → ~70–90% means the list is redirected wholesale to the injected category |
| **donor's target in top-10 (true → swapped)** | fraction of examples whose list contains the *donor's* target item | how specifically the injected information redirects the list, beyond the category level |

For the category variants, "swapped" puts in a different category. For `sasrec_raw` it puts in another user's SASRec history vector. `null_raw`'s side vector is random by design, so its "swapped" condition is a sanity check that should change nothing. `plain_token` has no side information to swap.

**3. Finer-grained diagnostic: output-distribution shift.** HR@10 cannot show effects that are real but too small to change the top-10, such as plain R3's latent in §1. This diagnostic, introduced in [Method](#method), measures how much the model's predicted distribution over the target title changes under each intervention, compared with the change from removing one history item (2,000 test examples; KL over target tokens with 95% CI, and the ratio to the one-item history drop):

| variant | swap side info | ratio | swap base `r` | ratio | mask | history drop (`r` recomputed / fixed) |
|---|---:|---:|---:|---:|---:|---:|
| `text_control` (category in prompt; epoch 2) | **1.134** [1.101, 1.166] | **28.2** | 0.0001 | 0.003 | 0.601 | 0.0401 / 0.0401 |
| `oracle_raw` (best: epoch 3) | **0.832** [0.810, 0.856] | **13.6** | 0.0012 | 0.019 | 1.262 | 0.0611 / 0.0606 |
| `oracle_token` (best: epoch 3) | **0.814** [0.793, 0.837] | **12.4** | 0.0010 | 0.015 | 0.628 | 0.0655 / 0.0648 |
| *`oracle_raw`, epoch-1 model* | *0.534* | *16.8* | *0.0002* | *0.008* | *0.439* | *0.0318 / 0.0302* |
| *`oracle_token`, epoch-1 model* | *0.445* | *12.1* | *0.0005* | *0.013* | *0.282* | *0.0368 / 0.0367* |
| `sasrec_raw` (epoch 2) | **0.061** [0.059, 0.064] | **1.10** | 0.0025 | 0.044 | 0.049 | 0.0556 / 0.0557 |
| `null_raw` (epoch 2) | 0.0001 | 0.001 | 0.0001 | 0.001 | 0.445 | 0.0670 / 0.0669 |
| `plain_token` (epoch 2) | — | — | 0.0001 | 0.001 | 0.0001 | 0.0664 / 0.0664 |
| *reference: v1 plain R3* | — | — | *0.00043* | *0.006* | *0.004* | *0.073 / 0.073* |
| *reference: margin 0.2 (§7)* | — | — | *0.0038* | *0.053* | *0.605* | *0.071 / 0.071* |

Swapping the side information also costs likelihood: ΔNLL +1.13 (text), +0.82 (oracle raw), +0.80 (oracle token), +0.034 [0.030, 0.038] (SASRec); null +0.0000.

*How to read table 3.*

- **Setup.** For each test example, the model is fed the prompt plus the true target title (teacher forcing). We compare its next-token distributions over the target-title tokens under the normal input ("raw") and under one modified input.
- **Thought vector.** It is `r' = r + side component`:
  - `r` is the **base thought**, pooled from the history as in plain R3;
  - the **side component** is `g·normalize(P(v))`, computed from the example's side vector `v` (target category embedding, SASRec vector, or a random category).
- **Donor.** Another test example with a different user and (except for SASRec) a different target category.

| column | what is changed | what it measures |
|---|---|---|
| **swap side info** | only the side component is replaced by the donor's; the base `r` is kept. For `text_control`, the category line in the prompt is replaced by the donor's category. | how much the output depends on *which* side information is supplied |
| **swap base `r`** | only the base `r` is replaced by the donor's; the own side component is kept. For plain/text runs (no side component) this is the whole thought vector. | how much the output depends on the history-derived part of the thought, the quantity §1 measured for plain R3 |
| **mask** | the whole thought vector is replaced by zeros | dependence on something being present in the slot (it mixes content and "out-of-distribution input" effects; see §1) |
| **history drop (`r` recomputed / fixed)** | the most recent history item is removed from the prompt; the thought is either recomputed from the shortened prompt or kept at its original value | the reference scale: the effect of a small real change to the evidence. Equal values mean the recomputed `r` adds nothing to what the history tokens already convey |

Values are the mean over examples of the per-token **KL divergence KL(p_raw ‖ p_modified)** over the full vocabulary, averaged over the target-title tokens. 0 means the output distribution is unchanged. Brackets are bootstrap 95% confidence intervals over examples.

**Ratio** = KL of that intervention ÷ KL of the history drop (`r` recomputed) for the same checkpoint:
- ≈ 0: the intervention barely matters compared with removing one history item;
- ≈ 1: it matters as much as a history item;
- ≫ 1: it matters far more.

Plain R3 checkpoints sit at 0.002–0.006 (§1).

**ΔNLL** = mean negative log-likelihood of the true target title under the modified input minus under the raw input. Positive means the modification makes the correct answer less likely.

**Reading, against the expectation table:**

- **H1 confirmed (row 1 of the table).**
  - The decoder reads the thought slot heavily when it carries information the history lacks. Swapping in another user's oracle category drops HR@10 by 96% (0.65 → 0.03), and moves the output distribution 12–14× as much as dropping a history item.
  - The swap redirects 89% of the top-10 into the injected category, versus 92% when the same information is given as text. Leaky HR@10 reaches 0.59–0.60, versus 0.62 for text.
  - The same architecture with an uninformative side vector (`null_raw`) is ignored completely (ratio 0.001, no steering), exactly like plain R3.
  - So the insensitivity of plain R3 is not an inability to read the slot. It is the absence of anything worth reading.
- **H3 rejected as the cause.**
  - The oracle works equally well at raw scale (‖r‖ ≈ 10²) and at token scale: ratio 13.6 vs 12.4, steering 89% vs 89%, HR@10 0.592 vs 0.597. At epoch 1 the token-scale model lagged (68% vs 76%); with full training the difference disappears.
  - Rescaling plain R3 to token scale (`plain_token`) does not increase content use (swap ratio 0.001). It even removes the presence dependence (mask KL 0.0001): at token scale the redundant `r` is ignored outright.
- **H5 / H7 rejected as blockers.**
  - A single spliced, off-manifold vector transmits information almost as well as text: oracle steering 89% vs 92%, donor-target hits 46–47% vs 51%.
  - The anchor/sink behaviour of plain R3 is how the decoder handles a *redundant* latent, not a limit of the channel.
  - The remaining gap to text is small, so channel efficiency is not a practical limitation.
- **The history-derived part of `r` stays unused in every variant.** Swapping the base `r` while keeping the side component gives ratios of 0.003–0.019, and 0.044 with SASRec.
  - The oracle models shrank the base `r` to norm ≈ 19–25 (versus 175–180 in `null_raw` / `plain_token`), letting the side component (norm 30) dominate the slot.
  - Recomputing `r` after a history change still adds nothing (drop-last recomputed ≈ fixed everywhere).
- **Collaborative information is read but does not improve ranking.**
  - `sasrec_raw` uses its CF component as much as a history item (ratio 1.10). Swapping it costs 27% relative HR@10 on the 1,000-example subset (0.157 → 0.115) and doubles hits on the donor's target.
  - Yet full-test HR@10 is 0.1196, not above plain R3 (0.1226–0.1231).
  - The decoder relies on the SASRec signal *instead of* part of what it would otherwise infer, rather than *in addition to* it. For ranking, the CF knowledge is largely substitutable for what the LLM already gets from titles. So the problem moves from "the decoder ignores the latent" to "what non-redundant information improves the ranking".

<a id="experiment-2"></a>
### Experiment 2: quality of CF vs category information (oracle ceiling)

**Question.** If the target item's information were available (an oracle ceiling, not a realistic setting), which representation of the target helps the decoder more: its **category** or its **collaborative-filtering (CF) representation**? This compares information *quality* at the ceiling. It complements the realistic finding that the history-based SASRec vector is read but does not improve ranking (Experiment 1). That finding is explained by SASRec and R3 largely knowing the same things: on the full test set, 81% of SASRec's hits@10 are also R3 hits, the two hit the same examples 6.6× more often than chance, and a perfect combination could add at most +1.7 points of HR@10.

#### Design

Identical to the `oracle_raw` run of Experiment 1 (same model, training, raw scale, evaluation). Only the per-item lookup table that supplies the target's side vector changes. Tables are built by `src/latent/build_sasrec_oracle_tables.py` from the Cornac SASRec checkpoint (trained on the training split) and indexed by the dataset's item ID (5,010 of 5,030 items are in SASRec's catalog).

| variant | slot content (for the **target** item) | groups | target's group: mean / median size | entropy of target group |
|---|---|---:|---:|---:|
| `oracle_raw` (Experiment 1) | gte embedding of its category path | 412 categories | 57 / 42 items | 6.83 bits |
| `oracle_cfitem` | its own SASRec item embedding (128-d, normalized) | item-level | 1 | ≈ identity |
| `oracle_cfcluster` | centroid of its SASRec cluster, spherical k-means with **k = 412** (same number of groups as categories) | 412 | 15 / 13 | 8.13 bits |
| `oracle_cfcluster150` | same with **k = 150**, chosen so the entropy matches the category oracle | 150 | 39 / 35 | 6.90 bits |

**Why two cluster variants.** Categories are very unbalanced (a few categories contain many targets), while k-means clusters are balanced. At the same k = 412, the CF cluster therefore carries more information (8.13 vs 6.83 bits; it narrows the target to ~15 candidates instead of ~57). `oracle_cfcluster` compares at equal *granularity*; `oracle_cfcluster150` compares at equal *information*, which isolates the quality of the grouping itself.

For reference, how predictable each group is from the history: the target shares the last history item's category 11.6% of the time, and its k = 412 CF cluster 5.8% of the time.

#### Expected results and how to read them

All results are leaky ceilings. The primary metric is full-test HR@10 / NDCG@10; swap/drop-last and steering are measured as in Experiment 1. The category oracle reached HR@10 0.592.

| outcome | reading |
|---|---|
| `oracle_cfitem` ≫ all others (near-perfect HR@10) | expected: an item-level embedding lets the decoder decode the target's identity. It confirms the channel's capacity, not the value of CF as a signal |
| `oracle_cfcluster` > `oracle_raw` | expected from information content alone (8.13 vs 6.83 bits); not by itself evidence that CF is better |
| `oracle_cfcluster150` > `oracle_raw` (equal information) | co-purchase groupings are a **more useful** kind of target information than semantic categories. A realistic predictor of the target's CF cluster is the more promising direction |
| `oracle_cfcluster150` ≈ `oracle_raw` | at equal information, the type of grouping matters little; usefulness is governed by how much the side information narrows the candidates |
| `oracle_cfcluster150` < `oracle_raw` | semantic categories align better with how the LLM ranks titles; category-level side information is the more promising direction |

#### Progress

| variant | job | checkpoint location | status |
|---|---|---|---|
| `oracle_cfitem` | 306733 | repo `output_dir/Musical_Instruments_R3_side_oracle_cfitem` | completed |
| `oracle_cfcluster` | 306734 | repo `output_dir/Musical_Instruments_R3_side_oracle_cfcluster` | completed |
| `oracle_cfcluster150` | 306743 | repo `output_dir/Musical_Instruments_R3_side_oracle_cfcluster150` | completed |

The first submissions (306727/306728) were cancelled after a minute because the scratch directory was at 486/500 GB; the runs write to the repo `output_dir` (home quota) instead. Each run takes ~12–13 h on a 3090 at micro-batch 1 (as for the Experiment 1 oracle reruns), plus evaluation. `python scripts/side_info_progress.py` includes these variants.

#### Results

All three runs completed. Best eval loss:
- `oracle_cfitem` at epoch 2 (0.109 → 0.045 → 0.045);
- `oracle_cfcluster` at epoch 3 (0.293 → 0.173 → 0.170);
- `oracle_cfcluster150` at epoch 2 (0.306 → 0.214 → 0.219).

Category oracle for reference: 0.313 → 0.215 → 0.214. Results are in `analysis/results/side_info/full_oracle_{cfitem,cfcluster,cfcluster150}/summary.json`. All numbers are leaky ceilings.

| oracle (target item's …) | information | full-test HR@10 | NDCG@10 | swap side info: KL (ratio to history drop) | swap ΔNLL | HR@10 true → swapped (first 1,000) | donor's target in top-10 after swap |
|---|---:|---:|---:|---:|---:|---|---:|
| category (`oracle_raw`) | 6.83 bits | 0.592 | 0.382 | 0.832 (13.6) | +0.82 | 0.647 → 0.026 | 46.9% |
| **CF cluster, k = 150** (equal information) | 6.90 bits | **0.593** | **0.370** | 0.606 (14.2) | +0.60 | 0.660 → 0.039 | 47.8% |
| CF cluster, k = 412 (equal granularity) | 8.13 bits | 0.822 | 0.535 | 0.896 (13.8) | +0.88 | 0.844 → 0.037 | 72.9% |
| CF item embedding | ≈ identity | 0.985* | 0.981 | 1.943 (273) | +1.94 | 0.987 → 0.005 | 97.6% |

\*Memorized lookup: 0.994 on targets seen as training targets, 0.000 on the 92 unseen ones (see reading). (The item-level ratio is inflated by a tiny denominator: when the target's identity is known, dropping a history item barely matters, KL 0.007.)

**Reading.**
- **At equal information, the kind of grouping does not matter.** The entropy-matched CF clustering and the semantic category give the same HR@10 (0.593 vs 0.592) and the same steering (donor-target hits 47.8% vs 46.9%). NDCG@10 is slightly lower for CF (0.370 vs 0.382), i.e. categories place the target marginally higher within the top-10. This is row 4 of the expectation table: co-purchase groupings are not a more useful *kind* of target information than semantic categories, nor less.
- **Usefulness tracks how much the information narrows the candidates.** HR@10 rises with the information content of the oracle: 0.59 at ~6.9 bits (category or CF-150), 0.82 at 8.1 bits (CF-412), 0.99 at item identity. The CF-412 advantage over category (+0.23 HR@10) is explained by information content, not quality: its clusters are balanced and leave ~15 candidates instead of ~57.
- **The item-level CF embedding is a memorized lookup, not general decoding.** Its HR@10 of 0.985 splits sharply by whether the test target ever appears as a *training* target (`oracle_seen_unseen.py`, job 307428):
  - 0.994 on a random sample of 500 test examples whose target was a training target;
  - **0.000** on all 92 test examples whose target never was.

  The model learned a lookup table "this SASRec vector → this title" for items seen as training targets, and the overall score is high because 98.9% of test targets are also training targets. It is a valid leaky ceiling ("if the target's identity were known"), but not evidence that identity can be decoded from an unfamiliar embedding. The group-level oracles (category, CF clusters) are shared across many items, so they cannot be memorized item by item in the same way.
- **In every variant the history-derived base `r` shrinks and stays unused.** Base-`r` norm is 7.7 (item), 9.6 (CF-412), 17.5 (CF-150) and 25.2 (category), versus ~175 without an oracle, and swap-base-`r` ratios are 0.01–0.06. The more informative the side vector, the more the model suppresses the redundant part of the slot.

**Implication.** Oracle quality is decided by information content, not by whether the grouping is semantic or collaborative. For a *realistic* method, the question becomes which target grouping can be **predicted from the history accurately enough**. At equal information, the target shares the last history item's category more often than its CF cluster (11.6% vs 5.8% at k = 412), which suggests categories may be easier to predict, though that is a weak heuristic. The degraded-oracle dose-response (true group with probability *p*, random otherwise) would give the accuracy a group predictor must reach to beat R3.

<a id="discussion"></a>
## Discussion: from a latent that is *used* to one that is *useful*

### What "oracle in the slot" means, and how to read its gains

The oracle variants of [Experiment 1](#experiment-1) add information about the **answer** to R3's latent slot, the `<|Thought|>` position where `r` is spliced in:

```
r' = r + g · normalize(P(e_cat(target)))
```

`e_cat(target)` is the gte-large embedding of the target item's category path (one of 412). It is added at training *and* test time, so at test time the model is told the category of the item it should recommend. It is called an oracle because this information is unavailable to a real recommender.

The oracle is a **probe of the channel**, not a candidate method: it asks whether the decoder *can* use the slot when the slot contains something the prompt does not. Category was chosen because it is informative without giving the item away, and it is genuinely new to the decoder (the last history item's category matches the target's only 14% of the time).

Full-test ranking (8,313 examples):

| model | HR@10 | NDCG@10 | HR@10 vs plain R3 |
|---|---:|---:|---:|
| plain R3 (v1) | 0.1226 | 0.0950 | — |
| `null_raw` (random category in the slot) | 0.1231 | 0.0951 | +0.4% |
| **`oracle_raw`** (target category in the slot) | **0.5922** | **0.3816** | +383% (≈ 4.8×) |
| **`oracle_token`** (same, token scale) | **0.5965** | **0.3827** | +387% (≈ 4.9×) |
| `text_control` (target category in the prompt) | 0.6160 | 0.3923 | +402% (≈ 5.0×) |
| `sasrec_raw` (SASRec vector in the slot) | 0.1196 | 0.0923 | −2.4% |

**The oracle gains are not improvements.** They are leaky: the model is partly told the answer. What they measure is **how much the decoder can extract from the latent slot**: nearly as much as from the same information written as text (0.59–0.60 vs 0.62). The only legitimate variant with new information, SASRec, does not improve ranking.

### Being used is necessary, not sufficient

| | latent is *used* | accuracy *improves* |
|---|---|---|
| plain R3 `r` | no | — |
| oracle in the slot | yes | yes, only because it leaks the answer's category |
| SASRec vector in the slot | yes (swapping it costs 27% HR@10 on the probed subset) | no (0.1196 vs 0.1226) |
| input masking / strict interface in the literature (below) | yes, by construction | not claimed |

Information asymmetry reliably makes a latent *used*. It improves accuracy only if the latent adds something the unrestricted decoder could not get (complementary information) or could not compute in one pass (extra computation). Current R3 provides neither: `plain_token`, whose `r` has no effect at all, scores the same as R3 (0.1216 vs 0.1226).

### What the information-asymmetry papers actually aim at

Both papers cited as remedies 1–2 in [Related work](#related-work) remove the decoder's direct access to the input. Neither removes *older* information specifically: that is our adaptation to R3. Their goal is **necessity / faithfulness**, not accuracy:

- **Viveiros et al., "What's Holding Back Latent Visual Reasoning?"** ([arXiv:2605.18445](https://arxiv.org/abs/2605.18445)).
  - The latents represent intermediate images (crops of the input image).
  - Hypothesis: they are bypassed because their content "is easily recoverable from the input directly".
  - They mask the input regions the crops correspond to, keeping the crops as latent supervision.
  - This is explicitly a **diagnostic**: it "deviates from realistic settings" and serves to tell failures of the modelling approach from failures of the data.
  - With masking, latent interventions matter (correct latents ~75–79% vs ~59–61% with noise, zeros or random latents; two readings of the paper gave 75% and 79% for the correct condition). That is not above the unmasked model, which gets ~79% whatever the latents.
  - Their lesson is about data design: latents matter only when their content cannot be recovered from the input.
- **Park et al., "Reason Through the Latent! Making Latent Visual Reasoning Necessary" (CVRR)** ([arXiv:2609.06746](https://arxiv.org/abs/2609.06746)).
  - Motivation: "a latent state containing task-relevant visual information is not necessarily one that the model actually uses", because the decoder keeps "alternative image-conditioned routes". For a claim that prediction depends on latent computation, informativeness is insufficient evidence.
  - They enforce the dependence architecturally: before decoding, the visual states and multimodal KV cache are removed, so the answer reaches the image only through a recurrent latent state.
  - Crucially, that latent **re-reads the full image at every recurrent step**. The latent keeps full access to the evidence; only the decoder loses it.
  - Under this strict interface CVRR scores 81.2% on V* and 52.7% MMVP pair accuracy, while other latent reasoners retrained under the same constraint reach 35.6–39.8% and 0.7–2.7%. Cost: ~10% latency for T=1→4.
  - The headline is that CVRR *retains* performance where others collapse, not that it beats unrestricted access.

Two lessons for R3:
1. Removing the decoder's access to information makes the latent necessary. By itself it can only match, not beat, a decoder that sees everything.
2. A bottleneck is only affordable when the latent has enough **capacity**. CVRR's latent is a recurrent computation that repeatedly re-reads the evidence. R3's `r` is one 1536-d vector from a single pooling step, so a full bottleneck ("decoder sees only `r`") would likely lose ranking: the CDs `reasoning_only` training cost ~10% HR@10.

### Why the CDs curriculum failed, and why a partial bottleneck differs

In the CDs attention curriculum ([Attempt 1](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-1)), each training batch used one of three decoder attention modes:
- `reasoning_only`: `r` visible, history not;
- `history_only`: history visible, `r` not;
- `both`: normal R3.

Inference always used `both`.

| CDs variant | trained on `both`? | content-dependence at inference? |
|---|---|---|
| original schedule (final epoch all `both`) | yes | none |
| alpha capped at 0.75 | never | yes, weak (mean/shuffle −0.007 HR@10), with `r` not net-helpful and −10% HR@10 |
| final epoch with ⅓ `both` batches | a little | none |
| 7b with alpha 0.9 | mostly | almost none |
| no curriculum | always | none |

The modes are trivially distinguishable from the input structure. So the decoder learned a **mode-conditional policy**: use `r` when the history is blocked, ignore it when the history is available. Inference is always "history available". Blocking made `r` non-redundant only *during training*; at test time `r` was redundant again (H1).

A **partial bottleneck** removes the mode switch. The decoder always sees only the last *k* history items, at training and at inference, while `r` is pooled from the full history. At test time the older items are still reachable only through `r`, so `r` stays non-redundant. The price is a truncated prompt, and the question it tests is whether `r` recovers what the truncation removes.

### Two ways to frame the goal

1. **Accuracy over R3.** The latent must add real capability:
   - *complementary* information: the SASRec result shows information that merely substitutes for title evidence does not help;
   - or *computation* the decoder cannot do in one pass, e.g. CVRR-style recurrent refinement of `r` that re-attends to the history hidden states.

   A partial bottleneck alone does not deliver this; it only tests whether `r` can carry history information.
2. **Faithfulness or efficiency at matched accuracy.** Information asymmetry is naturally useful here. For example, `r` compresses the older history so the decoder prompt holds only the last *k* items: shorter prompts, faster inference, histories longer than the context. As for CVRR, "the latent is used and accuracy matches full access" is then a real result.

Which framing the project adopts decides which experiments to prioritize (see [Next steps](#next-steps)).

## Related work

The phenomenon studied here, where latent reasoning tokens are informative or present but not used by the decoder, has been reported independently in several recent latent-reasoning papers across text reasoning, multimodal reasoning and recommendation. One paper names it the **latent bypass** problem. This section summarizes what they found, the causes they identify, and the remedies they propose, mapped onto the hypotheses above.

*Verification note.* LaRec, LASAR, "Final Checkpoints Are Not Enough" and "Do Latent Tokens Think?" were read from the full PDF text. The other entries are based on the papers' HTML/abstract pages; their exact numbers are less thoroughly verified.

### Papers reporting the same problem

| paper | setting | finding | cause identified | our hypothesis |
|---|---|---|---|---|
| Zhang et al., **"Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought"** ([arXiv:2512.21711](https://arxiv.org/abs/2512.21711)) | COCONUT, QA/reasoning, 7–8B models | swapping latent tokens across samples leaves MMLU accuracy unchanged (60.9% → 61.0%) while swapping explicit CoT drops it (62.8% → 43.4%); steering latents succeeds <10% of the time vs up to 50–94% for CoT; latents lie far from the vocabulary embedding manifold | speculative: latents encode whatever minimizes the answer loss most efficiently ("pseudo-reasoning") | H1, H3 (descriptive) |
| Viveiros et al., **"What's Holding Back Latent Visual Reasoning?"** ([arXiv:2605.18445](https://arxiv.org/abs/2605.18445)) | visual latent reasoning (LVR, Monet, ILVR, LanteRn) | replacing latents with zeros/noise/random crops keeps accuracy within ~2 points; even *oracle* latents do not help. Named the **"latent bypass problem"** | the information in the latents is "easily recoverable from the input image itself"; plus latent collapse (pairwise similarity 0.8–0.98) | **H1**, H4 |
| Zhang et al., **"Visual Latents Know More Than They Say: Unsilencing Latent Reasoning in MLLMs"** ([arXiv:2605.02735](https://arxiv.org/abs/2605.02735)) | multimodal LLMs | latents become semantically richer during training while their contribution to the answer is suppressed | "the autoregressive loss rapidly exploits shortcuts grounded in the visual input, while simultaneously driving latent tokens toward a transition-token-like role" | **H1, H5**, H6 |
| Park et al., **"Reason Through the Latent! Making Latent Visual Reasoning Necessary"** ([arXiv:2609.06746](https://arxiv.org/abs/2609.06746)) | visual LLMs | existing latent reasoners are not relied on when alternative pathways to the input exist | "visual information being present in a latent state does not imply that the model actually relies on that state" | H1; consistent with rejecting H4 |
| Cui et al., **"How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?"** ([arXiv:2602.22441](https://arxiv.org/abs/2602.22441), ICLR 2026) | Coconut-style latent reasoning | "pervasive shortcut behavior, where they achieve high accuracy without relying on latent reasoning" | supervision strength: "stronger supervision mitigates shortcut behavior but restricts the ability of latent representations to maintain diverse hypotheses" | H1, H6 |
| Jin et al., **"Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories"** ([arXiv:2607.06648](https://arxiv.org/abs/2607.06648)) | COCONUT, CODI on ProsQA / GSM | task accuracy rises while counterfactual responsiveness falls during training (final responsiveness: COCONUT 0.17, CODI 0.04, explicit CoT 0.73); sensitivity to norm-matched noise replacement declines alongside it | training dynamics; diagnostic only, no remedy | **H6** |
| Chen & Fu, **"Latent Recurrent Thoughts"** ([arXiv:2609.01117](https://arxiv.org/abs/2609.01117)) | SoftCoT, EBM-CoT with frozen LLMs | both "collapse to predicting a near-uniform latent across instances" and fall below a no-reasoning baseline on Countdown-4 (5.9% / 8.4% vs 27.8%) | the generic latent proposer ignores instance-specific information | H4 |

The pattern replicates across modalities, model families (1.5B–8B), single vs multiple latent tokens, and autoregressively fed-back vs pooled latents. This supports treating the R3 result as an instance of a general property of latent tokens trained only through output likelihood, not a quirk of R3.

The literature converges with this document on three points:

- **The input shortcut is the primary cause (H1).** Several papers show that latent information recoverable from the input is not used, and that it becomes used once it is not recoverable (below).
- **Informative ≠ used (H4 rejected).** Viveiros et al. and Park et al. both show that making latents informative does not by itself make them causal, matching v1 and collaborative InfoNCE here.
- **The latent is pushed into a structural role (H5).** "Transition-token-like role" (Zhang et al., Unsilencing) is the same phenomenon as the anchor/sink behaviour observed here (v0's catastrophic `mask` effect; category-align's 44% attention sink).

### Remedies proposed, mapped to our hypotheses

**1. Structural bottleneck: remove the input path so information must flow through the latent (targets H1, H6).** This is the most direct remedy and the one with the clearest causal evidence.
- *Park et al. (CVRR)* initialize a recurrent latent from the question state, update it while re-reading the image, and then "remove visual states and original multimodal KV cache before decoding", so only the final recurrent state carries image information to the answer. They report strong performance under this strict interface, with predictions causally sensitive to the latent content.
- *Viveiros et al.* mask the input regions that the latents correspond to. Latent interventions then matter: correct latents ~75–79% (two readings of the paper disagree) vs 59–61% under noise/zero/random replacement.
- *Relation to our work.* The CDs attention curricula (alpha=0.75, [7b](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-7b)) are a soft, training-only version of this. They produced content-dependence but lost it whenever the unrestricted path returned. The literature suggests a **permanent partial bottleneck** at inference as well, e.g. a decoder that sees `r` plus a truncated history rather than the full history.

**2. Put information in the latent that the input does not contain (targets H1).**
- *Viveiros et al.* design a synthetic rotation task whose intermediate states cannot be recovered from the input. Oracle latents then give 86% vs 25–26% with dummy latents, and 34% for a pause-token baseline.
- *Relation to our work.* This is the privileged-information oracle and collaborative-filtering-embedding test proposed in [Next steps](#next-steps). The [category-alignment pilot](#5-category-alignment-pilot-does-a-target-oriented-r-get-used) did not achieve it, because `r` remained a function of the history.

**3. Step-level supervision of latents against explicit-reasoning hidden states (targets H1/H6 via a stronger signal).** This is the approach taken by the two recommendation papers building on this line of work:
- *LaRec* (Xia et al., "LaRec: Unleashing LLM-based Latent Reasoning for Generative Recommendation", [arXiv:2607.24617](https://arxiv.org/abs/2607.24617)) cites R3 and criticizes its Gaussian-noise exploration ("blind random perturbations are prone to disrupting the semantic structure … leading to a collapse of reasoning logic"). It uses:
  - an InfoNCE "step-level alignment" between each latent step and the hidden state of the corresponding explicit CoT step;
  - a "process direction alignment" constraining successive latent steps to point toward the target;
  - personalized RL with exploration anchored in the user's historical interests.
- *LASAR* (Chen et al., "LASAR: Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation", [arXiv:2605.10207](https://arxiv.org/abs/2605.10207)) describes R3 as "only shallow single-step attention over history sequences". It diagnoses "representation drift" of unsupervised latents and uses step-wise **bidirectional KL** between the softmax-normalized activation profiles of latent hidden states and CoT-text hidden-state anchors, followed by GRPO with terminal-only KL and an adaptive-depth policy head. Two results are directly relevant here:
  - on **Instruments**, adding latents *without* alignment hurt (NDCG@10 0.0576 vs 0.0634 without latents), and KL alignment recovered it (0.0643);
  - on Beauty, MSE alignment was worse than no alignment (−11.6% NDCG@10) and cosine alignment neutral.
- *Caveats.* Neither paper intervenes on the latents at test time. Their ablations retrain without the latent component. So they show the method helps ranking, not that the decoder reads the latent's content; the swap/drop-last diagnostic of this document would test that. Cui et al. also warn that stronger supervision reduces shortcuts at the cost of latent diversity.

**4. Keep the latent on the input-embedding manifold (targets H3/H7).**
- *Soft Thinking* (Zhang et al., [arXiv:2505.15778](https://arxiv.org/abs/2505.15778), NeurIPS 2025) forms latent "concept tokens" as probability-weighted mixtures of token embeddings: on-manifold and at token scale by construction (training-free).
- *SoftCoT* (Xu et al., [arXiv:2502.12134](https://arxiv.org/abs/2502.12134), ACL 2025) maps assistant-model hidden states into the LLM's embedding space through a trainable projection module.
- *Chang et al., "Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention"* ([arXiv:2606.01243](https://arxiv.org/abs/2606.01243)) report weight tying as essential for COCONUT/CODI latents. They gain ~1.3–1.8 points by decode-time interventions that include projecting latents onto the token manifold.
- **No paper found explicitly measures the scale mismatch between fed-back hidden states and input embeddings as a cause of insensitivity.** These methods avoid it implicitly. The normalized-`r` experiment (H3) would address an open question.

**5. Anti-collapse by construction (targets H4).** *Latent Recurrent Thoughts* replaces the generic proposer with a small task-dedicated encoder and refines its latents through bounded residual corrections (`L⋆ = L⁽⁰⁾ + Δ`, with a λ‖Δ‖² penalty) fed to a frozen LLM. This is close in spirit to [Attempt 9](avoid_reasoning_collapse_and_shortcut_learning.md#attempt-9) (baseline-preserving residual correction).

**6. Inference-time latent optimization (sidesteps H6).** *Zhang et al. (Unsilencing)* freeze the backbone and optimize only the latents per instance: a query-guided contrastive warm-up, then natural-evolution-strategies RL with a confidence-progression reward. They report gains of up to +4–9 points on individual benchmarks. It avoids the training-time shortcut but adds per-query optimization cost.

**7. Checkpoint selection / monitoring (targets H6).** *Jin et al.* recommend evaluating faithfulness along the training trajectory rather than at the final checkpoint only. This matches the proposed use of the swap/drop-last ratio as a training-time monitor.

### Evidence-strength caveats

- **Zhang et al. (COCONUT):**
  - the swap result rests on 1,000 MMLU questions with one model and no confidence intervals;
  - steering success is judged by GPT-4o;
  - one sentence in its results contradicts its own Table 1 (attributing ≥50% steering success to COCONUT where the table shows it for CoT);
  - several references appear mis-cited (a computer-vision geometry paper, a wake-word detection paper and a signal-processing paper cited for LLM claims).
- **Recommendation papers (LaRec, LASAR):** they report ranking gains from retraining ablations only. Neither measures whether predictions depend on the latent's content at test time.
- **Remedies 1–2:** they come from visual reasoning, where the "input path" is a set of image tokens that can be masked or dropped. In R3 the history is also the primary evidence for the recommendation, so a bottleneck risks the ranking cost already seen with the CDs curricula. A partial bottleneck is the natural middle ground to test.

<a id="next-steps"></a>
## Next steps

Ordered by expected value, taking the [related work](#related-work) into account:

**Update after [Experiment 1](#experiment-1).** H1 is confirmed and H3/H5/H7 are not the blockers, so the priority shifts from "make the decoder read `r`" to "put information in `r` that is both absent from the title history *and* improves the ranking". The SASRec vector is read heavily but does not raise HR@10 (0.1196 vs 0.1226). Candidates:
- combine the CF vector with a decoder-side incentive (margin loss, §7);
- use CF signals less substitutable for title reasoning (e.g. item-item co-occurrence neighbours of the history, popularity/recency statistics, interactions outside the 10-item prompt window);
- test whether the SASRec-augmented decoder complements rather than replaces title evidence, e.g. by ensembling its scores with plain R3.

Items 1 and 3 below are answered; the others remain relevant.

1. **H1, decisive test: privileged-information oracle.** *Done ([Experiment 1](#experiment-1)): H1 confirmed with fully trained (3-epoch) oracle models.* Add a learned projection of the *target's* category embedding into `r`, at both training and test time. The leak makes ranking meaningless, but the diagnostic still applies:
   - if swap sensitivity jumps by orders of magnitude, redundancy is confirmed as the driver;
   - if the decoder still ignores `r` despite privileged information, the cause is architectural (H3/H5/H7).

   This mirrors Viveiros et al.'s non-recoverable-information design. The realistic follow-up adds collaborative-filtering user/item embeddings from an ID-based model trained on training interactions: information not in the title text, and no leak.
2. **H1/H6 remedy: partial bottleneck at train *and* inference.** The literature's most successful fix (Park et al.'s CVRR; Viveiros et al.'s masking) removes the input path so information must flow through the latent. The full bottleneck (`reasoning_only`) is too costly for recommendation, where the history is the primary evidence. Test a decoder that sees `r` plus a truncated history (e.g. the last k items), with the same mask used at inference, so there is no curriculum train/test mismatch of the kind that erased the CDs gains.
3. **H3: normalized-`r` 2×2.** *Mostly answered ([Experiment 1](#experiment-1)): scale is not the blocker (`plain_token`, `oracle_token`). The remaining token-scale + margin cell is low priority.* On Musical Instruments: {raw `r`, `r` rescaled to input-embedding scale before splicing} × {plain CE, + margin loss}, scored by the swap/drop-last ratio. The raw/CE cell exists (v0, v1). The raw/margin cell is done (§7: swap/drop-last 0.053, HR@10 0.1176). The token-scale/margin cell remains to be run once the `plain_token` result is in. No paper found tests the scale mismatch directly.
4. **CoT-anchored alignment (LASAR-style).** Align `r` to the hidden states of an explicit rationale with bidirectional KL (not MSE, which LASAR found harmful). Evaluate with the swap/drop-last diagnostic, which neither LaRec nor LASAR report. This requires generating rationales for the training data.
5. **H5: per-layer value contribution.** Measure `α · W_O W_V · RMSNorm(r)` at target positions, its norm relative to history tokens' contributions, and its change under a swap.
6. **H6: sensitivity over training.** Run the diagnostic on intermediate checkpoints (519/1038/1557) of v0/v1, following Jin et al.'s trajectory analysis.
7. **H7: on-manifold `r`.** Constrain `r` to the input-embedding manifold (a Soft-Thinking-style mixture of token embeddings, or the repository's `latent_soft_vocabulary` option) and compare with the rescaled-only variant from step 3.
8. **Monitoring.** For `r`-side auxiliary losses, track the value-projection gain; they consistently inflate ‖r‖ (§6).

**Success criterion.** The ratio `KL(shuffle) / KL(drop_last)` is a cheap (~6 min per checkpoint on 2,000 examples), generation-free measure of whether the decoder uses `r`'s content. Current values are 0.0017 (v0), 0.0059 (v1), 0.0073 (collaborative InfoNCE), 0.029 (category-align) and 0.053 (margin 0.2). It should rise by orders of magnitude for a method that genuinely fixes the problem. `mask`-based measures should not be used for this purpose: v0 vs v1 shows they reflect incidental presence-dependence, not use of content.

## Artifacts

- `analysis/results/decoder_sensitivity/musical_v0_ckpt1557/summary.json`, `per_example.npz` (job 304347)
- `analysis/results/decoder_sensitivity/musical_v1/summary.json`, `per_example.npz` (job 304346)
- `analysis/results/decoder_sensitivity/musical_collaborative_infonce/summary.json`, `per_example.npz` (job 304348)
- `analysis/results/decoder_sensitivity/musical_category_align/summary.json`, `per_example.npz` (job 304396), `category_probe.json` (job 304397)
- `analysis/results/decoder_sensitivity/latent_norm_trace.json` (job 304411)
- `analysis/results/decoder_sensitivity/musical_margin_m0.2/summary.json`, `per_example.npz` (job 304754; checkpoint from job 304312)
- Experiment 1 results: `analysis/results/side_info/full_<variant>/summary.json` for `oracle_raw` (job 304903), `oracle_token` (304904), `text_control` (304719), `null_raw` (304718), `sasrec_raw` (304720), `plain_token` (304717); epoch-1 oracle results in `full_oracle_{raw,token}_epoch1` (304780/304781)
- Experiment 2 seen/unseen check: `analysis/results/side_info/full_oracle_cfitem/seen_unseen.json` (job 307428; `src/prototype_grounding/oracle_seen_unseen.py`)
- Experiment 2 results: `analysis/results/side_info/full_oracle_{cfitem,cfcluster,cfcluster150}/summary.json` (jobs 306733, 306734, 306743); tables `data/Musical_Instruments/info/Musical_Instruments_sasrec_{item,cluster412,cluster150}.pt` built by `src/latent/build_sasrec_oracle_tables.py`
- Experiment 1 code: `src/latent/side_info_latent.py`, `src/latent/side_info_dataset.py`, options in `src/latent/latent_attention_train.py`, `src/prototype_grounding/side_info_eval.py`; launchers `scripts/side_info_musical.sbatch`, `scripts/side_info_eval_only.sbatch`; progress check `scripts/side_info_progress.py`
- Code: `src/prototype_grounding/decoder_sensitivity.py`, `category_probe.py`, `latent_norm_trace.py`; launchers `scripts/decoder_sensitivity_musical.sbatch`, `scripts/category_probe_musical.sbatch`, `scripts/latent_norm_trace.sbatch`
