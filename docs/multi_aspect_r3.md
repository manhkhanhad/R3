# Multi-Aspect R3 with Confidence-Gated Fusion

> Related (in the main checkout): `analysis/document/decoder_insensitivity_to_latent.md` (why the decoder ignores history-derived latents), `analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md`.

## Idea

Instead of a single latent thought `r`, R3 pools the history with K separate attention heads. Each produces a latent `r_k` that "reasons" about one aspect of the next item and is trained with its own loss. The aspect latents are fused into one thought vector, which the decoder uses for the recommendation as in plain R3.

**Expectation from the insensitivity analysis.** Every `r_k` is still computed from the history alone, so the decoder was expected to ignore the fused latent (hypothesis H1 of the insensitivity document). The aspect heads' *predictions* were expected to be useful when applied directly at ranking time, independently of the decoder.

## Design

Code on branch `multi-aspect-r3` (worktree `.claude/worktrees/multiAspectR3`):
- `src/latent/build_aspect_tables.py`, `multi_aspect_attention.py`, `multi_aspect_model.py`;
- options in `latent_attention_train.py`;
- evaluation `src/prototype_grounding/multi_aspect_eval.py`, `item_head_rerank.py`;
- launchers `scripts/multi_aspect_musical.sbatch`, `scripts/item_head_rerank.sbatch`.

**Aspects** (Musical Instruments, 5,030 items):

| aspect | classes | label source |
|---|---:|---|
| `item` | 5,030 | the target item itself (as in R3) |
| `category` | 412 | leaf category path |
| `sid_l1` | 32 | semantic id level 1: k-means on the Qwen3 title embeddings (PCA 64) |
| `sid_l2` | 512 | semantic id level 2: residual k-means inside each level-1 cluster (all 512 codes used) |

**Per-aspect latents and losses.** One `SelfAttentionLayer` pooler per aspect (as in R3) gives `r_k`; a head `LayerNorm → Linear` gives logits over that aspect's classes. Loss:

```
L = CE_recommendation(decoder | fused r) + 0.1 · Σ_k CE_k(head_k(r_k), code_k(target))
```

**Confidence-gated fusion.**

```
c_k = 1 − H(softmax(logits_k)) / log C_k          (confidence in [0,1], detached)
g   = softmax(softplus(a) · c + b)                 (a > 0: more confident → more weight)
r   = Σ_k g_k r_k
```

The confidences are detached so the recommendation loss cannot make a head overconfident to win the gate. The ablation `fusion=mean` uses `g_k = 1/K`.

**Implementation notes (bugs caught before the reported runs):**
- *Gradient-accumulation scale.* The model's `forward` takes `**kwargs`, so HF Trainer passes `num_items_in_batch` and does not divide the loss by gradient accumulation. The recommendation CE normalizes itself, but the aspect loss was multiplied 128× (logged loss 330 in the smoke test). It is now scaled by `1/grad_accum_steps`, as in Attempt 2. `category_alignment_attention.py` has the same pattern, so the earlier category-alignment run likely used an effective alignment weight far above its nominal 0.1.
- *bf16 gate parameters.* R3 trains in pure bf16. Near the gate scale's value (≈2.1) bf16 steps are ≈0.016, far larger than an update (≈3e-4), so the gate scale never moved. `gate_raw_scale` and `gate_bias` are now kept in fp32 (the first pair of runs was restarted for this).

**Runs.** Qwen2.5-1.5B-Instruct, 3 epochs, effective batch 128, lr 3e-4, best checkpoint by eval loss, RTX 3090 (micro-batch 1). Checkpoints: main checkout `output_dir/Musical_Instruments_R3_multiaspect_{confidence,mean}`.

| run | job | best epoch (eval loss) |
|---|---|---|
| confidence gate | 305053 | 2 (2.330) |
| uniform mean | 305054 | 3 (2.337) |

Eval loss includes the 0.1-weighted aspect losses, so it is not comparable with plain R3's.

## Results

Result files: main checkout `analysis/results/multi_aspect/full_{confidence,mean}/`.

### 1. Standard ranking through the decoder (10-beam, full test, 8,313 examples)

| model | HR@10 | NDCG@10 |
|---|---:|---:|
| plain R3 v1 (reference) | **0.1226** | **0.0950** |
| multi-aspect, confidence gate | 0.1201 | 0.0911 |
| multi-aspect, mean | 0.1179 | 0.0905 |

Through the decoder alone, the multi-aspect latent does not help (−2% HR@10).

### 2. The decoder ignores the fused latent

Swap diagnostic (first 2,000 test examples; KL of the target-title distribution when a latent is taken from another user, as a ratio to dropping one history item):

| model | swap fused `r` | swap one aspect `r_k` (item / category / sid_l1 / sid_l2) |
|---|---:|---|
| confidence gate | 0.0053 | 0.0019 / 0.0019 / 0.0019 / 0.0017 |
| mean | 0.0081 | 0.0032 / 0.0032 / 0.0031 / 0.0029 |
| *plain R3 (v0 / v1)* | *0.0017 / 0.0059* | — |

This matches plain R3: splitting the latent into history-derived aspects does not give it information the decoder lacks, so H1 applies unchanged.

### 3. The aspect heads learn real signal

Top-1 accuracy of each head on the target's code (first 2,000 test examples), against history-only rules:

| aspect | confidence-gate head | mean head | code of last history item | majority code in history |
|---|---:|---:|---:|---:|
| item | **10.3%** | 9.9% | 0.9% | 0.2% |
| category | **18.0%** | 16.7% | 14.4% | 10.8% |
| sid_l1 | **25.7%** | 25.2% | 22.1% | 20.5% |
| sid_l2 | 13.6% | 13.8% | 11.5% | 8.1% |

Every head beats the history heuristics. The item head predicts the exact next item 10.3% of the time, above the decoder's own HR@1 (≈ 0.070 for these models). It is effectively a strong classifier-style recommender.

### 4. The confidence gate is nearly uniform

| | item | category | sid_l1 | sid_l2 |
|---|---:|---:|---:|---:|
| mean gate | 0.233 | 0.289 | 0.236 | 0.243 |
| mean confidence | 0.26 | 0.37 | 0.27 | 0.28 |
| gate when head correct / wrong | 0.251 / 0.231 | 0.276 / 0.291 | 0.245 / 0.233 | 0.252 / 0.241 |

- The learned scale moved only from 2.13 to 2.15 and the biases stayed at ±0.01.
- Gates are barely higher when the head is right, and for category they are *lower*. Entropy-based confidence does not track correctness here.
- The gate therefore behaves almost like the uniform mean, consistent with the two runs' near-identical results.

### 5. Beam re-scoring with the aspect heads

20 decoder beams per example, reranked by `beam score + λ · Σ_k g_k log p_k(code_k(item))`; λ chosen on 2,000 validation examples (best NDCG@10) and applied to the full test set:

| model | decoder only (top-10 of 20 beams) | re-scored, all aspects (λ = 0.1) | re-scored, item head only |
|---|---|---|---|
| confidence gate | 0.1191 / 0.0890 | **0.1244 / 0.0962** | 0.1282 / 0.0994 |
| mean | 0.1184 / 0.0891 | 0.1207 / 0.0934 | 0.1246 / 0.0959 |

Values are HR@10 / NDCG@10; plain R3 v1 is 0.1226 / 0.0950.

- Re-scoring improves the confidence-gate model by +4.5% HR@10 and +8% NDCG@10, to slightly above plain R3.
- The item head alone gives most of the gain. The "item only" choice is post hoc (λ was tuned for the all-aspect combination); §6 tunes it on validation.
- **Ceiling:** the target is in the 20-beam list for only 14.9% of test examples, so beam re-ranking cannot recover items the decoder never generates.

### 6. Item head over the full catalog

Jobs 307493 / 307494; results in `analysis/results/multi_aspect/full_{confidence,mean}/item_head_rerank.json`.

**Setup.**
- Candidate set per example = item-head top 50 ∪ decoder 10-beam list.
- Each candidate is scored by the decoder's teacher-forced mean title log-likelihood `d`. This is validated: `d` ranks the decoder's own top beam first in 98% of test examples.
- Reranking uses `d + λ · log p_item` (or `+ λ · Σ_k g_k log p_k` for all aspects), with λ chosen on 2,000 validation examples.
- The item head alone ranks the full catalog.

Full test set (8,313 examples), HR@10 / NDCG@10:

| method | confidence-gate model | mean model |
|---|---|---|
| decoder 10-beam (standard) | 0.1196 / 0.0907 | 0.1174 / 0.0902 |
| **item head alone, full catalog** | **0.1287 / 0.0986** | 0.1223 / 0.0950 |
| decoder + item head (λ = 0.3) | **0.1306 / 0.0996** | 0.1214 / 0.0954 |
| decoder + all aspects, gated (λ = 0.2 / 0.05) | 0.1268 / 0.0974 | 0.1202 / 0.0920 |
| candidate recall (head top-50 ∪ beams) | 23.5% | 22.6% |
| *plain R3 v1* | *0.1226 / 0.0950* | |
| *SASRec (same split, Cornac)* | *0.0978 / 0.0802* | |

- **The item head alone is the best single ranker.** For the confidence-gate model it beats that model's own decoder (+7.6% HR@10), plain R3 (+5.0% HR@10, +3.8% NDCG@10) and SASRec (+32% HR@10). A single linear classifier on a pooled LLM representation, trained jointly with the other aspects, ranks better than generating titles.
- **Adding the decoder helps a little.** Combining the decoder with the item head (λ = 0.3) is the best overall: 0.1306 / 0.0996, which is +6.5% HR@10 and +4.8% NDCG@10 over plain R3.
- **The category/SID aspects add nothing on top of the item head.** Adding them through the gate is worse than the item head alone, consistent with the gate being uninformative (§4).
- **Gated vs mean training.** The mean-fusion model's item head is weaker (0.1223), and combining does not help it. Whether this reflects the gating or seed-to-seed variance is unknown with one run each.
- **Not yet verified:**
  - significance: needs a paired bootstrap against plain R3 per-example ranks, not yet computed;
  - whether the multi-aspect training matters: an R3 model trained with only the item head, or plain R3 plus a post-hoc item head, would isolate the contribution of the other aspects.

## Conclusions

1. The multi-aspect structure does not make the latent reasoning path useful. The decoder ignores the fused latent, as H1 predicts, and decoder-only ranking is slightly below plain R3.
2. The aspect heads learn predictive signal beyond simple history rules. The item head in particular ranks the full catalog better than the decoder, plain R3 and SASRec.
3. The best result comes from using the item head's scores directly: alone 0.1287 / 0.0986, combined with decoder likelihood 0.1306 / 0.0996, versus plain R3 0.1226 / 0.0950. The value of the design is in the heads' predictions at ranking time, not in the latent.
4. Entropy-confidence gating adds nothing measurable, and the category/SID aspects do not add to the item head in re-scoring. A gate that predicts head correctness, or validation-tuned per-aspect weights, would be needed.
5. All comparisons are single runs. A paired bootstrap against plain R3 and a second seed are needed before claiming an improvement.

## Next steps

1. **Significance:** paired bootstrap of per-example ranks, item head (± decoder) vs plain R3 v1.
2. **Ablations:**
   - item-head-only training (drop category/SID aspects);
   - a post-hoc item classifier on a frozen plain-R3 checkpoint.

   These tell whether joint multi-aspect training is what makes the item head strong.
3. **A second seed** for the confidence-gate run.
4. **Better gating:** train the gate to predict whether each head's top-1 is correct, instead of using entropy.
