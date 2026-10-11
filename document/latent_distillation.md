# Latent distillation: putting reasoning information into R3's latent

## Motivation

R3's decoder ignores the thought vector `r` because `r` holds nothing the prompt lacks
(analysis/document/decoder_insensitivity_to_latent.md, H1). That document's results
narrow the fix:

- **Non-redundant information in the slot is read.**
  - Oracle category: swap/drop-last ratio 13.6.
  - SASRec history vector: ratio 1.10.
- **It must also be complementary to improve ranking.** SASRec replaced title evidence
  instead of adding to it: HR@10 0.1196 vs 0.1226.
- **`r` is a function of the prompt.** It can only carry what the prompt contains, plus
  computation the decoder does not do in one pass.

Text-reasoning recommenders get that extra computation and knowledge by writing reasoning
tokens:
- each token is a full forward pass;
- larger models bring knowledge (SLIM, RDRec, KAR);
- R²ec learns reasoning by RL without a teacher.

There is no ground-truth latent to imitate. **Latent distillation** builds one from text:
a teacher R3 whose `r` is pooled from history + an LLM-written rationale, while its decoder
never sees the rationale. The teacher's `r` must then carry whatever the rationale adds,
and that becomes the target for a history-only student.

## Design

### Stage 0: rationales (`src/latent_distill/generate_rationales.py`)
- **Writer:** Qwen2.5-7B-Instruct (vLLM, greedy, ≤192 tokens). It reads only the history
  titles, never the target.
- **Content:** 3–5 sentences: what the customer is doing (instrument, level, project), what
  they need next and why, and the kind of product (type, features, brand), with no
  specific title.
- **Coverage:** one rationale per row of train/valid/test.
- **No leak:** the teacher is a legitimate, deployable recommender, since the rationale
  needs only the history.

### Stage 1: teacher (`RationaleLatentModel`, `src/latent/rationale_latent.py`)

    pool sequence:    R3 prompt with "### Analysis: <rationale>" appended to the input, then <|Thought|>
    decoder sequence: plain R3 prompt, <|Thought|> (= r pooled from the pool sequence), target

- **Recipe:** v1's, from Qwen2.5-1.5B-Instruct: lr 3e-4, batch 128, 3 epochs, early
  stopping. Cutoff is 1024 instead of 512.
- **Reference:** `text` mode puts the rationale in the decoder prompt (plain `LatentModel`).
- **Evaluation:** standard constrained 10-beam ranking on the full test set
  (`src/latent/rationale_eval.py`). The latent teacher is also evaluated with **shuffled**
  (another row's) and **empty** rationales on the first 2,000 rows.

| outcome | reading |
|---|---|
| text teacher > R3 v1 (0.1226) | the rationale carries useful, complementary information |
| latent teacher ≈ text teacher > R3, and shuffle ≪ true | `r` carries it and the decoder reads it: a "ground-truth latent" exists. Go to stage 2 |
| text teacher > R3, latent teacher ≈ R3 | the single-vector channel loses the rationale's information; the latent needs more capacity (several slots / steps) |
| text teacher ≈ R3 | a 7B model's history-only rationale adds nothing a fine-tuned 1.5B model doesn't infer; try a stronger writer, or self-generated reasoning with RL (R²ec-style) |

### Stage 2: student (after stage 1, not implemented yet)
The student is R3 on the plain prompt only, initialised from the teacher, so its decoder
already reads `r`. Its loss:

    L = CE(y) + α · KL(p_teacher(y | x, rationale) ‖ p_student(y | x, r_s)) + β · align(r_s, r_teacher)

- `align` is cosine or MSE on the normalised slot vector.
- The output KL alone could be absorbed by the decoder weights. The latent alignment keeps
  the knowledge in `r`.
- **Success criteria:**
  - HR@10 > 0.1226;
  - a swap/drop-last ratio far above v1's 0.006, measured with the diagnostic in
    decoder_insensitivity_to_latent.md.

## Runs

| job | what | status |
|---|---|---|
| 319656 | smoke: rationales for the first 512 rows of each split | done: 512 rationales in 32 s on one GPU; 6% hit the 192-token cap |
| 319657 / 319658 | smoke: latent / text teacher on those rows | done: train → save → constrained generation → metrics run end to end; predictions are valid catalog titles |
| 319662 | full rationales, Qwen2.5-7B-Instruct, train/valid/test (~83k rows) | done in 1 h 42 min; 6.3–7.1% hit the 192-token cap |
| 319663 | full latent teacher | done (results below) |
| 319664 | full text teacher | done: HR@10 0.1243, NDCG@10 0.0947 (vs v1: 129 hits gained / 115 lost) |
| 319665 | sanity: `rationale_eval.py --mode none` on R3 v1, first 1,000 test rows, vs v1's standard eval | done: HR@10 0.158 vs 0.158, NDCG@10 0.1257 vs 0.1248, identical top-1 on 93.7% of rows (batch padding / bf16). The new eval is comparable to the standard numbers |

## Results: stage 1

### Validation loss per epoch (target-title NLL, 8,312 rows)

| model | epoch 1 | epoch 2 (best) | epoch 3 |
|---|---:|---:|---:|
| R3 v1 (no rationale) | 0.4632 | 0.3982 | 0.4212 |
| latent teacher (rationale → `r` only) | 0.4615 | 0.3980 | 0.4194 |
| text teacher (rationale in the decoder prompt) | 0.4650 | 0.3984 | 0.4217 |

The rationale does not make the target more likely, **even when it is written into the prompt**.

### Ranking (constrained 10-beam generation, test set)

| model | rows | HR@10 | NDCG@10 | R3 v1 on the same rows | vs R3 v1: gained / lost hits |
|---|---:|---:|---:|---:|---|
| R3 v1 | 8,313 | 0.1226 | 0.0950 | — | — |
| latent teacher, true rationale | 8,313 | 0.1240 | 0.0949 | 0.1226 | 126 / 114 |
| text teacher (rationale in the decoder prompt) | 8,313 | 0.1243 | 0.0947 | 0.1226 | 129 / 115 |
| latent teacher, **shuffled** rationale (another row's) | 2,000 | 0.1560 | 0.1246 | 0.1575 | 29 / 32 |
| latent teacher, **empty** rationale | 2,000 | 0.1540 | 0.1239 | 0.1575 | 25 / 32 |

On the first 2,000 rows, the true rationale gives HR@10 0.1555. Shuffling it gives 0.1560 (identical top-1 for 89.8% of rows); emptying it gives 0.1540 (87.2%).

### Reading

- **No gain.**
  - HR@10 is 0.1240 vs 0.1226: +0.0014, with gained and lost hits nearly balanced (126 / 114), so the difference is noise.
  - The validation NLL is the same as R3 v1 to the third decimal.
- **No content dependence.** Swapping in another user's rationale costs nothing (0.1555 → 0.1560). The teacher's `r` is as ignorable as plain R3's.
- **The text teacher's validation loss also matches R3** (0.3984 vs 0.3982), so the information is missing at the source, not lost in the single-vector channel.
  - This is row 4 of the expectation table: a zero-shot 7B model's history-only rationale adds nothing a 1.5B model fine-tuned on 66k interactions does not already infer.
  - The rationales read as generic category-level guesses ("they will need strings; D'Addario or Elixir"). The fine-tuned recommender already captures that level of regularity from co-purchase data.
- **Implication for distillation.** There is no teacher signal to distil from these rationales. A useful reasoning teacher must know something the fine-tuned model does not. Options:
  - reasoning *optimised for the ranking* (R²ec: RL-trained, self-generated);
  - reasoning grounded in information beyond the titles (item descriptions, reviews, other users' behaviour).

  A larger zero-shot writer is not enough on its own.

## Why R3's RL stage cannot help (2026-10-11)

R3's second stage (`src/grpo_attention_tuning`) works as follows:
- it samples 8 noisy copies of `r` per row (r + N(0, 1.5²I), the first copy noise-free);
- the reward is the teacher-forced likelihood of the target title, exp(−mean token NLL);
- the advantage is measured against the noise-free copy;
- **only the pooler is trained**: backbone and LM head are frozen.

**Our runs never actually applied it.**
- The last Musical Instruments RL launch crashed at start-up: global batch 2×2 is not divisible by 8 generations.
- The merge step then picked up the SFT shard files left in the same directory, so `nomul_attn_lr_5e-5/final_result.json` is byte-identical to the SFT result.
- **0.1226 is the SFT number.**
- An earlier partial run (checkpoint-1800 of 2078 steps) logged KL = 0.0000 and clip ratio = 0.0000 on every step: the policy never moved.

**Signal probe** (`src/latent_distill/rl_signal_probe.py`, job 319847; R3 v1 SFT, 512 training rows, 8 samples per row):

| perturbation of `r` | within-group reward std / mean | groups with any \|advantage\| > 0.01 |
|---|---:|---:|
| RL noise, ‖noise‖ ≈ 0.90 ‖r‖ | 0.18% | 0.4% |
| replace `r` by other rows' `r` | 0.22% | 3.9% |
| (reference) reward std between rows | 0.14 on a mean of 0.75 | — |

Even replacing `r` outright moves the reward by 0.2%: the decoder ignores `r`, so every
sample in a group gets the same reward. The advantages are noise, and with the decoder
frozen the policy gradient on the pooler is ~0. RL on R3's latent cannot work while the
decoder does not read `r`. **RL needs samples whose outcomes differ.** Two ways to get them:
- sampled *recommendations* (outcome-level GRPO, the whole model trainable);
- sampled *text reasoning* (R²ec-style).

## LaRec-style latent pre-training (2026-10-11)

LaRec (Xia et al., arXiv 2607.24617) reports latent pre-training on the same Instruments
split (66,500 / 8,312 / 8,313 rows): HR@10 0.1238, NDCG@10 0.0964, against R3 v1's
0.1226 / 0.0950 here. It differs from the distillation above in three ways:

1. **The teacher CoT sees the target.** It is written to derive the purchase from the
   history without naming it, so it carries information the history lacks. Our
   history-only rationales carried none.
2. **K = 6 sequential latent steps** instead of one pooled vector.
3. **Supervision for each latent step.**
   - Step-level InfoNCE to the teacher's hidden state after each CoT step.
   - A direction loss pushing successive latent differences toward the target's embedding.

### Implementation (`src/latent/larec_latent.py`, `src/latent_distill/{generate_cot,teacher_states}.py`)

| component | LaRec | here | why |
|---|---|---|---|
| teacher CoT | Qwen-72B, history + target, K steps | Qwen2.5-7B-Instruct (largest available), same prompt idea, exactly 6 "Step k:" lines | |
| target leaks in CoT | "must not contain the target" (no procedure given) | **redaction**: the full title or any ≥4-word title run → "the product"; model numbers and brands absent from the history removed | the 7B named the target in 54% of rows, often verbatim in step 6. Dropping those rows would lose most of the data; after redaction 0/512 smoke rows still leak |
| teacher states h_k | final layer of the pre-training LLM after steps 1..k | frozen Qwen2.5-1.5B-Instruct, final normed state at the last token of step k | |
| latent steps | z_k from z_<k and history; "reasoning head" into embedding space (unspecified) | K = 6 slots; z_k = softmax(lm_head(s + MLP(s))) @ E, where s is the state before slot k (soft token, Soft-Thinking style) | the raw state rescaled to token norm gave initial target CE 15.3; the soft token gives 5.7 (R3: 4.6) |
| L_SA | InfoNCE, in-batch negatives, τ unreported | InfoNCE on the student state after reading z_k vs h_k, cosine, τ = 0.1, 255 negatives = same-step teacher states of random training rows | R3's micro-batch of 2 would give one negative |
| L_PA | 1 − cos(z_k − z_{k−1}, mean title-token embedding) | same | |
| weights | α = γ = 1 | α = γ = 1; auxiliary terms in training only, weighted by micro-batch token share | |
| recipe | lr ∈ {3e-4, 5e-5, 1e-5}, early stopping | R3 v1's: Qwen2.5-1.5B-Instruct, lr 3e-4, batch 128, 3 epochs, early stopping, cutoff 512 | fixed hyperparameters |

**Evaluation.**
- Standard constrained 10-beam ranking on the full test set.
- The first decoding step is keyed on the standard "### Response:" + thought prefix.
- **Swap control** on the first 2,000 rows: each user receives the next user's whole latent chain. It tests whether predictions depend on the latent content.

| job | what | status |
|---|---|---|
| 319865 / 319866 | smoke (first head): CoT → states → train → generate → swap | pipeline ran; eval loss 10.8 exposed the head problem |
| 319868 / 319869 | initial target CE: R3 4.59, LaRec raw head 15.33, soft head 5.70 | done |
| 319878 | smoke, soft head | done: eval loss 5.34 after 4 steps; step InfoNCE 3.81 vs chance 5.55 |
| 319876 | full: 6-step CoT (redacted) for 66,500 training rows + teacher states | running |
| 319877 | full: LaRec pre-training + test eval + swap control | queued (after 319876) |
