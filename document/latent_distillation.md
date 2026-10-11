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
| 319662 | full rationales, Qwen2.5-7B-Instruct, train/valid/test (~83k rows) | submitted |
| 319663 / 319664 | full latent / text teacher (3 epochs, test eval; latent also shuffle/empty on 2,000 rows) | submitted (after 319662) |
| 319665 | sanity: `rationale_eval.py --mode none` on R3 v1, first 1,000 test rows, vs v1's standard eval | submitted |
