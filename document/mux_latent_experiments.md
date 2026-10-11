# MUX-style category supervision of the R3 thought vector

**Question.** Does supervising R3's latent directly — MUX-style (arXiv 2607.18264), decoding it with the model's own tied unembedding and pulling it toward a multiplexed target — make the decoder depend on it? First aspect: the target item's category.

## Design

- `src/latent/mux_latent.py`. Two ways to build the thought vector r:
  - `pool`: R3's attention pooling over the prompt's final hidden states (unchanged);
  - `codi`: CODI-style, the final hidden state at the position before `<|Thought|>` through MLP + LayerNorm.
- Loss: `L = L_rec + β · KL(mux(category tokens) ‖ softmax(W r / τ))`, W = Qwen2.5's tied unembedding, τ = 1.
  - Target: the target item's category path, root dropped, segments leaf-first; weights ∝ 0.9^(j−1) over its tokens, repeats merged (8.4 tokens per path on average; 4,875 of 5,030 items).
  - Scaled by the micro-batch's share of label tokens under HF gradient accumulation.
- Grid: {pool, codi} × β ∈ {0, 0.1, 0.5}; all from Qwen2.5-1.5B-Instruct with the R3 v1 recipe (3 epochs, batch 128, lr 3e-4, cutoff 768). `pool_none` is R3 retrained with this script.
- Evaluation (`src/prototype_grounding/mux_eval.py`, Musical Instruments test split, 8,313 rows):
  1. category decoded from r: score each of the 412 paths by Σ a_j log softmax(W r)[token_j]; top-1/top-5, on R3 v1 hits/misses; baseline = category of the last history item;
  2. teacher-forced KL on the target title (first 2,000 rows) as a ratio to dropping the last history item: swap r (another user's), zero r, mean r, and replacing r by ‖r‖·normalize(Σ a_j W[token_j]) for the true / a donor's category;
  3. full-test constrained 10-beam ranking vs R3 v1 (gained/lost, exact McNemar).
- Jobs: smoke 319253/319254; full 319255–319260 (`scripts/mux_train.sbatch <variant> [full|smoke]`). Results: `analysis/results/mux_latent/full_<variant>/summary.json`.

## Results

| | pool_none | pool_mux0.1 | pool_mux0.5 | codi_none | codi_mux0.1 | codi_mux0.5 |
|---|---:|---:|---:|---:|---:|---:|
| category from r, top-1 (last-item baseline 11.6%) | 0.2% | 8.2% | 8.1% | 0.07% | 8.0% | 8.3% |
| top-5 | 1.0% | 15.7% | 15.5% | 0.6% | 15.6% | 15.7% |
| top-1 on R3 hits / misses | — | 45.4 / 2.9% | 44.5 / 3.0% | — | 44.3 / 2.9% | 46.2 / 3.0% |
| ‖r‖ | 88.6 | 45.5 | 37.4 | 39.2 | 39.4 | 39.3 |
| swap r (ratio to history drop) | 0.0020 | 0.0011 | 0.0018 | 0.0010 | 0.0013 | 0.0012 |
| inject true / donor category | 3.62 / 3.64 | 0.018 / 0.013 | 0.32 / 0.23 | 7.39 / 7.46 | 0.011 / 0.007 | 0.007 / 0.006 |
| zero r | 1.41 | 0.10 | 14.4 | 0.25 | 0.11 | 0.16 |
| HR@1 | 0.0742 | 0.0737 | 0.0734 | 0.0736 | 0.0731 | 0.0728 |
| HR@10 | 0.1241 | 0.1243 | 0.1213 | 0.1227 | **0.1247** | 0.1199 |
| NDCG@10 | 0.0955 | 0.0955 | 0.0940 | 0.0944 | 0.0956 | 0.0931 |
| gained / lost vs R3 v1 (HR@10 0.1221) | 123 / 106 | 125 / 107 | 125 / 132 | 115 / 110 | 127 / 105 | 112 / 130 |
| McNemar p | 0.29 | 0.26 | 0.71 | 0.79 | 0.17 | 0.27 |

## Reading

- **The supervision reaches r.** With β > 0 the category decodes from r at 8.0–8.3% top-1 (15.5–15.7% top-5) vs 0.1–0.2% without, for both builders and both weights — the same level everywhere, i.e. the ceiling of what the history supports for this predictor. It is still below the trivial last-history-item baseline (11.6%), and it is right almost only where R3 is already right (≈45% on R3 hits, ≈3% on misses).
- **The decoder still ignores r's content.** Swapping in another user's r moves the output 0.001–0.002 of a history-item drop in every run, supervised or not; injecting the *true* category barely differs from a donor's. Category content in r is derived from the history, which the decoder reads directly, so it has no reason to read r (H1 of `decoder_insensitivity_to_latent.md`).
- **Presence vs content.** pool_mux0.5 depends heavily on *something* being in the slot (zero r: 14.4) but not on which user's r it is (swap 0.0018): the anchor pattern of R3 v0. The large injection ratios of the unsupervised models (3.6, 7.4, true ≈ donor) are out-of-distribution disruption, not category reading.
- **Ranking is unchanged.** All six runs are within ±0.25 pt HR@10 of R3 v1 and of each other (no McNemar p < 0.15); the R3 retrain with this script (pool_none) scores 0.1241, so seed-level variation is ≈ 0.2 pt.
## Geometric collapse (job 319695; `src/prototype_grounding/latent_geometry.py`, first 2,000 test rows)

| | R3 v1 | pool_none | pool_mux0.1 | pool_mux0.5 | codi_none | codi_mux0.1 | codi_mux0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| near-duplicate share (NN cosine > 0.999) | 36.2% | 8.6% | 6.3% | 7.2% | **60.6%** | 5.9% | 7.7% |
| sample-specific share ‖r − μ‖/‖r‖ | 0.73 | 0.65 | 0.32 | 0.33 | **0.07** | 0.34 | 0.32 |
| cosine to mean | 0.72 | 0.77 | 0.94 | 0.94 | 0.996 | 0.93 | 0.94 |
| participation ratio | 12.3 | 60.9 | 10.4 | 10.3 | 4.7 | 10.0 | 9.9 |
| top-10 PC variance share | 68% | 30% | 75% | 75% | 73% | 77% | 79% |

- MUX removes the template collapse of the unsupervised CODI latent (61% → 6–8% near-duplicates; sample-specific share 0.07 → 0.33).
- All four MUX runs converge to the same geometry regardless of builder: a large shared component plus ~10 effective directions of sample-specific variation, few templates. The supervision sets the geometry.
- For the pooled latent, MUX trades spread (60 effective directions) for structure (a stronger shared component and ~10 directions).
- Unsupervised R3 geometry varies a lot between runs of the same recipe (v1 36% vs retrain 9% near-duplicates).
- **MUX fixes geometric collapse but not utilization collapse**: these structured latents are still ignored by the decoder (swap ratio 0.001–0.002).

## Conclusion

- **Conclusion.** Self-predicted aspect supervision makes the latent *interpretable* (it carries the category) but not *used*: the information is redundant with the prompt. To create dependence, the latent must carry information the decoder cannot get elsewhere during training — e.g. withholding part of the history from the decoding pass, or distilling information the model does not have (teacher rationales, retrieved collaborative evidence).
