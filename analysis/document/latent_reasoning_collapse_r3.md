# Latent Reasoning Collapse in R3 Recommendation Models

## Executive summary

The current R3 latent-reasoning recommender does not learn a rich, sample-specific reasoning space. Instead, it maps most interactions to a small collection of nearly collinear latent templates. The recommendation decoder strongly depends on receiving a normal-sized vector in the thought position, but depends only weakly on which in-distribution thought vector it receives.

We call this failure mode **latent reasoning collapse**:

> A model has latent reasoning collapse when its latent thought is required as an activation, but carries too little diverse, sample-specific information for replacing it with another valid thought to materially change the prediction.

The evidence indicates that collapse is already present after supervised fine-tuning (SFT) and becomes substantially stronger after verifier-free RL. This explains why zeroing the thought damages recommendation, while replacing it with the global mean or a thought from another user/target changes performance very little. It also explains why clustering produces redundant prototypes and why similarity-based prototype grounding does not improve recommendation.

The result does not show that latent reasoning is impossible. It shows that the present R3 architecture and objectives permit a shortcut in which the thought acts as a shared activation carrier while the decoder continues to obtain most sample-specific information directly from the visible history.

## 1. Models and data

Both checkpoints use `LatentModel` with a 1,536-dimensional, single-thought representation on `CDs_and_Vinyl`.

| Model | Checkpoint | Training stage | Checkpoint SHA-256 |
|---|---|---|---|
| SFT | `output_dir/CDs_and_Vinyl/latent` | Supervised fine-tuning only | `cd5501c9865046ca544795367445317a44085114fd927a4b8e01138096875799` |
| RL | `output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5` | SFT followed by verifier-free RL | `b03801e85325a091305fd9079e54905d25dd9c440fed24711d5e4936447dc5a5` |

The distribution analysis uses all 49,251 chronological training examples. PCA is centered and fitted on the complete latent bank, not a visualization subsample. Pairwise cosine statistics use 100,000 reproducibly sampled pairs with seed 42.

## 2. Causal evidence: activation dependence without content dependence

For the frozen RL checkpoint, the raw validation target NLL is 1.169466. The intervention results are:

| Thought intervention | Target NLL | Change versus raw | Interpretation |
|---|---:|---:|---|
| Raw thought | 1.169466 | 0 | Baseline |
| Zero vector | 2.170643 | +1.001177 | Removing the learned-manifold activation is destructive |
| Global train mean | 1.169722 | +0.000256 | No reliable content-specific effect |
| Shuffled user | 1.169743 | +0.000277 | No reliable user-specific effect |
| Shuffled target | 1.169566 | +0.000099 | No reliable target-specific effect |
| Gaussian noise 0.25× | 1.168888 | -0.000579 | Small noise does not reveal fragile reasoning content |
| Gaussian noise 0.50× | 1.166983 | -0.002483 | Likelihood slightly improves |
| Gaussian noise 1.00× | 1.163388 | -0.006079 | Even large noise is not strongly destructive |

Zero replacement is an out-of-distribution intervention. Its large effect establishes that a normal latent activation is needed, but it does not establish that the thought contains sample-specific reasoning. The in-distribution mean and shuffle interventions are the more informative tests: all preserve likelihood and ranking within very small margins.

The ranking results tell the same story. Zero replacement reduces HR@10 from 0.110606 to 0.072113 and NDCG@10 from 0.078384 to 0.041077. In contrast, mean, shuffled-user, and shuffled-target HR/NDCG remain close to the raw checkpoint. The complete intervention table is reported in [the experiment plan](prototype_grounding_experiment_plan.md).

### Interpretation

The model depends on the **presence and scale** of the thought slot more than on its **sample identity**. This is activation dependence, not yet reasoning dependence.

## 3. Geometric evidence: the RL thought space is collapsed

![RL latent-distribution overview](../figures/latent_distribution_overview.png)

The RL train bank has the following geometry:

| Diagnostic | RL result |
|---|---:|
| Latent dimension | 1,536 |
| Mean vector norm | 95.34 |
| Global-mean norm / mean vector norm | 0.9952 |
| Mean cosine to the global mean | 0.9952 |
| Mean random-pair cosine | 0.9905 |
| Mean same-user cosine | 0.9914 |
| Mean same-target cosine | 0.9906 |
| Exact FP16-distinct vectors | 348 of 49,251 |
| Samples represented by the top 10 templates | 82.12% |
| Samples represented by the top 20 templates | 91.57% |
| Largest-template coverage | 15.61% |
| PC1 share of centered variance | 77.05% |
| Top-two-PC share | 89.85% |
| Entropy effective rank | 2.56 |
| Participation ratio | 1.63 |

Although each vector has 1,536 coordinates, the centered bank behaves like a space with only a few effective dimensions. The global mean accounts for nearly all vector magnitude, and unrelated thoughts point in almost the same direction.

### All-vector RL PCA visualization

![All RL train vectors in centered PCA space](../figures/latent_pca_all_scatter.png)

The left panel uses exact PCA coordinates with logarithmic density, so repeated points remain visible. The right panel adds small display-only jitter to all 49,251 rows. Jitter is not used in PCA or any statistic.

The apparently small number of locations is a result, not subsampling: at stored FP16 precision, only 348 distinct vectors occur. Thousands of examples therefore occupy exactly the same plotted coordinate.

### Dominant RL templates

![RL exact-template frequencies](../figures/latent_template_frequency.png)

The first ten exact templates represent more than four-fifths of the training set. This concentration explains the severe effective-prototype collapse observed when spherical k-means was asked to produce 32–256 prototypes.

## 4. Collapse begins during SFT and becomes stronger after RL

The SFT-only checkpoint is also collapsed, but it retains considerably more variation than the RL checkpoint.

![All SFT train vectors in centered PCA space](../figures/sft_only_latent_pca_all_scatter.png)

| Diagnostic | SFT only | RL | Change after RL |
|---|---:|---:|---|
| Raw train target NLL ↓ | 0.714917 | 0.687539 | Improved by 0.027378 |
| Zero-mask train NLL | 1.795241 | 1.795336 | Essentially unchanged |
| Mean raw-over-zero NLL gain | 1.080324 | 1.107797 | Increased by 0.027473 |
| Mean vector norm | 72.59 | 95.34 | Increased by 31.3% |
| Global-mean norm / mean vector norm | 0.9926 | 0.9952 | More mean-dominated |
| Mean random-pair cosine | 0.9855 | 0.9905 | More collinear |
| Exact FP16-distinct vectors | 2,337 | 348 | Reduced by 85.1% |
| Top-10-template coverage | 32.21% | 82.12% | Increased by 49.91 points |
| PC1 centered variance | 48.27% | 77.05% | Increased by 28.78 points |
| Top-two centered variance | 67.76% | 89.85% | Increased by 22.09 points |
| Entropy effective rank | 4.82 | 2.56 | Reduced by 46.9% |
| Participation ratio | 3.35 | 1.63 | Reduced by 51.3% |

![SFT latent-distribution overview](../figures/sft_only_latent_distribution_overview.png)

SFT therefore establishes a low-diversity latent channel, and RL further concentrates that channel. RL lowers raw train NLL and increases the vector norm, while the zero-mask NLL remains almost identical. This is consistent with optimization strengthening a generally useful activation rather than creating diverse reasoning states.

This comparison should not be interpreted as proof that every RL method causes collapse. It is evidence about this R3 checkpoint and its verifier-free RL procedure.

## 5. Weak association with preference and utility

After subtracting a same-group-size permutation baseline, the fraction of centered latent variance associated with metadata is:

| Grouping variable | SFT only | RL |
|---|---:|---:|
| User identity | 5.15% | 10.19% |
| Target identity | 2.08% | 2.56% |
| Train-only CF item cluster | 0.61% | 0.79% |

There is some user-linked residual structure, and RL increases it. However, target and conventional collaborative-preference structure remain weak, while the overall bank becomes much more globally collapsed.

The evidence-utility score is

\[
s(r)=\operatorname{NLL}(r=0)-\operatorname{NLL}(r)
=\log P(y\mid X,r)-\log P(y\mid X,r=0).
\]

For the RL bank, utility has weak Pearson association with vector norm (-0.028), centered distance (-0.074), and cosine to the global mean (+0.033). Exact-template identity explains only 3.37% of utility variance. Consequently, selecting the top utility quartile does not isolate a clearly distinct reasoning geometry.

This utility is specifically an advantage over the zero-vector intervention. Because zero is far outside the learned manifold, it should not be treated as a direct measurement of semantic reasoning quality.

## 6. Consequence for prototype grounding

The collapse results predict the downstream prototype findings:

1. Clustering yields a few highly utilized directions and many duplicated or unsupported prototypes.
2. The E-top bank is not substantially more meaningful than E-all because zero-mask utility is dominated by activation presence.
3. Nearest-prototype replacement changes little because most valid thoughts and centroids lie in the same narrow cone.
4. Frozen residual grounding produces no consistent NLL or ranking gain.
5. A target-aware oracle can find mildly helpful directions, but cosine similarity cannot select them without using the answer.

Thus, prototype grounding is currently operating on collapsed activations rather than reusable collaborative reasoning units. Prototype mining should be retried only after the base model passes a stronger sample-specific dependence test.

## 7. Likely R3 shortcut mechanism

The R3 model computes a thought vector from the history and inserts it at the special thought-token position. During recommendation, however, the language-model decoder still receives the full history tokens as well as the thought token. It can therefore predict the next item through a direct path:

\[
X \longrightarrow \text{decoder} \longrightarrow y,
\]

while using the latent path primarily as a shared activation:

\[
X \longrightarrow r \approx \mu \longrightarrow \text{decoder}.
\]

Nothing in ordinary next-item likelihood forces different histories to produce meaningfully different thoughts when the decoder can already recover sample information from `X`. A constant or low-rank thought can become useful as a learned soft prompt, bias, or mode indicator. Verifier-free RL can then improve reward by amplifying this shared carrier instead of learning a semantically structured reasoning space.

The intervention evidence supports this explanation:

- zeroing removes the carrier and severely damages prediction;
- mean or shuffled valid vectors preserve the carrier and performance;
- even noise larger than ordinary within-bank variation has little ranking effect;
- RL increases norm and collapse while improving train NLL.

## 8. Recommended model revision

### 8.1 Remove or constrain the bypass

The strongest intervention is architectural. Make the prediction stage depend on the latent bottleneck rather than allowing unrestricted access to the same history used to create it. Candidate designs include:

- a reasoning encoder that sees the history and produces `r`, followed by a decoder that sees `r` but not the original history tokens;
- an attention mask in which the recommendation tokens attend to the thought token while the thought token summarizes the history;
- a limited residual history path whose capacity is explicitly controlled and ablated.

The first design provides the clearest test of whether `r` is sufficient. A softer history path may preserve accuracy but makes shortcut diagnosis harder.

### 8.2 Train sample-specific dependence

Use an in-distribution counterfactual objective, not zero masking alone. For a correct thought `r_i` and a shuffled thought `r_j`, require

\[
\log P(y_i\mid X_i,r_i)
-\log P(y_i\mid X_i,r_j) \ge m.
\]

A margin or pairwise-ranking loss can be added to the recommendation loss. Negatives should include other users, unrelated targets, hard nearby thoughts, and the global mean. The negative branch should use stop-gradient where necessary so the two branches cannot satisfy the objective by moving together.

### 8.3 Prevent dimensional and variance collapse

Apply the diversity constraint to centered thoughts, since the shared mean dominates raw geometry. Possible components are:

- per-dimension variance floors, as in VICReg-style regularization;
- covariance off-diagonal penalties or whitening;
- uniformity/contrastive losses on normalized centered thoughts;
- balanced discrete-code utilization if a codebook is trained;
- explicit penalties for dominant-template frequency and low effective rank.

Diversity by itself is insufficient: the decoder may still ignore diverse but irrelevant directions. It must be paired with the causal-dependence objective or bottleneck.

### 8.4 Align diversity with collaborative preference

Reasoning should not be different merely for the sake of being different. Positive pairs can be defined using leakage-safe shared preference structure—such as similar histories or train-only CF neighborhoods—while negatives should represent distinct preferences. This can make the latent geometry useful for later prototype retrieval.

## 9. Required acceptance tests before prototype mining

A revised checkpoint should satisfy all of the following on held-out validation data:

1. **In-distribution causal dependence:** shuffled-user, shuffled-target, and global-mean thoughts significantly worsen target NLL and ranking.
2. **Diversity:** random-pair cosine is materially below the current 0.985–0.990 range, exact-template concentration falls, and effective rank increases.
3. **Information alignment:** same-user/preference-neighbor similarity exceeds random similarity by a meaningful margin without using the validation/test target.
4. **Utility validity:** evidence utility remains predictive when the counterfactual is an in-distribution wrong thought, not only zero.
5. **No accuracy-only shortcut:** gains remain when the direct history-to-decoder path is removed or tightly controlled.
6. **Frozen prototype transfer:** useful-evidence prototypes beat raw thoughts, global mean, shuffled prototypes, and random controls before end-to-end Stage-2 training.

Only after these gates pass should collaborative prototype grounding be interpreted as reusable latent reasoning.

## 10. Interactive raw-vector t-SNE

A separate Plotly visualizer fits Barnes-Hut t-SNE directly to every raw 1,536-dimensional vector. It performs no PCA pre-reduction, centering, normalization, subsampling, clustering, or template mapping. Initialization is random; perplexity is 30; early exaggeration is 12; the run uses 1,000 iterations and seed 42.

- [Open the interactive RL raw-vector t-SNE](../figures/rl_raw_latent_tsne_plotly.html)
- [Open the interactive SFT raw-vector t-SNE](../figures/sft_only_raw_latent_tsne_plotly.html)

The displayed points are colored by history length. Hovering shows sample ID, user ID, target item, history length, target-NLL gain versus zero, raw target NLL, and the two t-SNE coordinates. Both HTML files include Plotly JavaScript and can be opened without an internet connection.

Commands:

```bash
python src/prototype_grounding/visualize_tsne_plotly.py \
  --evidence-dir analysis/evidence/CDs_and_Vinyl/train \
  --tag rl --title "R3 RL raw latent reasoning t-SNE" \
  --perplexity 30 --max-iter 1000 --seed 42 --color history_len

python src/prototype_grounding/visualize_tsne_plotly.py \
  --evidence-dir analysis/evidence/CDs_and_Vinyl/sft_only/train \
  --tag sft_only --title "R3 SFT-only raw latent reasoning t-SNE" \
  --perplexity 30 --max-iter 1000 --seed 42 --color history_len
```

The final KL divergences are 2.5765 for RL and 0.3469 for SFT. They describe each t-SNE optimization and should not be treated as a model-quality score.

Important caveat: the input contains many exactly duplicated stored vectors. Direct row-level t-SNE includes each duplicate as requested, but its repulsive objective can visually spread identical inputs. Apparent separation inside such clouds is therefore a property of t-SNE optimization, not evidence that the original latent vectors differ. PCA, exact-template counts, cosine similarity, and effective-rank statistics remain the appropriate evidence for collapse.

## 11. Reproducibility artifacts

- RL evidence: `analysis/evidence/CDs_and_Vinyl/train/`
- SFT evidence: `analysis/evidence/CDs_and_Vinyl/sft_only/train/`
- RL statistics: `analysis/results/latent_distribution/train_latent_distribution.json`
- SFT statistics: `analysis/results/latent_distribution/sft_only/train_latent_distribution.json`
- Numerical PCA/pair arrays: corresponding `train_latent_distribution_arrays.npz` files
- Analysis implementation: `src/prototype_grounding/experiment.py`, command `latent-distribution`
- Interactive raw-vector t-SNE implementation: `src/prototype_grounding/visualize_tsne_plotly.py`
- Raw t-SNE coordinates and run summaries: `analysis/results/latent_distribution/{rl,sft_only}/train_raw_latent_tsne_*`
- Full experiment record: [Prototype Grounding for Latent Recommendation](prototype_grounding_experiment_plan.md)

## Conclusion

The current R3 model exhibits a clear latent reasoning collapse failure mode. SFT produces an already narrow and repetitive latent space; verifier-free RL improves train likelihood while making that space substantially more concentrated. The decoder requires a valid latent activation but shows little dependence on its sample-specific content.

For this reason, the next research priority should not be more elaborate clustering of the existing thoughts. It should be training a causally necessary, diverse, preference-aligned latent bottleneck and verifying that in-distribution thought replacement changes recommendation. Prototype grounding becomes scientifically meaningful only after that base requirement is met.
