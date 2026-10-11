# General latent-reasoning techniques: information injection and bypass (2024–2026)

Saved by the coordinator from the researcher's hand-back. The researcher's own write to the shared checkout was blocked.

Verification:
- **FULL** (checked against arXiv full text): Coconut, CODI, Cui et al., "Do latent tokens think?", Viveiros et al., CVRR.
- **ABS:** abstract only.
- **SNIP:** search snippet only.

## Core finding
A latent being bypassed is the default outcome. A model relies on a latent only when:
1. the latent carries information that cannot be obtained another way; and
2. the alternative routes to that information are blocked.

Extra computation alone does not create reliance: pause tokens perform about the same as no chain-of-thought.

## Evidence of bypass (FULL)
- **Cui et al. (2602.22441).** Gaussian noise (σ=100) added to the latents:
  - ProsQA accuracy barely moves:
    - Coconut 99.4→99.4;
    - CODI 100→99.6;
    - SIM-CoT 100→97.6 (Llama).
  - CoLaR is the most latent-dependent: ProsQA 98.2→60.9, GSM8K 25.2→3.3.
  - Explicit chain-of-thought under the same noise collapses: 61.7→0.03 on GSM8K.
  - CODI with zero latents still gets about 30% on GSM8K.
  - Trade-off: stronger supervision gives fewer shortcuts but less diversity in the latents.
  - Their Coconut fix (cross-stage data reuse plus next-step-only supervision) brings ProsQA under noise down to 4.4%, i.e. the model now depends on its latents.
- **"Do latent tokens think?" (2512.21711).**
  - Swapping Coconut tokens across MMLU examples: 60.9→61.0, inconsistency 17.9%.
  - Swapping chain-of-thought tokens: 62.8→43.4, inconsistency 52.8%.
  - Coconut exploits planted shortcuts in MMLU and HotpotQA.
  - The paper's text contradicts its table on the LLaMA steering numbers; the table was used.
- **Coconut (2412.06769) ablations** (GSM8K / ProntoQA / ProsQA):

  | Setting | GSM8K | ProntoQA | ProsQA |
  |---|---|---|---|
  | Coconut | 34.1 | 99.8 | 97.0 |
  | Pause as thought | 24.1 | 100.0 | 96.6 |
  | Without the curriculum | 14.4 | 52.4 | 76.1 |

- **Viveiros et al. (2605.18445)**, on the vision-language models LVR, LanteRn, ILVR and Monet:
  - On every model and dataset, at least one replacement (noise, zero, random region, skipping the latents) stays within 2 points of the true latents.
  - LanteRn scores 79% with true latents and 79% with pause tokens.
  - Masking the relevant image region (the latent becomes the only path) opens a gap: 75% true vs 59% noise.
  - Synthetic Tetris task where latents are truly informative: 0.86 true vs 0.25 zeros (chance 0.25).
  - Collapse:
    - generated latents retrieve their own target latent in the top 1 only 0.3–3.3% of the time;
    - they are closer to other examples' predictions than to their own target 99–100% of the time;
    - cosine similarity among predictions is 0.82–0.98.
- **CVRR (2609.06746).**
  - Image states and the multimodal KV cache are deleted before the answer, so the recurrent state is the only path for image information.
  - Training is answer cross-entropy only: no latent target, no teacher. LoRA of 2.88M parameters (0.03% of the model), 4 recurrent steps.
  - On V\*, swapping the state with another example's drops accuracy 81.2→14.0; noise drops it to 35.6.
  - Other latent methods retrained under the same constraint collapse: LVR scores 35.6 on V\* and 0.7 on MMVP pairs.

## Where the latent's information comes from
- **Self-generated, curriculum from chain-of-thought text:** Coconut feeds its own hidden state back.
- **Teacher hidden states:**
  - CODI aligns one anchor token's hidden state, across all layers, with an L1 loss to the same model reading the chain of thought. Removing the loss drops GSM8K 43.7→24.5. The paper contradicts itself on the loss weight (γ=20 in one appendix, β=20 in another).
  - CoLaR predicts compressed embeddings of the chain-of-thought tokens.
- **Separate assistant model:** SoftCoT and SoftCoT++ use a frozen small model plus a trained projection (ABS/SNIP).
- **Extra computation:**
  - recurrent depth: Huginn, 3.5B parameters, 800B tokens (ABS);
  - CVRR;
  - Soft Thinking: training-free mixtures of token embeddings, up to +2.48 points, with a risk of collapse (SNIP).
- **Reward:**
  - RL: HRPO (a gated mix of hidden state and token embedding, ABS) and CoLaR's RL stage;
  - test-time optimisation: LatentSeek (ABS) and the "Silenced Visual Latents" fix (2605.02735, ABS, no numbers).
- **Retrieval:** LatentRAG (2605.06285, ABS) aligns latent sub-queries with a dense retriever and reports about 90% lower latency.
- **Formal analysis (2509.25239, ICML 2026, ABS):** latent thought suits parallel computation; chain-of-thought suits approximate counting and sampling.

## Methods without a teacher
- CVRR, HRPO, LatentSeek, the Silenced-latents fix, Huginn, Soft Thinking, pause tokens.
- Only CVRR has strong causal evidence, and it gets it by blocking the bypass rather than by adding information.

## Implications for R3 (the researcher's inference, untested in any source)
- **Add information the prompt lacks**, via one of:
  - CODI-style alignment to a teacher that sees privileged data (the target item or a rationale);
  - a CF/SASRec assistant, SoftCoT-style;
  - retrieval;
  - ranking-reward RL.
- **Make the latent necessary:** hide the history from the answer decoder, as CVRR does.
- **Test reliance with swaps, not probes:** swap the latent between users, add noise or zero it, and track the retrieval and collapse metrics of Viveiros et al.

## Gaps
- SoftCoT, SoftCoT++, HRPO, LatentSeek, Huginn and LatentRAG were checked at abstract or snippet level only.
- RING (2608.01630) and memory-token latent papers were not verified.
- The pause-token paper (2310.02226) was not fetched.
- No source applies swap tests in recommendation.

## Sources
- https://arxiv.org/html/2412.06769
- https://arxiv.org/html/2502.21074
- https://arxiv.org/html/2602.22441
- https://arxiv.org/html/2512.21711
- https://arxiv.org/html/2605.18445
- https://arxiv.org/html/2609.06746
- https://arxiv.org/abs/2605.02735
- https://arxiv.org/abs/2509.25239
- https://arxiv.org/abs/2505.16552
- https://arxiv.org/abs/2502.12134
- https://arxiv.org/html/2505.11484v2
- https://arxiv.org/html/2505.15778v1
- https://arxiv.org/abs/2502.05171
- https://arxiv.org/html/2505.18454v2
- https://arxiv.org/html/2505.13308v2
- https://arxiv.org/abs/2605.06285
