# Behaviour-grounded two-step latent reasoning (Attempt 12)

## Hypothesis

Other samples' reasoning can ground the current reasoning **when it comes with the behaviour it led to**. A donor's latent alone is another summary of a history (prototype grounding E4 and Attempts 10/11 showed it is redundant). The pair *(donor reasoning, donor's observed next item)* answers a different question: "what did users who reasoned like this actually do next?" That information is not in the query's own history.

## Why the previous outcome-memory pilot is not a test of this

`musical_margin2_memory2` used the right donor content but failed mechanically:

- its gate learned to close (0.44 → 0.003; delivered correction 0.005% of ‖r‖), so outcome = shuffled outcome = random = off;
- it was trained on top of a model that had already fit the training targets, so donor evidence could not lower the loss;
- evidence entered as a bounded (≤10%) additive correction to a large anchor vector.

## Model (`src/latent/grounded_reasoning.py`)

- **Step 1, own intuition.** `r0` = ordinary R3 pooling over the user's history.
- **Step 2, grounding (idea 5).** A learned retriever (`query(r0_i) · key(r0_j)`) selects the top 32 eligible donors. Each donor's value is `V([normalize(r0_j); e(y_j)])`, where `e(y_j)` is the Qwen3 title embedding of the donor's observed next item. Then
  `r1 = r0 + g · ‖r0‖ · normalize(Σ_j a_j V_j)`,
  with one learnable global scale `g` (initialised to 0.5) and no per-sample gate. `r1` goes into the thought slot.
- **Step-2 supervision.** `r1` must score the true next item over the full catalogue: `CE(normalize(P(r1)) · Eᵀ / 0.07, y)`, weight 0.2.
- **Behaviour-supervised retrieval (idea 4).** For each candidate donor, usefulness is how much grounding on that donor alone raises the log-probability of the query's *observed* next item: `u_j = log p(y_i | r1^(j)) − log p(y_i | r0)`. The retriever's distribution is distilled toward `softmax(u / 0.5)` with KL, weight 0.1. Eight random eligible donors are added per anchor during training for exploration.

## Leakage boundaries

- Donors are other users only.
- A donor's outcome must be strictly earlier than the query's latest history event (`history_cutoff`). Test rows are later than all train rows, so training mirrors test.
- Retrieval and the value path never see the query's target. The target is used only in the training losses.

## Training and inference

- Trained jointly from Qwen2.5-1.5B-Instruct, with grounding active from step 0, so donors matter before the decoder memorises the training targets.
- Training donors: a run-local FIFO of 20,000 recent training rows. Every row is accepted: observed outcomes need no correctness gate. Turnover is about 160 optimizer steps, versus about 780 for Attempt 10's gated queue.
- Inference donors: after training, the final model's `r0` for every train row is written to a persistent bank saved with the checkpoint.

## Controls (same checkpoint, `GROUNDING_MODE`)

| mode | meaning | what it isolates |
|---|---|---|
| `knn` | the method | — |
| `off` | `r1 = r0` | any effect of grounding at all |
| `shuffle_outcome` | same donors, outcomes permuted among them | behaviour (the thesis claim) |
| `latent_only` | same donors, outcome embeddings zeroed | reasoning alone vs reasoning + behaviour |
| `random` | random eligible donors | the learned retriever |

## Success criteria

1. Ranking (HR/NDCG@10) of `knn` beats plain R3 and the `off` control, with paired bootstrap CIs.
2. `shuffle_outcome` and `latent_only` both hurt relative to `knn`: the gain comes from donor behaviour.
3. `random` hurts relative to `knn`: the behaviour-supervised retriever matters.
4. Diagnostics: `evidence_scale` stays away from 0; `item_hit10_r1 > item_hit10_r0` on training batches; `utility_top_gain > 0`.

## Runs

- Smoke: job 323030 (512 train rows, 1 epoch, 16 test rows, controls `knn`/`off`).
- Launcher: `scripts/latent_grounded_reasoning_full.sbatch` (`CATEGORY`, `CODE_ROOT`, `OUTPUT_SUFFIX`, `CONTROLS`, sample overrides).

## Known risks

- Later epochs can still memorise the training targets. If `evidence_scale` or the knn–off gap shrinks over epochs, cross-fitted training is the next step.
- The usefulness signal is measured by the step-2 item scorer, not by a full decoder pass. That is cheap, but it is a proxy for decoder usefulness.
- The train-time FIFO donors and the test-time full bank come from slightly different encoders.
