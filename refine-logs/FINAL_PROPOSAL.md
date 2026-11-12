# Final Proposal: Cluster Evidence as a Trained Latent Variable (Posterior-Supervised Selection)

**Date**: 2026-10-28 (revised after user feedback: training-free selection/fusion is trivial, so it is a baseline only) · **Run**: `collab-reasoning-develop-2026-10-28` · **Branch**: `worktree-R3CollaborativeReasoning`

## Problem Anchor (frozen)

The ~8 decoded cluster-conditioned paths per user contain the answer: the best path reaches HR@10 0.2275 vs 0.1450 for the R3 path on the report rows. No selector recovers this headroom:
- decoder likelihood is below chance on R3 misses (0.073 vs 0.129);
- the best predictor reaches 22.7%;
- marginalization returns R3.

Every trained selector so far (the k=150 and k=20 verifiers) was trained on **which cluster contains the target**, and is confident only where R3 is already right. The anchored problem is **what a selector should be trained on**. Evidence delivery into the input is out of scope; it has five measured negatives.

## Method thesis

Treat the CF cluster as a latent variable of the recommender. Train the cluster prior on the **decoder's exact posterior**, i.e. which evidence makes the decoder produce the target, instead of on the cluster label. Include a null option (R3's own thought) so that deferring is learned rather than hand-tuned. Optionally, also train the decoder on the clusters the posterior favours.

## Model

```
p(y | x) = Σ_{c ∈ C(x) ∪ {∅}} p_η(c | x) · p_θ(y | x, c)          ∅ = R3's own thought (no evidence)
q(c | x, y) ∝ p_η(c | x) · p_θ(y | x, c)                            exact over the candidate set (teacher-forced)
C(x) = true cluster + top-15 of the mix predictor + 8 random        (24 + ∅; small fixed corpus → no stale index)
```

- **Prior / selector p_η.** A small head on R3's thought r (and the predictor logits), with 151 outputs (150 clusters + ∅). Loss: KL(q ‖ p_η), the Hindsight/PDist target.
  - A cluster label says only where the target lives. q says which evidence the decoder can **convert** into a hit. Locally, the decoder converts the true cluster about 63% of the time, and other clusters sometimes convert too. So q and the label can differ, and q's mass on ∅ is the defer signal.
- **Decoder p_θ.** Stage A: frozen (the `oracle_cfcluster150` checkpoint), as in RePlug. This avoids the slot-unlearning collapse of job 318248. Stage B, optional: EM-style training with loss −Σ_c SG(q(c)) log p_θ(y|x,c). The decoder is then trained on the clusters that work for this user, not only on the oracle cluster.
- **Inference.** Decode the top-K clusters under p_η plus ∅, and combine them with the **learned** weights p_η. A confident ∅ means use R3's own list; this is the learned deferral.

## Why this is not the failed verifier again

| | VRec-style verifiers (jobs 318248 / 318922) | this proposal |
|---|---|---|
| training target | cluster label (CE) | decoder posterior q(c\|x,y) (KL) |
| "don't intervene" | entropy gate (hand-designed) | ∅ is a class, trained on how often ∅ converts |
| what the decoder sees | a blend vector in the slot (ignored) | a hard single-cluster condition it was trained to trust (swap ratio ≈ 13) |
| hypotheses per row | 1 | K parallel decodes, weighted by p_η |

## Pre-registered go/no-go (E2-diag; measurement only, no training)

Compute q on 2,000 validation rows.
- **Kill.** On R3-miss rows, argmax q ≠ ∅ hits the target no more often than the 22.7% selector. The posterior is then as R3-correlated as the label, and the method is dead before any training.
- **Pass.** q identifies converting clusters on misses well above 22.7%. Train the prior (E3).

## Baselines

- R3 v1.
- Output RRF of v1 and fused CF (0.1346).
- Training-free path fusion and selection by p(c|x). This is a **baseline row, not a contribution**.
- Label-CE prior: same architecture, cluster-label target. This isolates the claim.
- Oracle best path (upper bound).

## Evidence requirements

- Full-test HR/NDCG @1,3,5,10,20 (@20 decoded with 10 beams), gained/lost vs R3, exact McNemar.
- HR@10 on R3-miss rows and on R3-hit rows (does the learned ∅ protect hits?).
- Selection accuracy on misses vs 22.7%.
- Risk–coverage curve of the ∅ probability.

## Contributions

1. A latent-variable formulation of CF evidence for an LLM latent recommender, with the prior trained on the exact decoder posterior. The survey and a fresh search found no recommender that supervises evidence selection this way; the external novelty check is pending.
2. A learned defer option (∅) that replaces hand-tuned gates.
3. Diagnostic: the oracle–selector gap, and below-chance conditioned likelihood, in latent-reasoning recommendation.
