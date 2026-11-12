# Refinement Report — what changed from raw C10b to the final proposal

1. **Success criterion hardened**: swap sensitivity alone → anchored complementarity audit (G1–G5), motivated by the audit-gaming counterexample (C25): a consistency-check model can pass raw+swap-collapse with zero useful latent prediction.
2. **Promotion separated from passage**: audit pass ≠ lead status; K4 (≥0.1226 vs original plain R3 on the identical-pool AND standard protocol) added as an explicit, separate gate.
3. **Producer specified** (reviewer gap #5): r̃ = g([r; s_u]), g fp32, trained only inside the correction fit; frozen title encoder; prespecified missing-target handling (excluded from fitting, counted in evaluation).
4. **Substitution diagnostic added** (G3 correction-alone) and **conventional comparator** (G5 SASRec-direct + scalar fusion) — both zero/near-zero-cost.
5. **Statistics tightened**: user-clustered paired bootstrap, repeated shuffle permutations, validation-only pilot decisions, Musical labeled exploratory, Toys/CDs confirmation.
6. **Complexity rejected**: C10 variants a/c/d, decoder adapters, MI penalties, GRPO — all explicitly out; failure retires the method rather than growing it.
7. **Backup pre-committed**: C1 recency-split frontier with its two reviewer-required controls, so a C10b failure has a named next step instead of an improvised pivot.
