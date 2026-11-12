# E1/E2 design — C1 recency-split bottleneck (for discussion, no code written yet)

Date: 2026-10-29. Status after P1: C11 promotion gate failed → C1 is the sole lead; this is the main training investment of the line.

## Idea recap

L7 hides the entire history from target tokens (strong content dependence, −11% HR@10). C1 relaxes it: target tokens see the **fixed instruction prefix + the last m history items + the fixed scaffold + r + earlier target tokens**; the **older** history is reachable only through r. Same mask in training, teacher-forced scoring, and KV-cache beam generation (Attempt-1 lesson). Pass 1 (computing r from the full history) is unchanged.

## Mask change (src/latent/latent_bottleneck.py + dataset)

Current `prompt_block` blocks `[end_of_matched_prefix, thought_pos)`. E1 adds a per-row `recency_start`:

```
block = valid & (pos < thought_pos) & ~visible_prefix & (pos < recency_start)
```

so `[recency_start, thought_pos)` — the last m titles plus the fixed "### Response:" scaffold — becomes visible. Position ids keep being recomputed over visible tokens (existing machinery), so the *number* of hidden older tokens does not leak.

**Where `recency_start` comes from (exact, not heuristic):** in `reasoning_dataset.get_history`, the history string is `"t1", "t2", ..., "tL"` in chronological order (most recent last). Build the prompt text in two pieces split at the opening quote of title `L−m+1`, tokenize the pieces separately, and concatenate the token lists; `recency_start` = len(first piece's tokens). This guarantees a clean token boundary (no title straddles the split). The collator carries `recency_start` next to `input_ids`; `LatentModel` passes it into `prompt_block`. Rows with history length ≤ m get `recency_start = end_of_prefix` (nothing hidden — effectively plain R3 on those rows; we report their fraction and, if non-trivial, metrics on the >m subset too).

Config: `latent_bottleneck = {"visible_prefix_ids": ..., "recency_m": m}`; checkpoint-name suffix `_recency_m{m}`.

## E1 runs

- Sweep m ∈ {2, 5, 10} on Musical, same recipe as L7 (same epochs, lr, batch; one latent token), new modules fp32, frozen code, outputs under `R3/output_dir/`.
- **Staged order**: run m=5 first together with its E2 baseline; launch m=2 and m=10 only after m=5's mask-integrity checks pass (saves 4 GPU-days if the implementation has a bug).
- Per-m **mask-integrity / exclusive-path checks** (cheap, before full eval):
  1. r fixed, older history swapped → max |Δlogit| ≈ 0 (L7's test gave 4.8e-4);
  2. r fixed, one of the last-m titles swapped → logits change (visible path alive);
  3. generation-time mask equals training mask on a probe batch.
- Eval: standard HR/NDCG @1,3,5,10 (@20 = @10 with 10 beams) + interventions raw/mean/shuffle_user, paired bootstrap, full test (n=8,313).

## E2 matched baselines (the review's blocking item #1)

For each trained m: an identical run — same recency mask, same budget (steps, lr, schedule, seed) — but the thought slot holds a **constant learned vector** instead of r (no r producer gradient). This isolates "how much accuracy does seeing the last m items buy by itself" from "what does r add on top".

## Claims / success criteria per split m (prespecified)

1. **K1 content dependence**: raw − mean and raw − shuffle_user significantly > 0 (paired CI).
2. **K2 r adds information**: E1(m) raw − E2(m) significantly > 0 on the same rows.
3. **K3 noninferiority vs plain R3**: paired noninferiority test, margin δ_NI = 0.005 HR@10 (plain R3 = 0.1226/0.0950).
4. Claims restricted to tested splits; the deliverable even if K3 fails at every m is the **content-dependence vs accuracy frontier** (L7 = m=0 endpoint, plain R3 = m=∞ endpoint).

## Budget

6 training runs ≈ 6 GPU-days (staged 2 + 4) + evals (~2 h each). Slurm per project defaults (researchshort, requeue, constraint l40s|a100|l40|a40|3090).

## Open questions for the user

1. Staged order (m=5 + baseline first) OK, or launch all 6 at once?
2. Should the fixed "### Response:" scaffold be visible (proposed: yes — it is user-independent) or stay hidden exactly as in L7?
3. E2 slot: constant **learned** vector (proposed) vs frozen train-mean r — learned is the stronger, fairer baseline.
4. Musical first, Toys/CDs replication (R1) only for the chosen operating point — confirm.
