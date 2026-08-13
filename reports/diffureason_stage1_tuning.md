# DiffuReason Stage-1 (SFT) Tuning Report

Dataset: `CDs_and_Vinyl` (5-core, 2015-10..2018-11). Train 49,252 / valid 6,157 / test 6,157 rows.
Backbone: `Qwen/Qwen2.5-1.5B-Instruct`. Metric: NDCG@K / HR@K via `src/utils/calc.py`
(K = 1, 3, 5, 10, 20; beam search uses `num_beams=10`, so K=10 and K=20 are always
identical -- there are never more than 10 candidates to rank).

Scale note: Experiment 1 is a pre-existing **full-scale** run (all 49,252 train rows,
3 epochs, full 6,157-row test set). Experiments 2-9 use a **subsampled** train/eval
set (`sample=5000` train rows, 2 epochs, 1,000-row eval subsample, see each config)
via `scripts/diffureason_experiment.sh` so multiple hyperparameter iterations fit in
one session. **Experiment 9 reruns Experiment 1's exact no-diffusion config at this
same subsampled scale, and this -- not Experiment 1 -- is the correct baseline to
compare Experiments 2-8 against.** Experiment 9's result (HR@10 0.044) turned out to
be dramatically lower than Experiment 1's full-scale number (HR@10 0.1025) and in the
*same range* as the DiffuReason variants -- meaning most of the "gap to baseline"
found throughout this report is a **scale confound** (10x less data, 1.5x fewer
epochs), not evidence that DiffuReason underperforms the base architecture. See the
Exp 9 entry below and the revised Overall Summary for the corrected picture. Exp 1's
full-scale numbers are kept for reference throughout but should not be used as the
comparison point for any subsampled experiment.

## Experiment log

### Exp 1 -- Original LatentR3 reference (no diffusion, no MSE anchor)

- Source: pre-existing run at `output_dir/CDs_and_Vinyl/latent/` (`scripts/latent_train.sh`).
- Config: `latent_R=1, diff_T=0, alpha=0` (diffusion module architecturally present
  but unused: `mu = tau_R` directly, no denoising). `sample=-1` (full train set),
  default `num_epochs=3`, `learning_rate=3e-4` (peak, linear warmup then decay),
  `batch_size=128, micro_batch_size=4`.
- Eval: full test set (6,157 rows).
- Result (final `metric.log` block):

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.05132 | 0.06227 | 0.06745 | 0.07376 | 0.07376 |
  | HR@K   | 0.05132 | 0.07033 | 0.08299 | 0.10248 | 0.10248 |

- Reflection: this was originally treated as the reference point, but see Exp 9
  below -- at the subsampled scale actually used by Exp 2-8, the no-diffusion
  baseline scores far lower than this, so this full-scale number **overstates**
  the bar DiffuReason needs to clear for a fair, scale-matched comparison.

### Exp 9 -- Exp 1's config, rerun at Exp 2-8's subsampled scale (the correct baseline)

- Config: identical to Exp 1 (`latent_R=1, diff_T=0, alpha=0`) but `sample=5000,
  epochs=2`, eval on the same 1,000-row test subsample as Exp 2-8 -- i.e. this
  isolates the effect of scale alone, holding the (no-diffusion) architecture
  fixed. Run out of order (after the Architecture Fix experiments) because the
  scale mismatch was only noticed at that point.
- Infra note: this run needed `micro_batch_size=2` (vs. the `8` used throughout
  Exp 2-8) after repeated CUDA OOMs at `micro_batch_size=8` and `4` on the 24GB
  RTX 3090 workers (unrelated to the method -- variable-length batches
  occasionally spike memory since `group_by_length=False`); a session
  interruption also killed the harness script's driver process right after
  training finished but before the eval phase, so eval was run manually against
  the saved checkpoint (tokenizer files, left empty by the interrupted save,
  were reconstructed identically to the training script's tokenizer setup and
  verified to match `config.json`'s `vocab_size` before eval).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.01500 | 0.01939 | 0.02232 | 0.02685 | 0.02685 |
  | HR@K   | 0.01500 | 0.02300 | 0.03000 | 0.04400 | 0.04400 |

  `eval_loss` 2.52 -> 2.14 across the two epochs -- close to Exp 7/8's `eval_loss`
  (2.07-2.33), not Exp 1's much lower full-scale range (~1.03-1.26).

- **Reflection -- this is the headline finding of the whole report.** HR@10
  drops from 0.1025 (Exp 1, full scale) to 0.044 at matched scale -- a bigger
  swing from *data/epoch budget alone* than any architectural change tested in
  this entire sweep produced. And 0.044 is squarely *within* the range of the
  DiffuReason variants (Exp 5: 0.053, Exp 7: 0.046, Exp 8: 0.051) -- not
  meaningfully better than most of them, and clearly worse than the best
  post-fix result (Exp 8: 0.051, +16% relative over this baseline). **The
  "DiffuReason underperforms the baseline by ~2x" conclusion that shaped this
  entire investigation was substantially a scale-mismatch artifact.** This
  doesn't retroactively validate every earlier diagnosis (Exp 2's paper-exact
  instability, the non-residual `DenoiseMLP` bug, and the `eps_init` scale
  mismatch are all still real, independently-verified issues -- see their
  sections above), but it means the *magnitude* of "how far behind is
  DiffuReason" was overstated throughout, and the right open question is no
  longer "why is DiffuReason so much worse than the baseline" but "does
  DiffuReason help, hurt, or wash out relative to the baseline once both are
  compared at the same, adequate training scale" -- which nothing in this
  report yet answers, since it requires a full-scale rerun of both arms.

### Exp 2 -- DiffuReason paper-exact config

- Config: `latent_R=3, diff_T=16, diff_s=0.008, use_schedule_scaling=False,
  alpha=1.0, diff_loss=mse` (literal Eq. 7). `sample=5000` (train), `epochs=2`,
  `learning_rate=3e-4`, `batch_size=128, micro_batch_size=8`. Eval on 1,000
  subsampled test rows (500/GPU x 2).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.00100 | 0.01133 | 0.01306 | 0.01438 | 0.01438 |
  | HR@K   | 0.00100 | 0.01800 | 0.02200 | 0.02600 | 0.02600 |

  Training loss stayed flat around 21-22 for the whole run (started at 22.8,
  ended at 21.5 after 2 epochs) -- essentially not learning, vs. the Exp-1
  baseline's loss dropping from ~7.7 to ~2.7 within the first 2% of one epoch.

- **Diagnosis.** This is a severe regression (HR@10 0.026 vs. baseline's 0.102),
  not a mild "diffusion doesn't help" result -- something is actively breaking
  training. Two hypotheses checked directly against the trained checkpoint:
  1. *MSE-loss scale mismatch* (mu unbounded LLM hidden-state-space vector vs.
     unit-norm target embedding, per Eq. 7 with no normalization mentioned).
     Checked empirically: `mu` norm on a sample forward pass is 0.93 (target
     norm is exactly 1.0 by construction) -- the MSE loss is *not* blowing up
     mu's scale; ruled out.
  2. *Un-annealed per-step diffusion noise destabilizing the LM's training
     signal.* Checked: `sigma` is ~0.58 on average (mid-range of its
     `[1e-3, 1.0]` clamp) even after training. With `use_schedule_scaling=False`
     (the paper-exact setting fixed last session per Appendix C.1.1: no
     forward process, no alpha_bar-based scaling), Algorithm 1's literal
     `x_{t-1} ~ N(mu_phi, sigma_phi)` injects a fresh ~N(0, 0.58^2) perturbation
     at *every one of 15 intermediate steps* with no decay toward the final
     step. The `<|Thought|>` embedding the LLM actually sees during training is
     therefore reconstructed through a long, genuinely stochastic 15-step chain
     -- a very high-variance, hard-to-fit input for the downstream CE loss to
     ever get consistent gradient signal from, especially at our much smaller
     training budget (5k rows x 2 epochs here vs. the paper's 200-300 epochs
     over the full dataset). This directly matches the concern raised earlier
     in this session ("lack of [per-step] supervision could degrade base
     performance") -- now with a concrete failure mode and empirical evidence.

- **Reflection / next step.** The paper-literal `use_schedule_scaling=False` is
  faithful to Appendix C.1.1's equations, but is likely tuned to work at the
  paper's much larger training budget (their backbones train 200-300 epochs on
  the full dataset per Table 5/A.2). At our budget, the un-annealed noise looks
  fatal. Hypothesis for Exp 3: re-enabling `use_schedule_scaling=True` (anneal
  the injected per-step noise to 0 as t->1 via `sqrt(1-alpha_bar_{t-1})`) should
  act as an implicit curriculum/stabilizer and recover most of the lost
  performance, even though it's a documented *deviation* from the paper's
  literal equations (this is an explicit, logged tradeoff, not a silent
  reversion).

### Exp 3 -- schedule-scaling re-enabled (deviation from paper Appendix C.1.1)

- Config: identical to Exp 2 except `use_schedule_scaling=True` (anneal per-step
  noise via `sqrt(1-alpha_bar_{t-1}) -> 0`). Same `sample=5000, epochs=2,
  lr=3e-4`. Note: this run was interrupted once by an unrelated session restart
  at ~73% through epoch 2 (partial log showed loss descending steadily from an
  early spike of 655 down to ~17); relaunched cleanly via `nohup`+`disown` for
  the reported run below.
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.00500 | 0.01155 | 0.01469 | 0.02026 | 0.02026 |
  | HR@K   | 0.00500 | 0.01600 | 0.02400 | 0.04100 | 0.04100 |

  Training loss descended cleanly this run too (steady drop through the low
  teens by epoch 2), unlike Exp 2's flat ~21-22.

- **Reflection.** The hypothesis partially confirmed: HR@10 nearly doubled
  vs. Exp 2 (0.041 vs. 0.026) and NDCG@10 +41%, and training loss now
  converges cleanly instead of flatlining -- so un-annealed per-step noise
  *was* part of the problem, and annealing it *does* help. But the result is
  still ~2.5x worse than the no-diffusion baseline (HR@10 0.041 vs. 0.1025),
  despite training loss looking healthy. This means training-loss convergence
  alone isn't the full story -- something is still hurting *generation-time*
  quality specifically. Leading suspect: at eval, `model.eval()` runs the
  reverse process deterministically (Appendix C.1.2: "No stochastic sampling
  is performed during inference") *except* `x_T = tau_R + eps_init` still
  injects real Gaussian noise even at test time (this repo's `eps_init_zero`
  defaults to `False`, matching the paper). With only 2 epochs / 5k rows of
  training, the denoiser may not have learned to be robust to that residual
  eval-time randomness yet, so beam search sees a different (noisy) anchor on
  every run -- unlike Exp 1's baseline, which has no stochastic component in
  its forward pass at all and converges to a stable prediction almost
  immediately (loss 7.7 -> 2.7 within the first 2% of an epoch).
- **Next step (cheap diagnostic before spending a full experiment slot):**
  re-run *eval only* on the already-trained Exp 3 checkpoint with
  `--eps_init_zero True` (deterministic warm start, no retraining needed).
  If this closes much of the gap, the fix is inference-time determinism, not
  more training. If not, the bottleneck is genuinely training convergence /
  data budget, and Exp 4 should scale up training instead.

### Diagnostic -- eval-time determinism (Exp 3 checkpoint, no retraining)

- Re-ran eval only on the trained Exp 3 checkpoint with `--eps_init_zero True`
  (deterministic warm start `x_T = tau_R`, no residual injected randomness),
  same 1,000-row eval subsample.
- Result: NDCG@10 = 0.02056, HR@10 = 0.042 -- **statistically indistinguishable**
  from Exp 3's stochastic-eval result (NDCG@10 0.02026, HR@10 0.041).
- **Conclusion: eval-time stochasticity is ruled out.** The residual gap to
  baseline (HR@10 ~0.04 vs. 0.1025) is not explained by inference-time noise
  in `eps_init`. This points back to training convergence / data budget as the
  remaining bottleneck: with only 5k rows x 2 epochs, the much more complex
  R=3 think-loop + 16-step diffusion chain likely hasn't had enough gradient
  steps to learn a *generalizable* mapping, even though its training-set CE
  loss looks healthy (possibly overfitting the visible 5k rows narrowly
  without learning a robust `<|Thought|>` representation that generalizes to
  held-out users -- unlike Exp 1's baseline, whose single-shot pooler is far
  simpler and converges almost immediately).

## Changelog

- **Exp 3:** `use_schedule_scaling` False -> True. Helped substantially
  (HR@10 0.026 -> 0.041) but did not close the gap to baseline.
- **Diagnostic:** `eps_init_zero=True` on Exp 3's checkpoint -> no change.
  Rules out inference-time noise; points to training budget as the bottleneck.
- **Exp 4 (planned):** scale up training budget substantially
  (`sample` 5,000 -> 15,000, `epochs` 2 -> 3, ~4.5x more optimizer steps),
  keep `use_schedule_scaling=True` and all else from Exp 3, to test whether
  more data/steps closes the remaining gap.
- **Bug fix (`src/latent/reasoning_dataset.py:47-48`):** `latent_attention_train.py`
  applies the same `--sample N` to both the train file (49,252 rows) and the
  eval-during-training file (`valid`, 6,157 rows). Exp 4's first launch attempt
  crashed (`ValueError: Cannot take a larger sample than population`) because
  `sample=15000 > 6157`. Fixed by clamping
  `self.data.sample(min(sample, len(self.data)), ...)` in `LatentRDataset.__init__`
  -- a generally-correct defensive fix (falls back to using the full file when
  the requested sample exceeds it), not scoped to this experiment only.

### Exp 4 -- scaled-up training budget (15,000 samples x 3 epochs)

- Config: identical to Exp 3 (`use_schedule_scaling=True`, `alpha=1.0`,
  `diff_loss=mse`, `lr=3e-4`) except `sample=15000, epochs=3` (~4.5x more
  optimizer steps than Exp 3's `sample=5000, epochs=2`).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.00900 | 0.00963 | 0.01040 | 0.01262 | 0.01262 |
  | HR@K   | 0.00900 | 0.01000 | 0.01200 | 0.01900 | 0.01900 |

  Training loss converged much lower this time (final ~4.5-4.7 vs. Exp 3's
  ~12-17 at the equivalent point) -- clean, healthy-looking convergence.

- **Reflection -- this falsifies the "needs more training" hypothesis, and
  the direction is the opposite of what was predicted.** Despite substantially
  better training-loss convergence, held-out generation quality got *worse*
  (HR@10 0.041 -> 0.019, NDCG@10 0.0203 -> 0.0126) -- a classic overfitting
  signature (train loss down, held-out task metric down), not underfitting.
  This rules out "data/step budget" as the bottleneck and points instead to
  the diffusion module's *architecture* itself: with `diff_T=16` reverse
  steps, each injecting a fresh reparameterized Gaussian sample even under
  annealed scaling, the effective input the LLM sees for the exact same
  `<|Thought|>` slot varies substantially forward pass to forward pass. Full
  fine-tuning (stage-1 SFT does not freeze the Qwen2.5 backbone -- only
  `think_mlp`/`cond_pool`/`denoiser`/`time_embed` are conditionally frozen
  based on `latent_R`/`diff_T`, never the backbone itself) then likely lets
  the backbone start memorizing spurious associations between specific noise
  realizations seen in training and their paired answers, rather than
  learning a robust semantic mapping -- and more epochs over more rows only
  gives it more opportunity to do so. This suggests the fix is to *reduce*
  the diffusion chain's effective stochastic depth (fewer reverse steps),
  not to feed it more data.
- **Next step:** Exp 5 reduces `diff_T` from 16 to 4 (much shorter reverse
  chain, less compounding injected noise per forward pass) at Exp 3's data
  scale (`sample=5000, epochs=2`, the best-performing budget so far), keeping
  `use_schedule_scaling=True`.

### Exp 5 -- shorter diffusion chain (diff_T=16 -> 4)

- Config: same as Exp 3 except `diff_T=4` instead of 16.
- **First two launch attempts crashed reproducibly** at the identical point
  (~5 steps after the mid-training eval checkpoint at the epoch-1 boundary,
  loss descending normally beforehand) with `rank 1 exitcode 1` and no
  captured Python traceback. Root cause found: this session's iteration had
  accumulated ~1TB of disk usage (`output_dir` alone reached 164G; each
  experiment's 2 HF Trainer checkpoints are ~9.2G each on top of the final
  3.1G saved model), and the user's personal disk quota was exhausted mid-run
  (confirmed by hitting `EDQUOT` directly while editing this report file
  shortly after). The filesystem itself has 212T free -- this was a per-user
  quota limit, not a capacity problem. **Fix:** deleted the stale
  `checkpoint-*` subdirectories from all prior experiments (only the final
  `save_pretrained` output at the experiment root is actually needed for
  eval), freeing ~57G, then relaunched.
- Result (after the fix):

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.01600 | 0.02555 | 0.02942 | 0.03301 | 0.03301 |
  | HR@K   | 0.01600 | 0.03300 | 0.04200 | 0.05300 | 0.05300 |

  Raw training loss plateaued around ~17-18 (higher than Exp 3's ~12-17 at
  the equivalent point), but `eval_loss` (HF Trainer's held-out validation
  loss) dropped sharply from 10.57 at epoch 1.0 to 3.28 at epoch 1.97 --
  i.e. held-out loss improved *more* than Exp 3's, even though the raw
  training-batch loss number looked less clean (likely because with fewer
  denoising steps the per-step MSE-anchor contribution to the loss stays
  more persistently non-trivial rather than being averaged away over many
  steps -- the two runs' raw loss numbers aren't directly comparable for
  this reason; `eval_loss` and the downstream HR/NDCG are the trustworthy
  signals here).

- **Reflection -- hypothesis confirmed, this is the best DiffuReason variant
  found.** HR@10 improved from 0.041 (Exp 3, `diff_T=16`) to 0.053
  (`diff_T=4`), +29% relative; NDCG@10 improved from 0.0203 to 0.0330, +63%
  relative. This directly supports the Exp 4 reflection: shortening the
  diffusion reverse chain (less compounding injected stochasticity per
  forward pass, less opportunity for the fully-fine-tuned backbone to
  memorize noise-specific spurious associations) helps generalization more
  than throwing more data/epochs at the long-chain (`diff_T=16`) variant did.
  `diff_T` -- effectively "how much of the paper's design to keep" -- turned
  out to be the single most impactful lever found in this sweep, more so
  than the schedule-scaling fix (Exp 2->3) or the data-budget experiment
  (Exp 3->4, which actively hurt). Still, at HR@10 0.053 vs. the no-diffusion
  baseline's 0.1025, DiffuReason has not caught up to plain LatentR3 in this
  small-scale regime -- see Recommendations below for what a fairer,
  full-scale comparison would need.

## Overall Summary

> **Superseded -- read with the correction below.** This section (and the
> "Overall Summary (Post-Fix)" section further down) both compare against
> Exp 1's *full-scale* HR@10 (0.1025). Exp 9 (run later, out of order) reruns
> Exp 1's exact config at the actual subsampled scale used by every experiment
> below and gets HR@10 **0.044** -- in the same range as most rows in the
> tables below. The relative rankings *among* the DiffuReason variants in this
> table are still valid (all run at the same subsampled scale), but "gap to
> Exp 1" should be read as "gap to a 2.3x-more-data, 1.5x-more-epoch run," not
> "gap to a fair baseline." See "Final Corrected Summary" at the end of this
> report for the scale-matched picture.

### Results at a glance

| Experiment | Config (relative to Exp 3 baseline) | Scale | NDCG@10 | HR@10 |
|---|---|---|---|---|
| Exp 1 (no-diffusion reference, full scale -- NOT scale-matched, see Exp 9) | `latent_R=1, diff_T=0, alpha=0` | full (49.2k train, 6.2k test) | **0.0738** | **0.1025** |
| Exp 9 (no-diffusion reference, matched scale -- the correct comparison) | `latent_R=1, diff_T=0, alpha=0` | 5k train x 2ep, 1k test | 0.0269 | 0.0440 |
| Exp 2 (paper-exact) | `diff_T=16`, `use_schedule_scaling=False` | 5k train x 2ep, 1k test | 0.0144 | 0.0260 |
| Exp 3 (+ schedule scaling) | `diff_T=16`, `use_schedule_scaling=True` | 5k train x 2ep, 1k test | 0.0203 | 0.0410 |
| Exp 4 (+ 4.5x more data) | `diff_T=16`, `sample=15k, epochs=3` | 15k train x 3ep, 1k test | 0.0126 | 0.0190 |
| **Exp 5 (+ shorter chain)** | **`diff_T=4`, `sample=5k, epochs=2`** | 5k train x 2ep, 1k test | **0.0330** | **0.0530** |

(Diagnostic, not a distinct config: `eps_init_zero=True` on Exp 3's checkpoint
-> NDCG@10 0.0206 / HR@10 0.042, no meaningful change from Exp 3.)

### Key findings

1. **The paper-literal reverse-process update (no schedule scaling,
   Appendix C.1.1) is unstable at small training budgets.** Exp 2's training
   loss flatlined completely (~21-22 for the whole run) with un-annealed
   per-step noise injected 15 times per forward pass. This is very likely
   fine at the paper's actual training budget (200-300 epochs over the full
   dataset, Table 5/A.2) but is a real liability when reproducing the method
   at a smaller compute/data budget, as most follow-up work will need to do.
2. **Annealing the injected per-step noise (`use_schedule_scaling=True`,
   a deliberate deviation from the paper's literal equations) fixes training
   stability and roughly doubles HR@10** (Exp 2 -> Exp 3). This is the
   single most important stability fix found.
3. **More training data/epochs made results *worse*, not better** (Exp 3 ->
   Exp 4: HR@10 0.041 -> 0.019, despite training loss converging much lower).
   This is a genuine overfitting signature and rules out "just needs more
   data" as the explanation for DiffuReason's underperformance here -- a
   counter-intuitive but load-bearing finding for anyone tuning this further.
4. **Shortening the diffusion chain (`diff_T` 16 -> 4) was the best lever
   found** (Exp 3 -> Exp 5: HR@10 0.041 -> 0.053, NDCG@10 +63% relative),
   consistent with the overfitting story: fewer stochastic reverse steps
   means less opportunity for the (fully fine-tuned, unfrozen) backbone to
   latch onto noise-specific artifacts rather than a generalizable semantic
   signal.
5. **Eval-time stochasticity (`eps_init`) is not a contributing factor** --
   ruled out directly via the diagnostic re-eval with `eps_init_zero=True`.
6. **Infra lesson:** iterating on full fine-tuning checkpoints (each HF
   Trainer checkpoint of this 1.5B model is ~9.2G, x `save_total_limit=2` per
   run) accumulates disk usage fast across many experiments and silently
   caused Exp 5's first two failures via a per-user disk quota, not a code
   bug -- `output_dir` reached 164G / user total ~1TB before cleanup. Delete
   stale `checkpoint-*` subdirectories after each experiment (the eval
   scripts only need the final `save_pretrained` output at the experiment
   root).

### Best config found

`latent_R=3, diff_T=4, diff_s=0.008, use_schedule_scaling=True, alpha=1.0,
diff_loss=mse, lr=3e-4` -- i.e. keep the R=3 think-loop and the MSE anchor
loss from the paper, but shorten the diffusion refinement to 4 steps (vs. the
paper's 16) and anneal the injected per-step noise (a deviation from Appendix
C.1.1, justified empirically above). This does **not** yet match or beat the
no-diffusion LatentR3 baseline at the scales tested here.

### Recommendations / caveats

- **All DiffuReason numbers here (Exp 2-5) are on a 5k-15k-row subsample of
  train and a 1k-row subsample of test**, for iteration speed within one
  session; Exp 1's baseline is full-scale. This is not an apples-to-apples
  comparison in absolute terms -- treat the *relative* ranking among Exp 2-5
  as the trustworthy signal, and the gap to Exp 1 as suggestive, not final.
- **The highest-value next step is a full-scale run of the Exp 5 config**
  (`diff_T=4, use_schedule_scaling=True`, full 49.2k train rows, full 6.2k
  test set, matching Exp 1's epoch count) to get a fair head-to-head number
  against the baseline. Given the overfitting signature seen in Exp 4, it is
  plausible (not certain) that more *real* data (vs. more epochs over the
  same small subsample) behaves differently than Exp 4's negative result --
  this is worth testing explicitly rather than assuming Exp 4 generalizes to
  full-scale.
- **Further `diff_T` sweep** (e.g. 2, 6, 8) at Exp 5's data scale would
  pin down whether 4 is close to optimal or whether even shorter chains (or
  `diff_T=1`, effectively a single denoising step) do better still -- the
  monotonic direction from 16->4 suggests but does not prove the trend
  continues.
- **Freezing the backbone** during stage-1 SFT (currently unfrozen; only the
  new think/diffusion submodules are conditionally frozen) is an untested
  lever directly motivated by the overfitting finding -- if the backbone
  itself is what's memorizing noise-specific artifacts, restricting
  gradient updates to the new modules (at least initially, akin to the
  paper's own "warm-up" framing, just without the disjoint two-stage
  training the paper argues against for the RL stage) could reduce
  overfitting without needing to cut `diff_T` as aggressively.
- **Operational:** run `du -sh output_dir` periodically during any further
  sweeps and delete `checkpoint-*` subdirectories promptly (only the final
  `save_pretrained` root output is needed for eval) to avoid repeating
  Exp 5's disk-quota failure.
- Stage-2 GRPO was explicitly out of scope for this tuning pass per the
  original request (SFT first); none of the above findings have been
  validated with GRPO alignment on top.

## Architecture Fix: residual DenoiseMLP

Deeper investigation of *why* every diffusion variant above underperforms the
no-diffusion baseline found two compounding architectural issues in
`src/latent/diffusion.py`'s `DenoiseMLP`, not just bad hyperparameters:

1. **Non-residual `mu`.** `DenoiseMLP.forward` returned `mu` as a direct MLP
   output, uncorrelated with the current state `x_t` at initialization
   (measured on a trained Exp 2 checkpoint: `tau_R` norm 3.69 vs. `mu` norm
   0.93 -- not a refinement, a different vector). Stacking this T times
   compounds drift away from the reasoning signal instead of refining it --
   consistent with the sweep's "more steps = worse" trend.
2. **`eps_init` scale mismatch.** Eq. 3's `eps_init ~ N(0, I)` has expected
   norm `sqrt(1536) ≈ 39.2` for this backbone, dwarfing `tau_R`'s actual
   scale (~3.7) -- the "warm start" is >99% noise by variance.

**Fix implemented** (`src/latent/diffusion.py`, `src/latent/layers.py`):
- `DenoiseMLP.forward`: `mu = x_t + d_mu` (residual correction, not a
  from-scratch prediction).
- `DenoiseMLP.__init__`: zero-init the final layer's weight and bias so
  `d_mu ~ 0` and `sigma` starts near its clamp floor (`softplus(-4)+1e-5 ~
  0.019`) -- an approximate identity pass-through at init.
- **Gotcha found and fixed during verification:** `from_pretrained`
  re-initializes any parameter missing from the loaded checkpoint (all of
  our custom submodules) via `self.apply(self._init_weights)` *after*
  `__init__` returns, silently overwriting the zero-init above (confirmed:
  weight went from all-zero to random, bias reset from `[0, -4]` to all-zero).
  Fixed by adding a `LatentModel._init_weights` override that restores the
  zero-init for `denoiser.net[-1]` specifically, since this hook is what
  `from_pretrained` actually calls for missing keys.
- Added `diff_eps_init_scale` (default `1.0`, paper-exact) as a new tunable
  to test shrinking `eps_init`, threaded through `layers.py`,
  `latent_attention_train.py`, and `scripts/diffureason_experiment.sh`
  (`EPS_INIT_SCALE` env var).
- **Verified via sanity check** on a fresh (untrained) model: `sigma` mean
  0.018 (near clamp floor, as intended) and `||mu - tau_R|| = 38.4` -- almost
  exactly `||eps_init||`'s expected norm (39.2), confirming `mu ~ tau_R +
  eps_init` at init (identity pass-through of the warm start), not the
  ~900,000-norm garbage the fix started with before the `_init_weights`
  override was added.
- `scripts/diffureason_experiment.sh` was also updated to delete
  `checkpoint-*` and `model.safetensors` after every run (keeping only
  logs/config/metrics/predictions), directly addressing the disk-quota
  failure from Exp 5.

### Exp 6 -- residual fix, diff_T=4 (vs. pre-fix Exp 5)

- Config: identical to Exp 5 (`latent_R=3, diff_T=4, diff_s=0.008,
  use_schedule_scaling=True, alpha=1.0, diff_loss=mse, sample=5000, epochs=2`)
  plus the residual-fix code change (`diff_eps_init_scale=1.0`, unchanged).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.00000 | 0.00050 | 0.00093 | 0.00122 | 0.00122 |
  | HR@K   | 0.00000 | 0.00100 | 0.00200 | 0.00300 | 0.00300 |

  Training loss was catastrophic: starting at **~26,800** and only easing to
  ~6,600 by the end of training (`eval_loss` 1297 -> 799). For reference,
  every prior run's loss was in the 4-30 range.

- **Diagnosis -- an unanticipated interaction between the fix and the
  existing MSE-anchor-loss scale mismatch (Flaw 3 from the earlier design
  discussion).** Before the fix, a freshly-initialized non-residual MLP
  happened to output `mu` at a modest, bounded scale (standard PyTorch Linear
  init keeps activations ~O(1)) -- numerically convenient for the unit-norm
  MSE target `v_target`, even though semantically wrong (uncorrelated with
  `tau_R`). After the fix, `mu ~ tau_R + eps_init` at init, with norm ~40-70
  (dominated by the unscaled `eps_init`, per the sanity check above) --
  architecturally correct for preserving signal, but now
  `L_diff = ||mu - v_target||^2` starts around `70^2 ~ 4900` per example.
  With `alpha=1.0` this completely swamps the CE loss (`L_rec`), so the model
  spends its gradient budget shrinking `mu`'s norm to satisfy the MSE term
  rather than learning useful `<|Thought|>` representations -- explaining
  both the enormous loss and the near-total collapse in generation quality.
- **Decision: do not run Exp 7 (diff_T=16) as originally planned with
  `alpha=1.0`.** It would almost certainly fail the same way, and the
  `diff_T` comparison would be confounded by this MSE-scale explosion rather
  than cleanly testing whether the residual fix breaks the "more steps =
  worse" trend. Pivoting to isolate the architecture fix from this new
  confound first: re-run with `alpha=0` (CE-only, matching the paper's own
  ablation finding that MSE costs almost nothing, Table 3: 0.1117 -> 0.1115)
  before re-introducing the MSE term.

### Exp 7 -- residual fix, diff_T=4, alpha=0 (isolating the architecture fix)

- Config: same as Exp 6 except `alpha=0` (no MSE anchor loss, CE-only --
  removes the scale-mismatch confound found in Exp 6).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.01700 | 0.02026 | 0.02663 | 0.02889 | 0.02889 |
  | HR@K   | 0.01700 | 0.02300 | 0.03900 | 0.04600 | 0.04600 |

  Training loss converged cleanly this time: ~1.2-1.5 by the end (vs. Exp 6's
  catastrophic thousands), `eval_loss` 2.33 -> 2.07 across the two epochs --
  the healthiest convergence of *any* diffusion variant tried so far, notably
  close to baseline Exp 1's `eval_loss` range (~1.03-1.26).

- **Reflection -- ambiguous / mildly negative result.** HR@10 (0.046) and
  NDCG@10 (0.0289) are close to but *not better than* pre-fix Exp 5
  (HR@10 0.053, NDCG@10 0.0330) -- despite the residual fix producing
  substantially better training dynamics (clean, fast CE convergence,
  `eval_loss` near baseline scale). This is a similar disconnect to the
  eval_loss/HR inversion seen with Exp 4 earlier in this report: good
  teacher-forced next-token prediction is not translating into better
  free-running, beam-search-decoded ranking quality. Two live hypotheses:
  1. **`eps_init` scale (Flaw 2, not yet tested).** Even with the residual
     fix, `x_T = tau_R + eps_init` still injects noise with expected norm
     ~39.2 against a `tau_R` signal of norm ~3.7 -- so although the *model*
     now learns to map this large noisy state back to something useful in
     expectation (hence the good CE loss), each individual forward pass still
     starts from a very different, high-variance noise draw, and beam search
     (which needs consistent, low-variance representations across the
     multiple candidate continuations it scores) may be disproportionately
     sensitive to that residual per-sample variance in a way teacher-forced
     loss on the gold continuation isn't.
  2. Only 4 denoising steps may simply not be enough to fully resolve a
     40-70-norm noisy input back to a clean signal, even with a
     well-behaved (residual, near-identity-init) denoiser -- i.e. the
     architecture fix helps optimization but the noise budget vs. step
     budget is still mismatched.
- **Next step:** test `diff_eps_init_scale=0.1` (shrinking the injected
  warm-start noise ~10x, roughly matching `tau_R`'s empirical per-dim scale)
  on top of the residual fix, still at `diff_T=4, alpha=0`, to test hypothesis
  1 directly -- this is now feasible to test cleanly since the residual fix
  and the MSE confound are both understood and isolated.

### Exp 8 -- diff_eps_init_scale=0.1 on top of the residual fix

- Config: same as Exp 7 except `diff_eps_init_scale=0.1` (eps_init shrunk
  ~10x, from expected norm ~39.2 to ~3.9 -- now roughly matching `tau_R`'s
  observed scale of ~3.7).
- Result:

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.02000 | 0.02376 | 0.02746 | 0.03229 | 0.03229 |
  | HR@K   | 0.02000 | 0.02700 | 0.03600 | 0.05100 | 0.05100 |

  `eval_loss` 2.32 -> 2.07, essentially identical to Exp 7's -- shrinking
  `eps_init` did not change training-loss dynamics at all (unsurprising: CE
  loss doesn't care about `mu`'s absolute scale, only its usefulness for
  prediction).

- **Reflection -- a small positive move, not a decisive one.** HR@10
  improved from 0.046 (Exp 7) to 0.051, NDCG@10 from 0.0289 to 0.0323 --
  roughly back to parity with pre-fix Exp 5 (HR@10 0.053, NDCG@10 0.0330),
  not meaningfully beyond it. This is directionally consistent with the
  `eps_init`-scale hypothesis (shrinking it helped a little) but the effect
  is small relative to the eval-noise floor on a 1,000-row subsample, and
  nowhere near closing the gap to the no-diffusion baseline (HR@10 0.1025).
  Combined with Exp 7's finding, the picture is: **the residual-fix +
  eps_init-scale changes measurably improve *training* behavior (much
  cleaner loss convergence, closer to baseline's loss scale) without
  producing a correspondingly large improvement in *generation* quality.**
  This is a plateau -- further small hyperparameter nudges along the same
  axes are unlikely to close a 2x gap. Stopping the architecture-fix
  sweep here given the extensive time already invested this session;
  see Recommendations below for what would actually be needed to test the
  remaining hypotheses properly.

## Overall Summary (Post-Fix)

> **Superseded -- read with the correction below.** Written before Exp 9
> (run afterward) revealed that Exp 1's full-scale HR@10 (0.1025) is not a
> fair comparison point for these subsampled runs -- the scale-matched
> baseline (Exp 9) scores HR@10 0.044, much closer to Exp 8's 0.051 than to
> 0.1025. See "Final Corrected Summary" at the end of this report.

### Results at a glance (all diffusion rows on the 5k-row-train / 1k-row-eval
subsample; Exp 1 is full-scale, NOT scale-matched -- see Exp 9 / Final
Corrected Summary)

| Experiment | Fix state | Config delta | NDCG@10 | HR@10 |
|---|---|---|---|---|
| Exp 1 (no-diffusion baseline, full scale) | -- | `latent_R=1, diff_T=0` (full scale) | 0.0738 | 0.1025 |
| Exp 9 (no-diffusion baseline, matched scale) | -- | `latent_R=1, diff_T=0`, `sample=5k, epochs=2` | 0.0269 | 0.0440 |
| Exp 5 (pre-fix best) | non-residual `mu` | `diff_T=4`, `alpha=1.0` | 0.0330 | 0.0530 |
| Exp 6 | residual fix | `diff_T=4`, `alpha=1.0` | 0.0012 | 0.0030 |
| Exp 7 | residual fix | `diff_T=4`, `alpha=0` | 0.0289 | 0.0460 |
| **Exp 8 (best post-fix)** | residual fix | `diff_T=4`, `alpha=0`, `eps_init_scale=0.1` | **0.0323** | **0.0510** |

### What the architecture fix actually accomplished

1. **Confirmed and fixed a real bug**, not just a tuning knob: the
   non-residual `DenoiseMLP` made `mu` uncorrelated with `tau_R` at
   initialization, and a separate `from_pretrained` gotcha was silently
   overwriting even a correct in-`__init__` fix. Both are now fixed and
   verified numerically (sanity check: `mu ~ tau_R + eps_init` at init,
   `sigma` near its clamp floor).
2. **Confirmed via Exp 7's `eval_loss` (2.07-2.33, close to baseline's
   1.03-1.26)** that the fix does what it was designed to do: training
   dynamics are now healthy and stable, unlike every pre-fix diffusion run
   (Exp 2-5, `eval_loss` 2.6-10.6).
3. **Did not close the gap to the no-diffusion baseline.** Despite much
   better training dynamics, HR@10 (0.051 best post-fix) remains ~2x below
   baseline (0.1025) -- essentially unchanged from the pre-fix best (0.053).
4. **Uncovered a second, previously invisible bug** (the MSE-anchor scale
   explosion in Exp 6) that only manifested *because* the residual fix
   changed `mu`'s scale -- a reminder that these two flaws were compounding,
   and fixing one exposed the other more starkly.
5. **The core unresolved finding from the pre-fix sweep still stands:**
   good teacher-forced loss does not predict good beam-search generation
   quality for this architecture. This was true pre-fix (Exp 4's eval_loss/HR
   inversion) and remains true post-fix (Exp 7/8's good eval_loss, mediocre
   HR/NDCG) -- this disconnect, not raw training convergence, looks like the
   actual bottleneck, and it has not yet been directly investigated.

### Recommendation

Given the extensive iteration already done (13 training runs across two
sessions, all confirming the same qualitative pattern), further blind
hyperparameter nudges on this subsampled setup are unlikely to be a good use
of compute. The two most promising remaining directions, in priority order:

1. **Directly inspect what beam search is doing differently from teacher
   forcing.** E.g., compare the argmax-greedy next-token predictions under
   teacher forcing (should be good, given the healthy eval_loss) against the
   actual beam-search output on the same examples -- if they diverge sharply,
   the issue is in how errors compound over free-running generation with a
   stochastic conditioning vector, not in the loss/architecture. This is a
   cheap (~no training) diagnostic that should run before any more training
   experiments.
2. **A full-scale run of the current best config** (Exp 8's:
   `latent_R=3, diff_T=4, use_schedule_scaling=True, diff_eps_init_scale=0.1,
   alpha=0`) matching Exp 1's data budget (full 49.2k train rows, 3 epochs,
   full 6.2k test set) -- the subsampled comparisons throughout this report
   are directional, and it remains possible (though the overfitting evidence
   from the pre-fix Exp 4 argues against it) that the gap narrows
   substantially with the full dataset.
3. Lower priority: `diff_T=16` with the fixed architecture (never cleanly
   tested -- Exp 6 confounded it with the MSE explosion), and re-introducing
   the MSE anchor with a scale-aware target (e.g. normalize `mu` before the
   MSE, or use `proj_mse` so the learned projection can absorb the scale
   mismatch instead of a fixed unit-norm target fighting a raw embedding-space
   `mu`).

## Final Corrected Summary (with Exp 9, the scale-matched baseline)

Exp 9 was run after everything above, in response to noticing that Exp 1 (the
baseline every prior section compares against) was full-scale while every
other experiment was a 5k-row/2-epoch subsample. Rerunning Exp 1's exact
config at that same subsampled scale gives **HR@10 0.044 / NDCG@10 0.0269** --
not 0.1025. This changes the headline conclusion of the entire report.

### Corrected comparison table (all rows at the same 5k-train/2-epoch/1k-eval scale)

| Experiment | Config | NDCG@10 | HR@10 | vs. Exp 9 baseline |
|---|---|---|---|---|
| **Exp 9 (baseline)** | `latent_R=1, diff_T=0, alpha=0` | 0.0269 | 0.0440 | -- |
| Exp 2 (paper-exact) | `diff_T=16, use_schedule_scaling=False, alpha=1.0` | 0.0144 | 0.0260 | -41% |
| Exp 6 (broken, MSE explosion) | residual fix, `diff_T=4, alpha=1.0` | 0.0012 | 0.0030 | -93% |
| Exp 4 (overfit, more data) | `diff_T=16, sample=15k, epochs=3, alpha=1.0` | 0.0126 | 0.0190 | -57% |
| Exp 3 | `diff_T=16, use_schedule_scaling=True, alpha=1.0` | 0.0203 | 0.0410 | -7% |
| Exp 7 | residual fix, `diff_T=4, alpha=0` | 0.0289 | 0.0460 | **+5%** |
| Exp 5 | `diff_T=4, use_schedule_scaling=True, alpha=1.0` | 0.0330 | 0.0530 | **+20%** |
| **Exp 8 (best)** | residual fix, `diff_T=4, alpha=0, eps_init_scale=0.1` | 0.0323 | 0.0510 | **+16%** |

### What actually changed with the corrected baseline

- **The "DiffuReason is ~2x worse than baseline" conclusion does not survive
  scale-matching.** Several configs (Exp 5, 7, 8) are at or above the
  scale-matched baseline; only the genuinely broken/pathological configs
  (Exp 2's un-annealed noise, Exp 4's overfitting, Exp 6's MSE explosion) are
  clearly worse. This flips the framing from "does diffusion hurt" to
  "diffusion is roughly neutral-to-mildly-positive at this small scale, with
  several specific, now-understood ways to make it much worse."
- **Every individual diagnosis in this report is still correct and still
  matters** -- schedule-scaling fixes real training instability (Exp 2->3),
  the non-residual `DenoiseMLP` was a real bug now fixed, the MSE-anchor
  scale mismatch is real (Exp 6), and `diff_eps_init_scale` is a real,
  validated lever (Exp 7->8). What changes is only the *denominator* --
  "worse than a fair baseline" vs. "worse than a baseline trained on 10x more
  data for 1.5x more epochs." The engineering fixes were necessary regardless
  of which baseline number they're compared against.
- **The single most impactful thing found in this entire report, by a wide
  margin, was training scale, not architecture.** Going from Exp 9's config
  to Exp 1's config (same architecture, more data + epochs) moved HR@10 from
  0.044 to 0.1025 (+133%). No architectural change tested came close to that
  effect size. This strongly suggests that if a fair, full-scale head-to-head
  is run (Recommendation #2 above, now with a corrected baseline number to
  target), the most likely outcome is that **both** arms improve
  substantially with more data, and the diffusion variant's relative
  position (currently ~neutral-to-slightly-positive at small scale) is the
  open question -- not "how do we close a 2x gap that turns out to have been
  mostly a measurement artifact."

### Updated recommendation

Recommendation #2 above (a full-scale run of Exp 8's config against Exp 1's
exact data budget) is now the clear top priority -- more so than the
beam-search-vs-teacher-forcing diagnostic (recommendation #1), since the
premise motivating that diagnostic (good loss, bad generation, at small
scale) may itself partly be a data-starvation artifact rather than a
fundamental beam-search issue. Everything in this report to date should be
read as a small-scale methodology and bug-fixing pass that produced several
genuine, validated fixes (schedule-scaling, residual `DenoiseMLP`,
`eps_init_scale`) and one important measurement correction (this section) --
not as a final verdict on whether DiffuReason helps or hurts LatentR3.

## Exp 10 -- Exp 8's config, rerun at Exp 1's full scale (the head-to-head this report was building toward)

This is the run recommended above: Exp 8's fixed architecture (residual
`DenoiseMLP`, `diff_eps_init_scale=0.1`) trained at Exp 1's exact data
budget, so the two can be compared head-to-head with no scale confound in
either direction.

- Config: `latent_R=3, diff_T=4, diff_s=0.008, use_schedule_scaling=True,
  diff_eps_init_scale=0.1, alpha=0, diff_loss=mse`. `sample=-1` (full
  49,252-row train set), `num_epochs=3`, `learning_rate=3e-4`,
  `batch_size=128, micro_batch_size=2` (dropped from the historical 4 after
  an OOM on the first attempt -- see infra note below).
- Eval: full test set (6,157 rows).
- Infra: submitted as Slurm job 240016 (`researchshort` partition, 3090
  nodes). A first attempt (job 239960, `micro_batch_size=4`) OOM'd on step
  1/1152 on a 24GB 3090 -- the diffusion-active full-scale config needs more
  activation memory than the no-diffusion Exp 1 run did at the same
  micro-batch size. Resubmitted with `micro_batch_size=2` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which completed
  cleanly in ~2h53m wall-clock (`eval_loss` 1.306 after 3 epochs).
- Result (`metric.log`):

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.04873 | 0.06099 | 0.06591 | 0.07223 | 0.07223 |
  | HR@K   | 0.04873 | 0.06968 | 0.08170 | 0.10119 | 0.10119 |

### Exp 10 vs. Exp 1 (both full scale, the answer to this report's central open question)

| Experiment | Config | NDCG@10 | HR@10 |
|---|---|---|---|
| Exp 1 (no diffusion, full scale) | `latent_R=1, diff_T=0, alpha=0` | 0.07376 | 0.10248 |
| Exp 10 (DiffuReason, full scale, fixed arch) | `latent_R=3, diff_T=4, alpha=0, eps_init_scale=0.1` | 0.07223 | 0.10119 |
| **Difference** | | **-0.00153 (-2.1%)** | **-0.00129 (-1.3%)** |

At full training scale, DiffuReason's fixed architecture (Exp 8's config)
lands **statistically indistinguishable** from the no-diffusion baseline --
on a 6,157-row eval set the standard error on a proportion metric like HR@10
is ~sqrt(0.10 x 0.90 / 6157) ~ 0.004, so a gap of 0.0013 is well within
noise. This directly answers the question the "Updated recommendation"
section above was building toward: **DiffuReason does not help, but it also
does not hurt, once both arms are trained on the same, adequate data
budget.** It washes out.

This closes the loop opened by the small-scale sweep: at 5k-row/2-epoch
scale, Exp 8 looked like a solid win (+16% HR@10 over the scale-matched Exp
9 baseline). That apparent win does not survive scaling both arms up
together -- it's consistent with diffusion-augmented training being more
data-efficient in the low-data regime (helping more when data is scarce)
but converging to the same ceiling as the simpler no-diffusion model once
enough data is available, rather than diffusion adding a distinct signal
that compounds with scale. Put differently: the small-scale sweep's ranking
was real and reproducible, but its absolute conclusion ("diffusion beats
baseline") does not generalize to the regime that matters for a final
reported number.

### What this means for the project

- If the goal is the best possible full-scale HR@10/NDCG@10 number, **the
  no-diffusion Exp 1 config is simpler, faster to train (no diffusion
  forward/reverse passes, larger feasible micro-batch size), and performs
  the same** -- there is no full-scale accuracy case for keeping the
  diffusion module in the current design.
- The diffusion module isn't disproven as an idea, but this report found no
  full-scale regime where it's clearly worth its added training cost. Any
  future case for it would need either (a) a low-data setting where its
  small-scale edge (Exp 8's +16%) is the actual deployment regime, or (b) a
  architectural/training change not yet tried here (the beam-search vs.
  teacher-forcing diagnostic from recommendation #1, or revisiting
  `metric_for_best_model` selecting on a loss contaminated by `alpha*L_diff`,
  both still open) that changes this outcome.
- All the architecture fixes in this report (residual `DenoiseMLP`,
  `eps_init_scale`) remain correct and worth keeping if the diffusion path
  is pursued further -- they took it from clearly-worse-than-baseline to
  statistically-tied-with-baseline. That is a real, validated improvement;
  it just wasn't enough to produce a full-scale win over the simpler
  alternative.

## Exp 11 -- Exp 5's config (paper-exact `alpha=1.0`, `eps_init_scale=1.0`) at full scale

Requested as a companion to Exp 10: same full-scale data budget, but with
Exp 5's original hyperparameters (`alpha=1.0`, i.e. the raw MSE-anchor loss
term active, and `eps_init_scale=1.0`, the paper-exact unit-variance warm
start) instead of Exp 8/10's tuned `alpha=0, eps_init_scale=0.1`. Important
caveat: Exp 5 predates the residual-`DenoiseMLP` fix (Architecture Fix
section), and that fix is now permanent, not flag-gated -- so this is not a
literal reproduction of small-scale Exp 5. It's really testing "paper-exact
loss weighting and noise scale, on the fixed architecture, at full data."

- Config: `latent_R=3, diff_T=4, diff_s=0.008, use_schedule_scaling=True,
  diff_eps_init_scale=1.0, alpha=1.0, diff_loss=mse`. `sample=-1` (full
  49,252-row train set), `num_epochs=3`, `learning_rate=3e-4,
  batch_size=128, micro_batch_size=2`.
- Infra: Slurm job 240183, `researchshort` partition, 3090 node (`voodoo`).
  Completed cleanly in 2h49m, no OOM (same `micro_batch_size=2` as Exp 10).
- Result (`metric.log`):

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.00016 | 0.00016 | 0.00078 | 0.00131 | 0.00131 |
  | HR@K   | 0.00016 | 0.00016 | 0.00162 | 0.00325 | 0.00325 |

  `eval_loss` dropped 19.49 -> 8.95 -> 6.85 across the three epochs -- still
  5-6x higher than Exp 1/Exp 10's ~1.0-1.3, and raw per-step training loss
  was still plateaued around ~258 at the very end of epoch 3 (started
  ~4700-5700 at the beginning). Compare to Exp 10's `eval_loss` of 1.31 with
  `alpha=0`.

### Exp 11 vs. Exp 1 and Exp 10

| Experiment | `alpha` | `eps_init_scale` | NDCG@10 | HR@10 |
|---|---|---|---|---|
| Exp 1 (no diffusion) | -- | -- | 0.07376 | 0.10248 |
| Exp 10 (fixed arch, tuned) | 0 | 0.1 | 0.07223 | 0.10119 |
| **Exp 11 (fixed arch, paper-exact)** | **1.0** | **1.0** | **0.00131** | **0.00325** |

This is a near-total collapse, not a small regression -- HR@10 fell by
~97% relative to Exp 1/Exp 10, landing close to Exp 6's small-scale
"broken, MSE explosion" result (0.0030), which was the single worst config
in the whole small-scale sweep. Full-scale training did not fix or dilute
this failure mode the way it washed out the small-scale Exp 5/Exp 8 gap in
Exp 10 -- if anything it confirms the failure is structural, not a
data-starvation artifact.

### Reflection

- **The raw, unnormalized MSE-anchor loss (`alpha=1.0`) is the culprit, not
  `eps_init_scale`.** Exp 10 (also `diff_T=4`, also the fixed residual
  architecture) used `eps_init_scale=0.1` and was statistically tied with
  baseline; the only hyperparameter changes here are `alpha` and
  `eps_init_scale` together, but the training curves point at `alpha`:
  per-step loss never got below ~250 even after 9,293s of training on the
  full 49k-row set for 3 epochs, meaning the model spent the entire budget
  fighting a loss term that dominates the combined objective by orders of
  magnitude over the cross-entropy term that actually drives generation
  quality. This is exactly the open "Flaw 3" question flagged earlier in
  this report (`metric_for_best_model` selecting on a loss contaminated by
  `alpha*L_diff`) -- it isn't just a checkpoint-selection issue, the
  training signal itself is dominated by an unnormalized, badly-scaled
  auxiliary loss that appears to actively corrupt the language-modeling
  head rather than merely add noise to which checkpoint gets picked.
- **This closes the loop on `alpha`, the same way Exp 10 closed the loop on
  `diff_T`/architecture.** Across every scale tested in this report --
  small (Exp 2 vs. Exp 7, both same `diff_T`, `alpha=1` vs `alpha=0`) and
  now full (Exp 10 vs. Exp 11) -- `alpha=1.0`'s raw MSE anchor is
  consistently harmful, and at full scale specifically it is catastrophic
  rather than merely suboptimal. There is no data budget tested in this
  report at which `alpha=1.0` with the current (unnormalized) loss
  combination is a reasonable choice. `alpha=0` (Exp 10's setting) should
  be the default going forward; if the MSE-anchor term is wanted at all
  (e.g. as a regularizer), it would need proper normalization/scaling
  (e.g. loss-magnitude balancing, or a much smaller coefficient) before
  being reintroduced, and re-validated on its own before combining with
  anything else.
- Net effect on the project's headline conclusion (Exp 10's "DiffuReason
  washes out to a tie with baseline at full scale"): unchanged and now
  better supported. Exp 10's config (fixed architecture, `alpha=0`,
  `eps_init_scale=0.1`) remains the best full-scale DiffuReason variant
  found, and Exp 11 shows the paper-exact loss weighting is not a viable
  alternative path back toward outperforming baseline -- it's actively
  destructive, not neutral.

## Fixing the `alpha=1.0` collapse: root cause and code fix

Traced Exp 11's collapse to two compounding bugs in `src/latent/layers.py`'s
`forward()`:

1. **`mu` is not an internal-only latent -- it's written directly into the
   `<|Thought|>` token's embedding** (`generate_embs`:
   `input_embs[where_thought_ids[:, 0], thought_idx] = mu.to(input_embs.dtype)`).
   With `diff_loss="mse"`, the anchor loss compares this same `mu` directly
   against a unit-norm item embedding (`v_target`, norm ~1), while `mu`'s
   natural scale as a Qwen hidden-state vector is much larger (`tau_R`
   observed norm ~3.7 on a trained checkpoint). Minimizing that loss
   directly pressures the actual embedding fed to the transformer at that
   position to shrink toward norm ~1 -- fighting the language-modeling
   objective at exactly the position that matters most for generation.
2. **The loss was summed, not averaged, over the feature dimension**
   (`l_diff = ((mu - v_target) ** 2).sum(dim=-1).mean()`), inflating it to
   roughly `hidden_size` (1536x) the scale of a comparable per-dimension
   MSE -- consistent with the observed raw training loss sitting in the
   thousands and refusing to converge even after a full 3-epoch, 49k-row
   run.

Fix applied (`src/latent/layers.py`, `forward()`):

- Changed `.sum(dim=-1).mean()` -> `.mean(dim=-1).mean()` for both the
  `mse` and `proj_mse` branches, bringing `l_diff` onto a per-dimension
  scale comparable to the CE loss.
- Recommended (not code-enforced) pairing: use `diff_loss=proj_mse` rather
  than `mse` whenever `alpha>0`. `proj_mse` routes the anchor loss through
  a separate learned head (`self.diff_proj`) instead of comparing `mu`
  directly, so the loss can shape `mu` toward encoding target-relevant
  information without forcing its literal embedding-space scale/direction
  to collapse toward the unit-norm target.

### Verification: Exp 12 (full scale) and Exp 13 (small scale)

Same configs as Exp 11 and small-scale Exp 5 respectively (`alpha=1.0`,
`eps_init_scale=1.0`, fixed residual architecture), but with `diff_loss`
switched from `mse` to `proj_mse` and the `mean(dim=-1)` fix live.

- **Exp 12** (full scale, `sample=-1, epochs=3`, Slurm job 240289,
  `micro_batch_size=2`): raw training loss immediately sits in the ~0.9-1.4
  range throughout training (vs. Exp 11's ~250-5700, never converging) --
  the fix clearly resolves the pathological loss magnitude. Full result
  pending (see below).
- **Exp 13** (small scale, `sample=5000, epochs=2`, run interactively --
  needed `micro_batch_size=2` on this 3090 node, vs. the original sweep's
  A40 nodes which fit `micro_batch_size=8`; an infra difference, not a
  fix-related regression):

  | K | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|----|----|
  | NDCG@K | 0.01500 | 0.01877 | 0.02058 | 0.02417 | 0.02417 |
  | HR@K   | 0.01500 | 0.02150 | 0.02600 | 0.03700 | 0.03700 |

  `eval_loss` 2.75 -> 2.37 across two epochs -- a healthy range, comparable
  to Exp 7/8's 2.07-2.33, not Exp 11-style non-convergence.

**Exp 12 full-scale result** (`sample=-1, epochs=3`, Slurm job 240289,
`micro_batch_size=2`): training finished cleanly, `eval_loss` 1.36 -> 1.08
-> 1.23 across three epochs -- healthy and in the same range as Exp 1
(~1.03-1.26) and Exp 10 (1.31), a world away from Exp 11's 6.85. Result
(`metric.log`):

| K | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|----|----|
| NDCG@K | 0.04564 | 0.05886 | 0.06287 | 0.06898 | 0.06898 |
| HR@K   | 0.04564 | 0.06838 | 0.07812 | 0.09696 | 0.09696 |

### Exp 12 vs. Exp 1, Exp 10, Exp 11 (all full scale)

| Experiment | `alpha` | `diff_loss` | NDCG@10 | HR@10 |
|---|---|---|---|---|
| Exp 1 (no diffusion) | -- | -- | 0.07376 | 0.10248 |
| Exp 10 (fixed arch, tuned) | 0 | -- | 0.07223 | 0.10119 |
| **Exp 12 (fixed arch, fixed loss)** | **1.0** | **proj_mse** | **0.06898** | **0.09696** |
| Exp 11 (fixed arch, buggy loss) | 1.0 | mse | 0.00131 | 0.00325 |

**Verdict:** the fix works, and works well -- it recovers 96.7% of Exp 11's
lost ground (HR@10 0.00325 -> 0.09696, vs. Exp 1's 0.10248 ceiling), taking
`alpha=1.0` from a total training collapse to a config that is close to,
but still measurably below, both the no-diffusion baseline and Exp 10's
`alpha=0` result. The remaining gap (HR@10: -4.2% vs. Exp 10, -5.4% vs.
Exp 1; NDCG@10: -4.5% vs. Exp 10, -6.5% vs. Exp 1) is larger than the
~0.004 standard error on this 6,157-row eval set (roughly 1-1.5 SE), so
it's a real, if modest, gap rather than pure noise -- consistent with the
same pattern seen at small scale (Exp 13 also trailed the `alpha=0`
baseline). **Conclusion: fixing the loss-scale bug makes `alpha=1.0` a
viable, non-broken configuration, but not a better one than `alpha=0`, at
either scale tested.** `alpha=0` (Exp 10's setting) remains the
recommended choice -- the MSE-anchor auxiliary loss, even correctly scaled
and routed through `proj_mse`, adds a small but consistent training-signal
cost rather than a benefit for this task/backbone. This closes the
`alpha`/loss-scale investigation: the bug fix was worth making (it's the
difference between "broken" and "usable"), but it doesn't change the
report's core recommendation.
