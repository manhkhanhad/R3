# Outcome-grounded latent memory pilot

Goal: test whether other users' observed transitions provide useful evidence
beyond near-duplicate history latents. Ranking improvement is the success
criterion; sensitivity alone is insufficient.

## Representation and integration

For a train row j, keep a frozen history latent r_j, history-derived retrieval
keys, its user identity, outcome timestamp, and the observed next item's title
embedding e(y_j). Given the query's history latent r_i, retrieve up to 32 eligible
donors and compute:

```
m_j = ValueMLP(concat(normalize(r_j), normalize(e(y_j))))
a_j = softmax(Query(r_i) dot Key(r_j))
r_supported = r_i + bounded(sigmoid(Gate(r_i)) * sum_j a_j m_j)
```

The value MLP encodes an observed transition rather than only a similar history.
Donor latents and item embeddings are normalized separately so the large norm
of R3 latents does not overwhelm the outcome input.
The decoder still sees the full query history. The correction starts at zero
and is bounded to 10% of the current latent norm. This is an initial conservative
setting, not a claim that 10% is optimal. The value encoder and attention are
trained through ordinary recommendation CE. The initial pilot disables the
paired margin loss (`pair_weight=0`).

This extends the existing projected-neighbor correction implementation, using
Attempt 10's checkpoint and its frozen 66,500-row train history-latent bank.
Both retrieval projections consume only history-derived latents, including the
head named `target_projection`; neither consumes the query's target item.
A learned target-predictive projection is distinct from supplying that target.
The frozen bank and retrieval heads remain fixed while the query encoder is
fine-tuned, so representation drift is a limitation to monitor.

## Leakage boundaries

- Bank provenance must match the training CSV and source checkpoint hashes.
  Donor item, user, and original row identities must align with extraction.
- Each donor must be from a different user. This excludes the query row and
  overlapping same-user training windows, including later windows exposing its
  target. No row's validation/test outcome is added to the bank.
- Donor outcome time must be strictly less than the latest timestamp already
  observed in the query history. The query's future target timestamp is never
  used to select memories. Equal-time events are conservatively excluded.
- Query time/user metadata is required at training and inference and expands
  with beam search. Missing metadata raises; no eligible donor gives an identity
  correction with finite gradients.
- Random retrieval obeys the same eligibility rule. Shuffled outcomes are
  permuted only among each query's eligible selected donors.

These are memory-access protections under the existing split protocol, not a
claim of a fully chronological experiment: the source checkpoint and retrieval
heads were trained on the full provided training split. A strict temporal model
comparison would also need chronological model fitting. Dataset timestamps are
coarse, so strict same-day exclusion may reduce useful donor coverage.

## Run

From the repository root, on a GPU allocation:

```bash
bash scripts/outcome_memory_musical.sh
```

Or submit the provided launcher:

```bash
sbatch scripts/outcome_memory_musical.sbatch
```

Small GPU smoke run:

```bash
TRAIN_SAMPLE=256 EVAL_SAMPLE=16 TEST_SAMPLE=16 bash scripts/outcome_memory_musical.sh
```

The launcher continues the pinned collaborative checkpoint for one epoch,
selects the best validation-loss checkpoint, and evaluates constrained top-10
recommendations under five conditions. It saves weights-only checkpoints to
limit disk usage. Override `BASE_MODEL`, `BANK_DIR`, `OUTPUT_DIR`,
`MEMORY_VALUE_MODE`, `MICRO_BATCH_SIZE`, `NUM_EPOCHS`, or sample limits via the
environment. The bank must match the selected source checkpoint.

| Control | Interpretation |
|---|---|
| outcome | Relevant donor history latents with their observed outcomes |
| latent_only | Same retrieval, zero outcome inputs |
| shuffle_outcome | Same retrieval, permuted eligible donor outcomes |
| off | Same trained decoder with the memory correction disabled |
| random | Random eligible donors, retaining each donor's outcome |

Outputs: `eval_<control>.json` in the model directory; the launcher prints
HR/NDCG and unknown-title counts through the existing ranking utility.
For independently trained latent-only comparison, use
`MEMORY_VALUE_MODE=latent_only`; its output directory is separate by default.
Inference ablations alone are interventions on an outcome-trained model, not
matched retraining results. Shuffling is ineffective with only one eligible
donor. Zero-initialization means pretraining smoke outputs should match memory-off.

Compare outcome against off, latent-only, random, and the original source
checkpoint. A win over off alone does not prove outcomes help: fine-tuning or
donor history summaries can also contribute. Require full-test ranking gains,
then inspect memory interventions to establish that the gain uses donor outcomes.
Text evidence and output-distribution KL controls remain follow-up experiments.

## Validation

Focused tests cover temporal/user exclusions, target-independent retrieval,
outcome contribution, no-donor gradients, metadata expansion, training gradients,
checkpoint round-trip, cached beam generation, and bank alignment. CPU tests
validate the integration, not recommendation quality. Full GPU training/evaluation
must produce the ranking evidence before concluding that the method helps.

## Two-phase margin → memory experiment

Implemented in `scripts/outcome_memory_two_phase_musical.sh` with the SLURM
wrapper `scripts/outcome_memory_two_phase_musical.sbatch`.

1. Start from Qwen2.5-1.5B-Instruct and train Attempt 2 for exactly two epochs:
   ordinary recommendation CE plus margin 0.2, weight 1.0. Micro-batch defaults
   to 2; size 1 is rejected because batch-shuffling cannot supply a wrong latent.
   Gradient checkpointing is enabled. Early stopping and best-checkpoint
   replacement are disabled, so phase 2 starts from the epoch-2 weights.
2. Measure latent dependence and diversity on validation (2,000 examples by
   default), then extract a fresh full training-split donor bank from those
   exact weights. Keep the bank provenance/hash checks and donor eligibility.
3. Attach the memory correction and train for two more epochs with ordinary CE.
   The margin checkpoint has no collaborative projection heads, so retrieval
   uses centered history-latent cosine rather than untrained projection heads.
   Then run the five ranking controls from the original memory pilot.

Default output root:
`scratchDirectory/R3/output_dir/musical_margin2_memory2/`, with separate
`phase1_margin` and `phase2_memory` directories. Bank:
`analysis/evidence/Musical_Instruments/musical_margin2_memory2/train`.
Existing one-phase results are separate and are not overwritten.

```bash
sbatch scripts/outcome_memory_two_phase_musical.sbatch
```

The job requests up to 48 hours, within the researchshort partition limit, to
cover both training phases, extraction, and the five ranking evaluations.
`RUN_TAG`/`RUN_ROOT` choose a separate experiment namespace. `SKIP_PHASE1=1`
and `SKIP_BANK_EXTRACTION=1` reuse completed artifacts; phase-2 provenance
checks still verify their agreement. The weights-only checkpoints do not
provide an exact optimizer-state resume. `PHASE1_MICRO_BATCH_SIZE` must remain
at least 2; `MICRO_BATCH_SIZE` independently controls phase 2.

Rationale: teach dependence on the user's latent before adding donor evidence.
This is a hypothesis about training order, not a guarantee of avoiding collapse
or retaining dependence once phase 2 returns to CE alone. Phase 1's diagnostic
separately reports geometry and swap sensitivity. Phase 2 now also logs the gate,
evidence norm, base-latent norm, and delivered correction after BF16 rounding,
so a small correction can be traced to its source instead of inferred from one
ratio. Independent phase-1 ranking and a matched extra-CE baseline remain useful
follow-up comparisons for attributing any improvements specifically to memory.

Validation: 16 focused tests passed, including active margin gradients,
phase-1-to-phase-2 checkpoint transfer without collaborative heads, zero-change
initialization, memory gradients, checkpoint reload, and leakage safeguards.
