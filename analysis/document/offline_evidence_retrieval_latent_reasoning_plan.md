# Externally Retrieved Evidence with Latent Composition Reasoning

## Status

Design plan only. No model or training code is changed by this document.

## Objective

Replace collaborative InfoNCE's relation-only queue regularization with an
explicit train-only memory whose contents come from **external behavioral and
semantic signals**, not from a pretrained R3 latent. Evidence selection is done
by a separate external retriever. The generated latent is not a retrieval query;
it is a transferable **composition policy** that determines how the already
retrieved evidence is combined and applied through the deployed reasoning path.

This separation is intentional:

```text
external retriever: history -> evidence IDs
latent composer:    history + retrieved evidence -> combination policy
decoder:            own state + composed evidence -> recommendation
```

The external retriever may be frozen and its top-K results cached. Gradients
from R3 must not turn the latent composer back into another evidence retriever.

The central acceptance condition is functional, not geometric:

> Correctly retrieved evidence must improve held-out recommendation relative to
> shuffled, random, latent-only, and disabled-evidence controls.

Higher latent rank, stronger user clustering, external retrieval recall, or
lower composition loss are diagnostics. They are not success criteria by
themselves.

## Motivation

Collaborative InfoNCE stores generated training latents in a FIFO and pulls a
current latent toward same-user or same-target examples. On CDs this produced a
much healthier and more collaboratively structured latent space, but replacing
the correct latent with the train mean or another sample's latent did not hurt
ranking. On Toys, it did not improve ranking and correct, mean, and shuffled
latents were again interchangeable. The loss therefore organizes `r` without
ensuring that its sample-specific content is used by the decoder.

Reusing a pretrained `r_j` as evidence is not sufficient. If the source
checkpoint is collapsed, its extracted bank is a large collection of the same
failure mode; retrieval can rearrange those vectors but cannot introduce the
missing information. The previous null same-user replacement and
collaborative-InfoNCE interventions are direct warnings against treating an R3
latent as evidence merely because it came from another row.

The high-level lesson borrowed from DPPM (arXiv:2610.11776) is to maintain two
complementary routes:

1. a direct evidence path that preserves and exposes relevant historical
   evidence; and
2. a current-reasoning or delta path that captures the query's own state and
   revision.

In DPPM, the memory compiler supplies segment representations and the evidence
path learns how to pool them. The useful analogy for R3 is therefore learned
evidence composition, not latent-query retrieval. This plan does not copy
DPPM's hypernetwork-generated LoRA architecture. An independent retriever
supplies the evidence set, a latent composer learns how to combine that set, and
ordinary R3 reasoning supplies the current path. R3's pretrained latent is
never used as a bank key or value.

## Relationship to existing experiments

The closest existing implementation is behaviour-grounded two-step reasoning
on branch `R3LatentCollaborativeReasoning`:

- `r0` queries other users' training latents;
- donor values contain both a donor latent and its observed next-item embedding;
- a utility teacher supervises the retriever; and
- retrieved evidence is added to `r0` to produce `r1`.

That experiment found a small behaviour-specific gain on CDs (`knn` over
`shuffle_outcome`: +0.0023 HR@10) that did not replicate on Toys. It also had
several limitations that this plan changes:

- training used a changing FIFO, whereas inference used a separately extracted
  full bank;
- donor utility came from an auxiliary item head rather than the deployed
  decoder;
- the evidence scale remained fixed at 0.5 because new parameters near one were
  updated in BF16; and
- no explicit loss required correct evidence to beat shuffled evidence through
  the deployed decoder.

The new experiment should reuse proven leakage checks and controls from that
implementation, but it deliberately removes donor `r_j` from the evidence
content. Existing R3 latents may be retained only as a diagnostic retrieval
baseline.

## Hypothesis

For a history `x_i`, an external method can retrieve a small set of evidence
units whose collaborative, semantic, or verified-rationale content is absent
from a collapsed R3 latent. A generated composition code can then specify how
those evidence units interact, which aspects to emphasize, and how strongly the
result should update the sample's own state. If the decoder is explicitly
trained to prefer the correct evidence and composition over matched
counterfactuals, the latent can represent a reusable reasoning operation rather
than duplicate the retriever.

## External signal hierarchy

Use signals in the following order of preference:

1. **Verified rationale prototypes.** Generate interaction-grounded rationales
   with an external LLM, embed and cluster them, and retain only prototypes that
   improve held-out within-train recommendation. Store the rationale embedding,
   facet/anchor fields, support, and verification score. The existing
   `LLMReasoningPrototype` bank on CDs is the most direct starting point.
2. **External collaborative state.** Frozen SASRec sequence/item embeddings,
   item-kNN/Markov transition distributions, or CF-cluster probabilities. These
   encode cross-user behavior without passing through R3's latent producer.
3. **External semantic state.** Frozen item-title, category, brand, and metadata
   embeddings. These are useful for interpreting a collaborative signal, but a
   title-only bank may be redundant with the decoder's prompt and should not be
   the sole evidence source.
4. **Observed training outcomes.** The donor's next-item embedding or outcome
   distribution is behavioral evidence. It may accompany a rationale or CF
   state, subject to strict temporal and user exclusion.

Start with one external source plus outcomes. Do not combine all four families
until a single-source causal test passes.

Before training R3, screen the proposed external signal for complementarity.
Measure its standalone top-K target coverage, coverage on rows the pinned R3
checkpoint misses, and the union ceiling with R3. Reject an external source that
mostly reproduces R3's hits: being independent of R3's parameters does not imply
that it contains complementary information.

## Textual rationale generation informed by OneReason

Reference: [OneReason Technical Report, Sections 5.4 and 7 and Appendix D.3](https://arxiv.org/html/2606.06260v1).

OneReason constructs traces through Persona Abstraction, Interest Expansion,
and Transition Inference. Historical item references remain machine-matchable.
Its quality dimensions are safety against answer leakage, consistency, logic,
factuality, and informativeness. Its teacher sees the target during construction,
although the trace must exclude target identifiers and target-only evidence.
Section 7 also measures target log-likelihood gain from reasoning; its SFT
traces can hurt prediction despite their structure, while RFT improves this
diagnostic. Thus fluent, structured text alone is insufficient evidence of
usefulness.

The following is our adaptation, rather than a reproduction of their pipeline.

### Generate short evidence-backed hypotheses

For each donor training-history prefix, give an external LLM only the ordered
interactions and available item metadata. Hide the next item from the generator.
An independent verifier may subsequently inspect the observed training outcome.
This reduces answer-conditioned rationalization, although history-only generation
still requires factuality and generalization checks.

1. **Preference abstraction.** Summarize supported preferences such as artist,
   genre, toy theme, brand, format, or product function. Avoid demographic or
   household labels inferred from purchases alone.
2. **Alternative continuations.** Produce one to three distinct hypotheses,
   each citing historical item IDs and a specific relation. Permit abstention
   when evidence is insufficient. Distinguish an observed pattern from an
   inferred motivation.
3. **Comparison.** Explain which continuation is better supported by repetition,
   recency, and the available sequence. State contradictory evidence and the
   conditions under which each alternative should receive less weight.

Do not invent searches, clicks, negative feedback, prices, compatible parts,
ages, or product attributes absent from the inputs. An interaction does not
establish satisfaction, ownership, gift intent, or an unmet need. Preserve event
order, but label a transition as uncertain when purchase-only records cannot
establish a meaningful progression.

For an illustrative CDs prefix containing two studio albums and a recent live
album by the same artist, the generator might produce:

```text
Hypothesis A: artist continuity
Evidence: H1, H2, H3 share an artist.
Implication: another release by this artist is a plausible continuation.

Hypothesis B: recording-format refinement
Evidence: H3 is a live release; H1 and H2 are studio releases.
Implication: live recordings are an alternative direction.
Limitation: one live release is weak evidence of a persistent format preference.

Comparison: artist continuity has repeated support; retain the live-format
alternative at lower confidence until additional evidence supports it.
```

These are hypothetical records, not measured examples from the dataset. For
Toys, use theme/brand continuity or attribute refinement when metadata supports
them; require explicit compatibility metadata before proposing expansions.

### Store rationale units that can be recombined

Store each hypothesis as its own evidence unit, with the donor's comparison as
optional context. A single donor's final judgement may not suit another history;
preserving alternatives lets the current latent composer make its own decision.

```text
RationaleUnit = {
    text,
    preference_facet,
    relation_type,
    antecedent_conditions,
    supporting_history_item_ids,
    supporting_metadata_fields,
    proposed_interest_direction,
    alternative_explanation,
    limitations_and_conflicts,
    source_history_prefix_id,
    source_users_and_max_evidence_time,
    generator_model_and_prompt_hash,
    verification_scores,
}
```

Begin with individual verified units. Cluster only if necessary for scale;
retain source membership and reverify any synthesized prototype. Do not merge
different artists, themes, or transition types into a generic rule that loses
the evidence needed to apply it.

The frozen external retriever matches the current history to the unit's
antecedent conditions. The latent composer receives the retrieved hypotheses,
support, and limitations, then determines their weights and interactions.
Generation and retrieval are performed independently of pretrained R3 latents.

### Generation prompt skeleton

```text
Input: ordered historical item IDs and their supplied metadata.
The next item is hidden.

Return a concise structured record with:
1. Supported preference factors, each citing history IDs and metadata fields.
2. Up to three plausible next-interest hypotheses.
3. For each hypothesis: antecedent condition, supporting IDs, relation,
   recommendation implication, uncertainty, and counterevidence.
4. A brief comparison explaining which hypotheses have stronger support.

Use only supplied facts. Separate observations from inferences. Do not invent
user attributes or behaviors. Do not name an unseen next item. If no informative
relation is supported, return insufficient_evidence rather than force a story.
```

### Verify grounding, transfer, and predictive use separately

- Automatically check referenced IDs, source fields, prefix boundaries, and
  exact evidence provenance before judging prose quality.
- Use an independent LLM pass to check entailment, unsupported intent claims,
  relation validity, informativeness, and target-only details.
- Record consistency with the observed training outcome as a noisy verification
  signal. A plausible alternative is not false merely because one logged next
  item follows another branch.
- Test the rule on disjoint training users satisfying its antecedent. Measure
  coverage and predictive improvement; do not qualify a rule using its own
  donor's next item alone.
- Before scaling latent composition, run a text-evidence pilot against no
  rationale, matched shuffled rationale, and a factual summary of equal length.
  This separates added reasoning value from repeated metadata and extra tokens.
- Then verify that the encoded rationale retains its useful distinctions and
  repeat the evidence/policy interventions through the deployed latent path.
  A failed text pilot motivates revisiting the rationale or reader; a successful
  text pilot with failed latent use points toward the encoding/composition
  interface. Neither comparison alone proves a unique failure cause.

## Evidence unit

For each external evidence unit `j`, store:

```text
Evidence_j = {
    external_key: k_ext_j,
    external_value: v_ext_j,
    evidence_type: rationale | cf_state | transition,
    rationale_text: optional,
    outcome_item_or_distribution: optional,
    quality_score: q_j,
    support: n_j,
    source_user_ids_or_cluster: provenance,
    source_time_range: provenance,
    external_encoder_sha256,
    evidence_builder_sha256,
    train_split_sha256,
}
```

The bank contains no `r_j`. A row-level behavioral value can initially be:

```text
v_ext_j = ExternalValueMLP([
    normalize(cf_state_j);
    normalize(outcome_embedding_j);
    quality_score_j
])
```

For a rationale prototype, use its frozen text embedding plus quality/support
features. The external embedding must be projected into R3 space with an
explicit preservation objective, such as reconstruction or pairwise-distance
distillation, so downstream CE cannot collapse all evidence values together.

Quality filtering should be based on external or held-out verification, not on
the collapsed R3 checkpoint's decoder correctness.

## Stage A: Build and verify the external evidence bank

1. Record the exact training CSV and its SHA-256.
2. Choose one external signal family for the first run.
3. Screen its held-out complementarity to the pinned R3 checkpoint and record
   standalone, R3-miss, and union coverage.
4. Fit or run the external evidence builder on the training split only.
5. Cross-fit row-level CF predictions so a training row is represented by a
   model that did not train on its user or fold.
6. For LLM rationales, generate from training interactions, cluster them, then
   verify canonical prototypes on disjoint training users before admission.
7. Select and freeze an external retrieval rule. Examples are SASRec/CF
   nearest neighbours, transition-rule lookup, or dense/BM25 retrieval over
   verified rationale prototypes.
8. Precompute and cache each training/validation/test history's eligible top-K
   evidence IDs where practical. Cache only IDs and retrieval scores; never
   cache the query target as a retrieval feature.
9. Save external keys, values, evidence type, quality, support, user/time
   provenance, and optional observed outcomes.
10. Save a summary containing shapes, dtypes, hashes, coverage, evidence-type
   counts, verification rates, timestamps, and duplicate statistics.
11. Treat the bank, retriever, and cached candidate sets as immutable throughout
    the first experiment.

The same external bank must be used during stage-B training and held-out
inference. A pinned R3 checkpoint is still needed to initialize the student, but
it does not generate the evidence bank.

### Optional cross-fitting

For row-level external evidence, split training users into folds. Produce each
row's CF state, rationale verification prediction, or evidence score using a
model that did not train on that user's fold. This prevents the external teacher
from becoming an in-sample target lookup and is preferable to a heuristic
correctness gate.

## Stage B: Latent evidence-composition reasoning

### External retrieval path

The external system retrieves evidence without using an R3 latent:

```text
h_ext_i = ExternalHistoryEncoder(history_i)
C_i = ExternalRetrieve(h_ext_i, immutable_bank, eligibility_rules, K)
```

`ExternalRetrieve` may be kNN in a frozen SASRec/CF space, a transition lookup,
or semantic retrieval over verified rationale prototypes. It is trained before
R3, uses only inference-available history, and receives no gradient from the R3
loss. This prevents the generated latent and the retriever from learning the
same function under different names.

Apply metadata eligibility before retrieval:

```text
eligible(i, j) = (u_j != u_i) and (t_j < c_i)
C_i = top-K externally scored eligible evidence units
```

During training, optionally add a few random eligible units to `C_i` so the
composer learns to suppress irrelevant evidence. These units are candidates for
composition only; R3 still does not retrieve from the full bank.

### Own-state path

Ordinary R3 supplies the current sample state:

```text
r_own_i = LatentRATT(x_i)
```

`r_own_i` is not compared with evidence keys and cannot alter evidence IDs. Its
role is to describe the current sample when generating the composition policy.

### Composition-policy path

Let the retrieved set contain fixed external values and metadata:

```text
E_i = {(v_ext_j, type_j, quality_j, retrieval_score_j): j in C_i}
c_i = CompositionCode(r_own_i, SetSummary(E_i))
a_ij, g_ij = ApplyPolicy(c_i, v_ext_j, metadata_j)
r_evidence_i = sum_{j in C_i} a_ij * (g_ij ⊙ ProjectExternal(v_ext_j))
```

Here `c_i` is the generated latent reasoning. It encodes a way to combine
evidence: mixture weights, feature gates, interaction/update coefficients, or a
small sequence of composition steps. `ApplyPolicy` operates only on the fixed
candidate set `C_i`. Reweighting items within that set is part of composition;
searching the global bank is not.

This is closest to DPPM's evidence path: evidence representations are supplied
first, then learned coordinate-wise or token-wise weights combine them. Begin
with one softmax mixture and feature gate. Add multi-step composition only after
the causal evidence test passes.

Observed outcomes may be evidence values, but they must not enter the external
retrieval query unless that feature is also available at inference for every
candidate. The current query target may supervise composition during training,
but may never determine `C_i` at evaluation.

### Dual-path fusion

Recommended first implementation:

```text
direction_i = normalize(EvidenceProjection(r_evidence_i))
r_final_i = r_own_i + alpha * ||r_own_i|| * direction_i
```

`alpha` and all new LayerNorm parameters must remain FP32 even when the backbone
uses BF16. Initialize `alpha` conservatively, but do not initialize the entire
evidence path to an exact zero if that prevents the composition losses from
training it.

An evidence-only ablation (`r_final = r_evidence`) tests whether the system can
operate as discrete retrieval, but is not the default because a retrieval error
would discard all query-specific reasoning.

If residual fusion again makes evidence identity invisible, the next variant
should expose `r_own` and the composed evidence as separate decoder positions.
Do not add this complexity to the first run.

## Functional composition supervision

### Evidence and composition utility

For each externally retrieved evidence unit, compute training-only utility while
holding the history and candidate set fixed:

```text
u_ij = log p_decoder(y_i | x_i, evidence_j)
       - log p_decoder(y_i | x_i, evidence_off)
teacher_ij = softmax(u_ij / tau_utility)
```

The target is legal for constructing a training label for combination weights.
It must not enter the external retrieval query or inference-time candidate
selection. Blend functional utility with the frozen retriever's relevance and
evidence-quality scores:

```text
composition_teacher = normalize_distribution(
    beta * external_relevance
    + gamma * evidence_quality
    + (1 - beta - gamma) * evidence_utility
)
```

Distill `a_i` toward this teacher. This teaches how to use a retrieved set; it
does not train R3 to reproduce the external retriever over the global bank. The
utility term prevents the composer from preserving externally plausible but
decoder-useless evidence.

Full decoder utility is the preferred teacher because it measures the deployed
path. If it is initially too expensive, use the existing item-head utility only
as a smoke-test proxy, then validate the chosen candidates with full decoder
utility before launching the full run.

### Losses

Use:

```text
L = L_recommendation
    + lambda_comp * KL(composition_teacher || a_i)
    + lambda_item * L_evidence_item
    + lambda_cf * L_counterfactual
    + lambda_transfer * L_policy_transfer
```

Set `lambda_transfer = 0` until the optional collaborative queue is enabled.

`L_evidence_item` requires `r_final` to identify the observed next item over the
full catalogue. It keeps the evidence direction semantically tied to behavior.

The counterfactual term directly requires evidence identity and its composition
to matter:

```text
gain_i = log p(y_i | x_i, correct_evidence)
         - log p(y_i | x_i, shuffled_evidence)
L_counterfactual = mean(max(0, margin - gain_i))
```

Use two matched counterfactuals:

1. keep `c_i` but shuffle evidence values among eligible retrieved sets; and
2. keep `E_i` but replace `c_i` with another sample's composition code.

The first tests whether the evidence matters. The second tests whether the
generated way of combining evidence is sample-specific. Match evidence count,
type, support, timestamp range, and vector norm where possible.

Unlike collaborative InfoNCE, no same-user or same-target relation and no
pretrained R3-latent similarity is assumed to define useful evidence or a
useful composition. Those quantities can remain diagnostics only.

## Optional collaborative queue: transfer composition policies

Retain the queue only after the non-queued composer passes the correct-evidence
test. The queue stores detached composition codes and metadata, not evidence
keys or values:

```text
QueueEntry_j = {
    composition_code: stopgrad(c_j),
    external_context_signature,
    evidence_type_signature,
    observed_utility,
    user_id,
    timestamp,
}
```

A queued code must be usable as an operator on the current sample's already
retrieved set:

```text
r_evidence_{j->i} = ApplyPolicy(c_j, E_i)
u_{j->i} = log p(y_i | x_i, r_evidence_{j->i})
           - log p(y_i | x_i, evidence_off)
```

Externally similar histories or compatible evidence-type signatures define the
candidate peer policies. Among those candidates, positive policies are codes
with positive transfer utility on `E_i`; matched negative policies have low or
negative utility. `L_policy_transfer` can distill the current policy toward a
useful queued policy or enforce a margin over a negative transferred policy.

This preserves the collaborative idea without asking a latent to perform
retrieval twice. The decisive queue ablation is functional: a correctly matched
peer policy must beat a shuffled peer policy when both operate on the same
`E_i`. Plain InfoNCE between composition codes is only an auxiliary diagnostic,
because it can improve geometry without changing decoder behavior.

## Training schedule

### Phase 1: frozen external bank and student initialization

- Load the pinned R3 checkpoint only as the student initialization.
- Freeze the evidence bank.
- Freeze the external retriever, history encoder, and evidence encoder.
- Load or compute fixed top-K evidence IDs; do not train a latent query.
- Freeze the backbone and decoder initially.
- Train the composition-code generator, within-set policy, evidence projection,
  item head, and fusion scale.
- Distill composition weights from external relevance/quality before relying on
  target-utility supervision.
- Keep all new normalization and scalar parameters in FP32.
- Verify on a small sample that every new module receives nonzero finite
  gradients and that correct/shuffled evidence and correct/shuffled policies
  produce different decoder scores.

### Phase 2: limited joint adaptation

- Unfreeze the latent producer and a small decoder adapter or the final decoder
  layers; avoid immediate full-model fine-tuning.
- Keep the frozen source bank fixed.
- Keep the external retriever fixed and continue all auxiliary losses so the
  composer does not drift toward an arbitrary CE shortcut.
- Select the checkpoint by held-out recommendation loss plus the causal evidence
  gate, not training retrieval accuracy alone.

### Phase 3: full comparison only after the gate passes

Run the complete held-out ranking suite. Add the collaborative composition queue
only after the non-queued system proves that evidence identity and composition
identity matter. Do not add multi-hop composition or multiple policy heads until
the single-step system passes.

## Leakage and provenance rules

For every query/evidence pair:

- exclude the query row;
- exclude all donors from the same user in the initial experiment;
- require donor outcome time to be strictly earlier than the query's latest
  observed history time;
- never add validation or test outcomes to the bank;
- never use the query target in retrieval features or inference metadata;
- apply the same eligibility rule to random and shuffled controls;
- validate that saved bank row order matches metadata and the training CSV;
- record student-checkpoint, external-encoder, evidence-builder, bank,
  data-split, and item-embedding hashes in every model and result artifact.

If timestamps are unavailable or ambiguous for a dataset, fail closed rather
than retrieving from the unrestricted bank.

## Controls

Evaluate all controls on the same trained checkpoint:

| Control | Meaning |
|---|---|
| `evidence` | Fixed external retrieval plus learned latent composition |
| `off` | Use `r_own` without evidence |
| `shuffle_outcome` | Same selected donors, permuted outcomes |
| `shuffle_policy` | Same externally retrieved evidence, another sample's composition code |
| `uniform_policy` | Same externally retrieved evidence, uniform combination |
| `external_rank` | Fixed external retrieval-score weights without a generated composition code |
| `external_only` | External evidence without observed-outcome features |
| `outcome_only` | Outcome features without rationale/CF evidence |
| `random_evidence` | Random eligible evidence with its matched values |
| `oracle_utility` | Training/diagnostic upper bound selected with the target; never an inference result |
| `collapsed_latent_bank` | Old `r_j` bank as a diagnostic baseline, never the proposed method |

Also retain the existing `raw`, `mask`, `mean`, and shuffled-latent interventions
to determine whether evidence makes the final reasoning sample-specific.

## Diagnostics

Log at least:

- recommendation CE;
- composition KL and teacher entropy;
- correct-versus-shuffled target log-probability gap;
- correct-policy-versus-shuffled-policy target log-probability gap;
- full-decoder utility of top-1, top-K mean, and best explored donor;
- evidence item NLL and hit@10 before/after fusion;
- external retriever recall of the highest-utility explored evidence unit;
- composition-weight entropy and per-evidence-type weight;
- evidence scale and delivered evidence/base-latent norm ratio;
- attention entropy and eligible-donor coverage;
- fraction of samples whose highest-weight evidence changes between checkpoints;
- external-bank representation drift, which should be zero;
- overlap and complementarity of the fixed external retriever with R3;
- evidence-projection reconstruction or distance-preservation loss;
- queued-policy transfer utility, when the optional queue is enabled;
- FP32 parameter deltas for the evidence scale and LayerNorm parameters.

Geometry metrics remain useful for diagnosis but are secondary: effective rank,
random/same-user/same-target cosine, exact-template frequency, and user/target
adjusted eta-squared.

## Acceptance criteria

### Smoke gate

- Bank hashes and row alignment validate.
- No query retrieves itself, a same-user donor, or a future outcome.
- All new modules receive finite nonzero gradients.
- FP32 evidence parameters move after optimizer steps.
- The retrieved path changes decoder output.
- Correct and shuffled evidence produce measurably different target scores.
- Correct and shuffled composition policies produce measurably different target
  scores on the same retrieved evidence.

### CDs full gate

On the complete CDs test split:

1. `evidence` must beat `shuffle_outcome` on HR@10 and NDCG@10 with paired 95%
   confidence intervals excluding zero.
2. `evidence` must beat `shuffle_policy` and `uniform_policy`, establishing that
   the generated composition policy matters.
3. `evidence` must beat `off`; this is weaker than criteria 1 and 2 but still
   required.
4. `evidence` should beat both `external_only` and `outcome_only`, establishing
   that their combination adds more than either external component alone.
5. `evidence` should beat `random_evidence`, establishing that the external
   retriever matters.
6. Standard ranking must at least preserve the pinned plain-R3 baseline.
7. Correct evidence must outperform mean/shuffled reasoning interventions.

Criteria 1 and 2 are the decisive causal gates. If either fails, do not
interpret improved geometry, retrieval recall, or composition accuracy as
success.

### Toys replication gate

Run Toys only after CDs passes. Use the same mechanism and predeclared
hyperparameters except for dataset-dependent bank sizes. Require the same
correct-versus-shuffled evidence and correct-versus-shuffled-policy results. A
gain over `off` without both causal gains repeats the previous weak Toys result
and does not count as replication.

## Ablation order

Run one change at a time:

1. Frozen external bank/retriever + uniform evidence composition.
2. Add the latent composition code and distill external relevance/quality.
3. Add the existing item-head utility teacher to validate functional
   composition.
4. Replace the proxy with full-decoder utility.
5. Add evidence-shuffle and policy-shuffle counterfactual margins.
6. Limited decoder adaptation.
7. Add the collaborative composition-policy queue.
8. Expose composed evidence as a separate decoder position if residual fusion
   is ignored.
9. Multi-step or multi-aspect composition only after single-step causal use
   passes.

Do not combine clustering, multiple composition heads, a learned verifier, and
a new decoder architecture in the first run; attribution would be impossible.

## Proposed repository layout

```text
src/latent/external_evidence_composer.py
src/latent/latent_attention_train_evidence_composer.py
src/latent/external_evidence_dataset.py
scripts/build_external_evidence_bank.py
scripts/cache_external_evidence_retrieval.py
scripts/evidence_composer_cds_full.sbatch
scripts/evidence_composer_toys_full.sbatch
tests/test_external_evidence_composer.py
analysis/document/offline_evidence_retrieval_latent_reasoning_plan.md
```

Prefer reusing eligibility, metadata, controls, checkpoint integration, and bank
round-trip logic from `outcome_memory.py` and the grounded-reasoning branch.

## Implementation milestones

1. **External bank schema and builder**
   - deterministic external-signal construction;
   - cross-fitting or held-out verification;
   - provenance hashes;
   - alignment validation;
   - train-only artifact.
2. **Frozen external retriever and dual-path composer**
   - eligible external top-K plus exploration;
   - cached candidate IDs and immutable retriever hashes;
   - latent composition code and within-set weighting/gating;
   - FP32 evidence modules;
   - residual fusion.
3. **Functional losses**
   - composition-utility teacher;
   - evidence item loss;
   - evidence-shuffle and policy-shuffle counterfactual margins.
4. **Controls and evaluation**
   - evidence/off/evidence-shuffle/policy-shuffle/uniform/random;
   - paired bootstrap;
   - raw/mask/mean/shuffled latent interventions.
5. **CDs smoke and full run**
   - pass all smoke gates;
   - run full ranking;
   - stop if the causal evidence gate fails.
6. **Collaborative policy queue**
   - store transferable composition codes only;
   - evaluate matched versus shuffled policy transfer on fixed evidence sets;
   - keep only if it improves the non-queued composer.
7. **Toys replication**
   - freeze the CDs-selected recipe;
   - rebuild only the dataset-specific bank;
   - evaluate the predeclared replication gate.

## Decision rule

Continue this research line only if externally grounded evidence beats matched
shuffled evidence and the learned policy beats shuffled/uniform composition on
held-out ranking. If the frozen external bank/retriever and functionally trained
composer still fail those tests, the bottleneck is no longer collapsed-bank
reuse, queue staleness, or relation quality. It is the decoder's ability to
consume composed latent evidence, and the next experiment should change the
evidence interface rather than add another retriever or geometry loss.
