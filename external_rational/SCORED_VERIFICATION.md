# Claim-level scored verification

Implementation: [`scored_verification.py`](scored_verification.py).

Live pilot findings and known remaining judge errors:
[`SCORED_VERIFICATION_FINDINGS.md`](SCORED_VERIFICATION_FINDINGS.md).

This adapts the quality evaluation in
[WhisperRec, Appendix D](https://arxiv.org/pdf/2607.26621#page=11) to the existing
Toys rationale units. It evaluates saved generations, leaving earlier inputs,
generation responses, and verification decisions intact. Invalid source
generations remain excluded; this command does not regenerate them.

## Run on the saved Toys pilot

```bash
python external_rational/scored_verification.py \
  --source-dir external_rational/results/toys_pilot_v2 \
  --output-dir external_rational/results/toys_scored_verification_v2 \
  --concurrency 2
```

The default endpoint/model is `http://album:8000/v1`, `Qwen/Qwen3-8B`.
Use `--base-url` and `--model` for a different judge. This makes a stronger,
separate judge possible; the default remains the same model as the generator.
No paid/external model is selected automatically. `--dry-run` validates sources
and prepares the job/config artifacts without sending requests. Repeating the
same command resumes successes and retries errors, preserving prior attempts.
Use a new output directory when prompts, code, inputs, model, or decoding settings
change. Do not run concurrent writers against the same output directory.

## What the judge receives

Only the history, one rationale, its claim list, and deterministic history
diagnostics enter the request. The next item, earlier judge decisions, and saved
assistant-audit decisions do not enter the prompt.

Each observation is assigned an `O` claim ID. Interpretation, continuation, and
standalone text are assigned `I1`, `C1`, and `T1`. Compound observations must be
supported in full. For every observation the judge must assess each cited event
once, rather than allowing a property from one product to justify all products.
The other claims must also identify relevant source events.

For factual observations, support means that the metadata establishes the claim.
For qualified inferences and recommendations, support means the event supplies
a relevant reason for a possible continuation. Direct observation of the user's
mental state is not required. The second prompt revision explicitly separates
these standards after the initial run rejected even qualified hypotheses.

An event check includes a support verdict, exact metadata quotes, and a reason.
The program checks event/field existence and quoted-substring existence.
It does not automatically prove that a quote entails the claim. The judge is
responsible for the semantic assessment, which can still be wrong.

Diagnostics include distinct product/timestamp counts, duplicate groups, and
ratings of cited products. Duplicate records do not create independent support.
The original events are preserved so the model can inspect their context.

## Scores and decisions

The judge returns nine integer scores from 1 to 5 with reasons:

- factuality;
- evidence selection;
- signal strength;
- intent translation;
- causal logic;
- recommendation fit;
- temporal reasoning;
- confidence calibration;
- task fulfillment.

The Toys adaptation replaces advertising fit with recommendation fit and omits
unavailable geographic/fulfillment requirements. It evaluates target-free
Exploration hypotheses rather than candidate evaluation or outcome attribution.

Python calculates the final decision. Acceptance requires factuality, evidence
selection, and causal logic each at least 4, no failed central claim, and no
flagged violation. These thresholds are our initial experimental policy, not a
filter threshold reported by WhisperRec. The other dimensions are diagnostic;
there is no average-score acceptance rule.

Hard checks and caps:

| Condition | Program action |
|---|---|
| Unsupported/contradictory central claim | Reject regardless of other scores |
| Observation passed overall but one cited event fails | Reject |
| Factual observation relabeled as a qualified inference | Reject |
| Fabricated behavior or target backfilling | Cap factuality at 2 and reject |
| Unsupported central product property | Cap factuality at 2 and reject; Toys extension |
| Fabricated context or unsupported need | Cap factuality at 3 and reject |
| Unhandled rating conflict or duplicate inflation flagged | Cap evidence selection at 3 and reject |
| Refinement/transition without two distinct cited products in the supplied order | Program flags unsupported progression and rejects |
| Missing claims, incomplete event coverage, invalid scores/quotes | Retry for correction; exclude if retries fail |

Policy v3 treats supplied interaction order as authoritative chronological order,
as specified for this dataset. Earlier/later and sequence-based transitions may
use that order even when timestamps tie or are absent. Distinct timestamps are
not an admission requirement. Attribute evidence must still support refinement or
transition; order alone does not prove a growing preference or causality. Exact
dates and elapsed durations require their own evidence. Earlier saved reports
used policy v2 and retain their original decisions; rejudge in a new directory
with the updated prompts to evaluate under this rule.

For this implementation, a rating-conflict violation requires an actual cited
rating of 1 or 2 on the dataset's 1–5 scale. An all-positive cited history cannot
be labeled a rating conflict merely because other motivations are possible;
that invalid judgment triggers schema correction. Duplicate-inflation findings
likewise require actual duplicate events. Uncertainty and conflicting evidence
that are handled appropriately need not be flagged as violations.

The judge's raw scores and the program's effective scores are both saved. Caps
are enforced even when the judge assigned a high raw score. Any flagged
violation blocks acceptance. A schema-correction request preserves negative
judgments rather than asking the model to make a rationale pass.

## Artifacts

| File | Contents |
|---|---|
| `config.json` | Judge/decoding settings, policy, code and prompt hashes |
| `jobs.jsonl` | Frozen unit/history snapshots and source-response hashes |
| `source_exclusions.json` | Invalid/missing source generations |
| `judgments/` | Exact requests, raw API responses, retries, scores, computed decisions |
| `all_rationales_review.jsonl` | All validly judged units and failed API/schema jobs |
| `accepted_rationales.jsonl` | Units admitted by the program's policy |
| `summary.json`, `review.md` | Aggregate counts and readable rejection reasons |

The report recomputes decisions from validated judgments rather than trusting a
saved decision field. Prior same-model acceptances are included in the report
for comparison, but are withheld from the new judge.

## Limits

These scores are a grounding screen. They do not establish predictive utility,
cross-user transfer, or a causal benefit from latent reasoning. A more elaborate
prompt also does not make Qwen3-8B an independent or reliable semantic verifier.
Inspect the saved claim checks and compare known pilot failures before scaling.

The focused test suite includes acceptance regressions where all nine raw scores
are 5 but an unsupported claim or a failed event check still forces rejection.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s external_rational -p 'test_*.py' -v
```
