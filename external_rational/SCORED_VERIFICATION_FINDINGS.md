# Scored verification results on the saved Toys pilot

The new verifier was tested against the same ten valid rationale units from
`results/toys_pilot_v2`. No rationale was regenerated and no previous generation,
verification, or assistant-audit artifact was modified. Four histories whose
generation failed citation checks remained excluded.

## Implementation and checks

- Per-claim, per-event evidence checks with exact metadata quote validation.
- Nine scored dimensions, preserving raw and program-capped scores.
- Program-controlled admission requiring factuality, evidence selection, and
  causal logic each at least 4, with no unsupported central claim or violation.
- Program rejection for unsupported refinement/transition over tied timestamps.
- Separate judge endpoint/model arguments, resumable results, and preserved retries.
- Targets, previous verification decisions, and assistant-audit labels withheld
  from judge requests.
- All 27 focused tests passed, including rejection despite perfect raw scores,
  complete claim/event coverage, source preservation, and caching/correction paths.

The thresholds are an experimental policy for this implementation. They are not
a filtering threshold established by the referenced paper.

## Live runs

Both runs used Qwen/Qwen3-8B at `http://album:8000/v1`, temperature 0, a 6,000-token
output limit, and two concurrent requests. Both finished with valid judgments
for all ten units and no unresolved API/schema errors.

| Run | Accepted | Rejected | Main finding |
|---|---:|---:|---|
| `toys_scored_verification_v1` | 0 | 10 | Judge frequently demanded direct proof of inferred preferences and invented conflicts from positive ratings |
| `toys_scored_verification_v2` | 3 | 7 | Clearer inference standards reduced blanket rejection, but semantic errors remained |

Between runs, the core three admission thresholds stayed fixed. The prompt
clarified the difference between factual entailment and support for a qualified
hypothesis. Schema validation was strengthened to disallow rating-conflict flags
without a cited actual low rating (1 or 2), and duplicate-inflation flags without
actual duplicate records. Invalid judgment structures triggered corrective
requests; they were not silently repaired.

These runs are a prompt/implementation pilot on ten examples, not an estimate
of verifier precision, recall, or recommendation performance.

## Remaining failures in the calibrated run

### A false acceptance remains

`Toys_and_Games:train:35712:R1` claims shared portability/versatility across a
carrying play mat, toy laptop, and toy vacuum. For the vacuum, the judge cites:

> 'Twist & turn' action that's just like the real thing

That quote exists, but does not establish portability between environments.
The judge nevertheless labeled the observation supported and assigned factuality
5. Because the semantic verdict and scores were positive and no violation was
flagged, the policy admitted it. The exact decision and reasoning are saved in
`results/toys_scored_verification_v2/judgments/000002.json`.

This demonstrates the boundary of program checks: they enforce a supplied
judgment consistently but cannot make a wrong entailment judgment correct.
The three entries in `accepted_rationales.jsonl` must not be treated as a trusted
external evidence bank.

### Questionable rejections also remain

The Star Wars continuity unit (`train:23344:R1`) was rejected after an initial
attempt cited nonexistent category quotations for a theme already supported by
brand/title information. The corrected judgment continued to focus on the
missing category quotation instead of reassessing available evidence.

The Munchkin unit (`train:25609:R1`) was rejected partly for inflated duplicate
support, which is a legitimate concern, but the judge also demanded direct proof
of user preference even though the output is intended as a qualified hypothesis.
Two distinct products still provide a possible continuity signal after duplicate
records are accounted for. Neither rejection establishes that the respective
rationale would be predictively useful or useless.

### Some hard failures are handled correctly

The Peppa refinement and Minecraft collectibility-progression units were
rejected. Their cited events do not provide distinct timestamps, so the program's
temporal constraint applies regardless of positive model scores. The Minecraft
history also contains a cited 2/5 rating that conflicts with an unqualified
positive preference explanation.

## Interpretation

The scored verification machinery is implemented and exercised end to end.
The current Qwen3-8B judge still has false acceptances and questionable
rejections. A lower acceptance rate is not evidence of a better verifier.

Use the separate judge configuration to evaluate a stronger or independent model
against the saved pilot and audit before scaling automatic admission. Keep the
history-only generation policy and retain claim-level artifacts for inspection.
Predictive and cross-user-transfer tests remain separate requirements.

See [`SCORED_VERIFICATION.md`](SCORED_VERIFICATION.md) for commands and policy
details; [`results/toys_scored_verification_v2/review.md`](results/toys_scored_verification_v2/review.md)
for all final decisions.
