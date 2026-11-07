# Toys rationale pilot

The pilot was run against `http://album:8000/v1` using `Qwen/Qwen3-8B`.
All code, prompts, source inputs, API responses, and results are under this directory.
The current results are in [`results/toys_pilot_v2/`](results/toys_pilot_v2/).

## Selection and execution

- Source: 53,898 Toys training rows, with no validation/test input or outcomes.
- Eligible pool: 33,908 prefixes with at least three events, from 9,332 users.
- Pilot: one seeded random eligible prefix per user, selecting 12 users; seed 42.
- Histories: 3–10 interactions; all 54 distinct history ASINs matched metadata.
- Targets were saved separately and never sent to either LLM stage.
- Generation temperature: 0.6; verification temperature: 0.0; maximum output 4,096
  tokens; Qwen thinking disabled; two concurrent requests in the revised run.

## Revised run results

| Measure | Count |
|---|---:|
| Histories attempted | 12 |
| Histories passing generation schema/reference/exact-quote checks | 8 |
| Histories excluded after three failed attempts | 4 |
| Valid rationale units from those eight histories | 10 |
| Units accepted by the separate same-model verifier call | 8 |
| Units marked for revision by that verifier | 2 |
| Units retained as grounded candidates in the assistant audit | 2 |

The run has finished, but `summary.json` reports `complete: false` because four
histories did not pass generation validation. These are recorded failures, not
pending API calls. They were excluded rather than silently repaired or admitted.
The verifier was run on all eight valid histories. Every API attempt is saved.

The 80% model acceptance rate applies only to the ten valid units. It is not a
grounding-accuracy estimate, an end-to-end yield estimate, or evidence of
recommendation improvement.

## What the audit found

Requiring exact quotations catches nonexistent metadata fields and invented
quotations. It does not prove that a quotation supports the observation around it.
For example, the generator combined a carrying play mat, a toy laptop with a
handle, and a toy vacuum into a portability rationale. The vacuum evidence was
only its twist-and-turn action. The verifier still accepted that rationale.

The verifier also accepted a purported progression toward collectible Minecraft
sets despite tied timestamps, and a Peppa Pig refinement claim without a
demonstrated narrowing of attributes. Rating conflicts were sometimes ignored.

The assistant audit retains two provisional continuity candidates: Star Wars
theme/brand continuity and Munchkin bath-toy continuity. These are grounded
observations with uncertain preference interpretations; they have not been shown
to add predictive information beyond the source item metadata. The Munchkin
history contains a duplicate ASIN/event, so its independent product support is two.
Full decisions are in
[`assistant_audit.json`](results/toys_pilot_v2/assistant_audit.json).

The generic `accepted_rationales.jsonl` contains model-screened units, including
false acceptances found by the assistant audit. Do not treat that file as a
trusted evidence bank. Use the audit to distinguish the two provisional
candidates from units requiring revision or exclusion.

## Data limitations and next experiment

Ten of twelve sampled histories have only one distinct timestamp. One history
contains repeated ASINs. A later large run should explicitly summarize unique
product support, flag duplicate events and timestamp ties, and constrain claims
about interest evolution accordingly. Static catalog metadata may also postdate
the original interactions; this pilot does not establish historical availability.

The next useful comparison is a stronger or independently trained grounding
verifier, followed by a text-evidence recommendation test against no rationale,
matched shuffled rationale, and an equally long factual summary. Avoid scaling
the same-model acceptance decisions into a trusted bank before those checks.

## Earlier run and checks

`results/toys_pilot_v1/` preserves the initial prompts inside saved requests. It
used weaker reference checks, generated 18 valid units from ten histories
(including one abstention), and suffered a temporary server interruption during
verification. Do not compare its incomplete verification rate to v2 as an
estimate of prompt improvement. Both versions used identical prepared histories.

The revised implementation passed 12 focused tests covering hidden-target
exclusion, chronological/alignment validation, metadata allowlisting, reference
and quotation existence, verifier gates, correction feedback, cached successes,
and preparation round trips. `git diff --check` passed. No R3 model or training
code was changed by this experiment.
