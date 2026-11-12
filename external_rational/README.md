# External rationale generation

Implementation and run guide: [External reasoning generation](../document/EXTERNAL_REASONING_GENERATION.md).

Completed Toys pilot results and quality findings:
[`PILOT_FINDINGS.md`](PILOT_FINDINGS.md). The same-model verifier has known false
acceptances in this pilot; consult the saved assistant audit before using units.

The new [claim-level scored verifier](SCORED_VERIFICATION.md) adds nine quality
scores, per-event claim checks, score caps, and program-enforced acceptance gates.
It can evaluate the saved Toys generations without repeating generation or
modifying the original pilot artifacts.
The [live scored-verifier findings](SCORED_VERIFICATION_FINDINGS.md) report
3 accepted and 7 rejected units in the calibrated run, including a remaining
false acceptance. Automatic admission is not yet reliable with the default judge.

Generate rationale units from training histories using the OpenAI-compatible
vLLM server at `http://album:8000/v1`, model `Qwen/Qwen3-8B`. No Python packages
beyond the standard library are required. Run commands from the repository root.

The first experiment uses Toys only. The four editable prompt files are in
[`prompts/`](prompts/). Their structure adapts the preference abstraction,
alternative hypotheses, and transition comparison from
[OneReason, Section 5.4](https://arxiv.org/html/2606.06260v1#S5.SS4).

`results/toys_pilot_v1` preserves the initial prompt experiment and its weaker
checks. The current prompts require exact source quotations and six explicit
verifier checks; use `results/toys_pilot_v2` with the commands below.

## Full training dataset

Generation and judging have separate entrypoints. Run generation first:

```bash
python -u external_rational/run_full_dataset.py \
  --output-dir external_rational/results/toys_full_qwen35_9b \
  --concurrency 2
```

After generation finishes, run the judge independently on its saved directory:

```bash
python -u external_rational/judge_full_dataset.py \
  --input-dir external_rational/results/toys_full_qwen35_9b/generation \
  --output-dir external_rational/results/toys_full_qwen35_9b/verification \
  --concurrency 2
```

Both scripts default to `http://sodag:8000/v1` and `Qwen/Qwen3.5-9B`. Each has
its own `--base-url`, `--model`, `--concurrency`, `--max-tokens`, `--timeout`, and
`--retries` options, so the judge can use a different server or model later.
Both scripts automatically show a progress bar with completed/total jobs,
percentage, elapsed time, estimated time remaining (`ETA`), estimated total stage
runtime, cached count, and error count. Updates follow actual completion order,
so one slow request cannot hide other completed jobs. Estimates appear after the
first uncached job finishes and use observed job durations and worker count;
cached/skipped jobs do not seed the estimate. Estimates are approximate and
include retries in observed job time. Interactive terminals refresh in place
once per second; redirected logs receive plain progress lines every 10 seconds
plus initial/final updates. Progress is written to stderr; use `2>&1` to capture
it together with stdout. Input preparation and report writing are outside the
inference-stage estimate. No progress dependency or extra flag is required.

Generation never invokes judging. Judging reads the saved generations and does
not generate or modify them. By default every prepared history must pass
generation checks. If some histories exhausted their generation retries, add
`--allow-partial` to the judge command to process only valid saved units. Failed
and missing histories are recorded in `verification/source_exclusions.json` and
`summary.json`; they are not admitted to the judge or counted as substantive
rationale rejections. If no valid rationale units remain, the judge refuses to
report success. Keep generation files fixed during judging. If later generation
retries add or change valid units, use a new verification output directory: the
existing judge run retains its frozen job list and configuration.
Its `--output-dir` defaults to a sibling
`verification/` directory when omitted.

Generation includes all nonempty training-history prefixes (`--selection
all-prefixes --limit 0 --min-history 1`) and supplies the latest 20 interactions
per prefix. Multiple prefixes from the same user and sparse histories are
included; weak evidence may produce abstentions. Validation/test data and
next-item targets are withheld. `--train`, `--metadata`, and `--dataset` allow
another dataset. `--min-history`, `--max-history`, `--selection`, and `--limit`
adjust coverage. Use `--prepare-only` to prepare inputs without inference.

To continue generation after a code-only update such as the progress display,
add `--resume` to the original generation command:

```bash
python -u external_rational/run_full_dataset.py \
  --output-dir external_rational/results/toys_full_qwen35_9b --resume
```

This permits a changed pipeline code hash while requiring identical prepared
data, selection settings, prompts, model, and decoding parameters. Prior
preparation and generation configs are archived in `preparation_history/` and
`generation/generations/config_history/`. Successful records remain unchanged;
failed records retain earlier attempts and use advanced retry seeds. Each new
attempt records its config hash. Changes to prompts, model, or data still require
a new run directory. `--resume` does not enable judging.

Repeat either script's command to resume its own work: successful records are
cached and failed records are retried, preserving earlier API attempts. Keep
data, prompts, and each stage's settings fixed; use a new output directory when
they change. Code-only generation changes may use the explicit --resume path above. Use one process per run directory. Finish/retry generation
before starting the judge so its frozen job list stays stable. Each script
returns a nonzero exit status if requests exhaust their retries, while saving
successful units and error reports. Invalid judge outputs remain unverified.

Outputs are separated under the run root:

- `preparation.json`: frozen generation preparation configuration.
- `generation/inputs.jsonl`, `data_manifest.json`, and `generations/`: full inputs,
  source coverage, and raw generation requests/responses.
- `generation/all_rationales_review.jsonl` and `review.md`: structured generated
  units, marked unverified by the generation report.
- `verification/accepted_rationales.jsonl`: units accepted by the judge,
  with `evidence`, `interpretation`, and `recommendation` fields.
- `verification/summary.json` and `review.md`: scored counts and decisions.

Use the verification report for quality decisions. The generation report does
not import scored results. Automated acceptance does not establish predictive
utility. The old combined `--stage` and `--judge-*` options have been replaced by
the two independent scripts above.

## Run the pilot

```bash
python external_rational/pipeline.py prepare \
  --output-dir external_rational/results/toys_pilot_v2 \
  --limit 12 --min-history 3 --max-history 20 --seed 42

python external_rational/pipeline.py generator \
  --output-dir external_rational/results/toys_pilot_v2

python external_rational/pipeline.py verifier \
  --output-dir external_rational/results/toys_pilot_v2

python external_rational/pipeline.py report \
  --output-dir external_rational/results/toys_pilot_v2
```

Preparation requires an empty output directory. Generation and verification
resume successful records automatically; failed records are retried. A change to
code, prompts, model, input content, or decoding configuration requires a new run
directory. Sampling and API seeds are recorded, but GPU inference need not be
bit-for-bit deterministic. Avoid concurrent writers to the same run directory.
Schema/reference/quotation failures trigger a correction request containing the
specific errors; the original response and every correction request are saved.

Use `--concurrency`, `--timeout`, `--retries`, `--base-url`, and `--model` to
configure each inference stage. `VLLM_API_KEY` is read if set and never saved.
Requests bypass environment HTTP proxies for the internal server. Qwen thinking
is disabled explicitly and JSON responses are requested; see the
[official Qwen vLLM guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html).
The requested output is the short, explicit rationale, not a private reasoning
trace. Full responses, usage, requests, and validation failures are preserved.

## Saved reasoning format

New generations and report exports use three explicit reasoning fields:
`evidence` (a list of factual statements with event/field/quote citations),
`interpretation` (a qualified preference hypothesis), and `recommendation`
(possible next-item attributes or directions). Readable reports display these as
three separate sections. Item titles identify products in prose; event IDs stay
in citation fields. Limitations and downweight conditions remain separate.

Legacy generation responses with `observations`, `continuation`, and `text`
remain readable. Report export maps `observations` to `evidence` and
`continuation` to `recommendation` without modifying original API responses or
inventing a new interpretation. A combined `text` field is no longer required
for new generations. The scored verifier checks every evidence observation,
the interpretation, and the recommendation; legacy standalone text is also
checked when present. Use a new run directory for future generation/verification
because schema, prompts, and code hashes have changed.

## Inputs and selection

Defaults use the existing Toys training CSV and `meta_Toys_and_Games.json`.
The pilot chooses one seeded random eligible prefix per user, then selects
12 users by seeded priority. The minimum original history length is three;
only the latest 20 events are supplied, preserving original event IDs/order.
This pilot therefore excludes sparse histories and does not represent all rows.

The generator sees item IDs/ASINs, original training titles, ratings, event
timestamps, and available category, brand, feature, and description fields.
HTML is removed and metadata length is bounded. Popularity ranks and `also_buy`
or `also_view` links are excluded. Missing metadata remains missing. Metadata is
a static catalog snapshot; its historical availability is not guaranteed.

The target, target title/rating/timestamp, and raw user identifier never enter
generation or grounding-verification requests. The target is saved in a separate
audit artifact. User hashes and evidence cutoff times are retained for later
retrieval exclusions. Input alignment and timestamp ordering fail closed.

## Artifacts under each result directory

| Artifact | Contents |
|---|---|
| `inputs.jsonl` | Exact target-free LLM inputs |
| `targets_audit_only.jsonl` | Separated next items; never loaded by inference stages |
| `provenance.jsonl` | Source row, user hash, evidence time, prefix hash |
| `metadata_subset.jsonl` | Selected catalog metadata |
| `data_manifest.json` | Data hashes, selection settings, metadata coverage |
| `generations/` | Prompts, model responses, parsed units, retries, decoding config |
| `verifications/` | Separate grounding-pass responses and decisions |
| `accepted_rationales.jsonl` | Units passing syntax/reference checks and LLM grounding screen |
| `all_rationales_review.jsonl` | Accepted, revision-required, rejected, or unverified units |
| `summary.json`, `review.md` | Counts and readable rationale/verification review |

`accept` means only that the separate model call passed a grounding screen.
By default the generator and verifier use the **same model**, so their errors
can correlate. Automatic checks establish reference and quotation existence,
but do not prove the quoted evidence entails the claim. Human review and an
independent/stronger verifier are useful next checks.

No held-out recommendation benefit, cross-user transfer, embedding quality, or
latent-composition benefit is established by this pipeline. Do those experiments
before treating the output as verified predictive evidence. Preserve donor-user
and time exclusions when retrieving these units downstream.

For a larger run, use a new directory and a larger `--limit`. `--limit 0` prepares
one eligible prefix per training user, not every prefix. Choose that scale after
reviewing the pilot. CDs can use the same pipeline with explicit `--train`,
`--metadata`, and `--dataset` arguments; no CDs generation is included here.

## Focused tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s external_rational -p 'test_*.py' -v
```
