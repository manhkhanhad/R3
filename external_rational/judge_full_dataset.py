#!/usr/bin/env python3
"""Judge previously generated structured rationales without running generation.

Standard library only. Defaults: sodag:8000, Qwen/Qwen3.5-9B.
Successful judgments are cached; rerun with unchanged settings to retry errors.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pipeline as p
import scored_verification as scored


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Prepared generation directory containing inputs.jsonl and generations/")
    parser.add_argument("--output-dir", type=Path, help="Defaults to a sibling verification/ directory")
    parser.add_argument("--base-url", default="http://sodag:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true", help="Judge valid saved generations and record failed/missing histories as exclusions")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.max_tokens < 1 or args.timeout < 1 or args.retries < 0:
        parser.error("Require positive concurrency/token limit/timeout and nonnegative retries")
    return args


def generation_complete(source):
    manifest = json.loads((source / "data_manifest.json").read_text())
    if p.sha_file(source / "inputs.jsonl") != manifest["inputs_sha256"]:
        raise ValueError("Prepared inputs changed")
    for index, history in enumerate(p.read_jsonl(source / "inputs.jsonl")):
        path = source / "generations" / f"{index:06d}.json"
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        if (record.get("status") != "ok" or record.get("input_hash") != p.digest(history)
                or p.validate_rationales(record.get("parsed", {}), history)):
            return False
    return True



def run(args):
    source = args.input_dir.resolve()
    verification = args.output_dir.resolve() if args.output_dir else source.parent / "verification"
    if source == verification or source in verification.parents:
        raise ValueError("Judge output must be outside the generation input directory")
    if not getattr(args, "allow_partial", False) and not generation_complete(source):
        raise ValueError("Finish generation before judging, or pass --allow-partial to judge valid saved units and exclude failed/missing histories")
    jobs, exclusions = scored.load_jobs(source)
    if exclusions:
        print(f"Partial verification: {len(jobs):,} valid rationale units; "
              f"{len(exclusions):,} failed/missing histories excluded. "
              "Keep generation files fixed for this verification run.", flush=True)
    if not jobs and exclusions:
        raise ValueError("No valid generated rationale units to judge; retry generation first")
    if not jobs:
        verification.mkdir(parents=True, exist_ok=True)
        scored.summarize(verification, jobs, exclusions)
        return 0
    judge_args = SimpleNamespace(source_dir=source, output_dir=verification,
                                 base_url=args.base_url, model=args.model, temperature=0.0,
                                 max_tokens=args.max_tokens, seed=args.seed, timeout=args.timeout,
                                 retries=args.retries, concurrency=args.concurrency, dry_run=False)
    try:
        scored.run(judge_args, prepared=(jobs, exclusions))
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("Judge errors remain unverified. Repeat this judge command with the same settings to retry them.", flush=True)
            return 1
    print(f"Accepted structured rationales: {verification / 'accepted_rationales.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
