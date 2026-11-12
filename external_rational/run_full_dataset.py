#!/usr/bin/env python3
"""Generate structured rationales for every training-history prefix.

Standard library only. Defaults: Toys, sodag:8000, Qwen/Qwen3.5-9B.
Run judging later with judge_full_dataset.py. Generation is resumable.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pipeline as p


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Run root; prepared inputs and responses are saved under generation/")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare inputs without calling vLLM")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run across code-only updates; data/prompts/model/decoding must match")
    parser.add_argument("--dataset", default="Toys_and_Games")
    parser.add_argument("--train", type=Path, default=p.TOYS_TRAIN)
    parser.add_argument("--metadata", type=Path, default=p.TOYS_META)
    parser.add_argument("--selection", choices=("all-prefixes", "one-per-user"), default="all-prefixes")
    parser.add_argument("--limit", type=int, default=0, help="0 processes the full eligible dataset")
    parser.add_argument("--min-history", type=int, default=1, help="Include sparse nonempty histories by default")
    parser.add_argument("--max-history", type=int, default=20, help="Latest interactions supplied per prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://sodag:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.concurrency < 1 or args.retries < 0
            or args.timeout < 1 or args.max_tokens < 1
            or args.min_history < 1 or args.max_history < args.min_history or args.limit < 0):
        parser.error("Require positive concurrency/timeout/token limits, nonnegative retries/limit, and 1 <= min-history <= max-history")
    return args


def prepare(args, source):
    settings = {"dataset": args.dataset, "train": str(args.train.resolve()),
                "metadata": str(args.metadata.resolve()), "selection": args.selection,
                "limit": args.limit, "min_history": args.min_history,
                "max_history": args.max_history, "seed": args.seed,
                "pipeline_sha256": p.sha_file(Path(p.__file__))}
    config_path = args.output_dir / "preparation.json"
    if not config_path.exists():
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory contains unrelated files; choose a new run root")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    p.save_compatible_config(config_path, settings, resume=getattr(args, "resume", False),
                             label="Preparation settings/code")
    manifest_path = source / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (p.sha_file(args.train) != manifest["train_sha256"]
                or p.sha_file(args.metadata) != manifest["metadata_sha256"]
                or p.sha_file(source / "inputs.jsonl") != manifest["inputs_sha256"]):
            raise ValueError("Prepared inputs or source data changed; use a new run directory")
        print("Using existing prepared inputs", flush=True)
    else:
        p.build_inputs(SimpleNamespace(**{**vars(args), "output_dir": source}))


def run(args):
    args.output_dir = args.output_dir.resolve()
    source = args.output_dir / "generation"
    prepare(args, source)
    if args.prepare_only:
        print(f"Prepared histories: {source / 'inputs.jsonl'}", flush=True)
        return 0
    generation_args = SimpleNamespace(**{**vars(args), "output_dir": source, "command": "generator"})
    try:
        p.run_stage(generation_args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            p.report(SimpleNamespace(output_dir=source))
            print("Generation has errors. Repeat this command to retry them.", flush=True)
            return 1
    p.report(SimpleNamespace(output_dir=source))
    print(f"Generated structured rationales: {source / 'all_rationales_review.jsonl'}", flush=True)
    print(f"Judge later with judge_full_dataset.py --input-dir {source}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
