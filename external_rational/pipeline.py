#!/usr/bin/env python3
"""Train-only rationale preparation, vLLM generation, and grounding verification.

Standard-library only. No R3 checkpoint, target, or future outcome enters an LLM
request. See README.md for artifacts and limitations.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from rationale_progress import concurrent_results


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROMPTS = ROOT / "prompts"
TOYS_TRAIN = REPO / "data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv"
TOYS_META = REPO / "data/Toys_and_Games/meta_Toys_and_Games.json"
RELATIONS = {"continuity", "refinement", "transition", "complementarity", "other"}


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def save_compatible_config(path, config, *, resume=False, label="Configuration"):
    """Archive code revisions while keeping inference/data settings immutable."""
    path = Path(path)
    compatible = lambda value: {key: item for key, item in value.items()
                                if key != "pipeline_sha256"}
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and previous != config:
        if not resume or compatible(previous) != compatible(config):
            raise ValueError(label + " changed; --resume permits code-only changes, otherwise use a new output directory")
    history = path.parent / (path.stem + "_history")
    history.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        write_json(history / (digest(previous) + ".json"), previous)
    write_json(history / (digest(config) + ".json"), config)
    write_json(path, config)
    accepted_hashes = {digest(config)}
    for archived in history.glob("*.json"):
        value = json.loads(archived.read_text())
        if compatible(value) == compatible(config):
            accepted_hashes.add(digest(value))
    return accepted_hashes


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        for row in rows:
            f.write(stable_json(row) + "\n")
    temp.replace(path)


def clean_text(value, limit):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value))).strip()[:limit]


def clean_metadata(raw):
    result = {}
    for key, limit in (("title", 300), ("brand", 160)):
        if text := clean_text(raw.get(key), limit):
            result[key] = text
    for key, count, limit in (("category", 10, 160), ("feature", 6, 200)):
        values = raw.get(key, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            texts = [clean_text(v, limit) for v in values if isinstance(v, str)]
            if texts := [v for v in texts if v][:count]:
                result[key] = texts
    values = raw.get("description", [])
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list):
        text = clean_text(" ".join(v for v in values if isinstance(v, str)), 800)
        if text:
            result["description"] = text
    return result


def parse_history(row, max_history):
    columns = ["history_item_id", "item_asins", "history_item_title",
               "history_rating", "history_timestamp"]
    fields = {key: ast.literal_eval(row[key]) for key in columns}
    lengths = {len(v) for v in fields.values() if isinstance(v, list)}
    if any(not isinstance(v, list) for v in fields.values()) or len(lengths) != 1:
        raise ValueError("History columns must be aligned lists")
    timestamps = fields["history_timestamp"]
    if timestamps != sorted(timestamps):
        raise ValueError("History timestamps are not chronological")
    if timestamps and max(timestamps) > int(row["timestamp"]):
        raise ValueError("History extends past its training outcome")
    events = []
    offset = max(0, len(timestamps) - max_history)
    for pos in range(offset, len(timestamps)):
        events.append({
            "event_id": f"H{pos + 1}",
            "item_id": str(fields["history_item_id"][pos]),
            "asin": str(fields["item_asins"][pos]),
            "sequence_position": pos + 1,
            "timestamp": timestamps[pos],
            "metadata": {"title": clean_text(fields["history_item_title"][pos], 300),
                         "rating": fields["history_rating"][pos]},
        })
    return events, len(timestamps)


def build_inputs(args):
    if args.min_history < 1 or args.max_history < args.min_history or args.limit < 0:
        raise ValueError("Require 1 <= min-history <= max-history and limit >= 0")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Preparation requires an empty output directory; use a new run directory")
    selection = getattr(args, "selection", "one-per-user")
    if selection not in {"one-per-user", "all-prefixes"}:
        raise ValueError("Unknown history selection mode")
    rng = random.Random(args.seed)
    per_user, all_prefixes, eligible_users = {}, [], set()
    counts = Counter()
    with args.train.open(newline="") as f:
        for index, row in enumerate(csv.DictReader(f)):
            counts["train_rows"] += 1
            events, full_length = parse_history(row, args.max_history)
            if full_length < args.min_history:
                continue
            counts["eligible_rows"] += 1
            user = row["user_id"]
            eligible_users.add(user)
            if selection == "all-prefixes":
                all_prefixes.append((index, index, row, events, full_length))
            else:
                # Seeded pilot sampling retains one eligible prefix per user.
                priority = rng.random()
                if user not in per_user or priority < per_user[user][0]:
                    per_user[user] = (priority, index, row, events, full_length)
    candidates = all_prefixes if selection == "all-prefixes" else sorted(per_user.values(), key=lambda x: x[0])
    selected = candidates[:args.limit] if args.limit else candidates
    if not selected:
        raise ValueError("No eligible training histories")
    counts["eligible_users"] = len(eligible_users)
    required = {e["asin"] for _, _, _, events, _ in selected for e in events}
    metadata = {}
    opener = gzip.open if args.metadata.suffix == ".gz" else open
    print(f"Scanning metadata for {len(required)} selected history ASINs", flush=True)
    with opener(args.metadata, "rt", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            asin = str(raw.get("asin", ""))
            if asin in required and asin not in metadata:
                metadata[asin] = clean_metadata(raw)
    counts["history_asins"] = len(required)
    counts["matched_metadata_asins"] = len(metadata)
    counts["histories"] = len(selected)
    inputs, targets, provenance = [], [], []
    for _, index, row, events, full_length in selected:
        history_id = f"{args.dataset}:train:{index}"
        for e in events:
            # Original training title takes precedence; retain the source title separately.
            source = metadata.get(e["asin"], {}).copy()
            source.pop("title", None)
            e["metadata"].update(source)
        inputs.append({"dataset": args.dataset, "history_id": history_id,
                       "interactions": events})
        targets.append({"history_id": history_id, "item_id": row["item_id"],
                        "asin": row["item_asin"], "title": row["item_title"],
                        "rating": float(row["rating"]), "timestamp": int(row["timestamp"])})
        provenance.append({"history_id": history_id, "train_row_index": index,
                           "source_user_hash": digest(row["user_id"]),
                           "full_history_length": full_length,
                           "max_evidence_timestamp": max(e["timestamp"] for e in events),
                           "source_history_hash": digest(events)})
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "targets_audit_only.jsonl", targets)
    write_jsonl(out / "provenance.jsonl", provenance)
    write_jsonl(out / "metadata_subset.jsonl", ({"asin": k, **v} for k, v in sorted(metadata.items())))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset, "split": "train", "seed": args.seed,
                "selection": ("all eligible training prefixes in CSV order" if selection == "all-prefixes"
                              else "one random eligible prefix per user, ordered by seeded priority"),
                "selection_mode": selection,
                "limit": args.limit, "min_history": args.min_history,
                "max_history": args.max_history, "train_path": str(args.train.resolve()),
                "train_sha256": sha_file(args.train),
                "metadata_path": str(args.metadata.resolve()),
                "metadata_sha256": sha_file(args.metadata),
                "inputs_sha256": sha_file(out / "inputs.jsonl"), "counts": dict(counts),
                "target_policy": "targets saved separately, never sent to generator or grounding verifier"}
    write_json(out / "data_manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2), flush=True)


def parse_json_response(text):
    if not isinstance(text, str):
        raise ValueError("Missing response content")
    # Some servers retain the empty Qwen thinking wrapper in non-thinking mode.
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object")
    return value


def rationale_evidence(rationale):
    """Read cited evidence from current or legacy rationale units."""
    return rationale.get("evidence", rationale.get("observations", []))


def rationale_recommendation(rationale):
    return rationale.get("recommendation", rationale.get("continuation", ""))


def structured_rationale(rationale):
    """Export the three reasoning components without changing source responses."""
    metadata = {key: value for key, value in rationale.items()
                if key not in {"evidence", "observations", "interpretation",
                               "recommendation", "continuation", "text"}}
    return {"evidence": rationale_evidence(rationale),
            "interpretation": rationale["interpretation"],
            "recommendation": rationale_recommendation(rationale), **metadata}


def rationale_markdown(rationale):
    if "interpretation" not in rationale:
        return "Judge error"
    evidence = "\n".join("- " + observation["statement"]
                         for observation in rationale_evidence(rationale))
    return ("**Evidence**\n\n" + evidence + "\n\n**Interpretation**\n\n"
            + rationale["interpretation"] + "\n\n**Recommendation**\n\n"
            + rationale_recommendation(rationale))


def validate_rationales(value, history):
    errors = []
    units = value.get("rationales")
    if value.get("status") not in {"ok", "insufficient_evidence"}:
        errors.append("Invalid status")
    if not isinstance(value.get("comparison"), str) or not value["comparison"].strip():
        errors.append("Missing comparison")
    if not isinstance(units, list) or len(units) > 3:
        return errors + ["rationales must be a list of at most three units"]
    if (value.get("status") == "ok") != bool(units):
        errors.append("Status must agree with whether rationales are present")
    events = {e["event_id"]: e for e in history["interactions"]}
    seen = set()
    for r in units:
        if not isinstance(r, dict):
            errors.append("Rationale must be an object")
            continue
        rid = r.get("rationale_id")
        if not isinstance(rid, str) or not re.fullmatch(r"R[1-3]", rid) or rid in seen:
            errors.append("Invalid or duplicate rationale_id")
        if isinstance(rid, str):
            seen.add(rid)
        structured = "evidence" in r or "recommendation" in r
        required = ("facet", "applicability", "interpretation") + (
            ("recommendation",) if structured else ("continuation", "text"))
        if structured and ("observations" in r or "continuation" in r):
            errors.append(f"{rid}: do not mix current and legacy reasoning fields")
        for field in required:
            if not isinstance(r.get(field), str) or not r[field].strip():
                errors.append(f"{rid}: missing {field}")
        if r.get("relation") not in RELATIONS:
            errors.append(f"{rid}: invalid relation")
        for field in ("limitations", "downweight_when"):
            if not isinstance(r.get(field), list) or not r[field] or any(
                    not isinstance(s, str) or not s.strip() for s in r[field]):
                errors.append(f"{rid}: {field} must contain strings")
        observations = rationale_evidence(r)
        if not isinstance(observations, list) or not observations:
            errors.append(f"{rid}: missing cited evidence")
            continue
        for obs in observations:
            if not isinstance(obs, dict):
                errors.append(f"{rid}: observation must be an object")
                continue
            refs, fields = obs.get("event_ids"), obs.get("metadata_fields")
            if not isinstance(obs.get("statement"), str) or not obs["statement"].strip():
                errors.append(f"{rid}: missing observation statement")
            if not isinstance(refs, list) or not refs or any(
                    not isinstance(ref, str) or ref not in events for ref in refs):
                errors.append(f"{rid}: invalid event reference")
                continue
            if not isinstance(fields, list) or not fields or any(
                    not isinstance(field, str) or any(
                        field not in events[ref]["metadata"] for ref in refs) for field in fields):
                errors.append(f"{rid}: invalid metadata field reference")
                continue
            quotes = obs.get("evidence_quotes")
            covered = set()
            if not isinstance(quotes, list) or not quotes:
                errors.append(f"{rid}: missing exact evidence quotes")
                continue
            for q in quotes:
                if not isinstance(q, dict):
                    errors.append(f"{rid}: quote must be an object")
                    continue
                ref, field, quote = q.get("event_id"), q.get("field"), q.get("quote")
                if ref not in refs or field not in fields or not isinstance(quote, str) or not quote.strip():
                    errors.append(f"{rid}: invalid quote reference")
                    continue
                source = events[ref]["metadata"][field]
                sources = source if isinstance(source, list) else [source]
                normalize = lambda s: " ".join(str(s).casefold().split())
                if not any(normalize(quote) in normalize(s) for s in sources):
                    errors.append(f"{rid}: quote not found in {ref}.{field}")
                else:
                    covered.add((ref, field))
            if covered != {(ref, field) for ref in refs for field in fields}:
                errors.append(f"{rid}: every cited event/field needs a matching exact quote")
    return errors


def validate_verification(value, generated):
    errors = []
    assessments = value.get("assessments")
    if not isinstance(assessments, list):
        return ["assessments must be a list"]
    expected = {r["rationale_id"] for r in generated["rationales"]}
    ids = []
    for a in assessments:
        if not isinstance(a, dict):
            errors.append("Assessment must be an object")
            continue
        ids.append(a.get("rationale_id"))
        if a.get("decision") not in {"accept", "revise", "reject"}:
            errors.append("Invalid verifier decision")
        for key in ("unsupported_claims", "incorrect_citations"):
            if not isinstance(a.get(key), list) or any(not isinstance(s, str) for s in a[key]):
                errors.append(f"Invalid {key}")
        if not isinstance(a.get("reason"), str) or not a["reason"].strip():
            errors.append("Missing verifier reason")
        checks = a.get("checks")
        names = {"grounding", "inference", "relation", "specificity", "applicability", "uncertainty"}
        if not isinstance(checks, dict) or set(checks) != names or any(type(v) is not bool for v in checks.values()):
            errors.append("Verifier must provide six boolean checks")
        elif a.get("decision") == "accept" and not all(checks.values()):
            errors.append("Acceptance contradicts a failed quality check")
        if a.get("decision") == "accept" and (a.get("unsupported_claims") or a.get("incorrect_citations")):
            errors.append("Acceptance contradicts flagged claims/citations")
    if any(not isinstance(rid, str) for rid in ids) or sorted(ids) != sorted(expected):
        errors.append("Verifier must assess each rationale exactly once")
    return errors


def make_messages(stage, history, generated=None):
    system = (PROMPTS / f"{stage}_system.txt").read_text()
    user = (PROMPTS / f"{stage}_user.txt").read_text()
    # Never pass an entire CSV row: history is explicitly allowlisted at preparation.
    user = user.replace("{{HISTORY_JSON}}", json.dumps(history, ensure_ascii=False))
    if stage == "verifier":
        user = user.replace("{{RATIONALES_JSON}}", json.dumps(generated, ensure_ascii=False))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_completion(args, messages, seed):
    body = {"model": args.model, "messages": messages, "temperature": args.temperature,
            "top_p": 0.8, "max_tokens": args.max_tokens, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"}}
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    # Internal album service should not be routed through a site HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=args.timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:1200]}") from exc


def run_stage(args):
    out = args.output_dir
    manifest = json.loads((out / "data_manifest.json").read_text())
    if sha_file(out / "inputs.jsonl") != manifest["inputs_sha256"]:
        raise ValueError("Inputs changed since preparation")
    histories = list(read_jsonl(out / "inputs.jsonl"))
    stage = args.command
    folder = out / ("generations" if stage == "generator" else "verifications")
    folder.mkdir(exist_ok=True)
    config = {"stage": stage, "base_url": args.base_url, "model": args.model,
              "temperature": args.temperature, "max_tokens": args.max_tokens,
              "seed": args.seed, "enable_thinking": False,
              "inputs_sha256": manifest["inputs_sha256"],
              "pipeline_sha256": sha_file(Path(__file__)),
              "prompt_hashes": {name: sha_file(PROMPTS / name) for name in
                                (f"{stage}_system.txt", f"{stage}_user.txt")}}
    config_path = folder / "config.json"
    compatible_hashes = save_compatible_config(
        config_path, config, resume=getattr(args, "resume", False), label="Stage configuration")

    def work(index_history):
        index, history = index_history
        path = folder / f"{index:06d}.json"
        generated = None
        if stage == "verifier":
            generation_path = out / "generations" / path.name
            if not generation_path.exists():
                return "missing_generation"
            source = json.loads(generation_path.read_text())
            if source.get("status") != "ok" or source.get("input_hash") != digest(history):
                return "invalid_generation"
            generated = source["parsed"]
            source_hash = digest(generated)
        else:
            source_hash = None
        previous = None
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("config_hash") not in compatible_hashes:
                raise ValueError(f"Saved record has incompatible configuration: {path}")
            if previous.get("status") == "ok" and previous.get("input_hash") == digest(history) and previous.get("source_hash") == source_hash:
                return "cached"
        prior_attempts = (previous.get("attempts", []) if previous
                          and previous.get("input_hash") == digest(history)
                          and previous.get("config_hash") in compatible_hashes
                          and previous.get("source_hash") == source_hash else [])
        for prior in prior_attempts:
            prior.setdefault("config_hash", previous["config_hash"])
        record = {"history_id": history["history_id"], "input_hash": digest(history),
                  "source_hash": source_hash, "config_hash": digest(config), "attempts": prior_attempts}
        if stage == "verifier" and not generated["rationales"]:
            record.update(status="ok", parsed={"assessments": []}, skipped="generator_abstained")
            write_json(path, record)
            return "abstained"
        messages = make_messages(stage, history, generated)
        record["messages"] = messages
        for attempt in range(args.retries + 1):
            started = time.monotonic()
            trace = {"attempt": len(record["attempts"]) + 1, "config_hash": digest(config), "messages": list(messages)}
            try:
                response = request_completion(args, messages, args.seed + index + len(record["attempts"]) * 100000)
                trace["response"] = response
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
                value = parse_json_response(choice["message"].get("content"))
                errors = validate_rationales(value, history) if stage == "generator" else validate_verification(value, generated)
                if errors:
                    messages = messages + [
                        {"role": "assistant", "content": choice["message"]["content"]},
                        {"role": "user", "content": "Your JSON failed validation: " + "; ".join(errors)
                         + ". Return a complete corrected JSON object. Use only the original supplied evidence; do not invent missing metadata or quotes."},
                    ]
                    raise ValueError("; ".join(errors))
                record.update(status="ok", parsed=value)
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"
                record["status"] = "error"
            trace["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["attempts"].append(trace)
            if record["status"] == "ok":
                break
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
        write_json(path, record)
        return record["status"]

    counts = Counter()
    for state in concurrent_results(work, enumerate(histories), total=len(histories),
                                    max_workers=args.concurrency, label=stage):
        counts[state] += 1
    print(json.dumps(dict(counts), indent=2), flush=True)
    if counts["error"] or counts["missing_generation"] or counts["invalid_generation"]:
        raise SystemExit(1)


def report(args):
    out = args.output_dir
    histories = list(read_jsonl(out / "inputs.jsonl"))
    provenance = {p["history_id"]: p for p in read_jsonl(out / "provenance.jsonl")}
    counts, accepted, review = Counter(), [], []
    for index, history in enumerate(histories):
        counts["histories"] += 1
        gp = out / "generations" / f"{index:06d}.json"
        vp = out / "verifications" / gp.name
        if not gp.exists():
            counts["generation_missing"] += 1
            continue
        generation = json.loads(gp.read_text())
        if generation.get("status") != "ok" or generation.get("input_hash") != digest(history):
            counts["generation_errors"] += 1
            continue
        counts["generation_ok"] += 1
        value = generation["parsed"]
        counts["generator_abstentions"] += value["status"] == "insufficient_evidence"
        counts["generated_units"] += len(value["rationales"])
        assessments = {}
        if vp.exists():
            verification = json.loads(vp.read_text())
            if verification.get("status") == "ok" and verification.get("source_hash") == digest(value) and verification.get("input_hash") == digest(history):
                counts["verification_ok"] += 1
                assessments = {a["rationale_id"]: a for a in verification["parsed"]["assessments"]}
            else:
                counts["verification_errors"] += 1
        else:
            counts["verification_missing"] += 1
        for rationale in value["rationales"]:
            a = assessments.get(rationale["rationale_id"], {"decision": "unverified"})
            counts[a["decision"]] += 1
            entry = {"unit_id": history["history_id"] + ":" + rationale["rationale_id"],
                     "history_id": history["history_id"], **structured_rationale(rationale),
                     "provenance": provenance[history["history_id"]],
                     "generation_config_hash": generation["config_hash"],
                     "verification": a,
                     "quality_status": "grounding_screen_only_not_predictively_verified"}
            review.append(entry)
            if a["decision"] == "accept":
                accepted.append(entry)
    write_jsonl(out / "accepted_rationales.jsonl", accepted)
    write_jsonl(out / "all_rationales_review.jsonl", review)
    summary = {"counts": dict(counts),
               "complete": counts["generation_ok"] == len(histories) and counts["verification_ok"] == len(histories),
               "acceptance_rate": counts["accept"] / counts["generated_units"] if counts["generated_units"] else None,
               "limitations": ["LLM grounding checks are fallible; inspect examples manually.",
                               "A separate call to the same model is not an independent model judge.",
                               "No held-out predictive/transfer benefit or latent-path benefit has been measured.",
                               "Source-user/time exclusion must be applied before downstream retrieval."]}
    write_json(out / "summary.json", summary)
    lines = ["# External rationale pilot", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for entry in review:
        lines.extend([f"## {entry['unit_id']} — {entry['verification']['decision']}", "",
                      rationale_markdown(entry), "", f"Verifier: {entry['verification'].get('reason', 'Not verified')}", ""])
    (out / "review.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--train", type=Path, default=TOYS_TRAIN)
    prep.add_argument("--metadata", type=Path, default=TOYS_META)
    prep.add_argument("--dataset", default="Toys_and_Games")
    prep.add_argument("--limit", type=int, default=12, help="0 selects all eligible histories under the selection mode")
    prep.add_argument("--selection", choices=("one-per-user", "all-prefixes"), default="one-per-user")
    prep.add_argument("--min-history", type=int, default=3)
    prep.add_argument("--max-history", type=int, default=20)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--output-dir", type=Path, required=True)
    for name in ("generator", "verifier"):
        run = sub.add_parser(name)
        run.add_argument("--output-dir", type=Path, required=True)
        run.add_argument("--base-url", default="http://album:8000/v1")
        run.add_argument("--model", default="Qwen/Qwen3-8B")
        run.add_argument("--concurrency", type=int, default=4)
        run.add_argument("--temperature", type=float, default=0.6 if name == "generator" else 0.0)
        run.add_argument("--max-tokens", type=int, default=4096)
        run.add_argument("--timeout", type=int, default=240)
        run.add_argument("--retries", type=int, default=2)
        run.add_argument("--seed", type=int, default=42)
        run.add_argument("--resume", action="store_true", help="Allow code-only updates while resuming unchanged inference settings")
    review = sub.add_parser("report")
    review.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        build_inputs(args)
    elif args.command in {"generator", "verifier"}:
        if args.concurrency < 1 or args.retries < 0:
            parser.error("concurrency must be positive and retries nonnegative")
        run_stage(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
