#!/usr/bin/env python3
"""Claim-level, scored Toys rationale verification inspired by WhisperRec.

Reads saved generations without changing them. All decisions are computed in
Python from validated judge output plus explicit local constraints.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import pipeline as p


DIMENSIONS = ("factuality", "evidence_selection", "signal_strength", "intent_translation",
              "causal_logic", "recommendation_fit", "temporal_reasoning",
              "confidence_calibration", "task_fulfillment")
VIOLATIONS = {"fabricated_behavior", "target_backfill", "fabricated_context",
              "unsupported_need", "unsupported_property", "rating_conflict",
              "duplicate_inflation", "unsupported_temporal_progression"}
POLICY = {"version": 3,
          "minimum_scores": {"factuality": 4, "evidence_selection": 4, "causal_logic": 4},
          "block_unsupported_central_claims": True,
          "history_order_is_authoritative": True,
          "require_distinct_products_for_refinement_or_transition": True}


def claims_for(rationale):
    claims = [{"claim_id": f"O{i+1}", "kind": "observation", "central": True,
               "text": obs["statement"], "required_event_ids": obs["event_ids"]}
              for i, obs in enumerate(p.rationale_evidence(rationale))]
    for cid, text, kind in (("I1", rationale["interpretation"], "inference"),
                            ("C1", p.rationale_recommendation(rationale), "recommendation")):
        claims.append({"claim_id": cid, "kind": kind, "central": True,
                       "text": text, "required_event_ids": []})
    if "text" in rationale:  # Saved legacy generations include a separate summary.
        claims.append({"claim_id": "T1", "kind": "summary", "central": True,
                       "text": rationale["text"], "required_event_ids": []})
    return claims


def diagnostics_for(history, rationale):
    events = history["interactions"]
    keys = Counter((e["asin"], e.get("timestamp"), p.stable_json(e["metadata"].get("rating"))) for e in events)
    refs = {ref for obs in p.rationale_evidence(rationale) for ref in obs["event_ids"]}
    return {"event_count": len(events), "distinct_product_count": len({e["asin"] for e in events}),
            "distinct_timestamp_count": len({e.get("timestamp") for e in events if e.get("timestamp") is not None}),
            "ordered_event_ids": [e["event_id"] for e in events],
            "sequence_policy": "Supplied interaction order is authoritative chronological order.",
            "duplicate_event_groups": [[e["event_id"] for e in events
                                        if (e["asin"], e.get("timestamp"), p.stable_json(e["metadata"].get("rating"))) == key]
                                       for key, count in keys.items() if count > 1],
            "cited_ratings": {e["event_id"]: e["metadata"].get("rating") for e in events if e["event_id"] in refs},
            "timestamp_warning": "Tied or missing timestamps do not invalidate supplied history order; elapsed durations still require evidence."}


def mechanical_violations(history, rationale):
    refs = {ref for obs in p.rationale_evidence(rationale) for ref in obs["event_ids"]}
    cited = [e for e in history["interactions"] if e["event_id"] in refs]
    if rationale["relation"] in {"refinement", "transition"} and len({e["asin"] for e in cited}) < 2:
        return [{"kind": "unsupported_temporal_progression", "claim_ids": [],
                 "origin": "program", "reason": "Refinement/transition lacks two distinct cited products in the ordered history."}]
    return []


def messages_for(history, rationale):
    user = (p.PROMPTS / "scored_verifier_user.txt").read_text()
    values = {"HISTORY_JSON": history, "RATIONALE_JSON": rationale,
              "CLAIMS_JSON": claims_for(rationale), "DIAGNOSTICS_JSON": diagnostics_for(history, rationale)}
    for key, value in values.items():
        user = user.replace("{{" + key + "}}", json.dumps(value, ensure_ascii=False))
    return [{"role": "system", "content": (p.PROMPTS / "scored_verifier_system.txt").read_text()},
            {"role": "user", "content": user}]


def validate_judgment(value, history, rationale):
    if not isinstance(value, dict):
        return ["Judgment must be an object"]
    errors = []
    if set(value) != {"claim_assessments", "scores", "violations"}:
        errors.append("Judgment must contain only claim_assessments, scores, violations")
    scores = value.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        errors.append("Provide all nine scores")
    else:
        for name, score in scores.items():
            if not isinstance(score, dict) or type(score.get("score")) is not int or not 1 <= score["score"] <= 5 or not isinstance(score.get("reason"), str) or not score["reason"].strip():
                errors.append(f"{name}: require integer score 1..5 and reason")
    claims = {c["claim_id"]: c for c in claims_for(rationale)}
    events = {e["event_id"]: e for e in history["interactions"]}
    assessments = value.get("claim_assessments")
    seen = []
    if not isinstance(assessments, list):
        errors.append("claim_assessments must be a list")
    else:
        for a in assessments:
            if not isinstance(a, dict) or not isinstance(a.get("claim_id"), str) or a["claim_id"] not in claims:
                errors.append("Unknown claim assessment")
                continue
            cid = a["claim_id"]
            seen.append(cid)
            if a.get("verdict") not in ("supported", "qualified_inference", "unsupported", "contradicted"):
                errors.append(f"{cid}: invalid verdict")
            if not isinstance(a.get("reason"), str) or not a["reason"].strip():
                errors.append(f"{cid}: missing reason")
            checks = a.get("event_checks")
            if not isinstance(checks, list) or not checks:
                errors.append(f"{cid}: event_checks required")
                continue
            checked = []
            for check in checks:
                if not isinstance(check, dict) or not isinstance(check.get("event_id"), str) or check["event_id"] not in events:
                    errors.append(f"{cid}: unknown event")
                    continue
                eid = check["event_id"]
                checked.append(eid)
                if check.get("support") not in ("supports", "does_not_support", "contradicts"):
                    errors.append(f"{cid}: invalid event support")
                if not isinstance(check.get("reason"), str) or not check["reason"].strip():
                    errors.append(f"{cid}: event reason required")
                quotes = check.get("evidence_quotes")
                if not isinstance(quotes, list):
                    errors.append(f"{cid}: evidence_quotes must be a list")
                    continue
                if check.get("support") == "supports" and not quotes:
                    errors.append(f"{cid}: supports requires exact evidence")
                for q in quotes:
                    if not isinstance(q, dict) or not isinstance(q.get("field"), str) or q["field"] not in events[eid]["metadata"] or not isinstance(q.get("quote"), str) or not q["quote"].strip():
                        errors.append(f"{cid}: invalid quote reference")
                        continue
                    source = events[eid]["metadata"][q["field"]]
                    sources = source if isinstance(source, list) else [source]
                    normalize = lambda s: " ".join(str(s).casefold().split())
                    if not any(normalize(q["quote"]) in normalize(s) for s in sources):
                        errors.append(f"{cid}: quote not found in {eid}.{q['field']}")
            required = claims[cid]["required_event_ids"]
            if len(set(checked)) != len(checked) or (required and sorted(checked) != sorted(required)):
                errors.append(f"{cid}: assess exactly each required event once")
        if sorted(seen) != sorted(claims):
            errors.append("Assess every supplied claim exactly once")
    violations = value.get("violations")
    if not isinstance(violations, list):
        errors.append("violations must be a list")
    else:
        for v in violations:
            if not isinstance(v, dict) or v.get("kind") not in VIOLATIONS or not isinstance(v.get("claim_ids"), list) or any(not isinstance(cid, str) or cid not in claims for cid in v["claim_ids"]) or not isinstance(v.get("reason"), str) or not v["reason"].strip():
                errors.append("Invalid violation")
                continue
            if v["kind"] == "rating_conflict":
                refs = {eid for obs in p.rationale_evidence(rationale) for eid in obs["event_ids"]}
                if not any(eid in refs and isinstance(e["metadata"].get("rating"), (int, float))
                           and 1 <= e["metadata"]["rating"] <= 2 for eid, e in events.items()):
                    errors.append("rating_conflict requires a cited actual low rating (1 or 2); positive ratings and possible motivations are not conflicts")
            if v["kind"] == "duplicate_inflation" and not diagnostics_for(history, rationale)["duplicate_event_groups"]:
                errors.append("duplicate_inflation requires actual duplicate source events")
    return errors


def apply_policy(value, history, rationale):
    errors = validate_judgment(value, history, rationale)
    if errors:
        raise ValueError("; ".join(errors))
    effective = {name: item["score"] for name, item in value["scores"].items()}
    violations = copy.deepcopy(value["violations"]) + mechanical_violations(history, rationale)
    reasons = []
    claim_lookup = {c["claim_id"]: c for c in claims_for(rationale)}
    for a in value["claim_assessments"]:
        observation = claim_lookup[a["claim_id"]]["kind"] == "observation"
        failed = a["verdict"] in {"unsupported", "contradicted"}
        if observation:
            failed |= a["verdict"] != "supported" or any(c["support"] != "supports" for c in a["event_checks"])
        elif a["verdict"] in {"supported", "qualified_inference"}:
            failed |= not any(c["support"] == "supports" for c in a["event_checks"])
        if any(c["support"] == "contradicts" for c in a["event_checks"]):
            failed = True
        if failed:
            reasons.append(f"Unsupported or contradictory central claim {a['claim_id']}")
            effective["factuality"] = min(effective["factuality"], 2 if observation else 3)
    for v in violations:
        kind = v["kind"]
        if kind in {"fabricated_behavior", "target_backfill", "unsupported_property"}:
            effective["factuality"] = min(effective["factuality"], 2)
        elif kind in {"fabricated_context", "unsupported_need", "unsupported_temporal_progression"}:
            effective["factuality"] = min(effective["factuality"], 3)
        if kind in {"rating_conflict", "duplicate_inflation"}:
            effective["evidence_selection"] = min(effective["evidence_selection"], 3)
        if kind == "unsupported_temporal_progression":
            effective["causal_logic"] = min(effective["causal_logic"], 2)
        reasons.append(f"Violation: {kind}: {v['reason']}")
    for name, threshold in POLICY["minimum_scores"].items():
        if effective[name] < threshold:
            reasons.append(f"{name}={effective[name]} below {threshold}")
    return {"decision": "reject" if reasons else "accept", "reasons": reasons,
            "raw_scores": {name: item["score"] for name, item in value["scores"].items()},
            "effective_scores": effective, "violations": violations, "policy": POLICY}


def load_jobs(source):
    manifest = json.loads((source / "data_manifest.json").read_text())
    if p.sha_file(source / "inputs.jsonl") != manifest["inputs_sha256"]:
        raise ValueError("Source inputs differ from their manifest")
    provenance = {r["history_id"]: r for r in p.read_jsonl(source / "provenance.jsonl")}
    jobs, exclusions = [], []
    for index, history in enumerate(p.read_jsonl(source / "inputs.jsonl")):
        path = source / "generations" / f"{index:06d}.json"
        if not path.exists():
            exclusions.append({"history_id": history["history_id"], "reason": "missing generation"})
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "ok" or record.get("input_hash") != p.digest(history) or p.validate_rationales(record.get("parsed", {}), history):
            exclusions.append({"history_id": history["history_id"], "reason": "generation failed source checks"})
            continue
        previous = {}
        old = source / "verifications" / path.name
        if old.exists():
            verification = json.loads(old.read_text())
            if verification.get("status") == "ok" and verification.get("input_hash") == p.digest(history) and verification.get("source_hash") == p.digest(record["parsed"]):
                previous = {a["rationale_id"]: a["decision"] for a in verification["parsed"]["assessments"]}
        for rationale in record["parsed"]["rationales"]:
            jobs.append({"unit_id": history["history_id"] + ":" + rationale["rationale_id"],
                         "history": history, "rationale": rationale,
                         "provenance": provenance[history["history_id"]],
                         "generation_record_sha256": p.sha_file(path),
                         "previous_decision": previous.get(rationale["rationale_id"], "unverified")})
    return jobs, exclusions


def summarize(out, jobs, exclusions):
    counts, all_units, accepted = Counter(), [], []
    for i, job in enumerate(jobs):
        record = json.loads((out / "judgments" / f"{i:06d}.json").read_text())
        if record.get("status") != "ok":
            counts["judge_errors"] += 1
            all_units.append({"unit_id": job["unit_id"],
                              **p.structured_rationale(job["rationale"]),
                              "provenance": job["provenance"],
                              "verification": {"decision": "unverified"}, "status": "error"})
            continue
        if record["job_hash"] != p.digest(job):
            raise ValueError("Stale judgment")
        # Recompute policy rather than trusting any persisted decision.
        decision = apply_policy(record["parsed"], job["history"], job["rationale"])
        counts[decision["decision"]] += 1
        if job["previous_decision"] == "accept":
            counts["previously_accepted"] += 1
            counts["previous_accept_now_" + decision["decision"]] += 1
        entry = {"unit_id": job["unit_id"], **p.structured_rationale(job["rationale"]),
                 "provenance": job["provenance"], "previous_decision": job["previous_decision"],
                 "verification": decision, "claim_assessments": record["parsed"]["claim_assessments"],
                 "quality_status": "scored_grounding_screen_only_not_predictively_verified"}
        all_units.append(entry)
        if decision["decision"] == "accept":
            accepted.append(entry)
    p.write_jsonl(out / "accepted_rationales.jsonl", accepted)
    p.write_jsonl(out / "all_rationales_review.jsonl", all_units)
    summary = {"units": len(jobs), "excluded_histories": exclusions, "counts": dict(counts),
               "all_units_judged": counts["judge_errors"] == 0,
               "acceptance_rate": counts["accept"] / len(jobs) if jobs else None,
               "limitations": ["Judge assessments can still be wrong; scoring is not proof of entailment.",
                               "Default judge is the same Qwen3-8B model used for generation.",
                               "No recommendation or cross-user-transfer gain is established."]}
    p.write_json(out / "summary.json", summary)
    lines = ["# Scored claim verification", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for entry in all_units:
        v = entry.get("verification", {})
        lines.extend([f"## {entry['unit_id']} — {v.get('decision', 'unverified')}", "",
                      p.rationale_markdown(entry), "",
                      "Reasons: " + "; ".join(v.get("reasons", [])), ""])
    (out / "review.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


def run(args, prepared=None):
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    if out == source or source in out.parents:
        raise ValueError("Choose an output directory outside the original pilot directory")
    jobs, exclusions = load_jobs(source) if prepared is None else prepared
    if not jobs:
        raise ValueError("No valid source rationale units")
    config = {"source_dir": str(source), "jobs_hash": p.digest(jobs), "base_url": args.base_url,
              "model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
              "seed": args.seed, "policy": POLICY,
              "code_hashes": {f.name: p.sha_file(f) for f in (Path(__file__), Path(p.__file__))},
              "prompt_hashes": {name: p.sha_file(p.PROMPTS / name) for name in
                                ("scored_verifier_system.txt", "scored_verifier_user.txt")}}
    if out.exists() and any(out.iterdir()) and not (out / "config.json").exists():
        raise ValueError("Output directory contains unrelated files")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Configuration changed; use a new output directory")
    p.write_json(path, config)
    p.write_jsonl(out / "jobs.jsonl", jobs)
    p.write_json(out / "source_exclusions.json", exclusions)
    folder = out / "judgments"
    folder.mkdir(exist_ok=True)
    if args.dry_run:
        print(f"Prepared {len(jobs)} units; excluded {len(exclusions)} invalid source histories.")
        return

    def work(index_job):
        index, job = index_job
        path = folder / f"{index:06d}.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous.get("status") == "ok" and previous.get("job_hash") == p.digest(job) and previous.get("config_hash") == p.digest(config):
            return "cached"
        record = {"unit_id": job["unit_id"], "job_hash": p.digest(job),
                  "config_hash": p.digest(config), "attempts": previous.get("attempts", []) if previous else []}
        messages = messages_for(job["history"], job["rationale"])
        for attempt in range(args.retries + 1):
            trace = {"messages": copy.deepcopy(messages)}
            started = time.monotonic()
            try:
                response = p.request_completion(args, messages, args.seed + index + (len(record["attempts"]) * 100000))
                trace["response"] = response
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
                parsed = p.parse_json_response(choice["message"].get("content"))
                errors = validate_judgment(parsed, job["history"], job["rationale"])
                if errors:
                    messages += [{"role": "assistant", "content": choice["message"]["content"]},
                                 {"role": "user", "content": "Correct the JSON schema/references: " + "; ".join(errors)
                                  + ". Return the full judgment. Preserve genuine negative verdicts; do not invent quotes to pass validation."}]
                    raise ValueError("; ".join(errors))
                record.update(status="ok", parsed=parsed,
                              decision=apply_policy(parsed, job["history"], job["rationale"]))
            except Exception as exc:
                record["status"] = "error"
                trace["error"] = f"{type(exc).__name__}: {exc}"
            trace["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["attempts"].append(trace)
            p.write_json(path, record)
            if record["status"] == "ok":
                break
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
        return record["status"]

    counts = Counter()
    for state in p.concurrent_results(work, enumerate(jobs), total=len(jobs),
                                      max_workers=args.concurrency, label="judge"):
        counts[state] += 1
    summarize(out, jobs, exclusions)
    if counts["error"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://album:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error("Require positive concurrency and nonnegative retries")
    run(args)


if __name__ == "__main__":
    main()
