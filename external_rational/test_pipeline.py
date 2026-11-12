"""Focused checks for target exclusion, evidence references, and verifier gates."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline as p


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "user_id": "u1", "history_item_id": "[1, 2, 3]",
            "item_asins": "['a1', 'a2', 'a3']",
            "history_item_title": "['Puzzle A', 'Puzzle B', 'Puzzle C']",
            "history_rating": "[4.0, 5.0, 4.0]",
            "history_timestamp": "[10, 20, 30]", "timestamp": "40",
            "item_id": "TARGET_ID_SENTINEL", "item_asin": "TARGET_ASIN_SENTINEL",
            "item_title": "TARGET_TITLE_SENTINEL", "rating": "5.0",
        }
        events, _ = p.parse_history(self.row, 20)
        self.history = {"dataset": "Toys", "history_id": "Toys:train:0", "interactions": events}
        self.generated = {
            "status": "ok", "comparison": "Repeated puzzle interactions support continuity.",
            "rationales": [{
                "rationale_id": "R1", "facet": "puzzles", "relation": "continuity",
                "applicability": "Repeated puzzle interactions",
                "observations": [{"event_ids": ["H1", "H2"], "metadata_fields": ["title"],
                                  "statement": "Both titles contain Puzzle",
                                  "evidence_quotes": [
                                      {"event_id": "H1", "field": "title", "quote": "Puzzle A"},
                                      {"event_id": "H2", "field": "title", "quote": "Puzzle B"}]}],
                "interpretation": "May indicate interest in puzzles",
                "continuation": "Additional puzzle products",
                "limitations": ["Titles alone do not establish desired difficulty"],
                "downweight_when": ["Recent interactions move to a different product type"],
                "text": "Repeated puzzle interactions may support puzzle continuity.",
            }],
        }

    def test_targets_never_enter_inputs_or_messages(self):
        for stage in ("generator", "verifier"):
            messages = p.make_messages(stage, self.history, self.generated)
            self.assertNotIn("SENTINEL", json.dumps(messages))

    def test_alignment_and_order_fail_closed(self):
        for changed in ({"history_item_title": "['one']"},
                        {"history_timestamp": "[20, 10, 30]"}, {"timestamp": "25"}):
            with self.assertRaises(ValueError):
                p.parse_history({**self.row, **changed}, 20)

    def test_truncation_preserves_original_positions(self):
        events, length = p.parse_history(self.row, 2)
        self.assertEqual(length, 3)
        self.assertEqual([e["event_id"] for e in events], ["H2", "H3"])

    def test_valid_and_abstention(self):
        self.assertEqual(p.validate_rationales(self.generated, self.history), [])
        value = {"status": "insufficient_evidence", "rationales": [], "comparison": "Too sparse"}
        self.assertEqual(p.validate_rationales(value, self.history), [])
        value["status"] = "ok"
        self.assertTrue(p.validate_rationales(value, self.history))

    def test_structured_reasoning_round_trip_preserves_citations(self):
        original = copy.deepcopy(self.generated["rationales"][0])
        structured = p.structured_rationale(original)
        self.assertEqual(structured["evidence"], original["observations"])
        self.assertEqual(structured["interpretation"], original["interpretation"])
        self.assertEqual(structured["recommendation"], original["continuation"])
        self.assertNotIn("text", structured)
        self.assertNotIn("observations", structured)
        self.assertNotIn("continuation", structured)
        self.assertEqual(p.structured_rationale(structured), structured)
        value = {**self.generated, "rationales": [structured]}
        self.assertEqual(p.validate_rationales(value, self.history), [])
        self.assertEqual(self.generated["rationales"][0], original)
        rendered = p.rationale_markdown(structured)
        for label in ("Evidence", "Interpretation", "Recommendation"):
            self.assertIn(label, rendered)

    def test_structured_reasoning_requires_all_three_parts_and_valid_quotes(self):
        structured = p.structured_rationale(self.generated["rationales"][0])
        for field in ("evidence", "interpretation", "recommendation"):
            changed = copy.deepcopy(structured)
            changed.pop(field)
            self.assertTrue(p.validate_rationales({**self.generated, "rationales": [changed]}, self.history))
        changed = copy.deepcopy(structured)
        changed["evidence"][0]["evidence_quotes"][0]["quote"] = "Invented feature"
        self.assertTrue(p.validate_rationales({**self.generated, "rationales": [changed]}, self.history))

    def test_bad_citations_and_fields_rejected(self):
        for field, changed in (("event_ids", ["H999"]), ("metadata_fields", ["compatibility"])):
            value = copy.deepcopy(self.generated)
            value["rationales"][0]["observations"][0][field] = changed
            self.assertTrue(p.validate_rationales(value, self.history))

    def test_missing_qualification_rejected(self):
        value = copy.deepcopy(self.generated)
        value["rationales"][0]["limitations"] = []
        self.assertTrue(p.validate_rationales(value, self.history))

    def test_verifier_must_cover_all_units_and_resolve_flags(self):
        self.assertTrue(p.validate_verification({"assessments": []}, self.generated))
        assessment = {"rationale_id": "R1", "decision": "accept", "reason": "Grounded",
                      "unsupported_claims": [], "incorrect_citations": [],
                      "checks": {key: True for key in ("grounding", "inference", "relation",
                                                       "specificity", "applicability", "uncertainty")}}
        self.assertEqual(p.validate_verification({"assessments": [assessment]}, self.generated), [])
        assessment["unsupported_claims"] = ["Invented recipient age"]
        self.assertTrue(p.validate_verification({"assessments": [assessment]}, self.generated))
        assessment["unsupported_claims"] = []
        assessment["checks"]["grounding"] = False
        self.assertTrue(p.validate_verification({"assessments": [assessment]}, self.generated))

    def test_quotes_must_match_and_cover_each_reference(self):
        value = copy.deepcopy(self.generated)
        value["rationales"][0]["observations"][0]["evidence_quotes"][0]["quote"] = "Invented feature"
        self.assertTrue(p.validate_rationales(value, self.history))
        value = copy.deepcopy(self.generated)
        value["rationales"][0]["observations"][0]["evidence_quotes"].pop()
        self.assertTrue(p.validate_rationales(value, self.history))

    def test_retry_gets_specific_feedback_and_success_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p.write_jsonl(out / "inputs.jsonl", [self.history])
            p.write_json(out / "data_manifest.json", {"inputs_sha256": p.sha_file(out / "inputs.jsonl")})
            invalid = copy.deepcopy(self.generated)
            invalid["rationales"][0]["observations"][0]["metadata_fields"] = ["invented"]
            response = lambda value: {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(value)}}]}
            args = SimpleNamespace(output_dir=out, command="generator", base_url="http://test/v1",
                                   model="test", temperature=0.0, max_tokens=100, seed=42,
                                   retries=1, concurrency=1)
            with patch.object(p, "request_completion", side_effect=[response(invalid), response(self.generated)]) as call, patch.object(p.time, "sleep"):
                p.run_stage(args)
                self.assertIn("invalid metadata field reference", call.call_args[0][1][-1]["content"])
                p.run_stage(args)
                self.assertEqual(call.call_count, 2)
            artifact = json.loads((out / "generations/000000.json").read_text())
            self.assertEqual(artifact["status"], "ok")
            self.assertEqual(len(artifact["attempts"]), 2)

    def test_qwen_wrapper_and_non_object_response(self):
        self.assertEqual(p.parse_json_response('<think></think>\n{"x":1}'), {"x": 1})
        with self.assertRaises(ValueError):
            p.parse_json_response("[]")

    def test_metadata_allowlist(self):
        value = p.clean_metadata({"title": "<b>Puzzle</b>", "brand": "Maker",
                                  "also_buy": ["FUTURE_ASIN"], "rank": ["#1"],
                                  "description": ["<p>Pieces &amp; patterns</p>"]})
        self.assertNotIn("also_buy", value)
        self.assertNotIn("rank", value)
        self.assertEqual(value["description"], "Pieces & patterns")

    def test_prepare_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train, meta, out = base / "train.csv", base / "meta.jsonl", base / "result"
            with train.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.row))
                writer.writeheader()
                writer.writerow(self.row)
            meta.write_text(json.dumps({"asin": "a1", "category": ["Puzzles"]}) + "\n")
            args = SimpleNamespace(output_dir=out, train=train, metadata=meta, dataset="Toys",
                                   min_history=3, max_history=20, limit=1, seed=42)
            p.build_inputs(args)
            inputs = (out / "inputs.jsonl").read_text()
            self.assertNotIn("SENTINEL", inputs)
            self.assertIn("TARGET_TITLE_SENTINEL", (out / "targets_audit_only.jsonl").read_text())
            manifest = json.loads((out / "data_manifest.json").read_text())
            self.assertEqual(manifest["inputs_sha256"], p.sha_file(out / "inputs.jsonl"))
            with self.assertRaises(ValueError):
                p.build_inputs(args)


if __name__ == "__main__":
    unittest.main()
