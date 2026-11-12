"""Acceptance regressions for claim-level scored verification."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline as p
import scored_verification as s


class ScoredVerificationTest(unittest.TestCase):
    def setUp(self):
        self.history = {"history_id": "toy:train:0", "dataset": "Toys", "interactions": [
            {"event_id": "H1", "asin": "a1", "timestamp": 1,
             "metadata": {"brand": "Maker", "title": "Toy A", "rating": 5.0}},
            {"event_id": "H2", "asin": "a2", "timestamp": 2,
             "metadata": {"brand": "Maker", "title": "Toy B", "rating": 4.0}}]}
        self.rationale = {
            "rationale_id": "R1", "facet": "brand", "relation": "continuity",
            "applicability": "Repeated brand in distinct products",
            "observations": [{"event_ids": ["H1", "H2"], "metadata_fields": ["brand"],
                              "statement": "Both products have the Maker brand",
                              "evidence_quotes": [{"event_id": e, "field": "brand", "quote": "Maker"}
                                                  for e in ("H1", "H2")]}],
            "interpretation": "May support brand continuity",
            "continuation": "Consider the same brand",
            "limitations": ["Other motivations remain possible"],
            "downweight_when": ["Recent products have low ratings for this brand"],
            "text": "Repeated Maker products may support continuity, with uncertain intent."}
        self.judgment = {"scores": {name: {"score": 5, "reason": "Evidence matches"}
                                    for name in s.DIMENSIONS}, "violations": [],
                         "claim_assessments": []}
        for claim in s.claims_for(self.rationale):
            ids = claim["required_event_ids"] or ["H1"]
            self.judgment["claim_assessments"].append({
                "claim_id": claim["claim_id"],
                "verdict": "supported" if claim["kind"] == "observation" else "qualified_inference",
                "event_checks": [{"event_id": eid, "support": "supports",
                                  "evidence_quotes": [{"field": "brand", "quote": "Maker"}],
                                  "reason": "Observed shared brand"} for eid in ids],
                "reason": "Supported observation or tentative inference"})

    def decide(self, judgment=None):
        return s.apply_policy(judgment or self.judgment, self.history, self.rationale)

    def test_supported_unit_passes(self):
        self.assertEqual(s.validate_judgment(self.judgment, self.history, self.rationale), [])
        self.assertEqual(self.decide()["decision"], "accept")

    def test_structured_reasoning_claims_cover_evidence_interpretation_recommendation(self):
        self.rationale = p.structured_rationale(self.rationale)
        self.judgment["claim_assessments"] = [a for a in self.judgment["claim_assessments"] if a["claim_id"] != "T1"]
        claims = s.claims_for(self.rationale)
        self.assertEqual([c["claim_id"] for c in claims], ["O1", "I1", "C1"])
        self.assertEqual(claims[0]["required_event_ids"], ["H1", "H2"])
        self.assertEqual(claims[-1]["text"], self.rationale["recommendation"])
        self.assertEqual(self.decide()["decision"], "accept")
        self.judgment["claim_assessments"][-1]["verdict"] = "unsupported"
        self.assertEqual(self.decide()["decision"], "reject")

    def test_unsupported_central_claim_blocks_perfect_scores(self):
        self.judgment["claim_assessments"][0]["verdict"] = "unsupported"
        result = self.decide()
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["raw_scores"]["factuality"], 5)
        self.assertEqual(result["effective_scores"]["factuality"], 2)

    def test_shared_property_must_hold_for_every_event(self):
        # Even a model's positive overall verdict cannot override a failed event check.
        check = self.judgment["claim_assessments"][0]["event_checks"][1]
        check["support"] = "does_not_support"
        self.assertEqual(self.decide()["decision"], "reject")

    def test_factual_observation_cannot_hide_as_inference(self):
        self.judgment["claim_assessments"][0]["verdict"] = "qualified_inference"
        self.assertEqual(self.decide()["decision"], "reject")

    def test_contradiction_blocks_qualified_summary(self):
        self.judgment["claim_assessments"][-1]["event_checks"][0]["support"] = "contradicts"
        self.assertEqual(self.decide()["decision"], "reject")

    def test_caps_override_judge_scores(self):
        for kind, cap in (("fabricated_behavior", 2), ("target_backfill", 2),
                          ("unsupported_property", 2), ("fabricated_context", 3),
                          ("unsupported_need", 3)):
            value = copy.deepcopy(self.judgment)
            value["violations"] = [{"kind": kind, "claim_ids": ["O1"], "reason": "Unsupported"}]
            result = self.decide(value)
            self.assertEqual(result["effective_scores"]["factuality"], cap)
            self.assertEqual(result["decision"], "reject")

    def test_missing_and_duplicate_claims_fail_schema(self):
        value = copy.deepcopy(self.judgment)
        value["claim_assessments"].pop()
        self.assertTrue(s.validate_judgment(value, self.history, self.rationale))
        value = copy.deepcopy(self.judgment)
        value["claim_assessments"].append(copy.deepcopy(value["claim_assessments"][0]))
        self.assertTrue(s.validate_judgment(value, self.history, self.rationale))

    def test_positive_ratings_cannot_be_fabricated_into_a_conflict(self):
        self.judgment["violations"] = [{"kind": "rating_conflict", "claim_ids": ["I1"],
                                        "reason": "The positive ratings might reflect other motivations"}]
        self.assertTrue(s.validate_judgment(self.judgment, self.history, self.rationale))
        self.history["interactions"][0]["metadata"]["rating"] = 2.0
        self.assertEqual(s.validate_judgment(self.judgment, self.history, self.rationale), [])
        self.assertEqual(self.decide()["decision"], "reject")

    def test_duplicate_violation_requires_duplicate_records(self):
        self.judgment["violations"] = [{"kind": "duplicate_inflation", "claim_ids": ["O1"], "reason": "Inflated count"}]
        self.assertTrue(s.validate_judgment(self.judgment, self.history, self.rationale))

    def test_missing_event_and_fake_quote_fail_schema(self):
        value = copy.deepcopy(self.judgment)
        value["claim_assessments"][0]["event_checks"].pop()
        self.assertTrue(s.validate_judgment(value, self.history, self.rationale))
        value = copy.deepcopy(self.judgment)
        value["claim_assessments"][0]["event_checks"][0]["evidence_quotes"][0]["quote"] = "Invented portability"
        self.assertTrue(s.validate_judgment(value, self.history, self.rationale))

    def test_scores_require_integers_and_thresholds(self):
        for bad in (True, 0, 6, "5", 4.5):
            value = copy.deepcopy(self.judgment)
            value["scores"]["factuality"]["score"] = bad
            self.assertTrue(s.validate_judgment(value, self.history, self.rationale))
        self.judgment["scores"]["causal_logic"]["score"] = 3
        self.assertEqual(self.decide()["decision"], "reject")

    def test_tied_timestamps_allow_supported_refinement_from_history_order(self):
        self.history["interactions"][1]["timestamp"] = 1
        self.rationale["relation"] = "refinement"
        self.assertEqual(self.decide()["decision"], "accept")
        diagnostics = s.diagnostics_for(self.history, self.rationale)
        self.assertEqual(diagnostics["ordered_event_ids"], ["H1", "H2"])

    def test_missing_timestamps_allow_supported_transition_from_history_order(self):
        for event in self.history["interactions"]:
            event.pop("timestamp")
        self.rationale["relation"] = "transition"
        self.assertEqual(self.decide()["decision"], "accept")
        self.assertEqual(s.diagnostics_for(self.history, self.rationale)["distinct_timestamp_count"], 0)

    def test_single_product_still_blocks_refinement(self):
        self.history["interactions"][1]["asin"] = "a1"
        self.rationale["relation"] = "refinement"
        result = self.decide()
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["effective_scores"]["causal_logic"], 2)
        self.assertEqual(result["violations"][0]["origin"], "program")

    def test_history_order_does_not_override_unsupported_preference_inference(self):
        self.history["interactions"][1]["timestamp"] = 1
        self.rationale["relation"] = "refinement"
        self.judgment["claim_assessments"][1]["verdict"] = "unsupported"
        self.assertEqual(self.decide()["decision"], "reject")

    def test_duplicate_diagnostics_do_not_inflate_support(self):
        event = copy.deepcopy(self.history["interactions"][0])
        event["event_id"] = "H3"
        self.history["interactions"].append(event)
        d = s.diagnostics_for(self.history, self.rationale)
        self.assertEqual(d["distinct_product_count"], 2)
        self.assertEqual(d["duplicate_event_groups"], [["H1", "H3"]])

    def test_messages_only_contain_allowed_inputs(self):
        text = json.dumps(s.messages_for(self.history, self.rationale))
        for excluded in ("TARGET_TITLE_SENTINEL", "previous_decision", "assistant_audit", "targets_audit_only"):
            self.assertNotIn(excluded, text)

    def test_live_path_correction_cache_and_original_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "scored"
            (source / "generations").mkdir(parents=True)
            p.write_jsonl(source / "inputs.jsonl", [self.history])
            p.write_json(source / "data_manifest.json", {"inputs_sha256": p.sha_file(source / "inputs.jsonl")})
            p.write_jsonl(source / "provenance.jsonl", [{"history_id": self.history["history_id"]}])
            generation = {"status": "ok", "input_hash": p.digest(self.history),
                          "parsed": {"status": "ok", "comparison": "Brand continuity",
                                     "rationales": [self.rationale]}}
            gp = source / "generations/000000.json"
            p.write_json(gp, generation)
            original_hash = p.sha_file(gp)
            args = SimpleNamespace(source_dir=source, output_dir=out, base_url="http://test/v1",
                                   model="test", temperature=0.0, max_tokens=1000, seed=42,
                                   dry_run=False, concurrency=1, retries=1)
            invalid = copy.deepcopy(self.judgment)
            invalid["scores"]["factuality"]["score"] = 6
            response = lambda value: {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
            with patch.object(p, "request_completion", side_effect=[response(invalid), response(self.judgment)]) as request, patch.object(s.time, "sleep"):
                s.run(args)
                self.assertIn("integer score", request.call_args[0][1][-1]["content"])
                s.run(args)
                self.assertEqual(request.call_count, 2)
            self.assertEqual(p.sha_file(gp), original_hash)
            self.assertEqual(len(list(p.read_jsonl(out / "accepted_rationales.jsonl"))), 1)
            self.assertTrue(json.loads((out / "summary.json").read_text())["all_units_judged"])


if __name__ == "__main__":
    unittest.main()
