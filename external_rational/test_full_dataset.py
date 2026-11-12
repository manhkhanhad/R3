"""Full-prefix selection and resumable runner checks without live inference."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline as p
import run_full_dataset as full
import judge_full_dataset as judge
import scored_verification as scored


class FullDatasetTest(unittest.TestCase):
    def prepare_fixture(self, base, prepare_only=False):
        train, metadata = base / "train.csv", base / "meta.jsonl"
        row = {"user_id": "same-user", "history_item_id": "[1]", "item_asins": "['a1']",
               "history_item_title": "['Toy A']", "history_rating": "[5.0]",
               "history_timestamp": "[1]", "timestamp": "3", "item_id": "target",
               "item_asin": "target-asin", "item_title": "HIDDEN_TARGET", "rating": "5.0"}
        second = {**row, "history_item_id": "[1, 2]", "item_asins": "['a1', 'a2']",
                  "history_item_title": "['Toy A', 'Toy B']", "history_rating": "[5.0, 4.0]",
                  "history_timestamp": "[1, 2]"}
        with train.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows([row, second])
        metadata.write_text(' {"asin":"a1","brand":"Maker"}\n{"asin":"a2","brand":"Maker"}\n')
        argv = ["--output-dir", str(base / "run"), "--train", str(train),
                "--metadata", str(metadata), "--retries", "0"]
        if prepare_only:
            argv.append("--prepare-only")
        return full.parse_args(argv)

    def generated(self):
        return {"status": "ok", "comparison": "Toy A supports a possible toy interest", "rationales": [{
            "rationale_id": "R1", "facet": "toys", "relation": "continuity",
            "applicability": "Histories containing Toy A", "evidence": [{
                "event_ids": ["H1"], "metadata_fields": ["title"], "statement": "Toy A appears in the history",
                "evidence_quotes": [{"event_id": "H1", "field": "title", "quote": "Toy A"}]}],
            "interpretation": "Toy A may suggest interest in similar toys",
            "recommendation": "Consider toys with similar attributes",
            "limitations": ["One observation"], "downweight_when": ["Recent low toy ratings"]}]}

    def response(self, args, messages, seed):
        generated = self.generated()
        if getattr(args, "command", None) == "generator":
            value = generated
        else:
            value = {"scores": {name: {"score": 5, "reason": "Grounded"} for name in scored.DIMENSIONS},
                     "violations": [], "claim_assessments": []}
            for claim in scored.claims_for(generated["rationales"][0]):
                value["claim_assessments"].append({"claim_id": claim["claim_id"],
                    "verdict": "supported" if claim["kind"] == "observation" else "qualified_inference",
                    "reason": "Supported by the toy title", "event_checks": [{
                        "event_id": "H1", "support": "supports", "reason": "Title matches",
                        "evidence_quotes": [{"field": "title", "quote": "Toy A"}]}]})
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    def test_full_selection_keeps_multiple_prefixes_per_user_and_sparse_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            args = self.prepare_fixture(base, prepare_only=True)
            self.assertEqual(full.run(args), 0)
            source = args.output_dir / "generation"
            inputs = list(p.read_jsonl(source / "inputs.jsonl"))
            self.assertEqual(len(inputs), 2)
            self.assertEqual([len(h["interactions"]) for h in inputs], [1, 2])
            self.assertNotIn("HIDDEN_TARGET", (source / "inputs.jsonl").read_text())
            manifest = json.loads((source / "data_manifest.json").read_text())
            self.assertEqual(manifest["counts"]["eligible_users"], 1)
            pilot = SimpleNamespace(**{**vars(args), "output_dir": base / "pilot", "selection": "one-per-user"})
            p.build_inputs(pilot)
            self.assertEqual(len(list(p.read_jsonl(base / "pilot/inputs.jsonl"))), 1)

    def test_generation_and_judge_run_separately_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp))
            args.concurrency = 6
            source = args.output_dir / "generation"
            judge_args = judge.parse_args(["--input-dir", str(source), "--concurrency", "3", "--retries", "0"])
            with patch.object(p, "request_completion", side_effect=self.response) as call:
                with patch.object(scored, "run") as verify:
                    self.assertEqual(full.run(args), 0)
                    verify.assert_not_called()
                self.assertEqual(call.call_count, 2)
                self.assertFalse((args.output_dir / "verification").exists())
                self.assertEqual(call.call_args[0][0].concurrency, 6)
                hashes = {f.name: p.sha_file(f) for f in (source / "generations").glob("[0-9]*.json")}
                self.assertEqual(judge.run(judge_args), 0)
                self.assertEqual(call.call_count, 4)
                self.assertEqual(call.call_args[0][0].concurrency, 3)
                self.assertEqual(hashes, {f.name: p.sha_file(f) for f in (source / "generations").glob("[0-9]*.json")})
                self.assertEqual(full.run(args), 0)
                self.assertEqual(judge.run(judge_args), 0)
                self.assertEqual(call.call_count, 4)
                args.min_history = 2
                with self.assertRaisesRegex(ValueError, "Preparation settings"):
                    full.run(args)
            accepted = list(p.read_jsonl(args.output_dir / "verification/accepted_rationales.jsonl"))
            self.assertEqual(len(accepted), 2)
            for unit in accepted:
                self.assertTrue({"evidence", "interpretation", "recommendation"} <= unit.keys())
                self.assertNotIn("text", unit)
            judge_args.output_dir = source / "judge"
            with self.assertRaisesRegex(ValueError, "outside"):
                judge.run(judge_args)

    def test_code_only_resume_preserves_successes_and_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp))
            def fail_first(args, messages, seed):
                if seed == args.seed:
                    raise RuntimeError("Interrupted request")
                return self.response(args, messages, seed)
            with patch.object(p, "request_completion", side_effect=fail_first):
                self.assertEqual(full.run(args), 1)
            folder = args.output_dir / "generation/generations"
            success_hash = p.sha_file(folder / "000001.json")
            old_config = json.loads((folder / "config.json").read_text())
            original_sha = p.sha_file
            def updated_code(path):
                return "updated-code-hash" if Path(path).resolve() == Path(p.__file__).resolve() else original_sha(path)
            with patch.object(p, "sha_file", side_effect=updated_code):
                with self.assertRaisesRegex(ValueError, "Preparation settings"):
                    full.run(args)
                args.resume = True
                with patch.object(p, "request_completion", side_effect=self.response) as call:
                    self.assertEqual(full.run(args), 0)
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(call.call_args[0][2], args.seed + 100000)
            self.assertEqual(p.sha_file(folder / "000001.json"), success_hash)
            retried = json.loads((folder / "000000.json").read_text())
            self.assertEqual(len(retried["attempts"]), 2)
            self.assertIn("error", retried["attempts"][0])
            self.assertEqual(retried["attempts"][0]["config_hash"], p.digest(old_config))
            self.assertEqual(retried["attempts"][1]["config_hash"], retried["config_hash"])
            archived = folder / "config_history" / (p.digest(old_config) + ".json")
            self.assertEqual(json.loads(archived.read_text()), old_config)

    def test_resume_still_rejects_model_prompt_and_preparation_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp))
            with patch.object(p, "request_completion", side_effect=self.response):
                self.assertEqual(full.run(args), 0)
            args.resume = True
            original_sha = p.sha_file
            def changed_prompt(path):
                return "changed-prompt" if Path(path).name == "generator_system.txt" else original_sha(path)
            with patch.object(p, "sha_file", side_effect=changed_prompt), patch.object(p, "request_completion") as call:
                with self.assertRaisesRegex(ValueError, "Stage configuration"):
                    full.run(args)
                call.assert_not_called()
            args.model = "different-model"
            with self.assertRaisesRegex(ValueError, "Stage configuration"):
                full.run(args)
            args.model = "Qwen/Qwen3.5-9B"
            args.min_history = 2
            with self.assertRaisesRegex(ValueError, "Preparation settings"):
                full.run(args)

    def test_partial_judge_excludes_failed_and_missing_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp))
            with patch.object(p, "request_completion", side_effect=self.response):
                self.assertEqual(full.run(args), 0)
            source = args.output_dir / "generation"
            failed_path = source / "generations/000001.json"
            failed = json.loads(failed_path.read_text())
            failed["status"] = "error"
            p.write_json(failed_path, failed)
            successful_hash = p.sha_file(source / "generations/000000.json")
            judge_args = judge.parse_args(["--input-dir", str(source), "--allow-partial", "--concurrency", "32"])
            with patch.object(p, "request_completion", side_effect=self.response) as call, patch.object(scored, "load_jobs", wraps=scored.load_jobs) as load:
                self.assertEqual(judge.run(judge_args), 0)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(judge.run(judge_args), 0)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(load.call_count, 2)  # One full-source scan per invocation.
            exclusions = json.loads((args.output_dir / "verification/source_exclusions.json").read_text())
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(exclusions[0]["history_id"], "Toys_and_Games:train:1")
            self.assertEqual(p.sha_file(source / "generations/000000.json"), successful_hash)
            failed_path.unlink()
            judge_args.output_dir = args.output_dir / "missing-verification"
            with patch.object(p, "request_completion", side_effect=self.response):
                self.assertEqual(judge.run(judge_args), 0)
            exclusions = json.loads((judge_args.output_dir / "source_exclusions.json").read_text())
            self.assertEqual(exclusions[0]["reason"], "missing generation")

    def test_partial_judge_does_not_report_success_when_no_units_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp), prepare_only=True)
            self.assertEqual(full.run(args), 0)
            judge_args = judge.parse_args(["--input-dir", str(args.output_dir / "generation"), "--allow-partial"])
            with patch.object(p, "request_completion") as call:
                with self.assertRaisesRegex(ValueError, "No valid"):
                    judge.run(judge_args)
                call.assert_not_called()

    def test_partial_generation_writes_report_and_does_not_start_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare_fixture(Path(tmp))
            def fail_first(args, messages, seed):
                if seed == args.seed:
                    raise RuntimeError("Test server interruption")
                return self.response(args, messages, seed)
            with patch.object(p, "request_completion", side_effect=fail_first), patch.object(scored, "run") as verify:
                self.assertEqual(full.run(args), 1)
                verify.assert_not_called()
            self.assertTrue((args.output_dir / "generation/summary.json").exists())
            judge_args = judge.parse_args(["--input-dir", str(args.output_dir / "generation")])
            with self.assertRaisesRegex(ValueError, "Finish generation"):
                judge.run(judge_args)
            with patch.object(p, "request_completion", side_effect=self.response):
                self.assertEqual(full.run(args), 0)
            retried = json.loads((args.output_dir / "generation/generations/000000.json").read_text())
            self.assertEqual(len(retried["attempts"]), 2)
            self.assertIn("error", retried["attempts"][0])
            self.assertEqual(retried["attempts"][1]["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
