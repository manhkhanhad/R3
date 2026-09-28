import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import experiment_tracking
from utils.ranking_metrics import calculate_ranking_metrics, infer_dataset_name


class _FakeRun:
    def __init__(self):
        self.log_code = mock.Mock()


class ExperimentTrackingTest(unittest.TestCase):
    def test_shared_evaluation_joins_run_without_finishing_it(self):
        fake_run = _FakeRun()
        fake_wandb = SimpleNamespace(
            Settings=mock.Mock(side_effect=lambda **kwargs: kwargs),
            init=mock.Mock(return_value=fake_run),
        )
        environment = {
            "WANDB_RUN_ID": "shared123",
            "WANDB_PROJECT": "recommendation",
            "WANDB_TAGS": "baseline",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ):
            result = experiment_tracking.init_evaluation_run("CDs_and_Vinyl")

        self.assertIs(result, fake_run)
        settings = fake_wandb.Settings.call_args.kwargs
        self.assertEqual(settings["mode"], "shared")
        self.assertFalse(settings["x_primary"])
        self.assertFalse(settings["x_update_finish_state"])
        init_args = fake_wandb.init.call_args.kwargs
        self.assertEqual(init_args["id"], "shared123")
        self.assertEqual(init_args["project"], "recommendation")
        self.assertEqual(init_args["tags"], ["CDs_and_Vinyl", "baseline"])
        code_args = fake_run.log_code.call_args
        self.assertEqual(code_args.args, (str(Path(__file__).resolve().parents[1]),))
        self.assertTrue(code_args.kwargs["include_fn"]("train.py"))
        self.assertTrue(code_args.kwargs["include_fn"]("config.yaml"))
        self.assertTrue(code_args.kwargs["include_fn"]("job.sbatch"))
        self.assertTrue(code_args.kwargs["include_fn"]("launch.sh"))
        self.assertFalse(code_args.kwargs["include_fn"]("results.json"))

    def test_code_logging_excludes_wandb_and_gitignored_paths(self):
        root = str(Path(__file__).resolve().parents[1])
        self.assertTrue(
            experiment_tracking._exclude_code(f"{root}/wandb/config.yaml", root)
        )

        with mock.patch(
            "experiment_tracking.subprocess.run",
            side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)],
        ) as check_ignore:
            self.assertTrue(
                experiment_tracking._exclude_code(f"{root}/analysis/report.py", root)
            )
            self.assertFalse(
                experiment_tracking._exclude_code(f"{root}/src/train.py", root)
            )

        first_command = check_ignore.call_args_list[0].args[0]
        self.assertEqual(first_command[-2:], ["--", "analysis/report.py"])
        self.assertIn("--no-index", first_command)

    def test_only_rank_zero_initializes_training(self):
        fake_wandb = SimpleNamespace(Settings=mock.Mock(), init=mock.Mock())
        with mock.patch.dict(os.environ, {"RANK": "1"}, clear=True), mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ):
            result = experiment_tracking.init_training_run("Toys_and_Games")

        self.assertIsNone(result)
        fake_wandb.init.assert_not_called()

    def test_callback_is_moved_after_log_enrichers(self):
        callback = object()
        trainer = SimpleNamespace(
            pop_callback=mock.Mock(return_value=callback),
            add_callback=mock.Mock(),
        )

        experiment_tracking.move_callback_to_end(trainer, "wandb-callback-type")

        trainer.pop_callback.assert_called_once_with("wandb-callback-type")
        trainer.add_callback.assert_called_once_with(callback)

    def test_logging_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"WANDB_DISABLED": "true"}, clear=True):
            self.assertEqual(experiment_tracking.wandb_report_to(), "none")
            self.assertIsNone(experiment_tracking.init_training_run("Books"))


class RankingMetricsTest(unittest.TestCase):
    def test_calculates_hit_rate_and_ndcg(self):
        with mock.patch(
            "pathlib.Path.open", mock.mock_open(read_data="A\t0\nB\t1\nC\t2\n")
        ):
            rows = [
                {"output": "A", "predict": ["A", "B"]},
                {"output": ["B"], "predict": ["C", "B"]},
            ]
            metrics = calculate_ranking_metrics(rows, "Books_5_test.txt")

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["unknown_predictions"], 0)
        self.assertEqual(metrics["hr@1"], 0.5)
        self.assertEqual(metrics["hr@3"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@3"], (1.0 + 1.0 / 1.5849625) / 2, places=6)

    def test_infers_dataset_from_standard_info_path(self):
        self.assertEqual(
            infer_dataset_name("data/CDs_and_Vinyl/info/items.txt"),
            "CDs_and_Vinyl",
        )


if __name__ == "__main__":
    unittest.main()
