"""Progress and runtime estimates for concurrent inference."""

import io
import threading
import unittest

from rationale_progress import ProgressBar, concurrent_results


class ProgressTest(unittest.TestCase):
    def test_fast_completion_is_reported_before_slow_first_submission(self):
        release = threading.Event()
        def work(index):
            if index == 0:
                if not release.wait(timeout=2):
                    raise RuntimeError("Slow request was not released")
                return "slow"
            return "fast"
        output = io.StringIO()
        results = concurrent_results(work, [0, 1], total=2, max_workers=2,
                                     label="generation", stream=output)
        try:
            self.assertEqual(next(results), "fast")
        finally:
            release.set()
        self.assertEqual(list(results), ["slow"])
        self.assertIn("2/2", output.getvalue())
        self.assertIn("100.0%", output.getvalue())

    def test_cached_results_do_not_seed_runtime_estimates(self):
        now = [0.0]
        progress = ProgressBar(10, "generation", 2, stream=io.StringIO(), clock=lambda: now[0])
        progress.update("cached", 0.001)
        self.assertIsNone(progress.estimate_remaining())
        now[0] = 10.0
        progress.update("ok", 20.0)
        self.assertAlmostEqual(progress.estimate_remaining(), 80.0)
        line = progress.format_line()
        self.assertIn("ETA 00:01:20", line)
        self.assertIn("est total 00:01:30", line)
        self.assertIn("cached=1", line)

    def test_log_output_has_heartbeat_and_no_carriage_returns(self):
        now = [0.0]
        output = io.StringIO()
        progress = ProgressBar(1, "judge", 1, stream=output, clock=lambda: now[0])
        progress.display(force=True)
        now[0] = 11.0
        progress.display()
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertIn("elapsed 00:00:11", output.getvalue())
        progress.update("error", 11.0)
        progress.display(force=True, final=True)
        self.assertEqual(progress.estimate_remaining(), 0.0)
        self.assertIn("errors=1", output.getvalue())
        self.assertNotIn("\r", output.getvalue())

    def test_empty_work_is_complete(self):
        output = io.StringIO()
        self.assertEqual(list(concurrent_results(lambda value: value, [], total=0,
                                                max_workers=2, label="generation", stream=output)), [])
        self.assertIn("100.0%", output.getvalue())
        self.assertIn("ETA 00:00:00", output.getvalue())


if __name__ == "__main__":
    unittest.main()
