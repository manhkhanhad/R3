import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "grpo_attention_tuning"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

collab_grpo = pytest.importorskip("collab_grpo")


def trainer(step, active, steps=20, need=8):
    t = object.__new__(collab_grpo.CollaborativeGRPOTrainer)
    t.calibration_steps, t.target_ratio, t.min_active_micro_batches = steps, 0.3, need
    t.collab_lambda = None
    t._grad_norms = {"grpo": [1.0] * active, "infonce": [2.0] * active}
    t._calibration_wait_logged = False
    t.state = SimpleNamespace(global_step=step)
    return t


def test_no_positives_keeps_lambda_zero_instead_of_crashing(capsys):
    t = trainer(step=20, active=0)
    t._maybe_finish_calibration()
    assert t.collab_lambda is None
    assert "lambda stays 0" in capsys.readouterr().out
    t._maybe_finish_calibration()                    # logged only once
    assert capsys.readouterr().out == ""


def test_calibrates_once_enough_active_micro_batches():
    t = trainer(step=25, active=8)
    t._maybe_finish_calibration()
    assert t.collab_lambda == pytest.approx(0.3 * 1.0 / 2.0)


def test_waits_for_calibration_steps_even_with_positives():
    t = trainer(step=5, active=50)
    t._maybe_finish_calibration()
    assert t.collab_lambda is None
