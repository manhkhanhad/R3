import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

import scheduled_attention
from latent_attention_train_scheduled import AttentionScheduleCallback
from scheduled_attention import ScheduledLatentModel


def make_schedule_model(alpha=0.0, equal_three_way_mix=False):
    model = object.__new__(ScheduledLatentModel)
    model.alpha = alpha
    model.equal_three_way_mix = equal_three_way_mix
    return model


@pytest.mark.parametrize(
    ("draw", "expected"),
    ((0.1, "history_only"), (0.5, "reasoning_only"), (0.9, "both")),
)
def test_equal_three_way_draws_all_modes(monkeypatch, draw, expected):
    model = make_schedule_model(alpha=0.75, equal_three_way_mix=True)
    monkeypatch.setattr(scheduled_attention.random, "random", lambda: draw)
    assert model._draw_mode() == expected


def test_final_epoch_switch_preserves_two_epoch_alpha_ramp():
    model = SimpleNamespace(alpha=0.0, equal_three_way_mix=False)
    callback = AttentionScheduleCallback(
        ramp_epochs=2.0,
        alpha_max=0.75,
        three_way_start_epoch=2.0,
    )

    callback.on_step_begin(None, SimpleNamespace(epoch=1.0), None, model=model)
    assert model.alpha == pytest.approx(0.375)
    assert model.equal_three_way_mix is False

    callback.on_step_begin(None, SimpleNamespace(epoch=2.0), None, model=model)
    assert model.alpha == pytest.approx(0.75)
    assert model.equal_three_way_mix is True


def test_default_schedule_behavior_is_unchanged(monkeypatch):
    model = make_schedule_model(alpha=0.75, equal_three_way_mix=False)
    monkeypatch.setattr(scheduled_attention.random, "random", lambda: 0.8)
    assert model._draw_mode() == "history_only"

    model.alpha = 1.0
    assert model._draw_mode() == "both"
