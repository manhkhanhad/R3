import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from collaborative_margin_attention import CollaborativeMarginLatentModel
from latent_attention_train_collaborative_margin import PhaseCallback


def make_model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    config.collaborative_contrastive = {"projection_dim": 4, "bank_size": 8}
    model = CollaborativeMarginLatentModel(config)
    model.positive_min_token_accuracy = 0.0
    model.train()
    return model


def batch(users, targets, samples):
    # Token 31 is the thought token (last embedding row).
    input_ids = torch.tensor([[1, 2, 31, 3, 4]] * len(users))
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=torch.tensor([[-100, -100, -100, 3, 4]] * len(users)),
        user_hash=torch.tensor(users),
        target_hash=torch.tensor(targets),
        sample_hash=torch.tensor(samples),
    )


def test_margin_phase_fills_queue_without_contrastive_gradient():
    model = make_model()
    out = model(**batch([1, 2], [5, 6], [0, 1]))
    assert model.bank_count.item() == 2
    assert model.last_user_contrastive_loss.item() == 0.0
    assert model.last_target_contrastive_loss.item() == 0.0
    out.loss.backward()
    assert model.user_projection[1].weight.grad is None


def test_contrastive_phase_uses_queue_and_keeps_margin():
    model = make_model()
    model(**batch([1, 2], [5, 6], [0, 1]))
    model.zero_grad()
    model.contrastive_enabled = True
    out = model(**batch([1, 2], [7, 8], [2, 3]))
    assert model.last_user_contrastive_loss.item() > 0.0
    assert model.last_margin_loss is not None
    out.loss.backward()
    assert model.user_projection[1].weight.grad.abs().sum() > 0


def test_margin_branch_changes_loss():
    model = make_model()
    model.margin_weight = 0.0
    no_margin = model(**batch([1, 2], [5, 6], [0, 1])).loss.item()
    model.margin_weight = 1.0
    with_margin = model(**batch([1, 2], [5, 6], [0, 1])).loss.item()
    # hinge >= margin when neg and pos share inputs up to r; never negative.
    assert with_margin >= no_margin
    assert model.last_margin_loss.item() >= 0.0


def test_phase_callback_switches_after_first_epoch():
    model = make_model()
    callback = PhaseCallback(margin_only_epochs=1, phase2_margin_weight=0.5)
    for epoch, expected in ((0.0, False), (0.99, False), (1.0, True), (2.5, True)):
        callback.on_step_begin(None, SimpleNamespace(epoch=epoch, global_step=0), None, model=model)
        assert model.contrastive_enabled is expected
    assert model.margin_weight == 0.5
