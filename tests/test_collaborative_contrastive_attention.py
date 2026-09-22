import sys
from pathlib import Path

import torch
import torch.nn as nn

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel


def make_model(bank_size=3, hidden=2):
    model = object.__new__(CollaborativeContrastiveLatentModel)
    nn.Module.__init__(model)
    model.bank_size = bank_size
    model.user_projection = nn.Linear(hidden, hidden, bias=False)
    model.target_projection = nn.Linear(hidden, hidden, bias=False)
    with torch.no_grad():
        model.user_projection.weight.copy_(torch.eye(hidden))
        model.target_projection.weight.copy_(torch.eye(hidden))
    model.temperature = 0.1
    model.max_user_positives = 16
    model.max_target_positives = 16
    model.num_negatives = 256
    model.register_buffer("bank_r", torch.zeros(bank_size, hidden, dtype=torch.float16), persistent=False)
    model.register_buffer("bank_user", torch.zeros(bank_size, dtype=torch.int64), persistent=False)
    model.register_buffer("bank_target", torch.zeros(bank_size, dtype=torch.int64), persistent=False)
    model.register_buffer("bank_sample", torch.zeros(bank_size, dtype=torch.int64), persistent=False)
    model.register_buffer("bank_nll", torch.zeros(bank_size), persistent=False)
    model.register_buffer("bank_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
    model.register_buffer("bank_count", torch.zeros((), dtype=torch.int64), persistent=False)
    return model


def push(model, values, quality=None):
    count = len(values)
    if quality is None:
        quality = torch.ones(count, dtype=torch.bool)
    model._push_bank(
        torch.tensor([[float(value), 0.0] for value in values]),
        torch.arange(count) + 10,
        torch.arange(count) + 20,
        torch.arange(count) + 30,
        torch.arange(count, dtype=torch.float32),
        quality,
    )


def test_fifo_retains_newest_entries_in_chronological_order():
    model = make_model(bank_size=3)
    push(model, [1, 2])
    push(model, [3, 4])
    bank_r, _, _, _, _ = model._bank_view()
    torch.testing.assert_close(bank_r.float()[:, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_only_correct_quality_latents_enter_corpus():
    model = make_model(bank_size=3)
    push(model, [1, 2, 3], quality=torch.tensor([False, True, False]))
    bank_r, _, _, _, _ = model._bank_view()
    assert model.bank_count.item() == 1
    assert bank_r[0, 0].item() == 2.0


def test_same_sample_is_not_accepted_as_a_positive():
    model = make_model()
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss, rate = model._relation_loss(
        anchor,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([7]), torch.tensor([8]), torch.tensor([9]),
        torch.tensor([7, 10]), torch.tensor([11, 12]), torch.tensor([9, 13]),
        model.user_projection, "user", 16,
    )
    assert loss.item() == 0.0
    assert rate.item() == 0.0


def test_info_nce_uses_valid_positive_and_unrelated_negative():
    model = make_model()
    anchor = torch.tensor([[0.5, 0.5]], requires_grad=True)
    loss, rate = model._relation_loss(
        anchor,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([7]), torch.tensor([8]), torch.tensor([9]),
        torch.tensor([7, 10]), torch.tensor([11, 12]), torch.tensor([14, 15]),
        model.user_projection, "user", 16,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert rate.item() == 1.0
    loss.backward()
    assert anchor.grad is not None


def test_quality_metric_requires_every_target_token_to_be_correct():
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[-100, 1, 2], [-100, 1, 2]])
    logits[0, 0, 1] = logits[0, 1, 2] = 5.0
    logits[1, 0, 1] = logits[1, 1, 3] = 5.0
    _, _, accuracy = CollaborativeContrastiveLatentModel._loss_and_quality(logits, labels)
    torch.testing.assert_close(accuracy, torch.tensor([1.0, 0.5]))
