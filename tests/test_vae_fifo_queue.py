import sys
from pathlib import Path

import torch
import torch.nn as nn

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from vae_decomposed_attention import VAEDecomposedLatentModel


def make_queue_model(queue_size=4, min_queue_size=0):
    model = object.__new__(VAEDecomposedLatentModel)
    nn.Module.__init__(model)
    model.z_queue_size = queue_size
    model.kl_min_queue_size = min_queue_size
    model.last_z_mean_sq = None
    model.last_z_var = None
    model.last_kl_active = False
    model.register_buffer("z_queue", torch.zeros(0, 1), persistent=False)
    return model


def test_z_queue_evicts_oldest_rows_first():
    model = make_queue_model(queue_size=3)

    model._push_queue(torch.tensor([[1.0], [2.0]]))
    model._push_queue(torch.tensor([[3.0], [4.0]]))

    torch.testing.assert_close(model.z_queue, torch.tensor([[2.0], [3.0], [4.0]]))


def test_kl_waits_until_fifo_queue_is_full():
    model = make_queue_model(queue_size=4, min_queue_size=4)

    first = model._queued_kl(torch.tensor([[2.0], [2.0]], requires_grad=True))
    second = model._queued_kl(torch.tensor([[2.0], [2.0]], requires_grad=True))
    third_z = torch.tensor([[2.0]], requires_grad=True)
    third = model._queued_kl(third_z)

    assert first.item() == 0.0
    assert second.item() == 0.0
    assert model.last_kl_active
    assert third.item() > 0.0
    third.backward()
    assert third_z.grad is not None
    torch.testing.assert_close(model.z_queue, torch.full((4, 1), 2.0))
