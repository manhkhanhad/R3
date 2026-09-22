import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from residual_correction_attention import ResidualCorrectionAttentionLayer
from layers import LatentModel


class FailingBase(nn.Module):
    """Fixed-mu mode must never call the source thought generator."""

    end_k = -1

    def forward(self, *args, **kwargs):
        raise AssertionError("base attention was called in fixed-mu mode")


def test_zero_init_preserves_mu_and_detaches_backbone():
    torch.manual_seed(0)
    layer = ResidualCorrectionAttentionLayer(
        FailingBase(), hidden_size=8, bottleneck_size=4, use_fixed_mu=True
    )
    mu = torch.arange(8, dtype=torch.float32)
    layer.set_fixed_mu(mu)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.long)
    thought_idx = torch.tensor([4, 4])

    output = layer(hidden, mask, thought_idx)
    torch.testing.assert_close(output, mu.expand_as(output))
    torch.testing.assert_close(layer.last_z, torch.zeros_like(layer.last_z))

    output.sum().backward()
    assert hidden.grad is None
    assert layer.residual_out.weight.grad is not None
    assert layer.residual_out.weight.grad.abs().sum() > 0


def test_fixed_mu_shape_is_validated():
    layer = ResidualCorrectionAttentionLayer(
        FailingBase(), hidden_size=8, bottleneck_size=4, use_fixed_mu=True
    )
    try:
        layer.set_fixed_mu(torch.zeros(7))
    except ValueError as error:
        assert "expected 8" in str(error)
    else:
        raise AssertionError("invalid fixed-mu shape was accepted")


def test_attempt9a_projects_residual_out_of_mu_subspace():
    layer = ResidualCorrectionAttentionLayer(
        FailingBase(), hidden_size=8, bottleneck_size=4,
        use_fixed_mu=True, orthogonal_to_mu=True,
    )
    mu = torch.zeros(8)
    mu[0] = 3.0
    layer.set_fixed_mu(mu)
    with torch.no_grad():
        layer.residual_out.weight.zero_()
        layer.residual_out.bias.fill_(1.0)

    hidden = torch.randn(2, 5, 8)
    output = layer(hidden, torch.ones(2, 5, dtype=torch.long), torch.tensor([4, 4]))
    z = output - mu
    torch.testing.assert_close(z[:, 0], torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(z @ mu, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(z[:, 1:], torch.ones(2, 7))


def test_attempt9_checkpoint_loads_through_both_model_paths(tmp_path):
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    config.latent_residual_correction = {
        "bottleneck_size": 4,
        "residual_scale": 1.0,
        "use_fixed_mu": True,
        "orthogonal_to_mu": True,
        "end_k": -1,
    }
    model = LatentModel(config)
    mu = torch.linspace(-1, 1, 8)
    model.attention.set_fixed_mu(mu)
    model.save_pretrained(tmp_path)

    reloaded = LatentModel.from_pretrained(tmp_path)
    torch.testing.assert_close(reloaded.attention.fixed_mu, mu)
    assert bool(reloaded.attention.fixed_mu_initialized)
    assert reloaded.attention.orthogonal_to_mu is True

    grpo_path = LATENT_DIR.parent / "grpo_attention_tuning" / "model.py"
    spec = importlib.util.spec_from_file_location("attempt9_grpo_model", grpo_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    e1_model = module.LatentModel.from_pretrained(tmp_path)
    torch.testing.assert_close(e1_model.attention.fixed_mu, mu)
    assert bool(e1_model.attention.fixed_mu_initialized)
    assert e1_model.attention.orthogonal_to_mu is True


def test_full_model_loss_trains_only_residual_path():
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    config.latent_residual_correction = {
        "bottleneck_size": 4,
        "residual_scale": 1.0,
        "use_fixed_mu": True,
        "end_k": -1,
    }
    model = LatentModel(config)
    model.attention.set_fixed_mu(torch.linspace(-1, 1, 8))
    model.requires_grad_(False)
    for parameter in model.attention.residual_parameters():
        parameter.requires_grad_(True)

    # LatentModel treats the final vocabulary id as <|Thought|>.
    input_ids = torch.tensor([[1, 2, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    output.loss.backward()

    assert model.attention.residual_out.weight.grad is not None
    assert model.attention.residual_out.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("attention.residual_")
    )
