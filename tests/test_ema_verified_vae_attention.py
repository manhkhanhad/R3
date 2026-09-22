import sys
from pathlib import Path

import torch
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from ema_verified_vae_attention import EMAVerifiedVAEDecomposedLatentModel
from layers import LatentModel


def make_model():
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    config.latent_decomposition = {
        "kind": "vae", "rho_mu": 1.0, "ema_momentum": 0.99,
    }
    config.latent_ema_verification = {
        "verify_weight": 0.05,
        "teacher_ema_momentum": 0.999,
        "history_blind": True,
        "shared_scaffold": "mu_ema",
    }
    model = EMAVerifiedVAEDecomposedLatentModel(config)
    model.kl_weight = 0.1
    model.mu_consistency_weight = 0.5
    model.verify_weight = 0.05
    model.grad_accum_steps = 1
    model.initialize_ema_teacher()
    return model


def test_verifier_gradient_reaches_z_but_not_teacher_parameters():
    torch.manual_seed(0)
    model = make_model()
    model.train()
    input_ids = torch.tensor([[1, 2, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    thought_idx = torch.tensor([2])
    z = torch.randn(1, 8, requires_grad=True)

    loss = model.verification_loss(
        input_ids, attention_mask, labels, thought_idx, z
    )
    loss.backward()

    assert z.grad is not None
    assert z.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.ema_teacher.parameters())
    assert model.ema_teacher.training is False


def test_verifier_is_invariant_to_masked_history_tokens():
    torch.manual_seed(1)
    model = make_model()
    model.eval()
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    thought_idx = torch.tensor([2])
    z = torch.randn(1, 8)

    first = model.verification_loss(
        torch.tensor([[1, 2, 31, 4, 5]]),
        attention_mask, labels, thought_idx, z,
    )
    second = model.verification_loss(
        torch.tensor([[9, 10, 31, 4, 5]]),
        attention_mask, labels, thought_idx, z,
    )
    torch.testing.assert_close(first, second, atol=1e-6, rtol=0)


def test_ema_update_and_checkpoint_exclusion(tmp_path):
    torch.manual_seed(2)
    model = make_model()
    model.teacher_ema_momentum = 0.5
    student_parameter = dict(model.named_parameters())["model.embed_tokens.weight"]
    teacher_parameter = dict(model.ema_teacher.named_parameters())[
        "model.embed_tokens.weight"
    ]
    before = teacher_parameter.detach().clone()
    with torch.no_grad():
        student_parameter.add_(2.0)

    model.update_ema_teacher()
    torch.testing.assert_close(teacher_parameter, before + 1.0)
    assert model.ema_teacher_updates == 1
    assert not any(key.startswith("ema_teacher.") for key in model.state_dict())

    model.save_pretrained(tmp_path)
    reloaded = LatentModel.from_pretrained(tmp_path)
    assert reloaded.config.latent_ema_verification["history_blind"] is True
    assert reloaded.config.latent_decomposition["kind"] == "vae"


def test_full_training_forward_uses_both_path_and_backpropagates():
    torch.manual_seed(3)
    model = make_model()
    model.train()
    model._draw_mode = lambda: (_ for _ in ()).throw(
        AssertionError("7c student must not draw a curriculum mode")
    )
    input_ids = torch.tensor([[1, 2, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])

    output = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    output.loss.backward()

    assert model.last_schedule_mode == "both"
    assert model.last_verify_loss is not None
    assert model.last_verify_loss.item() > 0
    assert model.attention.z_head.weight.grad is not None
    assert model.attention.z_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.ema_teacher.parameters())
