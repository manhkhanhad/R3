import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import Qwen2Config

GRPO_DIR = Path(__file__).resolve().parents[1] / "src" / "grpo_attention_tuning"
sys.path.insert(0, str(GRPO_DIR))

from collab_grpo import (  # noqa: E402
    CollaborativeGRPOTrainer, build_collab_module, load_projection_heads, target_nll,
)
from model import LatentModel  # noqa: E402

HIDDEN, VOCAB = 32, 64
THOUGHT = VOCAB - 1


def tiny_model():
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    model = LatentModel(config)
    for p in model.model.parameters():
        p.requires_grad = False
    collab = build_collab_module(HIDDEN, projection_dim=8, bank_size=16, nll_gate_window=32)
    collab.nll_gate_quantile = 1.0          # admit everything in the test
    collab.bank_max_per_user, collab.bank_max_per_target, collab.bank_dedup_cos = 2, 4, 0.9
    model.collab = collab
    return model


def fake_trainer(model, calibration_steps=1, step=0):
    trainer = object.__new__(CollaborativeGRPOTrainer)
    nn.Module.__init__  # trainer is not a module; set only what compute_loss reads
    trainer.model = model
    trainer.beta, trainer.num_iterations, trainer.temperature = 0.01, 1, 1.0
    trainer.epsilon_low, trainer.epsilon_high = 0.2, 0.28
    trainer.control = SimpleNamespace(should_evaluate=False)
    trainer.state = SimpleNamespace(global_step=step)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    trainer.calibration_steps, trainer.target_ratio = calibration_steps, 0.3
    trainer.collab_lambda = None if calibration_steps > 0 else 0.3
    trainer._grad_norms = {"grpo": [], "infonce": []}
    return trainer


def group_inputs(model, user, target, sample, generations=4, seed=0):
    """One GRPO group: same prompt repeated, row 0 noise-free."""
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(1, VOCAB - 2, (1, 6), generator=g)
    prompt = torch.cat([prompt, torch.tensor([[THOUGHT]])], dim=1).repeat(generations, 1)
    completion = torch.randint(1, VOCAB - 2, (1, 3), generator=g).repeat(generations, 1)
    noise = torch.randn(generations, HIDDEN, generator=g)
    noise[0] = 0
    with torch.no_grad():
        ids = torch.cat([prompt, completion], 1)
        mask = torch.ones_like(ids)
        embeds = model.generate_embs(ids, mask)
        embeds[torch.arange(generations), prompt.size(1) - 1] += noise
        logits = model(inputs_embeds=embeds, attention_mask=mask).logits[:, -4:-1]
        logps = torch.log_softmax(logits.float(), -1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)
    return {
        "prompt_ids": prompt, "prompt_mask": torch.ones_like(prompt),
        "completion_ids": completion, "completion_mask": torch.ones_like(completion),
        "noise": noise, "old_per_token_logps": logps, "ref_per_token_logps": logps.clone(),
        "advantages": torch.linspace(-1, 1, generations),
        "user_hash": torch.full((generations,), user), "target_hash": torch.full((generations,), target),
        "sample_hash": torch.full((generations,), sample),
    }


def test_only_the_noise_free_row_enters_the_queue():
    model = tiny_model()
    trainer = fake_trainer(model)
    trainer.my_get_per_token_logps = lambda m, ids, emb, mask, k: CollaborativeGRPOTrainer.my_get_per_token_logps(
        trainer, m, ids, emb, mask, k)
    loss = trainer.compute_loss(model, group_inputs(model, user=7, target=11, sample=1))
    assert torch.isfinite(loss)
    assert int(model.collab.bank_count) == 1
    assert model.collab._bank_view()[3].tolist() == [1]


def test_infonce_reaches_attention_and_heads_once_lambda_is_set():
    model = tiny_model()
    trainer = fake_trainer(model, calibration_steps=0)
    trainer.my_get_per_token_logps = lambda m, ids, emb, mask, k: CollaborativeGRPOTrainer.my_get_per_token_logps(
        trainer, m, ids, emb, mask, k)
    # queue: another sample of user 7 (positive) and an unrelated sample (negative)
    trainer.compute_loss(model, group_inputs(model, user=7, target=12, sample=2, seed=1))
    trainer.compute_loss(model, group_inputs(model, user=9, target=13, sample=3, seed=2))
    model.zero_grad()
    loss = trainer.compute_loss(model, group_inputs(model, user=7, target=11, sample=1, seed=3))
    loss.backward()
    assert trainer._metrics["train"]["user_infonce"][-1] > 0
    assert model.attention.query.weight.grad is not None
    assert model.collab.user_projection[1].weight.grad.abs().sum() > 0


def test_lambda_is_zero_during_calibration_then_set_from_gradient_ratio():
    model = tiny_model()
    trainer = fake_trainer(model, calibration_steps=2, step=0)
    trainer.my_get_per_token_logps = lambda m, ids, emb, mask, k: CollaborativeGRPOTrainer.my_get_per_token_logps(
        trainer, m, ids, emb, mask, k)
    for seed, (user, target, sample) in enumerate([(7, 12, 2), (9, 13, 3), (7, 11, 1), (7, 14, 4)]):
        trainer.compute_loss(model, group_inputs(model, user, target, sample, seed=seed))
    assert trainer.collab_lambda is None
    assert set(trainer._metrics["train"]["collab_lambda"]) == {0.0}
    assert len(trainer._grad_norms["infonce"]) >= 1
    trainer.state.global_step = 2
    trainer.compute_loss(model, group_inputs(model, 9, 15, 5, seed=9))
    med = lambda xs: float(torch.tensor(xs).median())
    expected = 0.3 * med(trainer._grad_norms["grpo"]) / med(trainer._grad_norms["infonce"])
    assert trainer.collab_lambda == pytest.approx(expected)


def test_target_nll_matches_masked_mean():
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    torch.testing.assert_close(target_nll(logps, mask), torch.tensor([2.0, 0.5]))


def test_projection_heads_load_from_stage1_checkpoint(tmp_path):
    source = build_collab_module(HIDDEN, projection_dim=8, bank_size=4)
    for p in source.parameters():
        nn.init.normal_(p)
    tensors = {k: v.contiguous() for k, v in source.state_dict().items() if "projection" in k}
    tensors["attention.query.weight"] = torch.zeros(2, 2)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    target = build_collab_module(HIDDEN, projection_dim=8, bank_size=4, checkpoint=str(tmp_path))
    torch.testing.assert_close(target.user_projection[1].weight, source.user_projection[1].weight)
    torch.testing.assert_close(target.target_projection[0].bias, source.target_projection[0].bias)
    with pytest.raises(KeyError):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
        load_projection_heads(build_collab_module(HIDDEN, projection_dim=8, bank_size=4), str(tmp_path))
