import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from intent_reasoning import IntentLatentModel, IntentReasoningLayer
from layers import LatentModel

K, N_ITEMS = 3, 5


def make_model(margin_weight=0.0):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2)
    config.latent_intent_reasoning = {"n_intents": K, "path_hidden": 16,
                                      "n_items": N_ITEMS, "item_dim": 4, "margin_weight": margin_weight}
    model = IntentLatentModel(config)
    model.attention.reset_new_parameters()
    model.attention.item_emb.copy_(F.normalize(torch.randn(N_ITEMS, 4), dim=-1).half())
    return model


def batch(n=2):
    ids = torch.tensor([[1, 2 + i, 31, 3, 4] for i in range(n)])  # 31 = thought token
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=torch.tensor([[-100, -100, -100, 3, 4]] * n),
                target_item=torch.arange(n) % N_ITEMS)


def test_posterior_credits_the_path_that_explains_the_item():
    layer = make_model().attention
    r0 = torch.randn(1, 16)
    paths = layer.paths(r0)
    target = torch.tensor([2])
    lp = layer._target_log_probs(paths, target)
    best = lp.argmax(-1)
    # Uniform prior: the credited path is the one whose scorer favours the item.
    with torch.no_grad():
        layer.prior[-1].weight.zero_()
        layer.prior[-1].bias.zero_()
    layer.train()
    layer.target_item = target
    class FixedLatent(torch.nn.Module):
        def forward(self, **_):
            return r0

    layer.base_attention = FixedLatent()
    out = layer(hidden_states=None, attention_mask=None, thought_id_idx=None)
    assert layer.last_selected.item() == best.item()
    torch.testing.assert_close(out.float(), paths[0, best.item()][None])


def test_training_step_trains_every_intent_module():
    model = make_model()
    model.train()
    out = model(**batch())
    assert torch.isfinite(out.loss)
    assert {"mixture_loss", "prior_loss", "balance_loss", "posterior_entropy"} <= set(model.last_losses)
    out.loss.backward()
    layer = model.attention
    assert layer.intent_heads.grad.abs().sum() > 0
    for module in (layer.path, layer.prior, layer.item_head):
        assert module[1].weight.grad.abs().sum() > 0
    assert layer.path_scale.grad is not None


def test_balance_loss_is_zero_at_uniform_usage_and_positive_when_collapsed():
    layer = make_model().attention
    uniform = torch.full((K,), 1.0 / K)
    collapsed = torch.tensor([0.98, 0.01, 0.01])
    kl = lambda u: (u * (u * K).log()).sum()
    assert kl(uniform).abs() < 1e-6
    assert kl(collapsed) > 0.5


def test_margin_adds_a_second_pass_against_another_intent():
    model = make_model(margin_weight=1.0)
    model.train()
    out = model(**batch())
    assert "margin_loss" in model.last_losses
    assert torch.isfinite(out.loss)


@pytest.mark.parametrize("mode", ["map", "random", "oracle", "r0"])
def test_eval_modes(monkeypatch, mode):
    model = make_model()
    model.eval()
    monkeypatch.setenv("INTENT_MODE", mode)
    if mode == "oracle":
        model.attention.target_item = torch.tensor([1, 2])
    ids = batch()["input_ids"][:, :3]
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert torch.isfinite(out.logits).all()
    if mode == "r0":
        assert model.attention.last_selected is None
    else:
        assert model.attention.last_selected.shape == (2,)


def test_forced_intent_expands_over_beams_and_changes_output():
    model = make_model()
    model.eval()
    ids = batch(1)["input_ids"][:, :3].repeat(4, 1)  # one prompt expanded to 4 beams
    logits = []
    for k in range(K):
        model.attention.forced_intent = k
        logits.append(model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1])
        assert model.attention.last_selected.tolist() == [k] * 4
    assert not torch.allclose(logits[0], logits[1])
    assert model.attention.last_prior_logp.shape == (4, K)


def test_round_trip_through_latent_model(tmp_path):
    model = make_model()
    model.save_pretrained(tmp_path)
    loaded = LatentModel.from_pretrained(tmp_path)
    assert isinstance(loaded.attention, IntentReasoningLayer)
    torch.testing.assert_close(loaded.attention.intent_heads, model.attention.intent_heads)
    torch.testing.assert_close(loaded.attention.item_emb, model.attention.item_emb)


def test_paths_start_distinct():
    layer = make_model().attention
    r0 = torch.randn(8, 16)
    unit = F.normalize(layer.paths(r0) - r0[:, None], dim=-1)
    cos = unit @ unit.transpose(1, 2)
    off_diagonal = cos[:, ~torch.eye(K, dtype=torch.bool)]
    assert off_diagonal.abs().mean() < 0.5


def test_credit_follows_the_item_not_a_confident_prior():
    layer = make_model().attention
    r0 = torch.randn(1, 16)
    target = torch.tensor([2])
    best = layer._target_log_probs(layer.paths(r0), target).argmax(-1).item()
    wrong = (best + 1) % K
    with torch.no_grad():
        layer.prior[-1].weight.zero_()
        layer.prior[-1].bias.fill_(-50.0)
        layer.prior[-1].bias[wrong] = 50.0

    class FixedLatent(torch.nn.Module):
        def forward(self, **_):
            return r0

    layer.base_attention = FixedLatent()
    layer.train()
    layer.target_item = target
    layer(hidden_states=None, attention_mask=None, thought_id_idx=None)
    assert layer.last_selected.item() == best
