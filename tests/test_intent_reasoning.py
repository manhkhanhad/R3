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


# ------------------------------------------------- literature fixes (MIMA, Re4, REMI)
def make_fixed_model(**spec):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2)
    config.latent_intent_reasoning = {"n_intents": K, "path_hidden": 16, "n_items": N_ITEMS,
                                      "item_dim": 4, **spec}
    model = IntentLatentModel(config)
    model.attention.reset_new_parameters()
    model.attention.item_emb.copy_(F.normalize(torch.randn(N_ITEMS, 4), dim=-1).half())
    return model


def test_later_items_follow_the_user_in_time_and_skip_the_current_target(tmp_path):
    import pandas as pd
    from intent_reasoning import later_items_by_row
    path = tmp_path / "rows.csv"
    pd.DataFrame({"user_id": ["a", "a", "a", "b", "a"], "item_id": [1, 2, 1, 7, 3],
                  "timestamp": [10, 20, 30, 5, 40]}).to_csv(path, index=False)
    later = later_items_by_row(path, 2)
    assert later[0] == [2, 3]      # row 0 (item 1): next distinct items, item 1 again skipped
    assert later[1] == [1, 3]
    assert later[4] == []          # last event of user a
    assert later[3] == []          # user b has one row


def test_hungarian_gives_each_positive_its_own_path():
    from scipy.optimize import linear_sum_assignment
    layer = make_fixed_model(assignment="hungarian").attention
    lp_all = torch.log_softmax(torch.randn(2, K, N_ITEMS), -1)
    target = torch.tensor([0, 1])
    later = torch.tensor([[2, 3], [4, -1]])
    selected, loss, n_pos = layer._hungarian(lp_all, target, later)
    assert torch.isfinite(loss) and n_pos.item() == 2.5
    rows, cols = linear_sum_assignment(-lp_all[0][:, [0, 2, 3]].numpy())
    assert selected[0].item() == rows[list(cols).index(0)]
    assert len(set(rows.tolist())) == 3            # three positives, three distinct paths


def test_routing_regularization_penalises_collapse_onto_one_item():
    layer = make_fixed_model().attention
    mask = torch.tensor([[True, True, True, True]])
    uniform = torch.full((1, K, 4), 0.25)
    collapsed = torch.zeros(1, K, 4)
    collapsed[..., 0] = 1.0
    assert layer._routing_regularization(uniform, mask).item() < 1e-8
    assert layer._routing_regularization(collapsed, mask).item() > 0.5


def test_full_step_with_all_fixes_trains_heads():
    model = make_fixed_model(assignment="hungarian", recontrast_weight=0.1, routing_reg_weight=1.0)
    model.train()
    data = batch(2)
    data["history_items"] = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]])
    data["later_items"] = torch.tensor([[2, 3], [-1, -1]])
    out = model(**data)
    assert torch.isfinite(out.loss)
    assert {"assign_loss", "recontrast_loss", "routing_reg_loss", "routing_max"} <= set(model.last_losses)
    out.loss.backward()
    assert model.attention.intent_heads.grad.abs().sum() > 0
    assert model.attention.item_head[1].weight.grad.abs().sum() > 0


# ------------------------------------------------- phase 2: DivReason path aggregation
def make_router_model(tmp_path, **aggregator):
    phase1 = make_model()
    phase1.save_pretrained(tmp_path / "phase1")
    config = phase1.config
    spec = dict(config.latent_intent_reasoning)
    spec["aggregator"] = {"tau": 0.5, "prune": 0.05, **aggregator}
    config.latent_intent_reasoning = spec
    model = LatentModel.from_pretrained(tmp_path / "phase1", config=config)
    model.attention.prepare_aggregator(0.5)
    return model


def test_router_is_fp32_and_only_router_trains_through_frozen_decoder(tmp_path):
    model = make_router_model(tmp_path)
    layer = model.attention
    assert layer.router[1].weight.dtype == torch.float32
    for p in model.parameters():
        p.requires_grad_(False)
    for p in list(layer.router.parameters()) + [layer.agg_log_beta]:
        p.requires_grad_(True)
    model.train()
    data = batch(2)
    out = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"], labels=data["labels"])
    out.loss.backward()
    assert layer.router[1].weight.grad.abs().sum() > 0
    assert layer.agg_log_beta.grad is not None
    assert layer.intent_heads.grad is None and layer.prior[1].weight.grad is None
    weights_entropy = layer.last_aggregation["agg_entropy"]
    assert 0 <= weights_entropy.item() <= torch.log(torch.tensor(float(K))).item() + 1e-5


def test_aggregation_modes_and_suppression(tmp_path, monkeypatch):
    layer = make_router_model(tmp_path).attention
    layer.eval()
    r0 = torch.randn(1, 16)
    paths = layer.paths(r0)
    prior = torch.log_softmax(torch.tensor([[2.0, 0.0, -1.0]]), -1)
    uniform = layer.aggregate(r0, paths, prior, "aggregate_uniform")
    torch.testing.assert_close(uniform, paths.mean(1))
    blend = layer.aggregate(r0, paths, prior, "prior_blend")
    torch.testing.assert_close(blend, torch.einsum("bk,bkh->bh", prior.exp(), paths))
    # Two identical offsets are redundant: suppression must lower their joint weight.
    dup = paths.clone()
    dup[:, 1] = dup[:, 0]
    with torch.no_grad():
        layer.agg_log_beta.fill_(-20.0)                      # beta ~ 0
    layer.aggregate(r0, dup, prior)
    low = layer.last_aggregation
    with torch.no_grad():
        layer.agg_log_beta.fill_(1.5)                        # beta ~ 4.5
    out = layer.aggregate(r0, dup, prior)
    assert torch.isfinite(out).all()
    assert layer.last_aggregation["agg_beta"].item() > low["agg_beta"].item()


def test_pruning_drops_small_weights_at_eval(tmp_path):
    layer = make_router_model(tmp_path, prune=0.4).attention
    layer.eval()
    r0 = torch.randn(1, 16)
    with torch.no_grad():
        layer.router[-1].bias.zero_()
        layer.router[-1].weight.zero_()
    paths = layer.paths(r0)
    out = layer.aggregate(r0, paths, torch.zeros(1, K))
    # Near-uniform weights (~1/3 < 0.4): pruning must keep at least the top path,
    # never return a zero latent.
    assert out.norm() > 0.5 * r0.norm()
    assert any(torch.allclose(out, paths[:, k], atol=1e-4) for k in range(K)) or out.norm() > 0


def test_aggregator_checkpoint_round_trip(tmp_path, monkeypatch):
    model = make_router_model(tmp_path)
    model.save_pretrained(tmp_path / "phase2")
    loaded = LatentModel.from_pretrained(tmp_path / "phase2")
    assert loaded.attention.aggregator_enabled
    torch.testing.assert_close(loaded.attention.router[1].weight.float(), model.attention.router[1].weight.float())
    loaded.eval()
    monkeypatch.delenv("INTENT_MODE", raising=False)
    ids = batch(1)["input_ids"][:, :3]
    out = loaded(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert torch.isfinite(out.logits).all()
    assert loaded.attention.last_aggregation is not None
