import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from grounded_reasoning import GroundedReasoningLatentModel, GroundedReasoningLayer, history_cutoff
from layers import LatentModel

N_ITEMS, BANK = 6, 4


def make_model():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2)
    config.latent_grounded_reasoning = {
        "topn": 3, "explore": 1, "key_dim": 8, "value_hidden": 16, "queue_size": 8,
        "n_items": N_ITEMS, "item_dim": 4, "bank_size": BANK,
    }
    model = GroundedReasoningLatentModel(config)
    model.attention.reset_new_parameters()
    model.attention.item_emb.copy_(torch.nn.functional.normalize(torch.randn(N_ITEMS, 4), dim=-1).half())
    return model


def batch(users, items, query_times, outcome_times):
    ids = torch.tensor([[1, 2 + i, 31, 3, 4] for i in range(len(users))])  # 31 = thought token
    return dict(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        labels=torch.tensor([[-100, -100, -100, 3, 4]] * len(users)),
        query_user=torch.tensor(users), target_item=torch.tensor(items),
        query_time=torch.tensor(query_times), outcome_time=torch.tensor(outcome_times),
    )


def fill_queue(model):
    model.train()
    model(**batch([1, 2, 3, 4], [0, 1, 2, 3], [5, 5, 5, 5], [1, 2, 3, 4]))


def test_history_cutoff_uses_latest_history_event():
    assert history_cutoff({"history_timestamp": "[3, 9, 7]"}) == 9


def test_eligibility_excludes_same_user_and_future_outcomes():
    layer = make_model().attention
    layer.query_time, layer.query_user = torch.tensor([10]), torch.tensor([7])
    eligible = layer._eligible(1, torch.tensor([7, 8, 8, 9]), torch.tensor([1, 1, 10, 11]))
    assert eligible.tolist() == [[False, True, False, False]]


def test_metadata_expands_over_beams():
    layer = make_model().attention
    layer.query_time, layer.query_user = torch.tensor([10, 20]), torch.tensor([1, 2])
    eligible = layer._eligible(4, torch.tensor([3]), torch.tensor([15]))
    assert eligible.flatten().tolist() == [False, False, True, True]


def test_training_fills_queue_and_trains_every_new_module():
    model = make_model()
    fill_queue(model)
    assert model.attention.queue_count.item() == 4
    model.zero_grad()
    out = model(**batch([5, 6], [1, 2], [9, 9], [8, 8]))
    assert torch.isfinite(out.loss)
    assert {"item_loss", "retrieval_loss"} <= set(model.last_losses)
    out.loss.backward()
    layer = model.attention
    for module in (layer.query, layer.key, layer.value, layer.item_head):
        assert module[1].weight.grad.abs().sum() > 0
    assert layer.evidence_scale.grad is not None
    assert model.attention.queue_count.item() == 6


def test_no_eligible_donor_is_identity():
    model = make_model()
    fill_queue(model)
    model.eval()
    layer = model.attention
    layer.query_time, layer.query_user = torch.tensor([0]), torch.tensor([9])
    r0 = torch.randn(1, 16)
    assert torch.equal(layer.refine(r0), r0)


def test_controls_change_the_grounded_latent(monkeypatch):
    model = make_model()
    fill_queue(model)
    model.eval()
    layer = model.attention
    layer.query_time, layer.query_user = torch.tensor([9]), torch.tensor([9])
    r0 = torch.randn(1, 16)
    results = {}
    for mode in ("off", "knn", "latent_only", "shuffle_outcome"):
        monkeypatch.setenv("GROUNDING_MODE", mode)
        results[mode] = layer.refine(r0)
    assert torch.equal(results["off"], r0)
    assert not torch.allclose(results["knn"], r0)
    assert not torch.allclose(results["knn"], results["latent_only"])
    assert not torch.allclose(results["knn"], results["shuffle_outcome"])


def test_inference_bank_round_trips_through_latent_model(tmp_path):
    model = make_model()
    data = batch([1, 2, 3, 4], [0, 1, 2, 3], [5, 5, 5, 5], [1, 2, 3, 4])
    loader = [{k: v[i:i + 2] for k, v in data.items()} for i in (0, 2)]
    assert model.build_inference_bank(loader, torch.device("cpu")) == BANK
    model.save_pretrained(tmp_path)
    loaded = LatentModel.from_pretrained(tmp_path)
    assert isinstance(loaded.attention, GroundedReasoningLayer)
    assert bool(loaded.attention.bank_ready)
    torch.testing.assert_close(loaded.attention.bank_r, model.attention.bank_r)
    assert loaded.attention.bank_user.tolist() == [1, 2, 3, 4]
    loaded.eval()
    loaded.attention.query_time, loaded.attention.query_user = torch.tensor([9]), torch.tensor([1])
    ids = torch.tensor([[1, 2, 31]])
    out = loaded(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert torch.isfinite(out.logits).all()
    assert loaded.attention.stats["donor_rate"].item() == 1.0


def test_unknown_mode_is_rejected(monkeypatch):
    model = make_model()
    fill_queue(model)
    model.attention.query_time, model.attention.query_user = torch.tensor([9]), torch.tensor([9])
    monkeypatch.setenv("GROUNDING_MODE", "bogus")
    with pytest.raises(ValueError):
        model.attention.refine(torch.randn(1, 16))
