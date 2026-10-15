import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel
from collaborative_pair_features import build_pair_features
from latent_attention_train_collaborative import prefill_candidate_bank


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


def make_title_cf_model():
    model = make_model(bank_size=3)
    model.title_feature_dim = 2
    model.cf_topk = 2
    model.title_positive_floor = 0.25
    model.cf_projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.cf_projection.weight.copy_(torch.eye(2))
    model.register_buffer("bank_title", torch.zeros(3, 2, dtype=torch.float16), persistent=False)
    model.register_buffer("bank_item", torch.full((3,), -1, dtype=torch.int64), persistent=False)
    model.register_buffer(
        "cf_neighbor_ids", torch.tensor([[1, -1], [0, -1], [-1, -1]]), persistent=False,
    )
    return model


def test_title_and_item_features_follow_fifo_order():
    model = make_title_cf_model()
    for index in range(4):
        model._push_bank(
            torch.tensor([[float(index), 0.0]]), torch.tensor([index]),
            torch.tensor([index]), torch.tensor([index]), torch.tensor([0.0]),
            torch.tensor([True]), torch.tensor([[float(index), 1.0]]),
            torch.tensor([index]),
        )
    bank_title, bank_item = model._bank_aux_view()
    torch.testing.assert_close(bank_title.float()[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(bank_item, torch.tensor([1, 2, 3]))


def test_title_weighted_user_positives_change_loss():
    model = make_title_cf_model()
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    identifiers = (
        torch.tensor([7]), torch.tensor([8]), torch.tensor([9]),
        torch.tensor([7, 7, 10]), torch.tensor([11, 12, 13]),
        torch.tensor([14, 15, 16]),
    )
    uniform, _ = model._relation_loss(
        anchor, candidates, *identifiers, model.user_projection, "user", 16,
    )
    weighted, _ = model._relation_loss(
        anchor, candidates, *identifiers, model.user_projection, "user", 16,
        anchor_title=torch.tensor([[1.0, 0.0]]),
        candidate_title=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    assert weighted.item() < uniform.item()
    weighted.backward()
    assert anchor.grad is not None


def test_cf_neighbor_is_positive_across_users_and_not_a_negative():
    model = make_title_cf_model()
    latent = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], requires_grad=True)
    losses = model._collaborative_losses(
        latent, torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22]),
        torch.tensor([30, 31, 32]), torch.tensor([True, True, True]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1, 2]),
    )
    user_loss, target_loss, cf_loss, _, _, cf_rate = losses
    assert user_loss.item() == 0.0
    assert target_loss.item() == 0.0
    assert cf_loss.item() > 0.0
    assert cf_rate.item() > 0.0
    cf_loss.backward()
    assert latent.grad is not None


def test_cf_neighbors_are_built_only_from_training_interactions(tmp_path):
    frame = pd.DataFrame({
        "user_id": ["a", "a", "b", "b", "c"],
        "item_id": [1, 2, 1, 2, 3],
        "item_title": ["Electric Guitar", "Acoustic Guitar", "Electric Guitar",
                       "Acoustic Guitar", "Piano Keyboard"],
    })
    train_file = tmp_path / "train.csv"
    frame.to_csv(train_file, index=False)
    item_to_index, titles, neighbors, report = build_pair_features(
        train_file, title_dim=2, cf_topk=2, cf_min_common=2,
    )
    assert titles.shape[0] == 3
    assert neighbors[item_to_index[1], 0] == item_to_index[2]
    assert neighbors[item_to_index[3], 0] == -1
    assert report["cf_directed_edges"] == 2


def make_tiny_training_model():
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    config.collaborative_contrastive = {
        "projection_dim": 4, "bank_size": 3, "title_feature_dim": 2, "cf_topk": 2,
    }
    model = CollaborativeContrastiveLatentModel(config)
    model.cf_neighbor_ids = torch.tensor([[1, -1], [0, -1], [-1, -1]])
    model.positive_min_token_accuracy = 0.0
    return model


def tiny_batch():
    return {
        "input_ids": torch.tensor([
            [1, 2, 31, 3, 4], [1, 2, 31, 5, 6], [1, 2, 31, 7, 8],
        ]),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
        "labels": torch.tensor([
            [-100, -100, -100, 3, 4],
            [-100, -100, -100, 5, 6],
            [-100, -100, -100, 7, 8],
        ]),
        "user_hash": torch.tensor([10, 11, 12]),
        "target_hash": torch.tensor([20, 21, 22]),
        "sample_hash": torch.tensor([30, 31, 32]),
        "target_item_index": torch.tensor([0, 1, 2]),
        "target_title_features": torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        ]),
    }


def test_title_cf_training_forward_and_queue_prefill():
    model = make_tiny_training_model()
    batch = tiny_batch()
    rows = [{key: value[index] for key, value in batch.items()} for index in range(3)]
    seen, accepted = prefill_candidate_bank(
        model, rows, lambda group: {
            key: torch.stack([row[key] for row in group]) for key in group[0]
        }, batch_size=3, seed=42, min_token_accuracy=0.0,
    )
    assert seen == accepted == 3
    assert model.bank_count.item() == 3
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.cf_projection[1].weight.grad is not None


def make_diverse_model(bank_size=4, per_user=0, per_target=0, dedup=0.0, window=8, gate=0.0):
    model = make_model(bank_size=bank_size)
    model.bank_max_per_user = per_user
    model.bank_max_per_target = per_target
    model.bank_dedup_cos = dedup
    model.nll_gate_quantile = gate
    model.nll_gate_window = window
    model.register_buffer("bank_time", torch.zeros(bank_size, dtype=torch.int64), persistent=False)
    model.register_buffer("bank_valid", torch.zeros(bank_size, dtype=torch.bool), persistent=False)
    model.register_buffer("bank_clock", torch.zeros((), dtype=torch.int64), persistent=False)
    model.register_buffer("nll_history", torch.zeros(window), persistent=False)
    model.register_buffer("nll_history_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
    model.register_buffer("nll_history_count", torch.zeros((), dtype=torch.int64), persistent=False)
    return model


def push_rows(model, rows):
    """rows: (r0, r1, user, target, sample)."""
    rows = torch.tensor(rows, dtype=torch.float32)
    model._push_bank(
        rows[:, :2], rows[:, 2].long(), rows[:, 3].long(), rows[:, 4].long(),
        torch.zeros(len(rows)), torch.ones(len(rows), dtype=torch.bool),
    )


def test_diverse_queue_without_limits_matches_fifo():
    model = make_diverse_model(bank_size=3, per_target=100)  # active but never binding
    push_rows(model, [(1, 0, 1, 11, 1), (2, 0, 2, 12, 2)])
    push_rows(model, [(3, 0, 3, 13, 3), (4, 0, 4, 14, 4)])
    torch.testing.assert_close(model._bank_view()[0].float()[:, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_per_user_cap_replaces_that_users_oldest_entry():
    model = make_diverse_model(bank_size=4, per_user=2)
    push_rows(model, [(1, 0, 7, 11, 1), (1, 1, 8, 12, 2), (0, 1, 7, 13, 3)])
    push_rows(model, [(-1, 0, 7, 14, 4)])  # user 7 already holds 2: replace its oldest (sample 1)
    bank_r, bank_user, _, bank_sample, _ = model._bank_view()
    assert sorted(bank_sample.tolist()) == [2, 3, 4]
    assert (bank_user == 7).sum() == 2
    assert int(model.last_queue_replaced) == 1


def test_per_target_cap_replaces_that_targets_oldest_entry():
    model = make_diverse_model(bank_size=4, per_target=1)
    push_rows(model, [(1, 0, 1, 50, 1), (0, 1, 2, 60, 2)])
    push_rows(model, [(-1, 0, 3, 50, 3)])
    _, _, bank_target, bank_sample, _ = model._bank_view()
    assert sorted(bank_sample.tolist()) == [2, 3]
    assert (bank_target == 50).sum() == 1


def test_near_duplicate_of_same_user_refreshes_slot_but_other_users_do_not():
    model = make_diverse_model(bank_size=4, dedup=0.9)
    push_rows(model, [(1, 0, 7, 11, 1)])
    push_rows(model, [(1, 0.1, 7, 12, 2)])   # cos ~0.995 with user 7's entry: replaces it
    push_rows(model, [(1, 0.1, 8, 13, 3)])   # same vector, different user: added
    push_rows(model, [(0, 1, 7, 14, 4)])     # user 7 but dissimilar: added
    _, bank_user, _, bank_sample, _ = model._bank_view()
    assert bank_sample.tolist() == [2, 3, 4]
    assert bank_user.tolist() == [7, 8, 7]


def test_full_diverse_queue_evicts_globally_oldest():
    model = make_diverse_model(bank_size=2, per_user=5)
    push_rows(model, [(1, 0, 1, 11, 1), (2, 0, 2, 12, 2)])
    push_rows(model, [(3, 0, 3, 13, 3)])
    assert model._bank_view()[3].tolist() == [2, 3]


def test_relative_nll_gate_keeps_best_quantile_of_recent_samples():
    model = make_diverse_model(window=8, gate=0.25)
    first = model._nll_gate(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    assert first.tolist() == [False, False, False, True]   # 25% quantile of {1,2,3,4} = 1.75
    second = model._nll_gate(torch.tensor([0.5, 5.0, 6.0, 7.0]))
    # window now {4,3,2,1,0.5,5,6,7}: 25% quantile = 1.75, so only 0.5 passes
    assert second.tolist() == [True, False, False, False]
    third = model._nll_gate(torch.tensor([9.0, 9.0, 9.0, 9.0]))  # oldest four evicted
    assert third.tolist() == [False, False, False, False]
    torch.testing.assert_close(model.last_nll_gate_threshold, torch.quantile(
        torch.tensor([0.5, 5.0, 6.0, 7.0, 9.0, 9.0, 9.0, 9.0]), 0.25))


def test_training_forward_with_nll_gate_and_diverse_queue():
    model = make_tiny_training_model()
    model.positive_min_token_accuracy = 0.0
    model.nll_gate_quantile = 0.25
    model.bank_max_per_user, model.bank_max_per_target, model.bank_dedup_cos = 2, 4, 0.9
    batch = tiny_batch()
    for _ in range(3):
        output = model(**batch)
        assert torch.isfinite(output.loss)
        output.loss.backward()
    assert model.last_nll_gate_threshold is not None
    assert 0 < int(model.bank_count) <= 3 * 3
    assert int(model.bank_count) == int(model.bank_valid.sum())
    assert 0.0 < float(model.last_queue_user_diversity) <= 1.0
    users = model._bank_view()[1]
    assert int(torch.unique(users, return_counts=True)[1].max()) <= 2
