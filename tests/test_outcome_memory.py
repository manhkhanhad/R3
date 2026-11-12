import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'latent'))

from layers import LatentModel
from latent_attention_train_projected_correction import PairedCorrectionModel
from outcome_memory import OutcomeMemoryDataset, OutcomeMemoryLayer, history_cutoff, user_key
from projected_correction_training_utils import build_bank


class FixedAttention(nn.Module):
    def forward(self, hidden_states, attention_mask, thought_id_idx):
        return hidden_states.new_tensor([[2., 0.]]).expand(len(hidden_states), -1)


def make_layer():
    spec = dict(outcome_grounded=True, bank_size=4, n_items=4, item_dim=2,
                topn=4, projection_dim=2, key_dim=2, value_hidden=4,
                user_weight=1., max_delta_ratio=.1)
    layer = OutcomeMemoryLayer(FixedAttention(), 2, spec)
    layer.reset_new_parameters()
    raw = torch.tensor([[2., 0.], [0., 2.], [1., 1.], [-2., 0.]])
    projection = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2, bias=False))
    layer.initialize_from_attempt10(projection, projection, raw)
    layer.bank_time.copy_(torch.tensor([10, 20, 30, 40]))
    layer.bank_user.copy_(torch.tensor([1, 2, 3, 4]))
    layer.bank_item.copy_(torch.arange(4))
    layer.item_emb.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    layer.query_time = torch.tensor([30])
    layer.query_user = torch.tensor([1])
    return layer


def test_retrieval_excludes_self_user_ties_and_future():
    layer = make_layer()
    for mode in ('knn', 'random'):
        index, valid = layer._select(torch.tensor([[2., 0.]]), mode)
        assert index[valid].tolist() == [1]
    with torch.no_grad():
        layer.bank_item.copy_(torch.tensor([3, 2, 1, 0]))
    # Donor outcomes do not affect retrieval.
    index, valid = layer._select(torch.tensor([[2., 0.]]), 'knn')
    assert index[valid].tolist() == [1]
    layer.query_user = None
    with pytest.raises(ValueError, match='requires query_time and query_user'):
        layer.refine(torch.tensor([[2., 0.]]))


def test_no_eligible_memory_is_identity_with_finite_gradients():
    layer = make_layer()
    layer.query_time = torch.tensor([10])
    with torch.no_grad():
        layer.value[-1].weight.fill_(1.)
        layer.value[-1].bias.fill_(1.)
    raw = torch.tensor([[2., 0.]], requires_grad=True)
    out = layer.refine(raw)
    torch.testing.assert_close(out, raw)
    out.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_outcomes_change_values_not_keys_and_controls_are_effective(monkeypatch):
    layer = make_layer()
    with torch.no_grad():
        layer.value[0].weight.fill_(1.)
        layer.value[1].weight.copy_(torch.eye(4))
        layer.value[-1].weight.copy_(torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]]))
    raw = torch.tensor([[2., 0.]])
    index_before, valid_before = layer._select(raw, 'knn')
    with_outcome = layer.refine(raw)
    with torch.no_grad():
        layer.item_emb[1].copy_(torch.tensor([1., 0.]))
    changed = layer.refine(raw)
    assert not torch.allclose(with_outcome, changed)
    index_after, valid_after = layer._select(raw, 'knn')
    assert torch.equal(index_before, index_after)
    assert torch.equal(valid_before, valid_after)
    monkeypatch.setenv('OUTCOME_MEMORY_VALUE_MODE', 'latent_only')
    latent_only = layer.refine(raw)
    with torch.no_grad():
        layer.item_emb.fill_(100.)
    torch.testing.assert_close(layer.refine(raw), latent_only)
    monkeypatch.setenv('NEIGHBOR_REFINEMENT_MODE', 'off')
    torch.testing.assert_close(layer.refine(raw), raw)
    assert (changed - raw).norm() <= .1 * raw.norm() + 1e-6


def test_metadata_uses_observed_history_and_expands_per_beam():
    row = pd.Series(dict(user_id='alice', history_timestamp='[10, 15]', timestamp=999))
    assert history_cutoff(row) == 15
    row['timestamp'] = 1000000
    assert history_cutoff(row) == 15
    assert user_key('alice') == user_key('alice') != user_key('bob')
    layer = make_layer()
    layer.query_time = torch.tensor([30, 15])
    layer.query_user = torch.tensor([1, 2])
    eligible = layer.eligible(6)
    assert torch.equal(eligible[:3], torch.tensor([[False, True, False, False]]).expand(3, -1))
    assert torch.equal(eligible[3:], torch.tensor([[True, False, False, False]]).expand(3, -1))
    with pytest.raises(ValueError, match='align'):
        layer.eligible(3)
    with pytest.raises(ValueError, match='nonempty'):
        history_cutoff({'history_timestamp': '[]'})


def tiny_model():
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config.collaborative_contrastive = dict(projection_dim=8, bank_size=2)
    model = PairedCorrectionModel(config)
    spec = dict(outcome_grounded=True, projected=True, bank_size=2, n_items=2,
                item_dim=4, topn=2, projection_dim=8, key_dim=8,
                value_hidden=8, max_delta_ratio=.1)
    layer = OutcomeMemoryLayer(model.attention, 16, spec)
    layer.reset_new_parameters()
    layer.initialize_from_attempt10(model.user_projection, model.target_projection, torch.randn(2, 16))
    layer.bank_time.copy_(torch.tensor([10, 20]))
    layer.bank_user.copy_(torch.tensor([1, 2]))
    layer.bank_item.copy_(torch.tensor([0, 1]))
    layer.item_emb.copy_(torch.randn(2, 4))
    model.attention = layer
    del model.user_projection, model.target_projection
    delattr(config, 'collaborative_contrastive')
    config.latent_neighbor_refinement = spec
    model.pair_weight = 0
    return model


def test_training_reload_and_beam_generation_share_strict_memory(tmp_path):
    model = tiny_model()
    ids = torch.tensor([[1, 4, 31, 5, 6], [1, 7, 31, 8, 9]])
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :3] = -100
    meta = dict(query_time=torch.tensor([30, 30]), query_user=torch.tensor([1, 2]))
    model.train()
    result = model(input_ids=ids, attention_mask=mask, labels=labels, **meta)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.attention.value[-1].weight.grad.abs().sum() > 0
    assert model.attention.query_time is None and model.attention.query_user is None
    model.eval()
    model.save_pretrained(tmp_path)
    loaded = LatentModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids, attention_mask=mask, **meta).logits,
                                   loaded(input_ids=ids, attention_mask=mask, **meta).logits)
        # Two prompts, three beams, multiple cached decoding steps.
        generated = loaded.generate(ids[:, :3], attention_mask=mask[:, :3], **meta,
                                    num_beams=3, num_return_sequences=3, max_new_tokens=3)
    assert generated.shape[0] == 6
    assert loaded.attention.query_time is None and loaded.attention.query_user is None
    with pytest.raises(ValueError, match='requires query_time and query_user'):
        loaded(input_ids=ids, attention_mask=mask)


def test_bank_rejects_misaligned_user_metadata(tmp_path):
    layer = make_layer()
    frame = pd.DataFrame(dict(user_id=['a', 'b', 'c', 'd'], item_id=[0, 1, 2, 3], timestamp=[10, 20, 30, 40]))
    train_path = tmp_path / 'train.csv'
    frame.to_csv(train_path, index=False)
    torch.save(torch.randn(4, 2), tmp_path / 'thoughts.pt')
    meta = pd.DataFrame(dict(sample_id=[0, 1, 2, 3], user_id=['wrong', 'b', 'c', 'd'], target_item=[0, 1, 2, 3]))
    meta.to_csv(tmp_path / 'metadata.csv', index=False)
    item_path = tmp_path / 'items.pt'
    torch.save({'embeddings': torch.randn(4, 2)}, item_path)
    with pytest.raises(ValueError, match='aligned'):
        build_bank(layer, str(tmp_path), str(train_path), str(item_path))
    meta['user_id'] = frame.user_id
    meta.to_csv(tmp_path / 'metadata.csv', index=False)
    build_bank(layer, str(tmp_path), str(train_path), str(item_path))
    assert layer.bank_user.tolist() == [user_key(u) for u in frame.user_id]


class CharacterTokenizer:
    bos_token_id = None
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def test_dataset_train_and_test_use_identical_history_metadata(tmp_path):
    path = tmp_path / 'samples.csv'
    pd.DataFrame([dict(user_id='alice', history_item_title='["Guitar"]',
                       history_item_id='[1]', history_timestamp='[15]',
                       item_id=2, item_title='Tuner', timestamp=999)]).to_csv(path, index=False)
    kwargs = dict(train_file=str(path), tokenizer=CharacterTokenizer(),
                  max_len=2048, category='music instruments', K=0)
    train = OutcomeMemoryDataset(**kwargs)[0]
    test = OutcomeMemoryDataset(**kwargs, test=True)[0]
    assert train['query_time'] == test['query_time'] == 15
    assert train['query_user'] == test['query_user'] == user_key('alice')
    assert 'labels' in train and 'labels' not in test
    assert test['input_ids'] == train['input_ids'][:len(test['input_ids'])]


def test_shuffled_outcomes_never_pull_in_ineligible_values(monkeypatch):
    layer = make_layer()
    layer.query_time = torch.tensor([35])
    raw = torch.tensor([[2., 0.]])
    seen = []
    hook = layer.value.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    monkeypatch.setenv('OUTCOME_MEMORY_VALUE_MODE', 'shuffle_outcome')
    layer.refine(raw)
    hook.remove()
    index, valid = layer._select(raw, 'knn')
    outcome_values = seen[0][0, valid[0], 2:]
    expected = layer.item_emb[layer.bank_item[index[0, valid[0]]]].float()
    assert torch.equal(outcome_values, expected.roll(1, dims=0))
    # Eligible donors are only rows 1 and 2; same-user row 0 and future row 3
    # never contribute an outcome to the valid attention slots.
    assert set(index[valid].tolist()) == {1, 2}


def test_history_retrieval_works_without_collaborative_heads():
    layer = make_layer()
    layer.retrieval_space = 'history'
    raw = torch.tensor([[2., 0.], [0., 2.], [1., 1.], [-2., 0.]])
    layer.initialize_history_bank(raw)
    index, valid = layer._select(torch.tensor([[2., 0.]]), 'knn')
    assert index[valid].tolist() == [1]
    assert all(not p.requires_grad for p in layer.user_projection.parameters())
    assert all(not p.requires_grad for p in layer.target_projection.parameters())
    torch.testing.assert_close(layer.bank_key.float(),
                               F.normalize(raw - raw.mean(0), dim=-1), atol=5e-4, rtol=5e-4)


def test_margin_phase_transfers_exactly_into_memory_phase(tmp_path):
    from margin_loss_attention import MarginLatentModel
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64)
    phase1 = MarginLatentModel(config)
    ids = torch.tensor([[1, 4, 31, 5, 6], [1, 7, 31, 8, 9]])
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :3] = -100
    phase1.train()
    result = phase1(input_ids=ids, attention_mask=mask, labels=labels)
    assert phase1.last_margin_loss > 0
    result.loss.backward()
    assert torch.isfinite(phase1.attention.value.weight.grad).all()
    phase1.eval().save_pretrained(tmp_path / 'phase1')
    phase2 = LatentModel.from_pretrained(tmp_path / 'phase1')
    spec = dict(outcome_grounded=True, projected=True, retrieval_space='history',
                bank_size=2, n_items=2, item_dim=4, topn=2, projection_dim=8,
                key_dim=8, value_hidden=8, max_delta_ratio=.1)
    layer = OutcomeMemoryLayer(phase2.attention, 16, spec)
    layer.reset_new_parameters()
    layer.initialize_history_bank(torch.randn(2, 16))
    layer.bank_time.copy_(torch.tensor([10, 20]))
    layer.bank_user.copy_(torch.tensor([1, 2]))
    layer.bank_item.copy_(torch.tensor([0, 1]))
    layer.item_emb.copy_(torch.randn(2, 4))
    phase2.attention = layer
    phase2.config.latent_neighbor_refinement = spec
    meta = dict(query_time=torch.tensor([30, 30]), query_user=torch.tensor([1, 2]))
    phase2.eval()
    with torch.no_grad():
        torch.testing.assert_close(phase1(input_ids=ids, attention_mask=mask).logits,
                                   phase2(input_ids=ids, attention_mask=mask, **meta).logits)
    phase2.train()
    out = phase2(input_ids=ids, attention_mask=mask, labels=labels, **meta)
    out.loss.backward()
    assert layer.value[-1].weight.grad.abs().sum() > 0
    assert layer.last_raw_norm > 0
    assert layer.last_gate_mean >= 0
    assert layer.last_delivered_refinement_ratio == 0
    phase2.eval().save_pretrained(tmp_path / 'phase2')
    reloaded = LatentModel.from_pretrained(tmp_path / 'phase2').eval()
    with torch.no_grad():
        torch.testing.assert_close(phase2(input_ids=ids, attention_mask=mask, **meta).logits,
                                   reloaded(input_ids=ids, attention_mask=mask, **meta).logits)


def test_margin_training_rejects_microbatch_one_before_loading():
    from latent_attention_train_margin import train
    with pytest.raises(ValueError, match='micro_batch_size >= 2'):
        train(base_model='must-not-be-loaded', micro_batch_size=1, margin_weight=1.)
