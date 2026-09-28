import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "latent"))

from projected_neighbor_refinement import ProjectedNeighborRefinementLayer
from latent_attention_train_projected_correction import PairedCorrectionModel
from layers import LatentModel


class FixedAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        return self.value[None].expand(hidden_states.size(0), -1)


def make_layer():
    spec = {"bank_size": 2, "n_items": 2, "item_dim": 2, "topn": 1,
            "key_dim": 2, "value_hidden": 4, "projection_dim": 2,
            "user_weight": 1.0, "max_delta_ratio": 0.1}
    layer = ProjectedNeighborRefinementLayer(FixedAttention([2.0, 0.0]), 2, spec)
    layer.reset_new_parameters()
    raw = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    user = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2, bias=False))
    target = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        user[1].weight.copy_(torch.eye(2))
        target[1].weight.copy_(torch.eye(2))
    layer.initialize_from_attempt10(user, target, raw)
    layer.bank_key.copy_(F.normalize(raw.half(), dim=-1))
    layer.bank_item.copy_(torch.tensor([0, 1]))
    layer.bank_time.copy_(torch.tensor([10, 20]))
    return layer


def test_projected_retrieval_respects_training_time():
    layer = make_layer()
    query = torch.tensor([[2.0, 0.0]])
    index, valid = layer._select(query, "knn")
    assert index.item() == 0 and valid.item()
    layer.query_time = torch.tensor([10])
    _, valid = layer._select(query, "knn")
    assert not valid.item()


def test_correction_starts_at_checkpoint_and_stays_bounded():
    layer = make_layer()
    args = (torch.zeros(1, 1, 2), torch.ones(1, 1), torch.tensor([0]))
    raw = layer.base_attention(*args)
    torch.testing.assert_close(layer(*args), raw)
    with torch.no_grad():
        layer.value[-1].bias.fill_(10.0)
    corrected = layer(*args)
    assert (corrected - raw).norm().item() > 0
    assert (corrected - raw).norm().item() <= 0.1 * raw.norm().item() + 1e-6
    assert torch.equal(layer.bank_raw.float(), torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_paired_training_and_plain_model_inference_share_corrected_path():
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64)
    config.collaborative_contrastive = {"projection_dim": 8, "bank_size": 2}
    model = PairedCorrectionModel(config)
    spec = {"projected": True, "bank_size": 2, "n_items": 2, "item_dim": 4,
            "topn": 1, "projection_dim": 8, "key_dim": 8,
            "value_hidden": 8, "max_delta_ratio": 0.1}
    layer = ProjectedNeighborRefinementLayer(model.attention, 16, spec)
    layer.reset_new_parameters()
    layer.initialize_from_attempt10(model.user_projection, model.target_projection,
                                    torch.randn(2, 16))
    layer.bank_key.copy_(F.normalize(layer.bank_raw, dim=-1))
    layer.bank_time.copy_(torch.tensor([1, 2]))
    layer.bank_mean.copy_(layer.bank_raw.float().mean(0))
    model.attention = layer
    del model.user_projection, model.target_projection
    delattr(config, "collaborative_contrastive")
    config.latent_neighbor_refinement = spec
    model.train()
    ids = torch.tensor([[1, 2, 31, 3, 4], [5, 6, 31, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, 3, 4], [-100, -100, -100, 7, 8]])
    mask = torch.ones_like(ids)
    for _ in range(2):  # mean and shuffled controls both execute
        result = model(input_ids=ids, attention_mask=mask, labels=labels,
                       query_time=torch.tensor([3, 3]))
        assert torch.isfinite(result.loss)
        result.loss.backward()
        model.zero_grad(set_to_none=True)
    model.eval()
    loaded = LatentModel(config)
    loaded.load_state_dict(model.state_dict(), strict=True)
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.generate_embs(ids, mask), loaded.generate_embs(ids, mask))
