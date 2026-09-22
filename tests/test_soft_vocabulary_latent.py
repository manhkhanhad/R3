import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from layers import LatentModel


def tiny_config(soft_vocabulary=True):
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    if soft_vocabulary:
        config.latent_soft_vocabulary = {
            "top_k": 3, "temperature": 0.7, "exclude_last_n": 1,
        }
    return config


class FixedAttention(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = nn.Parameter(latent)
        self.end_k = -1

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        return self.latent.unsqueeze(0).expand(hidden_states.size(0), -1)


def test_disabled_projection_is_exact_identity():
    model = LatentModel(tiny_config(soft_vocabulary=False))
    latent = torch.randn(2, 8)
    assert model.project_reasoning_latent(latent) is latent


def test_projection_matches_topk_vocabulary_mixture_and_excludes_thought():
    torch.manual_seed(0)
    model = LatentModel(tiny_config())
    latent = torch.randn(2, 8, requires_grad=True)

    # Even an artificially dominant final-token logit cannot select the
    # randomly initialized <|Thought|> entry because exclude_last_n=1.
    with torch.no_grad():
        model.lm_head.weight[-1].fill_(10_000)

    actual = model.project_reasoning_latent(latent)
    normalized = model.model.norm(latent)
    logits = model.lm_head(normalized)[:, :-1]
    values, ids = logits.topk(3, dim=-1)
    probabilities = F.softmax(values.float() / 0.7, dim=-1)
    expected = (
        probabilities.unsqueeze(-1)
        * F.embedding(ids, model.model.embed_tokens.weight).float()
    ).sum(dim=-2).to(latent.dtype)

    torch.testing.assert_close(actual, expected)
    assert (ids < 31).all()
    actual.sum().backward()
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0


def test_generate_embs_inserts_soft_latent_and_checkpoint_keeps_config(tmp_path):
    torch.manual_seed(1)
    model = LatentModel(tiny_config())
    raw = torch.randn(8)
    model.attention = FixedAttention(raw.clone())
    input_ids = torch.tensor([[1, 2, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)

    expected = model.project_reasoning_latent(raw.unsqueeze(0))[0]
    embeddings = model.generate_embs(input_ids, attention_mask)
    torch.testing.assert_close(embeddings[0, 2], expected)

    embeddings[0, 2].sum().backward()
    assert model.attention.latent.grad is not None
    assert model.attention.latent.grad.abs().sum() > 0

    # The standard ranking loader reconstructs LatentModel from config.json.
    model.save_pretrained(tmp_path)
    reloaded = LatentModel.from_pretrained(tmp_path)
    assert reloaded.config.latent_soft_vocabulary == {
        "top_k": 3, "temperature": 0.7, "exclude_last_n": 1,
    }
