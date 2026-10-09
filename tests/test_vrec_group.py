import sys
from pathlib import Path

import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))

from vrec_group_latent import VRecGroupLatentModel  # noqa: E402

VOCAB, HIDDEN = 40, 32


def tiny_vrec(k=4):
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    config.latent_vrec = {"k": k, "weight": 0.5, "gate": "entropy", "label_smoothing": 0.0}
    torch.manual_seed(0)
    model = VRecGroupLatentModel(config)
    model.reset_verifier(0)
    model.set_group_table({"cluster_id": torch.tensor([0, 1, 2, 3, -1]), "valid": torch.ones(5, dtype=torch.bool)})
    return model


def batch(n=2, length=6):
    ids = torch.randint(1, VOCAB - 1, (n, length))
    ids[:, 3] = VOCAB - 1  # thought token
    labels = ids.clone()
    labels[:, :4] = -100
    return ids, torch.ones_like(ids), labels


def test_slot_keeps_r_when_uncertain_and_moves_to_prototype_when_confident():
    model = tiny_vrec()
    r = torch.randn(2, HIDDEN) * 50
    torch.testing.assert_close(model.slot_from_probs(r, torch.full((2, 4), 0.25)), r)
    onehot = torch.nn.functional.one_hot(torch.tensor([1, 3]), 4).float()
    w = torch.nn.functional.normalize(model.verifier.weight[[1, 3]].float(), dim=-1)
    torch.testing.assert_close(model.slot_from_probs(r, onehot), w * r.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(model.slot_from_probs(r, None), r)


def test_forward_trains_verifier_and_zero_override():
    model = tiny_vrec()
    model.train()
    ids, mask, labels = batch()
    base = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    out = model(input_ids=ids, attention_mask=mask, labels=labels, evidence_item_id=torch.tensor([1, 4]))
    assert torch.isfinite(out.loss) and out.loss > base - 1e-6
    out.loss.backward()
    assert model.verifier.weight.grad.abs().sum() > 0
    model.eval()
    model.evidence_override = "zero"
    latent = torch.randn(2, HIDDEN)
    torch.testing.assert_close(model.observe_reasoning_latent(latent), latent)
