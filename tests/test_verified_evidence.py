import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))

from verified_evidence_latent import VerifiedEvidenceLatentModel, evidence_component, oracle_mix_rate  # noqa: E402

VOCAB, HIDDEN, DIM, K = 40, 32, 8, 3


def tiny_model(gate="entropy", mix_start=0.0, mix_examples=0):
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    config.latent_side_info = {"kind": "oracle_category", "dimension": DIM, "init_norm": 30.0}
    config.latent_evidence = {"k": K, "gate": gate, "weight": 0.5, "label_smoothing": 0.0,
                              "oracle_mix_start": mix_start, "oracle_mix_examples": mix_examples, "verifier_width": 16}
    torch.manual_seed(0)
    model = VerifiedEvidenceLatentModel(config)
    model.reset_verifier(0)
    centroids = torch.randn(K, DIM)
    cid = torch.tensor([0, 0, 1, 2, 2, -1])
    table = {"embeddings": torch.stack([centroids[c] if c >= 0 else torch.zeros(DIM) for c in cid]),
             "cluster_id": cid, "valid": cid >= 0}
    model.set_evidence_table(table)
    return model, centroids


def batch(n=2, length=6):
    ids = torch.randint(1, VOCAB - 1, (n, length))
    ids[:, 3] = VOCAB - 1  # thought token
    labels = ids.clone()
    labels[:, :4] = -100
    return ids, torch.ones_like(ids), labels


def test_evidence_component_gate():
    ev = torch.nn.functional.normalize(torch.randn(K, HIDDEN), dim=-1)
    onehot = torch.tensor([[0.0, 1.0, 0.0]])
    comp, conf = evidence_component(onehot, ev, 30.0)
    torch.testing.assert_close(conf, torch.ones(1))
    torch.testing.assert_close(comp[0], 30.0 * ev[1])  # confident: exactly the evidence the oracle decoder reads
    uniform = torch.full((1, K), 1 / K)
    comp, conf = evidence_component(uniform, ev, 30.0)
    assert conf.abs().max() < 1e-6 and comp.norm() < 1e-4  # uncertain: nothing in the slot
    _, conf = evidence_component(uniform, ev, 30.0, gate="none")
    torch.testing.assert_close(conf, torch.ones(1))
    with pytest.raises(ValueError):
        evidence_component(uniform, ev, 30.0, gate="bad")


def test_oracle_mix_rate_decays_to_zero():
    assert oracle_mix_rate(0.5, 0, 100) == 0.5
    assert math.isclose(oracle_mix_rate(0.5, 50, 100), 0.25)
    assert oracle_mix_rate(0.5, 150, 100) == 0.0
    assert oracle_mix_rate(0.5, 0, 0) == 0.0


def test_evidence_table_centroids_and_labels():
    model, centroids = tiny_model()
    torch.testing.assert_close(model.evidence_centroids, centroids)
    assert model.item_cluster.tolist() == [0, 0, 1, 2, 2, -1]
    assert model.evidence_vectors().shape == (K, HIDDEN)


def test_forward_adds_verifier_loss_and_trains_verifier():
    model, _ = tiny_model()
    model.train()
    ids, mask, labels = batch()
    base = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    out = model(input_ids=ids, attention_mask=mask, labels=labels, evidence_item_id=torch.tensor([2, 5]))
    assert torch.isfinite(out.loss) and out.loss > base - 1e-6  # item 5 has no cluster: one valid label
    out.loss.backward()
    assert model.verifier[0].weight.grad.abs().sum() > 0
    assert model.side_projection.weight.grad.abs().sum() > 0
    assert model._seen == 2


def test_oracle_mix_puts_true_evidence_in_slot():
    model, _ = tiny_model(mix_start=1.0, mix_examples=10 ** 6)
    model.train()
    seen = {}
    orig = model.combine
    model.combine = lambda latent, comp: seen.setdefault("comp", comp) is not None and orig(latent, comp)
    ids, mask, labels = batch()
    model(input_ids=ids, attention_mask=mask, labels=labels, evidence_item_id=torch.tensor([0, 3]))
    expected = 30.0 * model.evidence_vectors()[torch.tensor([0, 2])]
    torch.testing.assert_close(seen["comp"], expected, rtol=1e-4, atol=1e-4)
    assert model.last_evidence["conf"].shape == (2,)  # logged confidence is the verifier's own


def test_eval_mode_ignores_labels_for_the_slot_and_zero_override():
    model, _ = tiny_model(mix_start=1.0, mix_examples=10 ** 6)
    model.eval()
    ids, mask, _ = batch()
    with torch.no_grad():
        latent = torch.randn(2, HIDDEN)
        a = model.observe_reasoning_latent(latent)
        model.evidence_override = "zero"
        b = model.observe_reasoning_latent(latent)
    torch.testing.assert_close(b, latent)
    expected = latent + model.last_evidence["component"]  # last_evidence from the zero call is zero
    assert torch.allclose(b, expected)
    assert not torch.allclose(a, latent)  # the verifier's evidence (not the oracle) is added in eval


def test_verifier_loss_scaled_under_gradient_accumulation():
    model, _ = tiny_model()
    model.train()
    ids, mask, labels = batch()
    items = torch.tensor([0, 3])
    tokens = int((labels[:, 1:] != -100).sum())
    with torch.no_grad():
        aux = (model(input_ids=ids, attention_mask=mask, labels=labels, evidence_item_id=items).loss
               - model(input_ids=ids, attention_mask=mask, labels=labels).loss)
        total = 4 * tokens  # this micro-batch holds a quarter of the accumulated label tokens
        aux_acc = (model(input_ids=ids, attention_mask=mask, labels=labels, evidence_item_id=items,
                         num_items_in_batch=total).loss
                   - model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=total).loss)
    assert aux > 0
    torch.testing.assert_close(aux_acc, aux / 4, rtol=1e-4, atol=1e-5)
