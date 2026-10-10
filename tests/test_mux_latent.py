import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))

from mux_latent import CodiLatentLayer, MuxLatentModel, accumulation_share, category_segments, mux_kl, mux_table  # noqa: E402

VOCAB, HIDDEN = 40, 32


class CharTokenizer:
    """One token per word (ids from a fixed map) - enough to test the table."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]


def test_category_segments_drop_root_and_reverse():
    assert category_segments("Musical Instruments > A > B C") == ["B C", "A"]
    assert category_segments("Musical Instruments") == ["Musical Instruments"]


def test_mux_table_geometric_leaf_first_and_merged():
    tok = CharTokenizer()
    ids, w = mux_table(["Musical Instruments > x > y z", "Musical Instruments > a > a", "bad"], [True, True, False], tok, rho=0.5)
    # row 0: tokens y, z, x with weights 1, .5, .25 normalized
    torch.testing.assert_close(w[0, :3], torch.tensor([1.0, 0.5, 0.25]) / 1.75)
    assert [ids[0, k].item() for k in range(3)] == [tok.vocab["y"], tok.vocab["z"], tok.vocab["x"]]
    # row 1: repeated token merged into one entry with weight 1
    assert w[1, 0].item() == 1.0 and w[1, 1:].sum() == 0
    assert w[2].sum() == 0  # invalid row


def test_mux_kl_matches_definition():
    torch.manual_seed(0)
    logp = F.log_softmax(torch.randn(2, 10), -1)
    ids = torch.tensor([[3, 5, 0], [1, 0, 0]])
    w = torch.tensor([[0.6, 0.4, 0.0], [1.0, 0.0, 0.0]])
    target = torch.zeros(2, 10)
    target[0, 3], target[0, 5], target[1, 1] = 0.6, 0.4, 1.0
    expected = (target * (target.clamp_min(1e-12).log() - logp)).sum(-1)
    torch.testing.assert_close(mux_kl(logp, ids, w), expected)


def test_codi_layer_reads_position_before_thought():
    layer = CodiLatentLayer(HIDDEN)
    layer.reset(0)
    h = torch.randn(2, 5, HIDDEN)
    out = layer(h, torch.ones(2, 5), torch.tensor([3, 4]))
    torch.testing.assert_close(out, layer.norm(layer.mlp(h[[0, 1], [2, 3]])))


def tiny_model(builder, weight):
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         tie_word_embeddings=True)
    config.latent_mux = {"builder": builder, "weight": weight, "tau": 1.0}
    torch.manual_seed(0)
    model = MuxLatentModel(config)
    model.tie_weights()
    model.reset_latent_builder(0)
    model.set_mux_table(torch.tensor([[2, 3], [5, 0], [0, 0]]), torch.tensor([[0.7, 0.3], [1.0, 0.0], [0.0, 0.0]]))
    return model


def batch(n=2, length=6):
    ids = torch.randint(1, VOCAB - 1, (n, length))
    ids[:, 3] = VOCAB - 1  # thought token
    labels = ids.clone()
    labels[:, :4] = -100
    return ids, torch.ones_like(ids), labels


def test_forward_adds_mux_loss_for_both_builders_and_trains_the_builder():
    for builder in ("pool", "codi"):
        model = tiny_model(builder, 0.5)
        model.train()
        ids, mask, labels = batch()
        base = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        out = model(input_ids=ids, attention_mask=mask, labels=labels, mux_item_id=torch.tensor([0, 2]))
        assert torch.isfinite(out.loss) and out.loss > base  # only row 0 has a target
        out.loss.backward()
        grads = [p.grad for p in model.attention.parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads) > 0
        assert model.lm_head.weight is model.model.embed_tokens.weight  # tied: W is the input embedding


def test_mux_loss_off_and_accumulation_share():
    model = tiny_model("pool", 0.0)
    ids, mask, labels = batch()
    torch.manual_seed(1)
    a = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    b = model(input_ids=ids, attention_mask=mask, labels=labels, mux_item_id=torch.tensor([0, 1])).loss
    torch.testing.assert_close(a, b)
    tokens = int((labels[:, 1:] != -100).sum())
    share = accumulation_share({"labels": labels, "num_items_in_batch": 4 * tokens})
    assert math.isclose(float(share), 0.25)
