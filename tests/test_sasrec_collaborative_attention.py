import sys
from pathlib import Path

import torch
import torch.nn as nn

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from sasrec_collaborative_attention import SASRecCollaborativeLatentModel


def make_model(bank_size=4, hidden=2, sasrec_dim=2, item_table=None):
    model = object.__new__(SASRecCollaborativeLatentModel)
    nn.Module.__init__(model)
    model.bank_size = bank_size
    model.sequence_projection = nn.Linear(hidden, hidden, bias=False)
    model.item_projection = nn.Linear(hidden, hidden, bias=False)
    with torch.no_grad():
        model.sequence_projection.weight.copy_(torch.eye(hidden))
        model.item_projection.weight.copy_(torch.eye(hidden))
    if item_table is None:
        item_table = torch.eye(sasrec_dim)
    model.register_buffer("sasrec_item_table", item_table.float(), persistent=False)
    for name, shape, dtype in (
        ("bank_r", (bank_size, hidden), torch.float16),
        ("bank_sequence", (bank_size, sasrec_dim), torch.float16),
        ("bank_item", (bank_size,), torch.int64),
        ("bank_user", (bank_size,), torch.int64),
        ("bank_sample", (bank_size,), torch.int64),
    ):
        model.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=False)
    model.register_buffer("bank_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
    model.register_buffer("bank_count", torch.zeros((), dtype=torch.int64), persistent=False)
    model.temperature = 0.1
    model.max_sequence_positives = 16
    model.max_item_positives = 16
    model.num_negatives = 256
    model.min_positive_similarity = 0.3
    model.max_negative_similarity = 0.2
    model.similarity_weight_power = 1.0
    model.exclude_same_user_positives = False
    return model


def test_fifo_keeps_newest_entries_in_order_and_skips_low_quality():
    model = make_model(bank_size=3)
    for start in (0, 2):
        values = torch.tensor([start + 1.0, start + 2.0])
        model._push_bank(
            torch.stack([values, values], dim=1), torch.ones(2, 2), torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long), values.long(), torch.tensor([True, True]),
        )
    model._push_bank(
        torch.ones(1, 2) * 9, torch.ones(1, 2), torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long), torch.tensor([9]), torch.tensor([False]),
    )
    bank = model._bank_view()
    assert bank["bank_sample"].tolist() == [2, 3, 4]
    torch.testing.assert_close(bank["bank_r"].float()[:, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_weighted_loss_emphasizes_more_similar_positive():
    model = make_model()
    anchors = torch.tensor([[1.0, 0.0]])
    # Candidate 0 is the well-aligned positive, candidate 1 a misaligned
    # positive, candidate 2 the negative.
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    eligible = torch.tensor([[True, True, False]])
    negatives = torch.tensor([[False, False, True]])

    def loss_for(sim_good, sim_bad):
        similarity = torch.tensor([[sim_good, sim_bad, 0.0]])
        loss, rate, _ = model._relation_loss(anchors, candidates, similarity, eligible, negatives, 16)
        assert rate.item() == 1.0
        return loss.item()

    # Upweighting the positive the anchor already agrees with lowers the loss.
    assert loss_for(0.9, 0.3) < loss_for(0.3, 0.9)
    # Equal similarities reduce to the unweighted per-positive mean.
    ell_good = torch.logaddexp(torch.tensor(10.0), torch.tensor(0.0)) - 10.0
    ell_bad = torch.logaddexp(torch.tensor(0.0), torch.tensor(0.0)) - 0.0
    assert abs(loss_for(0.5, 0.5) - ((ell_good + ell_bad) / 2).item()) < 1e-5


def test_topk_keeps_most_similar_positives():
    model = make_model()
    anchors = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    similarity = torch.tensor([[0.4, 0.9, 0.0]])
    eligible = torch.tensor([[True, True, False]])
    negatives = torch.tensor([[False, False, True]])
    _, _, positive_sim = model._relation_loss(anchors, candidates, similarity, eligible, negatives, 1)
    assert abs(positive_sim.item() - 0.9) < 1e-6


def test_pair_selection_uses_sasrec_similarity_and_excludes_self():
    # Items 0 and 1 are near-identical in SASRec space; item 2 is orthogonal.
    table = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]), dim=-1)
    model = make_model(item_table=table)
    seq_close, seq_far = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    # Bank: 0 = same sample (stale copy), 1 = similar item, 2 = unrelated
    # in both spaces (the only valid negative), 3 = similar sequence only.
    model._push_bank(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        torch.stack([seq_close, seq_far, seq_far, seq_close]),
        torch.tensor([0, 1, 2, 2]), torch.tensor([5, 6, 7, 8]),
        torch.tensor([100, 101, 102, 103]), torch.ones(4, dtype=torch.bool),
    )
    r = torch.tensor([[1.0, 0.0]], requires_grad=True)
    seq_loss, item_loss, seq_rate, item_rate, seq_sim, item_sim = model._collaborative_losses(
        r, seq_close[None], torch.tensor([0]), torch.tensor([5]), torch.tensor([100]),
        torch.tensor([False]),
    )
    assert seq_rate.item() == 1.0 and item_rate.item() == 1.0
    # Only candidate 3 is a sequence positive and only candidate 1 an item
    # positive; the stale self entry (sim 1.0) is excluded from both.
    assert abs(seq_sim.item() - 1.0) < 1e-3
    assert abs(item_sim.item() - table[0] @ table[1]) < 1e-3
    (seq_loss + item_loss).backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
