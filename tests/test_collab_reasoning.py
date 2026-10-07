import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))
sys.path.insert(0, str(ROOT / "src" / "collab_reasoning"))

from collab_latent import CAND_TOKEN, CandidateSlotLatentModel, candidate_line  # noqa: E402
import cf_sources as S  # noqa: E402
from build_candidates import calibrate, canonical  # noqa: E402
from train_sasrec_folds import user_fold  # noqa: E402


def knn_context(seqs, n):
    ctx = object.__new__(S.Context)
    ctx.n = n
    ctx.train_seq = seqs
    users = sorted(seqs)
    ctx.user_index = {u: i for i, u in enumerate(users)}
    ri, ci = [], []
    for u, seq in seqs.items():
        for it in set(seq):
            ri.append(ctx.user_index[u]); ci.append(it)
    X = sp.csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(len(users), n))
    C = (X.T @ X).toarray().astype(np.float32)
    ctx.count = np.diag(C).copy()
    np.fill_diagonal(C, 0.0)
    ctx.C0 = C
    return ctx


def brute_knn(seqs, n, history, weights, leave_out):
    kept = {u: s for u, s in seqs.items() if u != leave_out}
    X = np.zeros((len(kept), n), np.float32)
    for k, s in enumerate(kept.values()):
        X[k, list(set(s))] = 1
    C = X.T @ X
    cnt = np.diag(C).copy()
    np.fill_diagonal(C, 0)
    out = np.zeros(n, np.float32)
    for h, w in zip(history, weights):
        if cnt[h] == 0:
            continue
        for i in range(n):
            if cnt[i] > 0:
                out[i] += w * C[h, i] / np.sqrt(cnt[h] * cnt[i])
    return out


def test_leave_user_out_knn_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 12
    seqs = {f"u{k}": list(rng.choice(n, size=rng.integers(3, 7), replace=False)) for k in range(8)}
    ctx = knn_context(seqs, n)
    history = seqs["u3"][:3]
    weights = S.recency_weights(len(history))
    fast_train = S._knn_scores(ctx, [np.array(history)], [weights], np.array(["u3"]), "train")[0]
    fast_test = S._knn_scores(ctx, [np.array(history)], [weights], np.array(["u3"]), "test")[0]
    np.testing.assert_allclose(fast_train, brute_knn(seqs, n, history, weights, "u3"), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fast_test, brute_knn(seqs, n, history, weights, None), rtol=1e-5, atol=1e-6)


class TinyBase(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)


def make_slot_model(vocab=10, hidden=4, dim=3, k=2):
    model = object.__new__(CandidateSlotLatentModel)
    nn.Module.__init__(model)
    model.model = TinyBase(vocab, hidden)
    model.cand_k = k
    model.cand_token_id = vocab - 2
    model.cand_token_norm = 2.0
    model.cand_projection = nn.Linear(dim, hidden)
    model.register_buffer("cand_table", torch.empty(0, dim), persistent=False)
    model.register_buffer("cand_valid", torch.empty(0, dtype=torch.bool), persistent=False)
    model._cand_items = None
    model.set_cand_table(torch.randn(5, dim), torch.tensor([True, True, True, False, True]))
    return model


def test_candidate_embeddings_scale_and_padding():
    model = make_slot_model()
    emb = model.candidate_embeddings(torch.tensor([[0, -1], [3, 4]]))
    norms = emb.norm(dim=-1)
    torch.testing.assert_close(norms[0, 0], torch.tensor(2.0))
    assert norms[0, 1] == 0 and norms[1, 0] == 0  # padding and unknown items are zero slots
    torch.testing.assert_close(norms[1, 1], torch.tensor(2.0))


def test_embed_with_candidates_fills_slots_in_order_and_expands_beams():
    model = make_slot_model()
    c = model.cand_token_id
    ids = torch.tensor([[1, c, c, 9], [2, c, c, 9]])
    model.set_cand_items(torch.tensor([[0, 1], [2, 4]]))
    out = model.embed_with_candidates(ids)
    expected = model.candidate_embeddings(torch.tensor([[0, 1], [2, 4]]))
    torch.testing.assert_close(out[0, 1:3], expected[0])
    torch.testing.assert_close(out[1, 1:3], expected[1])
    torch.testing.assert_close(out[0, 0], model.model.embed_tokens.weight[1])
    # two beams per prompt: rows are expanded consecutively
    beams = model.embed_with_candidates(ids.repeat_interleave(2, dim=0))
    torch.testing.assert_close(beams[1, 1:3], expected[0])
    torch.testing.assert_close(beams[2, 1:3], expected[1])


def test_embed_with_candidates_rejects_wrong_slot_count():
    model = make_slot_model()
    c = model.cand_token_id
    model.set_cand_items(torch.tensor([[0, 1]]))
    with pytest.raises(ValueError):
        model.embed_with_candidates(torch.tensor([[1, c, 9, 9]]))


def test_candidate_line_text_and_slots():
    titles = {0: "Capo", 1: "Strings"}
    assert candidate_line("text", [0, -1, 1], titles).endswith('"Capo", "Strings"')
    assert candidate_line("text", [-1, -1], titles).endswith("none")
    assert candidate_line("slots", [0, -1], titles).endswith(CAND_TOKEN * 2)


def test_calibrate_matches_target_rate_in_expectation():
    table = np.arange(10000)[:, None].repeat(3, 1)
    hit = np.ones(10000, dtype=bool)
    out, info = calibrate(table, hit, 0.25, seed=0)
    assert abs(info["q"] - 0.75) < 1e-9
    kept = (out == table).all(1).mean()
    assert 0.23 < kept < 0.28
    _, none = calibrate(table, hit * False, 0.25)
    assert none["q"] == 0.0


def test_canonical_maps_duplicate_titles_and_padding():
    titles = ["a", "b", "a"]
    title_id = {"a": 0, "b": 1}
    np.testing.assert_array_equal(canonical(np.array([[2, 1, -1]]), titles, title_id), [[0, 1, -1]])


def test_user_fold_is_stable_and_balanced():
    folds = [user_fold(f"user{i}", 4) for i in range(4000)]
    assert user_fold("user7", 4) == folds[7]
    counts = np.bincount(folds, minlength=4)
    assert counts.min() > 850
