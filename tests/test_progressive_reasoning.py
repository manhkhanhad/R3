"""ReaRec recurrence + PRL on the latent-only bottleneck (continuous_thoughts.py,
progressive_reasoning.py), on a tiny random Qwen2 in fp32 on CPU."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from continuous_thoughts import ContinuousThoughts, info_nce  # noqa: E402
from layers import LatentModel  # noqa: E402
from progressive_reasoning import prl_loss, step_logits, step_loss  # noqa: E402

THOUGHT, PAD = 31, 0
PREFIX = [5, 6, 7]
SCAFFOLD = [20, 21]
TARGET = [3, 4, 8, 1]
K = 3


def make_model(k=K, cl=None, seed=0, prl=True):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=PAD, eos_token_id=1)
    config._attn_implementation = "eager"
    config.latent_continuous_thoughts = {"k": k, "source": "hidden", "recurrence": "rearec",
                                         "token_norm": 1.0, "cl": cl}
    config.latent_bottleneck = {"visible_prefix_ids": PREFIX}
    if prl:
        config.latent_prl = {"temp_scale": 5.0, "weight": 1.0, "from_step": 0}
    model = LatentModel(config).float()
    model.thoughts.token_norm.fill_(float(model.model.embed_tokens.weight[:-1].norm(dim=-1).mean()))
    model.thoughts.reset_parameters()
    with torch.no_grad():  # make the step embeddings matter at this scale
        model.thoughts.reason_pos_emb.normal_(0.0, 1.0)
    return model.eval()


def batch(histories, k=K, target=TARGET):
    rows = [PREFIX + h + SCAFFOLD + [THOUGHT] * k + target for h in histories]
    width = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    labels = torch.full_like(ids, -100)
    for i, r in enumerate(rows):
        labels[i, len(r) - len(target):len(r)] = torch.tensor(target)
    return ids, mask, labels


def prepared(model, ids, mask):
    """Spliced embeddings + bottleneck block, as LatentModel.forward builds them."""
    model._latent_bottleneck_inputs(model.config.latent_bottleneck, ids, mask, ids, None)
    return model.generate_embs(ids, mask)


def logits_of(model, ids, mask, labels, step, thoughts=None):
    model.thoughts.override = (lambda x: thoughts) if thoughts is not None else None
    try:
        with torch.no_grad():
            embs = prepared(model, ids, mask)
            return step_logits(model, embs, mask, labels, model._bottleneck_block,
                               model._bottleneck_thought_pos, K, step)[0]
    finally:
        model.thoughts.override = None


def naive(model, ids):
    """Uncached recurrence for one row: thoughts t_1..t_K and outputs h_1..h_K."""
    first = ids[0].tolist().index(THOUGHT)
    emb = model.model.embed_tokens(ids[:, :first])
    thoughts, outs = [], []
    with torch.no_grad():
        h = model.model(inputs_embeds=emb).last_hidden_state[:, -1]
        for step in range(model.thoughts.k):
            t = model.thoughts.project(h, step)
            thoughts.append(t)
            emb = torch.cat([emb, t[:, None]], dim=1)
            h = model.model(inputs_embeds=emb).last_hidden_state[:, -1]
            outs.append(h)
    return torch.stack(thoughts, 1), torch.stack(outs, 1)


def test_old_coconut_config_is_rejected():
    with pytest.raises(ValueError, match="frozen code"):
        ContinuousThoughts(16, {"k": 2, "source": "hidden"})


def test_cached_recurrence_matches_naive_and_batches():
    model = make_model()
    hists = [[10, 11, 12], [13]]
    ids, mask, _ = batch(hists)
    with torch.no_grad():
        prepared(model, ids, mask)
    batched = model.thoughts.last_thoughts
    for i, h in enumerate(hists):
        single = torch.tensor([PREFIX + h + SCAFFOLD + [THOUGHT] * K])
        expect, _ = naive(model, single)
        assert torch.allclose(batched[i], expect[0], atol=1e-5)
    # step embeddings make t_k depend on k even for the same state
    assert not torch.allclose(model.thoughts.project(torch.zeros(1, 16), 0), model.thoughts.project(torch.zeros(1, 16), 1))


def test_step_K_reproduces_main_loss():
    model = make_model()
    ids, mask, labels = batch([[10, 11, 12], [13]])
    with torch.no_grad():
        main = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        embs = prepared(model, ids, mask)
        mine, _ = step_loss(model, embs, mask, labels, model._bottleneck_block, model._bottleneck_thought_pos,
                            K, K, temperature=1.0)
    assert torch.allclose(main, mine, atol=1e-5)


@pytest.mark.parametrize("step", [1, 2])
def test_step_j_sees_only_thoughts_up_to_j(step):
    model = make_model()
    torch.manual_seed(1)
    fixed = torch.randn(1, K, 16)
    a_ids, a_mask, a_lab = batch([[10, 11, 12]])
    b_ids, b_mask, b_lab = batch([[13]])  # other history, other length
    base = logits_of(model, a_ids, a_mask, a_lab, step, fixed)
    assert torch.allclose(base, logits_of(model, b_ids, b_mask, b_lab, step, fixed), atol=1e-5)
    later = fixed.clone()
    later[:, step:] = torch.randn(1, K - step, 16)
    assert torch.allclose(base, logits_of(model, a_ids, a_mask, a_lab, step, later), atol=1e-5)
    own = fixed.clone()
    own[:, step - 1] = torch.randn(16)
    assert not torch.allclose(base[0], logits_of(model, a_ids, a_mask, a_lab, step, own)[0], atol=1e-3)


def test_step_j_equals_a_model_with_j_thoughts():
    """Hiding thoughts j+1..K == a prompt that only has j thoughts (positions too)."""
    step = 2
    model = make_model()
    fixed = torch.randn(1, K, 16)
    ids, mask, labels = batch([[10, 11, 12]])
    got = logits_of(model, ids, mask, labels, step, fixed)
    short_ids, short_mask, short_labels = batch([[10, 11, 12]], k=step)
    # same weights, a j-thought prompt, plain bottleneck main pass
    model.thoughts.k = model.config.latent_continuous_thoughts["k"] = step
    model.thoughts.override = lambda x: fixed[:, :step]
    try:
        with torch.no_grad():
            logits = model(input_ids=short_ids, attention_mask=short_mask).logits[0]
    finally:
        model.thoughts.k = model.config.latent_continuous_thoughts["k"] = K
        model.thoughts.override = None
    first = short_ids[0].tolist().index(THOUGHT) + step - 1
    assert torch.allclose(got, logits[first:first + len(TARGET)], atol=1e-5)


def test_step0_reads_history_not_thoughts():
    model = make_model()
    fixed = torch.randn(1, K, 16)
    a = batch([[10, 11, 12]])
    base = logits_of(model, *a, 0, fixed)
    assert torch.allclose(base, logits_of(model, *a, 0, torch.randn(1, K, 16)), atol=1e-5)
    assert not torch.allclose(base, logits_of(model, *batch([[13, 14, 15]]), 0, fixed), atol=1e-3)


def test_contrastive_branch_leaves_clean_thoughts_and_matches_naive():
    cl = {"weight": 1.0, "noise": 1e-7, "temperature": 0.07, "similarity": "dot"}
    model = make_model(cl=cl)
    hists = [[10, 11, 12], [13], [14, 15]]
    ids, mask, _ = batch(hists)
    with torch.no_grad():
        prepared(model, ids, mask)
        clean_eval = model.thoughts.last_thoughts
        assert model.thoughts.last_cl_loss is None  # eval: no noisy branch
        model.train()
        prepared(model, ids, mask)
    assert torch.allclose(model.thoughts.last_thoughts, clean_eval, atol=1e-6)
    outs = torch.cat([naive(model, torch.tensor([PREFIX + h + SCAFFOLD + [THOUGHT] * K]))[1] for h in hists])
    # noise ~0: the noisy outputs equal the clean ones, so the loss is InfoNCE(clean, clean)
    assert torch.allclose(model.thoughts.last_cl_loss, info_nce(outs, outs, 0.07, "dot"), rtol=1e-3, atol=1e-4)


def test_train_forward_adds_prl_and_cl_with_hf_normalisation():
    cl = {"weight": 1.0, "noise": 1e-7, "temperature": 0.07, "similarity": "dot"}
    model = make_model(cl=cl)
    hists = [[10, 11, 12], [13]]
    ids, mask, labels = batch(hists)
    with torch.no_grad():
        main = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        embs = prepared(model, ids, mask)
        prl, per_step = prl_loss(model, embs, mask, labels, model.config.latent_prl)
        outs = torch.cat([naive(model, torch.tensor([PREFIX + h + SCAFFOLD + [THOUGHT] * K]))[1] for h in hists])
        contrast = info_nce(outs, outs, 0.07, "dot")
        model.train()
        total = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = int(labels[:, 1:].ne(-100).sum())
        halved = model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=2 * n).loss
    assert sorted(per_step) == [0, 1, 2]
    assert torch.allclose(total, main + prl + contrast, rtol=1e-4, atol=1e-4)
    assert torch.allclose(halved, total / 2, rtol=1e-4, atol=1e-4)
    assert set(model.last_prl_stats) == {"step0", "step1", "step2", "cl"}


def test_max_temp_schedule():
    from progressive_reasoning import step_temperature
    paper = {"temp_scale": 5.0}
    capped = {"temp_scale": 5.0, "max_temp": 25.0}
    assert [step_temperature(j, 4, paper) for j in range(4)] == [625.0, 125.0, 25.0, 5.0]
    got = [step_temperature(j, 4, capped) for j in range(4)]
    assert got == pytest.approx([25.0, 25 ** 0.75, 5.0, 25 ** 0.25])
    # K=2 with max_temp = scale^2 is the paper's schedule
    assert [step_temperature(j, 2, capped) for j in range(2)] == pytest.approx([25.0, 5.0])


def test_prl_loss_uses_capped_temperature():
    model = make_model()
    model.config.latent_prl = {"temp_scale": 5.0, "weight": 1.0, "from_step": 0, "max_temp": 25.0}
    ids, mask, labels = batch([[10, 11, 12]])
    with torch.no_grad():
        embs = prepared(model, ids, mask)
        _, means = prl_loss(model, embs, mask, labels, model.config.latent_prl)
        for j in range(K):
            logits, targets = step_logits(model, embs, mask, labels, model._bottleneck_block,
                                          model._bottleneck_thought_pos, K, j)
            expect = torch.nn.functional.cross_entropy(logits / 25.0 ** ((K - j) / K), targets)
            assert torch.allclose(means[j], expect, atol=1e-5)


def test_temperature_annealing():
    model = make_model()
    ids, mask, labels = batch([[10, 11, 12]])
    with torch.no_grad():
        embs = prepared(model, ids, mask)
        _, means = prl_loss(model, embs, mask, labels, model.config.latent_prl)
        for j in range(K):
            logits, targets = step_logits(model, embs, mask, labels, model._bottleneck_block,
                                          model._bottleneck_thought_pos, K, j)
            expect = torch.nn.functional.cross_entropy(logits / 5.0 ** (K - j), targets)
            assert torch.allclose(means[j], expect, atol=1e-5)
