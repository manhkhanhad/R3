"""L7 latent-only bottleneck: target-side logits must not depend on the history
tokens once r is held fixed, in teacher forcing and in KV-cache generation."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from latent_bottleneck import prompt_block  # noqa: E402
from layers import LatentModel  # noqa: E402

THOUGHT, PAD = 31, 0
PREFIX = [5, 6, 7]
SCAFFOLD = [20, 21]  # stands in for "\n\n### Response: \n"
TARGET = [3, 4, 8]


class FixedLatent(torch.nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.register_buffer("vector", vector)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        return self.vector.expand(thought_id_idx.numel(), -1)


def make_model(bottleneck=True, attn="eager", seed=0):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=PAD, eos_token_id=1)
    config._attn_implementation = attn
    if bottleneck:
        config.latent_bottleneck = {"visible_prefix_ids": PREFIX}
    model = LatentModel(config).eval()
    model.attention = FixedLatent(torch.randn(16))
    return model


def row(history):
    return PREFIX + history + SCAFFOLD + [THOUGHT] + TARGET


def target_logits(model, history):
    ids = torch.tensor([row(history)])
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    t = ids[0].tolist().index(THOUGHT)
    return logits[t:]  # positions predicting TARGET[0..] and after


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_target_logits_ignore_history_when_r_fixed(attn):
    model = make_model(attn=attn)
    a = target_logits(model, [10, 11, 12])
    b = target_logits(model, [13])  # different titles and different length
    assert torch.allclose(a, b, atol=1e-5)


def test_without_bottleneck_history_changes_logits():
    model = make_model(bottleneck=False)
    a = target_logits(model, [10, 11, 12])
    b = target_logits(model, [13])
    assert not torch.allclose(a, b, atol=1e-3)


def test_latent_still_drives_prediction():
    model = make_model()
    a = target_logits(model, [10, 11, 12])
    model.attention.vector.copy_(torch.randn(16))
    b = target_logits(model, [10, 11, 12])
    assert not torch.allclose(a, b, atol=1e-3)


def test_right_padded_batch_matches_single_rows():
    model = make_model()
    rows = [row([10, 11, 12]), row([13])]
    width = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    for i, hist in enumerate(([10, 11, 12], [13])):
        t = rows[i].index(THOUGHT)
        assert torch.allclose(logits[i, t:len(rows[i])], target_logits(model, hist), atol=1e-5)


def test_prefix_mismatch_hides_everything_before_thought():
    ids = torch.tensor([[9, 6, 7, 10, 20, THOUGHT, 3]])  # truncated/foreign prefix
    block, pos = prompt_block(ids, torch.ones_like(ids), THOUGHT, PREFIX)
    assert pos.tolist() == [5]
    assert block[0].tolist() == [True] * 5 + [False] * 2
    ids = torch.tensor([[PAD, PAD] + row([10])])  # left padding: prefix matched after pads
    mask = torch.tensor([[0, 0] + [1] * len(row([10]))])
    block, _ = prompt_block(ids, mask, THOUGHT, PREFIX)
    assert block[0].tolist()[:6] == [False, False, False, False, False, True]


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("num_beams", [1, 3])
def test_kv_cache_generation_ignores_history(attn, num_beams):
    model = make_model(attn=attn)
    prompts = [row([10, 11, 12])[:-len(TARGET)], row([13])[:-len(TARGET)]]
    width = max(map(len, prompts))
    ids = torch.tensor([[PAD] * (width - len(p)) + p for p in prompts])
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
    config = GenerationConfig(max_new_tokens=4, min_new_tokens=4, num_beams=num_beams,
                              num_return_sequences=num_beams, do_sample=False,
                              pad_token_id=PAD, eos_token_id=1, suppress_tokens=[THOUGHT])
    outputs = {}
    for use_cache in (True, False):
        with torch.no_grad():
            out = model.generate(ids, attention_mask=mask, generation_config=config, use_cache=use_cache,
                                 return_dict_in_generate=True, output_scores=True)
        seqs = out.sequences[:, width:].view(2, num_beams, -1)
        assert torch.equal(seqs[0], seqs[1]), "generation differs across histories with fixed r"
        outputs[use_cache] = out.sequences
        if num_beams > 1:
            scores = out.sequences_scores.view(2, num_beams)
            assert torch.allclose(scores[0], scores[1], atol=1e-5)
        else:
            for step in out.scores:
                assert torch.allclose(step[0], step[1], atol=1e-5)
    assert torch.equal(outputs[True], outputs[False])

    # greedy cached generation equals teacher-forced argmax under the same mask
    if num_beams == 1:
        generated = outputs[True][1, width:].tolist()
        ids_tf = torch.tensor([prompts[1] + generated])
        with torch.no_grad():
            logits = model(input_ids=ids_tf, attention_mask=torch.ones_like(ids_tf)).logits[0]
        t = len(prompts[1]) - 1
        assert logits[t:t + 4].argmax(-1).tolist() == generated
