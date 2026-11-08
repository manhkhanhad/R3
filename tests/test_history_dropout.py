"""History dropout (option C): title/item segmentation, p = 0 is plain R3, p = 1 hides
every title token from the target side (template still visible), eval uses eval_p,
and KV-cache generation reuses the block drawn at prefill."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from history_dropout import dropout_block, title_items  # noqa: E402
from layers import LatentModel  # noqa: E402

THOUGHT, PAD, SEP = 31, 0, 30
PREFIX = [5, 6, 7]
HEADER = [8, 9]
RESPONSE = [20]
SCAFFOLD = RESPONSE + [21]
TARGET = [3, 4, 1]
SPEC = {"header_ids": HEADER, "response_ids": RESPONSE, "separator_ids": [SEP], "p": 0.5, "unit": "item", "eval_p": 0.0}


class FixedLatent(torch.nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.register_buffer("vector", vector)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        return self.vector.expand(thought_id_idx.numel(), -1)


def make_model(p=None, eval_p=0.0, unit="item", attn="eager", seed=0):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=PAD, eos_token_id=1)
    config._attn_implementation = attn
    if p is not None:
        config.latent_bottleneck = {"dropout": {**SPEC, "p": p, "eval_p": eval_p, "unit": unit}}
    model = LatentModel(config).eval()
    model.attention = FixedLatent(torch.randn(16))
    return model


def history(items):
    out = []
    for it in items:
        out += [SEP] + it + [SEP]
    return out


def row(items):
    return PREFIX + HEADER + history(items) + SCAFFOLD + [THOUGHT] + TARGET


def logits_after_thought(model, ids_list, train=False):
    ids = torch.tensor([ids_list])
    model.train(train)
    with torch.set_grad_enabled(False):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    model.eval()
    t = ids_list.index(THOUGHT)
    return logits[t:]


def test_title_items_segmentation():
    ids = torch.tensor([row([[10, 11], [12]])])
    pos = torch.tensor([ids[0].tolist().index(THOUGHT)])
    title, item = title_items(ids, torch.ones_like(ids), SPEC, pos)
    r = ids[0].tolist()
    start = len(PREFIX) + len(HEADER)
    assert title[0].tolist() == [False] * start + [False, True, True, False, False, True, False] + [False] * (len(r) - start - 7)
    assert item[0, start + 1:start + 3].tolist() == [0, 0]
    assert item[0, start + 5].item() == 1
    assert (item[0][~title[0]] == -1).all()


def test_truncated_row_has_no_titles():
    ids = torch.tensor([[9, 10, 11, SEP, 12] + SCAFFOLD + [THOUGHT] + TARGET])  # header cut off
    pos = torch.tensor([ids[0].tolist().index(THOUGHT)])
    title, _ = title_items(ids, torch.ones_like(ids), SPEC, pos)
    assert not title.any()


def test_dropout_block_item_and_token_units():
    ids = torch.tensor([row([[10, 11], [12], [13, 14]])])
    pos = torch.tensor([ids[0].tolist().index(THOUGHT)])
    assert not dropout_block(ids, torch.ones_like(ids), SPEC, pos, 0.0).any()
    title, item = title_items(ids, torch.ones_like(ids), SPEC, pos)
    assert torch.equal(dropout_block(ids, torch.ones_like(ids), SPEC, pos, 1.0), title)
    g = torch.Generator().manual_seed(3)
    block = dropout_block(ids, torch.ones_like(ids), SPEC, pos, 0.5, generator=g)
    assert (block <= title).all()
    for k in range(3):  # whole items are dropped or kept together
        inside = block[0][item[0] == k]
        assert inside.all() or not inside.any()
    tok = dropout_block(ids, torch.ones_like(ids), {**SPEC, "unit": "token"}, pos, 0.5, generator=g)
    assert (tok <= title).all()


def test_p_zero_equals_plain_model():
    plain = make_model(p=None)
    dropped = make_model(p=0.0)
    dropped.attention.vector.copy_(plain.attention.vector)
    r = row([[10, 11], [12]])
    assert torch.allclose(logits_after_thought(plain, r, train=True), logits_after_thought(dropped, r, train=True), atol=1e-5)
    assert torch.allclose(logits_after_thought(plain, r), logits_after_thought(dropped, r), atol=1e-5)


def test_p_one_hides_titles_but_keeps_template():
    model = make_model(p=1.0)
    a = logits_after_thought(model, row([[10, 11], [12]]), train=True)
    b = logits_after_thought(model, row([[13, 14], [15]]), train=True)  # same item count, other titles
    assert torch.allclose(a, b, atol=1e-5)
    c = logits_after_thought(model, row([[10, 11], [12]]))  # eval: eval_p = 0, history visible
    assert not torch.allclose(a, c, atol=1e-3)
    d = logits_after_thought(model, row([[13, 14], [15]]))
    assert not torch.allclose(c, d, atol=1e-3)


def test_eval_p_applies_in_eval_mode():
    model = make_model(p=0.0, eval_p=1.0)
    a = logits_after_thought(model, row([[10, 11], [12]]))
    b = logits_after_thought(model, row([[13, 14], [15]]))
    assert torch.allclose(a, b, atol=1e-5)


def test_partial_dropout_changes_only_dropped_items():
    model = make_model(p=0.5)
    model._history_dropout_generator = torch.Generator().manual_seed(0)
    ids = torch.tensor([row([[10, 11], [12], [13, 14]])])
    pos = torch.tensor([ids[0].tolist().index(THOUGHT)])
    model.train()
    # draw the same block as the forward will (same generator state) to know what was dropped
    g = torch.Generator().manual_seed(0)
    block = dropout_block(ids, torch.ones_like(ids), {**SPEC, "p": 0.5}, pos, 0.5, generator=g)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    assert model._bottleneck_block is not None
    assert torch.equal(model._bottleneck_block, block)
    model.eval()
    assert (block <= title_items(ids, torch.ones_like(ids), SPEC, pos)[0]).all()
    assert logits.shape[0] == ids.shape[1]


@pytest.mark.parametrize("num_beams", [1, 3])
def test_kv_cache_generation_matches_teacher_forcing_with_eval_dropout(num_beams):
    model = make_model(p=0.5, eval_p=1.0)
    prompts = [row([[10, 11], [12]])[:-len(TARGET)], row([[13]])[:-len(TARGET)]]
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
        outputs[use_cache] = out.sequences
    assert torch.equal(outputs[True], outputs[False])
    if num_beams == 1:
        generated = outputs[True][1, width:].tolist()
        ids_tf = torch.tensor([prompts[1] + generated])
        with torch.no_grad():
            logits = model(input_ids=ids_tf, attention_mask=torch.ones_like(ids_tf)).logits[0]
        t = len(prompts[1]) - 1
        assert logits[t:t + 4].argmax(-1).tolist() == generated
