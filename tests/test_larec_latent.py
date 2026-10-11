import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))
sys.path.insert(0, str(ROOT / "src" / "latent_distill"))
sys.path.insert(0, str(ROOT / "tests"))

from generate_cot import leaks, parse_steps, redact  # noqa: E402
from larec_latent import LaRecLatentModel, LaRecLatentRDataset  # noqa: E402
from test_rationale_latent import FakeTokenizer  # noqa: E402

VOCAB, HIDDEN, K = 40, 32, 3
THOUGHT = VOCAB - 1


def tiny(**spec):
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, eos_token_id=1)
    config.latent_larec = {"k": K, "tau": 0.1, "negatives": 7, "token_norm": 2.0, **spec}
    torch.manual_seed(0)
    model = LaRecLatentModel(config)
    model.head.reset(0)
    with torch.no_grad():
        model.head.down.weight.normal_(0, 0.1)  # non-trivial head for the tests
    model.set_teacher_states(torch.randn(20, K, HIDDEN), torch.tensor([True] * 19 + [False]))
    return model


def batch():
    ids = torch.tensor([[2, 3, 4, THOUGHT, THOUGHT, THOUGHT, 5, 6, 1],
                        [7, 8, 9, THOUGHT, THOUGHT, THOUGHT, 10, 11, 1]])
    labels = ids.clone()
    labels[:, :6] = -100
    return ids, torch.ones_like(ids), labels


def test_latents_are_sequential_and_scaled():
    model = tiny().eval()
    ids, mask, _ = batch()
    with torch.no_grad():
        embeds = model.generate_embs(ids, mask)
        z = model._trace["latents"]
        # z_1 is the expected embedding of the next-token distribution at the last prompt token
        state = model.model(input_ids=ids[:, :3]).last_hidden_state[:, -1]
        logits = model.lm_head(model.head(state))
        logits[:, -1] = float("-inf")
        torch.testing.assert_close(z[:, 0], logits.softmax(-1) @ model.model.embed_tokens.weight)
        torch.testing.assert_close(embeds[:, 3:6], z)
        # z_1 depends only on the prompt: changing the target leaves the chain unchanged
        ids2 = ids.clone()
        ids2[:, 6:8] = 12
        model.generate_embs(ids2, mask)
        torch.testing.assert_close(model._trace["latents"], z)
        # changing the history changes the chain
        ids3 = ids.clone()
        ids3[0, 0] = 13
        model.generate_embs(ids3, mask)
        assert not torch.allclose(model._trace["latents"][0], z[0])


def test_aux_losses_only_in_training_and_reach_the_head():
    model = tiny()
    ids, mask, labels = batch()
    rows = torch.tensor([0, 19])  # row 19 has no valid CoT: no step alignment for it
    model.eval()
    plain = model(input_ids=ids, attention_mask=mask, labels=labels, larec_row=rows).loss
    model.train()
    out = model(input_ids=ids, attention_mask=mask, labels=labels, larec_row=rows)
    assert out.loss > plain + 0.1  # InfoNCE over 8 candidates starts near log 8
    out.loss.backward()
    assert model.head.up.weight.grad.abs().sum() > 0
    assert model.model.embed_tokens.weight.grad is not None


def test_step_alignment_prefers_matching_teacher_state():
    model = tiny(negatives=7)
    student = model.teacher_states[[0, 1]].clone()
    assert model.step_alignment(student, torch.tensor([0, 1])) < model.step_alignment(student, torch.tensor([1, 0]))


def test_swap_gives_each_row_the_next_rows_chain():
    model = tiny().eval()
    ids, mask, _ = batch()
    with torch.no_grad():
        model.generate_embs(ids, mask)
        z = model._trace["latents"]
        model.swap_shift = 1
        embeds = model.generate_embs(ids, mask)
    torch.testing.assert_close(embeds[0, 3:6], z[1])
    torch.testing.assert_close(embeds[1, 3:6], z[0])


def test_dataset_has_k_consecutive_slots(tmp_path):
    frame = pd.DataFrame({"history_item_title": [str(["Capo"])], "item_title": ["Strings"], "item_id": [3],
                          "history_item_id": [str([1])]})
    path = tmp_path / "train" / "x.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    tok = FakeTokenizer()
    ds = LaRecLatentRDataset(str(path), tok, max_len=4096, category="m", k=4)
    ex = ds[0]
    thought = tok.encode("<|Thought|>")[0]
    at = ex["input_ids"].index(thought)
    assert ex["input_ids"][at:at + 4] == [thought] * 4 and thought not in ex["input_ids"][at + 4:]
    assert all(v == -100 for v in ex["labels"][:at + 4]) and ex["labels"][at + 4] != -100
    assert len(ex["labels"]) == len(ex["input_ids"]) and ex["larec_row"] == 0


def test_cot_parsing_and_leak_filter():
    text = "Step 1: plays guitar.\nStep 2: needs strings.\nStep 3: nylon for classical."
    assert parse_steps(text, 3) == ["plays guitar.", "needs strings.", "nylon for classical."]
    assert parse_steps(text, 4) is None
    target = "D'Addario EJ45 Pro-Arte Nylon Classical Guitar Strings"
    assert leaks(["they want D'Addario strings"], target)  # brand not in the history
    assert not leaks(["they like D'Addario strings"], target, ["D'Addario EJ16 Phosphor Bronze"])  # brand in history
    assert leaks(["next: EJ45 strings"], target, ["D'Addario EJ16"])  # model number
    assert not leaks(["next: classical guitar strings"], target)  # a 3-word product type is allowed
    assert leaks(["so they buy Pro-Arte Nylon Classical Guitar strings"], target)  # 4-word title run
    steps = ["Next they buy the D'Addario EJ45 Pro-Arte Nylon Classical Guitar Strings.", "Nylon strings fit."]
    clean = redact(steps, target)
    assert clean == ["Next they buy the the product.", "Nylon strings fit."] and not leaks(clean, target)
    assert redact(["A D'Addario EJ45 set."], target, ["D'Addario EJ16"]) == ["A D'Addario set."]
