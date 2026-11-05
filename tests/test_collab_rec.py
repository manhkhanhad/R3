import sys
from pathlib import Path

import pytest
import torch
from transformers import DataCollatorForSeq2Seq, GenerationConfig, PreTrainedTokenizerFast, Qwen2Config
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "latent"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collab_rec_latent import CollabLatentModel

VOCAB = 40
THOUGHT = VOCAB - 1


def make_model(mode, dim=8):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=4, pad_token_id=0, eos_token_id=1)
    config.collab_rec = {"support_mode": mode, "rationale_dim": dim}
    model = CollabLatentModel(config).eval()
    if mode == "text":
        tokens = torch.randint(2, THOUGHT, (6, 5))
        mask = torch.ones(6, 5, dtype=torch.bool)
        mask[:, :2] = False  # left padding
        tokens[~mask] = 0
        model.set_support_table(tokens, mask)
    else:
        model.set_support_table(torch.randn(6, dim))
    return model


def batch():
    ids = torch.tensor([[3, 4, 5, THOUGHT, 6, 7], [0, 3, 5, THOUGHT, 8, 9]])
    mask = (ids != 0).long()
    mask[0, 0] = 1
    labels = torch.full_like(ids, -100)
    labels[:, 4:] = ids[:, 4:]
    return ids, mask, labels


@pytest.mark.parametrize("mode", ["text", "emb"])
def test_thought_depends_on_supports_and_trains(mode):
    model = make_model(mode)
    ids, mask, labels = batch()
    model.set_support_ids(torch.tensor([[0, 1, 2], [3, 4, -1]]))
    e1 = model.generate_embs(ids, mask)
    model.set_support_ids(torch.tensor([[5, 4, 3], [0, 1, -1]]))
    e2 = model.generate_embs(ids, mask)
    model.set_support_ids(None)
    assert not torch.allclose(e1[:, 3], e2[:, 3])  # thought slot changes with supports
    assert torch.allclose(e1[:, :3], e2[:, :3]) and torch.allclose(e1[:, 4:], e2[:, 4:])
    # padding entries (-1) are ignored
    model.set_support_ids(torch.tensor([[0, 1, 2], [3, 4, -1]]))
    e3 = model.generate_embs(ids, mask)
    model.set_support_ids(torch.tensor([[0, 1, 2], [3, 4, -1]]))
    assert torch.allclose(e1, e3)

    model.train()
    out = model(input_ids=ids, attention_mask=mask, labels=labels, support_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]))
    out.loss.backward()
    head = model.injectnet.W_q_2 if mode == "text" else model.support_proj
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model._support_ids is None


@pytest.mark.parametrize("mode", ["text", "emb"])
def test_no_supports_path(mode):
    model = make_model(mode)
    ids, mask, labels = batch()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.isfinite(out.loss)
    model.set_support_ids(torch.zeros(2, 0, dtype=torch.long))
    assert torch.isfinite(model(input_ids=ids, attention_mask=mask, labels=labels).loss)


def test_beam_generation_matches_per_row_supports():
    model = make_model("text")
    ids, mask, _ = batch()
    prompt, pmask = ids[:, :4], mask[:, :4]
    support = torch.tensor([[0, 1, 2], [3, 4, 5]])
    cfg = GenerationConfig(num_beams=3, num_return_sequences=3, max_new_tokens=3, pad_token_id=0, eos_token_id=1,
                           do_sample=False)
    model.set_support_ids(support)
    joint = model.generate(prompt, attention_mask=pmask, generation_config=cfg).tolist()
    single = []
    for r in range(2):
        model.set_support_ids(support[r:r + 1])
        single += model.generate(prompt[r:r + 1], attention_mask=pmask[r:r + 1], generation_config=cfg).tolist()
    model.set_support_ids(None)
    strip = lambda seqs: [[t for t in s if t != 0] for s in seqs]  # noqa: E731
    assert strip(joint) == strip(single)


def test_collator_stacks_support_ids():
    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "a": 1}, unk_token="[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]")
    collate = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    out = collate([{"input_ids": [1, 1], "attention_mask": [1, 1], "labels": [-100, 1], "support_ids": [3, 4]},
                   {"input_ids": [1], "attention_mask": [1], "labels": [1], "support_ids": [5, 6]}])
    assert out["support_ids"].tolist() == [[3, 4], [5, 6]]


def test_loss_average_switch_reaches_trainer(tmp_path):
    from transformers import Trainer, TrainingArguments
    model = make_model("text")
    for flag in (False, True):
        model.accepts_loss_kwargs = flag
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True))
        assert trainer.model_accepts_loss_kwargs is flag


def test_rationale_infonce_train_only_and_trains():
    model = make_model("text")
    ids, mask, labels = batch()
    tokens = torch.randint(2, THOUGHT, (6, 5))
    rmask = torch.ones(6, 5, dtype=torch.bool)
    rmask[:, :1] = False
    model.set_rationale_table(tokens, rmask)
    support = torch.tensor([[0, 1, 2], [3, 4, 5]])
    own = torch.tensor([2, 5])
    model.train()
    base = model(input_ids=ids, attention_mask=mask, labels=labels, support_ids=support, own_row=own).loss
    model.rationale_beta, model.aux_negatives = 0.1, 4
    torch.manual_seed(1)
    out = model(input_ids=ids, attention_mask=mask, labels=labels, support_ids=support, own_row=own)
    assert len(model.aux_log) == 1 and model.aux_log[0] > 0
    assert torch.allclose(out.loss, base + 0.1 * torch.tensor(model.aux_log[0]), atol=1e-4)
    out.loss.backward()
    assert model.injectnet.W_q.weight.grad.abs().sum() > 0  # pooling of rationales gets gradient
    assert model._thought is None and model._own_rows is None
    model.eval()
    with torch.no_grad():
        ev = model(input_ids=ids, attention_mask=mask, labels=labels, support_ids=support, own_row=own).loss
    assert torch.allclose(ev, base, atol=1e-5)  # eval loss = NTP only


def test_collator_passes_own_row():
    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "a": 1}, unk_token="[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]")
    collate = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    out = collate([{"input_ids": [1, 1], "attention_mask": [1, 1], "labels": [-100, 1], "support_ids": [3, 4], "own_row": 7},
                   {"input_ids": [1], "attention_mask": [1], "labels": [1], "support_ids": [5, 6], "own_row": 9}])
    assert out["own_row"].tolist() == [7, 9]
