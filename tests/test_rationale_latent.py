import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))

from rationale_latent import RationaleCollator, RationaleLatentModel, RationaleLatentRDataset  # noqa: E402
from reasoning_dataset import LatentRDataset  # noqa: E402

VOCAB, HIDDEN = 40, 32
THOUGHT = VOCAB - 1


def tiny():
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    return RationaleLatentModel(config).eval()


def test_decoder_reads_r_pooled_from_the_pool_sequence_only():
    model = tiny()
    ids = torch.tensor([[1, 2, 3, THOUGHT, 5, 6], [4, 4, 3, THOUGHT, 7, 8]])
    mask = torch.ones_like(ids)
    pool_a = torch.tensor([[1, 2, 3, 9, 9, THOUGHT], [4, 4, 3, 10, 11, THOUGHT]])
    pool_b = torch.tensor([[1, 2, 3, 12, 13, THOUGHT], [4, 4, 3, 14, 15, THOUGHT]])
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask, pool_input_ids=pool_a, pool_attention_mask=torch.ones_like(pool_a)).logits
        a2 = model(input_ids=ids, attention_mask=mask, pool_input_ids=pool_a, pool_attention_mask=torch.ones_like(pool_a)).logits
        b = model(input_ids=ids, attention_mask=mask, pool_input_ids=pool_b, pool_attention_mask=torch.ones_like(pool_b)).logits
    torch.testing.assert_close(a, a2)
    assert not torch.allclose(a[:, 3:], b[:, 3:])  # the rationale reaches the output through r
    torch.testing.assert_close(a[:, :3], b[:, :3])  # positions before the slot are untouched
    assert model._pool is None


def test_pool_latent_is_repeated_for_beams_and_matches_unexpanded():
    model = tiny()
    ids = torch.tensor([[1, 2, THOUGHT], [3, 4, THOUGHT]])
    pool = torch.tensor([[1, 2, 9, THOUGHT], [3, 4, 10, THOUGHT]])
    with torch.no_grad():
        model.set_pool(pool, torch.ones_like(pool))
        single = model.generate_embs(ids, torch.ones_like(ids))
        model.set_pool(pool, torch.ones_like(pool))
        beams = model.generate_embs(ids.repeat_interleave(3, 0), torch.ones(6, 3, dtype=torch.long))
    torch.testing.assert_close(beams, single.repeat_interleave(3, 0))


def test_gradient_reaches_pooler_through_pool_sequence():
    model = tiny().train()
    ids = torch.tensor([[1, 2, THOUGHT, 5, 6]])
    labels = torch.tensor([[-100, -100, -100, 5, 6]])
    pool = torch.tensor([[1, 2, 9, THOUGHT]])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                pool_input_ids=pool, pool_attention_mask=torch.ones_like(pool))
    out.loss.backward()
    assert model.attention.query.weight.grad.abs().sum() > 0


class FakeTokenizer:
    """Character-level stand-in with the HF interface ``Tokenizer`` uses."""
    bos_token_id, eos_token_id, pad_token_id, padding_side = None, 1, 0, "right"

    def encode(self, s):
        return [ord(c) % 1000 + 2 for c in s.replace("<|Thought|>", "\x00")]


@pytest.fixture
def csv(tmp_path):
    frame = pd.DataFrame({
        "history_item_title": [str(["Guitar picks", "Capo"]), str(["Drum sticks"]), str(["Reeds"])],
        "item_title": ["Strings", "Drum head", "Ligature"],
        "item_id": [3, 4, 5], "history_item_id": [str([1, 2]), str([6]), str([7])],
    })
    path = tmp_path / "test" / "x.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


def test_prompts_match_plain_r3_and_rationale_aligns_with_rows(csv):
    tok, rats = FakeTokenizer(), ["plays guitar", "drummer", "clarinet"]
    plain = LatentRDataset(str(csv), tok, max_len=4096, category="music instruments")
    latent = RationaleLatentRDataset(str(csv), tok, max_len=4096, category="music instruments", rationales=rats, mode="latent")
    text = RationaleLatentRDataset(str(csv), tok, max_len=4096, category="music instruments", rationales=rats, mode="text")
    for i in range(3):
        assert latent[i]["input_ids"] == plain[i]["input_ids"]
        assert latent[i]["labels"] == plain[i]["labels"]
        no_analysis, _ = latent.prompt_tokens(i)
        assert plain[i]["input_ids"][:len(no_analysis)] == no_analysis
        decoded = "".join(chr(t - 2) for t in latent[i]["pool_input_ids"])
        assert decoded.endswith(f"### Analysis: \n{rats[i]}\n\n### Response: \n\x00")
        assert latent[i]["pool_input_ids"] == text[i]["input_ids"][:len(latent[i]["pool_input_ids"])]
        assert text[i]["labels"][-len(plain[i]["labels"]):] != []

    # a shard written by utils/split.py keeps the original row in "Unnamed: 0"
    shard = csv.parent / "shard.csv"
    pd.read_csv(csv).iloc[1:].to_csv(shard, index=True)
    sharded = RationaleLatentRDataset(str(shard), tok, max_len=4096, category="music instruments", rationales=rats, mode="latent")
    assert sharded[0]["pool_input_ids"] == latent[1]["pool_input_ids"]


def test_collator_pads_pool_sequences_separately(csv):
    tok = FakeTokenizer()
    data = RationaleLatentRDataset(str(csv), tok, max_len=4096, category="m", rationales=["a", "bbbbbbbbbbbb", "c"], mode="latent")

    class HFTok(FakeTokenizer):
        model_input_names = ["input_ids", "attention_mask"]

        def pad(self, features, **kw):
            n = max(len(f["input_ids"]) for f in features)
            n = -(-n // 8) * 8
            return {"input_ids": torch.tensor([f["input_ids"] + [0] * (n - len(f["input_ids"])) for f in features]),
                    "attention_mask": torch.tensor([f["attention_mask"] + [0] * (n - len(f["attention_mask"])) for f in features])}

    feats = [dict(data[i]) for i in range(3)]
    batch = RationaleCollator(HFTok())(feats)
    lengths = [len(data[i]["pool_input_ids"]) for i in range(3)]
    assert batch["pool_input_ids"].shape == (3, -(-max(lengths) // 8) * 8)
    assert batch["pool_attention_mask"].sum(1).tolist() == lengths
