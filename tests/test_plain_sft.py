import sys
from pathlib import Path

import pandas as pd

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from plain_sft_dataset import PlainSFTDataset
from plain_sft_eval import allowed_catalog_tokens, build_catalog_trie


class CharacterTokenizer:
    bos_token_id = None
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(character) for character in text]


def test_plain_sft_masks_prompt_and_has_no_latent_token(tmp_path):
    data_file = tmp_path / "sample.csv"
    pd.DataFrame([{
        "history_item_title": '["Piano"]',
        "history_item_id": "[1]",
        "item_title": "Guitar",
        "item_id": 2,
        "user_id": 3,
    }]).to_csv(data_file, index=False)
    tokenizer = CharacterTokenizer()
    train_data = PlainSFTDataset(
        train_file=str(data_file), tokenizer=tokenizer,
        max_len=512, category="music instruments",
    )
    example = train_data[0]
    target_ids = tokenizer.encode('"Guitar"\n') + [tokenizer.eos_token_id]
    assert example["input_ids"][-len(target_ids):] == target_ids
    assert example["labels"] == (
        [-100] * (len(example["input_ids"]) - len(target_ids)) + target_ids
    )
    assert "<|Thought|>" not in "".join(
        chr(token_id) for token_id in example["input_ids"] if token_id
    )

    test_data = PlainSFTDataset(
        train_file=str(data_file), tokenizer=tokenizer,
        max_len=512, test=True, category="music instruments",
    )
    assert "labels" not in test_data[0]
    assert test_data[0]["input_ids"] == example["input_ids"][:-len(target_ids)]


def test_catalog_trie_allows_only_catalog_completions(tmp_path):
    info_file = tmp_path / "items.txt"
    info_file.write_text("Guitar\t1\nPiano\t2\n")
    tokenizer = CharacterTokenizer()
    trie = build_catalog_trie(str(info_file), tokenizer)
    quote = ord('"')
    assert allowed_catalog_tokens(trie, [], tokenizer.eos_token_id) == [quote]
    assert set(allowed_catalog_tokens(trie, [quote], tokenizer.eos_token_id)) == {ord("G"), ord("P")}
    completion = tokenizer.encode('"Guitar"\n')
    assert allowed_catalog_tokens(trie, completion, tokenizer.eos_token_id) == [tokenizer.eos_token_id]
    assert allowed_catalog_tokens(
        trie, completion + [tokenizer.eos_token_id], tokenizer.eos_token_id,
    ) == [tokenizer.eos_token_id]
    assert allowed_catalog_tokens(
        trie, tokenizer.encode('"GuitarX'), tokenizer.eos_token_id,
    ) == [tokenizer.eos_token_id]
