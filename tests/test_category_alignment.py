import csv
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "latent"))

from category_alignment_attention import CategoryAlignedLatentModel
from category_alignment_data import category_paths, item_asins


def test_category_paths_use_asins_and_decode_entities(tmp_path):
    split = tmp_path / "train.csv"
    with split.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_asin", "item_id"])
        writer.writeheader()
        writer.writerows([{"item_asin": "A", "item_id": 1}, {"item_asin": "B", "item_id": 0}])
    metadata = tmp_path / "meta.json"
    with metadata.open("w") as handle:
        for row in [
            {"asin": "A", "category": ["Musical Instruments", "Guitar &amp; Bass Accessories", "Picks"]},
            {"asin": "A", "category": ["Musical Instruments", "Guitar & Bass Accessories", "Picks"]},
            {"asin": "B", "category": []},
        ]:
            handle.write(json.dumps(row) + "\n")
    assert category_paths(metadata, item_asins([split])) == [
        None, "Musical Instruments > Guitar & Bass Accessories > Picks"
    ]


def test_alignment_adds_loss_and_skips_unlabeled_items():
    config = Qwen2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4,
    )
    config.category_alignment = {"dimension": 8, "weight": 0.25}
    model = CategoryAlignedLatentModel(config)
    model.set_category_targets(torch.nn.functional.normalize(torch.randn(2, 8), dim=-1), torch.tensor([True, False]))
    input_ids = torch.tensor([[1, 2, 47, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 3]])

    aligned = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    category_item_id=torch.tensor([0]))
    ce_only = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.allclose(aligned.loss, ce_only.loss + 0.25 * model.last_align_loss, atol=1e-5)
    aligned.loss.backward()
    assert model.category_projection.weight.grad is not None
    assert model.attention.query.weight.grad is not None

    missing = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    category_item_id=torch.tensor([1]))
    assert model.last_align_loss.item() == 0
    assert torch.allclose(missing.loss, ce_only.loss, atol=1e-5)


def test_trainer_reports_eval_loss_for_category_model(tmp_path):
    config = Qwen2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4,
    )
    config.category_alignment = {"dimension": 8, "weight": 0.1}
    model = CategoryAlignedLatentModel(config)
    model.set_category_targets(torch.nn.functional.normalize(torch.randn(1, 8), dim=-1), torch.tensor([True]))
    example = {
        "input_ids": torch.tensor([1, 2, 47, 3]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, -100, 3]),
        "category_item_id": torch.tensor(0),
    }
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), per_device_eval_batch_size=1,
                               label_names=["labels"], remove_unused_columns=False,
                               report_to="none"),
        eval_dataset=[example],
        data_collator=lambda rows: {key: torch.stack([row[key] for row in rows]) for key in rows[0]},
    )
    assert "eval_loss" in trainer.evaluate()
