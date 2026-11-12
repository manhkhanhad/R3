"""Attempt 11: train plain R3 with SASRec-similarity-weighted collaborative InfoNCE."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq

from experiment_tracking import init_training_run, move_callback_to_end, wandb_report_to
from sasrec_collaborative_attention import SASRecCollaborativeLatentModel
from sasrec_collaborative_dataset import SASRecCollaborativeDataset
from sasrec_pair_features import build_sasrec_features, similarity_report


class SASRecCollaborativeLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attr, key in (
            ("last_ce_loss", "ce_loss"),
            ("last_sequence_contrastive_loss", "sequence_infonce_loss"),
            ("last_item_contrastive_loss", "item_infonce_loss"),
            ("last_sequence_positive_rate", "sequence_positive_rate"),
            ("last_item_positive_rate", "item_positive_rate"),
            ("last_sequence_positive_similarity", "sequence_positive_similarity"),
            ("last_item_positive_similarity", "item_positive_similarity"),
            ("last_queue_accept_rate", "queue_accept_rate"),
            ("last_queue_count", "queue_count"),
        ):
            value = getattr(target, attr, None)
            logs[key] = value.item() if value is not None else None


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "data/Musical_Instruments/train/Musical_Instruments_5_2015-1-2018-11.csv",
    eval_file: str = "data/Musical_Instruments/valid/Musical_Instruments_5_2015-1-2018-11.csv",
    sasrec_checkpoint: str = "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/Musical_Instruments_5_2015-1-2018-11/models/SASRec/2026-09-26_00-07-12-933295.pkl",
    output_dir: str = "./output_test/",
    sample: int = -1,
    seed: int = 0,
    end_k: int = -1,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    category: str = "Musical_Instruments",
    K: int = 0,
    projection_dim: int = 256,
    temperature: float = 0.1,
    sequence_contrastive_weight: float = 0.1,
    item_contrastive_weight: float = 0.1,
    bank_size: int = 10_000,
    max_sequence_positives: int = 16,
    max_item_positives: int = 16,
    num_negatives: int = 256,
    min_positive_similarity: float = 0.3,
    max_negative_similarity: float = 0.2,
    similarity_weight_power: float = 1.0,
    exclude_same_user_positives: bool = False,
    positive_min_token_accuracy: float = 1.0,
    positive_max_nll: float = float("inf"),
):
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    category_dict = {
        "Office_Products": "office products", "Books": "books",
        "steam": "games", "CDs_and_Vinyl": "musics",
        "Toys_and_Games": "toys and games", "Video_Games": "video games",
        "Musical_Instruments": "music instruments",
        "Sports_and_Outdoors": "sports and outdoors",
        "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products",
        "Movies": "movie", "yelp": "resturant",
    }
    category_text = category_dict[category]
    gradient_accumulation_steps = batch_size // micro_batch_size
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("SASRec collaborative R3 is single-GPU only: its FIFO corpus is run-local.")
    if max_negative_similarity > min_positive_similarity:
        raise ValueError("max_negative_similarity must not exceed min_positive_similarity")

    item_table, sasrec_rows = build_sasrec_features(sasrec_checkpoint, [train_file, eval_file])
    train_sequences, train_targets = sasrec_rows[train_file]
    unknown = int((train_targets < 0).sum())
    print(f"SASRec features: items={item_table.shape}, train_rows={len(train_targets)}, unknown_targets={unknown}", flush=True)
    print(f"SASRec similarity: {similarity_report(item_table, train_sequences, train_targets, bank=bank_size, topk=max_sequence_positives)}", flush=True)

    config = AutoConfig.from_pretrained(base_model)
    config.sasrec_collaborative = {
        "projection_dim": projection_dim,
        "bank_size": bank_size,
        "sasrec_dim": item_table.shape[1],
        "num_sasrec_items": item_table.shape[0],
    }
    model = SASRecCollaborativeLatentModel.from_pretrained(
        base_model, config=config, torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.sasrec_item_table = torch.as_tensor(item_table, dtype=torch.float32)
    model.temperature = temperature
    model.sequence_contrastive_weight = sequence_contrastive_weight
    model.item_contrastive_weight = item_contrastive_weight
    model.max_sequence_positives = max_sequence_positives
    model.max_item_positives = max_item_positives
    model.num_negatives = num_negatives
    model.min_positive_similarity = min_positive_similarity
    model.max_negative_similarity = max_negative_similarity
    model.similarity_weight_power = similarity_weight_power
    model.exclude_same_user_positives = exclude_same_user_positives
    model.positive_min_token_accuracy = positive_min_token_accuracy
    model.positive_max_nll = positive_max_nll
    model.grad_accum_steps = gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in additional_special_tokens:
        additional_special_tokens.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1

    print(
        "Plain-R3 SASRec collaborative InfoNCE (Attempt 11): "
        f"epochs={num_epochs}, queue={bank_size}, projection={projection_dim}, "
        f"temperature={temperature}, sequence_weight={sequence_contrastive_weight}, "
        f"item_weight={item_contrastive_weight}, min_pos_sim={min_positive_similarity}, "
        f"max_neg_sim={max_negative_similarity}, weight_power={similarity_weight_power}, "
        f"exclude_same_user={exclude_same_user_positives}, "
        f"min_token_accuracy={positive_min_token_accuracy}, max_nll={positive_max_nll}",
        flush=True,
    )
    dataset_kwargs = dict(tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category_text, K=K)
    train_data = SASRecCollaborativeDataset(
        train_file=train_file, seed=seed,
        sasrec_sequence_vectors=train_sequences, sasrec_targets=train_targets,
        **dataset_kwargs,
    )
    val_data = SASRecCollaborativeDataset(
        train_file=eval_file,
        sasrec_sequence_vectors=sasrec_rows[eval_file][0], sasrec_targets=sasrec_rows[eval_file][1],
        **dataset_kwargs,
    )
    hf_train_dataset = HFDataset.from_dict(
        {key: [value[key] for value in train_data] for key in train_data[0]}
    )
    hf_val_dataset = HFDataset.from_dict(
        {key: [value[key] for value in val_data] for key in val_data[0]}
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[SASRecCollaborativeLogCallback()],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
