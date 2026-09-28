"""Train the isolated vocabulary-native soft-latent R3 ablation."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from experiment_tracking import init_training_run, wandb_report_to


class SoftVocabularyLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attribute, key in (
            ("last_soft_vocab_entropy", "soft_vocab_entropy"),
            ("last_soft_vocab_max_probability", "soft_vocab_max_probability"),
            ("last_soft_vocab_raw_norm", "raw_latent_norm"),
            ("last_soft_vocab_norm", "soft_latent_norm"),
        ):
            value = getattr(target, attribute, None)
            if value is not None:
                logs[key] = value.item()


def train(
    base_model: str = "/data1/huggingface/models--Qwen--Qwen2.5-1.5B/",
    train_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    sample: int = 1024,
    seed: int = 0,
    end_k: int = -1,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    local_rank: int = 0,
    deepspeed: str = "./deepspeed.json",
    category: str = "Toys_and_Games",
    K: int = 0,
    soft_vocab_top_k: int = 64,
    soft_vocab_temperature: float = 1.0,
    wandb_run_name: str = None,
):
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    category_dict = {
        "Office_Products": "office products", "Books": "books",
        "steam": "games", "CDs_and_Vinyl": "musics",
        "Toys_and_Games": "toys and games", "Video_Games": "video games",
        "Musical_Instruments": "music instruments",
        "Sports_and_Outdoors": "sports and outdoors",
        "Pet_Supplies": "pet supplies",
        "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
        "yelp": "resturant",
    }
    category_text = category_dict[category]
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps //= world_size

    if soft_vocab_top_k <= 0:
        raise ValueError("soft_vocab_top_k must be positive")
    if soft_vocab_temperature <= 0:
        raise ValueError("soft_vocab_temperature must be positive")

    config = AutoConfig.from_pretrained(base_model)
    config.latent_soft_vocabulary = {
        "top_k": soft_vocab_top_k,
        "temperature": soft_vocab_temperature,
        # The last vocabulary entry is the newly initialized <|Thought|>
        # token. It must not become a generic one-token shortcut.
        "exclude_last_n": 1,
    }
    model = LatentModel.from_pretrained(
        base_model, config=config, torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in additional_special_tokens:
        additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1

    print(
        "Vocabulary-native soft latent only: "
        f"top_k={soft_vocab_top_k}, temperature={soft_vocab_temperature}, "
        f"exclude_last_n=1, epochs={num_epochs}, added_tokens={num_added_toks}"
    )

    train_data = LatentRDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_text, K=K,
    )
    val_data = LatentRDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, category=category_text, K=K,
    )
    hf_train_dataset = HFDataset.from_dict(
        {key: [value[key] for value in train_data] for key in train_data[0]}
    )
    hf_val_dataset = HFDataset.from_dict(
        {key: [value[key] for value in val_data] for key in val_data[0]}
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
            save_total_limit=5,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=1),
            SoftVocabularyLogCallback(),
        ],
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
