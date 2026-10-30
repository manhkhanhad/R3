"""Full-parameter Qwen SFT baseline without the R3 latent reasoning module."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from experiment_tracking import init_training_run, wandb_report_to
from plain_sft_dataset import PlainSFTDataset


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "data/Musical_Instruments/train/Musical_Instruments_5_2015-1-2018-11.csv",
    eval_file: str = "data/Musical_Instruments/valid/Musical_Instruments_5_2015-1-2018-11.csv",
    output_dir: str = "/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_plain_sft/model",
    sample: int = -1,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    cutoff_len: int = 512,
    resume_from_checkpoint: str = None,
    category: str = "music instruments",
    dataset_name: str = "Musical_Instruments",
    instruction: str = None,
):
    if batch_size < micro_batch_size or batch_size % micro_batch_size:
        raise ValueError("batch_size must be a multiple of micro_batch_size")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This SFT launcher expects one GPU")

    init_training_run(
        dataset=dataset_name,
        config={
            "method": "plain_sft",
            "base_model": base_model,
            "train_file": train_file,
            "eval_file": eval_file,
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "dataset": dataset_name, "instruction": instruction,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    train_data = PlainSFTDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category, instruction=instruction,
    )
    eval_data = PlainSFTDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category, instruction=instruction,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=20,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            report_to=wandb_report_to(),
            remove_unused_columns=False,
            seed=seed,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
