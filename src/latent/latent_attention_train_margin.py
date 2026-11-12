"""Training entry point for the in-distribution counterfactual margin-loss ablation.

Copy of latent_attention_train.py that swaps in MarginLatentModel and exposes
--margin/--margin_weight. latent_attention_train.py itself is untouched; delete
this file plus margin_loss_attention.py to remove the approach entirely.

Unlike latent_attention_train_scheduled.py, this uses vanilla unrestricted
("both") attention throughout -- no attention masking curriculum. The margin
loss is computed directly under that same deployed configuration, so it
targets content-dependence (mean/shuffle vs raw) without relying on a training
proxy transferring to it.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
import fire
import torch

import transformers
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

from margin_loss_attention import MarginLatentModel
from reasoning_dataset import LatentRDataset
from experiment_tracking import init_training_run, wandb_report_to


class MarginLossLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        ce = getattr(target, "last_ce_loss", None)
        margin = getattr(target, "last_margin_loss", None)
        if ce is not None:
            logs["ce_loss"] = ce.item()
        if margin is not None:
            logs["margin_loss"] = margin.item()


def train(
    # model/data params
    base_model: str = "/data1/huggingface/models--Qwen--Qwen2.5-1.5B/",  # the only required argument
    train_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    sample: int = 1024,
    seed: int = 0,
    # attention hyper parameters
    end_k: int = -1,

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve

    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

    local_rank: int = 0,
    deepspeed: str = "./deepspeed.json",
    category: str = "Toys_and_Games",
    K: int = 0,

    # margin-loss hyperparameters
    margin: float = 0.2,
    margin_weight: float = 1.0,
    wandb_run_name: str = None,
    gradient_checkpointing: bool = False,
    eval_sample: int = None,
    early_stopping_patience: int = 1,
    load_best_model_at_end: bool = True,
    save_only_model: bool = False,
):
    if margin_weight > 0 and micro_batch_size < 2:
        raise ValueError("margin training requires micro_batch_size >= 2; accumulation does not create shuffle donors")
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")

    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    )
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = MarginLatentModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.margin = margin
    model.margin_weight = margin_weight
    # See margin_loss_attention.py: Trainer will not divide our custom loss by
    # gradient_accumulation_steps on its own (it assumes a model with **kwargs
    # in forward() already normalized via num_items_in_batch, which we don't
    # use), so we must do it ourselves or the effective gradient is
    # gradient_accumulation_steps times too large.
    model.grad_accum_steps = gradient_accumulation_steps

    ##############################################################################
    ##################
    ##################   add a special token here to represent latent thought
    ##################

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    print("We have added", num_added_toks, "tokens")
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    ##################
    ##############################################################################

    train_data = LatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K)
    val_data = LatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample if eval_sample is None else eval_sample, category=category, K=K)

    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    print("train_data 0 keys:", train_data[0].keys())
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
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
            load_best_model_at_end=load_best_model_at_end,
            save_only_model=save_only_model,
            gradient_checkpointing=gradient_checkpointing,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=([EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
                   if early_stopping_patience > 0 else []) + [MarginLossLogCallback()],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
