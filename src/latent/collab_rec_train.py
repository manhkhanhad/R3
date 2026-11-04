"""Train CollabRec on R3 (thought = attention over collaborative supports; NTP loss only).

Same protocol as latent_attention_train.py: HF Trainer, warmup 20, epoch eval/save,
best checkpoint by valid loss, early stopping patience 1.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback

from collab_rec_dataset import (SOURCES, TARGET_FORMATS, CollabLatentRDataset, build_support_table, default_collab_dir,
                                load_support_rows)
from collab_rec_latent import CollabLatentModel
from experiment_tracking import init_training_run, wandb_report_to

CATEGORY = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics",
            "Toys_and_Games": "toys and games", "Video_Games": "video games",
            "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
            "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
            "yelp": "resturant"}


def collab_spec(category, support_source, support_mode, support_k, collab_dir, rationale_tag, max_support_len,
                target_format="r3"):
    if support_source not in SOURCES:
        raise ValueError(f"support_source must be one of {SOURCES}")
    if support_source == "oracle_motive":
        support_k = 2
    if support_source == "none":
        support_k = 0
    return {"support_source": support_source, "support_mode": support_mode, "support_k": int(support_k),
            "collab_dir": os.path.abspath(collab_dir or default_collab_dir(category)),
            "rationale_tag": rationale_tag, "max_support_len": int(max_support_len),
            "target_format": target_format}


class StopAfterSteps(transformers.TrainerCallback):
    """Check runs: stop after N optimizer steps while keeping the full-run LR schedule."""

    def __init__(self, steps):
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.steps:
            control.should_training_stop = True
        return control


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B",
    train_file: str = "data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/Toys_and_Games/valid/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    category: str = "Toys_and_Games",
    sample: int = -1,
    eval_sample: int = None,
    seed: int = 0,
    # CollabRec
    support_source: str = "peer",  # peer | motive | oracle_motive | none
    support_mode: str = "text",  # text | emb
    support_k: int = 10,
    collab_dir: str = None,  # default: <IntuRec>/data/<category>/collab
    rationale_tag: str = "v5",
    max_support_len: int = 64,
    # training
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    resume_from_checkpoint: str = None,
    save_only_model: bool = True,
    save_total_limit: int = 2,
    wandb_run_name: str = None,
    # IntuRec-matching options (defaults = R3 protocol)
    target_format: str = "r3",  # r3: '"title"\n' + <|endoftext|> | end_predict: '"title"' + <|EndPredict|> (IntuRec)
    loss_average: str = "batch",  # batch: token mean over the 128-row batch (HF) | micro_batch: mean of micro-batch means (IntuRec)
    stop_after_steps: int = 0,  # >0: check run, stop after N optimizer steps (full LR schedule), no eval/save
):
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"target_format must be one of {TARGET_FORMATS}")
    if loss_average not in ("batch", "micro_batch"):
        raise ValueError("loss_average must be batch or micro_batch")
    spec = collab_spec(category, support_source, support_mode, support_k, collab_dir, rationale_tag,
                       max_support_len, target_format)
    init_training_run(dataset=category, run_name=wandb_run_name,
                      config={"base_model": base_model, "train_file": train_file, "eval_file": eval_file,
                              "output_dir": output_dir, "dataset": category, "collab_rec": spec})
    print("collab_rec:", spec)
    category_text = CATEGORY[category]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size // world_size

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # <|Thought|> must stay the last token id (the model finds it as num_embeddings - 1)
    extra = ["<|EndPredict|>"] if target_format == "end_predict" else []
    tokens = tokenizer.additional_special_tokens + extra + ["<|Thought|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    table, mask = build_support_table(spec, tokenizer)
    config = AutoConfig.from_pretrained(base_model)
    if support_mode == "emb" and table is not None:
        spec["rationale_dim"] = int(table.shape[1])
    config.collab_rec = spec
    model = CollabLatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    model.config.loss_type = "ce"
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    if target_format == "end_predict":
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EndPredict|>")
    # IntuRec averaging: without loss kwargs the Trainer divides each micro-batch's mean by the accumulation steps
    model.accepts_loss_kwargs = loss_average == "batch"
    print(f"target_format={target_format} loss_average={loss_average} micro_batch_size={micro_batch_size} "
          f"grad_accum={gradient_accumulation_steps}")
    if table is not None:
        model.set_support_table(table, mask)
        print(f"support table {tuple(table.shape)}")

    def rows_for(path, split):
        rows = load_support_rows(spec, split)
        if rows is not None:
            n = len(pd.read_csv(path, usecols=[0]))
            assert len(rows) == n, f"{split}: {len(rows)} support rows for {n} CSV rows"
        return rows

    train_data = CollabLatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
                                      seed=seed, category=category_text, support_rows=rows_for(train_file, "train"),
                                      target_format=target_format)
    val_data = CollabLatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                                    sample=sample if eval_sample is None else eval_sample, category=category_text,
                                    support_rows=rows_for(eval_file, "valid"), target_format=target_format)
    print("train example keys:", train_data[0].keys(), "support_ids:", train_data[0].get("support_ids"))
    ex = train_data[0]
    print("train example target tokens:", [tokenizer.decode([t]) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])

    check = stop_after_steps > 0
    from datasets import Dataset as HFDataset
    hf_train = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_val,
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
            eval_strategy="no" if check else "epoch",
            save_strategy="no" if check else "epoch",
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
            load_best_model_at_end=not check,
            label_names=["labels"],
            ddp_find_unused_parameters=True if world_size > 1 else None,  # R3's unused self.attention
            report_to=wandb_report_to(),
            remove_unused_columns=False,
            seed=seed,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[StopAfterSteps(stop_after_steps)] if check else [EarlyStoppingCallback(early_stopping_patience=1)],
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if check:
        losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
        for s in range(10, len(losses) + 1, 10):
            print(f"check step {s}: running mean loss {sum(losses[:s]) / s:.4f}  step loss {losses[s - 1]:.4f}")
        return
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
