"""Latent-free SFT on the R3 prompt, optionally with each row's text rationale (rationale_sft_dataset.py).

Same recipe as R3's trainer (latent_attention_train.py): lr 3e-4, batch 128, 3 epochs, warmup 20,
bf16, evaluation each epoch, early stopping (patience 1), best checkpoint by validation loss.
Only the <|Thought|> latent is gone; mode=off is the latent-free control, mode=own adds the
row's own rationale. One GPU.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback

from experiment_tracking import init_training_run, wandb_report_to
from rationale_sft_dataset import RationaleSFTDataset
import rationale_text

CATEGORY_NAMES = {"Toys_and_Games": "toys and games", "Musical_Instruments": "music instruments",
                  "CDs_and_Vinyl": "musics", "Video_Games": "video games"}


def train(
    category: str = "Toys_and_Games",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    rationale_dir: str = "",
    mode: str = "own",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1536,
    resume_from_checkpoint: str = None,
):
    if not (train_file and eval_file and output_dir):
        raise ValueError("train_file, eval_file and output_dir are required")
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be a multiple of micro_batch_size")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This SFT launcher expects one GPU")
    train_rationales = eval_rationales = None
    if mode != "off":
        if not rationale_dir:
            raise ValueError(f"mode {mode!r} needs rationale_dir with train.json / valid.json")
        train_rationales = rationale_text.load(os.path.join(rationale_dir, "train.json"), train_file)
        eval_rationales = rationale_text.load(os.path.join(rationale_dir, "valid.json"), eval_file)

    init_training_run(
        dataset=category,
        config={"method": f"rationale_sft_{mode}", "base_model": base_model, "train_file": train_file,
                "eval_file": eval_file, "output_dir": output_dir, "rationale_dir": rationale_dir,
                "mode": mode, "learning_rate": learning_rate, "batch_size": batch_size,
                "cutoff_len": cutoff_len, "dataset": category},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    name = CATEGORY_NAMES[category]
    common = dict(tokenizer=tokenizer, max_len=cutoff_len, sample=-1, seed=seed, category=name, mode=mode)
    train_data = RationaleSFTDataset(train_file=train_file, rationales=train_rationales, **common)
    eval_data = RationaleSFTDataset(train_file=eval_file, rationales=eval_rationales, **common)
    lengths = sorted(len(x["input_ids"]) for x in train_data.inputs)
    print(f"train sequence length: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
          f"truncated at {cutoff_len}: {sum(l >= cutoff_len for l in lengths)}", flush=True)

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
            save_total_limit=1,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=wandb_report_to(),
            remove_unused_columns=False,
            seed=seed,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.config.use_cache = True
    model.save_pretrained(output_dir)   # best checkpoint (load_best_model_at_end)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
