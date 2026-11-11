"""Train MWC-M: margin + WIDER warm-up, then same-motive InfoNCE (motive_contrastive.py)."""

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
from latent_attention_train_grounded_margin import CATEGORY_TEXT, GroundedLogCallback
from motive_contrastive import MotiveDataset, MotiveMarginLatentModel, build_positive_index


class MotivePhaseCallback(transformers.TrainerCallback):
    """Contrastive term on once ``warmup_epochs`` epochs have elapsed (from state.epoch, so resumes land right)."""

    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs
        self.announced = None

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        target = model.module if hasattr(model, "module") else model
        enabled = (state.epoch or 0.0) >= self.warmup_epochs - 1e-6
        target.contrastive_enabled = enabled
        if enabled != self.announced:
            print(f"[phase] epoch {state.epoch or 0.0:.3f}: motive InfoNCE {'ON' if enabled else 'off'}; "
                  f"bank filled {float(target.bank_filled.float().mean()):.3f}", flush=True)
            self.announced = enabled


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "",
    eval_file: str = "",
    item_emb_file: str = "",
    output_dir: str = "./output_test/",
    category: str = "Toys_and_Games",
    sample: int = -1,
    eval_sample: int = None,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    K: int = 0,
    margin: float = 0.2,
    margin_weight: float = 1.0,
    wider_weight: float = 0.1,
    wider_momentum: float = 0.99,
    warmup_epochs: int = 1,
    motive_weight: float = 0.1,
    temperature: float = 0.1,
    within_threshold: float = 0.8,
    max_positives: int = 16,
    num_negatives: int = 256,
    projection_dim: int = 256,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
):
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    grad_accum = batch_size // micro_batch_size
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in special:
        special.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    text = CATEGORY_TEXT[category]

    item_emb = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"].float()
    index, items, users, stats = build_positive_index(train_file, item_emb, within_threshold)
    print(f"motive positives (threshold {within_threshold}): {stats}", flush=True)

    train_data = MotiveDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                               sample=sample, seed=seed, category=text, K=K)
    val_data = MotiveDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample if eval_sample is None else eval_sample, category=text, K=K)
    to_hf = lambda data: HFDataset.from_dict({k: [row[k] for row in data] for k in data[0]})
    hf_train = to_hf(train_data)
    # Validation rows are not train rows: no bank id (contrastive term is train-only anyway).
    hf_val = to_hf(val_data).remove_columns(["row_id"])

    config = AutoConfig.from_pretrained(base_model)
    config.latent_motive = {"bank_size": int(stats["rows"]), "projection_dim": projection_dim,
                            "temperature": temperature, "weight": motive_weight,
                            "max_positives": max_positives, "num_negatives": num_negatives,
                            "within_threshold": within_threshold, "warmup_epochs": warmup_epochs}
    model = MotiveMarginLatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    model.projection.float()
    model.set_positive_index(index, items, users)
    model.grad_accum_steps = grad_accum
    model.config.loss_type = "ce"
    model.margin, model.margin_weight = margin, margin_weight
    if wider_weight > 0:
        model.config.latent_wider = {"weight": float(wider_weight), "momentum": float(wider_momentum)}
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    print(f"MWC-M: margin={margin} x{margin_weight}; WIDER={getattr(model.config, 'latent_wider', None)}; "
          f"motive={config.latent_motive}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size, per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum, warmup_steps=20, num_train_epochs=num_epochs,
            learning_rate=learning_rate, bf16=True, logging_steps=1, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, output_dir=output_dir,
            load_best_model_at_end=False, report_to=wandb_report_to(), remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[GroundedLogCallback(), MotivePhaseCallback(warmup_epochs)],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
