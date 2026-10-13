"""Train plain R3 with a margin-only phase, then margin + collaborative InfoNCE."""

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

from collaborative_contrastive_dataset import CollaborativeContrastiveDataset
from collaborative_margin_attention import CollaborativeMarginLatentModel
from experiment_tracking import init_training_run, move_callback_to_end, wandb_report_to
from latent_attention_train_collaborative import CollaborativeLogCallback


class CollaborativeMarginLogCallback(CollaborativeLogCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attr, key in (
            ("last_margin_loss", "margin_loss"),
            ("last_contrastive_enabled", "contrastive_enabled"),
        ):
            value = getattr(target, attr, None)
            logs[key] = value.item() if value is not None else None


class PhaseCallback(transformers.TrainerCallback):
    """Enable InfoNCE once ``margin_only_epochs`` full epochs have elapsed.

    Derived from ``state.epoch`` on every step, so a resumed run lands in the
    correct phase.
    """

    def __init__(self, margin_only_epochs, phase2_margin_weight):
        self.margin_only_epochs = margin_only_epochs
        self.phase2_margin_weight = phase2_margin_weight
        self.announced = None

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        target = model.module if hasattr(model, "module") else model
        enabled = (state.epoch or 0.0) >= self.margin_only_epochs - 1e-6
        target.contrastive_enabled = enabled
        if enabled:
            target.margin_weight = self.phase2_margin_weight
        if enabled != self.announced:
            print(
                f"[phase] epoch={state.epoch:.3f} step={state.global_step}: "
                f"contrastive_enabled={enabled}, margin_weight={target.margin_weight}",
                flush=True,
            )
            self.announced = enabled


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    category: str = "Toys_and_Games",
    K: int = 0,
    margin_only_epochs: int = 1,
    margin: float = 0.2,
    margin_weight: float = 1.0,
    phase2_margin_weight: float = None,
    projection_dim: int = 256,
    temperature: float = 0.1,
    user_contrastive_weight: float = 0.1,
    target_contrastive_weight: float = 0.1,
    bank_size: int = 10_000,
    max_user_positives: int = 16,
    max_target_positives: int = 16,
    num_negatives: int = 256,
    positive_min_token_accuracy: float = 1.0,
    positive_max_nll: float = float("inf"),
    wandb_run_name: str = None,
):
    if margin_weight > 0 and micro_batch_size < 2:
        raise ValueError("margin training requires micro_batch_size >= 2; accumulation does not create shuffle donors")
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if phase2_margin_weight is None:
        phase2_margin_weight = margin_weight
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Collaborative margin R3 is single-GPU only (local FIFO corpus).")

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

    config = AutoConfig.from_pretrained(base_model)
    config.collaborative_contrastive = {"projection_dim": projection_dim, "bank_size": bank_size}
    model = CollaborativeMarginLatentModel.from_pretrained(
        base_model, config=config, torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.margin = margin
    model.margin_weight = margin_weight
    model.contrastive_enabled = margin_only_epochs <= 0
    model.temperature = temperature
    model.user_contrastive_weight = user_contrastive_weight
    model.target_contrastive_weight = target_contrastive_weight
    model.cf_contrastive_weight = 0.0
    model.max_user_positives = max_user_positives
    model.max_target_positives = max_target_positives
    model.num_negatives = num_negatives
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
        "Plain-R3 margin -> collaborative InfoNCE: "
        f"epochs={num_epochs}, margin_only_epochs={margin_only_epochs}, "
        f"margin={margin}, margin_weight={margin_weight}, "
        f"phase2_margin_weight={phase2_margin_weight}, queue={bank_size}, "
        f"projection={projection_dim}, temperature={temperature}, "
        f"user_weight={user_contrastive_weight}, target_weight={target_contrastive_weight}, "
        f"min_token_accuracy={positive_min_token_accuracy}, max_nll={positive_max_nll}"
    )
    train_data = CollaborativeContrastiveDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_text, K=K,
    )
    val_data = CollaborativeContrastiveDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, category=category_text, K=K,
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
        callbacks=[
            PhaseCallback(margin_only_epochs, phase2_margin_weight),
            CollaborativeMarginLogCallback(),
        ],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
