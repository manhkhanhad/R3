"""Train disentangled-interest R3: K reasoning paths, posterior-credited decoder."""

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
from intent_reasoning import IntentDataset, IntentLatentModel

CATEGORY_TEXT = {
    "Office_Products": "office products", "Books": "books", "steam": "games",
    "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games",
    "Video_Games": "video games", "Musical_Instruments": "music instruments",
    "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
    "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant",
}


class IntentCallback(transformers.TrainerCallback):
    """Explore (sample the credited path) early; log intent diagnostics."""

    def __init__(self, explore_epochs):
        self.explore_epochs = explore_epochs

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        target = model.module if hasattr(model, "module") else model
        target.attention.explore = (state.epoch or 0.0) < self.explore_epochs

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None:
            return
        target = model.module if hasattr(model, "module") else model
        for key, value in getattr(target, "last_losses", {}).items():
            logs[key] = value.item() if torch.is_tensor(value) else value


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "",
    eval_file: str = "",
    item_emb_file: str = "",
    output_dir: str = "./output_test/",
    category: str = "Musical_Instruments",
    sample: int = -1,
    eval_sample: int = None,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    K: int = 0,
    n_intents: int = 4,
    path_hidden: int = 1024,
    init_scale: float = 0.5,
    tau_item: float = 0.07,
    mixture_weight: float = 0.2,
    prior_weight: float = 0.1,
    balance_weight: float = 0.01,
    margin: float = 0.2,
    margin_weight: float = 0.0,
    explore_epochs: float = 0.1,
    posterior_uses_prior: bool = False,
    gradient_checkpointing: bool = False,
    wandb_run_name: str = None,
):
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir,
                                                "dataset": category})
    gradient_accumulation_steps = batch_size // micro_batch_size

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in special:
        special.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    text = CATEGORY_TEXT[category]
    train_data = IntentDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                               sample=sample, seed=seed, category=text, K=K)
    val_data = IntentDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample if eval_sample is None else eval_sample, category=text, K=K)
    to_hf = lambda data: HFDataset.from_dict({k: [row[k] for row in data] for k in data[0]})
    hf_train, hf_val = to_hf(train_data), to_hf(val_data)

    item_table = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"].float()
    config = AutoConfig.from_pretrained(base_model)
    config.latent_intent_reasoning = {
        "n_intents": n_intents, "path_hidden": path_hidden,
        "init_scale": init_scale, "tau_item": tau_item, "mixture_weight": mixture_weight,
        "prior_weight": prior_weight, "balance_weight": balance_weight, "margin": margin,
        "margin_weight": margin_weight, "posterior_uses_prior": posterior_uses_prior, "n_items": int(item_table.size(0)),
        "item_dim": int(item_table.size(1)), "item_emb_file": os.path.abspath(item_emb_file),
    }
    model = IntentLatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    model.attention.reset_new_parameters(init_scale)
    model.attention.item_emb.copy_(item_table.to(model.attention.item_emb.dtype))
    model.grad_accum_steps = gradient_accumulation_steps
    model.config.loss_type = "ce"
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    print(f"Intent R3: {config.latent_intent_reasoning}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20, num_train_epochs=num_epochs, learning_rate=learning_rate,
            bf16=True, logging_steps=1, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="no", output_dir=output_dir,
            load_best_model_at_end=False, gradient_checkpointing=gradient_checkpointing,
            report_to=wandb_report_to(), remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[IntentCallback(explore_epochs)],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
