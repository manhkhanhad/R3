"""Train margin + WIDER + behaviour-grounded donors R3 (grounded_margin.py)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq

from experiment_tracking import init_training_run, move_callback_to_end, wandb_report_to
from grounded_reasoning import GroundedReasoningDataset
from grounded_margin import MarginGroundedLatentModel


class GroundedLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None:
            return
        target = model.module if hasattr(model, "module") else model
        for key, value in getattr(target, "last_losses", {}).items():
            logs[key] = value.item() if torch.is_tensor(value) else value


CATEGORY_TEXT = {
    "Office_Products": "office products", "Books": "books", "steam": "games",
    "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games",
    "Video_Games": "video games", "Musical_Instruments": "music instruments",
    "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
    "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant",
}


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
    topn: int = 32,
    explore: int = 8,
    key_dim: int = 256,
    value_hidden: int = 1024,
    queue_size: int = 20_000,
    init_scale: float = 0.5,
    tau_item: float = 0.07,
    tau_util: float = 0.5,
    item_weight: float = 0.2,
    retrieval_weight: float = 0.1,
    bank_batch_size: int = 8,
    gradient_checkpointing: bool = False,
    margin: float = 0.2,
    margin_weight: float = 1.0,
    wider_weight: float = 0.1,
    wider_momentum: float = 0.99,
    slot_bank_rows: int = 10000,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
):
    if margin_weight > 0 and micro_batch_size < 2:
        raise ValueError("margin needs micro_batch_size >= 2 (in-batch shuffle donors)")
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("grounded reasoning is single-GPU only (local FIFO donors)")
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir,
                                                "dataset": category})
    gradient_accumulation_steps = batch_size // micro_batch_size

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in special:
        special.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    text = CATEGORY_TEXT[category]
    train_data = GroundedReasoningDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                                          sample=sample, seed=seed, category=text, K=K)
    val_data = GroundedReasoningDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                                        sample=sample if eval_sample is None else eval_sample,
                                        category=text, K=K)
    to_hf = lambda data: HFDataset.from_dict({k: [row[k] for row in data] for k in data[0]})
    hf_train, hf_val = to_hf(train_data), to_hf(val_data)
    item_table = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"].float()
    config = AutoConfig.from_pretrained(base_model)
    config.latent_grounded_reasoning = {
        "topn": topn, "explore": explore, "key_dim": key_dim, "value_hidden": value_hidden,
        "queue_size": queue_size, "init_scale": init_scale, "tau_item": tau_item,
        "tau_util": tau_util, "item_weight": item_weight, "retrieval_weight": retrieval_weight,
        "n_items": int(item_table.size(0)), "item_dim": int(item_table.size(1)),
        "bank_size": len(hf_train), "item_emb_file": os.path.abspath(item_emb_file),
    }
    model = MarginGroundedLatentModel.from_pretrained(base_model, config=config,
                                                         torch_dtype=torch.bfloat16)
    layer = model.attention
    layer.reset_new_parameters(init_scale)
    layer.item_emb.copy_(item_table.to(layer.item_emb.dtype))
    model.keep_new_modules_fp32()
    model.grad_accum_steps = gradient_accumulation_steps
    model.config.loss_type = "ce"
    model.margin, model.margin_weight = margin, margin_weight
    if wider_weight > 0:
        model.config.latent_wider = {"weight": float(wider_weight), "momentum": float(wider_momentum)}
    print(f"margin={margin} x{margin_weight}; WIDER={getattr(model.config, 'latent_wider', None)}; "
          f"evidence_scale dtype={layer.evidence_scale.dtype}", flush=True)

    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    print(f"Grounded R3: {config.latent_grounded_reasoning}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20, num_train_epochs=num_epochs, learning_rate=learning_rate,
            bf16=True, logging_steps=1, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, output_dir=output_dir,
            load_best_model_at_end=False, gradient_checkpointing=gradient_checkpointing,
            report_to=wandb_report_to(), remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[GroundedLogCallback()],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Inference donors: the final model's own-history latent for every train row.
    loader = DataLoader(hf_train, batch_size=bank_batch_size, shuffle=False, collate_fn=collator)
    filled = model.build_inference_bank(loader, next(model.parameters()).device)
    print(f"inference bank filled with {filled} train rows", flush=True)
    # Grounded slot vectors r1 of train rows: mean / shuffle_user interventions and geometry.
    subset = torch.randperm(len(hf_train), generator=torch.Generator().manual_seed(0))[:slot_bank_rows].tolist()
    slot_loader = DataLoader(hf_train.select(subset), batch_size=bank_batch_size, shuffle=False, collate_fn=collator)
    vectors, users = model.build_slot_bank(slot_loader, next(model.parameters()).device, slot_bank_rows)
    torch.save(vectors.half(), os.path.join(output_dir, "slot_bank.pt"))
    torch.save(users, os.path.join(output_dir, "slot_bank_users.pt"))
    print(f"slot bank: {tuple(vectors.shape)}, evidence_scale={float(layer.evidence_scale):.4f}", flush=True)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
