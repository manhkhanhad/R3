"""Train G3: margin + WIDER(r0) + CF-prior donor evidence with an evidence-swap margin (cf_grounded_margin.py)."""

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
from cf_grounded_margin import CFDonorIndex, CFGroundedDataset, CFGroundedMarginLatentModel
from latent_attention_train_grounded_margin import CATEGORY_TEXT, GroundedLogCallback


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
    key_dim: int = 256,
    value_hidden: int = 1024,
    init_scale: float = 0.5,
    tau_item: float = 0.07,
    tau_util: float = 0.1,
    item_weight: float = 0.2,
    retrieval_weight: float = 0.1,
    centre: bool = True,
    mean_momentum: float = 0.99,
    evidence_margin: float = 0.2,
    evidence_weight: float = 1.0,
    margin: float = 0.2,
    margin_weight: float = 1.0,
    wider_weight: float = 0.1,
    wider_momentum: float = 0.99,
    bank_batch_size: int = 8,
    slot_bank_rows: int = 10000,
    gradient_checkpointing: bool = False,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
):
    if micro_batch_size < 2:
        raise ValueError("G3 margins need micro_batch_size >= 2 (in-batch swaps)")
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

    index = CFDonorIndex(train_file, topn)
    CFGroundedDataset.donor_index = index
    train_data = CFGroundedDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                                   sample=sample, seed=seed, category=text, K=K)
    val_data = CFGroundedDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                                 sample=sample if eval_sample is None else eval_sample, category=text, K=K)
    to_hf = lambda data: HFDataset.from_dict({k: [row[k] for row in data] for k in data[0]})
    hf_train, hf_val = to_hf(train_data), to_hf(val_data)
    has = lambda d: sum(1 for x in d["donor_idx"] if x[0] >= 0) / len(d)
    print(f"CF donors: train rows with donor {has(hf_train):.3f}, valid {has(hf_val):.3f}; bank {len(index)}", flush=True)

    item_table = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"].float()
    config = AutoConfig.from_pretrained(base_model)
    config.latent_cf_grounded = {
        "topn": topn, "key_dim": key_dim, "value_hidden": value_hidden, "init_scale": init_scale,
        "tau_item": tau_item, "tau_util": tau_util, "item_weight": item_weight,
        "retrieval_weight": retrieval_weight, "centre": centre, "mean_momentum": mean_momentum,
        "evidence_margin": evidence_margin, "evidence_weight": evidence_weight,
        "n_items": int(item_table.size(0)), "item_dim": int(item_table.size(1)), "bank_size": len(index),
        "train_file": os.path.abspath(train_file), "item_emb_file": os.path.abspath(item_emb_file),
    }
    model = CFGroundedMarginLatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    layer = model.attention
    layer.reset_new_parameters(init_scale)
    layer.item_emb.copy_(item_table.to(layer.item_emb.dtype))
    layer.bank_item.copy_(torch.as_tensor(index.item))
    model.keep_new_modules_fp32()
    model.grad_accum_steps = gradient_accumulation_steps
    model.config.loss_type = "ce"
    model.margin, model.margin_weight = margin, margin_weight
    if wider_weight > 0:
        model.config.latent_wider = {"weight": float(wider_weight), "momentum": float(wider_momentum)}
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    print(f"G3: margin={margin} x{margin_weight}; evidence margin={evidence_margin} x{evidence_weight}; "
          f"WIDER(r0)={getattr(model.config, 'latent_wider', None)}; spec={config.latent_cf_grounded}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size, per_device_eval_batch_size=micro_batch_size,
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

    subset = torch.randperm(len(hf_train), generator=torch.Generator().manual_seed(0))[:slot_bank_rows].tolist()
    slot_loader = DataLoader(hf_train.select(subset), batch_size=bank_batch_size, shuffle=False, collate_fn=collator)
    vectors, users = model.build_slot_bank(slot_loader, next(model.parameters()).device, slot_bank_rows)
    torch.save(vectors.half(), os.path.join(output_dir, "slot_bank.pt"))
    torch.save(users, os.path.join(output_dir, "slot_bank_users.pt"))
    print(f"slot bank: {tuple(vectors.shape)}, evidence_scale={float(layer.evidence_scale):.4f}, "
          f"|evidence_mean|={float(layer.evidence_mean.norm()):.4f}", flush=True)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
