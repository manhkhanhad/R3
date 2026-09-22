"""Training entry point for Attempt 7c: EMA-verified 7b."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import fire
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq

from ema_verified_vae_attention import EMAVerifiedVAEDecomposedLatentModel
from reasoning_dataset import LatentRDataset


class EMATeacherUpdateCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        target = model.module if hasattr(model, "module") else model
        target.update_ema_teacher()


class VerifiedDecompositionLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attribute, key in (
            ("last_ce_loss", "ce_loss"),
            ("last_verify_loss", "verify_loss"),
            ("last_kl_loss", "kl_loss"),
            ("last_mu_consistency_loss", "mu_consistency_loss"),
            ("last_z_mean_sq", "z_mean_sq"),
            ("last_z_var", "z_var"),
        ):
            value = getattr(target, attribute, None)
            logs[key] = value.item() if value is not None else None
        logs["mu_ema_norm"] = target.attention.mu_ema.norm().item()
        logs["z_norm"] = (
            target.attention.last_z.detach().norm(dim=-1).mean().item()
            if target.attention.last_z is not None else None
        )
        logs["ema_teacher_updates"] = target.ema_teacher_updates


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
    rho_mu: float = 1.0,
    mu_ema_momentum: float = 0.99,
    kl_weight: float = 1.0,
    mu_consistency_weight: float = 0.5,
    z_queue_size: int = 256,
    kl_min_queue_size: int = 0,
    verify_weight: float = 0.05,
    teacher_ema_momentum: float = 0.999,
):
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
    if world_size != 1:
        raise RuntimeError(
            "Attempt 7c is single-GPU only: its z queue, mu EMA, and EMA teacher "
            "are intentionally local and unsynchronized."
        )
    if not 0 <= kl_min_queue_size <= z_queue_size:
        raise ValueError("kl_min_queue_size must be between 0 and z_queue_size")
    if not 0 <= teacher_ema_momentum < 1:
        raise ValueError("teacher_ema_momentum must be in [0, 1)")
    if verify_weight < 0:
        raise ValueError("verify_weight must be non-negative")

    config = AutoConfig.from_pretrained(base_model)
    config.latent_decomposition = {
        "kind": "vae", "rho_mu": rho_mu, "ema_momentum": mu_ema_momentum,
    }
    config.latent_ema_verification = {
        "verify_weight": verify_weight,
        "teacher_ema_momentum": teacher_ema_momentum,
        "history_blind": True,
        "shared_scaffold": "mu_ema",
    }
    model = EMAVerifiedVAEDecomposedLatentModel.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.kl_weight = kl_weight
    model.mu_consistency_weight = mu_consistency_weight
    model.z_queue_size = z_queue_size
    model.kl_min_queue_size = kl_min_queue_size
    model.verify_weight = verify_weight
    model.teacher_ema_momentum = teacher_ema_momentum
    model.grad_accum_steps = gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in additional_special_tokens:
        additional_special_tokens.append("<|Thought|>")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1

    # Initialize only after checkpoint loading and tokenizer resize so teacher
    # and student start from identical decoder weights and vocabulary shapes.
    model.initialize_ema_teacher()
    print(
        "Attempt 7c EMA-verified 7b: student_mode=both, "
        f"rho_mu={rho_mu}, kl_weight={kl_weight}, "
        f"mu_consistency_weight={mu_consistency_weight}, "
        f"verify_weight={verify_weight}, teacher_ema={teacher_ema_momentum}, "
        f"z_queue_size={z_queue_size}, kl_min_queue_size={kl_min_queue_size}"
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
            save_total_limit=1,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to="none",
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
        callbacks=[EMATeacherUpdateCallback(), VerifiedDecompositionLogCallback()],
    )
    model.config.use_cache = False
    model.ema_teacher.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
