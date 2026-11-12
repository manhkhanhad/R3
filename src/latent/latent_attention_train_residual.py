"""Train Attempt 9: a baseline-preserving latent residual correction.

Start from a strong completed R3 checkpoint.  Its decoder/backbone are frozen,
and its train-set mean thought is installed as the fixed ``mu`` at the thought
slot.  A new zero-initialized encoder learns ``z_i`` from detached history
states, giving the unchanged second pass ``F(x_i, mu + z_i)``.

Only the residual encoder is trainable.  At initialization ``z_i == 0`` for
every row, so the model exactly reproduces the source checkpoint's mean-thought
condition; training cannot damage the source decoder or direct history path.
"""

from __future__ import annotations

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from residual_correction_attention import ResidualCorrectionAttentionLayer
from experiment_tracking import init_training_run, wandb_report_to


class ResidualLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        attention = target.attention
        z = getattr(attention, "last_z", None)
        mu = getattr(attention, "last_mu", None)
        logs["residual_z_norm"] = (
            z.detach().norm(dim=-1).mean().item() if z is not None else None
        )
        logs["base_mu_norm"] = (
            mu.detach().norm(dim=-1).mean().item() if mu is not None else None
        )
        if z is not None and mu is not None:
            mu_unit = torch.nn.functional.normalize(mu.detach().float(), dim=-1, eps=1e-8)
            z_float = z.detach().float()
            parallel_sq = ((z_float * mu_unit).sum(dim=-1) ** 2).mean()
            total_sq = z_float.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
            logs["residual_parallel_energy_fraction"] = (parallel_sq / total_sq).item()


def _load_train_mean(path: str, hidden_size: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("thoughts", "vectors", "embeddings"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor) or obj.ndim != 2:
        raise ValueError(f"{path} must contain a [N, H] thought tensor")
    if obj.size(1) != hidden_size:
        raise ValueError(
            f"{path} has hidden size {obj.size(1)}; expected {hidden_size}"
        )
    return obj.float().mean(dim=0)


def _install_residual_wrapper(
    model: LatentModel,
    train_mean: torch.Tensor,
    bottleneck_size: int,
    residual_scale: float,
    orthogonal_to_mu: bool,
    end_k: int,
) -> None:
    if getattr(model.config, "latent_residual_correction", None):
        raise ValueError("base_model is already an Attempt-9 residual checkpoint")

    spec = {
        "bottleneck_size": bottleneck_size,
        "residual_scale": residual_scale,
        "use_fixed_mu": True,
        "orthogonal_to_mu": orthogonal_to_mu,
        "end_k": end_k,
    }
    wrapper = ResidualCorrectionAttentionLayer(
        base_attention=model.attention,
        hidden_size=model.config.hidden_size,
        **spec,
    )

    # The source checkpoint is already bf16.  Cast only trainable modules so
    # their matmuls match the detached backbone states; keep fixed_mu in fp32.
    device = model.get_input_embeddings().weight.device
    dtype = model.get_input_embeddings().weight.dtype
    for module in (
        wrapper.residual_pool,
        wrapper.residual_norm,
        wrapper.residual_in,
        wrapper.residual_out,
    ):
        module.to(device=device, dtype=dtype)
    wrapper.to(device=device)
    wrapper.set_fixed_mu(train_mean)
    model.attention = wrapper
    model.config.latent_residual_correction = spec
    model.config.architectures = ["LatentModel"]

    # Freeze first, then explicitly reopen only the correction path.
    model.requires_grad_(False)
    for parameter in wrapper.residual_parameters():
        parameter.requires_grad_(True)


def train(
    base_model: str = "output_dir/CDs_and_Vinyl_R3_vae7b_nocurriculum/latent/checkpoint-1152",
    mean_thoughts_path: str = "analysis/evidence/CDs_and_Vinyl/latent_vae7b_nocurriculum_checkpoint_1152/train/thoughts.pt",
    train_file: str = "data/CDs_and_Vinyl/train/CDs_and_Vinyl_5_2015-10-2018-11.csv",
    eval_file: str = "data/CDs_and_Vinyl/valid/CDs_and_Vinyl_5_2015-10-2018-11.csv",
    output_dir: str = "output_dir/CDs_and_Vinyl_R3_attempt9_residual/latent",
    sample: int = -1,
    seed: int = 0,
    category: str = "CDs_and_Vinyl",
    K: int = 0,
    end_k: int = -1,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    bottleneck_size: int = 512,
    residual_scale: float = 1.0,
    orthogonal_to_mu: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
):
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    category_dict = {
        "Office_Products": "office products", "Books": "books", "steam": "games",
        "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games",
        "Video_Games": "video games", "Musical_Instruments": "music instruments",
        "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
        "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant",
    }
    category_prompt = category_dict[category]
    gradient_accumulation_steps = batch_size // micro_batch_size
    if gradient_accumulation_steps < 1:
        raise ValueError("batch_size must be at least micro_batch_size")

    model = LatentModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    thought_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
    if thought_id != len(tokenizer) - 1:
        raise ValueError(
            "base_model must already contain <|Thought|> as its final token; "
            "use a completed R3 checkpoint, not the raw backbone"
        )

    train_mean = _load_train_mean(mean_thoughts_path, model.config.hidden_size)
    _install_residual_wrapper(
        model, train_mean, bottleneck_size, residual_scale, orthogonal_to_mu, end_k
    )
    model.config.loss_type = "ce"
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Attempt 9 residual correction: trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.3f}%), mu_norm={train_mean.norm().item():.4f}, "
        f"bottleneck={bottleneck_size}, scale={residual_scale}, "
        f"orthogonal_to_mu={orthogonal_to_mu}"
    )

    train_data = LatentRDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_prompt, K=K,
    )
    val_data = LatentRDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_prompt, K=K,
    )
    from datasets import Dataset as HFDataset
    hf_train = HFDataset.from_dict(
        {key: [row[key] for row in train_data] for key in train_data[0]}
    )
    hf_val = HFDataset.from_dict(
        {key: [row[key] for row in val_data] for key in val_data[0]}
    )

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
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[ResidualLogCallback()],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
