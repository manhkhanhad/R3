"""Training entry point for Attempt 6: explicit r = mu + z decomposition with
alpha-capped attention schedule + contrastive/geometry pressure on z.

Copy of latent_attention_train_scheduled.py that swaps in
DecomposedContrastiveLatentModel. Delete this file plus
decomposed_attention.py and decomposed_contrastive_attention.py (and the
`latent_decomposition` config branch in layers.py / grpo model.py) to remove
the approach entirely.
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
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from decomposed_contrastive_attention import DecomposedContrastiveLatentModel
from latent_attention_train_scheduled import AttentionScheduleCallback
from reasoning_dataset import LatentRDataset
from experiment_tracking import init_training_run, wandb_report_to


class DecompositionLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attr, key in (
            ("last_ce_loss", "ce_loss"),
            ("last_contrastive_loss", "contrastive_loss"),
            ("last_var_loss", "var_loss"),
            ("last_cov_loss", "cov_loss"),
        ):
            value = getattr(target, attr, None)
            logs[key] = value.item() if value is not None else None
        logs["mu_norm"] = target.attention.mu.detach().float().norm().item()
        logs["z_mean_norm"] = target.attention.z_mean.norm().item()


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

    # attention-schedule hyperparameters
    ramp_epochs: float = -1.0,
    alpha_max: float = 0.75,

    # decomposition hyperparameters
    rho: float = 1.0,  # ||z|| = rho * ||mu||
    ema_momentum: float = 0.99,
    mu_init_norm: float = 1.0,

    # contrastive/geometry hyperparameters (applied to z only)
    temperature: float = 0.1,
    contrastive_weight: float = 1.0,
    geom_var_weight: float = 1.0,
    geom_cov_weight: float = 0.04,
    geom_var_target: float = 1.0,
    wandb_run_name: str = None,
):

    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    )
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        raise RuntimeError("Attempt 6 is single-GPU only: per-batch schedule draws and the EMA centering buffer are not DDP-synchronized")

    if ramp_epochs < 0:
        ramp_epochs = max(num_epochs - 1, 1e-8)

    # The decomposition flag must be on the config before from_pretrained so
    # __init__ builds DecomposedSelfAttentionLayer; it is then saved into the
    # checkpoint's config.json so every downstream loader rebuilds it.
    config = AutoConfig.from_pretrained(base_model)
    config.latent_decomposition = {"rho": rho, "ema_momentum": ema_momentum}
    model = DecomposedContrastiveLatentModel.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.temperature = temperature
    model.contrastive_weight = contrastive_weight
    model.geom_var_weight = geom_var_weight
    model.geom_cov_weight = geom_cov_weight
    model.geom_var_target = geom_var_target
    model.grad_accum_steps = gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    print("We have added", num_added_toks, "tokens")
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    # mu starts as the <|Thought|> embedding direction at a controlled norm: a
    # zero mu would zero z's scale (||z|| = rho * ||mu||) and freeze the residual.
    with torch.no_grad():
        thought_embed = model.model.embed_tokens.weight[len(tokenizer) - 1].float()
        model.attention.mu.copy_((thought_embed / thought_embed.norm().clamp(min=1e-6) * mu_init_norm).to(model.attention.mu.dtype))
    print(f"mu initialized with norm {model.attention.mu.float().norm().item():.4f} (rho={rho})")

    train_data = LatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K)
    val_data = LatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category, K=K)

    print("LOAD DATA FINISHED")

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
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            # No early stopping: eval loss under an alpha ramp is expected to be
            # noisy (see latent_attention_train_scheduled.py).
            AttentionScheduleCallback(ramp_epochs=ramp_epochs, alpha_max=alpha_max),
            DecompositionLogCallback(),
        ],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
