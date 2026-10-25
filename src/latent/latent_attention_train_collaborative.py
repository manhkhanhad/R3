"""Train plain R3 with user/item supervised multi-positive InfoNCE."""

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

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel
from collaborative_contrastive_dataset import CollaborativeContrastiveDataset
from collaborative_pair_features import build_pair_features
from experiment_tracking import init_training_run, move_callback_to_end, wandb_report_to


class CollaborativeLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for attr, key in (
            ("last_ce_loss", "ce_loss"),
            ("last_user_contrastive_loss", "user_infonce_loss"),
            ("last_target_contrastive_loss", "target_infonce_loss"),
            ("last_cf_contrastive_loss", "cf_infonce_loss"),
            ("last_user_positive_rate", "user_positive_rate"),
            ("last_target_positive_rate", "target_positive_rate"),
            ("last_cf_positive_rate", "cf_positive_rate"),
            ("last_queue_accept_rate", "queue_accept_rate"),
            ("last_queue_count", "queue_count"),
            ("last_nll_gate_threshold", "nll_gate_threshold"),
            ("last_queue_user_diversity", "queue_user_diversity"),
            ("last_queue_replaced", "queue_replaced"),
        ):
            value = getattr(target, attr, None)
            logs[key] = value.item() if value is not None else None


@torch.no_grad()
def prefill_candidate_bank(model, dataset, collator, batch_size, seed, min_token_accuracy):
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collator,
    )
    model.eval()
    seen = 0
    accepted = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        embeds = model.generate_embs(batch["input_ids"], batch["attention_mask"])
        positions = model._thought_positions(batch["input_ids"])
        rows = torch.arange(len(positions), device=device)
        latent = embeds[rows, positions].float()
        output = model._run_second_pass(embeds, batch["attention_mask"], None, None, 0)
        _, example_nll, accuracy = model._loss_and_quality(output.logits, batch["labels"])
        quality = (accuracy >= min_token_accuracy) & (example_nll <= model.positive_max_nll)
        model._push_bank(
            latent, batch["user_hash"], batch["target_hash"], batch["sample_hash"],
            example_nll, quality, batch.get("target_title_features"),
            batch.get("target_item_index"),
        )
        seen += len(positions)
        accepted += int(quality.sum().item())
        if seen % 1024 < batch_size or int(model.bank_count.item()) == model.bank_size:
            print(f"queue prefill: seen={seen}, accepted={accepted}, bank={int(model.bank_count.item())}", flush=True)
        if int(model.bank_count.item()) == model.bank_size:
            break
    model.train()
    print(
        f"queue prefill complete: seen={seen}, accepted={accepted}, "
        f"bank={int(model.bank_count.item())}/{model.bank_size}", flush=True,
    )
    if int(model.bank_count.item()) == 0:
        raise RuntimeError("No examples met the queue prefill accuracy threshold")
    return seen, accepted


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
    nll_gate_quantile: float = 0.0,
    nll_gate_window: int = 1024,
    bank_max_per_user: int = 0,
    bank_max_per_target: int = 0,
    bank_dedup_cos: float = 0.0,
    save_steps: int = 0,             # 0 = save once per epoch (default behaviour)
    save_total_limit: int = 1,       # 0 = keep every checkpoint
    save_only_model: bool = False,
    title_cf_positives: bool = False,
    title_positive_floor: float = 0.25,
    cf_contrastive_weight: float = 0.1,
    cf_topk: int = 32,
    cf_min_common: int = 2,
    cf_min_cosine: float = 0.1,
    prefill_queue: bool = False,
    prefill_batch_size: int = 8,
    wandb_run_name: str = None,
):
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
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        raise RuntimeError(
            "Collaborative contrastive R3 is single-GPU only because its FIFO corpus "
            "is intentionally local and unsynchronized."
        )

    if title_cf_positives:
        item_to_index, title_vectors, cf_neighbors, feature_report = build_pair_features(
            train_file, cf_topk=cf_topk, cf_min_common=cf_min_common,
            cf_min_cosine=cf_min_cosine, seed=seed,
        )
        print(f"pair features: {feature_report}", flush=True)
    else:
        item_to_index = title_vectors = cf_neighbors = None

    config = AutoConfig.from_pretrained(base_model)
    config.collaborative_contrastive = {
        "projection_dim": projection_dim,
        "bank_size": bank_size,
        "title_feature_dim": 0 if title_vectors is None else title_vectors.shape[1],
        "cf_topk": 0 if cf_neighbors is None else cf_neighbors.shape[1],
        "nll_gate_window": nll_gate_window,
    }
    model = CollaborativeContrastiveLatentModel.from_pretrained(
        base_model, config=config, torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    model.temperature = temperature
    model.user_contrastive_weight = user_contrastive_weight
    model.target_contrastive_weight = target_contrastive_weight
    model.cf_contrastive_weight = cf_contrastive_weight if title_cf_positives else 0.0
    model.title_positive_floor = title_positive_floor
    if cf_neighbors is not None:
        model.cf_neighbor_ids = torch.as_tensor(cf_neighbors, dtype=torch.int64)
    model.max_user_positives = max_user_positives
    model.max_target_positives = max_target_positives
    model.num_negatives = num_negatives
    model.positive_min_token_accuracy = positive_min_token_accuracy
    model.positive_max_nll = positive_max_nll
    model.nll_gate_quantile = nll_gate_quantile
    model.bank_max_per_user = bank_max_per_user
    model.bank_max_per_target = bank_max_per_target
    model.bank_dedup_cos = bank_dedup_cos
    model.grad_accum_steps = gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in additional_special_tokens:
        additional_special_tokens.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1

    print(
        "Plain-R3 collaborative InfoNCE: "
        f"epochs={num_epochs}, queue={bank_size}, projection={projection_dim}, "
        f"temperature={temperature}, user_weight={user_contrastive_weight}, "
        f"target_weight={target_contrastive_weight}, "
        f"min_token_accuracy={positive_min_token_accuracy}, max_nll={positive_max_nll}, "
        f"nll_gate_quantile={nll_gate_quantile} (window {nll_gate_window}), "
        f"per_user_cap={bank_max_per_user}, per_target_cap={bank_max_per_target}, "
        f"dedup_cos={bank_dedup_cos}"
    )
    train_data = CollaborativeContrastiveDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_text, K=K,
        item_to_index=item_to_index, title_vectors=title_vectors,
    )
    val_data = CollaborativeContrastiveDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, category=category_text, K=K,
        item_to_index=item_to_index, title_vectors=title_vectors,
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
    if prefill_queue:
        model.to("cuda")
        prefill_candidate_bank(
            model, hf_train_dataset, collator, prefill_batch_size, seed,
            positive_min_token_accuracy,
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
            save_strategy="steps" if save_steps > 0 else "epoch",
            save_steps=save_steps if save_steps > 0 else 500,
            output_dir=output_dir,
            save_total_limit=save_total_limit or None,
            save_only_model=save_only_model,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[CollaborativeLogCallback()],
    )
    # Trainer registers reporting integrations before user callbacks. Move W&B
    # behind CollaborativeLogCallback so it sees the enriched log dictionary.
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
