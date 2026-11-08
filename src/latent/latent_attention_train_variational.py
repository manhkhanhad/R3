"""Training entry point for the variational line (analysis/document/research_notes/
Latent utilization/research_log.md, variational framing 2026-11-08).

  C  history dropout   --history_dropout_p P [--history_dropout_unit item|token]
                        [--history_dropout_eval_p 0]        (history_dropout.py; plain LatentModel)
  A  variational latent --cvae [--cvae_dim 64 --kl_beta 1 --kl_free_nats 0 --kl_warmup 0.3
                        --kl_schedule linear|cyclical --prior_prob 0 --aux_weight 0
                        --aux_prior_weight 0 --item_embeddings_file ...]   (cvae_latent.py)

C and A combine.  Recipe otherwise = plain R3 stage 1 (next-item CE, lr 3e-4, batch 128,
<= 3 epochs, early stopping on the validation loss, which for A is the inference-path CE
with z = mu_p).  latent_attention_train.py is untouched.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
import fire
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback

from experiment_tracking import init_training_run, wandb_report_to
from layers import LatentModel
from reasoning_dataset import LatentRDataset

CATEGORY_TEXT = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics",
                 "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments",
                 "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
                 "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B",
    train_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    sample: int = 1024,
    eval_sample: int = None,
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
    save_only_model: bool = True,
    save_total_limit: int = 1,
    wandb_run_name: str = None,
    # C: stochastic history dropout in the recommendation pass
    history_dropout_p: float = 0.0,
    history_dropout_unit: str = "item",
    history_dropout_eval_p: float = 0.0,
    # A: variational latent
    cvae: bool = False,
    cvae_dim: int = 64,
    posterior_dim: int = 256,
    logvar_init: float = -2.0,
    kl_beta: float = 1.0,
    kl_free_nats: float = 0.0,
    kl_warmup: float = 0.3,  # fraction of training (per cycle for cyclical)
    kl_schedule: str = "linear",
    kl_cycles: int = 4,
    prior_prob: float = 0.0,  # rows decoded from a prior sample instead of a posterior sample
    aux_weight: float = 0.0,  # target-item auxiliary from z_q
    aux_prior_weight: float = 0.0,  # ... and from mu_p
    aux_tau: float = 0.05,
    item_embeddings_file: str = None,  # data/<cat>/info/<cat>_qwen3emb_1536.pt (rows = item ids)
    probe_rows: int = 64,
):
    init_training_run(dataset=category, run_name=wandb_run_name, config={
        "base_model": base_model, "output_dir": output_dir, "dataset": category, "learning_rate": learning_rate,
        "history_dropout_p": history_dropout_p, "history_dropout_unit": history_dropout_unit,
        "history_dropout_eval_p": history_dropout_eval_p, "cvae": cvae, "cvae_dim": cvae_dim, "kl_beta": kl_beta,
        "kl_free_nats": kl_free_nats, "kl_warmup": kl_warmup, "kl_schedule": kl_schedule, "prior_prob": prior_prob,
        "aux_weight": aux_weight, "aux_prior_weight": aux_prior_weight})
    category_text = CATEGORY_TEXT[category]
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model_kwargs = {}
    model_class = LatentModel
    aux_table = None
    if cvae:
        from cvae_latent import CVAELatentModel
        model_class = CVAELatentModel
        config = AutoConfig.from_pretrained(base_model)
        spec = {"dim": int(cvae_dim), "posterior_dim": int(posterior_dim), "logvar_init": float(logvar_init), "aux": None}
        if aux_weight > 0 or aux_prior_weight > 0:
            if not item_embeddings_file:
                raise ValueError("aux weights need item_embeddings_file")
            aux_table = torch.load(item_embeddings_file, map_location="cpu", weights_only=False)["embeddings"].float()
            spec["aux"] = {"file": os.path.abspath(item_embeddings_file), "emb_dim": int(aux_table.shape[1]), "tau": float(aux_tau)}
        config.latent_cvae = spec
        model_kwargs["config"] = config
    model = model_class.from_pretrained(base_model, torch_dtype=torch.bfloat16, **model_kwargs)
    model.attention.end_k = end_k
    model.config.loss_type = "ce"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    additional_special_tokens.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1

    if cvae:
        from cvae_latent import init_cvae
        token_norm = float(model.model.embed_tokens.weight[:-1].float().norm(dim=-1).mean())
        model.config.latent_cvae["token_norm"] = token_norm
        init_cvae(model, model.config.latent_cvae)
        model.kl_beta = 0.0 if kl_warmup > 0 else float(kl_beta)
        model.kl_free_nats = float(kl_free_nats)
        model.prior_prob = float(prior_prob)
        model.aux_weight = float(aux_weight)
        model.aux_prior_weight = float(aux_prior_weight)
        model.grad_accum_steps = gradient_accumulation_steps
        if aux_table is not None:
            model.aux.set_table(aux_table)
        print("variational latent:", model.config.latent_cvae, {"kl_beta": kl_beta, "kl_free_nats": kl_free_nats,
              "kl_warmup": kl_warmup, "kl_schedule": kl_schedule, "prior_prob": prior_prob, "aux_weight": aux_weight,
              "aux_prior_weight": aux_prior_weight})

    dataset_class, extra = LatentRDataset, {}
    if cvae:
        from cvae_latent import TargetRowsDataset
        dataset_class = TargetRowsDataset
    train_data = dataset_class(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                               category=category_text, K=K, **extra)
    val_data = dataset_class(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample if eval_sample is None else eval_sample, category=category_text, K=K, **extra)

    if history_dropout_p > 0 or history_dropout_eval_p > 0:
        from history_dropout import build_spec, title_items
        spec = build_spec(tokenizer, category_text, history_dropout_p, history_dropout_unit, history_dropout_eval_p)
        model.config.latent_bottleneck = {"dropout": spec}
        probe = torch.tensor(train_data[0]["input_ids"])[None]
        thought = (probe[0] == len(tokenizer) - 1).nonzero().flatten()[:1]
        title, item = title_items(probe, torch.ones_like(probe), spec, thought)
        n_items = int(item.max()) + 1
        print(f"history dropout: p={history_dropout_p} unit={history_dropout_unit} eval_p={history_dropout_eval_p}; "
              f"row 0: {n_items} items, titles = {tokenizer.decode(probe[0][title[0]])!r}")
        if n_items < 1:
            raise RuntimeError("history dropout: no history titles found in row 0")
    print("LOAD DATA FINISHED")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    hf_train = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if cvae:
        from cvae_latent import TargetCollator
        collator = TargetCollator(collator)
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size, per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps, warmup_steps=20, num_train_epochs=num_epochs,
            learning_rate=learning_rate, bf16=True, logging_steps=1, optim="adamw_torch", eval_strategy="epoch",
            save_strategy="epoch", output_dir=output_dir, save_total_limit=save_total_limit,
            save_only_model=save_only_model, load_best_model_at_end=True,
            label_names=["labels"] if cvae else None, ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length, report_to=wandb_report_to(), remove_unused_columns=False),
        data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    if cvae:
        from cvae_latent import CVAEDiagnostics, CVAELogCallback, KLSchedule
        trainer.add_callback(KLSchedule(model, kl_beta, kl_warmup, kl_schedule, kl_cycles))
        trainer.add_callback(CVAELogCallback())
        step = max(1, len(val_data) // probe_rows)
        rows = [val_data[i] for i in range(0, len(val_data), step)][:probe_rows]
        trainer.add_callback(CVAEDiagnostics(rows, collator, output_dir))
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
