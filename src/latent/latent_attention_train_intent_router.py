"""Phase 2 of disentangled-interest R3: DivReason-style path aggregation.

Loads a phase-1 intent checkpoint (trained paths), adds a router + redundancy
suppression over the K paths, freezes everything else, and trains the router through
the decoder's recommendation loss.  ``fit_file`` defaults to the validation split:
the phase-1 model has memorised its training rows, so a selector fit on them gets
little signal (cf. the outcome-memory gate that learned to close).
"""

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
from latent_attention_train_intent import CATEGORY_TEXT
from layers import LatentModel
from reasoning_dataset import LatentRDataset


class RouterLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None:
            return
        layer = (model.module if hasattr(model, "module") else model).attention
        for key, value in (layer.last_aggregation or {}).items():
            logs[key] = value.item() if torch.is_tensor(value) else value


def train(
    base_model: str = "",
    fit_file: str = "",
    output_dir: str = "./output_test/",
    category: str = "Musical_Instruments",
    fit_sample: int = -1,
    holdout_fraction: float = 0.1,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    cutoff_len: int = 512,
    tau: float = 0.5,
    beta: float = 0.5,
    prune: float = 0.05,
    wandb_run_name: str = None,
):
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir,
                                                "dataset": category, "phase": "router"})
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data = LatentRDataset(train_file=fit_file, tokenizer=tokenizer, max_len=cutoff_len,
                          sample=fit_sample, seed=seed, category=CATEGORY_TEXT[category], K=0)
    rows = [data[i] for i in range(len(data))]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    n_hold = max(1, int(len(rows) * holdout_fraction))
    to_hf = lambda items: HFDataset.from_dict({k: [row[k] for row in items] for k in items[0]})
    hf_fit = to_hf([rows[i] for i in order[n_hold:]])
    hf_hold = to_hf([rows[i] for i in order[:n_hold]])
    print(f"router fit rows {len(hf_fit)}, held-out rows {len(hf_hold)} from {fit_file}", flush=True)

    config = AutoConfig.from_pretrained(base_model)
    spec = dict(config.latent_intent_reasoning)
    spec["aggregator"] = {"tau": tau, "prune": prune, "beta": beta}
    config.latent_intent_reasoning = spec
    model = LatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    layer = model.attention
    layer.prepare_aggregator(beta)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(layer.router.parameters()) + [layer.agg_log_beta]
    for parameter in trainable:
        parameter.requires_grad_(True)
    print(f"trainable router parameters: {sum(p.numel() for p in trainable)}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_fit, eval_dataset=hf_hold, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=5, num_train_epochs=num_epochs, learning_rate=learning_rate,
            bf16=True, logging_steps=1, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="no", output_dir=output_dir,
            load_best_model_at_end=False, report_to=wandb_report_to(), remove_unused_columns=False,
        ),
        data_collator=collator,
        callbacks=[RouterLogCallback()],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
