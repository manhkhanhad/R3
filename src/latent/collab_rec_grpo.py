"""R3 stage 2 (stock noise-GRPO) on a CollabRec stage-1 checkpoint.

Same algorithm and settings as src/grpo_attention_tuning/train_noise_grpo.py (NoiseGRPORecTrainer:
Gaussian noise (sigma 1.5) on the thought, reward = -PPL of the gold target, LLM frozen). Differences,
all needed to run it on a CollabLatentModel:
  * the policy is CollabLatentModel; only the module producing r (InjectNet in text mode) is trained,
    as R3 trains only its attention layer;
  * each row's support ids travel with the batch and are set on the model before every generate_embs;
  * prompts/targets come from CollabLatentRDataset with the checkpoint's target_format (end_predict for C4).
Kept as in stock R3: the reference log-probs use the policy's r, so the KL term is ~0.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "grpo_attention_tuning"))

import fire
import pandas as pd
import torch
import trl
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from collab_rec_dataset import CollabLatentRDataset, build_support_table, load_support_rows
from collab_rec_latent import CollabLatentModel
from collab_rec_train import CATEGORY
from experiment_tracking import init_training_run, wandb_report_to
from grpo_trainer import NoiseGRPORecTrainer


def r_module_names(model):
    return ["injectnet"] if model.support_mode == "text" else ["support_proj", "support_attn"]


class CollabNoiseGRPOTrainer(NoiseGRPORecTrainer):
    """NoiseGRPORecTrainer with the batch's support ids set on the policy around every generate_embs."""

    def _with_supports(self, support_ids, fn):
        self.model._support_ids = support_ids
        try:
            return fn()
        finally:
            self.model._support_ids = None

    def _generate_and_score_completions(self, inputs):
        support_ids = inputs.get("support_ids")
        out = self._with_supports(support_ids, lambda: super(CollabNoiseGRPOTrainer, self)
                                  ._generate_and_score_completions(inputs))
        out["support_ids"] = support_ids
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self._with_supports(inputs.get("support_ids"), lambda: super(CollabNoiseGRPOTrainer, self)
                                   .compute_loss(model, inputs, return_outputs, num_items_in_batch))


def train(
    base_model: str,  # CollabRec stage-1 checkpoint (config.json carries collab_rec)
    train_file: str = "data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    category: str = "Toys_and_Games",
    lr: float = 1e-5,
    num_epochs: int = 1,
    batch_size: int = 256,
    micro_batch_size: int = 8,
    sample: int = -1,
    seed: int = 42,
    cutoff_len: int = 512,
    num_generations: int = 8,
    beta: float = 0.01,
    num_iterations: int = 1,
    epsilon: float = 0.2,
    epsilon_high: float = 0.28,
    max_completion_length: int = 256,
    save_steps: int = 200,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
):
    model = CollabLatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    spec = model.config.collab_rec
    target_format = spec.get("target_format", "r3")
    init_training_run(dataset=category, run_name=wandb_run_name,
                      config={"base_model": base_model, "train_file": train_file, "output_dir": output_dir,
                              "dataset": category, "collab_rec": spec, "stage": "noise_grpo"})
    print("collab_rec:", spec)

    table, mask = build_support_table(spec, tokenizer)
    if table is not None:
        model.set_support_table(table, mask)
        print(f"support table {tuple(table.shape)}")

    for p in model.parameters():
        p.requires_grad = False
    names = r_module_names(model)
    for name in names:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable modules {names}: {n_train} parameters")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    rows = load_support_rows(spec, "train")
    if rows is not None:
        n = len(pd.read_csv(train_file, usecols=[0]))
        assert len(rows) == n, f"train: {len(rows)} support rows for {n} CSV rows"
    train_data = CollabLatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
                                      seed=seed, category=CATEGORY[category], support_rows=rows,
                                      target_format=target_format)
    ex = train_data[0]
    print("train example keys:", list(ex.keys()), "support_ids:", ex.get("support_ids"))
    print("train example target tokens:",
          [tokenizer.decode([t]) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    hf_train = HFDataset.from_dict({k: [v[k] for v in train_data] for k in ex.keys()})

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size // world_size
    trainer = CollabNoiseGRPOTrainer(
        prefix_allowed_tokens_fn=None,  # unused by the noise trainer (no generation; completions are gold targets)
        model=model,
        reward_funcs=[lambda completions, **kwargs: [0.0] * len(completions)],  # unused: reward = -PPL
        args=trl.GRPOConfig(
            warmup_steps=200,
            num_generations=num_generations,
            max_prompt_length=512,
            temperature=1.0,
            max_completion_length=max_completion_length,
            learning_rate=lr,
            beta=beta,
            num_iterations=num_iterations,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type="cosine",
            eval_strategy="no",
            save_steps=save_steps,
            save_total_limit=1,
            load_best_model_at_end=False,
            report_to=wandb_report_to(),
            bf16=True,
            logging_steps=1,
            remove_unused_columns=False,
        ),
        train_dataset=hf_train,
        processing_class=tokenizer,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
