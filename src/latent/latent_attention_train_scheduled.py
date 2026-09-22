"""Training entry point for the attention-pattern-schedule ablation.

Copy of latent_attention_train.py that swaps in ScheduledLatentModel and adds
an epoch-driven alpha schedule via a TrainerCallback. latent_attention_train.py
itself is untouched; delete this file plus scheduled_attention.py to remove the
approach entirely.

Schedule: alpha ramps linearly 0 -> 1 over the first (num_epochs - 1) epochs,
then holds at >= 1 (vanilla R3, unrestricted attention) for the remaining
epoch(s). ramp_epochs defaults to num_epochs - 1 but is exposed as its own CLI
flag so the length of the final "both" phase can be swept independently of
num_epochs.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fire
import torch

import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from scheduled_attention import ScheduledLatentModel
from reasoning_dataset import LatentRDataset


class AttentionScheduleCallback(transformers.TrainerCallback):
    def __init__(
        self,
        ramp_epochs: float,
        alpha_max: float = 1.0,
        three_way_start_epoch: float = -1.0,
    ):
        self.ramp_epochs = max(ramp_epochs, 1e-8)
        # alpha_max >= 1.0 preserves the original behavior: ramp 0->1, then
        # keep growing past 1.0 (forcing deterministic "both" for the rest of
        # training). alpha_max < 1.0 instead ramps 0->alpha_max and holds
        # flat there -- alpha never reaches 1.0, so "both" is never triggered
        # and the model trains exclusively on the stochastic
        # reasoning_only/history_only mix for the entire run.
        self.alpha_max = alpha_max
        # Negative preserves every existing schedule. At or after a
        # non-negative threshold, sample history_only, reasoning_only, and
        # both uniformly.
        self.three_way_start_epoch = three_way_start_epoch

    def _set_schedule(self, model, alpha: float, progress: float):
        target = model.module if hasattr(model, "module") else model
        target.alpha = alpha
        target.equal_three_way_mix = (
            self.three_way_start_epoch >= 0
            and progress >= self.three_way_start_epoch
        )

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        progress = state.epoch or 0.0
        if progress >= self.ramp_epochs:
            if self.alpha_max >= 1.0:
                alpha = 1.0 + (progress - self.ramp_epochs)
            else:
                alpha = self.alpha_max
        else:
            alpha = self.alpha_max * (progress / self.ramp_epochs)
        self._set_schedule(model, alpha, progress)

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        logs["schedule_alpha"] = getattr(target, "alpha", None)
        logs["schedule_equal_three_way"] = getattr(target, "equal_three_way_mix", False)
        logs["schedule_mode"] = getattr(target, "last_schedule_mode", None)


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
    ramp_epochs: float = -1.0,  # -1 means num_epochs - 1 (default: linear ramp over all but the last epoch)
    alpha_max: float = 1.0,  # <1.0: ramp 0->alpha_max and hold there, never triggering deterministic "both"
    three_way_start_epoch: float = -1.0,
):

    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    )
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if ramp_epochs < 0:
        ramp_epochs = max(num_epochs - 1, 1e-8)

    model = ScheduledLatentModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # custom 4D attention bias is only exercised against the eager path
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"

    ##############################################################################
    ##################
    ##################   add a special token here to represent latent thought
    ##################

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    print("We have added", num_added_toks, "tokens")
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    ##################
    ##############################################################################

    train_data = LatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K)
    val_data = LatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category, K=K)

    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
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
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none",
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            # No EarlyStoppingCallback here (unlike latent_attention_train.py): eval loss is
            # expected to be noisy/non-monotonic during the alpha ramp (epochs 1..num_epochs-1),
            # since most of that eval history reflects a model still adapting to alternating
            # history_only/reasoning_only training batches. Stopping early on that noise would
            # prevent ever reaching the final alpha>=1 epoch, which is the one that actually
            # matters (a full epoch of unrestricted "both" training/eval, matching deployment).
            AttentionScheduleCallback(
                ramp_epochs=ramp_epochs,
                alpha_max=alpha_max,
                three_way_start_epoch=three_way_start_epoch,
            ),
        ],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
