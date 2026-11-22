"""Train MWC-M: margin + WIDER warm-up, then same-motive InfoNCE (motive_contrastive.py)."""

import glob
import os
import shutil
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
from latent_attention_train_grounded_margin import CATEGORY_TEXT, GroundedLogCallback
from motive_contrastive import (HingeLabelCollator, MotiveDataset, MotiveMarginLatentModel, build_hard_negative_index,
                                build_positive_index)


class SingleCheckpointCallback(transformers.TrainerCallback):
    """Storage: delete the previous checkpoint *before* the next epoch save, so a run never holds two
    (HF writes the new one first and rotates after; ~9 GB each with optimizer state).  A preemption during
    the save then restarts from scratch instead of resuming; it never leaves two full checkpoints."""

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_save:
            remove_checkpoints(args.output_dir, "before epoch save")


def remove_checkpoints(output_dir, why):
    for path in sorted(glob.glob(os.path.join(output_dir, "checkpoint-*"))):
        shutil.rmtree(path, ignore_errors=True)
        print(f"[storage] removed {path} ({why})", flush=True)


class MotivePhaseCallback(transformers.TrainerCallback):
    """Contrastive term on once ``warmup_epochs`` epochs have elapsed (from state.epoch, so resumes land right)."""

    def __init__(self, warmup_epochs, margin_epochs=-1, early_margin_mode="fixed", late_margin_mode="",
                 early_margin_weight=1.0, late_margin_weight=-1.0):
        self.warmup_epochs = warmup_epochs
        self.margin_epochs = margin_epochs   # > 0: history-swap hinge only for the first margin_epochs (MWC-M3b)
        # MW-3b-L: with late_margin_mode set, the hinge is not switched off after margin_epochs but switched
        # to late_margin_mode (e.g. novel / adaptive) for the remaining epochs.
        # MW-3b-lambda: with late_margin_weight >= 0, the hinge stays on after margin_epochs at that weight.
        self.early_margin_mode = early_margin_mode
        self.late_margin_mode = late_margin_mode
        self.early_margin_weight = early_margin_weight
        self.late_margin_weight = late_margin_weight
        self.announced = None
        self.margin_announced = None

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        target = model.module if hasattr(model, "module") else model
        enabled = (state.epoch or 0.0) >= self.warmup_epochs - 1e-6
        target.contrastive_enabled = enabled
        if enabled != self.announced:
            print(f"[phase] epoch {state.epoch or 0.0:.3f}: motive InfoNCE {'ON' if enabled else 'off'}; "
                  f"bank filled {float(target.bank_filled.float().mean()):.3f}", flush=True)
            self.announced = enabled
        if self.margin_epochs > 0:
            early = (state.epoch or 0.0) < self.margin_epochs - 1e-6
            if self.late_margin_mode or self.late_margin_weight >= 0:
                target.margin_active = True
                if self.late_margin_mode:
                    target.margin_mode = self.early_margin_mode if early else self.late_margin_mode
                if self.late_margin_weight >= 0:
                    target.margin_weight = self.early_margin_weight if early else self.late_margin_weight
                phase = f"{target.margin_mode} x{target.margin_weight}"
            else:
                target.margin_active = early
                phase = "ON" if early else "off"
            if phase != self.margin_announced:
                print(f"[phase] epoch {state.epoch or 0.0:.3f}: history-swap hinge {phase}", flush=True)
                self.margin_announced = phase


def reference_weights(path, fn, tau, n_rows, power=1.0, cut=0.25):
    """MW-R: per-train-row hinge weight from the frozen reference LM's NLL of the target (reference_score.py).
    rank: percentile of NLL_ref in [0, 1]; sigmoid: sigmoid((NLL_ref - median) / tau).  Low NLL_ref (the history
    already explains the target) -> small weight.  cut (MW-nc with copy_row = reference-NLL quantile < cut, e.g. Q1):
    weight 0 on copy rows, 1 elsewhere, so the refweight loss sum(w*hinge)/sum(w) equals skipcopy's.
    Indexed by row_id (CSV row position)."""
    ref = torch.load(path, map_location="cpu", weights_only=False)
    nll, rows = ref["nll_mean"].double(), ref["row_id"].long()
    if fn == "rank":
        w = nll.argsort().argsort().double() / max(1, len(nll) - 1)
    elif fn == "sigmoid":
        w = torch.sigmoid((nll - nll.median()) / tau)
    elif fn == "cut":
        if not 0.0 < cut < 1.0:
            raise ValueError(f"ref_cut must be in (0, 1), got {cut}")
        pct = nll.argsort().argsort().double() / max(1, len(nll) - 1)
        w = (pct >= cut).double()
        print(f"MW-nc (copy_row = reference-NLL percentile < {cut}): {int((w == 0).sum())} of {len(w)} train rows "
              f"are copy rows (no hinge)", flush=True)
    else:
        raise ValueError(f"unknown ref_weight_fn {fn}")
    w = w.clamp(min=0.0) ** power   # MW-R-H2: sharpen (power > 1) toward the zero-weight-on-copy-like-rows limit
    out = torch.full((n_rows,), float("nan"), dtype=torch.float64)
    out[rows] = w
    if torch.isnan(out).any():
        raise ValueError(f"reference file {path} misses {int(torch.isnan(out).sum())} of {n_rows} train rows")
    c = ref["is_copy"].bool()
    print(f"MW-R weights ({fn}, tau {tau}, power {power}): mean {w.mean():.3f}; copy {w[c].mean():.3f} vs non-copy {w[~c].mean():.3f}",
          flush=True)
    return out.float()


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "",
    eval_file: str = "",
    item_emb_file: str = "",
    output_dir: str = "./output_test/",
    category: str = "Toys_and_Games",
    sample: int = -1,
    eval_sample: int = None,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    K: int = 0,
    margin: float = 0.2,
    margin_weight: float = 1.0,
    wider_weight: float = 0.1,
    wider_momentum: float = 0.99,
    warmup_epochs: int = 1,
    motive_weight: float = 0.1,
    temperature: float = 0.1,
    within_threshold: float = 0.8,
    max_positives: int = 16,
    num_negatives: int = 256,
    projection_dim: int = 256,
    within_mode: str = "user",
    margin_mode: str = "fixed",
    margin_epochs: int = -1,
    late_margin_mode: str = "",
    late_margin_weight: float = -1.0,
    adaptive_alpha: float = 0.5,
    selfnorm_cap: float = 2.0,
    lagrange_delta: float = 0.1,
    lagrange_lr: float = 0.01,
    lagrange_momentum: float = 0.99,
    lagrange_mu_max: float = 5.0,
    lagrange_mu_init: float = 1.0,
    ref_nll_file: str = "",
    ref_weight_fn: str = "rank",
    ref_tau: float = 0.5,
    ref_power: float = 1.0,
    ref_cut: float = 0.25,
    band_upper: float = -1.0,
    band_beta: float = 1.0,
    hinge_negative: str = "batch",
    copy_swap_weight: float = 0.0,
    distill_weight: float = 0.0,
    distill_model: str = "",
    copy_power: float = 1.0,
    hard_neg_cap: int = 32,
    sasrec_file: str = "",
    sasrec_k: int = 20,
    resume_from_checkpoint: str = None,
    wandb_run_name: str = None,
    train_seed: int = None,
):
    # train_seed: None keeps the old behaviour (no global seeding before model init; HF Trainer default seed 42).
    # An int seeds module init, data order and dropout (transformers.set_seed + TrainingArguments seed/data_seed).
    if train_seed is not None:
        transformers.set_seed(train_seed)
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    init_training_run(dataset=category, config={"base_model": base_model, "output_dir": output_dir, "dataset": category})
    grad_accum = batch_size // micro_batch_size
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in special:
        special.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    text = CATEGORY_TEXT[category]

    item_emb = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"].float()
    index, items, users, stats = build_positive_index(train_file, item_emb, within_threshold, within_mode=within_mode,
                                                      sasrec_file=sasrec_file or None, sasrec_k=sasrec_k)
    print(f"motive positives (threshold {within_threshold}, within={within_mode}, k={sasrec_k}): {stats}", flush=True)

    modes = ("fixed", "self", "selfnorm", "novel", "adaptive", "skipcopy", "lagrange", "refweight")
    if margin_mode not in modes or (late_margin_mode and late_margin_mode not in modes):
        raise ValueError(f"unknown margin_mode {margin_mode} / late_margin_mode {late_margin_mode}")
    if (late_margin_mode or late_margin_weight >= 0) and margin_epochs <= 0:
        raise ValueError("late_margin_mode / late_margin_weight need margin_epochs > 0 (the epoch at which they switch)")
    MotiveDataset.novel_hinge = "novel" in (margin_mode, late_margin_mode)
    train_data = MotiveDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                               sample=sample, seed=seed, category=text, K=K)
    val_data = MotiveDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample if eval_sample is None else eval_sample, category=text, K=K)
    to_hf = lambda data: HFDataset.from_dict({k: [row[k] for row in data] for k in data[0]})
    hf_train = to_hf(train_data)
    # Validation rows are not train rows: no bank id (contrastive term is train-only anyway).
    hf_val = to_hf(val_data).remove_columns(["row_id"])

    config = AutoConfig.from_pretrained(base_model)
    config.latent_motive = {"bank_size": int(stats["rows"]), "projection_dim": projection_dim,
                            "temperature": temperature, "weight": motive_weight,
                            "max_positives": max_positives, "num_negatives": num_negatives,
                            "within_threshold": within_threshold, "warmup_epochs": warmup_epochs,
                            "within_mode": within_mode, "sasrec_k": sasrec_k,
                            "margin_mode": margin_mode, "margin_epochs": margin_epochs, "late_margin_mode": late_margin_mode, "late_margin_weight": late_margin_weight, "adaptive_alpha": adaptive_alpha, "selfnorm_cap": selfnorm_cap,
                            "lagrange": {"delta": lagrange_delta, "lr": lagrange_lr, "momentum": lagrange_momentum, "mu_max": lagrange_mu_max, "mu_init": lagrange_mu_init},
                            "reference": {"file": ref_nll_file, "fn": ref_weight_fn, "tau": ref_tau, "power": ref_power, "cut": ref_cut},
                            "band": {"upper": band_upper, "beta": band_beta}, "train_seed": train_seed,
                            "hinge_negative": hinge_negative, "hard_neg_cap": hard_neg_cap,
                            "copy_swap_weight": copy_swap_weight, "distill_weight": distill_weight, "distill_model": distill_model, "copy_power": copy_power}
    model = MotiveMarginLatentModel.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)
    model.projection.float()
    model.set_positive_index(index, items, users)
    model.grad_accum_steps = grad_accum
    model.config.loss_type = "ce"
    model.margin, model.margin_weight = margin, margin_weight
    model.margin_mode = margin_mode
    model.adaptive_alpha = adaptive_alpha
    model.selfnorm_cap = selfnorm_cap
    model.lagrange_delta, model.lagrange_lr = lagrange_delta, lagrange_lr
    model.lagrange_momentum, model.lagrange_mu_max = lagrange_momentum, lagrange_mu_max
    model.band_upper, model.band_beta = band_upper, band_beta
    # fp32 (from_pretrained casts buffers to bf16, where mu steps of lr would round away); resume reloads it.
    model.lagrange_state = torch.tensor([lagrange_mu_init, 0.0, 0.0], dtype=torch.float32)
    if margin_mode == "refweight":
        model.set_ref_weights(reference_weights(ref_nll_file, ref_weight_fn, ref_tau, int(stats["rows"]), ref_power, ref_cut))
    if copy_swap_weight > 0 or distill_weight > 0:
        if margin_mode != "refweight":
            raise ValueError("copy_swap_weight / distill_weight need margin_mode=refweight (they use 1 - w_i)")
        model.copy_swap_weight, model.distill_weight = float(copy_swap_weight), float(distill_weight)
        model.copy_power = float(copy_power)
        if distill_weight > 0:
            from transformers import AutoModelForCausalLM
            model.set_teacher(AutoModelForCausalLM.from_pretrained(distill_model, torch_dtype=torch.bfloat16))
        print(f"MW-R-H-S/D: swapped-pass CE weight {copy_swap_weight}, distillation weight {distill_weight} from '{distill_model}' (row weight (1 - w_i)^{copy_power})", flush=True)
    if hinge_negative == "history":
        hard_ptr, hard_idx, hstats = build_hard_negative_index(train_file, cap=hard_neg_cap)
        model.set_hard_negative_index(hard_ptr, hard_idx)
        print(f"MW-HN history-matched hard negatives (same last item, other user, different next item; cap {hard_neg_cap}): {hstats}", flush=True)
    elif hinge_negative != "batch":
        raise ValueError(f"unknown hinge_negative {hinge_negative}")
    if wider_weight > 0:
        model.config.latent_wider = {"weight": float(wider_weight), "momentum": float(wider_momentum)}
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")["input_ids"][0] == len(tokenizer) - 1
    print(f"MWC-M: margin={margin} x{margin_weight}; WIDER={getattr(model.config, 'latent_wider', None)}; "
          f"margin_mode={margin_mode}, margin_epochs={margin_epochs}, adaptive_alpha={adaptive_alpha}; motive={config.latent_motive}", flush=True)

    collator = HingeLabelCollator(DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
    trainer = transformers.Trainer(
        model=model, train_dataset=hf_train, eval_dataset=hf_val, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size, per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum, warmup_steps=20, num_train_epochs=num_epochs,
            learning_rate=learning_rate, bf16=True, logging_steps=1, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, output_dir=output_dir,
            load_best_model_at_end=False, report_to=wandb_report_to(), remove_unused_columns=False,
            **({"seed": train_seed, "data_seed": train_seed} if train_seed is not None else {}),
        ),
        data_collator=collator,
        callbacks=[GroundedLogCallback(), MotivePhaseCallback(warmup_epochs, margin_epochs, margin_mode, late_margin_mode,
                                                                margin_weight, late_margin_weight), SingleCheckpointCallback()],
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Keep only the final model: the training log goes next to it, then the resume checkpoint is dropped.
    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
    remove_checkpoints(output_dir, "training finished, final model saved")


if __name__ == "__main__":
    fire.Fire(train)
