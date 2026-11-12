"""Continue Attempt 10 with a projected-neighbour correction and paired controls.

The fixed bank and projection heads come from the same Attempt-10 checkpoint.
Training alternates mean and earlier-train-latent controls while holding the
history fixed. Controls are detached references, so the paired loss cannot be
satisfied by directly degrading a control branch in the current update.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import fire
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset as HFDataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq
from layers import LatentModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel
from experiment_tracking import init_training_run, move_callback_to_end, wandb_report_to
from projected_correction_training_utils import CATEGORY_TEXT, NeighborLogCallback, TimedDataset, TwoRateTrainer, build_bank
from projected_neighbor_refinement import ProjectedNeighborRefinementLayer
from outcome_memory import OutcomeMemoryDataset, OutcomeMemoryLayer


class PairedCorrectionModel(CollaborativeContrastiveLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.pair_weight = 0.3
        self.pair_margin = 0.01
        self.pair_step = 0
        self.last_pair_loss = None
        self.last_control_nll = None
        self.last_own_nll = None

    @staticmethod
    def _example_nll(logits, labels):
        shifted = logits[:, :-1].float().contiguous()
        target = labels[:, 1:].contiguous()
        mask = target != -100
        nll = F.cross_entropy(
            shifted.reshape(-1, shifted.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view_as(target)
        return (nll * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    @torch.no_grad()
    def _control(self, raw, times):
        layer = self.attention
        if self.pair_step % 2 == 0:
            return layer.bank_mean.to(raw).expand_as(raw)
        eligible = (layer.eligible(len(raw)) if isinstance(layer, OutcomeMemoryLayer) else
                    layer.bank_time[None, :] < times.to(layer.bank_time.device)[:, None])
        random_score = torch.rand(eligible.shape, device=eligible.device).masked_fill(~eligible, -1)
        idx = random_score.argmax(dim=1)
        sampled = layer.bank_raw[idx].to(raw)
        return torch.where(eligible.any(dim=1, keepdim=True), sampled,
                           layer.bank_mean.to(raw).expand_as(raw))

    def forward(self, input_ids=None, attention_mask=None, labels=None, query_time=None,
                inputs_embeds=None, query_user=None, **kwargs):
        layer = self.attention
        if not isinstance(layer, ProjectedNeighborRefinementLayer):
            raise TypeError("PairedCorrectionModel requires projected neighbour refinement")
        previous_time = layer.query_time
        previous_user = getattr(layer, "query_user", None)
        layer.query_time = query_time
        if isinstance(layer, OutcomeMemoryLayer):
            layer.query_user = query_user
        try:
            if not self.training or labels is None or input_ids is None or inputs_embeds is not None or self.pair_weight == 0:
                return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                       labels=labels, inputs_embeds=inputs_embeds, **kwargs)
            if query_time is None:
                raise ValueError("query_time is required during paired correction training")
            embeds = self.generate_embs(input_ids, attention_mask)
            positions = self._thought_positions(input_ids)
            rows = torch.arange(len(positions), device=embeds.device)
            raw = embeds[rows, positions].detach()
            own = super().forward(inputs_embeds=embeds, attention_mask=attention_mask, **kwargs)
            own_nll = self._example_nll(own.logits, labels)
            with torch.no_grad():
                control_embeds = embeds.detach().clone()
                control_raw = self._control(raw, query_time)
                control_embeds[rows, positions] = layer.refine(control_raw).to(control_embeds.dtype)
                control = super().forward(inputs_embeds=control_embeds,
                                          attention_mask=attention_mask, **kwargs)
                control_nll = self._example_nll(control.logits, labels)
            pair_loss = F.relu(self.pair_margin + own_nll - control_nll).mean()
            # Average per supervised token, matching the ordinary CE objective.
            token_counts = (labels[:, 1:] != -100).sum(-1)
            ce = (own_nll * token_counts).sum() / token_counts.sum().clamp_min(1)
            self.last_pair_loss = pair_loss.detach()
            self.last_control_nll = control_nll.mean().detach()
            self.last_own_nll = own_nll.mean().detach()
            self.pair_step += 1
            return CausalLMOutputWithPast(
                loss=ce + self.pair_weight * pair_loss, logits=own.logits,
                past_key_values=own.past_key_values, hidden_states=own.hidden_states,
                attentions=own.attentions,
            )
        finally:
            layer.query_time = previous_time
            if isinstance(layer, OutcomeMemoryLayer):
                layer.query_user = previous_user


class PairedCorrectionLogCallback(NeighborLogCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        super().on_log(args, state, control, logs=logs, model=model, **kwargs)
        if logs is not None and model is not None:
            for name in ("last_pair_loss", "last_control_nll", "last_own_nll"):
                value = getattr(model, name, None)
                if value is not None:
                    logs[name.removeprefix("last_")] = value.item()


def train(base_model: str, bank_dir: str, train_file: str, eval_file: str,
          item_emb_file: str, output_dir: str, category: str = "Musical_Instruments",
          topn: int = 32, max_delta_ratio: float = 0.1, user_weight: float = 0.5,
          pair_weight: float = 0.3, pair_margin: float = 0.01,
          seed: int = 42, sample: int = -1, eval_sample: int = -1,
          batch_size: int = 128, micro_batch_size: int = 4, num_epochs: int = 1,
          learning_rate: float = 5e-5, module_learning_rate: float = 5e-4,
          cutoff_len: int = 512, gradient_checkpointing: bool = False,
          outcome_grounded: bool = False, memory_value_mode: str = "outcome",
          retrieval_space: str = "auto"):
    if memory_value_mode not in {"outcome", "latent_only", "shuffle_outcome"}:
        raise ValueError("unknown memory_value_mode")
    if batch_size % micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if not 0 <= pair_weight or pair_margin < 0:
        raise ValueError("pair_weight and pair_margin must be nonnegative")
    init_training_run(dataset=category, config={"base_model": base_model,
                                                "output_dir": output_dir, "dataset": category})
    transformers.set_seed(seed)
    source_config = AutoConfig.from_pretrained(base_model)
    collaborative = getattr(source_config, "collaborative_contrastive", None) or {}
    if retrieval_space == "auto":
        retrieval_space = "projected" if collaborative else "history"
    if retrieval_space not in {"history", "projected"}:
        raise ValueError("retrieval_space must be auto, history, or projected")
    if not collaborative and (not outcome_grounded or pair_weight != 0 or retrieval_space != "history"):
        raise ValueError("plain/margin sources require outcome_grounded, history retrieval, and pair_weight=0")
    model_class = PairedCorrectionModel if collaborative else LatentModel
    model = model_class.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    source_user = getattr(model, "user_projection", None)
    source_target = getattr(model, "target_projection", None)
    with open(os.path.join(bank_dir, "summary.json"), encoding="utf-8") as stream:
        bank_summary = json.load(stream)
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if sha256(os.path.join(base_model, "model.safetensors")) != bank_summary.get("checkpoint_sha256"):
        raise ValueError("train bank and source checkpoint have different model hashes")
    if bank_summary.get("split") != "train" or sha256(train_file) != bank_summary.get("split_sha256"):
        raise ValueError("retrieval bank must match the training split")
    raw = torch.load(os.path.join(bank_dir, "thoughts.pt"), map_location="cpu", weights_only=True).float()
    metadata = pd.read_csv(os.path.join(bank_dir, "metadata.csv"))
    frame = pd.read_csv(train_file)
    if len(frame) != len(raw) or len(metadata) != len(raw) or \
            (metadata.target_item.values != frame.item_id.values).any():
        raise ValueError("bank thoughts and metadata must align with the training CSV")
    item_emb = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"]
    spec = {"projected": True, "mode": "knn", "topn": topn,
            "projection_dim": collaborative.get("projection_dim", 256),
            "key_dim": 256, "value_hidden": 512, "bank_size": len(raw),
            "n_items": len(item_emb), "item_dim": int(item_emb.shape[1]),
            "max_delta_ratio": max_delta_ratio, "user_weight": user_weight}
    if outcome_grounded:
        spec.update(outcome_grounded=True, memory_value_mode=memory_value_mode,
                    retrieval_space=retrieval_space,
                    eligibility="other_user_outcome_before_latest_history",
                    source_checkpoint_sha256=bank_summary["checkpoint_sha256"],
                    train_split_sha256=bank_summary["split_sha256"])
    layer_class = OutcomeMemoryLayer if outcome_grounded else ProjectedNeighborRefinementLayer
    layer = layer_class(model.attention, model.config.hidden_size, spec)
    layer.reset_new_parameters()
    for module in (layer.query, layer.key, layer.value, layer.gate,
                   layer.user_projection, layer.target_projection):
        module.to(torch.bfloat16)
    build_bank(layer, bank_dir, train_file, item_emb_file)
    if outcome_grounded and retrieval_space == "history":
        layer.initialize_history_bank(raw)
    else:
        layer.initialize_from_attempt10(source_user, source_target, raw)
    model.attention = layer
    if collaborative:
        del model.user_projection, model.target_projection
        delattr(model.config, "collaborative_contrastive")
    model.config.latent_neighbor_refinement = spec
    model.config.architectures = ["LatentModel"]
    model.pair_weight, model.pair_margin = pair_weight, pair_margin

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer("<|Thought|>")["input_ids"][0] != len(tokenizer) - 1:
        raise ValueError("source tokenizer has an unexpected thought token")
    text = CATEGORY_TEXT[category]
    dataset_class = OutcomeMemoryDataset if outcome_grounded else TimedDataset
    train_data = dataset_class(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                              sample=sample, seed=seed, category=text, K=0)
    val_data = dataset_class(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                            sample=eval_sample, seed=seed, category=text, K=0)
    to_hf = lambda d: HFDataset.from_dict({k: [x[k] for x in d] for k in d[0]})
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                     return_tensors="pt", padding=True)
    trainer = TwoRateTrainer(
        model=model, train_dataset=to_hf(train_data), eval_dataset=to_hf(val_data),
        processing_class=tokenizer, data_collator=collator, callbacks=[PairedCorrectionLogCallback()],
        module_lr=module_learning_rate,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            num_train_epochs=num_epochs, learning_rate=learning_rate, bf16=True,
            logging_steps=1, warmup_steps=20, optim="adamw_torch",
            eval_strategy="epoch", save_strategy="epoch" if outcome_grounded else "steps", save_steps=25,
            save_total_limit=1 if outcome_grounded else 2, save_only_model=outcome_grounded,
            load_best_model_at_end=outcome_grounded,
            gradient_checkpointing=gradient_checkpointing,
            output_dir=output_dir,
            report_to=wandb_report_to(), remove_unused_columns=False, seed=seed,
        ),
    )
    move_callback_to_end(trainer, transformers.integrations.WandbCallback)
    model.config.use_cache = False
    checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    trainer.train(resume_from_checkpoint=checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
