import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
import fire
import torch

import transformers
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from category_alignment_attention import CategoryAlignedLatentModel
from category_alignment_dataset import CategoryAlignedLatentRDataset
from side_info_latent import SideInfoLatentModel
from side_info_dataset import SideInfoLatentRDataset, CategoryTextLatentRDataset
from multi_aspect_model import MultiAspectLatentModel, MultiAspectLatentRDataset
from staged_refine_model import StagedRefineLatentModel, StagedRefineLatentRDataset
from experiment_tracking import init_training_run, wandb_report_to
# torch.backends.cuda.enable_cudnn_sdp(False)

class AspectLogCallback(transformers.TrainerCallback):
    """Adds per-aspect losses, accuracies, gates and confidences to the trainer logs."""

    def __init__(self, model):
        self.model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        stats = getattr(self.model, "last_aspect_stats", None)
        if logs is not None and stats and "loss" in logs:
            logs.update({k: round(float(v), 4) for k, v in stats.items()})


class StagedRefineCallback(transformers.TrainerCallback):
    """Stage switch / oracle warm-up for staged-refinement R3, plus per-head logs.

    Stage 1 for the first ``stage1_epochs`` epochs, stage 2 afterwards. In stage 2
    the oracle fraction decays linearly from 1 to 0 over ``warmup_epochs``.
    Validation statistics per head are appended to <output_dir>/epoch_metrics.jsonl.
    """

    def __init__(self, model, stage1_epochs, warmup_epochs):
        self.model, self.stage1_epochs, self.warmup_epochs = model, stage1_epochs, warmup_epochs

    def on_step_begin(self, args, state, control, **kwargs):
        att = self.model.attention
        epoch = float(state.epoch or 0.0)
        stage2 = epoch >= self.stage1_epochs - 1e-6
        att.stage.fill_(2 if stage2 else 1)
        if stage2 and self.warmup_epochs > 0:
            att.oracle_alpha = max(0.0, 1.0 - (epoch - self.stage1_epochs) / self.warmup_epochs)
        else:
            att.oracle_alpha = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        stats = getattr(self.model, "last_stats", None)
        if logs is not None and stats and "loss" in logs:
            logs.update({k: round(float(v), 4) for k, v in stats.items()})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import json
        summary = self.model.eval_summary()
        record = {"epoch": round(float(state.epoch or 0.0), 3), "step": state.global_step,
                  "stage": int(self.model.attention.stage), "eval_loss": (metrics or {}).get("eval_loss"), **summary}
        print("staged-refine validation:", record, flush=True)
        if state.is_world_process_zero:
            with open(os.path.join(args.output_dir, "epoch_metrics.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        self.model.reset_eval_stats()


def train(
    # model/data params
    base_model: str = "/data1/huggingface/models--Qwen--Qwen2.5-1.5B/",  # the only required argument
    train_file: str="data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str="data/train/Toys_and_Games_5_2017-1-2018-11.csv",
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
    deepspeed: str ="./deepspeed.json",
    category: str="Toys_and_Games",
    K: int = 0,
    wandb_run_name: str = None,
    category_embeddings_file: str = None,
    category_alignment_weight: float = 0.0,
    # decoder-insensitivity tests (analysis/document/decoder_insensitivity_to_latent.md)
    side_info_kind: str = None,  # oracle_category | random_category | sasrec_sequence | none
    side_info_file: str = None,  # category-embedding .pt, or SASRec checkpoint for sasrec_sequence
    side_init_norm: float = 30.0,
    side_token_scale: bool = False,
    category_text_file: str = None,  # text positive control: category-embedding .pt with "paths"
    # multi-aspect R3 (src/latent/multi_aspect_attention.py)
    aspect_file: str = None,  # .pt from build_aspect_tables.py
    aspects: str = "item,category,sid_l1,sid_l2",
    aspect_loss_weight: float = 0.1,
    aspect_fusion: str = "confidence",  # confidence | mean
    # staged-refinement R3 (src/latent/staged_refine_attention.py)
    refine_cluster_file: str = None,  # .pt from build_sasrec_cluster_table.py
    refine_mode: str = "centroid",  # vrec | centroid
    refine_stage1_epochs: float = 2.0,
    refine_warmup_epochs: float = 0.0,  # stage-2 oracle warm-up length (alpha 1 -> 0); 0 = none
    refine_loss_weight: float = 0.1,
    refine_init_norm: float = 30.0,  # centroid mode: initial gain (oracle_raw value)
    save_only_model: bool = False,  # checkpoints without optimizer state (~3 GB instead of ~9 GB)
    save_total_limit: int = 5,

):

    init_training_run(
        dataset=category,
        run_name=wandb_run_name,
        config={
            "base_model": base_model,
            "train_file": train_file,
            "eval_file": eval_file,
            "output_dir": output_dir,
            "dataset": category,
            "category_embeddings_file": category_embeddings_file,
            "category_alignment_weight": category_alignment_weight,
            "side_info_kind": side_info_kind,
            "side_info_file": side_info_file,
            "side_init_norm": side_init_norm,
            "side_token_scale": side_token_scale,
            "category_text_file": category_text_file,
            "aspect_file": aspect_file,
            "aspects": aspects,
            "aspect_loss_weight": aspect_loss_weight,
            "aspect_fusion": aspect_fusion,
            "refine_cluster_file": refine_cluster_file,
            "refine_mode": refine_mode,
            "refine_stage1_epochs": refine_stage1_epochs,
            "refine_warmup_epochs": refine_warmup_epochs,
            "refine_loss_weight": refine_loss_weight,
        },
    )

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

    if category_alignment_weight < 0:
        raise ValueError("category_alignment_weight must be nonnegative")
    use_category_alignment = category_alignment_weight > 0
    if use_category_alignment and not category_embeddings_file:
        raise ValueError("category_embeddings_file is required when category alignment is enabled")
    category_targets = None
    model_class = LatentModel
    model_kwargs = {}
    if use_category_alignment:
        category_targets = torch.load(category_embeddings_file, map_location="cpu", weights_only=True)
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.category_alignment = {
            "dimension": int(category_targets["embeddings"].shape[1]),
            "weight": category_alignment_weight,
        }
        model_class = CategoryAlignedLatentModel
        model_kwargs["config"] = model_config

    use_multi_aspect = aspect_file is not None
    aspect_table = None
    if use_multi_aspect:
        if category_alignment_weight > 0 or side_info_kind is not None:
            raise ValueError("multi-aspect cannot be combined with category alignment or side information")
        from transformers import AutoConfig
        aspect_table = torch.load(aspect_file, map_location="cpu", weights_only=True)
        names = [a for a in aspects.split(",") if a]
        rows = [aspect_table["names"].index(a) for a in names]
        aspect_codes = aspect_table["codes"][rows]
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_multi_aspect = {
            "names": names,
            "num_classes": [int(aspect_table["num_classes"][i]) for i in rows],
            "n_items": int(aspect_codes.shape[1]),
            "fusion": aspect_fusion,
            "loss_weights": [aspect_loss_weight] * len(names),
            "source": os.path.abspath(aspect_file),
        }
        model_class = MultiAspectLatentModel
        model_kwargs["config"] = model_config

    use_staged = refine_cluster_file is not None
    cluster_table = None
    if use_staged:
        if use_multi_aspect or use_category_alignment or side_info_kind is not None:
            raise ValueError("staged refinement cannot be combined with other latent options")
        from transformers import AutoConfig
        cluster_table = torch.load(refine_cluster_file, map_location="cpu", weights_only=True)
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_staged_refine = {
            "mode": refine_mode,
            "num_clusters": int(cluster_table["k"]),
            "centroid_dim": int(cluster_table["centroids"].shape[1]),
            "n_items": int(cluster_table["cluster_id"].shape[0]),
            "init_norm": refine_init_norm,
            "loss_weight": refine_loss_weight,
            "stage1_epochs": refine_stage1_epochs,
            "warmup_epochs": refine_warmup_epochs,
            "source": os.path.abspath(refine_cluster_file),
        }
        model_class = StagedRefineLatentModel
        model_kwargs["config"] = model_config

    use_side_info = side_info_kind is not None
    if use_side_info and use_category_alignment:
        raise ValueError("side-information and category alignment cannot be combined")
    side_table = side_valid = None
    side_rows = {}
    if use_side_info:
        from transformers import AutoConfig
        if side_info_kind in ("oracle_category", "random_category"):
            targets = torch.load(side_info_file, map_location="cpu", weights_only=True)
            side_table, side_valid = targets["embeddings"].float(), targets["valid"].bool()
        elif side_info_kind == "sasrec_sequence":
            from sasrec_pair_features import build_sasrec_features
            _, rows = build_sasrec_features(side_info_file, [train_file, eval_file])
            side_rows = {train_file: 0, eval_file: len(rows[train_file][0])}
            side_table = torch.cat([torch.from_numpy(rows[train_file][0]), torch.from_numpy(rows[eval_file][0])])
        elif side_info_kind != "none":
            raise ValueError(f"unknown side_info_kind {side_info_kind}")
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_side_info = {
            "kind": side_info_kind,
            "source": os.path.abspath(side_info_file) if side_info_file else None,
            "dimension": int(side_table.shape[1]) if side_table is not None else 1,
            "init_norm": side_init_norm,
            "token_scale": side_token_scale,
        }
        model_class = SideInfoLatentModel
        model_kwargs["config"] = model_config

    model = model_class.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
        # device_map=device_map,
    )
    model.attention.end_k = end_k
    if use_multi_aspect:
        model.attention.aspect_codes.copy_(aspect_codes)
        model.grad_accum_steps = gradient_accumulation_steps
        # Gate parameters in fp32: with pure-bf16 training, updates of ~lr (3e-4) are
        # below bf16 resolution near the scale's value (~2.1, spacing 0.016) and are lost.
        att = model.attention
        att.gate_raw_scale = torch.nn.Parameter(att.gate_raw_scale.detach().float())
        att.gate_bias = torch.nn.Parameter(att.gate_bias.detach().float())
        print("multi-aspect:", model.config.latent_multi_aspect)
    if use_staged:
        att = model.attention
        att.cluster_codes.copy_(cluster_table["cluster_id"])
        att.centroids.copy_(cluster_table["centroids"].float())
        att.stage.fill_(1)
        model.grad_accum_steps = gradient_accumulation_steps
        if refine_mode == "centroid":
            # fp32 gain: bf16 spacing near 30 (0.125) would swallow updates of ~lr
            att.gain = torch.nn.Parameter(att.gain.detach().float())
        print("staged refinement:", model.config.latent_staged_refine)
    model.config.loss_type = "ce"
    if use_category_alignment:
        model.set_category_targets(category_targets["embeddings"], category_targets["valid"])
    
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
    if use_side_info:
        # token scale = mean input-embedding norm of the real vocabulary (fixed at init)
        token_norm = float(model.model.embed_tokens.weight[:-1].float().norm(dim=-1).mean())
        model.config.latent_side_info["token_norm"] = token_norm
        model.side_token_norm = token_norm
        if side_table is not None:
            model.set_side_table(side_table, side_valid)
        print("side information:", model.config.latent_side_info)
    
    ##################
    ##############################################################################

    dataset_class = CategoryAlignedLatentRDataset if use_category_alignment else LatentRDataset
    extra_train, extra_val = {}, {}
    if use_side_info and side_info_kind != "none":
        dataset_class = SideInfoLatentRDataset
        if side_info_kind == "sasrec_sequence":
            if sample > 0:
                raise ValueError("sasrec_sequence side information requires sample=-1 (row-indexed table)")
            extra_train = {"side_key": "row", "row_offset": side_rows[train_file]}
            extra_val = {"side_key": "row", "row_offset": side_rows[eval_file]}
    if category_text_file:
        if use_side_info or use_category_alignment:
            raise ValueError("category text control is a plain-R3 run")
        paths = torch.load(category_text_file, map_location="cpu", weights_only=True)
        category_paths = {i: p for i, p in enumerate(paths["paths"]) if paths["valid"][i]}
        dataset_class = CategoryTextLatentRDataset
        extra_train = extra_val = {"category_paths": category_paths}
    if use_multi_aspect:
        dataset_class = MultiAspectLatentRDataset
    if use_staged:
        dataset_class = StagedRefineLatentRDataset
    train_data = dataset_class(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K, **extra_train)
    val_data = dataset_class(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category, K=K, **extra_val)

    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    print("train_data 0 keys:", train_data[0].keys())
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
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
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
            # staged runs: eval loss is not comparable across the stage switch; keep the final model
            load_best_model_at_end=not use_staged,
            label_names=["labels"] if (use_category_alignment or use_side_info or use_multi_aspect or use_staged) else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = ([] if use_staged else [EarlyStoppingCallback(early_stopping_patience=1)])
            + ([AspectLogCallback(model)] if use_multi_aspect else [])
            + ([StagedRefineCallback(model, refine_stage1_epochs, refine_warmup_epochs)] if use_staged else []),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
