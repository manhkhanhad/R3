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
from collab_latent import CAND_TOKEN, CandidateSlotLatentModel, CollabCandidateLatentRDataset
from verified_evidence_latent import EvidenceLatentRDataset, VerifiedEvidenceLatentModel
from vrec_group_latent import VRecGroupLatentModel
from mux_latent import MuxLatentModel, MuxLatentRDataset, mux_table
from experiment_tracking import init_training_run, wandb_report_to
# torch.backends.cuda.enable_cudnn_sdp(False)

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
    # collaborative candidates (document/collaborative_reasoning_experiments.md)
    collab_candidates_file: str = None,  # build_candidates.py table: {"train": [n, K], "valid": [n, K], ...}
    collab_mode: str = "text",  # text: candidate titles in the prompt | slots: <|Cand|> slots with SASRec item vectors
    collab_item_table: str = None,  # slots: per-item vectors {"embeddings": [n_items, d], "valid": [n_items]}
    collab_titles_file: str = None,  # text: catalog "<title>\t<item_id>" file (default: data/<category>/info/*.txt)
    # trained evidence verifier (document/collaborative_reasoning_experiments.md)
    verified_evidence_from: str = None,  # oracle_category side-info checkpoint to start from (replaces base_model)
    vrec_groups_file: str = None,  # VRec-style nudge toward coarse groups: build_group_table.py table (base_model = an R3 checkpoint)
    evidence_weight: float = 0.5,  # weight of the verifier cross-entropy
    evidence_gate: str = "entropy",  # entropy | none
    evidence_label_smoothing: float = 0.1,
    evidence_oracle_mix_start: float = 0.5,  # fraction of rows given the true evidence at the start
    evidence_oracle_mix_epochs: float = 1.0,  # linear decay of that fraction to 0
    # MUX-style latent supervision (document/mux_latent_experiments.md)
    latent_builder: str = None,  # pool (R3 attention pooling) | codi (MLP+LayerNorm on h[thought-1]); enables MuxLatentModel
    mux_category_file: str = None,  # category table with "paths" and "valid" (target of the mux loss)
    mux_weight: float = 0.0,  # weight of KL(mux(category tokens) || softmax(W r / tau)); 0 = no supervision
    mux_tau: float = 1.0,
    mux_rho: float = 0.9,  # geometric decay of the mux weights, leaf-first
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
            "collab_candidates_file": collab_candidates_file,
            "collab_mode": collab_mode,
            "collab_item_table": collab_item_table,
            "verified_evidence_from": verified_evidence_from,
            "vrec_groups_file": vrec_groups_file,
            "latent_builder": latent_builder,
            "mux_category_file": mux_category_file,
            "mux_weight": mux_weight,
            "mux_tau": mux_tau,
            "mux_rho": mux_rho,
            "evidence_weight": evidence_weight,
            "evidence_gate": evidence_gate,
            "evidence_oracle_mix_start": evidence_oracle_mix_start,
            "evidence_oracle_mix_epochs": evidence_oracle_mix_epochs,
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
        elif side_info_kind == "row_table":
            rows = torch.load(side_info_file, map_location="cpu", weights_only=False)
            side_rows = {train_file: 0, eval_file: len(rows["train"])}
            side_table = torch.cat([rows["train"].float(), rows["valid"].float()])
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

    use_collab = collab_candidates_file is not None
    if use_collab and (use_side_info or use_category_alignment or category_text_file):
        raise ValueError("collaborative candidates are a plain-R3 variant")
    if use_collab and collab_mode not in ("text", "slots"):
        raise ValueError("collab_mode must be text or slots")
    if use_collab:
        collab_candidates = torch.load(collab_candidates_file, map_location="cpu", weights_only=False)
    if use_collab and collab_mode == "slots":
        from transformers import AutoConfig
        item_table = torch.load(collab_item_table, map_location="cpu", weights_only=False)
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_candidate_slots = {
            "k": int(collab_candidates["k"]),
            "dimension": int(item_table["embeddings"].shape[1]),
            "source": os.path.abspath(collab_item_table),
            "candidates": os.path.abspath(collab_candidates_file),
        }
        model_class = CandidateSlotLatentModel
        model_kwargs["config"] = model_config

    use_evidence = verified_evidence_from is not None
    evidence_table = None
    if use_evidence:
        if use_side_info or use_category_alignment or use_collab or category_text_file:
            raise ValueError("the evidence verifier starts from an oracle side-info checkpoint and adds nothing else")
        from transformers import AutoConfig
        base_model = verified_evidence_from
        model_config = AutoConfig.from_pretrained(base_model)
        spec = getattr(model_config, "latent_side_info", None)
        if not spec or spec.get("kind") != "oracle_category" or spec.get("token_scale"):
            raise ValueError("verified_evidence_from must be a raw-scale oracle_category checkpoint")
        evidence_table = torch.load(spec["source"], map_location="cpu", weights_only=False)
        if "cluster_id" not in evidence_table:
            raise ValueError("the oracle side table has no cluster_id: evidence must be clusters")
        model_config.latent_evidence = {
            "k": int(evidence_table["cluster_id"].max()) + 1,
            "source": spec["source"],
            "weight": evidence_weight,
            "gate": evidence_gate,
            "label_smoothing": evidence_label_smoothing,
            "oracle_mix_start": evidence_oracle_mix_start,
            "oracle_mix_examples": 0,  # set below from the training-set size
            "start_checkpoint": os.path.abspath(verified_evidence_from),
        }
        model_class = VerifiedEvidenceLatentModel
        model_kwargs["config"] = model_config

    use_vrec = vrec_groups_file is not None
    if use_vrec:
        if use_evidence or use_side_info or use_category_alignment or use_collab or category_text_file:
            raise ValueError("the VRec-style verifier is added to a plain R3 checkpoint and nothing else")
        from transformers import AutoConfig
        evidence_table = torch.load(vrec_groups_file, map_location="cpu", weights_only=False)
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_vrec = {
            "k": int(evidence_table["cluster_id"].max()) + 1,
            "source": os.path.abspath(vrec_groups_file),
            "weight": evidence_weight,
            "gate": evidence_gate,
            "label_smoothing": evidence_label_smoothing,
            "start_checkpoint": os.path.abspath(base_model),
        }
        model_class = VRecGroupLatentModel
        model_kwargs["config"] = model_config

    use_mux = latent_builder is not None
    if use_mux:
        if use_vrec or use_evidence or use_side_info or use_category_alignment or use_collab or category_text_file:
            raise ValueError("the MUX latent is a plain-R3 variant")
        if mux_weight > 0 and not mux_category_file:
            raise ValueError("mux_category_file is required when mux_weight > 0")
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(base_model)
        if not getattr(model_config, "tie_word_embeddings", False):
            raise ValueError("the mux loss decodes r with the tied unembedding; this backbone is untied")
        model_config.latent_mux = {
            "builder": latent_builder,
            "weight": mux_weight,
            "tau": mux_tau,
            "rho": mux_rho,
            "category_file": os.path.abspath(mux_category_file) if mux_category_file else None,
        }
        model_class = MuxLatentModel
        model_kwargs["config"] = model_config

    model = model_class.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
        # device_map=device_map,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    if use_evidence:
        model.reset_verifier(seed)
        model.set_evidence_table(evidence_table)
        print("evidence verifier:", model.config.latent_evidence)
    if use_mux:
        model.reset_latent_builder(seed)
    if use_vrec:
        model.reset_verifier(seed)
        model.set_group_table(evidence_table)
        print("VRec-style verifier:", model.config.latent_vrec)
    if use_category_alignment:
        model.set_category_targets(category_targets["embeddings"], category_targets["valid"])
    
    ##############################################################################
    ##################                                 
    ##################   add a special token here to represent latent thought
    ##################

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    if use_collab and collab_mode == "slots":
        additional_special_tokens.append(CAND_TOKEN)  # before the thought token, which must stay last
    if "<|Thought|>" not in additional_special_tokens:  # already there when starting from an R3 checkpoint
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
    if use_collab and collab_mode == "slots":
        token_norm = float(model.model.embed_tokens.weight[:-2].float().norm(dim=-1).mean())
        cand_id = tokenizer.convert_tokens_to_ids(CAND_TOKEN)
        assert cand_id == len(tokenizer) - 2
        model.config.latent_candidate_slots.update({"token_norm": token_norm, "cand_token_id": cand_id})
        model.cand_token_norm, model.cand_token_id = token_norm, cand_id
        model.set_cand_table(item_table["embeddings"], item_table["valid"])
        print("candidate slots:", model.config.latent_candidate_slots)
    
    ##################
    ##############################################################################

    dataset_class = CategoryAlignedLatentRDataset if use_category_alignment else LatentRDataset
    extra_train, extra_val = {}, {}
    if use_side_info and side_info_kind != "none":
        dataset_class = SideInfoLatentRDataset
        if side_info_kind in ("sasrec_sequence", "row_table"):
            if sample > 0:
                raise ValueError(f"{side_info_kind} side information requires sample=-1 (row-indexed table)")
            extra_train = {"side_key": "row", "row_offset": side_rows[train_file]}
            extra_val = {"side_key": "row", "row_offset": side_rows[eval_file]}
    if category_text_file:
        if use_side_info or use_category_alignment:
            raise ValueError("category text control is a plain-R3 run")
        paths = torch.load(category_text_file, map_location="cpu", weights_only=True)
        category_paths = {i: p for i, p in enumerate(paths["paths"]) if paths["valid"][i]}
        dataset_class = CategoryTextLatentRDataset
        extra_train = extra_val = {"category_paths": category_paths}
    if use_collab:
        if collab_titles_file is None:
            import glob
            collab_titles_file = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(train_file)), "info", "*.txt")))[0]
        titles = {}
        for line in open(collab_titles_file):
            t, i = line.rstrip("\n").rsplit("\t", 1)
            titles[int(i)] = t.strip(' \n"')
        dataset_class = CollabCandidateLatentRDataset
        extra_train = {"candidates": collab_candidates["train"], "titles": titles, "mode": collab_mode}
        extra_val = {"candidates": collab_candidates["valid"], "titles": titles, "mode": collab_mode}
    if use_evidence or use_vrec:
        dataset_class = EvidenceLatentRDataset
    if use_mux:
        dataset_class = MuxLatentRDataset
        if mux_category_file:
            cats = torch.load(mux_category_file, map_location="cpu", weights_only=False)
            mux_ids, mux_w = mux_table(cats["paths"], cats["valid"], tokenizer, rho=mux_rho)
            model.set_mux_table(mux_ids, mux_w)
            print("mux targets:", int((mux_w.sum(-1) > 0).sum()), "items; example", tokenizer.decode(mux_ids[1][mux_w[1] > 0]))
    train_data = dataset_class(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K, **extra_train)
    val_data = dataset_class(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category, K=K, **extra_val)

    print("LOAD DATA FINISHED")
    if use_evidence:
        mix_examples = int(evidence_oracle_mix_epochs * len(train_data))
        model.evidence_mix_examples = mix_examples
        model.config.latent_evidence["oracle_mix_examples"] = mix_examples

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
            load_best_model_at_end=True,
            label_names=["labels"] if (use_category_alignment or use_side_info or use_collab or use_evidence or use_vrec or use_mux) else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
