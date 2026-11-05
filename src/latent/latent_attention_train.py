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
from experiment_tracking import init_training_run, wandb_report_to
# torch.backends.cuda.enable_cudnn_sdp(False)


class BottleneckSchedule(transformers.TrainerCallback):
    """Latent-only bottleneck for epochs [0, n), full attention from epoch n on.

    The bottleneck is read from model.config.latent_bottleneck at every forward
    (layers.py), so removing the key switches training, epoch-end evaluation and the
    saved config to plain R3. Checkpoints saved at the end of epochs < n keep the
    bottleneck in their config. Also correct after resuming from a checkpoint, since
    state.epoch is restored before on_epoch_begin.
    """

    def __init__(self, model, n):
        self.model, self.n = model, n
        self.spec = model.config.latent_bottleneck

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch + 1e-4)
        latent = epoch < self.n
        self.model.config.latent_bottleneck = self.spec if latent else None
        print(f"[BottleneckSchedule] epoch {epoch}: {'latent-only bottleneck' if latent else 'full attention'}", flush=True)

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
    save_only_model: bool = False,  # checkpoints without optimizer state (~3 GB instead of ~9 GB)
    save_total_limit: int = 5,
    # L7 structural bottleneck: target tokens see only the latent + fixed instruction
    # prefix, never the history, in training and generation (latent_bottleneck.py)
    latent_only: bool = False,
    # Continuous thoughts (continuous_thoughts.py): "hidden" = Coconut-style K-step
    # thoughts replacing the pooled r; "constant" = K learned vectors (baseline C)
    thought_source: str = None,
    n_thoughts: int = 1,
    # Gist bottleneck (latent_bottleneck.py): with latent_only, the thought tokens
    # read the history at every layer; the target reads it only through them
    latent_gist: bool = False,
    # Soft bottleneck (latent_bottleneck.py): with latent_only, a finite negative
    # attention bias on the history for target-side queries instead of hiding it
    history_bias: float = None,
    # Staged training: with latent_only, the bottleneck is active for the first
    # latent_only_epochs epochs and removed afterwards (full attention, plain R3), so the
    # saved model is evaluated with history visible. Disables early stopping / best-model reload.
    latent_only_epochs: int = None,
    # History-only pooling (history_pooling.py): r pools over the history title tokens only
    pool_history_only: bool = False,
    # B1 steered reading (steered_reading.py): the thought selects which k history
    # items the recommendation pass reads; selector trained toward the oracle items
    steered_k: int = 0,
    steered_rule: str = "learned",  # learned | recent (fixed-rule baseline)
    steered_aux_weight: float = 0.2,
    item_embeddings_file: str = None,  # Qwen3 title embeddings (oracle items)
    # SASRec-informed pooling (candidate_pooling.py): r's pooling also attends to the
    # row's SASRec top-C candidates; prompt unchanged; base_model = a trained R3 checkpoint
    candidate_kind: str = None,  # title | sasrec
    candidates_file: str = None,  # sasrec_candidates.py output
    title_vectors_file: str = None,  # candidate_title_vectors.py output
    n_candidates: int = 10,
    candidate_shuffle_seed: int = None,  # control: every row gets another user's candidates
    candidate_source: str = "sasrec",  # sasrec | random (C random non-history items per row)
    candidate_anneal: str = None,  # "start,end" training fractions: branch scale 1 -> 0
    eval_sample: int = None,  # validation rows (default: sample)
    pool_diagnostics: bool = False,  # log R3 pooling concentration / candidate weights per evaluation
    # Category-conditioned latents (category_latent.py): additive | inject (IntuRec-style);
    # trained with the target category (additive) / target + predicted list (inject)
    category_injection: str = None,
    category_prep_file: str = None,  # category_prep.py output
    # Diffusion-refined thought (diffusion_refine.py): r' = g*normalize(P(diffuse(r))),
    # denoiser target = the target item's vector from an item_oracle_prep.py table
    diffusion_refine_file: str = None,
    diffusion_mode: str = "diffusion",  # diffusion | regress (deterministic control)
    diffusion_steps: int = 32,
    diffusion_weight: float = 1.0,
    diffusion_ce_weight: float = 1.0,
    diffusion_tau: float = 0.05,
    diffusion_chain_prob: float = 0.5,
    diffusion_width: int = 2048,
    diffusion_depth: int = 3,

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
            "latent_only": latent_only,
            "thought_source": thought_source,
            "n_thoughts": n_thoughts,
            "latent_gist": latent_gist,
            "history_bias": history_bias,
            "latent_only_epochs": latent_only_epochs,
            "pool_history_only": pool_history_only,
            "steered_k": steered_k,
            "steered_rule": steered_rule,
            "steered_aux_weight": steered_aux_weight,
            "candidate_kind": candidate_kind,
            "candidates_file": candidates_file,
            "n_candidates": n_candidates,
            "candidate_shuffle_seed": candidate_shuffle_seed,
            "candidate_source": candidate_source,
            "candidate_anneal": candidate_anneal,
            "category_injection": category_injection,
            "diffusion_refine_file": diffusion_refine_file,
            "diffusion_mode": diffusion_mode,
            "diffusion_steps": diffusion_steps,
            "diffusion_weight": diffusion_weight,
            "diffusion_ce_weight": diffusion_ce_weight,
            "diffusion_chain_prob": diffusion_chain_prob,
            "learning_rate": learning_rate,
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

    if thought_source is not None:
        if use_side_info or use_category_alignment or category_text_file:
            raise ValueError("continuous thoughts are a plain-R3 variant; do not combine with other options")
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_continuous_thoughts = {"k": n_thoughts, "source": thought_source}
        if steered_k > 0:
            if latent_only:
                raise ValueError("steered reading does not combine with latent_only")
            if thought_source != "hidden" and steered_rule == "learned":
                raise ValueError("learned selection needs thought_source=hidden "
                                 "(constant thoughts only with the fixed recent rule)")
            model_config.latent_steered_reading = {"k": steered_k, "rule": steered_rule,
                                                   "aux_weight": steered_aux_weight, "dim": 256}
        model_kwargs["config"] = model_config
    elif n_thoughts != 1:
        raise ValueError("n_thoughts > 1 needs thought_source")

    category_prep = None
    if category_injection is not None:
        if use_side_info or use_category_alignment or category_text_file or thought_source or latent_only \
                or candidate_kind:
            raise ValueError("category latents are a plain-R3 variant; do not combine with other options")
        from transformers import AutoConfig
        category_prep = torch.load(category_prep_file, map_location="cpu", weights_only=False)
        model_config = AutoConfig.from_pretrained(base_model)
        if category_injection == "additive":
            model_config.latent_side_info = {
                "kind": "oracle_category", "source": os.path.abspath(category_prep_file),
                "dimension": int(category_prep["embeddings"].shape[1]), "init_norm": side_init_norm,
                "token_scale": False, "table": "category",
            }
            model_class = SideInfoLatentModel
        elif category_injection == "inject":
            from category_latent import CategoryInjectModel
            tok0 = AutoTokenizer.from_pretrained(base_model)
            model_config.latent_category_inject = {
                "k": int(category_prep["lists"]["train"].shape[1]), "source": os.path.abspath(category_prep_file),
                "token_ids": category_prep["token_ids"],
                "unknown_token_ids": tok0.encode("unknown", add_special_tokens=False),
                "pad_id": int(tok0.pad_token_id if tok0.pad_token_id is not None else tok0.eos_token_id),
            }
            model_class = CategoryInjectModel
        else:
            raise ValueError(f"unknown category_injection {category_injection}")
        model_kwargs["config"] = model_config

    diffusion_prep = None
    if diffusion_refine_file is not None:
        if category_injection is not None or use_side_info or use_category_alignment or category_text_file \
                or thought_source or latent_only or candidate_kind or pool_history_only:
            raise ValueError("diffusion refinement is a plain-R3 variant; do not combine with other options")
        if diffusion_mode not in ("diffusion", "regress"):
            raise ValueError(f"unknown diffusion_mode {diffusion_mode}")
        from transformers import AutoConfig
        from diffusion_refine import DiffusionRefineModel
        diffusion_prep = torch.load(diffusion_refine_file, map_location="cpu", weights_only=False)
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.latent_diffusion_refine = {
            "source": os.path.abspath(diffusion_refine_file), "kind": diffusion_prep.get("kind"),
            "dimension": int(diffusion_prep["embeddings"].shape[1]), "mode": diffusion_mode,
            "steps": diffusion_steps, "weight": diffusion_weight, "ce_weight": diffusion_ce_weight,
            "tau": diffusion_tau, "chain_prob": diffusion_chain_prob,
            "width": diffusion_width, "depth": diffusion_depth, "eval_seed": 0,
        }
        model_class = DiffusionRefineModel
        model_kwargs["config"] = model_config

    candidates = None
    if candidate_kind is not None:
        if use_side_info or use_category_alignment or category_text_file or thought_source or latent_only:
            raise ValueError("candidate pooling is a plain-R3 variant; do not combine with other options")
        from transformers import AutoConfig
        from candidate_pooling import CandidatePoolingModel
        candidates = torch.load(candidates_file, map_location="cpu", weights_only=False)
        model_config = AutoConfig.from_pretrained(base_model)
        if candidate_source not in ("sasrec", "random"):
            raise ValueError(f"unknown candidate_source {candidate_source}")
        model_config.latent_candidate_pooling = {
            "kind": candidate_kind, "k": n_candidates,
            "candidates_file": os.path.abspath(candidates_file),
            "sasrec_dim": int(candidates["item_embeddings"].shape[1]),
            "shuffle_seed": candidate_shuffle_seed,
            "source": candidate_source,
            "n_items": int(candidates["item_embeddings"].shape[0]),
            # fire parses "a,b" as a tuple; accept a string too
            "anneal": [float(x) for x in (candidate_anneal.split(",") if isinstance(candidate_anneal, str)
                                          else candidate_anneal)] if candidate_anneal else None,
        }
        if title_vectors_file:
            model_config.latent_candidate_pooling.update(
                title_vectors=os.path.abspath(title_vectors_file),
                title_norm=float(torch.load(title_vectors_file, map_location="cpu", weights_only=False)["mean_norm"]))
        elif candidate_kind == "sasrec":
            model_config.latent_candidate_pooling["sasrec_scale"] = "hidden"
        else:
            raise ValueError("candidate_kind=title needs title_vectors_file")
        model_class = CandidatePoolingModel
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
    if thought_source is not None:
        # HF re-initialised the (missing) thought weights; restore the intended fp32
        # init and put thoughts at the mean input-token-embedding norm.
        token_norm = float(model.model.embed_tokens.weight[:-1].float().norm(dim=-1).mean())
        model.config.latent_continuous_thoughts["token_norm"] = token_norm
        model.thoughts.token_norm.fill_(token_norm)
        model.thoughts.reset_parameters()
        print("continuous thoughts:", model.config.latent_continuous_thoughts)
        if steered_k > 0:
            model.steer.float()
            for layer in (model.steer.query, model.steer.key):
                layer.reset_parameters()
            print("steered reading:", model.config.latent_steered_reading)
    if candidates is not None:
        print("candidate pooling:", model.config.latent_candidate_pooling)
    if category_injection == "additive":
        # fp32 side branch, category-indexed table (+ one invalid row for unknown categories)
        model.side_projection.float().reset_parameters()
        with torch.no_grad():
            model.side_gain.data = torch.tensor(float(side_init_norm), device=model.side_gain.device)
        table = torch.cat([category_prep["embeddings"], torch.zeros(1, category_prep["embeddings"].shape[1])])
        valid = torch.ones(len(table), dtype=torch.bool); valid[-1] = False
        model.set_side_table(table, valid)
        print("category latent (additive):", {k: v for k, v in model.config.latent_side_info.items()})
    elif category_injection == "inject":
        print("category latent (inject): k =", model.config.latent_category_inject["k"])
    if diffusion_prep is not None:
        # fp32 denoiser, freshly initialised (HF leaves missing weights uninitialised)
        model.diffuse.float()
        for m in model.diffuse.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)):
                m.reset_parameters()
        with torch.no_grad():
            model.diffuse.gain.fill_(30.0)
        model.set_item_table(diffusion_prep["embeddings"], diffusion_prep["valid"])
        print("diffusion refinement:", model.config.latent_diffusion_refine)
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
    if n_thoughts != 1:
        extra_train = {**extra_train, "n_thoughts": n_thoughts}
        extra_val = {**extra_val, "n_thoughts": n_thoughts}
    if steered_k > 0:
        from steered_reading import SteeredReadingDataset
        dataset_class = SteeredReadingDataset
        item_embeddings = torch.load(item_embeddings_file, map_location="cpu", weights_only=False)["embeddings"].float()
        extra_train = {**extra_train, "item_embeddings": item_embeddings}
        extra_val = {**extra_val, "item_embeddings": item_embeddings}
    if category_prep is not None:
        from category_latent import CategoryRowsDataset
        dataset_class = CategoryRowsDataset
        unknown = len(category_prep["catalog"])
        extra_train = {"target": category_prep["target"]["train"], "lists": category_prep["lists"]["train"],
                       "injection": category_injection, "mode": "oracle", "unknown": unknown}
        extra_val = {"target": category_prep["target"]["valid"], "lists": category_prep["lists"]["valid"],
                     "injection": category_injection, "mode": "oracle", "unknown": unknown}
    if diffusion_prep is not None:
        # the target item index rides in ``side_index`` (CategoryRowsDataset, mode oracle)
        from category_latent import CategoryRowsDataset
        dataset_class = CategoryRowsDataset
        unknown = len(diffusion_prep["catalog"])
        extra_train = {"target": diffusion_prep["target"]["train"], "lists": diffusion_prep["lists"]["train"],
                       "injection": "additive", "mode": "oracle", "unknown": unknown}
        extra_val = {"target": diffusion_prep["target"]["valid"], "lists": diffusion_prep["lists"]["valid"],
                     "injection": "additive", "mode": "oracle", "unknown": unknown}
    if candidates is not None:
        from candidate_pooling import CandidateRowsDataset
        dataset_class = CandidateRowsDataset
        rand = {"random_seed": 42, "n_items": model.config.latent_candidate_pooling["n_items"]} \
            if candidate_source == "random" else {}
        extra_train = {"candidates": candidates["candidates"]["train"], "logp": candidates["logp"]["train"],
                       "n_candidates": n_candidates, "shuffle_seed": candidate_shuffle_seed, **rand}
        extra_val = {"candidates": candidates["candidates"]["valid"], "logp": candidates["logp"]["valid"],
                     "n_candidates": n_candidates, "shuffle_seed": candidate_shuffle_seed, **rand}
    train_data = dataset_class(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K, **extra_train)
    val_data = dataset_class(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample if eval_sample is None else eval_sample, category=category, K=K, **extra_val)

    if history_bias is not None and not latent_only:
        raise ValueError("history_bias needs latent_only")
    if latent_only:
        if use_side_info or use_category_alignment or category_text_file:
            raise ValueError("latent_only is a plain-R3 run; do not combine with other options")
        prefix_ids = [int(t) for t in train_data.instruction_tokens()]
        assert prefix_ids == [int(t) for t in val_data.instruction_tokens()]
        model.config.latent_bottleneck = {"visible_prefix_ids": prefix_ids}
        if latent_gist:
            if thought_source != "constant" or n_thoughts < 2:
                raise ValueError("latent_gist uses learned thought tokens (thought_source=constant), "
                                 "n_thoughts = latents + 1 readout >= 2")
            model.config.latent_bottleneck["gist"] = True
        if history_bias is not None:
            if latent_gist or float(history_bias) >= 0:
                raise ValueError("history_bias is a negative bias for the plain (non-gist) bottleneck")
            model.config.latent_bottleneck["history_bias"] = float(history_bias)
        print("latent bottleneck: visible prefix =", repr(tokenizer.decode(prefix_ids)))
    if latent_only_epochs is not None:
        if not latent_only or latent_gist or history_bias is not None or thought_source is not None:
            raise ValueError("latent_only_epochs is for the plain-R3 hard bottleneck (latent_only only)")
        if not 0 < int(latent_only_epochs) < num_epochs:
            raise ValueError("need 0 < latent_only_epochs < num_epochs")

    if pool_history_only:
        if (thought_source is not None or latent_only or use_side_info or use_category_alignment
                or category_text_file or steered_k > 0 or candidates is not None):
            raise ValueError("pool_history_only is a plain-R3 option; do not combine with other options")
        from history_pooling import build_spec, history_title_mask
        model.config.latent_pool_history = build_spec(tokenizer, train_data.category)
        probe = torch.tensor(train_data[0]["input_ids"])[None]
        shown = history_title_mask(probe, torch.ones_like(probe), model.config.latent_pool_history,
                                   model.model.embed_tokens.num_embeddings - 1)[0]
        print("history pooling: keys =", repr(tokenizer.decode(probe[0][shown])))

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
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if steered_k > 0:
        from steered_reading import SteeredCollator
        collator = SteeredCollator(collator)
    if candidates is not None:
        from candidate_pooling import CandidateCollator
        collator = CandidateCollator(collator)
    if category_prep is not None or diffusion_prep is not None:
        from category_latent import CategoryCollator
        collator = CategoryCollator(collator)
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
            load_best_model_at_end=latent_only_epochs is None,
            label_names=["labels"] if (use_category_alignment or use_side_info or steered_k > 0
                                       or candidates is not None or category_prep is not None
                                       or diffusion_prep is not None) else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=wandb_report_to(),
            remove_unused_columns=False
        ),
        data_collator=collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if latent_only_epochs is None else [],
    )
    if latent_only_epochs is not None:
        trainer.add_callback(BottleneckSchedule(model, int(latent_only_epochs)))
    if thought_source is not None:
        from continuous_thoughts import ThoughtDiagnostics
        # 64 validation rows spread over the file (adjacent rows share a user)
        step = max(1, len(val_data) // 64)
        probe_rows = [val_data[i] for i in range(0, len(val_data), step)][:64]
        trainer.add_callback(ThoughtDiagnostics(probe_rows, trainer.data_collator, output_dir))
    if candidates is not None and model.config.latent_candidate_pooling.get("anneal"):
        from candidate_pooling import AnnealCallback
        trainer.add_callback(AnnealCallback())
    if pool_diagnostics:
        from candidate_pooling import PoolingDiagnostics
        step = max(1, len(val_data) // 64)  # 64 validation rows spread over the file
        probe_rows = [val_data[i] for i in range(0, len(val_data), step)][:64]
        trainer.add_callback(PoolingDiagnostics(probe_rows, trainer.data_collator, output_dir))
    if diffusion_prep is not None:
        from diffusion_refine import AuxLogCallback
        trainer.add_callback(AuxLogCallback(model))
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
