import fire
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
import torch
import json
from transformers import GenerationConfig,  AutoTokenizer
from layers import LatentModel
from reasoning_dataset import  LatentRDataset
from transformers import  LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data
from experiment_tracking import init_evaluation_run
from utils.ranking_metrics import calculate_ranking_metrics

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# torch.backends.cuda.enable_cudnn_sdp(False)

def main(
    base_model: str = "path/to/your/model",
    info_file: str = "data/info/Toys_and_Games_5_2017-1-2018-11.txt",
    category: str = "Toys_and_Games",
    test_data_path: str = "data/test/Toys_and_Games_5_2017-1-2018-11.csv",
    result_json_data: str = "temp.json",
    batch_size: int = 8,
    sample: int = 200,
    end_k: int = -1,
    seed: int = 0,
    temperature: float=1.0,
    guidance_scale: float=1.0,
    length_penalty: float=1.0,
    wandb_run_name: str = None,
    # continuous-thought interventions (continuous_thoughts.py): raw | mask | mean | shuffle
    thought_mode: str = "raw",
    thought_seed: int = 42,
    # B1 steered reading (steered_reading.py): learned | recent | all (selection intervention)
    selection_mode: str = "learned",
    item_embeddings_file: str = None,  # optional: oracle-agreement statistics
    # SASRec-informed pooling (candidate_pooling.py): raw (trained source and
    # schedule) | shuffle (another user's SASRec candidates) | drop (no candidate
    # branch: plain R3 pooling) | real (SASRec candidates, for random-source models)
    # | guided (branch scale 1, for annealed models)
    candidate_mode: str = "raw",
    # category latents (category_latent.py).  additive: oracle | shuffle | pred1 |
    # mix<K> (one beam search per predicted category, merged by category score +
    # item score); inject: oracle | pred | shuffle
    category_mode: str = None,
    # diffusion-refined thought (diffusion_refine.py): raw (sampled x0_hat) | oracle
    # (target item's vector) | shuffle (another user's x0_hat) | mean (mean x0_hat)
    diffusion_eval_mode: str = "raw",
    # recursive refinement (recursive_refine.py): number of cycles at test time (default: as trained)
    recursive_cycles: int = None,
):
    dataset_name = category
    wandb_run = init_evaluation_run(
        dataset=dataset_name,
        run_name=wandb_run_name,
        config={
            "base_model": base_model,
            "test_data_path": test_data_path,
            "dataset": dataset_name,
        },
    )
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "STEAM": "games", "Movies": "movie",
                     "yelp": "resturant" }
    category = category_dict[category]

    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(base_model)
    candidate_spec = getattr(base_config, "latent_candidate_pooling", None)
    side_spec = getattr(base_config, "latent_side_info", None)
    side_spec = side_spec if side_spec and side_spec.get("table") == "category" else None
    inject_spec = getattr(base_config, "latent_category_inject", None)
    model_class = LatentModel
    if side_spec:
        from side_info_latent import SideInfoLatentModel
        model_class = SideInfoLatentModel
    elif inject_spec:
        from category_latent import CategoryInjectModel
        model_class = CategoryInjectModel
    elif category_mode is not None:
        raise ValueError("category_mode needs a category-latent checkpoint")
    diffusion_spec = getattr(base_config, "latent_diffusion_refine", None)
    if diffusion_spec:
        from diffusion_refine import DiffusionRefineModel
        model_class = DiffusionRefineModel
        if diffusion_spec.get("mode") == "recursive":
            from recursive_refine import RecursiveRefineModel
            model_class = RecursiveRefineModel
    elif diffusion_eval_mode != "raw":
        raise ValueError("diffusion_eval_mode needs a diffusion-refine checkpoint")
    if diffusion_eval_mode not in ("raw", "oracle", "shuffle", "mean"):
        raise ValueError(f"unknown diffusion_eval_mode {diffusion_eval_mode}")
    if candidate_spec:
        from candidate_pooling import CandidatePoolingModel
        model_class = CandidatePoolingModel
    elif candidate_mode != "raw":
        raise ValueError("candidate_mode needs a candidate-pooling checkpoint")
    if candidate_mode not in ("raw", "shuffle", "drop", "real", "guided"):
        raise ValueError(f"unknown candidate_mode {candidate_mode}")
    model = model_class.from_pretrained(base_model, torch_dtype=torch.bfloat16, use_flash_attention_2=False)
    model.attention.end_k = end_k

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if thought_mode == "delete":  # gist bottleneck: hide the latent tokens from the target side
        if not (getattr(model.config, "latent_bottleneck", None) or {}).get("gist"):
            raise ValueError("thought_mode=delete needs a gist-bottleneck checkpoint")
        model.config.latent_bottleneck["delete_latents"] = True
        thought_mode = "raw"
    thoughts_spec = getattr(model.config, "latent_continuous_thoughts", None)
    n_thoughts = int(thoughts_spec["k"]) if thoughts_spec else 1
    if thought_mode != "raw" and not thoughts_spec:
        raise ValueError("thought_mode needs a continuous-thought checkpoint")
    hash_dict = get_prefix_data(info_file, tokenizer, n_thoughts=n_thoughts)
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    steer = getattr(model.config, "latent_steered_reading", None)
    dataset_class, extra = LatentRDataset, {}
    if steer:
        from steered_reading import SteeredReadingDataset, pad_index
        dataset_class = SteeredReadingDataset
        if item_embeddings_file:
            extra["item_embeddings"] = torch.load(item_embeddings_file, map_location="cpu", weights_only=False)["embeddings"].float()
        model.steer.rule_override = selection_mode
    elif selection_mode != "learned":
        raise ValueError("selection_mode needs a steered-reading checkpoint")
    category_prep = None
    mixture = 0
    if side_spec or inject_spec:
        from category_latent import CategoryRowsDataset, load_fp32_params
        category_prep = torch.load((side_spec or inject_spec)["source"], map_location="cpu", weights_only=False)
        mode = category_mode or "oracle"
        if side_spec:
            model.side_projection.float()
            model.side_gain.data = model.side_gain.data.float()
            load_fp32_params(model, base_model, "side_")
            table = torch.cat([category_prep["embeddings"], torch.zeros(1, category_prep["embeddings"].shape[1])])
            valid = torch.ones(len(table), dtype=torch.bool); valid[-1] = False
            test_target = category_prep["target"]["test"].clone()
            if mode.startswith(("nn:", "noise:")):
                test_target, table, valid = perturb_oracle(mode, test_target, category_prep["embeddings"].float(),
                                                           table, valid, seed)
                mode = "oracle"
            model.set_side_table(table, valid)
            if mode == "pred1":
                mode = "pred:0"
            elif mode.startswith("mix"):
                mixture, mode = int(mode[3:]), "pred:0"
        else:
            test_target = category_prep["target"]["test"]
        dataset_class = CategoryRowsDataset
        extra = {"target": test_target, "lists": category_prep["lists"]["test"],
                 "injection": "additive" if side_spec else "inject", "mode": mode,
                 "unknown": len(category_prep["catalog"])}
    if candidate_spec:
        from candidate_pooling import CandidateRowsDataset
        dataset_class = CandidateRowsDataset
        cand = torch.load(candidate_spec["candidates_file"], map_location="cpu", weights_only=False)
        # the test CSV here is split.py's copy of the test split (same rows, same order)
        extra = {"candidates": cand["candidates"]["test"], "logp": cand["logp"]["test"],
                 "n_candidates": int(candidate_spec["k"]),
                 "shuffle_seed": thought_seed if candidate_mode == "shuffle" else None}
        if candidate_spec.get("source") == "random" and candidate_mode == "raw":
            extra.update(random_seed=42, n_items=int(candidate_spec["n_items"]))
        model.cand_disabled = candidate_mode == "drop"
        model.eval_cand_scale = 1.0 if candidate_mode == "guided" else 0.0
    val_dataset=dataset_class(train_file=test_data_path, tokenizer=tokenizer,max_len=2560, category=category, test=True,K=4, seed=seed,
                              sample=sample, n_thoughts=n_thoughts, **extra)
        
    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()

    def evaluate(
            encodings,
            cf_logits,
            temperature=1.0,
            num_beams=10,
            max_new_tokens=128,
            guidance_scale=1.0,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        minLen = min([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": [], "attention_mask": []}

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            padding_encodings["attention_mask"].append([0] * (maxLen - L) + _["attention_mask"])

        
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            # top_p=0,
            # top_k=10,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        with torch.no_grad():
            ccc = CFEnhancedLogitsProcessor(
                guidance_scale=guidance_scale,
                cf_logits=cf_logits,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                cf_dict=None,
                unconditional_ids=None,
                model=model,
                tokenizer=tokenizer,
                num_beams=num_beams,
                n_thoughts=n_thoughts,
            )
            logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), ccc])

            generation_output = model.generate(
                torch.tensor([_ for _ in padding_encodings["input_ids"]]).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
                attention_mask=torch.tensor([_ for _ in padding_encodings["attention_mask"]]).to(device),
            )
    
        s = generation_output.sequences[:, minLen - (4 + n_thoughts):]
        # sequence_scores = [[0 for i in range(len(generation_output.scores))] for _ in range(num_beams)]
        # for i in range(num_beams):
        #     for j in range(L, len(generation_output.sequences[i])):
        #         beam_index = generation_output.beam_indices[i][j - L]
        #         if beam_index != -1:
        #             sequence_scores[i][j - L] = generation_output.scores[j - L][beam_index][generation_output.sequences[i][j]].item()
        scores = generation_output.sequences_scores.tolist()
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split("Response:")[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        real_scores = [scores[i * num_beams: (i + 1) * num_beams] for i in range(len(scores) // num_beams)]
        return real_outputs, real_scores, None
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
    scores = []
    seq_scores = []
    selections = []
    cand_mass = []
    replacement = None
    if thought_mode != "raw":
        replacement = thought_replacements(model, tokenizer, new_encodings, val_dataset.data, thought_mode,
                                           thought_seed, result_json_data)
    x0_bank = None
    if recursive_cycles is not None:
        if not (diffusion_spec and diffusion_spec.get("mode") == "recursive"):
            raise ValueError("recursive_cycles needs a recursive-refine checkpoint")
        model.eval_cycles = int(recursive_cycles)
    if diffusion_spec:
        x0_bank = diffusion_x0_bank(model, tokenizer, new_encodings, val_dataset.data, diffusion_spec,
                                    diffusion_eval_mode, thought_seed, result_json_data)
    grounded = getattr(model.config, "latent_grounded_reasoning", None)
    if grounded:
        from grounded_reasoning import row_metadata
        if not bool(model.attention.bank_ready):
            raise RuntimeError("grounded reasoning checkpoint has no inference bank")
        meta = [row_metadata(val_dataset.data.iloc[i]) for i in range(len(val_dataset.data))]
    intent = getattr(model.config, "latent_intent_reasoning", None)
    intent_mode = os.environ.get("INTENT_MODE", "map")
    intent_predictions = []
    if intent:
        targets = [int(val_dataset.data.iloc[i]["item_id"]) for i in range(len(val_dataset.data))]
    for idx, encodings in enumerate(tqdm(new_encodings)):
        if grounded:
            rows = meta[idx * batch_size: idx * batch_size + len(encodings)]
            model.attention.query_time = torch.tensor([m["query_time"] for m in rows], device=device)
            model.attention.query_user = torch.tensor([m["query_user"] for m in rows], device=device)
        if intent and intent_mode == "oracle":
            batch_targets = targets[idx * batch_size: idx * batch_size + len(encodings)]
            model.attention.target_item = torch.tensor(batch_targets, device=device)
        if intent and intent_mode == "mixture":
            # One beam search per intent; merge by log p(k|x) + sequence score.
            merged = [dict() for _ in encodings]
            per_intent = [dict() for _ in encodings]
            for k in range(model.attention.n_intents):
                model.attention.forced_intent = k
                out_k, score_k, _ = evaluate(encodings, cf_logits=None, temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
                prior = model.attention.last_prior_logp.view(len(encodings), -1, model.attention.n_intents)[:, 0, k]
                for row, (items, item_scores) in enumerate(zip(out_k, score_k)):
                    per_intent[row][k] = items
                    for item, s in zip(items, item_scores):
                        value = prior[row].item() + s
                        if value > merged[row].get(item, float("-inf")):
                            merged[row][item] = value
            model.attention.forced_intent = None
            output = [sorted(m, key=m.get, reverse=True)[:10] for m in merged]
            score = [[m[i] for i in o] for m, o in zip(merged, output)]
            seq_score = None
            intent_predictions.extend(per_intent)
        else:
            if replacement is not None:
                rows = replacement[idx * batch_size: idx * batch_size + len(encodings)].to(device)
                model.thoughts.override = lambda x, rows=rows: rows.repeat_interleave(x.size(0) // rows.size(0), 0).to(x)
            if candidate_spec:
                model.cand_override = (torch.tensor([e["cand_items"] for e in encodings]),
                                       torch.tensor([e["cand_logp"] for e in encodings]))
            if inject_spec:
                model.cat_override = torch.tensor([e["cat_list"] for e in encodings])
            if x0_bank is not None:
                model.x0_override = x0_bank[idx * batch_size: idx * batch_size + len(encodings)].to(device)
            if side_spec and not mixture:
                model.set_side_index(torch.tensor([e["side_index"] for e in encodings]))
            if steer:
                width = max(len(e["input_ids"]) for e in encodings)
                model.steer.item_index = pad_index([e["item_index"] for e in encodings], width, "left")
                model.steer.item_sim = (torch.tensor([e["item_sim"] for e in encodings], dtype=torch.float32)
                                        if item_embeddings_file else None)
            if side_spec and mixture:
                # one beam search per predicted category; merge by category score + item score
                start = idx * batch_size
                rows_lists = category_prep["lists"]["test"][val_dataset.data.index[start:start + len(encodings)]]
                rows_scores = category_prep["scores"]["test"][val_dataset.data.index[start:start + len(encodings)]]
                merged = [dict() for _ in encodings]
                for j in range(mixture):
                    model.set_side_index(rows_lists[:, j])
                    out_j, score_j, _ = evaluate(encodings, cf_logits=None, temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
                    for r, (items, item_scores) in enumerate(zip(out_j, score_j)):
                        for it, s in zip(items, item_scores):
                            v = float(rows_scores[r, j]) + s
                            if v > merged[r].get(it, float("-inf")):
                                merged[r][it] = v
                output = [sorted(m, key=m.get, reverse=True)[:10] for m in merged]
                score = [[m[i] for i in o] for m, o in zip(merged, output)]
                seq_score = None
            else:
                output, score, seq_score = evaluate(encodings, cf_logits=None, temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
            if inject_spec and model.last_cat_weights is not None:
                w = model.last_cat_weights[:: model.last_cat_weights.size(0) // len(encodings)]
                cand_mass.extend(w.float().cpu().tolist())
            if steer:
                chosen = model.steer.last_selection[:: model.steer.last_selection.size(0) // len(encodings)]
                selections.extend([torch.nonzero(row).squeeze(1).tolist() for row in chosen.cpu()])
                model.steer.item_index = model.steer.item_sim = None
            if replacement is not None:
                model.thoughts.override = None
        if intent:
            model.attention.target_item = None
        if candidate_spec and model.last_cand_weights is not None:
            w = model.last_cand_weights[:: model.last_cand_weights.size(0) // len(encodings)]
            cand_mass.extend(w.float().cpu().tolist())
        if idx == 0:
            print(output)
            print(score)
        outputs = outputs + output
        scores = scores + score
        seq_scores.append(seq_score)
    
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        test["predict_score"] = scores[i]
        if selections:
            test["selected_items"] = selections[i]
        if cand_mass:
            test["candidate_weights"] = cand_mass[i]
        if intent_predictions:
            test["intent_predict"] = {str(k): v for k, v in intent_predictions[i].items()}
        # test["predict_seq_score"] = seq_scores[i]

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

    if wandb_run is not None:
        metrics = calculate_ranking_metrics(test_data, info_file)
        wandb_run.log({f"evaluation_shard/{key}": value for key, value in metrics.items()})
        wandb_run.finish()

def diffusion_x0_bank(model, tokenizer, batches, frame, spec, mode, seed, result_path):
    """Refined vectors x0_hat of every test row (prompt only, as in generation).

    Writes the nearest-item retrieval readout of the raw x0_hat (HR/NDCG of the
    target among the catalogue items, centred cosine; also split by whether the
    target occurs as a training target) to diffusion_x0.json next to the
    results.  Returns the [N, d] override for ``mode`` (None for raw).
    """
    from diffusion_refine import nearest_item_metrics

    prep = torch.load(spec["source"], map_location="cpu", weights_only=False)
    model.set_item_table(prep["embeddings"], prep["valid"])
    rows = torch.as_tensor(frame.index.to_numpy())
    target = prep["target"]["test"][rows].long()
    parts, cycle_parts = [], []
    for encodings in batches:
        width = max(len(e["input_ids"]) for e in encodings)
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(e["input_ids"])) + e["input_ids"] for e in encodings])
        mask = torch.tensor([[0] * (width - len(e["input_ids"])) + e["attention_mask"] for e in encodings])
        with torch.no_grad():
            model.generate_embs(ids.to(device), mask.to(device))
        parts.append(model.last_x0.float().cpu())
        if getattr(model, "last_cycles", None) is not None:
            cycle_parts.append(model.last_cycles.float().cpu())
    bank = torch.cat(parts)
    item_x0, item_valid = model.item_x0.cpu(), model.item_valid.cpu()
    seen = prep["seen_target"]["test"][rows].bool()
    readout = {"all": nearest_item_metrics(bank, target, item_x0, item_valid), "n": len(bank),
               "seen": nearest_item_metrics(bank[seen], target[seen], item_x0, item_valid),
               "unseen": nearest_item_metrics(bank[~seen], target[~seen], item_x0, item_valid),
               "n_unseen": int((~seen).sum())}
    if cycle_parts:
        # recursive refinement: nearest-item readout of the answer after every cycle
        cycles = torch.cat(cycle_parts, dim=1)
        readout["per_cycle"] = [nearest_item_metrics(c, target, item_x0, item_valid) for c in cycles]
        readout["per_cycle_step_change"] = [0.0] + [
            (cycles[i] - cycles[i - 1]).norm(dim=-1).mean().item() for i in range(1, len(cycles))]
    print("diffusion x0 retrieval:", json.dumps(readout))
    out_dir = os.path.dirname(os.path.abspath(result_path))
    with open(os.path.join(out_dir, "diffusion_x0.json"), "w") as f:
        json.dump(readout, f, indent=2)
    if mode == "raw":
        return None
    if mode == "oracle":
        return item_x0[target].clone()
    if mode == "mean":
        return bank.mean(0, keepdim=True).expand_as(bank).clone()
    users = frame["user_id"].astype(str).to_numpy()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(bank), generator=generator)
    for i in range(len(bank)):
        while users[int(perm[i])] == users[i]:
            perm[i] = int(torch.randint(len(bank), (1,), generator=generator))
    return bank[perm].clone()


def perturb_oracle(mode, target, embeddings, table, valid, seed):
    """Perturbed target-item oracles (item_oracle_prep.py tables).

    ``nn:<k>``: the target's k-th nearest other item (cosine in the table's space).
    ``noise:<s>``: variance-preserving noise on the item-specific (mean-centred) part, norm kept:
    v' = mu + ||c|| * normalize(sqrt(1 - s^2) c/||c|| + s u), c = v - mu, u a random unit
    vector (centred cos ~ sqrt(1 - s^2)); one noised vector per test row, appended to the table.
    Rows without a vector (target == n) keep the zero row.
    """
    import torch.nn.functional as F

    n = embeddings.shape[0]
    ok = target < n
    target = target.clone()
    if mode.startswith("nn:"):
        k = int(mode[3:])
        unit = F.normalize(embeddings, dim=-1)
        rows = torch.nonzero(ok).squeeze(-1)
        for s in range(0, len(rows), 1024):
            r = rows[s:s + 1024]
            sim = unit[target[r]] @ unit.T
            sim[torch.arange(len(r)), target[r]] = -2.0  # exclude the target itself
            sim[:, ~valid[:n]] = -2.0
            target[r] = sim.topk(k, dim=-1).indices[:, -1]
        return target, table, valid
    s = float(mode[len("noise:"):])
    g = torch.Generator().manual_seed(seed)
    mu = embeddings[valid[:n]].mean(0)  # noise acts on the item-specific part (tables share a large mean)
    v = embeddings[target[ok]] - mu
    norm = v.norm(dim=-1, keepdim=True)
    u = F.normalize(torch.randn(v.shape, generator=g), dim=-1)
    noisy = mu + F.normalize((1 - s * s) ** 0.5 * v / norm + s * u, dim=-1) * norm
    print(f"noise {s}: mean centred cos(v', v) {F.cosine_similarity(noisy - mu, v, dim=-1).mean():.4f}, "
          f"raw cos {F.cosine_similarity(noisy, v + mu, dim=-1).mean():.4f}")
    start = table.shape[0]
    table = torch.cat([table, noisy])
    valid = torch.cat([valid, torch.ones(len(noisy), dtype=torch.bool)])
    target[ok] = start + torch.arange(len(noisy))
    return target, table, valid


def thought_replacements(model, tokenizer, batches, frame, mode, seed, result_path):
    """Thoughts each test row receives under an intervention ([N, K, H], fp32).

    The raw thoughts of every row are computed once (prompt only, left-padded as in
    generation) and saved as thought_bank.pt next to the results (shared by modes); ``mean`` gives every row the mean
    thought block, ``shuffle`` the block of a random row from a different user,
    ``mask`` zeros.
    """
    bank_path = os.path.join(os.path.dirname(os.path.abspath(result_path)), "thought_bank.pt")
    if os.path.exists(bank_path):
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    else:
        parts = []
        for encodings in batches:
            width = max(len(e["input_ids"]) for e in encodings)
            ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(e["input_ids"])) + e["input_ids"] for e in encodings])
            mask = torch.tensor([[0] * (width - len(e["input_ids"])) + e["attention_mask"] for e in encodings])
            with torch.no_grad():
                model.generate_embs(ids.to(device), mask.to(device))
            parts.append(model.thoughts.last_thoughts.float().cpu())
        bank = torch.cat(parts)
        torch.save(bank, bank_path)
    if mode == "mask":
        return torch.zeros_like(bank)
    if mode == "mean":
        return bank.mean(0, keepdim=True).expand_as(bank).clone()
    if mode == "shuffle":
        users = frame["user_id"].astype(str).to_numpy()
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(bank), generator=generator)
        for i in range(len(bank)):
            while users[int(perm[i])] == users[i]:
                perm[i] = int(torch.randint(len(bank), (1,), generator=generator))
        return bank[perm].clone()
    raise ValueError(f"unknown thought_mode {mode}")


if __name__ == '__main__':
    fire.Fire(main)



