"""Constrained-beam ranking evaluation for rationale runs (document/latent_distillation.md).

Same generation as ``latent_attention_eval.py`` (10 beams over catalog titles), with
the rationale either pooled into ``r`` only (``mode=latent``), written into the
decoder prompt (``mode=text``), or absent (``mode=none``: plain R3).  For the
latent mode, ``rationale_override`` swaps the rationale for a control:
``shuffle`` (another row's rationale) or ``empty`` (no analysis text).
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import numpy as np
import torch
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

from layers import LatentModel
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data
from rationale_latent import RationaleLatentModel, RationaleLatentRDataset
from utils.ranking_metrics import calculate_ranking_metrics

CATEGORIES = {"Musical_Instruments": "music instruments", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games"}


def left_pad(rows, value):
    length = max(len(r) for r in rows)
    return [[value] * (length - len(r)) + list(r) for r in rows]


def main(base_model: str, info_file: str, test_data_path: str, result_json_data: str,
         rationale_file: str = None, mode: str = "latent", rationale_override: str = None,
         category: str = "Musical_Instruments", batch_size: int = 8, num_beams: int = 10,
         limit: int = -1, seed: int = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = RationaleLatentModel if mode == "latent" else LatentModel
    model = model_class.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    rationales = None
    if mode != "none":
        with open(rationale_file) as handle:
            rationales = json.load(handle)["rationales"]
        if rationale_override == "shuffle":
            perm = np.random.default_rng(seed).permutation(len(rationales))
            rationales = [rationales[j] for j in perm]
        elif rationale_override == "empty":
            rationales = [""] * len(rationales)
        elif rationale_override is not None:
            raise ValueError("rationale_override must be shuffle or empty")
    dataset = RationaleLatentRDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560,
                                      category=CATEGORIES[category], test=True, sample=-1,
                                      rationales=rationales, mode=mode)
    encodings = [dataset[i] for i in range(len(dataset))]
    test_data = dataset.get_all()
    if limit > 0:
        encodings, test_data = encodings[:limit], test_data[:limit]

    hash_dict = get_prefix_data(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

    outputs, scores = [], []
    for start in range(0, len(encodings), batch_size):
        batch = encodings[start:start + batch_size]
        ids = torch.tensor(left_pad([b["input_ids"] for b in batch], tokenizer.pad_token_id), device=device)
        mask = torch.tensor(left_pad([b["attention_mask"] for b in batch], 0), device=device)
        if mode == "latent":
            pool = left_pad([b["pool_input_ids"] for b in batch], tokenizer.pad_token_id)
            pool_mask = left_pad([[1] * len(b["pool_input_ids"]) for b in batch], 0)
            model.set_pool(torch.tensor(pool, device=device), torch.tensor(pool_mask, device=device))
        config = GenerationConfig(num_beams=num_beams, num_return_sequences=num_beams, max_new_tokens=128,
                                  pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id)
        processor = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                              prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, cf_dict=None,
                                              unconditional_ids=None, model=model, tokenizer=tokenizer,
                                              num_beams=num_beams)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=mask, generation_config=config, return_dict_in_generate=True,
                                 output_scores=True,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(1.0), processor]))
        text = tokenizer.batch_decode(out.sequences[:, ids.shape[1]:], skip_special_tokens=True)
        seq_scores = out.sequences_scores.tolist()
        for i in range(len(batch)):
            outputs.append(text[i * num_beams:(i + 1) * num_beams])
            scores.append(seq_scores[i * num_beams:(i + 1) * num_beams])
        if start == 0:
            print(outputs[0], scores[0], flush=True)
        if start // batch_size % 50 == 0:
            print(f"{start + len(batch)}/{len(encodings)}", flush=True)

    for row, predict, score in zip(test_data, outputs, scores):
        row["predict"], row["predict_score"] = predict, score
        row.pop("dedup", None)
    os.makedirs(os.path.dirname(os.path.abspath(result_json_data)), exist_ok=True)
    with open(result_json_data, "w") as handle:
        json.dump(test_data, handle, indent=1)
    metrics = calculate_ranking_metrics(test_data, info_file)
    print(json.dumps({k: metrics[k] for k in ("hr@10", "ndcg@10", "hr@5", "ndcg@5") if k in metrics}))


if __name__ == "__main__":
    fire.Fire(main)
