"""Evaluate an R3 category-prediction model and compare its category accuracy
with plain R3 (item model) and history heuristics.

Generation: constrained 10-beam search over the distinct category paths, so each
test example gets a ranked list of 10 categories.

Comparison (all on test rows whose target has a valid category):
* category-R3: its ranked categories;
* plain R3: categories of its top-10 recommended items, deduplicated in rank order
  (the category the item model implicitly predicts);
* last history item's category; history categories by frequency (ties: most recent);
  history-by-frequency then filled with the train-set category prior; train prior alone.

Metrics: top-1/3/5/10 exact-path accuracy, MRR@10, and top-1 accuracy at category depth 3.
"""

import ast
import json
import os
import sys
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fire
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

from category_dataset import CategoryRDataset, load_category_paths, write_category_info_file
from layers import LatentModel
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data

KS = (1, 3, 5, 10)


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x is not None and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def score(ranked, targets):
    res = {f"top{k}": float(np.mean([t in r[:k] for r, t in zip(ranked, targets)])) for k in KS}
    res["mrr@10"] = float(np.mean([1.0 / (r[:10].index(t) + 1) if t in r[:10] else 0.0 for r, t in zip(ranked, targets)]))
    depth = lambda p: " > ".join(p.split(" > ")[:3])
    res["top1_depth3"] = float(np.mean([bool(r) and depth(r[0]) == depth(t) for r, t in zip(ranked, targets)]))
    res["mean_distinct_in_top10"] = float(np.mean([len(r[:10]) for r in ranked]))
    return res


def title_to_item(info_file):
    out = {}
    for line in open(info_file):
        t, i = line.rstrip("\n").rsplit("\t", 1)
        out.setdefault(t.strip(' \n"'), int(i))
    return out


def generate(base_model, test_file, category_file, cat_info_file, batch_size):
    device = "cuda"
    model = LatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    hash_dict = get_prefix_data(cat_info_file, tokenizer)
    allowed = lambda b, ids: hash_dict.get(get_hash(ids), [])
    ds = CategoryRDataset(train_file=test_file, tokenizer=tokenizer, max_len=2560, sample=-1,
                          category="music instruments", test=True, K=4, category_file=category_file)
    enc = [ds[i] for i in range(len(ds))]
    cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0, max_new_tokens=128,
                           pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    preds = []
    for s in range(0, len(enc), batch_size):
        batch = enc[s:s + batch_size]
        L = max(len(e["input_ids"]) for e in batch)
        ids = torch.tensor([[tokenizer.pad_token_id] * (L - len(e["input_ids"])) + e["input_ids"] for e in batch], device=device)
        mask = torch.tensor([[0] * (L - len(e["input_ids"])) + e["attention_mask"] for e in batch], device=device)
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=allowed,
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=mask, generation_config=cfg, return_dict_in_generate=True,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        dec = tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)
        dec = [x.strip(' \n"') for x in dec]
        preds += [dec[i * 10:(i + 1) * 10] for i in range(len(batch))]
        if s % (batch_size * 50) == 0:
            print(f"generate {s + len(batch)}/{len(enc)}", flush=True)
    return ds.data, preds


def main(
    base_model: str,
    out_dir: str,
    test_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/test/Musical_Instruments_5_2015-1-2018-11.csv",
    train_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/train/Musical_Instruments_5_2015-1-2018-11.csv",
    category_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt",
    item_info_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/info/Musical_Instruments_5_2015-1-2018-11.txt",
    r3_predictions: str = "/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/latent/final_result.json",
    batch_size: int = 8,
    limit: int = -1,
):
    os.makedirs(out_dir, exist_ok=True)
    paths = load_category_paths(category_file)
    cat_info = os.path.join(out_dir, "category_catalog.txt")
    catalog = set(write_category_info_file(paths, cat_info))

    pred_file = os.path.join(out_dir, "category_predictions.json")
    if os.path.exists(pred_file):
        saved = json.load(open(pred_file))
        rows, preds = pd.DataFrame(saved["rows"]), saved["predictions"]
    else:
        test = test_file
        if limit > 0:
            test = os.path.join(out_dir, "test_subset.csv")
            pd.read_csv(test_file).iloc[:limit].to_csv(test, index=False)
        data, preds = generate(base_model, test, category_file, cat_info, batch_size)
        rows = data[["user_id", "item_id", "history_item_id"]].copy()
        rows["history_item_id"] = rows["history_item_id"].astype(str)
        json.dump({"rows": rows.to_dict("records"), "predictions": preds}, open(pred_file, "w"))

    targets = [paths[int(i)] for i in rows["item_id"]]
    catalog_rate = float(np.mean([p in catalog for pr in preds for p in pr]))

    # baselines from the history and the train-set prior
    train = pd.read_csv(train_file, usecols=["item_id"])
    prior = [c for c, _ in Counter(paths[int(i)] for i in train.item_id if int(i) in paths).most_common(10)]
    last, freq, freq_prior = [], [], []
    for h in rows["history_item_id"]:
        hist = [paths[int(i)] for i in ast.literal_eval(h) if int(i) in paths]
        last.append(hist[-1:])
        counts = Counter(hist)
        recency = {c: j for j, c in enumerate(hist)}
        by_freq = sorted(counts, key=lambda c: (-counts[c], -recency[c]))
        freq.append(by_freq)
        freq_prior.append(dedup(by_freq + prior))
    results = {
        "n": len(targets), "catalog_match_rate": catalog_rate,
        "category_r3": score([dedup(p) for p in preds], targets),
        "last_history_item": score(last, targets),
        "history_by_frequency": score(freq, targets),
        "history_then_prior": score(freq_prior, targets),
        "train_prior": score([prior] * len(targets), targets),
    }

    # plain R3: categories of its recommended items (same test-row order; rows filtered to valid targets)
    if r3_predictions and os.path.exists(r3_predictions):
        r3 = json.load(open(r3_predictions))
        full = pd.read_csv(test_file, usecols=["item_id"])
        keep = [k for k, i in enumerate(full.item_id) if int(i) in paths]
        if limit > 0:
            keep = [k for k in keep if k < limit]
        t2i = title_to_item(item_info_file)
        r3_ranked = [dedup([paths.get(t2i.get(x.strip(' \n"'))) for x in r3[k]["predict"]]) for k in keep]
        assert len(r3_ranked) == len(targets)
        results["plain_r3_implicit"] = score(r3_ranked, targets)
        results["plain_r3_source"] = r3_predictions
    json.dump(results, open(os.path.join(out_dir, "category_accuracy.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
