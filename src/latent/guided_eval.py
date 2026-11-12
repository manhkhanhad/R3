"""Evaluate an item-R3 model guided by a precomputed latent (guided_latent.py).

Constrained 10-beam item generation on the test split, as latent_attention_eval.py,
under two conditions:
* true:    each example gets its own guide vector;
* swapped: each example gets the guide of a donor example (different user and,
           when available, different target category) -- tests whether the guide is used.

Reports HR/NDCG@{1,5,10} and the share of the top-10 in the example's own vs the
donor's target category (steering).
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fire
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

from category_dataset import load_category_paths
from guided_latent import GuidedLatentModel, load_guide
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data
from reasoning_dataset import LatentRDataset


def metrics(ranks, ks=(1, 5, 10)):
    ranks = np.asarray(ranks)
    out = {}
    for k in ks:
        hit = ranks <= k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return out


def main(
    checkpoint: str,
    guide_test_file: str,
    out_dir: str,
    test_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/test/Musical_Instruments_5_2015-1-2018-11.csv",
    info_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/info/Musical_Instruments_5_2015-1-2018-11.txt",
    category_file: str = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt",
    batch_size: int = 8,
    limit: int = -1,
    conditions: str = "true,swapped",
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    model = GuidedLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    guide = load_guide(guide_test_file)
    model.set_guide_table(guide)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = True
    hash_dict = get_prefix_data(info_file, tokenizer)
    allowed = lambda b, ids: hash_dict.get(get_hash(ids), [])

    frame = pd.read_csv(test_file)
    ds = LatentRDataset(train_file=test_file, tokenizer=tokenizer, max_len=2560, category="music instruments", test=True, K=4, sample=-1)
    assert len(ds) == len(frame) == len(guide), "guide table must be row-aligned with the test CSV"
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    titles = [str(t).strip(' \n"') for t in frame.item_title]
    paths = load_category_paths(category_file)
    t2cat = {}
    for line in open(info_file):
        t, i = line.rstrip("\n").rsplit("\t", 1)
        t2cat.setdefault(t.strip(' \n"'), paths.get(int(i)))
    target_cat = [paths.get(int(i)) for i in frame.item_id]
    users = frame.user_id.tolist()
    donor = []
    for i in range(n):
        j = (i + n // 2) % n
        for _ in range(n):
            if users[j] != users[i] and target_cat[j] != target_cat[i]:
                break
            j = (j + 1) % n
        donor.append(j)

    cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0, max_new_tokens=128,
                           pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    results = {"checkpoint": checkpoint, "guide_test_file": guide_test_file, "n": n}
    for cond in conditions.split(","):
        index = list(range(n)) if cond == "true" else donor
        preds = []
        for s in range(0, n, batch_size):
            batch = [ds[i] for i in range(s, min(s + batch_size, n))]
            L = max(len(e["input_ids"]) for e in batch)
            ids = torch.tensor([[tokenizer.pad_token_id] * (L - len(e["input_ids"])) + e["input_ids"] for e in batch], device=device)
            mask = torch.tensor([[0] * (L - len(e["input_ids"])) + e["attention_mask"] for e in batch], device=device)
            model.set_guide_index(index[s:s + len(batch)])
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=allowed,
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=mask, generation_config=cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
            dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
            preds += [dec[k * 10:(k + 1) * 10] for k in range(len(batch))]
            if s % (batch_size * 100) == 0:
                print(f"{cond}: {s + len(batch)}/{n}", flush=True)
        model.set_guide_index(None)
        ranks = [p.index(titles[i]) + 1 if titles[i] in p else 10 ** 9 for i, p in enumerate(preds)]
        res = metrics(ranks)
        valid = [i for i in range(n) if target_cat[i] is not None]
        res["top10_in_own_category"] = float(np.mean([np.mean([t2cat.get(x) == target_cat[i] for x in preds[i]]) for i in valid]))
        res["top10_in_donor_category"] = float(np.mean([np.mean([t2cat.get(x) == target_cat[donor[i]] for x in preds[i]])
                                                         for i in valid if target_cat[donor[i]] is not None]))
        res["catalog_match_rate"] = float(np.mean([x in t2cat for p in preds for x in p]))
        results[cond] = res
        json.dump({"predictions": preds, "ranks": ranks}, open(os.path.join(out_dir, f"predictions_{cond}.json"), "w"))
        json.dump(results, open(os.path.join(out_dir, "guided_results.json"), "w"), indent=2)
        print(cond, json.dumps(res), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
