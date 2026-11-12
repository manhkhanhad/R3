#!/usr/bin/env python3
"""HR@10 of an oracle side-information checkpoint, split by whether the test
target item ever appears as a *training target*.

Tests whether an item-level oracle (e.g. the target's SASRec embedding) is
decoded by a general embedding -> title mapping or by memorizing the mapping for
items seen as training targets. Uses the same prompts, side lookup and
constrained 10-beam generation as side_info_eval.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from side_info_eval import VariantPromptDataset, load
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from latent_grpo_dataset import get_prefix_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seen_sample", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    model, tokenizer, device, spec = load(a.checkpoint)
    assert spec and spec["kind"] == "oracle_category", "expects an item-indexed oracle checkpoint"

    train_targets = set(pd.read_csv(E.SPLIT_FILES["train"]).item_id)
    ds = VariantPromptDataset("test", tokenizer, -1)
    unseen = [i for i, r in enumerate(ds.rows) if r["target_item"] not in train_targets]
    seen_all = [i for i, r in enumerate(ds.rows) if r["target_item"] in train_targets]
    seen = list(np.random.default_rng(0).choice(seen_all, size=min(a.seen_sample, len(seen_all)), replace=False))

    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    pad = tokenizer.pad_token_id
    cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                           pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    model.config.use_cache = True

    def hits(indices):
        out, examples = [], []
        for s in range(0, len(indices), a.batch_size):
            rows = [ds.rows[i] for i in indices[s:s + a.batch_size]]
            prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
            L = max(map(len, prompts))
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
            mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
            model.set_side_index(torch.tensor([r["target_item"] for r in rows]))
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                             prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
            with torch.no_grad():
                gen = model.generate(ids, attention_mask=mask, generation_config=cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
            dec = [x.strip(' \n"') for x in tokenizer.batch_decode(gen.sequences[:, L:], skip_special_tokens=True)]
            for j, r in enumerate(rows):
                pred = dec[j * 10:(j + 1) * 10]
                tgt = r["target_title"].strip(' \n"')
                out.append(tgt in pred)
                if len(examples) < 5:
                    examples.append({"target": tgt, "top3": pred[:3]})
        return float(np.mean(out)), len(out), examples

    res = {"checkpoint": str(a.checkpoint), "n_test": len(ds), "n_unseen_as_train_target": len(unseen)}
    for name, idx in (("unseen_as_train_target", unseen), ("seen_as_train_target_sample", seen)):
        hr, n, ex = hits(idx)
        res[name] = {"HR@10": hr, "n": n, "examples": ex}
        print(name, hr, n, flush=True)
    out = E.ANALYSIS / "results/side_info" / a.tag / "seen_unseen.json"
    E.atomic_json(out, res)
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:2000])


if __name__ == "__main__":
    main()
