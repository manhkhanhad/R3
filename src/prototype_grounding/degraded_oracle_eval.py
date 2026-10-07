#!/usr/bin/env python3
"""Degraded-oracle dose-response for the CF-cluster oracle (idea 5's accuracy threshold).

The ``oracle_cfcluster150`` checkpoint (Experiment 2 of decoder_insensitivity_to_latent.md)
was trained with the target's true CF cluster (k = 150) in the thought slot.  At test time
this script gives it the true cluster with probability ``p`` and the cluster of a uniformly
random catalog item otherwise, and records HR@10 / NDCG@10 as a function of ``p``.  The
``p`` at which HR@10 reaches plain R3 (0.1226) is a rough accuracy a target-group predictor
must reach to help.  It is an approximation: the model was trained on a perfect oracle and
trusts the slot fully, while a model trained with a noisy predictor would learn to discount it.

``--plugin`` also feeds the group predictor's top-1 cluster (build_candidates.py) as a
realistic, non-leaky input to the same model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from latent_grpo_dataset import get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from side_info_eval import VariantPromptDataset, load

sys.path.insert(0, str(E.ROOT / "src" / "collab_reasoning"))
import data as D  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ps", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    p.add_argument("--plugin", type=Path, default=None, help="group_pred150_*.pt from build_candidates.py")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--gen_batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    model, tokenizer, device, spec = load(a.checkpoint)
    if spec is None or spec["kind"] != "oracle_category":
        raise ValueError("expects an oracle_category side-information checkpoint")
    clusters = torch.load(spec["source"], map_location="cpu", weights_only=False)
    cid = clusters["cluster_id"].numpy()
    ds = VariantPromptDataset("test", tokenizer, a.limit)
    n = len(ds)
    targets = np.array([r["target_item"] for r in ds.rows])
    valid_items = np.nonzero(cid >= 0)[0]
    pad = tokenizer.pad_token_id
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)

    def run(index):
        model.config.use_cache = True
        ranks = []
        for s in range(0, n, a.gen_batch_size):
            rows = ds.rows[s:s + a.gen_batch_size]
            prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
            L = max(map(len, prompts))
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
            masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
            model.set_side_index(torch.as_tensor(index[s:s + len(rows)]))
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                             prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=masks, generation_config=gen_cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
            dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
            for i, r in enumerate(rows):
                pr = dec[i * 10:(i + 1) * 10]
                t = r["target_title"].strip(' \n"')
                ranks.append(pr.index(t) + 1 if t in pr else 10 ** 9)
        model.set_side_index(None)
        model.config.use_cache = False
        return E.recommendation_metrics(np.array(ranks))

    out_path = D.RESULTS / "degraded_oracle" / a.tag / "summary.json"
    summary = {"checkpoint": str(a.checkpoint), "n": n, "doses": {}}
    rng = np.random.default_rng(0)
    for prob in [float(x) for x in a.ps.split(",")]:
        keep = rng.random(n) < prob
        index = np.where(keep, targets, rng.choice(valid_items, size=n))
        acc = float((cid[index] == cid[targets])[cid[targets] >= 0].mean())
        res = run(index)
        summary["doses"][str(prob)] = {"cluster_accuracy": acc, **res}
        print(prob, summary["doses"][str(prob)], flush=True)
        E.atomic_json(out_path, summary)
    if a.plugin is not None:
        rep = torch.load(a.plugin, map_location="cpu", weights_only=False)["test_top1_rep_item"].numpy()[:n]
        acc = float((cid[rep] == cid[targets])[cid[targets] >= 0].mean())
        summary["plugin_top1"] = {"cluster_accuracy": acc, **run(rep)}
        print("plugin", summary["plugin_top1"], flush=True)
        E.atomic_json(out_path, summary)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
