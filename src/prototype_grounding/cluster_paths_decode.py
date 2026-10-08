#!/usr/bin/env python3
"""Parallel cluster-hypothesis paths for latent marginalization (idea 2).

Treats the target's CF cluster c as a latent variable:

    p(item | x) = sum_c p(c | x) * p(item | x, c)

* p(item | x, c): the ``oracle_cfcluster150`` checkpoint (trained with the target's
  cluster centroid in the thought slot), decoded with hypothesis cluster c in the slot.
  Its training assumed the slot is correct, which is what conditioning on c means.
* p(c | x): a cluster predictor (build_candidates.py fused predictor, cluster_seq_predictor.py).
* plus a plain R3 v1 path (no cluster), which the aggregation may mix in.

This script only decodes: for each test row it runs constrained 10-beam search on the
R3 path and on every cluster in the union of the predictors' top-K, and saves each
path's titles and beam scores.  cluster_paths_aggregate.py combines them.

Usage: python cluster_paths_decode.py --rows 3000 --topk 5 --predictors fused,grufull
"""

from __future__ import annotations

import argparse
import json
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

TABLES = {
    "fused": "group_pred150_{tag}.pt",
    "grufull": "cluster_grufull_{tag}.pt",
    "mix": "cluster_mix_{tag}.pt",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_model", type=Path, required=True, help="oracle_cfcluster150 checkpoint")
    p.add_argument("--r3_model", type=Path, required=True, help="plain R3 v1 checkpoint")
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--predictors", default="fused,grufull")
    p.add_argument("--table_tag", default="full")
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--out", type=Path, default=D.RESULTS / "cluster_paths" / "full")
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    a.out.mkdir(parents=True, exist_ok=True)

    cluster_model, tokenizer, device, spec = load(a.cluster_model)
    if spec is None or spec["kind"] != "oracle_category":
        raise ValueError("cluster_model must be an oracle_category side-information checkpoint")
    cid = torch.load(spec["source"], map_location="cpu", weights_only=False)["cluster_id"].numpy()
    k = int(cid.max()) + 1
    rep = np.full(k, -1, np.int64)  # any member item: every member's side vector is the cluster centroid
    for it in range(len(cid)):
        if cid[it] >= 0 and rep[cid[it]] < 0:
            rep[cid[it]] = it
    r3_model, _, _ = E.load_model(a.r3_model)

    ds = VariantPromptDataset("test", tokenizer, a.rows)
    n = len(ds)
    table_dir = D.DATA / "info" / "collab_reasoning"
    probs = {name: torch.load(table_dir / TABLES[name].format(tag=a.table_tag), weights_only=False)["test"][:n].float().numpy()
             for name in a.predictors.split(",")}
    paths = [sorted({int(c) for pr in probs.values() for c in np.argsort(-pr[i])[:a.topk]}) for i in range(n)]

    pad = tokenizer.pad_token_id
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)

    def decode(model, rows, side_index=None):
        """Top-10 titles and length-normalized beam log-probabilities per row."""
        model.config.use_cache = True
        prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
        L = max(map(len, prompts))
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
        masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
        if side_index is not None:
            model.set_side_index(torch.as_tensor(side_index))
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                         prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=masks, generation_config=gen_cfg, return_dict_in_generate=True,
                                 output_scores=True,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        if side_index is not None:
            model.set_side_index(None)
        model.config.use_cache = False
        dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
        sc = out.sequences_scores.float().cpu().numpy()
        return [(dec[i * 10:(i + 1) * 10], sc[i * 10:(i + 1) * 10].tolist()) for i in range(len(rows))]

    # flatten (row, cluster) jobs so every generate call has a full batch
    records = [{"row": i, "target": ds.rows[i]["target_title"], "target_cluster": int(cid[ds.rows[i]["target_item"]]),
                "probs": {name: {str(c): float(pr[i, c]) for c in paths[i]} for name, pr in probs.items()},
                "paths": {}} for i in range(n)]
    B = a.gen_batch_size
    for s in range(0, n, B):
        rows = ds.rows[s:s + B]
        for i, res in zip(range(s, s + len(rows)), decode(r3_model, rows)):
            records[i]["paths"]["r3"] = {"titles": res[0], "scores": res[1]}
        if s % (B * 50) == 0:
            print(f"r3 path {s + len(rows)}/{n}", flush=True)
    jobs = [(i, c) for i in range(n) for c in paths[i]]
    print(f"{len(jobs)} cluster paths ({len(jobs) / n:.2f} per row)", flush=True)
    for s in range(0, len(jobs), B):
        chunk = jobs[s:s + B]
        res = decode(cluster_model, [ds.rows[i] for i, _ in chunk], side_index=[rep[c] for _, c in chunk])
        for (i, c), r in zip(chunk, res):
            records[i]["paths"][str(c)] = {"titles": r[0], "scores": r[1]}
        if s % (B * 100) == 0:
            print(f"cluster paths {s + len(chunk)}/{len(jobs)}", flush=True)
            with (a.out / "paths.json").open("w") as f:
                json.dump(records, f)
    with (a.out / "paths.json").open("w") as f:
        json.dump(records, f)
    print(f"wrote {a.out / 'paths.json'}", flush=True)


if __name__ == "__main__":
    main()
