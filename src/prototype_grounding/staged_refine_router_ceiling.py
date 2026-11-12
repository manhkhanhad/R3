#!/usr/bin/env python3
"""Step 0 of the aspect router: how much could a router that gates the cluster aspect gain?

For each example the decoder is run (constrained 10-beam) under fixed refinement settings:

  none    c = 0                      -> thought = r_item (aspect switched off)
  pred    one-hot(argmax p), c = 1   -> the head's top-1 cluster at full strength
  oracle  one-hot(true cluster), c=1 -> reference ceiling

and per-row ranks are combined into routers:

  router_correct  pred where argmax p == true cluster, else none   (perfect "aspect correct" router)
  router_helps    better of pred / none per row                     (perfect "aspect helps" router)
  confidence      pred where max p >= tau, else none; tau tuned on validation, also the
                  best tau on test (optimistic) and the AUROC of max p for "aspect correct"

Row groups (opportunity / risk) count where the aspect can gain or lose against `none`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from latent_grpo_dataset import get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash

MISS = 10 ** 9
CONDITIONS = ("none", "pred", "oracle")


def transform_for(condition, num_clusters):
    if condition == "none":
        return lambda p, c: (p, torch.zeros_like(c))
    if condition == "pred":
        return lambda p, c: (F.one_hot(p.argmax(-1), num_clusters).float(), torch.ones_like(c))
    return None  # oracle uses force_oracle


def decode(model, tokenizer, device, codes, split, limit, batch_size, condition, beams=10):
    """Per-row rank of the target in the 10-beam list, plus the head's max p and top-1 cluster."""
    att = model.attention
    ds = E.PromptDataset(split, tokenizer, limit)
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    cfg = GenerationConfig(num_beams=beams, num_return_sequences=beams, length_penalty=1.0,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    pad = tokenizer.pad_token_id
    model.config.use_cache = True
    ranks, conf, top1 = [], [], []
    for s in range(0, len(ds), batch_size):
        rows = ds.rows[s:s + batch_size]
        frame = ds.frame.iloc[s:s + batch_size]
        prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
        L = max(map(len, prompts))
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
        masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                         prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=beams)
        att.transform = transform_for(condition, att.num_clusters)
        att.force_oracle = condition == "oracle"
        att.oracle_codes = torch.tensor([int(codes[int(i)]) for i in frame.item_id], device=device) if condition == "oracle" else None
        try:
            with torch.no_grad():
                out = model.generate(ids, attention_mask=masks, generation_config=cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        finally:
            att.transform, att.force_oracle, att.oracle_codes = None, False, None
        # the prompt pass ran on the beam-expanded batch; one row per example
        pr = F.softmax(att.last["logits"].float()[::beams], -1).cpu()
        conf.append(pr.max(-1).values)
        top1.append(pr.argmax(-1))
        dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
        for i, row in enumerate(rows):
            cands = dec[i * beams:(i + 1) * beams]
            t = row["target_title"].strip(' \n"')
            ranks.append(cands.index(t) + 1 if t in cands else MISS)
        if s % (batch_size * 100) == 0:
            print(f"{split}/{condition}: {s + len(rows)}/{len(ds)}", flush=True)
    model.config.use_cache = False
    target = np.array([int(codes[int(i)]) for i in ds.frame.item_id])
    return {"rank": np.array(ranks), "conf": torch.cat(conf).numpy(), "top1": torch.cat(top1).numpy(), "target": target}


def metrics(ranks):
    return E.recommendation_metrics(np.asarray(ranks))


def auroc(score, label):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    pos = label.sum(); neg = len(label) - pos
    return float((r[label].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))


def gate(none, pred, use):
    return np.where(use, pred, none)


def analyse(d):
    none, pred, oracle = d["none"]["rank"], d["pred"]["rank"], d["oracle"]["rank"]
    conf, correct = d["none"]["conf"], d["none"]["top1"] == d["none"]["target"]
    hit = lambda r: r <= 10
    out = {"n": int(len(none)), "head_top1_accuracy": float(correct.mean()),
           "conf_auroc_for_correct": auroc(conf, correct),
           "none": metrics(none), "pred": metrics(pred), "oracle": metrics(oracle),
           "router_correct": metrics(gate(none, pred, correct)),
           "router_helps": metrics(np.minimum(none, pred)),
           "groups": {
               "opportunity_correct_and_none_miss": int((correct & ~hit(none)).sum()),
               "risk_wrong_and_none_hit": int((~correct & hit(none)).sum()),
               "pred_gains_over_none": int((hit(pred) & ~hit(none)).sum()),
               "pred_loses_vs_none": int((~hit(pred) & hit(none)).sum()),
               "none_hit_rate_when_correct": float(hit(none)[correct].mean()) if correct.any() else None,
               "none_hit_rate_when_wrong": float(hit(none)[~correct].mean()),
               "pred_hit_rate_when_correct": float(hit(pred)[correct].mean()) if correct.any() else None,
               "pred_hit_rate_when_wrong": float(hit(pred)[~correct].mean())}}
    return out


def confidence_curve(d, taus):
    none, pred, conf = d["none"]["rank"], d["pred"]["rank"], d["none"]["conf"]
    return {f"{t:.2f}": metrics(gate(none, pred, conf >= t)) for t in taus}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Musical_Instruments")
    p.add_argument("--valid_limit", type=int, default=2000)
    p.add_argument("--test_limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category(a.category)
    E.seed_all(42)
    model, tokenizer, device = E.load_model(a.checkpoint)
    att = model.attention
    if not hasattr(att, "transform") or int(att.stage) < 2:
        raise ValueError("needs a stage-2 staged-refinement checkpoint")
    codes = att.cluster_codes.cpu()
    out_dir = E.ANALYSIS / "results/staged_refine" / a.tag
    res = {"checkpoint": str(a.checkpoint), "mode": att.mode}
    data = {}
    for split, limit in (("valid", a.valid_limit), ("test", a.test_limit)):
        data[split] = {c: decode(model, tokenizer, device, codes, split, limit, a.batch_size, c) for c in CONDITIONS}
        res[split] = analyse(data[split])
        print(split, json.dumps(res[split], indent=1), flush=True)
    taus = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    vcurve, tcurve = confidence_curve(data["valid"], taus), confidence_curve(data["test"], taus)
    best_v = max(vcurve, key=lambda t: vcurve[t]["NDCG@10"])
    best_t = max(tcurve, key=lambda t: tcurve[t]["NDCG@10"])
    res["confidence_gate"] = {"tau_from_valid": best_v, "test_at_valid_tau": tcurve[best_v],
                              "tau_best_on_test_optimistic": best_t, "test_at_best_tau": tcurve[best_t],
                              "test_curve": tcurve}
    E.atomic_json(out_dir / "router_ceiling.json", res)  # creates out_dir
    for split in ("valid", "test"):
        np.savez(out_dir / f"router_ceiling_{split}.npz",
                 **{f"{c}_{k}": v for c, dd in data[split].items() for k, v in dd.items()})
    print(json.dumps({k: res["confidence_gate"][k] for k in ("tau_from_valid", "test_at_valid_tau", "tau_best_on_test_optimistic", "test_at_best_tau")}, indent=1), flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
