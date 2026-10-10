#!/usr/bin/env python3
"""Evaluate a MUX-latent checkpoint (src/latent/mux_latent.py; document/mux_latent_experiments.md).

1. Category decoded from the thought vector: each of the 412 category paths is scored
   by sum_j a_j log softmax(W r)[token_j] (its own mux target); accuracy of the argmax
   path (top-1/top-5) on the full test split, overall and on R3 v1's hits/misses.
2. Does the decoder read r? Teacher-forced KL on the target title (first ``--limit``
   rows), as a ratio to dropping the last history item (r recomputed), for:
   swap_r (a donor's r), zero_r, mean_r, and two category injections that replace r by
   ||r|| * normalize(sum_j a_j W[token_j]) for the true / a donor's category.
3. Ranking by constrained 10-beam generation on the full test split vs R3 v1
   (gained/lost rows, exact McNemar).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from decoder_sensitivity import ci, compare, target_log_probs
from latent_grpo_dataset import get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from side_info_eval import CATEGORY_FILE, VariantPromptDataset

sys.path.insert(0, str(E.ROOT / "src" / "latent"))
sys.path.insert(0, str(E.ROOT / "src" / "collab_reasoning"))
import data as D  # noqa: E402
from mux_latent import MuxLatentModel, mux_table  # noqa: E402

TOPK = 5


def load(checkpoint):
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(checkpoint)
    if getattr(config, "latent_mux", None) is None:
        model, tokenizer, device = E.load_model(checkpoint)  # plain R3 (e.g. v1) as a reference
        return model, tokenizer, device
    model, info = MuxLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"missing keys when loading {checkpoint}: {missing}")
    model = model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--r3_predictions", required=True, help="R3 v1 test predictions (0.json, 1.json)")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--gen_limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    model, tokenizer, device = load(a.checkpoint)
    out_dir = D.RESULTS.parent / "mux_latent" / a.tag
    spec = getattr(model.config, "latent_mux", None)
    summary = {"checkpoint": a.checkpoint, "latent_mux": spec}

    full_ds = VariantPromptDataset("test", tokenizer, -1)
    n = len(full_ds)
    rows = D.load_rows("test")
    r3 = D.load_r3_predictions(a.r3_predictions, rows)
    r3_hit = np.array([t in pr for t, pr in zip(rows.target, r3)])

    cats = torch.load(CATEGORY_FILE, map_location="cpu", weights_only=True)
    rho = spec["rho"] if spec else 0.9
    item_ids, item_w = mux_table(cats["paths"], cats["valid"], tokenizer, rho=rho)
    path_of = {i: x for i, x in enumerate(cats["paths"]) if cats["valid"][i]}
    uniq = sorted(set(path_of.values()))
    rep = {path: i for i, path in sorted(path_of.items(), reverse=True)}  # an item carrying each path
    path_ids = item_ids[[rep[x] for x in uniq]].to(device)
    path_w = item_w[[rep[x] for x in uniq]].to(device)
    tgt_path = np.array([uniq.index(path_of[t]) if t in path_of else -1 for t in rows.target])
    W = model.lm_head.weight
    pad = tokenizer.pad_token_id
    collate = lambda x: E.collate(x, pad)

    def batches(ds, count):
        for s in range(0, count, a.batch_size):
            b = collate(ds.rows[s:min(s + a.batch_size, count)])
            yield s, [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]

    def log_probs(r):
        return F.log_softmax(model.lm_head(r.to(W.dtype)).float(), dim=-1)

    def category_vec(items, norm):
        """||r|| * normalize(sum_j a_j W[token_j]) for each item's category."""
        v = (item_w[items].to(device).unsqueeze(-1) * W[item_ids[items].to(device)].float()).sum(1)
        return F.normalize(v, dim=-1) * norm.unsqueeze(-1)

    # 1. category decoded from r
    r_all, ranks = [], []
    with torch.no_grad():
        for s, (ids, mask, labels, pos) in batches(full_ds, n):
            r, _ = E.thought_embeds(model, ids, mask, pos)
            r_all.append(r.cpu())
            lp = log_probs(r)
            scores = (path_w.unsqueeze(0) * lp[:, path_ids]).sum(-1)  # [B, P]
            ranks.append(torch.argsort(scores, dim=-1, descending=True)[:, :TOPK].cpu())
            if s % (a.batch_size * 500) == 0:
                print(f"[{a.tag}] category {s}/{n}", flush=True)
    r_all, ranks = torch.cat(r_all), torch.cat(ranks).numpy()
    ok = tgt_path >= 0
    top1 = (ranks[:, 0] == tgt_path) & ok
    top5 = (ranks == tgt_path[:, None]).any(1) & ok
    last_cat = np.array([uniq.index(path_of[h[-1]]) if h and h[-1] in path_of else -2 for h in rows.history])
    summary["category_from_r"] = {
        "paths": len(uniq), "n_valid": int(ok.sum()),
        "top1": float(top1[ok].mean()), "top5": float(top5[ok].mean()),
        "top1_r3_hit": float(top1[ok & r3_hit].mean()), "top1_r3_miss": float(top1[ok & ~r3_hit].mean()),
        "last_history_item_category_baseline": float((last_cat == tgt_path)[ok].mean()),
        "r_norm_mean": float(r_all.norm(dim=-1).mean()),
    }
    E.atomic_json(out_dir / "summary.json", summary)
    print(json.dumps(summary["category_from_r"]), flush=True)

    # 2. teacher-forced sensitivity
    lim = min(a.limit, n)
    donor = []
    for i in range(lim):
        j = (i + lim // 2) % lim
        for _ in range(lim):
            if rows.user[j] != rows.user[i] and tgt_path[j] != tgt_path[i]:
                break
            j = (j + 1) % lim
        donor.append(j)
    mean_r = r_all[:lim].mean(0).to(device)
    drop = VariantPromptDataset("test", tokenizer, lim, drop_last=True)
    it_drop = batches(drop, lim)
    targets = torch.as_tensor(rows.target)
    results = {}
    with torch.no_grad():
        for s, (ids, mask, labels, pos) in batches(full_ds, lim):
            sl = slice(s, s + len(ids))
            dn = torch.tensor(donor[sl])
            r = r_all[sl].to(device)
            _, embeds = E.thought_embeds(model, ids, mask, pos)
            raw = target_log_probs(model, E.substitute(embeds, pos, r), mask, labels)
            norm = r.norm(dim=-1)
            variants = {
                "swap_r": r_all[dn].to(device), "zero_r": torch.zeros_like(r), "mean_r": mean_r.expand_as(r),
                "inject_true_category": category_vec(targets[sl], norm),
                "inject_donor_category": category_vec(targets[dn], norm),
            }
            for name, vec in variants.items():
                lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
                for k, v in compare(raw, lp, labels, labels).items():
                    results.setdefault(name, {}).setdefault(k, []).append(v)
            _, (ids_d, mask_d, labels_d, pos_d) = next(it_drop)
            rd, emb_d = E.thought_embeds(model, ids_d, mask_d, pos_d)
            lp = target_log_probs(model, E.substitute(emb_d, pos_d, rd), mask_d, labels_d)
            for k, v in compare(raw, lp, labels, labels_d).items():
                results.setdefault("drop_last_recomputed", {}).setdefault(k, []).append(v)
    summary["output_shift"] = {k: {m: ci(np.concatenate(v)) for m, v in d.items()} for k, d in results.items()}
    ref = summary["output_shift"]["drop_last_recomputed"]["kl"]["mean"]
    summary["ratio_to_drop_last"] = {k: v["kl"]["mean"] / ref for k, v in summary["output_shift"].items()}
    E.atomic_json(out_dir / "summary.json", summary)
    print({k: round(v, 4) for k, v in summary["ratio_to_drop_last"].items()}, flush=True)

    # 3. ranking
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    g = n if a.gen_limit < 0 else min(a.gen_limit, n)
    model.config.use_cache = True
    preds = []
    for s in range(0, g, a.gen_batch_size):
        batch_rows = full_ds.rows[s:min(s + a.gen_batch_size, g)]
        prompts = [r["input_ids"][:r["prompt_len"]] for r in batch_rows]
        L = max(map(len, prompts))
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
        masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                         prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=masks, generation_config=gen_cfg, return_dict_in_generate=True,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
        preds += [dec[i * 10:(i + 1) * 10] for i in range(len(batch_rows))]
        if s % (a.gen_batch_size * 100) == 0:
            print(f"[{a.tag}] generate {s + len(batch_rows)}/{g}", flush=True)
    ranks_t = np.array([pr.index(r["target_title"].strip(' \n"')) + 1 if r["target_title"].strip(' \n"') in pr else 10 ** 9
                        for pr, r in zip(preds, full_ds.rows[:g])])
    hit, base = ranks_t <= 10, r3_hit[:g]
    gained, lost = int((hit & ~base).sum()), int((~hit & base).sum())
    summary["ranking"] = {"n": g, **E.recommendation_metrics(ranks_t), "r3_HR@10_same_rows": float(base.mean()),
                          "gained_vs_r3": gained, "lost_vs_r3": lost,
                          "mcnemar_p": float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0,
                          "HR@10_category_right": float(hit[top1[:g]].mean()) if top1[:g].any() else None,
                          "HR@10_category_wrong": float(hit[~top1[:g] & ok[:g]].mean())}
    E.atomic_json(out_dir / "predictions.json", [{"row": i, "top10": pr} for i, pr in enumerate(preds)])
    E.atomic_json(out_dir / "summary.json", summary)
    print(summary["ranking"], flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
