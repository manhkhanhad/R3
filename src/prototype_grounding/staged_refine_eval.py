#!/usr/bin/env python3
"""Evaluate a staged-refinement R3 checkpoint (src/latent/staged_refine_attention.py).

1. Cluster-head probe (first --limit test examples): top-1/top-5 accuracy on the
   target's SASRec cluster vs history rules (last item's cluster, majority, prior),
   mean confidence, and confidence when right/wrong.
2. Teacher-forced sensitivity (first --limit test examples): KL of the target-title
   distribution vs the raw thought, as a ratio to dropping the last history item:
     swap_refinement  own r_item, donor's cluster prediction and confidence
     swap_r_item      donor's r_item, own cluster prediction
     swap_all         donor's whole thought
     oracle           own r_item, true cluster (one-hot, confidence 1)
   (stage-1 checkpoints: only swap_all, since the thought is r_item.)
3. Generation (first --gen_limit test examples, constrained 10-beam): HR/NDCG with
   the model's own prediction (raw) and with the true cluster (oracle-at-test).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from decoder_sensitivity import ci, compare, target_log_probs, DropLastHistory
from latent_grpo_dataset import get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash


def first_pass(model, ids, mask, pos):
    """Thought, and the parts it is built from (r_item, cluster logits, p, conf)."""
    att = model.attention
    with torch.no_grad():
        h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        thought = att(h, mask, pos)
        embeds = model.model.embed_tokens(ids).clone()
    last = att.last
    p, conf = (last["p"], last["conf"]) if "p" in last else att.refinement_inputs(last["logits"])
    return thought.float(), last["r_item"].float(), last["logits"].float(), p.float(), conf.float(), embeds


def build(att, r_item, p, conf):
    if int(att.stage) < 2:
        return r_item
    with torch.no_grad():
        return att.refine(r_item, p, conf)[0]


def probe(model, ds, device, codes, batch_size):
    col = lambda x: E.collate(x, ds.tokenizer.pad_token_id)
    top5, conf, norms_item, norms_comp = [], [], [], []
    for s in range(0, len(ds), batch_size):
        b = col(ds.rows[s:s + batch_size])
        ids, mask, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        thought, r_item, logits, _, _, _ = first_pass(model, ids, mask, pos)
        top5.append(logits.topk(5, -1).indices.cpu())
        conf.append(F.softmax(logits, -1).max(-1).values.cpu())
        norms_item.append(r_item.norm(dim=-1).cpu())
        norms_comp.append((thought - r_item).norm(dim=-1).cpu())
    top5, conf = torch.cat(top5), torch.cat(conf)
    prior = int(torch.bincount(codes[codes >= 0]).argmax())
    hit1, hit5, last, major, pri, c_ok, c_bad = [], [], [], [], [], [], []
    for i, row in enumerate(ds.frame.itertuples()):
        t = int(codes[int(row.item_id)])
        if t < 0:
            continue
        hist = [int(codes[h]) for h in ast.literal_eval(row.history_item_id) if codes[h] >= 0]
        ok = int(top5[i, 0]) == t
        hit1.append(ok)
        hit5.append(t in top5[i].tolist())
        last.append(bool(hist) and hist[-1] == t)
        major.append(bool(hist) and Counter(hist).most_common(1)[0][0] == t)
        pri.append(t == prior)
        (c_ok if ok else c_bad).append(float(conf[i]))
    m = lambda x: float(np.mean(x)) if x else None
    return {"n": len(hit1), "head_top1": m(hit1), "head_top5": m(hit5), "last_history_item": m(last),
            "history_majority": m(major), "most_frequent_cluster": m(pri), "mean_confidence": float(conf.mean()),
            "confidence_when_correct": m(c_ok), "confidence_when_wrong": m(c_bad),
            "norm_r_item": float(torch.cat(norms_item).mean()), "norm_thought_minus_r_item": float(torch.cat(norms_comp).mean())}


def sensitivity(model, tokenizer, device, codes, limit, batch_size):
    att = model.attention
    ds = E.PromptDataset("test", tokenizer, limit)
    ds_d = DropLastHistory("test", tokenizer, limit)
    n = len(ds)
    users = [r["user_id"] for r in ds.rows]
    donor = []
    for i in range(n):
        j = (i + n // 2) % n
        while users[j] == users[i]:
            j = (j + 1) % n
        donor.append(j)
    col = lambda rows: E.collate(rows, tokenizer.pad_token_id)
    R, P, C = [], [], []
    for s in range(0, n, batch_size):
        b = col(ds.rows[s:s + batch_size])
        ids, mask, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        _, r_item, _, p, conf, _ = first_pass(model, ids, mask, pos)
        R.append(r_item.cpu()); P.append(p.cpu()); C.append(conf.cpu())
    R, P, C = torch.cat(R), torch.cat(P), torch.cat(C)
    targets = torch.tensor([int(codes[int(i)]) for i in ds.frame.item_id])
    stage2 = int(att.stage) >= 2
    res = {}
    for s in range(0, n, batch_size):
        b, bd = col(ds.rows[s:s + batch_size]), col(ds_d.rows[s:s + batch_size])
        ids, mask, labels, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        thought, r_item, _, p, conf, embeds = first_pass(model, ids, mask, pos)
        d = torch.tensor(donor[s:s + len(ids)])
        rd, pd, cd = R[d].to(device), P[d].to(device), C[d].to(device)
        raw = target_log_probs(model, E.substitute(embeds, pos, thought), mask, labels)
        variants = {"swap_all": build(att, rd, pd, cd)}
        if stage2:
            t = targets[s:s + len(ids)].to(device)
            ok = t >= 0
            p_or = torch.where(ok[:, None], F.one_hot(t.clamp_min(0), att.num_clusters).float(), p)
            c_or = torch.where(ok, torch.ones_like(conf), conf)
            variants.update(swap_refinement=build(att, r_item, pd, cd), swap_r_item=build(att, rd, p, conf),
                            oracle=build(att, r_item, p_or, c_or))
        for name, vec in variants.items():
            lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
            for key, v in compare(raw, lp, labels, labels).items():
                res.setdefault(name, {}).setdefault(key, []).append(v)
        ids_d, mask_d, labels_d, pos_d = [bd[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        thought_d, _, _, _, _, embeds_d = first_pass(model, ids_d, mask_d, pos_d)
        lp = target_log_probs(model, E.substitute(embeds_d, pos_d, thought_d), mask_d, labels_d)
        for key, v in compare(raw, lp, labels, labels_d).items():
            res.setdefault("drop_last_recomputed", {}).setdefault(key, []).append(v)
    out = {v: {k: ci(np.concatenate(x)) for k, x in dct.items()} for v, dct in res.items()}
    ref = out["drop_last_recomputed"]["kl"]["mean"]
    return {"n": n, "output_shift": out, "ratio_to_drop_last": {v: dct["kl"]["mean"] / ref for v, dct in out.items()}}


def generate_ranks(model, tokenizer, device, codes, limit, batch_size, oracle, beams=10):
    att = model.attention
    ds = E.PromptDataset("test", tokenizer, limit)
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    cfg = GenerationConfig(num_beams=beams, num_return_sequences=beams, length_penalty=1.0,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    pad = tokenizer.pad_token_id
    model.config.use_cache = True
    ranks = []
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
        att.force_oracle = oracle
        att.oracle_codes = torch.tensor([int(codes[int(i)]) for i in frame.item_id], device=device) if oracle else None
        try:
            with torch.no_grad():
                out = model.generate(ids, attention_mask=masks, generation_config=cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        finally:
            att.force_oracle, att.oracle_codes = False, None
        dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
        for i, row in enumerate(rows):
            cands = dec[i * beams:(i + 1) * beams]
            t = row["target_title"].strip(' \n"')
            ranks.append(cands.index(t) + 1 if t in cands else 10 ** 9)
        if s % (batch_size * 50) == 0:
            print(f"generate (oracle={oracle}): {s + len(rows)}/{len(ds)}", flush=True)
    model.config.use_cache = False
    return E.recommendation_metrics(np.array(ranks))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Musical_Instruments")
    p.add_argument("--limit", type=int, default=2000, help="examples for probe/sensitivity")
    p.add_argument("--gen_limit", type=int, default=2000, help="examples for raw vs oracle generation")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--skip", default="", help="comma list of: probe,sensitivity,generation")
    a = p.parse_args()
    E.configure_category(a.category)
    E.seed_all(42)
    skip = set(a.skip.split(","))
    model, tokenizer, device = E.load_model(a.checkpoint)
    att = model.attention
    if not hasattr(att, "cluster_codes") or int((att.cluster_codes >= 0).sum()) == 0:
        raise ValueError("not a staged-refinement checkpoint, or cluster_codes did not load")
    codes = att.cluster_codes.cpu()
    out_dir = E.ANALYSIS / "results/staged_refine" / a.tag
    summary = {"checkpoint": str(a.checkpoint), "mode": att.mode, "stage": int(att.stage),
               "gain": float(att.gain) if getattr(att, "gain", None) is not None else None}
    print(summary, flush=True)
    if "probe" not in skip:
        summary["probe"] = probe(model, E.PromptDataset("test", tokenizer, a.limit), device, codes, a.batch_size)
        print(summary["probe"], flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    if "sensitivity" not in skip:
        summary["sensitivity"] = sensitivity(model, tokenizer, device, codes, a.limit, a.batch_size)
        print(summary["sensitivity"]["ratio_to_drop_last"], flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    if "generation" not in skip:
        summary["generation"] = {"n": a.gen_limit,
                                 "raw": generate_ranks(model, tokenizer, device, codes, a.gen_limit, a.batch_size, False)}
        if int(att.stage) >= 2:
            summary["generation"]["oracle"] = generate_ranks(model, tokenizer, device, codes, a.gen_limit, a.batch_size, True)
        print(summary["generation"], flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
