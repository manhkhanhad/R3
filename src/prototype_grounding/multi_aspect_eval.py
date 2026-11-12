#!/usr/bin/env python3
"""Evaluate a multi-aspect R3 checkpoint (src/latent/multi_aspect_attention.py).

1. Aspect probes: top-1 accuracy of each aspect head on the target's code,
   vs history-only heuristics (code of the last history item, majority code).
2. Gates: mean gate / confidence per aspect, and whether gates follow correctness.
3. Teacher-forced sensitivity (first --limit test examples), KL vs raw on the
   target title, as a ratio to the one-item history drop:
   swap_all (donor's fused r) and swap_<aspect> (donor's r_k for one aspect,
   own gates).
4. Beam re-scoring: generate --beams candidates (constrained beam search),
   then rerank by  seq_score + lam * sum_k g_k log p_k(code_k(item)).
   lam is chosen on --valid_limit validation examples and applied to test.
   lam = 0 is the decoder alone (top-10 of the beam list).
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

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def aspect_forward(model, ids, mask, pos):
    """Run the first pass; return (fused r, per-aspect latents, gates, logits, input embeds)."""
    with torch.no_grad():
        h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        fused = model.attention(h, mask, pos)
        att = model.attention
        embeds = model.model.embed_tokens(ids).clone()
    return fused.float(), att.last_latents.float(), att.last_gate.float(), [l.float() for l in att.last_logits], embeds


def probes(model, ds, device, codes, names, batch_size):
    frame = ds.frame
    preds, gates, confs = [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda x: E.collate(x, ds.tokenizer.pad_token_id))
    for b in loader:
        ids, mask, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        _, _, g, logits, _ = aspect_forward(model, ids, mask, pos)
        preds.append(torch.stack([l.argmax(-1) for l in logits], 1).cpu())
        gates.append(g.cpu())
        confs.append(model.attention.last_confidence.float().cpu())
    preds, gates, confs = torch.cat(preds), torch.cat(gates), torch.cat(confs)
    out = {}
    for k, name in enumerate(names):
        acc, last, major, g_ok, g_bad = [], [], [], [], []
        for i, row in enumerate(frame.itertuples()):
            t = int(codes[k, int(row.item_id)])
            if t < 0:
                continue
            hist = [int(codes[k, h]) for h in ast.literal_eval(row.history_item_id) if codes[k, h] >= 0]
            ok = int(preds[i, k]) == t
            acc.append(ok)
            last.append(bool(hist) and hist[-1] == t)
            major.append(bool(hist) and Counter(hist).most_common(1)[0][0] == t)
            (g_ok if ok else g_bad).append(float(gates[i, k]))
        out[name] = {"head_top1": float(np.mean(acc)), "last_history_item": float(np.mean(last)),
                     "history_majority": float(np.mean(major)), "n": len(acc),
                     "mean_gate": float(gates[:, k].mean()), "mean_confidence": float(confs[:, k].mean()),
                     "gate_when_head_correct": float(np.mean(g_ok)) if g_ok else None,
                     "gate_when_head_wrong": float(np.mean(g_bad)) if g_bad else None}
    return out


def sensitivity(model, tokenizer, device, names, limit, batch_size):
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
    # per-aspect latents and gates for all rows (donor lookups)
    lat_all = []
    for s in range(0, n, batch_size):
        b = col(ds.rows[s:s + batch_size])
        ids, mask, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        lat_all.append(aspect_forward(model, ids, mask, pos)[1].cpu())
    lat_all = torch.cat(lat_all)
    res = {}
    for s in range(0, n, batch_size):
        b, bd = col(ds.rows[s:s + batch_size]), col(ds_d.rows[s:s + batch_size])
        ids, mask, labels, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        fused, lat, g, _, embeds = aspect_forward(model, ids, mask, pos)
        raw = target_log_probs(model, E.substitute(embeds, pos, fused), mask, labels)
        dlat = lat_all[torch.tensor(donor[s:s + len(ids)])].to(device)
        variants = {"swap_all": (g.unsqueeze(-1) * dlat).sum(1)}
        for k, name in enumerate(names):
            mixed = lat.clone()
            mixed[:, k] = dlat[:, k]
            variants[f"swap_{name}"] = (g.unsqueeze(-1) * mixed).sum(1)
        for vname, vec in variants.items():
            lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
            for key, v in compare(raw, lp, labels, labels).items():
                res.setdefault(vname, {}).setdefault(key, []).append(v)
        ids_d, mask_d, labels_d, pos_d = [bd[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        fused_d, _, _, _, embeds_d = aspect_forward(model, ids_d, mask_d, pos_d)
        lp = target_log_probs(model, E.substitute(embeds_d, pos_d, fused_d), mask_d, labels_d)
        for key, v in compare(raw, lp, labels, labels_d).items():
            res.setdefault("drop_last_recomputed_r", {}).setdefault(key, []).append(v)
    out = {v: {k: ci(np.concatenate(x)) for k, x in d.items()} for v, d in res.items()}
    ref = out["drop_last_recomputed_r"]["kl"]["mean"]
    return {"n": n, "output_shift": out, "ratio_to_drop_last": {v: d["kl"]["mean"] / ref for v, d in out.items()}}


def generate_candidates(model, tokenizer, device, split, limit, beams, batch_size, codes):
    ds = E.PromptDataset(split, tokenizer, limit)
    title_to_id = {}
    for line in E.INFO_FILE.open():
        t, i = line.rstrip("\n").rsplit("\t", 1)
        title_to_id.setdefault(t.strip(' \n"'), int(i))
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    cfg = GenerationConfig(num_beams=beams, num_return_sequences=beams, length_penalty=1.0,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    pad = tokenizer.pad_token_id
    model.config.use_cache = True
    recs = []
    for s in range(0, len(ds), batch_size):
        rows = ds.rows[s:s + batch_size]
        prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
        L = max(map(len, prompts))
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
        masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                         prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=beams)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=masks, generation_config=cfg, return_dict_in_generate=True,
                                 output_scores=True,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
        att = model.attention
        # the prompt pass ran on the beam-expanded batch; keep one row per example
        logp = [F.log_softmax(l.float(), -1)[::beams].cpu() for l in att.last_logits]
        gate = att.last_gate.float()[::beams].cpu()
        dec = tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)
        dec = [x.strip(' \n"') for x in dec]
        seq = out.sequences_scores.float().cpu().view(len(rows), beams)
        for i, row in enumerate(rows):
            cands = dec[i * beams:(i + 1) * beams]
            item_ids = [title_to_id.get(c, -1) for c in cands]
            asp = np.zeros((beams, len(logp)), dtype=np.float32)
            for j, it in enumerate(item_ids):
                for k in range(len(logp)):
                    code = int(codes[k, it]) if it >= 0 else -1
                    asp[j, k] = float(logp[k][i, code]) if code >= 0 else -30.0
            recs.append({"target": row["target_title"].strip(' \n"'), "cands": cands, "seq": seq[i].numpy(),
                         "aspect_logp": asp, "gate": gate[i].numpy()})
        if s % (batch_size * 50) == 0:
            print(f"generate {split}: {s + len(rows)}/{len(ds)}", flush=True)
    model.config.use_cache = False
    return recs


def rerank_metrics(recs, lam, mask=None):
    ranks = []
    for r in recs:
        g = r["gate"] if mask is None else r["gate"] * mask
        score = r["seq"] + lam * (r["aspect_logp"] * g[None]).sum(1)
        order = np.argsort(-score, kind="stable")[:10]
        top = [r["cands"][j] for j in order]
        ranks.append(top.index(r["target"]) + 1 if r["target"] in top else 10 ** 9)
    return E.recommendation_metrics(np.array(ranks))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Musical_Instruments")
    p.add_argument("--limit", type=int, default=2000, help="examples for probes/sensitivity")
    p.add_argument("--valid_limit", type=int, default=2000)
    p.add_argument("--test_limit", type=int, default=-1)
    p.add_argument("--beams", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument("--skip", default="", help="comma list of: probes,sensitivity,rerank")
    a = p.parse_args()
    E.configure_category(a.category)
    E.seed_all(42)
    skip = set(a.skip.split(","))
    model, tokenizer, device = E.load_model(a.checkpoint)
    att = model.attention
    if not hasattr(att, "aspect_codes"):
        raise ValueError("not a multi-aspect checkpoint")
    codes, names = att.aspect_codes.cpu(), att.names
    if int((codes >= 0).sum()) == 0:
        raise ValueError("aspect_codes buffer is empty: checkpoint did not load it")
    out_dir = E.ANALYSIS / "results/multi_aspect" / a.tag
    summary = {"checkpoint": str(a.checkpoint), "aspects": names, "fusion": att.fusion,
               "gate_scale": float(att.gate_scale()), "gate_bias": att.gate_bias.float().tolist()}

    if "probes" not in skip:
        summary["probes"] = probes(model, E.PromptDataset("test", tokenizer, a.limit), device, codes, names, a.batch_size)
        print(summary["probes"], flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    if "sensitivity" not in skip:
        summary["sensitivity"] = sensitivity(model, tokenizer, device, names, a.limit, a.batch_size)
        print(summary["sensitivity"]["ratio_to_drop_last"], flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    if "rerank" not in skip:
        valid = generate_candidates(model, tokenizer, device, "valid", a.valid_limit, a.beams, a.gen_batch_size, codes)
        grid = {str(l): rerank_metrics(valid, l) for l in LAMBDAS}
        best = max(LAMBDAS, key=lambda l: grid[str(l)]["NDCG@10"])
        test = generate_candidates(model, tokenizer, device, "test", a.test_limit, a.beams, a.gen_batch_size, codes)
        per_aspect = {}
        for k, name in enumerate(names):
            m = np.zeros(len(names), dtype=np.float32)
            m[k] = 1.0
            per_aspect[name] = rerank_metrics(test, best, m / max(float(np.mean([r["gate"][k] for r in test])), 1e-6))
        summary["rerank"] = {"beams": a.beams, "valid_n": len(valid), "valid_grid": grid, "chosen_lambda": best,
                             "test_n": len(test),
                             "test_decoder_only": rerank_metrics(test, 0.0),
                             "test_rescored": rerank_metrics(test, best),
                             "test_rescored_single_aspect": per_aspect,
                             "test_grid": {str(l): rerank_metrics(test, l) for l in LAMBDAS},
                             "candidate_recall@beams": float(np.mean([r["target"] in r["cands"] for r in test]))}
        print({k: summary["rerank"][k] for k in ("chosen_lambda", "test_decoder_only", "test_rescored", "candidate_recall@beams")}, flush=True)
        E.atomic_json(out_dir / "summary.json", summary)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
