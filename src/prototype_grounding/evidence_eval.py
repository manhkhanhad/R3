#!/usr/bin/env python3
"""Evaluate a trained evidence-verifier checkpoint (src/latent/verified_evidence_latent.py).

1. Verifier on the full test split: top-1/top-5 accuracy of the predicted evidence
   (target's CF cluster), confidence, and a selective table (accuracy and coverage
   above confidence thresholds), on all rows and on R3 v1's misses.  The number
   of R3 misses where the verifier is confident and right bounds the gain.
2. Does the decoder read the evidence?  Teacher-forced KL on the target title
   (first ``--limit`` rows) for: swap_evidence (a donor's evidence component),
   zero_evidence, oracle_evidence (the true cluster at full confidence), swap_r,
   and the one-item history drop as the reference scale.
3. Ranking by constrained 10-beam generation on the full test split (or
   ``--gen_limit`` rows), compared row by row with R3 v1 (gained/lost, exact
   McNemar), plus the zero-evidence ablation on the first ``--ablate_limit`` rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from decoder_sensitivity import ci, compare, target_log_probs
from latent_grpo_dataset import get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from side_info_eval import VariantPromptDataset

sys.path.insert(0, str(E.ROOT / "src" / "latent"))
sys.path.insert(0, str(E.ROOT / "src" / "collab_reasoning"))
import data as D  # noqa: E402
from verified_evidence_latent import VerifiedEvidenceLatentModel, evidence_component  # noqa: E402

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def load(checkpoint: Path):
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, info = VerifiedEvidenceLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]  # tied to embed_tokens
    if missing:
        raise RuntimeError(f"missing keys when loading {checkpoint}: {missing}")
    model = model.to(device).eval()
    model.config.use_cache = False
    model.set_evidence_table(torch.load(model.config.latent_evidence["source"], map_location="cpu", weights_only=False))
    return model, tokenizer, device


def selective(conf, right, miss):
    rows = []
    for th in THRESHOLDS:
        s = conf >= th
        sm = s & miss
        rows.append({"threshold": th, "coverage": float(s.mean()), "acc": float(right[s].mean()) if s.any() else None,
                     "coverage_miss": float(sm.sum() / max(miss.sum(), 1)),
                     "acc_miss": float(right[sm].mean()) if sm.any() else None,
                     "miss_confident_right": int((right & sm).sum())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--r3_predictions", type=Path, required=True, help="R3 v1 test predictions (0.json, 1.json)")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--gen_limit", type=int, default=-1)
    p.add_argument("--ablate_limit", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    model, tokenizer, device = load(a.checkpoint)
    out_dir = D.RESULTS / "verified_evidence" / a.tag
    summary = {"checkpoint": str(a.checkpoint), "evidence_spec": model.config.latent_evidence}

    full_ds = VariantPromptDataset("test", tokenizer, -1)
    n = len(full_ds)
    test_rows = D.load_rows("test")
    if len(test_rows) != n:
        raise ValueError("test split mismatch")
    r3 = D.load_r3_predictions(a.r3_predictions, test_rows)
    r3_hit = np.array([t in pr for t, pr in zip(test_rows.target, r3)])
    item_cluster = model.item_cluster.cpu().numpy()
    tc = item_cluster[test_rows.target]
    pad = tokenizer.pad_token_id
    collate = lambda x: E.collate(x, pad)
    evidence = model.evidence_vectors()
    gain = model.side_gain.float()

    def batches(ds, count):
        for s in range(0, count, a.batch_size):
            b = collate(ds.rows[s:min(s + a.batch_size, count)])
            yield s, [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]

    # 1. verifier on the full test split
    probs, r_base = [], []
    with torch.no_grad():
        for s, (ids, mask, labels, pos) in batches(full_ds, n):
            r, _ = E.thought_embeds(model, ids, mask, pos)
            r_base.append(r.cpu())
            probs.append(model.verifier_logits(r).softmax(-1).cpu())
            if s % (a.batch_size * 500) == 0:
                print(f"[{a.tag}] verifier {s}/{n}", flush=True)
    probs, r_base = torch.cat(probs), torch.cat(r_base)
    _, conf = evidence_component(probs, torch.zeros(probs.shape[1], 1), 1.0, model.evidence_gate)
    conf = conf.numpy()
    pn = probs.numpy()
    top5 = np.argsort(-pn, 1)[:, :5]
    valid = tc >= 0
    right = (pn.argmax(1) == tc) & valid
    miss = ~r3_hit
    summary["verifier"] = {
        "n": n, "r3_hr10": float(r3_hit.mean()),
        "top1": float(right[valid].mean()), "top5": float((top5 == tc[:, None]).any(1)[valid].mean()),
        "top1_r3_miss": float(right[valid & miss].mean()), "top1_r3_hit": float(right[valid & ~miss].mean()),
        "mean_conf": float(conf.mean()), "mean_conf_right": float(conf[right].mean()),
        "mean_conf_wrong": float(conf[valid & ~right].mean()), "mean_max_prob": float(pn.max(1).mean()),
        "selective_by_conf": selective(conf, right, miss),
        "selective_by_max_prob": selective(pn.max(1), right, miss),
    }
    E.atomic_json(out_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary["verifier"].items() if not k.startswith("selective")}), flush=True)

    # 2. teacher-forced: does the decoder read the evidence component?
    lim = min(a.limit, n)
    users = test_rows.user
    donor = []
    for i in range(lim):
        j = (i + lim // 2) % lim
        for _ in range(lim):
            if users[j] != users[i] and tc[j] != tc[i]:
                break
            j = (j + 1) % lim
        donor.append(j)
    drop = VariantPromptDataset("test", tokenizer, lim, drop_last=True)
    it_drop = batches(drop, lim)
    oracle_probs = torch.nn.functional.one_hot(torch.as_tensor(np.maximum(tc, 0)), probs.shape[1]).float()
    results = {}

    def comp(pr):
        return evidence_component(pr.to(device), evidence, gain, model.evidence_gate)[0]

    with torch.no_grad():
        for s, (ids, mask, labels, pos) in batches(full_ds, lim):
            sl = slice(s, s + len(ids))
            dn = torch.tensor(donor[sl])
            rb = r_base[sl].to(device)
            _, embeds = E.thought_embeds(model, ids, mask, pos)
            own = comp(probs[sl])
            raw_lp = target_log_probs(model, E.substitute(embeds, pos, model.combine(rb, own)), mask, labels)
            variants = {
                "swap_evidence": model.combine(rb, comp(probs[dn])),
                "zero_evidence": model.combine(rb, None),
                "oracle_evidence": model.combine(rb, comp(oracle_probs[sl])),
                "swap_r": model.combine(r_base[dn].to(device), own),
            }
            for name, vec in variants.items():
                lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
                for k, v in compare(raw_lp, lp, labels, labels).items():
                    results.setdefault(name, {}).setdefault(k, []).append(v)
            _, (ids_d, mask_d, labels_d, pos_d) = next(it_drop)
            rd, emb_d = E.thought_embeds(model, ids_d, mask_d, pos_d)
            vec = model.combine(rd, comp(model.verifier_logits(rd).softmax(-1).cpu()))
            lp = target_log_probs(model, E.substitute(emb_d, pos_d, vec), mask_d, labels_d)
            for k, v in compare(raw_lp, lp, labels, labels_d).items():
                results.setdefault("drop_last_recomputed", {}).setdefault(k, []).append(v)
    summary["output_shift"] = {name: {k: ci(np.concatenate(v)) for k, v in d.items()} for name, d in results.items()}
    ref = summary["output_shift"]["drop_last_recomputed"]["kl"]["mean"]
    summary["ratio_to_drop_last"] = {k: v["kl"]["mean"] / ref for k, v in summary["output_shift"].items()}
    E.atomic_json(out_dir / "summary.json", summary)
    print({k: round(v, 4) for k, v in summary["ratio_to_drop_last"].items()}, flush=True)

    # 3. ranking
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)

    def generate(count):
        model.config.use_cache = True
        preds = []
        for s in range(0, count, a.gen_batch_size):
            rows = full_ds.rows[s:min(s + a.gen_batch_size, count)]
            prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
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
            preds += [dec[i * 10:(i + 1) * 10] for i in range(len(rows))]
            if s % (a.gen_batch_size * 100) == 0:
                print(f"[{a.tag}] generate {s + len(rows)}/{count}", flush=True)
        model.config.use_cache = False
        return preds

    def ranking(preds, count):
        ranks = np.array([pr.index(r["target_title"].strip(' \n"')) + 1 if r["target_title"].strip(' \n"') in pr else 10 ** 9
                          for pr, r in zip(preds, full_ds.rows[:count])])
        hit = ranks <= 10
        base = r3_hit[:count]
        gained, lost = int((hit & ~base).sum()), int((~hit & base).sum())
        pval = binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
        return hit, {"n": count, **E.recommendation_metrics(ranks), "r3_HR@10_same_rows": float(base.mean()),
                     "gained_vs_r3": gained, "lost_vs_r3": lost, "mcnemar_p": float(pval)}

    g = n if a.gen_limit < 0 else min(a.gen_limit, n)
    preds = generate(g)
    hit, summary["ranking"] = ranking(preds, g)
    sel = np.array([c >= 0.3 for c in conf[:g]])
    summary["ranking"]["HR@10_conf>=0.3"] = float(hit[sel].mean()) if sel.any() else None
    summary["ranking"]["r3_HR@10_conf>=0.3"] = float(r3_hit[:g][sel].mean()) if sel.any() else None
    E.atomic_json(out_dir / "predictions.json", [{"row": i, "top10": pr} for i, pr in enumerate(preds)])
    E.atomic_json(out_dir / "summary.json", summary)
    print(summary["ranking"], flush=True)

    if a.ablate_limit > 0:
        model.evidence_override = "zero"
        ab = min(a.ablate_limit, g)
        hit0, summary["ranking_zero_evidence"] = ranking(generate(ab), ab)
        model.evidence_override = None
        summary["ranking_zero_evidence"]["with_evidence_HR@10_same_rows"] = float(hit[:ab].mean())
        E.atomic_json(out_dir / "summary.json", summary)
        print(summary["ranking_zero_evidence"], flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
