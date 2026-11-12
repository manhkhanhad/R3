#!/usr/bin/env python3
"""Distribution of the cluster head's confidence (max_j p_j) for a staged-refinement checkpoint.

Per test example: confidence, whether the head's top-1 is the target's cluster, the
target cluster's probability, and the decoder's HR@10 from the standard-eval json.
Reports quantiles, a histogram, and accuracy / HR@10 / refinement norm per bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import experiment as E
from staged_refine_eval import first_pass

BINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test_json", type=Path, required=True, help="standard-eval 0.json for per-example HR@10")
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    model, tokenizer, device = E.load_model(a.checkpoint)
    att = model.attention
    codes = att.cluster_codes.cpu()
    ds = E.PromptDataset("test", tokenizer, a.limit)
    col = lambda x: E.collate(x, tokenizer.pad_token_id)
    conf, top1, p_true, comp = [], [], [], []
    for s in range(0, len(ds), a.batch_size):
        b = col(ds.rows[s:s + a.batch_size])
        ids, mask, pos = [b[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        thought, r_item, logits, _, _, _ = first_pass(model, ids, mask, pos)
        pr = F.softmax(logits, -1).cpu()
        conf.append(pr.max(-1).values)
        top1.append(pr.argmax(-1))
        t = torch.tensor([int(codes[int(i)]) for i in ds.frame.item_id.iloc[s:s + len(ids)]])
        p_true.append(torch.where(t >= 0, pr[torch.arange(len(t)), t.clamp_min(0)], torch.full_like(t, -1, dtype=pr.dtype)))
        comp.append((thought - r_item).norm(dim=-1).cpu())
        if s % (a.batch_size * 200) == 0:
            print(f"{s + len(ids)}/{len(ds)}", flush=True)
    conf, top1, p_true, comp = [torch.cat(x).numpy() for x in (conf, top1, p_true, comp)]
    target = np.array([int(codes[int(i)]) for i in ds.frame.item_id])
    valid = target >= 0
    correct = top1 == target

    preds = json.load(open(a.test_json))[:len(ds)]
    hit10 = np.array([r["output"].strip(' \n"') in [x.strip(' \n"') for x in r["predict"][:10]] for r in preds])

    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    out = {"checkpoint": str(a.checkpoint), "n": int(len(conf)), "mean": float(conf.mean()), "std": float(conf.std()),
           "min": float(conf.min()), "max": float(conf.max()),
           "quantiles": {str(q): float(np.quantile(conf, q)) for q in qs},
           "uniform_floor": 1.0 / att.num_clusters, "bins": []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        mv = m & valid
        out["bins"].append({"range": f"[{lo:.2f}, {min(hi, 1.0):.2f})", "n": int(m.sum()), "share": float(m.mean()),
                            "top1_accuracy": float(correct[mv].mean()) if mv.any() else None,
                            "mean_p_true_cluster": float(p_true[mv].mean()) if mv.any() else None,
                            "decoder_hr@10": float(hit10[m].mean()) if m.any() else None,
                            "refinement_norm": float(comp[m].mean()) if m.any() else None})
    path = E.ANALYSIS / "results/staged_refine" / a.tag / "confidence.json"
    E.atomic_json(path, out)
    np.savez(path.with_suffix(".npz"), confidence=conf, correct=correct, valid=valid, p_true=p_true, hit10=hit10, component_norm=comp)
    print(json.dumps(out, indent=1), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
