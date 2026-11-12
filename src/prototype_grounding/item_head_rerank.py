#!/usr/bin/env python3
"""Use a multi-aspect checkpoint's item head over the full catalog.

For each example:
  * item head alone: rank all catalog items by log p_item (HR/NDCG@10);
  * candidate set = head top-``--head_k`` items  ∪  decoder 10-beam list;
  * every candidate is scored by the decoder's teacher-forced mean token
    log-likelihood of its title (prompt KV cache reused, so the thought is
    spliced once), and reranked by

        d(item) + lam * log p_item(item)                 ("item")
        d(item) + lam * sum_k g_k log p_k(code_k(item))  ("all aspects")

lam is chosen on --valid_limit validation examples (decoder beams generated
here) and applied to test (decoder beams read from the standard-eval json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import experiment as E
from multi_aspect_eval import generate_candidates

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def catalog(tokenizer):
    titles, ids = {}, {}
    for line in E.INFO_FILE.open():
        t, i = line.rstrip("\n").rsplit("\t", 1)
        t = t.strip(' \n"')
        titles.setdefault(int(i), t)
        ids.setdefault(t, int(i))
    enc = {}
    for i, t in titles.items():
        ids_ = tokenizer.encode(f'"{t}"\n')
        while ids_ and ids_[0] == tokenizer.bos_token_id:
            ids_.pop(0)
        while ids_ and ids_[-1] == tokenizer.eos_token_id:
            ids_.pop()
        enc[i] = ids_ + [tokenizer.eos_token_id]
    return titles, ids, enc


@torch.no_grad()
def score_example(model, prompt, cand_ids, enc, device, item_k):
    """Decoder mean log-lik of each candidate title + all aspect log-probs + gates."""
    p = torch.tensor([prompt], device=device)
    out = model(input_ids=p, attention_mask=torch.ones_like(p), use_cache=True)
    att = model.attention
    asp_logp = [F.log_softmax(l.float()[0], -1) for l in att.last_logits]
    gate = att.last_gate.float()[0]
    first = F.log_softmax(out.logits[0, -1].float(), -1)
    seqs = [enc[c] for c in cand_ids]
    T = max(map(len, seqs))
    C = len(seqs)
    tok = torch.full((C, T), model.config.eos_token_id, device=device)
    msk = torch.zeros((C, T), dtype=torch.long, device=device)
    for j, s in enumerate(seqs):
        tok[j, :len(s)] = torch.tensor(s, device=device)
        msk[j, :len(s)] = 1
    cache = out.past_key_values
    cache.batch_repeat_interleave(C)
    am = torch.cat([torch.ones((C, len(prompt)), dtype=torch.long, device=device), msk], 1)
    lo = model(input_ids=tok, attention_mask=am, past_key_values=cache, use_cache=True).logits.float()
    lp = F.log_softmax(lo, -1)
    tok_lp = torch.zeros((C, T), device=device)
    tok_lp[:, 0] = first[tok[:, 0]]
    if T > 1:
        tok_lp[:, 1:] = lp[:, :-1].gather(-1, tok[:, 1:].unsqueeze(-1)).squeeze(-1)
    d = ((tok_lp * msk).sum(1) / msk.sum(1)).cpu().numpy()
    return d, [a.cpu() for a in asp_logp], gate.cpu()


def run_split(model, tokenizer, device, split, limit, beams_per_row, head_k, codes, names, enc):
    ds = E.PromptDataset(split, tokenizer, limit)
    item_k = names.index("item")
    recs = []
    for i, row in enumerate(ds.rows):
        prompt = row["input_ids"][:row["prompt_len"]]
        # head ranking needs the item head over the catalog: one prompt pass first
        with torch.no_grad():
            p = torch.tensor([prompt], device=device)
            model(input_ids=p, attention_mask=torch.ones_like(p), use_cache=False)
            head = F.log_softmax(model.attention.last_logits[item_k].float()[0], -1).cpu()
        head_top = head.argsort(descending=True)[:head_k].tolist()
        cands = list(dict.fromkeys(beams_per_row[i] + head_top))
        cands = [c for c in cands if c in enc]
        d, asp, gate = score_example(model, prompt, cands, enc, device, item_k)
        asp_c = np.array([[float(asp[k][codes[k, c]]) if codes[k, c] >= 0 else -30.0 for k in range(len(names))] for c in cands])
        recs.append({"target": row["target_item"], "cands": cands, "d": d, "asp": asp_c, "gate": gate.numpy(),
                     "head_rank": int((head > head[row["target_item"]]).sum()) + 1,
                     "beams": beams_per_row[i]})
        if i % 500 == 0:
            print(f"{split}: {i + 1}/{len(ds)}", flush=True)
    return recs


def metrics_from_ranks(ranks):
    return E.recommendation_metrics(np.array(ranks))


def rerank(recs, lam, mode, item_k):
    ranks = []
    for r in recs:
        if mode == "item":
            s = r["d"] + lam * r["asp"][:, item_k]
        else:
            s = r["d"] + lam * (r["asp"] * r["gate"][None]).sum(1)
        order = np.argsort(-s, kind="stable")[:10]
        top = [r["cands"][j] for j in order]
        ranks.append(top.index(r["target"]) + 1 if r["target"] in top else 10 ** 9)
    return metrics_from_ranks(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--test_json", type=Path, required=True, help="standard-eval 0.json (decoder 10-beam lists)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--valid_limit", type=int, default=2000)
    ap.add_argument("--test_limit", type=int, default=-1)
    ap.add_argument("--head_k", type=int, default=50)
    a = ap.parse_args()
    E.configure_category("Musical_Instruments")
    model, tokenizer, device = E.load_model(a.checkpoint)
    att = model.attention
    codes, names = att.aspect_codes.cpu(), att.names
    item_k = names.index("item")
    titles, title_ids, enc = catalog(tokenizer)

    # decoder beams: valid generated here, test from the standard-eval output
    vgen = generate_candidates(model, tokenizer, device, "valid", a.valid_limit, 10, 4, codes)
    vbeams = [[title_ids[c] for c in r["cands"] if c in title_ids] for r in vgen]
    tj = json.load(open(a.test_json))
    if a.test_limit > 0:
        tj = tj[:a.test_limit]
    tbeams = [[title_ids[x.strip(' \n"')] for x in r["predict"] if x.strip(' \n"') in title_ids] for r in tj]

    valid = run_split(model, tokenizer, device, "valid", a.valid_limit, vbeams, a.head_k, codes, names, enc)
    test = run_split(model, tokenizer, device, "test", a.test_limit, tbeams, a.head_k, codes, names, enc)

    out = {"checkpoint": str(a.checkpoint), "head_k": a.head_k, "valid_n": len(valid), "test_n": len(test)}
    for split, recs in (("valid", valid), ("test", test)):
        beam_ranks = [r["beams"].index(r["target"]) + 1 if r["target"] in r["beams"][:10] else 10 ** 9 for r in recs]
        out[split] = {
            "decoder_beam": metrics_from_ranks(beam_ranks),
            "item_head_full_catalog": metrics_from_ranks([r["head_rank"] if r["head_rank"] <= 10 else 10 ** 9 for r in recs]),
            "candidate_recall": float(np.mean([r["target"] in r["cands"] for r in recs])),
            # sanity: teacher-forced d should rank the decoder's own top beam first among its beams
            "top_beam_agreement": float(np.mean([
                r["beams"][0] == max(r["beams"], key=lambda c: r["d"][r["cands"].index(c)])
                for r in recs if r["beams"] and all(c in r["cands"] for c in r["beams"])])),
            "grid_item": {str(l): rerank(recs, l, "item", item_k) for l in LAMBDAS},
            "grid_all": {str(l): rerank(recs, l, "all", item_k) for l in LAMBDAS},
        }
    for mode in ("item", "all"):
        best = max(LAMBDAS, key=lambda l: out["valid"][f"grid_{mode}"][str(l)]["NDCG@10"])
        out[f"chosen_lambda_{mode}"] = best
        out[f"test_rescored_{mode}"] = out["test"][f"grid_{mode}"][str(best)]
    path = E.ANALYSIS / "results/multi_aspect" / a.tag / "item_head_rerank.json"
    E.atomic_json(path, out)
    for k in ("decoder_beam", "item_head_full_catalog", "candidate_recall", "top_beam_agreement"):
        print(k, out["test"][k], flush=True)
    for mode in ("item", "all"):
        print(mode, out[f"chosen_lambda_{mode}"], out[f"test_rescored_{mode}"], flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
