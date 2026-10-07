#!/usr/bin/env python3
"""Evaluate the collaborative-candidate R3 variants (document/collaborative_reasoning_experiments.md).

Checkpoint kinds (``--mode``):

* ``plain``: plain R3 (no candidates).  With ``--predict_only`` this writes the
  top-10 predictions of a split (used for R3 v1 on the validation split, which
  the residual scorer of build_candidates.py is trained on).
* ``text``: candidate titles written into the prompt (idea 2);
* ``slots``: candidate SASRec item vectors in ``<|Cand|>`` slots (ideas 3/4).

Measurements on the test split:

1. Teacher-forced output shift on the target title (first ``--limit`` rows),
   as in decoder_sensitivity.py: ``swap_cands`` (a donor row's candidates, a
   different user, thought recomputed), ``no_cands`` (empty candidate list),
   ``swap_r`` (own candidates, donor's thought), ``mask`` (zero thought) and the
   one-item history drop as the reference.  Reported as ratios to the history drop.
2. Constrained 10-beam generation (first ``--gen_limit`` rows) with own, donor's
   and no candidates: HR/NDCG@10, hits on the donor's target, and the copy rate
   (share of the top-10 that are own / donor candidates).
3. ``--full_ranking``: all test rows with own candidates; per-example top-10 are
   saved, and HR@10 is reported on the rows R3 v1 misses and as a union with v1.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from decoder_sensitivity import ci, compare, target_log_probs
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from latent_grpo_dataset import get_prefix_data

sys.path.insert(0, str(E.ROOT / "src" / "latent"))
sys.path.insert(0, str(E.ROOT / "src" / "collab_reasoning"))
from collab_latent import candidate_line  # noqa: E402
import data as D  # noqa: E402


class CandidatePromptDataset(E.PromptDataset):
    """Prompts identical to training, with a per-row candidate line (``cands[row]``)."""

    def __init__(self, split, tokenizer, limit=-1, mode="plain", cands=None, titles=None, drop_last=False):
        self.mode, self.cands, self.titles, self.drop_last = mode, cands, titles, drop_last
        super().__init__(split, tokenizer, limit)

    def _encode(self, sample_id, row):
        history = ast.literal_eval(row.history_item_title)
        if self.drop_last and len(history) > 1:
            history = history[:-1]
        titles = ", ".join(f'"{x}"' for x in history)
        extra = ""
        items = []
        if self.mode in ("text", "slots"):
            items = [int(x) for x in self.cands[sample_id]]
            extra = "\n" + candidate_line(self.mode, items, self.titles)
        instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request. \n\n### Instruction:\n"
            f"Given a list of {E.CATEGORY_PROMPT} the user recetenly enjoy, please write a new {E.CATEGORY_PROMPT} that the user may bought\n"
        )
        prompt = (
            f"### User Input: \nThe user has palyed the following {E.CATEGORY_PROMPT}s before: {titles}{extra}"
            f"\n\n### Response: \n{E.THOUGHT}"
        )
        target = f'"{str(row.item_title)}"\n'
        ids = self._strip_edges(self.tokenizer.encode(instruction))
        if self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        ids += self._strip_edges(self.tokenizer.encode(prompt))
        prompt_len = len(ids)
        ids += self._strip_edges(self.tokenizer.encode(target)) + [self.tokenizer.eos_token_id]
        return {"sample_id": int(sample_id), "input_ids": ids, "prompt_len": prompt_len, "cand_items": items,
                "user_id": str(row.user_id), "target_item": int(row.item_id), "target_title": str(row.item_title)}


def load(checkpoint: Path, mode: str):
    if mode != "slots":
        model, tokenizer, device = E.load_model(checkpoint)
        return model, tokenizer, device
    from collab_latent import CandidateSlotLatentModel
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, info = CandidateSlotLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"missing keys when loading {checkpoint}: {missing}")
    model = model.to(device).eval()
    model.config.use_cache = False
    table = torch.load(model.config.latent_candidate_slots["source"], map_location="cpu", weights_only=False)
    model.set_cand_table(table["embeddings"], table["valid"])
    return model, tokenizer, device


def thought_and_embeds(model, mode, ids, mask, pos, cands):
    """First R3 pass: the thought vector and the input embeddings (with candidate slots)."""
    if mode != "slots":
        return E.thought_embeds(model, ids, mask, pos)
    with torch.no_grad():
        model.set_cand_items(cands)
        embeds = model.embed_with_candidates(ids)
        model.set_cand_items(None)
        hidden = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        thought = model.attention(hidden, mask, pos)
        embeds = embeds.clone()
        embeds[torch.arange(ids.size(0), device=ids.device), pos] = thought.to(embeds.dtype)
    return thought.float(), embeds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=("plain", "text", "slots"), required=True)
    p.add_argument("--candidates", type=Path, default=None, help="build_candidates.py table (text/slots)")
    p.add_argument("--split", default="test")
    p.add_argument("--predict_only", action="store_true", help="only write top-10 predictions for --split")
    p.add_argument("--predict_limit", type=int, default=-1, help="--predict_only rows (-1: whole split)")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--gen_limit", type=int, default=1000)
    p.add_argument("--full_ranking", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    out_dir = D.RESULTS / "eval" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load(a.checkpoint, a.mode)
    titles = D.item_titles()
    cands = None
    if a.mode != "plain":
        if a.candidates is None and a.mode == "slots":
            a.candidates = Path(model.config.latent_candidate_slots["candidates"])
        cands = torch.load(a.candidates, map_location="cpu", weights_only=False)[a.split].numpy()
    pad = tokenizer.pad_token_id

    title_to_id = {}
    for line in E.INFO_FILE.open():
        t, i = line.rstrip("\n").rsplit("\t", 1)
        title_to_id.setdefault(t.strip(' \n"'), int(i))
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)

    def generate(ds, count):
        model.config.use_cache = True
        preds = []
        for s in range(0, count, a.gen_batch_size):
            rows = ds.rows[s:s + a.gen_batch_size]
            prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
            L = max(map(len, prompts))
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
            masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
            if a.mode == "slots":
                model.set_cand_items(torch.tensor([r["cand_items"] for r in rows]))
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                             prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=masks, generation_config=gen_cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
            dec = [x.strip(' \n"') for x in tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)]
            preds += [dec[i * 10:(i + 1) * 10] for i in range(len(rows))]
            if s % (a.gen_batch_size * 50) == 0:
                print(f"[{a.tag}] generate {s + len(rows)}/{count}", flush=True)
        model.config.use_cache = False
        if a.mode == "slots":
            model.set_cand_items(None)
        return preds

    def score(preds, rows, own=None, donor_rows=None, donor_cands=None):
        ranks, copy_own, copy_donor, hit_donor = [], [], [], []
        for i, (pr, row) in enumerate(zip(preds, rows)):
            tgt = row["target_title"].strip(' \n"')
            ranks.append(pr.index(tgt) + 1 if tgt in pr else 10 ** 9)
            ids = [title_to_id.get(x, -1) for x in pr]
            if own is not None:
                copy_own.append(np.mean([x in set(own[i]) for x in ids]))
            if donor_rows is not None:
                hit_donor.append(float(donor_rows[i]["target_title"].strip(' \n"') in pr))
                copy_donor.append(np.mean([x in set(donor_cands[i]) for x in ids]))
        res = {**E.recommendation_metrics(np.array(ranks)),
               "catalog_match_rate": float(np.mean([x in title_to_id for pr in preds for x in pr]))}
        if copy_own:
            res["copy_rate_own_candidates"] = float(np.mean(copy_own))
        if donor_rows is not None:
            res.update({"donor_target_hit@10": float(np.mean(hit_donor)),
                        "copy_rate_donor_candidates": float(np.mean(copy_donor))})
        return res, np.array(ranks)

    if a.predict_only:
        ds = CandidatePromptDataset(a.split, tokenizer, a.predict_limit, a.mode, cands, titles)
        preds = generate(ds, len(ds))
        res, _ = score(preds, ds.rows)
        with (out_dir / "predictions.json").open("w") as f:
            json.dump([{"top10": pr, "target": r["target_title"]} for pr, r in zip(preds, ds.rows)], f)
        E.atomic_json(out_dir / "summary.json", {"checkpoint": str(a.checkpoint), "split": a.split, "n": len(ds), **res})
        print(res, flush=True)
        return

    # ---- teacher-forced interventions
    base = CandidatePromptDataset("test", tokenizer, a.limit, a.mode, cands, titles)
    n = len(base)
    users = [r["user_id"] for r in base.rows]
    donor = []
    for i in range(n):
        j = (i + n // 2) % n
        for _ in range(n):
            differs = cands is None or list(cands[j]) != list(cands[i])
            if users[j] != users[i] and differs:
                break
            j = (j + 1) % n
        donor.append(j)
    summary = {"checkpoint": str(a.checkpoint), "mode": a.mode, "candidates": str(a.candidates), "n": n, "output_shift": {}}
    results = {}
    collate = lambda x: E.collate(x, pad)

    def batches(ds):
        for s in range(0, len(ds), a.batch_size):
            rows = ds.rows[s:s + a.batch_size]
            b = collate(rows)
            c = torch.tensor([r["cand_items"] for r in rows]) if a.mode == "slots" else None
            yield s, [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")], c

    variants = {"drop_last_recomputed_r": CandidatePromptDataset("test", tokenizer, a.limit, a.mode, cands, titles, drop_last=True)}
    if cands is not None:
        swapped = cands.copy()
        swapped[:n] = cands[np.array(donor)]
        variants["swap_cands"] = CandidatePromptDataset("test", tokenizer, a.limit, a.mode, swapped, titles)
        variants["no_cands"] = CandidatePromptDataset("test", tokenizer, a.limit, a.mode, np.full_like(cands, -1), titles)
    r_base = []
    for _, (ids, mask, labels, pos), c in batches(base):
        r_base.append(thought_and_embeds(model, a.mode, ids, mask, pos, c)[0].cpu())
    r_base = torch.cat(r_base)
    iters = {k: batches(v) for k, v in variants.items()}
    for s, (ids, mask, labels, pos), c in batches(base):
        B = len(ids)
        _, embeds = thought_and_embeds(model, a.mode, ids, mask, pos, c)
        raw_lp = target_log_probs(model, embeds, mask, labels)
        own_r = r_base[s:s + B].to(device)
        for name, vec in (("mask", torch.zeros_like(own_r)), ("swap_r", r_base[torch.tensor(donor[s:s + B])].to(device))):
            lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
            for k, v in compare(raw_lp, lp, labels, labels).items():
                results.setdefault(name, {}).setdefault(k, []).append(v)
        for name, it in iters.items():
            _, (ids_v, mask_v, labels_v, pos_v), c_v = next(it)
            _, emb_v = thought_and_embeds(model, a.mode, ids_v, mask_v, pos_v, c_v)
            lp = target_log_probs(model, emb_v, mask_v, labels_v)
            for k, v in compare(raw_lp, lp, labels, labels_v).items():
                results.setdefault(name, {}).setdefault(k, []).append(v)
        if s % (a.batch_size * 100) == 0:
            print(f"[{a.tag}] teacher-forced {s + B}/{n}", flush=True)
    for name, d in results.items():
        summary["output_shift"][name] = {k: ci(np.concatenate(v)) for k, v in d.items()}
    ref = summary["output_shift"]["drop_last_recomputed_r"]["kl"]["mean"]
    summary["ratio_to_drop_last"] = {k: v["kl"]["mean"] / ref for k, v in summary["output_shift"].items()}
    E.atomic_json(out_dir / "summary.json", summary)
    print({k: round(v, 5) for k, v in summary["ratio_to_drop_last"].items()}, flush=True)

    # ---- generation with own / donor / no candidates
    g = min(a.gen_limit, n)
    donor_rows = [base.rows[j] for j in donor[:g]]
    own_c = [r["cand_items"] for r in base.rows[:g]] if cands is not None else None
    gen = {"true": score(generate(base, g), base.rows[:g], own_c)[0]}
    if cands is not None:
        donor_c = [cands[j].tolist() for j in donor[:g]]
        gen["swap_cands"] = score(generate(variants["swap_cands"], g), base.rows[:g], own_c, donor_rows, donor_c)[0]
        gen["no_cands"] = score(generate(variants["no_cands"], g), base.rows[:g], own_c)[0]
    summary["generation"] = {"n": g, **gen}
    E.atomic_json(out_dir / "summary.json", summary)
    print(summary["generation"], flush=True)

    if a.full_ranking:
        full = CandidatePromptDataset("test", tokenizer, -1, a.mode, cands, titles)
        preds = generate(full, len(full))
        own_all = [r["cand_items"] for r in full.rows] if cands is not None else None
        res, ranks = score(preds, full.rows, own_all)
        test_rows = D.load_rows("test")
        v1 = np.array(D.load_r3_predictions(D.V1_PREDICTIONS, test_rows))
        v1_hit = np.array([titles[t] in {titles[x] for x in row if x >= 0} for t, row in zip(test_rows.target, v1)])
        hit = ranks <= 10
        res.update({"hr10_on_v1_miss": float(hit[~v1_hit].mean()), "hr10_on_v1_hit": float(hit[v1_hit].mean()),
                    "union_with_v1_hr10": float((hit | v1_hit).mean()), "v1_hr10": float(v1_hit.mean())})
        if cands is not None:
            cand_hit = (cands[:len(full)] == test_rows.target[:, None]).any(1)
            res.update({"candidates_hr10": float(cand_hit.mean()),
                        "hr10_when_target_in_candidates": float(hit[cand_hit].mean()),
                        "hr10_when_target_not_in_candidates": float(hit[~cand_hit].mean())})
        summary["full_ranking"] = {"n": len(full), **res}
        with (out_dir / "full_predictions.json").open("w") as f:
            json.dump([{"top10": pr, "target": r["target_title"]} for pr, r in zip(preds, full.rows)], f)
        E.atomic_json(out_dir / "summary.json", summary)
        print(summary["full_ranking"], flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
