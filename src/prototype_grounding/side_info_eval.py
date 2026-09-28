#!/usr/bin/env python3
"""Evaluate the decoder-insensitivity side-information tests.

Supports three checkpoint types:

* side-information checkpoints (``config.latent_side_info``; src/latent/side_info_latent.py):
  oracle_category, random_category, sasrec_sequence, none (token-scale);
* the category-text positive control (``--category_text``): plain R3 trained with
  the target's category path written into the prompt;
* plain R3 baselines (no side channel).

Measurements (Musical Instruments test split):

1. Teacher-forced output shift on the target title (as in decoder_sensitivity.py),
   for: mask; swap_side (the side component, or the category text, of a donor
   example with a different target category / different user); swap_r (donor's
   base thought, own side component); swap_all (donor's full thought);
   drop_last_{recomputed,fixed}_r (history-change reference).
2. Steering by constrained 10-beam generation on ``--gen_limit`` examples, with
   the true and the donor side information: HR@10, and the fraction of the top-10
   in the own / donor target category, plus hits on the donor's target item.
3. Optional full-test ranking with the true side information (``--full_ranking``).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

import experiment as E
from decoder_sensitivity import ci, compare, target_log_probs
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash
from latent_grpo_dataset import get_prefix_data

sys.path.insert(0, str(E.ROOT / "src" / "latent"))
CATEGORY_FILE = E.ROOT / "data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt"
CATEGORY_TEXT_LINE = "The category of the item to recommend is: {path}"


class VariantPromptDataset(E.PromptDataset):
    """PromptDataset with optional drop-last history and category-text line.

    ``category_item`` maps row position -> item id whose category path is written
    (the row's own target by default; a donor's for the text swap).
    """

    def __init__(self, split, tokenizer, limit=-1, drop_last=False, category_paths=None, category_item=None):
        self.drop_last = drop_last
        self.category_paths = category_paths
        self.category_item = category_item
        self._pos = 0
        super().__init__(split, tokenizer, limit)

    def _encode(self, sample_id, row):
        row = row.copy()
        history = ast.literal_eval(row.history_item_title)
        if self.drop_last and len(history) > 1:
            history = history[:-1]
        titles = ", ".join(f'"{x}"' for x in history)
        extra = ""
        if self.category_paths is not None:
            item = int(row.item_id) if self.category_item is None else int(self.category_item[self._pos])
            extra = "\n" + CATEGORY_TEXT_LINE.format(path=self.category_paths.get(item, "unknown"))
        self._pos += 1
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
        return {
            "sample_id": int(sample_id), "input_ids": ids, "prompt_len": prompt_len,
            "user_id": str(row.user_id), "target_item": int(row.item_id),
            "history_len": len(history), "target_title": str(row.item_title),
        }


def load(checkpoint: Path):
    config = AutoConfig.from_pretrained(checkpoint)
    spec = getattr(config, "latent_side_info", None)
    if spec is None:
        model, tokenizer, device = E.load_model(checkpoint)
        return model, tokenizer, device, None
    from side_info_latent import SideInfoLatentModel
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, info = SideInfoLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]  # tied to embed_tokens
    if missing:
        raise RuntimeError(f"missing keys when loading {checkpoint}: {missing}")
    model = model.to(device).eval()
    model.config.use_cache = False
    kind = spec["kind"]
    source = spec.get("source")
    if source and not Path(source).is_absolute():
        source = str(E.ROOT / source)
    if kind in ("oracle_category", "random_category"):
        cat = torch.load(source, map_location="cpu", weights_only=True)
        model.set_side_table(cat["embeddings"], cat["valid"])
    elif kind == "sasrec_sequence":
        from sasrec_pair_features import build_sasrec_features
        test_file = str(E.SPLIT_FILES["test"])
        _, rows = build_sasrec_features(source, [test_file])
        model.set_side_table(torch.from_numpy(rows[test_file][0]))
    return model, tokenizer, device, spec


def lookup_component(model, index):
    """Side component for explicit table rows (also for the random_category model)."""
    index = index.to(model.side_table.device)
    vec = model.side_table[index].to(model.side_projection.weight.dtype)
    comp = F.normalize(model.side_projection(vec).float(), dim=-1) * model.side_gain.float()
    return comp * model.side_valid[index].unsqueeze(-1).float()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category_text", action="store_true", help="checkpoint is the category-text control")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--gen_limit", type=int, default=1000)
    p.add_argument("--full_ranking", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=8)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    model, tokenizer, device, spec = load(a.checkpoint)
    kind = spec["kind"] if spec else ("category_text" if a.category_text else "plain")
    side = spec is not None and kind != "none"

    cat = torch.load(CATEGORY_FILE, map_location="cpu", weights_only=True)
    paths = {i: x for i, x in enumerate(cat["paths"]) if cat["valid"][i]}
    text_paths = paths if kind == "category_text" else None

    base = VariantPromptDataset("test", tokenizer, a.limit, category_paths=text_paths)
    n = len(base)
    targets = [r["target_item"] for r in base.rows]
    users = [r["user_id"] for r in base.rows]
    # donor: different user and, when possible, a different valid target category
    donor = []
    for i in range(n):
        j = (i + n // 2) % n
        for _ in range(n):
            if users[j] != users[i] and (paths.get(targets[j]) != paths.get(targets[i]) or kind == "sasrec_sequence"):
                break
            j = (j + 1) % n
        donor.append(j)
    donor_items = [targets[j] for j in donor]
    rng = np.random.default_rng(0)
    valid_items = np.array(sorted(paths))
    # side index per row: target item id (category kinds), row index (sasrec), fixed random item (null)
    if kind == "oracle_category":
        own_idx = torch.tensor(targets)
    elif kind == "random_category":
        own_idx = torch.tensor(rng.choice(valid_items, size=n))
    elif kind == "sasrec_sequence":
        own_idx = torch.arange(n)
    else:
        own_idx = None
    donor_idx = own_idx[torch.tensor(donor)] if own_idx is not None else None

    pad = tokenizer.pad_token_id
    collate = lambda x: E.collate(x, pad)

    def batches(ds):
        for s in range(0, len(ds), a.batch_size):
            b = collate(ds.rows[s:s + a.batch_size])
            yield s, [b[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]

    # pass 1: base thoughts (before the side channel)
    r_base = []
    for _, (ids, mask, labels, pos) in batches(base):
        r_base.append(E.thought_embeds(model, ids, mask, pos)[0].cpu())
    r_base = torch.cat(r_base)

    def full(rb, idx):
        if not side:
            return model.combine(rb, None).float() if spec else rb
        return model.combine(rb, lookup_component(model, idx)).float()

    drop = VariantPromptDataset("test", tokenizer, a.limit, drop_last=True, category_paths=text_paths)
    swap_text = (VariantPromptDataset("test", tokenizer, a.limit, category_paths=paths, category_item=donor_items)
                 if kind == "category_text" else None)
    results = {}
    it_drop = batches(drop)
    it_text = batches(swap_text) if swap_text is not None else None
    for s, (ids, mask, labels, pos) in batches(base):
        B = len(ids)
        sl = slice(s, s + B)
        rb = r_base[sl].to(device)
        rb_d = r_base[torch.tensor(donor[sl])].to(device)
        oi = own_idx[sl] if own_idx is not None else None
        di = donor_idx[sl] if donor_idx is not None else None
        _, embeds = E.thought_embeds(model, ids, mask, pos)
        r_true = full(rb, oi)
        raw_lp = target_log_probs(model, E.substitute(embeds, pos, r_true), mask, labels)
        variants = {"mask": torch.zeros_like(r_true), "swap_r": full(rb_d, oi), "swap_all": full(rb_d, di)}
        if side:
            variants["swap_side"] = full(rb, di)
        for name, vec in variants.items():
            lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
            for k, v in compare(raw_lp, lp, labels, labels).items():
                results.setdefault(name, {}).setdefault(k, []).append(v)
        # history-change reference
        _, (ids_d, mask_d, labels_d, pos_d) = next(it_drop)
        rb_new, emb_d = E.thought_embeds(model, ids_d, mask_d, pos_d)
        for name, vec in (("drop_last_recomputed_r", full(rb_new, oi)), ("drop_last_fixed_r", r_true)):
            lp = target_log_probs(model, E.substitute(emb_d, pos_d, vec), mask_d, labels_d)
            for k, v in compare(raw_lp, lp, labels, labels_d).items():
                results.setdefault(name, {}).setdefault(k, []).append(v)
        if it_text is not None:
            _, (ids_t, mask_t, labels_t, pos_t) = next(it_text)
            rb_t, emb_t = E.thought_embeds(model, ids_t, mask_t, pos_t)
            lp = target_log_probs(model, E.substitute(emb_t, pos_t, rb_t), mask_t, labels_t)
            for k, v in compare(raw_lp, lp, labels, labels_t).items():
                results.setdefault("swap_side", {}).setdefault(k, []).append(v)
        if s % (a.batch_size * 100) == 0:
            print(f"[{a.tag}] teacher-forced {s + B}/{n}", flush=True)

    summary = {"checkpoint": str(a.checkpoint), "kind": kind, "n": n, "side_spec": spec, "output_shift": {}}
    for name, d in results.items():
        summary["output_shift"][name] = {k: ci(np.concatenate(v)) for k, v in d.items()}
    ref = summary["output_shift"]["drop_last_recomputed_r"]["kl"]["mean"]
    summary["ratio_to_drop_last"] = {k: v["kl"]["mean"] / ref for k, v in summary["output_shift"].items()}
    if side:
        summary["side_gain"] = float(model.side_gain)
    summary["r_base_norm_mean"] = float(r_base.norm(dim=-1).mean())
    out_dir = E.ANALYSIS / "results/side_info" / a.tag
    E.atomic_json(out_dir / "summary.json", summary)
    print({k: round(v, 5) for k, v in summary["ratio_to_drop_last"].items()}, flush=True)

    # generation: steering and ranking
    title_to_id = {}
    for line in E.INFO_FILE.open():
        t, i = line.rstrip("\n").rsplit("\t", 1)
        title_to_id.setdefault(t.strip(' \n"'), int(i))
    prefix = get_prefix_data(str(E.INFO_FILE), tokenizer)
    gen_cfg = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)

    def generate(ds, idx, count):
        model.config.use_cache = True
        preds = []
        for s in range(0, count, a.gen_batch_size):
            rows = ds.rows[s:s + a.gen_batch_size]
            prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
            L, m = max(map(len, prompts)), min(map(len, prompts))
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=device)
            masks = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=device)
            if spec is not None:
                model.set_side_index(None if idx is None else idx[s:s + len(rows)])
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                                             prefix_allowed_tokens_fn=lambda b, x: prefix.get(get_hash(x), []),
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=masks, generation_config=gen_cfg, return_dict_in_generate=True,
                                     logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), proc]))
            # left-padded prompts all end at position L: decode only generated tokens
            dec = tokenizer.batch_decode(out.sequences[:, L:], skip_special_tokens=True)
            dec = [x.strip(' \n"') for x in dec]
            preds += [dec[i * 10:(i + 1) * 10] for i in range(len(rows))]
            if s % (a.gen_batch_size * 50) == 0:
                print(f"[{a.tag}] generate {s + len(rows)}/{count}", flush=True)
        model.config.use_cache = False
        if spec is not None:
            model.set_side_index(None)
        return preds

    def score(preds, rows, donor_rows=None):
        ranks, cat_own, cat_donor, hit_donor = [], [], [], []
        for i, (pr, row) in enumerate(zip(preds, rows)):
            tgt = row["target_title"].strip(' \n"')
            ranks.append(pr.index(tgt) + 1 if tgt in pr else 10 ** 9)
            cats = [paths.get(title_to_id.get(x)) for x in pr]
            cat_own.append(np.mean([c is not None and c == paths.get(row["target_item"]) for c in cats]))
            if donor_rows is not None:
                d = donor_rows[i]
                cat_donor.append(np.mean([c is not None and c == paths.get(d["target_item"]) for c in cats]))
                hit_donor.append(float(d["target_title"].strip(' \n"') in pr))
        res = {**E.recommendation_metrics(np.array(ranks)), "top10_in_own_category": float(np.mean(cat_own)),
               "catalog_match_rate": float(np.mean([x in title_to_id for pr in preds for x in pr])),
               "examples": [{"target": rows[i]["target_title"], "top3": preds[i][:3]} for i in range(min(3, len(preds)))]}
        if donor_rows is not None:
            res.update({"top10_in_donor_category": float(np.mean(cat_donor)), "donor_target_hit@10": float(np.mean(hit_donor))})
        return res

    g = min(a.gen_limit, n)
    gen = {}
    donor_rows = [base.rows[j] for j in donor[:g]]
    true_preds = generate(base, own_idx, g)
    gen["true"] = score(true_preds, base.rows[:g], donor_rows)
    if side:
        gen["swap_side"] = score(generate(base, donor_idx, g), base.rows[:g], donor_rows)
    elif kind == "category_text":
        gen["swap_side"] = score(generate(swap_text, None, g), base.rows[:g], donor_rows)
    summary["generation"] = {"n": g, **gen}
    E.atomic_json(out_dir / "summary.json", summary)
    print(summary["generation"], flush=True)

    if a.full_ranking:
        full_ds = VariantPromptDataset("test", tokenizer, -1, category_paths=text_paths)
        nf = len(full_ds)
        if kind == "oracle_category":
            idx = torch.tensor([r["target_item"] for r in full_ds.rows])
        elif kind == "sasrec_sequence":
            idx = torch.arange(nf)
        else:
            idx = None
        preds = generate(full_ds, idx, nf)
        summary["full_ranking"] = {"n": nf, **score(preds, full_ds.rows)}
        E.atomic_json(out_dir / "summary.json", summary)
        print(summary["full_ranking"], flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
