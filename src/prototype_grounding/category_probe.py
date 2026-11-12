#!/usr/bin/env python3
"""Check how well a checkpoint's thought vector predicts the target item's category.

For a category-aligned checkpoint, the trained ``category_projection`` head is
applied to r and matched (cosine) against the unique category-path embeddings.
History-only heuristics (category of the last history item, most frequent
category in the history) and the train-set frequency prior are reported as
references for how much of the target category is recoverable from x alone.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import DataLoader

import experiment as E


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--category_embeddings", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Musical_Instruments")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    E.configure_category(a.category)

    cat = torch.load(a.category_embeddings, map_location="cpu", weights_only=True)
    paths, valid = cat["paths"], cat["valid"]
    path_of = {i: paths[i] for i in range(len(paths)) if valid[i]}
    uniq = sorted(set(path_of.values()))
    uidx = {x: i for i, x in enumerate(uniq)}
    first_item = {}
    for i, x in path_of.items():
        first_item.setdefault(x, i)
    U = F.normalize(torch.stack([cat["embeddings"][first_item[x]] for x in uniq]).float(), dim=-1)

    frame = pd.read_csv(E.SPLIT_FILES[a.split]).iloc[:a.limit]
    train = pd.read_csv(E.SPLIT_FILES["train"])
    prior = Counter(path_of[i] for i in train.item_id if i in path_of).most_common(1)[0][0]

    model, tokenizer, device = E.load_model(a.checkpoint)
    with safe_open(str(a.checkpoint / "model.safetensors"), "pt") as f:
        W = f.get_tensor("category_projection.weight").float().to(device)
        b = f.get_tensor("category_projection.bias").float().to(device)
    ds = E.PromptDataset(a.split, tokenizer, a.limit)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=lambda x: E.collate(x, tokenizer.pad_token_id))
    rs = []
    for batch in loader:
        ids, mask, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        rs.append(E.thought_embeds(model, ids, mask, pos)[0])
    r = torch.cat(rs)
    scores = (F.normalize(r @ W.T + b, dim=-1) @ U.to(device).T).cpu()
    ranking = scores.argsort(dim=-1, descending=True)

    hits = {"model_top1": [], "model_top5": [], "last_history_item": [], "history_majority": [], "train_prior": []}
    for i, row in enumerate(frame.itertuples()):
        tgt = path_of.get(int(row.item_id))
        if tgt is None:
            continue
        t = uidx[tgt]
        hits["model_top1"].append(int(ranking[i, 0] == t))
        hits["model_top5"].append(int(t in ranking[i, :5].tolist()))
        hist = [path_of[h] for h in ast.literal_eval(row.history_item_id) if h in path_of]
        hits["last_history_item"].append(int(bool(hist) and hist[-1] == tgt))
        hits["history_majority"].append(int(bool(hist) and Counter(hist).most_common(1)[0][0] == tgt))
        hits["train_prior"].append(int(prior == tgt))
    summary = {k: float(np.mean(v)) for k, v in hits.items()}
    summary.update({"n": len(hits["model_top1"]), "n_categories": len(uniq), "checkpoint": str(a.checkpoint)})
    out = E.ANALYSIS / "results/decoder_sensitivity" / a.tag / "category_probe.json"
    E.atomic_json(out, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
