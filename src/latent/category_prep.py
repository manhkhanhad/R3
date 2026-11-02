"""Inputs for category-conditioned latents (category_latent.py), Toys level-3.

Writes one .pt with
* ``catalog``: the level-3 category paths (categories.txt order); index
  ``len(catalog)`` = unknown category.
* ``embeddings`` [n_cats, d]: gte-large-en-v1.5 CLS embeddings of the paths
  (root "Toys & Games > " removed), L2-normalised (additive injection).
* ``token_ids``: per category, Qwen tokens of the path without the root
  (IntuRec-style injection encodes these).
* ``target`` {split: [N]}: the row's target category index.
* ``lists`` {split: [N, K]} and ``scores`` {split: [N, K]}: predicted top-K
  categories.  train: SASRec category model (item IDs + history categories)
  out of fold (F folds, each scored by a model trained on the other folds);
  valid: the model trained on all training rows; test: the LLM SFT category
  model's constrained beam top-K and its sequence scores (eval_category.json).

Usage (from the R3 main checkout):
  category_prep.py --output output_dir/category_latent/prep/Toys_category_prep.pt
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--category_dir", default="output_dir/category_prediction/Toys_and_Games_level3/data")
    p.add_argument("--llm_eval", default="output_dir/category_prediction/Toys_and_Games_category_level3/eval_category.json")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.join(here, "..", "..", "scripts")]
    import category_prediction_data as cpd
    from sasrec_category import SASRecCategory
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = [l.split("\t")[0] for l in open(os.path.join(args.category_dir, "categories.txt"))]
    cidx = {c: i for i, c in enumerate(catalog)}
    unknown = len(catalog)
    strip = lambda c: c.split(" > ", 1)[1] if " > " in c else c
    paths = cpd.category_map(os.path.join(args.data_dir, "meta_Toys_and_Games.json"), args.level)

    gte_tok = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
    gte = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True).eval()
    with torch.no_grad():
        b = gte_tok([strip(c) for c in catalog], padding=True, return_tensors="pt")
        emb = F.normalize(gte(**b).last_hidden_state[:, 0].float(), dim=-1)
    qtok = AutoTokenizer.from_pretrained(args.base_model)
    token_ids = [qtok.encode(strip(c), add_special_tokens=False) for c in catalog]

    frames = {s: pd.read_csv(os.path.join(args.data_dir, s, SPLIT_FILE)) for s in ("train", "valid", "test")}
    target = {s: torch.tensor([cidx.get(paths.get(a), unknown) for a in f["item_asin"]]) for s, f in frames.items()}

    # SASRec category model inputs (item IDs + history categories), as sasrec_category.py
    n_items = 1 + max(max(int(f["item_id"].max()), max(max(ast.literal_eval(h)) for h in f["history_item_id"]))
                      for f in frames.values())
    item_cat = {}
    for f in frames.values():
        for a_s, i_s in zip(f["item_asins"], f["history_item_id"]):
            for a, i in zip(ast.literal_eval(a_s), ast.literal_eval(i_s)):
                if paths.get(a) in cidx:
                    item_cat[i] = cidx[paths[a]]
        for a, i in zip(f["item_asin"], f["item_id"]):
            if paths.get(a) in cidx:
                item_cat[int(i)] = cidx[paths[a]]

    def tensors(f, max_len=10):
        items, cats = [], []
        for h in f["history_item_id"]:
            ids = ast.literal_eval(h)[-max_len:]
            pad = max_len - len(ids)
            items.append([0] * pad + [i + 1 for i in ids])
            cats.append([0] * pad + [item_cat.get(i, unknown) + 1 for i in ids])
        return torch.tensor(items), torch.tensor(cats)

    X = {s: tensors(f) for s, f in frames.items()}

    @torch.no_grad()
    def predict(model, items, cats):
        model.eval()
        out = [F.log_softmax(model(items[i:i + 1024].to(device), cats[i:i + 1024].to(device)).float(), -1).cpu()
               for i in range(0, len(items), 1024)]
        return torch.cat(out)

    def fit(rows):
        model = SASRecCategory(n_items, len(catalog), 10, 64, 2, 2, 0.2, True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        items, cats = X["train"][0][rows], X["train"][1][rows]
        y = target["train"][rows]
        ok = y < unknown
        items, cats, y = items[ok], cats[ok], y[ok]
        va_ok = target["valid"] < unknown
        best, state, bad = -1.0, None, 0
        for epoch in range(200):
            model.train()
            order = torch.randperm(len(y))
            for i in range(0, len(order), 256):
                b = order[i:i + 256]
                loss = F.cross_entropy(model(items[b].to(device), cats[b].to(device)), y[b].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
            acc = (predict(model, *X["valid"]).argmax(1)[va_ok] == target["valid"][va_ok]).float().mean().item()
            if acc > best:
                best, bad, state = acc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        model.load_state_dict(state)
        print(f"  stopped at epoch {epoch}, best valid top-1 {best:.4f}", flush=True)
        return model

    n = len(frames["train"])
    fold = torch.from_numpy(np.random.default_rng(args.seed).integers(0, args.folds, n))
    lists = {"train": torch.zeros(n, args.k, dtype=torch.long)}
    scores = {"train": torch.zeros(n, args.k)}
    for f in range(args.folds):
        print(f"fold {f}", flush=True)
        held = fold == f
        model = fit(torch.nonzero(~held).squeeze(1))
        s, i = predict(model, X["train"][0][held], X["train"][1][held]).topk(args.k, -1)
        lists["train"][held], scores["train"][held] = i, s
    print("all training rows", flush=True)
    model = fit(torch.arange(n))
    s, i = predict(model, *X["valid"]).topk(args.k, -1)
    lists["valid"], scores["valid"] = i, s
    llm = json.load(open(args.llm_eval))
    assert len(llm) == len(frames["test"])
    lists["test"] = torch.tensor([[cidx.get(c.strip().strip('"'), unknown) for c in r["predict"][:args.k]] for r in llm])
    scores["test"] = torch.tensor([r["predict_score"][:args.k] for r in llm])
    for split in ("train", "valid", "test"):
        ok = target[split] < unknown
        hit = (lists[split] == target[split][:, None])
        print(split, "rows", len(ok), "with category", int(ok.sum()),
              "top-1", round(hit[ok, :1].any(1).float().mean().item(), 4),
              f"top-{args.k}", round(hit[ok].any(1).float().mean().item(), 4), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"catalog": catalog, "embeddings": emb, "token_ids": token_ids, "target": target,
                "lists": lists, "scores": scores, "args": vars(args)}, args.output)


if __name__ == "__main__":
    main()
