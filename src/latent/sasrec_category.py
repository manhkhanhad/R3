"""SASRec-style next-item *category* prediction (transformer over item IDs).

A causal transformer over the history item IDs (optionally plus each history
item's category embedding); the last position's state goes through a softmax
over the level-k category catalog (scripts/category_prediction_data.py).
Trained with cross-entropy on the category CSVs, early stopping on valid top-1.
Writes per-test-row JSON in the ranking-eval format (``output`` / ``predict``,
top-10 category paths), so scripts/category_vs_item_analysis.py can read it.

Usage: sasrec_category.py --data_dir <category data dir> --output <json> [--history_categories]
"""

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


class SASRecCategory(nn.Module):
    def __init__(self, n_items, n_cats, max_len, dim, layers, heads, dropout, history_categories):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)  # id + 1; 0 = pad
        self.cat = nn.Embedding(n_cats + 2, dim, padding_idx=0) if history_categories else None  # +1 unknown
        self.pos = nn.Embedding(max_len, dim)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_cats)
        self.heads = heads

    def forward(self, items, cats):
        # items / cats: [B, L], left-padded with 0, so position -1 is the last history item
        length = items.size(1)
        x = self.item(items) * self.item.embedding_dim ** 0.5
        if self.cat is not None:
            x = x + self.cat(cats)
        x = self.drop(x + self.pos(torch.arange(length, device=items.device)))
        # causal + padded keys hidden; every position keeps itself, else left-padded
        # queries see no key and their NaNs leak into the real positions
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=items.device), 1)
        blocked = causal.unsqueeze(0) | (items == 0).unsqueeze(1)
        blocked &= ~torch.eye(length, dtype=torch.bool, device=items.device)
        h = self.encoder(x, mask=blocked.repeat_interleave(self.heads, 0))
        return self.head(self.norm(h[:, -1]))


def load(path, item_cat, cat_index, max_len, missing):
    frame = pd.read_csv(path)
    items, cats, targets = [], [], []
    for hist, title in zip(frame["history_item_id"], frame["item_title"]):
        ids = ast.literal_eval(hist)[-max_len:]
        pad = max_len - len(ids)
        items.append([0] * pad + [i + 1 for i in ids])
        cats.append([0] * pad + [item_cat.get(i, len(cat_index)) + 1 for i in ids])
        targets.append(cat_index.get(title, -1) if title != missing else -1)
    return torch.tensor(items), torch.tensor(cats), torch.tensor(targets), frame


@torch.no_grad()
def predict(model, items, cats, device, batch=1024):
    model.eval()
    out = [model(items[i:i + batch].to(device), cats[i:i + batch].to(device)).float().cpu()
           for i in range(0, len(items), batch)]
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--raw_data_dir", default="data/Toys_and_Games")
    p.add_argument("--output", required=True)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--history_categories", action="store_true")
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing", default="Unknown")
    args = p.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
    import category_prediction_data as cpd

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = [line.split("\t")[0] for line in open(os.path.join(args.data_dir, "categories.txt"))]
    cat_index = {c: i for i, c in enumerate(catalog)}
    name = os.path.basename(os.path.normpath(args.raw_data_dir))
    paths = cpd.category_map(os.path.join(args.raw_data_dir, f"meta_{name}.json"), args.level)
    item_cat, n_items = {}, 0
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(os.path.join(args.raw_data_dir, split, SPLIT_FILE))
        for asins, ids in zip(frame["item_asins"], frame["history_item_id"]):
            for a, i in zip(ast.literal_eval(asins), ast.literal_eval(ids)):
                if a in paths:
                    item_cat[i] = cat_index[paths[a]]
                n_items = max(n_items, i + 1)
        for a, i in zip(frame["item_asin"], frame["item_id"]):
            if a in paths:
                item_cat[int(i)] = cat_index[paths[a]]
            n_items = max(n_items, int(i) + 1)

    data = {s: load(os.path.join(args.data_dir, s, SPLIT_FILE), item_cat, cat_index, args.max_len, args.missing)
            for s in ("train", "valid", "test")}
    model = SASRecCategory(n_items, len(catalog), args.max_len, args.dim, args.layers, args.heads,
                           args.dropout, args.history_categories).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr_items, tr_cats, tr_y, _ = data["train"]
    va_items, va_cats, va_y, _ = data["valid"]
    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr_y))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            loss = F.cross_entropy(model(tr_items[b].to(device), tr_cats[b].to(device)), tr_y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        acc = (predict(model, va_items, va_cats, device).argmax(1) == va_y).float().mean().item()
        print(f"epoch {epoch}: train loss {total / len(tr_y):.4f}, valid top-1 {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    print(f"best valid top-1 {best:.4f}")

    te_items, te_cats, te_y, frame = data["test"]
    top = predict(model, te_items, te_cats, device).topk(10, dim=1).indices
    rows = [{"output": f'"{title}"\n', "predict": [catalog[j] for j in top[r].tolist()]}
            for r, title in enumerate(frame["item_title"])]
    known = te_y >= 0
    for k in (1, 3, 5, 10):
        hit = (top[:, :k] == te_y.unsqueeze(1)).any(1)
        print(f"test HR@{k} {hit[known].float().mean().item():.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(rows, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
