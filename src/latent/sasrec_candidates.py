"""SASRec next-item candidates for SASRec-informed pooling (candidate_pooling.py).

A causal transformer over history item IDs scored by dot product with the
shared item table (as in SASRec), trained with full-catalog cross-entropy at
the last position only (one target per CSV row).  Candidates for training rows
are produced out of fold: rows are split into F folds and each fold is scored
by a model trained on the other folds, so no row's own target was ever a
training target of the model that ranks it (valid/test rows: the model trained
on all training rows).  Every split thus gets candidates of comparable quality.
History items are excluded from the candidates (as in the Cornac exports).

Output (.pt): ``candidates`` / ``logp`` {split: [N, C]} (item ids, log-probs
renormalised over the non-history catalog), ``item_embeddings`` [n_items, d]
(all-training-rows model), ``recall`` per split.  Row order = the split CSV.

Usage: sasrec_candidates.py --data_dir data/Toys_and_Games --output <.pt>
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


class SASRec(nn.Module):
    def __init__(self, n_items, max_len, dim, layers, heads, dropout):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)  # id + 1; 0 = pad
        self.pos = nn.Embedding(max_len, dim)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, items):
        # items: [B, L], left-padded with 0; returns scores over item ids 0..n_items-1
        length = items.size(1)
        x = self.item(items) * self.item.embedding_dim ** 0.5
        x = self.drop(x + self.pos(torch.arange(length, device=items.device)))
        # causal + padded keys hidden; every position keeps itself (else NaN from empty rows)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=items.device), 1)
        blocked = causal.unsqueeze(0) | (items == 0).unsqueeze(1)
        blocked &= ~torch.eye(length, dtype=torch.bool, device=items.device)
        h = self.norm(self.encoder(x, mask=blocked.repeat_interleave(self.heads, 0))[:, -1])
        return h @ self.item.weight[1:].t()


def load(path, max_len):
    frame = pd.read_csv(path)
    items, full = [], []
    for hist in frame["history_item_id"]:
        ids = ast.literal_eval(hist)
        full.append(ids)
        ids = ids[-max_len:]
        items.append([0] * (max_len - len(ids)) + [i + 1 for i in ids])
    return torch.tensor(items), torch.tensor(frame["item_id"].astype(int).values), full


@torch.no_grad()
def top_candidates(model, items, history, c, device, batch=1024):
    """Top-c non-history items and their log-probs (softmax over the non-history catalog)."""
    model.eval()
    ids, logp = [], []
    for i in range(0, len(items), batch):
        s = model(items[i:i + batch].to(device)).float()
        for r, h in enumerate(history[i:i + batch]):
            s[r, h] = -float("inf")
        s = F.log_softmax(s, -1)
        v, j = s.topk(c, -1)
        ids.append(j.cpu()); logp.append(v.cpu())
    return torch.cat(ids), torch.cat(logp)


def recall(top, y, ks=(1, 5, 10, 20)):
    return {k: (top[:, :k] == y[:, None]).any(1).float().mean().item() for k in ks if k <= top.size(1)}


def fit(args, n_items, tr, va, device, tag):
    torch.manual_seed(args.seed)
    model = SASRec(n_items, args.max_len, args.dim, args.layers, args.heads, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_x, tr_y = tr
    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr_y))
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            loss = F.cross_entropy(model(tr_x[b].to(device)), tr_y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        r10 = recall(top_candidates(model, va[0], va[2], 10, device)[0], va[1])[10]
        if r10 > best:
            best, bad = r10, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    print(f"[{tag}] stopped at epoch {epoch}, best valid recall@10 {best:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--output", required=True)
    p.add_argument("--n_candidates", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: load(os.path.join(args.data_dir, s, SPLIT_FILE), args.max_len) for s in ("train", "valid", "test")}
    n_items = sum(1 for _ in open(os.path.join(args.data_dir, "info", SPLIT_FILE.replace(".csv", ".txt"))))
    tr_x, tr_y, tr_h = data["train"]
    va = data["valid"]
    c = args.n_candidates
    out = {"candidates": {}, "logp": {}, "recall": {}, "args": vars(args), "n_items": n_items}

    fold = torch.from_numpy(np.random.default_rng(args.seed).integers(0, args.folds, len(tr_y)))
    cand = torch.zeros(len(tr_y), c, dtype=torch.long)
    logp = torch.zeros(len(tr_y), c)
    for f in range(args.folds):
        held = fold == f
        model = fit(args, n_items, (tr_x[~held], tr_y[~held]), va, device, f"fold {f}")
        idx = torch.nonzero(held).squeeze(1)
        cand[idx], logp[idx] = top_candidates(model, tr_x[idx], [tr_h[i] for i in idx.tolist()], c, device)
    out["candidates"]["train"], out["logp"]["train"] = cand, logp

    model = fit(args, n_items, (tr_x, tr_y), va, device, "all training rows")
    for split in ("valid", "test"):
        x, _, h = data[split]
        out["candidates"][split], out["logp"][split] = top_candidates(model, x, h, c, device)
    out["item_embeddings"] = model.item.weight[1:].detach().float().cpu()
    for split in ("train", "valid", "test"):
        out["recall"][split] = recall(out["candidates"][split], data[split][1])
        print(split, {k: round(v, 4) for k, v in out["recall"][split].items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)


if __name__ == "__main__":
    main()
