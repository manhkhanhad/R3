"""Embed GEARec v2 reasoning (best candidate per row) and per-row history text with mxbai (GPU).

Same encoder and normalisation as GEARec's V3 rank check (SentenceTransformer.encode, L2-normalised),
so reasoning vectors live in the space of GEARec's catalog item embeddings (outputs/item_emb_mxbai.pt).

Writes <out>/<split>.pt = {"reasoning": [N,1024] float16 (zeros where missing), "has_reasoning": [N] bool,
"history": [N,1024] float16}; rows follow the R3 CSV order.
"""

import argparse
import ast
import json
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

V2 = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/v2_target_reasoning"
DATA = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
RUNS = {"train": "full_train_bo8_fullhist_r100", "valid": "full_valid_bo8_r100", "test": "full_test_bo8_r100"}


def encode(model, texts, bs):
    v = model.encode(texts, batch_size=bs, convert_to_tensor=True, show_progress_bar=True).float()
    return (v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)).half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mixedbread-ai/mxbai-embed-large-v1")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = SentenceTransformer(a.model, device="cuda")
    for split, run in RUNS.items():
        df = pd.read_csv(f"{DATA}/{split}/{STEM}.csv", usecols=["user_id", "item_id", "history_item_title"])
        n = len(df)
        reasoning = [""] * n
        for line in open(f"{V2}/{run}/judged.jsonl"):
            s = json.loads(line)
            i = s["sample_id"]
            assert df.iloc[i]["user_id"] == s["user_id"] and int(df.iloc[i]["item_id"]) == s["target_id"]
            reasoning[i] = " ".join(((s.get("best") or {}).get("reasoning") or "").split())
        has = torch.tensor([bool(r) for r in reasoning])
        idx = [i for i in range(n) if reasoning[i]]
        R = torch.zeros(n, 1024, dtype=torch.float16)
        R[idx] = encode(model, [reasoning[i] for i in idx], a.batch)
        hist = ["; ".join(ast.literal_eval(h)) for h in df["history_item_title"]]
        H = encode(model, hist, a.batch)
        torch.save({"reasoning": R, "has_reasoning": has, "history": H}, f"{a.out}/{split}.pt")
        print(f"{split}: rows {n}, with reasoning {int(has.sum())}", flush=True)


if __name__ == "__main__":
    main()
