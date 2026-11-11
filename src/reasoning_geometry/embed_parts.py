"""Embed the history / target parts (split_parts.py) with mxbai; same encoder as embed.py.

Writes <out>/<split>_parts.pt = {"history_part": [N,1024] fp16, "target_part": [N,1024] fp16,
"has": [N] bool (row has a target part)}; zeros where a part is empty.
"""

import argparse
import json

import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mixedbread-ai/mxbai-embed-large-v1")
    a = ap.parse_args()
    model = SentenceTransformer(a.model, device="cuda")
    for split in ("train", "valid", "test"):
        d = json.load(open(f"{a.parts}/{split}.json"))
        res = {}
        for field in ("history_part", "target_part"):
            texts = d[field]
            idx = [i for i, t in enumerate(texts) if t]
            v = model.encode([texts[i] for i in idx], batch_size=256, convert_to_tensor=True, show_progress_bar=True).float()
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            M = torch.zeros(len(texts), v.shape[1], dtype=torch.float16)
            M[idx] = v.half().cpu()
            res[field] = M
        res["has"] = torch.tensor([bool(t) for t in d["target_part"]])
        torch.save(res, f"{a.out}/{split}_parts.pt")
        print(f"{split}: rows {len(d['target_part'])}, with target part {int(res['has'].sum())}", flush=True)


if __name__ == "__main__":
    main()
