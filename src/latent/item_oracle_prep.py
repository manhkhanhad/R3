"""Target-item oracle latents: per-item vector tables in category_prep.py's format.

The additive category latent (category_latent.py / side_info_latent.py,
r' = r + g * normalize(P(v))) is reused with v = a vector of the *target item*
instead of its category.  Question: if the latent explicitly encodes the
target, does the decoder copy it, and does that generalise to items never seen
as training targets (reading) or only work for seen ones (lookup)?

``--kind``:
* ``r3title``: R3's own last-layer representation of the title (mean over the
  quoted title tokens; candidate_title_vectors.py output).
* ``qwen3``: Qwen3-Embedding title embedding (data/<cat>/info/*_qwen3emb_1536.pt).
* ``sasrec``: SASRec item embedding (interaction-only; sasrec_candidates.py).
* ``charac``: item characteristics without the title (brand, category path,
  features, description from the metadata), encoded like ``r3title`` by the R3
  backbone (mean over tokens, <= --max_tokens).  Items without metadata text
  are invalid.

Output keys (as category_prep.py): ``catalog`` (one entry per item id;
index n_items = unknown), ``embeddings`` [n_items, d], ``target`` {split: [N]}
(item id, n_items if the item has no vector), ``lists`` / ``scores``
{split: [N, 5]} = SASRec top-5 candidates / log-probs (``pred1`` = SASRec's
top-1 item vector), plus ``valid`` [n_items], ``seen_target`` {split: [N]}
(target item occurs as a training target), ``kind``.

Usage (from the R3 main checkout):
  item_oracle_prep.py --kind r3title --output output_dir/item_oracle/prep/Toys_item_r3title.pt
"""

import argparse
import ast
import html
import json
import os
import re

import pandas as pd
import torch

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


def item_asins(data_dir):
    id_asin = {}
    for split in ("train", "valid", "test"):
        f = pd.read_csv(os.path.join(data_dir, split, SPLIT_FILE))
        id_asin.update(zip(f.item_id.astype(int), f.item_asin))
        for a_s, i_s in zip(f.item_asins, f.history_item_id):
            id_asin.update(zip(ast.literal_eval(i_s), ast.literal_eval(a_s)))
    return id_asin


def clean(text):
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(text)))
    return re.sub(r"\s+", " ", text).strip()


def characteristics(meta_file, asins):
    """asin -> 'Brand: ...; Category: ...; Features: ...; Description: ...' (no title)."""
    out = {}
    with open(meta_file) as handle:
        for line in handle:
            a = line[line.find('"asin": "') + 9:].split('"', 1)[0] if '"asin": "' in line else None
            if a not in asins:
                continue
            rec = json.loads(line)
            parts = []
            if clean(rec.get("brand", "")):
                parts.append("Brand: " + clean(rec["brand"]))
            cats = [clean(c) for c in rec.get("category") or [] if clean(c)]
            if cats:
                parts.append("Category: " + " > ".join(cats))
            feats = [clean(x) for x in rec.get("feature") or [] if clean(x)]
            if feats:
                parts.append("Features: " + "; ".join(feats))
            desc = " ".join(clean(x) for x in rec.get("description") or [] if clean(x))
            if desc:
                parts.append("Description: " + desc)
            if parts:
                out[rec["asin"]] = ". ".join(parts)
    return out


@torch.no_grad()
def encode_texts(base_model, texts, max_tokens, batch_size=32):
    from transformers import AutoTokenizer, Qwen2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_model)
    backbone = Qwen2Model.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device).eval()
    vecs = torch.zeros(len(texts), backbone.config.hidden_size)
    for s in range(0, len(texts), batch_size):
        enc = [tok.encode(t, add_special_tokens=False)[:max_tokens] for t in texts[s:s + batch_size]]
        width = max(map(len, enc))
        x = torch.tensor([e + [tok.eos_token_id] * (width - len(e)) for e in enc], device=device)
        m = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc], device=device)
        h = backbone(input_ids=x, attention_mask=m).last_hidden_state.float()
        vecs[s:s + len(enc)] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu()
    return vecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["r3title", "qwen3", "sasrec", "charac"])
    p.add_argument("--output", required=True)
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--candidates", default="output_dir/candidate_pooling/prep/Toys_sasrec_candidates.pt")
    p.add_argument("--r3_title_vectors", default="output_dir/candidate_pooling/prep/Toys_r3_title_vectors.pt")
    p.add_argument("--qwen3", default="data/Toys_and_Games/info/Toys_and_Games_qwen3emb_1536.pt")
    p.add_argument("--base_model", default="/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/Toys_and_Games/latent")
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()

    cand = torch.load(args.candidates, map_location="cpu", weights_only=False)
    n = int(cand["n_items"])
    valid = torch.ones(n, dtype=torch.bool)
    texts = None
    if args.kind == "r3title":
        emb = torch.load(args.r3_title_vectors, map_location="cpu", weights_only=False)["vectors"].float()
    elif args.kind == "qwen3":
        emb = torch.load(args.qwen3, map_location="cpu", weights_only=False)["embeddings"].float()
    elif args.kind == "sasrec":
        emb = cand["item_embeddings"].float()
    else:
        id_asin = item_asins(args.data_dir)
        name = os.path.basename(os.path.normpath(args.data_dir))
        char = characteristics(os.path.join(args.data_dir, f"meta_{name}.json"), set(id_asin.values()))
        texts = [char.get(id_asin.get(i), "") for i in range(n)]
        valid = torch.tensor([bool(t) for t in texts])
        emb = encode_texts(args.base_model, [t or "unknown" for t in texts], args.max_tokens)
        emb[~valid] = 0
    assert emb.shape[0] == n, (emb.shape, n)

    train = pd.read_csv(os.path.join(args.data_dir, "train", SPLIT_FILE))
    seen = set(train.item_id.astype(int))
    out = {"kind": args.kind, "catalog": list(range(n)), "embeddings": emb, "valid": valid,
           "target": {}, "lists": {}, "scores": {}, "seen_target": {}}
    for split in ("train", "valid", "test"):
        f = pd.read_csv(os.path.join(args.data_dir, split, SPLIT_FILE))
        y = torch.tensor(f.item_id.astype(int).values)
        out["target"][split] = torch.where(valid[y], y, torch.tensor(n))
        out["seen_target"][split] = torch.tensor([int(i) in seen for i in y.tolist()])
        lists = cand["candidates"][split][:, :5].clone()
        out["lists"][split] = torch.where(valid[lists], lists, torch.tensor(n))
        out["scores"][split] = cand["logp"][split][:, :5].clone()
    if texts is not None:
        out["texts"] = texts
    print(f"{args.kind}: {n} items, dim {emb.shape[1]}, valid {valid.float().mean():.4f}, "
          f"mean norm {emb[valid].norm(dim=-1).mean():.2f}; test targets seen as training targets "
          f"{out['seen_target']['test'].float().mean():.4f}, test targets with a vector "
          f"{(out['target']['test'] < n).float().mean():.4f}")
    if texts is not None:
        print("example:", next(t for t in texts if t)[:300])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)


if __name__ == "__main__":
    main()
