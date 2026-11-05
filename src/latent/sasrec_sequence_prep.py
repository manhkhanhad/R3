"""Toys `sasrec_raw`: per-row SASRec history vectors in category_prep.py's format.

Replicates the Musical `sasrec_raw` side-information run
(analysis/document/decoder_insensitivity_to_latent.md, Experiment 1) on Toys
through the additive category-latent path (fp32 side modules):

    r' = r + g * normalize(P(v)),   v = SASRec sequence embedding of the row's history

The Cornac SASRec (128-d) is trained on the train split only, so training rows
get in-sample vectors (as in the Musical run).  The "catalog" here is the list
of rows (train, valid, test concatenated); ``target[split]`` points every row
at its own vector, so the eval's ``oracle`` mode = own vector and ``shuffle`` =
another user's vector.  ``lists`` are all ``unknown`` so that ``pred1`` gives
the zero side vector (= drop).

Usage (from the R3 main checkout):
  sasrec_sequence_prep.py --output output_dir/sasrec_latent/prep/Toys_sasrec_sequence.pt
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sasrec_pair_features import encode_rows, load_sasrec  # noqa: E402

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"
SASREC = ("/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
          "Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--sasrec", default=SASREC)
    args = p.parse_args()

    rec = load_sasrec(args.sasrec)
    vecs, sizes, hits = [], {}, {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(os.path.join(args.data_dir, split, SPLIT_FILE), usecols=["item_asins", "item_asin"])
        v, tgt = encode_rows(rec, frame)
        vecs.append(torch.from_numpy(v))
        sizes[split] = len(frame)
        # in-sample check: SASRec's own top-1 / top-10 hit rate on this split (history not excluded)
        items = rec.model.item_emb.weight.detach().float()[: rec.num_items]
        scores = torch.from_numpy(v) @ torch.nn.functional.normalize(items, dim=-1).T
        top = scores.topk(10, dim=-1).indices.numpy()
        ok = tgt >= 0
        hits[split] = (float((top[ok, 0] == tgt[ok]).mean()), float((top[ok] == tgt[ok, None]).any(1).mean()))
    emb = torch.cat(vecs)
    n = emb.shape[0]
    out = {"kind": "sasrec_sequence", "catalog": list(range(n)), "embeddings": emb,
           "valid": torch.ones(n, dtype=torch.bool), "target": {}, "lists": {}, "scores": {}, "sizes": sizes,
           "sasrec": args.sasrec}
    start = 0
    for split in ("train", "valid", "test"):
        m = sizes[split]
        out["target"][split] = torch.arange(start, start + m)
        out["lists"][split] = torch.full((m, 5), n, dtype=torch.long)
        out["scores"][split] = torch.zeros(m, 5)
        start += m
    for split, (h1, h10) in hits.items():
        print(f"{split}: {sizes[split]} rows; SASRec cosine top-1 {h1:.4f} top-10 {h10:.4f} (history not excluded)")
    print(f"table {tuple(emb.shape)}, mean norm {emb.norm(dim=-1).mean():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)


if __name__ == "__main__":
    main()
