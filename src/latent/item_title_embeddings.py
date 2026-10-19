"""Item-title embedding table for a category (``<category>_qwen3emb_1536.pt``).

Same recipe and format as the existing CDs / Musical tables: Qwen3-Embedding-8B,
title text only (no instruction), last-token pooling, first ``dim`` dimensions,
L2-normalized; saved as ``{"embeddings": float32 [n_items, dim], "dim", "model",
"normalized"}`` with row ``i`` = item id ``i``.

Info-file lines are ``<title>\\t<item_id>``; the id is the last field (a few titles
contain tabs).  ``--check_table`` re-embeds a sample of another category's titles and
compares them with its stored table, to confirm the recipe matches.
Needs transformers >= 4.51 (the ``racer`` env).
"""

import argparse
import os

import torch

from explanation_embeddings import Encoder


def read_titles(info_file):
    titles = {}
    for line in open(info_file):
        line = line.rstrip("\n")
        if not line:
            continue
        title, item_id = line.rsplit("\t", 1)
        titles[int(item_id)] = title.strip()
    ids = sorted(titles)
    if ids != list(range(len(ids))):
        raise ValueError(f"{info_file}: item ids are not 0..{len(ids) - 1}")
    return [titles[i] for i in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--check_table", default="", help="existing <category>_qwen3emb_<dim>.pt")
    parser.add_argument("--check_info", default="", help="that category's info file")
    parser.add_argument("--check_n", type=int, default=256)
    args = parser.parse_args()

    encoder = Encoder(args.model, args.dim, "last", max_length=512)
    if args.check_table:
        reference = torch.load(args.check_table, map_location="cpu", weights_only=False)["embeddings"].float()
        check_titles = read_titles(args.check_info)
        sample = torch.randperm(len(check_titles), generator=torch.Generator().manual_seed(0))[: args.check_n]
        fresh = encoder([check_titles[i] for i in sample], args.batch_size)
        cosine = (fresh * torch.nn.functional.normalize(reference[sample], dim=-1)).sum(-1)
        print(f"recipe check vs {args.check_table}: cosine mean {cosine.mean():.5f}, min {cosine.min():.5f}", flush=True)
        if cosine.mean() < 0.99:
            raise SystemExit("re-embedded titles do not match the reference table; refusing to write")

    titles = read_titles(args.info_file)
    embeddings = encoder(titles, args.batch_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"embeddings": embeddings.float(), "dim": embeddings.size(1), "model": args.model,
                "normalized": True}, args.output)
    print(f"saved {args.output}: {tuple(embeddings.shape)}", flush=True)


if __name__ == "__main__":
    main()
