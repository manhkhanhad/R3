"""Build fixed item-category targets for category-aligned R3 training."""

import argparse
import csv
import html
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"


def item_asins(split_files):
    asin_to_id = {}
    for path in split_files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                asin, item_id = row["item_asin"], int(row["item_id"])
                if asin in asin_to_id and asin_to_id[asin] != item_id:
                    raise ValueError(f"Conflicting item IDs for ASIN {asin}")
                asin_to_id[asin] = item_id
    ids = sorted(asin_to_id.values())
    if ids != list(range(len(ids))):
        raise ValueError("Item IDs must be contiguous and start at zero")
    return asin_to_id


def category_paths(metadata_file, asin_to_id):
    paths = [None] * len(asin_to_id)
    with open(metadata_file) as handle:
        for line in handle:
            record = json.loads(line)
            item_id = asin_to_id.get(record.get("asin"))
            if item_id is None:
                continue
            category = record.get("category") or []
            if not isinstance(category, list):
                raise ValueError(f"Invalid category for item {item_id}")
            path = " > ".join(html.unescape(str(part)).strip() for part in category)
            if not path:
                continue
            if paths[item_id] is not None and paths[item_id] != path:
                raise ValueError(f"Conflicting category paths for item {item_id}")
            paths[item_id] = path
    return paths


@torch.inference_mode()
def encode_paths(paths, model_name=MODEL_NAME, batch_size=32, device=None):
    device = device or "cuda"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly to run on CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    unique_paths = sorted({path for path in paths if path is not None})
    vectors = {}
    for start in range(0, len(unique_paths), batch_size):
        batch = unique_paths[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        output = model(**{key: value.to(device) for key, value in tokens.items()})
        embeddings = F.normalize(output.last_hidden_state[:, 0].float(), dim=-1).cpu()
        vectors.update(zip(batch, embeddings))
    dimension = model.config.hidden_size
    embeddings = torch.zeros(len(paths), dimension, dtype=torch.float32)
    valid = torch.zeros(len(paths), dtype=torch.bool)
    for item_id, path in enumerate(paths):
        if path is not None:
            embeddings[item_id] = vectors[path]
            valid[item_id] = True
    return embeddings, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    asin_to_id = item_asins(args.split)
    paths = category_paths(args.metadata, asin_to_id)
    embeddings, valid = encode_paths(paths, batch_size=args.batch_size, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "valid": valid, "paths": paths, "model_name": MODEL_NAME}, args.output)
    print(f"Saved {valid.sum().item()}/{len(paths)} item categories to {args.output}")


if __name__ == "__main__":
    main()
