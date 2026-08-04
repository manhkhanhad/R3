"""Generate item-title embeddings with Qwen3-Embedding-8B for the MSE anchor loss.

The info file is tab-separated "title\tid" (including "pad_title 0"). Embeddings
are MRL-truncated to `dim` dims (Qwen3-Embedding supports custom output dims)
and L2-normalized AFTER truncation, then saved indexed by item id.

Usage:
  python src/utils/gen_item_embeddings.py \
    --info_file data/CDs_and_Vinyl_3_2022-10-2023-10_randNeg/info/CDs_and_Vinyl.txt \
    --output_file data/CDs_and_Vinyl_3_2022-10-2023-10_randNeg/info/CDs_and_Vinyl_qwen3emb_1536.pt
"""
import os

import fire
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states, attention_mask):
    # per the Qwen3-Embedding model card (left padding -> last position)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def main(
    info_file: str,
    output_file: str = None,
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    dim: int = 1536,
    batch_size: int = 32,
    max_length: int = 64,
):
    if output_file is None:
        base = os.path.splitext(os.path.basename(info_file))[0]
        output_file = os.path.join(os.path.dirname(info_file), f"{base}_qwen3emb_{dim}.pt")

    titles_by_id = {}
    with open(info_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            title, item_id = line.rsplit("\t", 1)
            titles_by_id[int(item_id)] = title
    n_items = len(titles_by_id)
    assert sorted(titles_by_id) == list(range(n_items)), \
        "item ids must be contiguous 0..N-1"
    titles = [titles_by_id[i] for i in range(n_items)]
    print(f"{n_items} items from {info_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda().eval()

    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, n_items, batch_size)):
            batch = titles[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(model.device)
            out = model(**inputs)
            emb = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
            emb = emb.float()[:, :dim]                # MRL truncation
            emb = F.normalize(emb, p=2, dim=-1)       # normalize AFTER truncation
            all_embs.append(emb.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    assert embeddings.shape == (n_items, dim)
    torch.save({
        "embeddings": embeddings,
        "dim": dim,
        "model": model_name,
        "normalized": True,
    }, output_file)
    print(f"saved {tuple(embeddings.shape)} -> {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
