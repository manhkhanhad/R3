"""Extract category-R3 thought vectors (r_cat) for every row of a split CSV.

The prompt is category-R3's own prompt (history + category instruction +
<|Thought|>), and r_cat is its pooled latent, computed exactly as in
LatentModel.generate_embs. Rows are kept in CSV order (no filtering), so row i
of the saved tensor belongs to CSV row i.

Output: <out_dir>/<name>.pt with {"latents": float16 [N, H], "csv": path, "checkpoint": path}.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fire
import torch
from transformers import AutoTokenizer

from category_dataset import CategoryRDataset
from layers import LatentModel


@torch.no_grad()
def main(checkpoint: str, csv_files: str, names: str, out_dir: str, category_file: str, batch_size: int = 16):
    """csv_files / names: comma-separated, same order (e.g. train,valid,test)."""
    # fire turns "a,b,c" into a tuple; accept both
    split = lambda x: list(x) if isinstance(x, (list, tuple)) else str(x).split(",")
    csv_files, names = split(csv_files), split(names)
    assert len(csv_files) == len(names)
    os.makedirs(out_dir, exist_ok=True)
    model = LatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pad = tokenizer.eos_token_id
    thought_id = model.model.embed_tokens.num_embeddings - 1
    for path, name in zip(csv_files, names):
        ds = CategoryRDataset(train_file=path, tokenizer=tokenizer, max_len=2560, sample=-1, test=True,
                              category="music instruments", K=4, category_file=category_file, filter_invalid=False)
        out = []
        for s in range(0, len(ds), batch_size):
            batch = [ds[i]["input_ids"] for i in range(s, min(s + batch_size, len(ds)))]
            L = max(map(len, batch))
            ids = torch.tensor([x + [pad] * (L - len(x)) for x in batch], device="cuda")  # right padding
            mask = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in batch], device="cuda")
            pos = torch.tensor([len(x) - 1 for x in batch], device="cuda")
            assert bool((ids[torch.arange(len(batch)), pos] == thought_id).all())
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            out.append(model.attention(hidden_states=hidden, attention_mask=mask, thought_id_idx=pos).float().cpu())
            if s % (batch_size * 200) == 0:
                print(f"{name}: {s + len(batch)}/{len(ds)}", flush=True)
        latents = torch.cat(out)
        assert latents.shape[0] == len(ds.data)
        torch.save({"latents": latents.half(), "csv": path, "checkpoint": checkpoint}, os.path.join(out_dir, f"{name}.pt"))
        print(f"{name}: saved {tuple(latents.shape)}; mean norm {latents.norm(dim=-1).mean():.2f}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
