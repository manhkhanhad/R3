"""GPU self-check of continuous thoughts (continuous_thoughts.py), fp32.

1. Padding/batching: a row's thoughts are the same whether it is computed alone,
   in a right-padded batch (training) or a left-padded batch (generation).
2. KV cache: the cached K-step recurrence equals a naive recomputation that
   re-runs the whole prefix + thoughts-so-far without a cache.
3. Splicing: generate_embs puts the K thoughts exactly at the K thought positions.
Works on a fresh base model (random identity-init projection) or a checkpoint.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
from transformers import AutoConfig, AutoTokenizer

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from continuous_thoughts import first_thought_positions


def main(test_file: str, model_path: str = "Qwen/Qwen2.5-1.5B", k: int = 4, rows: int = 4,
         category: str = "toys and games", tolerance: float = 2e-3):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    fresh = not getattr(config, "latent_continuous_thoughts", None)
    if fresh:
        config.latent_continuous_thoughts = {"k": k, "source": "hidden", "token_norm": 1.0}
    model = LatentModel.from_pretrained(model_path, config=config, torch_dtype=torch.float32).to(device).eval()
    if fresh:
        tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + ["<|Thought|>"]})
        model.resize_token_embeddings(len(tokenizer))
        model.thoughts.reset_parameters()
        model.thoughts.token_norm.fill_(float(model.model.embed_tokens.weight[:-1].norm(dim=-1).mean()))
    k = model.thoughts.k
    thought_id = model.model.embed_tokens.num_embeddings - 1
    assert tokenizer.convert_tokens_to_ids("<|Thought|>") == thought_id
    ds = LatentRDataset(train_file=test_file, tokenizer=tokenizer, max_len=2560, sample=rows, seed=1,
                        category=category, test=True, n_thoughts=k)
    prompts = [ds[i]["input_ids"] for i in range(rows)]
    pad = tokenizer.eos_token_id

    def batch(side):
        width = max(map(len, prompts))
        ids, mask = [], []
        for p in prompts:
            fill = [pad] * (width - len(p))
            ids.append(p + fill if side == "right" else fill + p)
            mask.append([1] * len(p) + [0] * len(fill) if side == "right" else [0] * len(fill) + [1] * len(p))
        return torch.tensor(ids, device=device), torch.tensor(mask, device=device)

    def thoughts_of(ids, mask):
        with torch.no_grad():
            first = first_thought_positions(ids, thought_id, k)
            return model.thoughts(model.model, ids, mask, first)

    alone = torch.cat([thoughts_of(torch.tensor([p], device=device), torch.ones(1, len(p), dtype=torch.long, device=device))
                       for p in prompts])
    right = thoughts_of(*batch("right"))
    left = thoughts_of(*batch("left"))

    naive = []
    for p in prompts:
        first = p.index(thought_id)
        emb = model.model.embed_tokens(torch.tensor([p[:first]], device=device))
        steps = []
        with torch.no_grad():
            for _ in range(k):
                h = model.model(inputs_embeds=emb).last_hidden_state[:, -1]
                t = model.thoughts.project(h)
                steps.append(t)
                emb = torch.cat([emb, t.to(emb.dtype)[:, None]], dim=1)
        naive.append(torch.stack(steps, dim=1))
    naive = torch.cat(naive)

    ids, mask = batch("left")
    with torch.no_grad():
        embs = model.generate_embs(ids, mask)
    first = first_thought_positions(ids, thought_id, k)
    spliced = torch.stack([embs[b, first[b]:first[b] + k] for b in range(rows)]).float()

    scale = alone.norm(dim=-1).mean().item()
    checks = {
        "right-padded batch vs alone": (right - alone).abs().max().item(),
        "left-padded batch vs alone": (left - alone).abs().max().item(),
        "cached recurrence vs naive recompute": (alone - naive).abs().max().item(),
        "spliced embeddings vs thoughts": (spliced - left).abs().max().item(),
    }
    print(f"k={k}, rows={rows}, thought norm {scale:.3f}, prompt lengths {[len(p) for p in prompts]}")
    spread = (alone.flatten(1) - alone.flatten(1).mean(0)).norm(dim=1) / alone.flatten(1).norm(dim=1)
    print(f"per-sample share of thoughts (fresh projection): {spread.mean().item():.3f}")
    failed = False
    for name, value in checks.items():
        ok = value <= tolerance * max(scale, 1.0)
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max |diff| = {value:.3e}")
    if failed:
        raise SystemExit("continuous-thought check failed")
    print("continuous-thought check passed")


if __name__ == "__main__":
    fire.Fire(main)
