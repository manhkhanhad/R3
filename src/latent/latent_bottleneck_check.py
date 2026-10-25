"""Checkpoint-level check of the L7 latent-only bottleneck (run on GPU after training).

For pairs (a, b) of test rows it compares a's target logits under
  * history swap: b's prompt (history) + a's thought/target, with r forced to r_a;
  * r swap:       a's prompt, with r forced to r_b.
With the bottleneck the first must be ~0 (numerical noise only) while the second is
not.  On a plain-R3 checkpoint the history swap is large.  Runs in fp32 so the
noise floor is small.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
from transformers import AutoTokenizer

from layers import LatentModel
from reasoning_dataset import LatentRDataset

CATEGORIES = {"Musical_Instruments": "music instruments", "CDs_and_Vinyl": "musics",
              "Toys_and_Games": "toys and games"}


def main(checkpoint: str, test_file: str, category: str = "Musical_Instruments",
         pairs: int = 32, tolerance: float = 1e-2, dtype: str = "float32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentModel.from_pretrained(checkpoint, torch_dtype=getattr(torch, dtype)).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    thoughts = getattr(model.config, "latent_continuous_thoughts", None)
    ds = LatentRDataset(train_file=test_file, tokenizer=tokenizer, max_len=2560,
                        sample=2 * pairs, seed=0, category=CATEGORIES[category], K=0,
                        n_thoughts=int(thoughts["k"]) if thoughts else 1)
    thought = model.model.embed_tokens.num_embeddings - 1
    print("latent_bottleneck config:", getattr(model.config, "latent_bottleneck", None) is not None)

    def latent_of(ids):
        if thoughts:  # continuous thoughts: the latent is the whole [1, K, H] block
            logits_of(ids)
            return model.thoughts.last_thoughts.clone()
        captured = {}
        model.observe_reasoning_latent = lambda latent: captured.setdefault("r", latent)
        logits_of(ids)
        del model.observe_reasoning_latent
        return captured["r"].detach()

    def logits_of(ids, r=None):
        if r is not None and thoughts:
            model.thoughts.override = lambda block: r.to(block.dtype).expand_as(block)
        elif r is not None:
            model.observe_reasoning_latent = lambda latent: r.to(latent.dtype).expand_as(latent)
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float()
        if r is not None and thoughts:
            model.thoughts.override = None
        elif r is not None:
            del model.observe_reasoning_latent
        return out[ids.index(thought):-1]  # positions predicting the target tokens

    hist, rswap = [], []
    for i in range(pairs):
        a, b = ds[2 * i]["input_ids"], ds[2 * i + 1]["input_ids"]
        ta, tb = a.index(thought), b.index(thought)
        r_a, r_b = latent_of(a), latent_of(b)
        base = logits_of(a, r_a)
        hybrid = b[:tb] + a[ta:]
        hist.append((logits_of(hybrid, r_a) - base).abs().max().item())
        rswap.append((logits_of(a, r_b) - base).abs().max().item())
    hist_max = max(hist)
    rswap_mean = sum(rswap) / len(rswap)
    print(f"history swap, r fixed: max |dlogit| = {hist_max:.3e} (mean {sum(hist)/len(hist):.3e})")
    print(f"r swap, history fixed: mean max |dlogit| = {rswap_mean:.3e}")
    if getattr(model.config, "latent_bottleneck", None) is not None and hist_max > tolerance:
        raise SystemExit(f"BOTTLENECK LEAK: history swap changed target logits by {hist_max:.3e}")
    print("bottleneck check passed" if hist_max <= tolerance else "history reaches the target (no bottleneck)")


if __name__ == "__main__":
    fire.Fire(main)
