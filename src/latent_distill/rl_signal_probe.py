"""How much learning signal does R3's latent-noise GRPO stage get? (document/latent_distillation.md)

``src/grpo_attention_tuning`` samples G noisy copies of the thought vector per row
(r + N(0, 1.5^2 I), the first copy noise-free), scores each by the teacher-forced
likelihood of the target title, exp(-mean token NLL), and uses the reward relative
to the noise-free copy as the advantage; only the pooler is trained.  If the
decoder ignores r's content, rewards within a group are nearly identical and the
policy gradient vanishes.  This script measures, per row:

  - the within-group reward spread (std / mean) under that noise,
  - the same spread when r is replaced by other rows' r (a much larger move),
  - ||noise|| / ||r||.

Usage: python src/latent_distill/rl_signal_probe.py --checkpoint <R3 ckpt> --csv <train csv> --rows 512
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "latent"))

import fire
import numpy as np
import torch
from transformers import AutoTokenizer

from layers import LatentModel
from reasoning_dataset import LatentRDataset


@torch.no_grad()
def target_rewards(model, ids, labels, slot, latents):
    """exp(-mean target NLL) for each latent in ``latents`` [G, H], same prompt + target."""
    g = latents.shape[0]
    embeds = model.model.embed_tokens(ids).repeat(g, 1, 1)
    embeds[:, slot] = latents.to(embeds.dtype)
    logits = model(inputs_embeds=embeds, attention_mask=torch.ones(g, ids.shape[1], device=ids.device)).logits.float()
    logp = logits[:, :-1].log_softmax(-1)
    tgt = labels[0, 1:]
    keep = tgt != -100
    tok = logp[:, keep].gather(-1, tgt[keep].unsqueeze(0).unsqueeze(-1).expand(g, -1, 1)).squeeze(-1)
    return (tok.mean(-1)).exp()


@torch.no_grad()
def main(checkpoint: str, csv: str, out: str, rows: int = 512, groups: int = 8, noise_std: float = 1.5, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    model = LatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data = LatentRDataset(train_file=csv, tokenizer=tokenizer, max_len=1024, sample=rows, seed=seed,
                          category="music instruments")
    thought = model.model.embed_tokens.num_embeddings - 1

    latents, items = [], []
    for ex in data:
        ids = torch.tensor([ex["input_ids"]], device=device)
        r = model.generate_embs(ids, torch.ones_like(ids))
        slot = int((ids[0] == thought).nonzero()[0, 0])
        latents.append(r[0, slot].float())
        items.append((ids, torch.tensor([ex["labels"]], device=device), slot))
    bank = torch.stack(latents)

    stats = {"noise": [], "swap": []}
    norm_ratio = []
    for i, (ids, labels, slot) in enumerate(items):
        r = bank[i]
        noise = torch.randn(groups, r.shape[0], device=device) * noise_std
        noise[0] = 0
        norm_ratio.append(float(noise[1:].norm(dim=-1).mean() / r.norm()))
        others = bank[torch.randperm(len(bank), device=device)[:groups - 1]]
        for name, group in (("noise", r + noise), ("swap", torch.cat([r[None], others]))):
            rew = target_rewards(model, ids, labels, slot, group).cpu().numpy()
            stats[name].append({"mean": float(rew.mean()), "std": float(rew.std()),
                                "max_adv": float(np.abs(rew[1:] - rew[0]).max())})

    summary = {"checkpoint": checkpoint, "rows": len(items), "groups": groups, "noise_std": noise_std,
               "noise_norm_over_r_norm": float(np.mean(norm_ratio)), "r_norm_mean": float(bank.norm(dim=-1).mean())}
    for name, s in stats.items():
        mean = np.array([x["mean"] for x in s])
        std = np.array([x["std"] for x in s])
        adv = np.array([x["max_adv"] for x in s])
        summary[name] = {"reward_mean": float(mean.mean()),
                         "within_group_std_mean": float(std.mean()),
                         "within_group_std_over_mean": float((std / np.maximum(mean, 1e-9)).mean()),
                         "max_abs_advantage_median": float(np.median(adv)),
                         "frac_groups_with_adv_gt_0.01": float((adv > 0.01).mean()),
                         "between_row_reward_std": float(mean.std())}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    fire.Fire(main)
