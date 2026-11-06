"""Geometry of a bank of latents r (experiment.py extract -> thoughts.pt), WIDER's measures.

Usage: latent_rank.py <thoughts.pt> [<output.json>]
Reports on the unit-normalised vectors z = r/||r||:
  effective rank  exp(entropy of sigma_i^2 / sum sigma^2)   (uncentred, as in WIDER)
  top-1 energy    sigma_1^2 / sum sigma^2
  mean cos to the mean direction, mean pairwise cos (sampled)
and on raw r: residual share ||r - mean r|| / ||r|| (per-user part), mean norm.
"""

import json
import sys

import torch
import torch.nn.functional as F

r = torch.load(sys.argv[1], map_location="cpu", weights_only=True).float()
r = r.reshape(r.shape[0], -1, r.shape[-1])[:, 0]
z = F.normalize(r, dim=-1)
s = torch.linalg.svdvals(z)
p = s ** 2 / (s ** 2).sum()
out = {
    "n": int(r.shape[0]),
    "effective_rank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())),
    "top1_energy": float(p[0]),
    "mean_cos_to_mean_direction": float((z @ F.normalize(z.mean(0), dim=0)).mean()),
    "residual_share": float(((r - r.mean(0)).norm(dim=-1) / r.norm(dim=-1)).mean()),
    "mean_norm": float(r.norm(dim=-1).mean()),
}
g = torch.Generator().manual_seed(0)
i, j = torch.randint(len(z), (20000,), generator=g), torch.randint(len(z), (20000,), generator=g)
keep = i != j
out["mean_pairwise_cos"] = float((z[i[keep]] * z[j[keep]]).sum(-1).mean())
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
