"""Diffusion-refined R3 thought: r' = g * normalize(P(diffuse(r))), prompt unchanged.

R3's pooled thought ``r`` conditions a small x0-predicting denoiser (DreamRec /
DiffSBR style) whose target is the *target item's* vector from an
item_oracle_prep.py table (``r3title``: R3's own title representation).  The
thought slot then carries only the refined vector:

    x0      = sqrt(d) * normalize(v_target - mu)          (centred item space)
    x_t     = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps
    x0_hat  = f(x_t, t, c(r))
    r'      = g * normalize(P(x0_hat))

Training loss = next-token NLL (through r', so the decoder trains f and r)
+ ``weight`` * (||x0_hat - x0||^2 / d + ``ce_weight`` * catalogue CE of
x0_hat against all item vectors).  What the decoder sees in training:
``x0_hat`` from one denoising step at a random t, or with probability
``chain_prob`` the full reverse chain from noise (gradient through the last
step only), which is what it sees at test time.  At test time the chain
starts from a fixed seeded noise vector shared by every row, so beam copies of
a prompt get the same r'.

``mode = "regress"`` is the deterministic control: x0_hat = f(0, 0, c(r)),
trained with the same losses.

Evaluation overrides (``x0_override``): a [B, d] tensor replaces x0_hat
(oracle / shuffle / mean interventions in latent_attention_eval.py).
New modules are fp32 (prefix ``diffuse.``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

from layers import LatentModel


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Denoiser(nn.Module):
    """f(x_t, t, c) -> x0_hat (MLP with residual blocks)."""

    def __init__(self, hidden_size, dim, width, depth):
        super().__init__()
        self.dim = dim
        self.cond = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.x_in = nn.Linear(dim, width)
        self.t_in = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
            for _ in range(depth)
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        # P: refined vector -> thought slot (hidden size); g: its norm
        self.proj = nn.Linear(dim, hidden_size)
        self.gain = nn.Parameter(torch.tensor(30.0))

    def forward(self, x_t, t, c):
        h = self.x_in(x_t) + self.t_in(timestep_embedding(t, 256)) + c
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class DiffusionRefineModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_diffusion_refine
        self.spec = spec
        dim = int(spec["dimension"])
        self.diffuse = Denoiser(config.hidden_size, dim, int(spec.get("width", 2048)), int(spec.get("depth", 3)))
        steps = int(spec.get("steps", 32))
        # DDPM linear schedule rescaled to T steps (Nichol & Dhariwal), capped below 1
        scale = 1000 / steps
        betas = torch.linspace(float(spec.get("beta_start", 1e-4)) * scale, float(spec.get("beta_end", 0.02)) * scale,
                               steps, dtype=torch.float64).clamp(max=0.999)
        abar = torch.cumprod(1 - betas, 0)
        self.register_buffer("abar", abar.float(), persistent=False)          # abar[t-1] for t = 1..T
        self.register_buffer("item_x0", torch.empty(0, dim), persistent=False)  # [n_items, d], x0 space
        self.register_buffer("item_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self._target = None
        self.x0_override = None
        self.last_x0 = None
        self.last_aux = None
        self.last_aux_parts = None

    # ------------------------------------------------------------------ setup
    def set_item_table(self, embeddings, valid):
        """Raw item vectors -> centred, sqrt(d)-scaled unit vectors (the x0 space)."""
        embeddings = torch.as_tensor(embeddings).float()
        valid = torch.as_tensor(valid).bool()
        mu = embeddings[valid].mean(0)
        x0 = F.normalize(embeddings - mu, dim=-1) * math.sqrt(embeddings.shape[1])
        x0[~valid] = 0
        device = self.model.embed_tokens.weight.device
        self.item_x0, self.item_valid = x0.to(device), valid.to(device)

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        from category_latent import load_fp32_params

        model = super().from_pretrained(path, *args, **kwargs)
        model.diffuse.float()
        load_fp32_params(model, str(path), "diffuse.")
        return model

    # -------------------------------------------------------------- diffusion
    @property
    def steps(self):
        return self.abar.numel()

    def denoise(self, x_t, t, c):
        if self.spec.get("mode", "diffusion") == "regress":
            zero_t = torch.zeros(c.size(0), dtype=torch.long, device=c.device)
            return self.diffuse(torch.zeros_like(x_t), zero_t, c)
        return self.diffuse(x_t, t, c)

    def sample(self, c, grad_last=False):
        """Deterministic DDIM chain (x0-prediction) from a fixed seeded noise vector."""
        d = self.item_x0.shape[1]
        if self.spec.get("mode", "diffusion") == "regress":
            return self.denoise(torch.zeros(c.size(0), d, device=c.device), None, c)
        g = torch.Generator(device="cpu").manual_seed(int(self.spec.get("eval_seed", 0)))
        x = torch.randn(d, generator=g).to(c.device).expand(c.size(0), d)
        for step in range(self.steps, 0, -1):
            t = torch.full((c.size(0),), step, device=c.device, dtype=torch.long)
            with torch.set_grad_enabled(grad_last and step == 1 and torch.is_grad_enabled()):
                x0 = self.denoise(x, t, c)
            if step == 1:
                return x0
            a, a_prev = self.abar[step - 1], self.abar[step - 2]
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = (a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps).detach()

    def to_slot(self, x0):
        return F.normalize(self.diffuse.proj(x0), dim=-1) * self.diffuse.gain

    # ------------------------------------------------------------------ hooks
    def observe_reasoning_latent(self, latent):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            c = self.diffuse.cond(latent.float())
            if self.x0_override is not None:
                over = self.x0_override.to(c.device).float()
                x0_hat = over.repeat_interleave(c.size(0) // over.size(0), 0)
            elif self.training and self._target is not None:
                x0_hat = self._training_x0(c, self._target.to(c.device))
            else:
                x0_hat = self.sample(c)
            self.last_x0 = x0_hat.detach()
            return self.to_slot(x0_hat).to(latent.dtype)

    def _training_x0(self, c, target):
        n = self.item_x0.shape[0]
        ok = (target < n) & self.item_valid[target.clamp(max=n - 1)]
        x0 = self.item_x0[target.clamp(max=n - 1)]
        t = torch.randint(1, self.steps + 1, (c.size(0),), device=c.device)
        a = self.abar[t - 1][:, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
        x0_hat = self.denoise(x_t, t, c)

        mse = ((x0_hat - x0) ** 2).mean(-1)
        logits = F.normalize(x0_hat, dim=-1) @ F.normalize(self.item_x0, dim=-1).T / float(self.spec.get("tau", 0.05))
        logits = logits.masked_fill(~self.item_valid[None], float("-inf"))
        ce = F.cross_entropy(logits, target.clamp(max=n - 1), reduction="none")
        okf = ok.float()
        denom = okf.sum().clamp(min=1)
        mse, ce = (mse * okf).sum() / denom, (ce * okf).sum() / denom
        self.last_aux = mse + float(self.spec.get("ce_weight", 1.0)) * ce
        self.last_aux_parts = {"mse": mse.detach(), "ce": ce.detach()}

        # decoder input: one-step x0_hat, or the test-time chain for some rows (and rows without a target)
        use_chain = (torch.rand(c.size(0), device=c.device) < float(self.spec.get("chain_prob", 0.5))) | ~ok
        if use_chain.any():
            chained = self.sample(c[use_chain], grad_last=True)
            x0_hat = x0_hat.clone()
            x0_hat[use_chain] = chained
        return x0_hat

    def forward(self, side_index=None, **kwargs):
        # side_index = target item (training only; never used outside training)
        self._target = side_index if self.training else None
        self.last_aux = None
        try:
            output = super().forward(**kwargs)
        finally:
            self._target = None
        if self.training and self.last_aux is not None and kwargs.get("labels") is not None:
            output.loss = output.loss + float(self.spec.get("weight", 1.0)) * self.last_aux
        return output


class AuxLogCallback(TrainerCallback):
    """Adds the latest diffusion loss parts (mse, ce) to the Trainer logs."""

    def __init__(self, model):
        self.model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        parts = getattr(self.model, "last_aux_parts", None)
        if logs is not None and parts:
            logs.update({f"diffusion_{k}": float(v) for k, v in parts.items()})


def nearest_item_metrics(x0_hat, target, item_x0, item_valid, ks=(1, 3, 5, 10, 20)):
    """HR/NDCG of the nearest catalogue items (cosine in x0 space) to x0_hat."""
    sim = F.normalize(x0_hat.float(), dim=-1) @ F.normalize(item_x0.float(), dim=-1).T
    sim[:, ~item_valid] = -2.0
    rank = (sim > sim.gather(1, target[:, None])).sum(1)  # 0-based rank of the target
    out = {}
    for k in ks:
        hit = (rank < k).float()
        out[f"HR@{k}"] = hit.mean().item()
        out[f"NDCG@{k}"] = (hit / torch.log2(rank.float() + 2)).mean().item()
    out["centred_cos"] = F.cosine_similarity(x0_hat.float(), item_x0[target].float(), dim=-1).mean().item()
    return out
