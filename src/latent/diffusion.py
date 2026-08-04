"""DiffuReason-style diffusion refinement modules (arXiv 2602.09744).

Shared by stage-1 SFT (src/latent) and stage-2 GRPO (src/grpo_attention_tuning).
All sigma / log-density math is kept in fp32 for bf16 training stability.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar_t = cos^2(((t/T + s) / (1 + s)) * pi/2), t = 0..T (Eq. 17)."""
    t = torch.arange(T + 1, dtype=torch.float32)
    return torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal encoding of integer timesteps t [B] -> [B, dim] (fp32)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal encoding followed by a 2-layer SiLU MLP -> [B, hidden_size]."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.hidden_size)
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


class DenoiseMLP(nn.Module):
    """3-layer SiLU MLP denoiser: [c, x_t + t_emb] (2d) -> 4d -> 4d -> 2d = (mu, log_sigma)."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        d = hidden_size
        self.net = nn.Sequential(
            nn.Linear(2 * d, 4 * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, 4 * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, 2 * d),
        )

    def forward(self, x_t, t_emb, c):
        model_dtype = x_t.dtype
        out = self.net(torch.cat([c, x_t + t_emb], dim=-1))
        mu, log_sigma = out.float().chunk(2, dim=-1)
        sigma = (F.softplus(log_sigma) + 1e-5).clamp(1e-3, 1.0)
        return mu.to(model_dtype), sigma  # sigma stays fp32


class AttentionPooling(nn.Module):
    """Learnable-query attention pooling over the R thinking tokens -> condition c."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) * (hidden_size ** -0.5))
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: [B, R, d] -> c: [B, d]
        K = self.key(taus)
        V = self.value(taus)
        scores = (K @ self.query.to(K.dtype)) * self.scale  # [B, R]
        weights = F.softmax(scores.float(), dim=-1).to(V.dtype)
        return (weights.unsqueeze(1) @ V).squeeze(1)


def gaussian_log_prob(z, mu, sigma):
    """Diagonal-Gaussian log density summed over dims, constant dropped (Eq. 20). fp32."""
    z, mu, sigma = z.float(), mu.float(), sigma.float()
    return -0.5 * ((((z - mu) / sigma) ** 2) + 2 * sigma.log()).sum(dim=-1)


def reverse_process(
    denoiser: DenoiseMLP,
    time_embed: TimestepEmbedding,
    tau_R: torch.Tensor,
    c: torch.Tensor,
    T: int,
    alpha_bar: torch.Tensor,
    training: bool,
    use_schedule_scaling: bool = True,
    eps_init: torch.Tensor = None,
    noises: list = None,
    generator: torch.Generator = None,
):
    """Run the learned reverse process from x_T = tau_R + eps_init (Eq. 3-5).

    training=True (or noises replayed): intermediate steps sample reparameterized
    x_{t-1} = mu + scale_t * sigma * eps_t; each eps_t is recorded so a rollout
    trajectory can be replayed exactly in a gradient pass (stage-2 GRPO).
    training=False and no replay: deterministic mean-following (Appendix C.1).

    Returns (mu, sigma, used_noises) where used_noises = [eps_init, eps_T, ..., eps_2]
    (no noise is consumed at the final t=1 step).
    """
    replay = noises is not None
    if replay:
        eps_init = noises[0]
        step_noises = noises[1:]
    else:
        if eps_init is None:
            eps_init = torch.randn(
                tau_R.shape, device=tau_R.device, dtype=torch.float32, generator=generator
            ).to(tau_R.dtype)
        step_noises = None

    used_noises = [eps_init]
    x_t = tau_R + eps_init
    mu, sigma = None, None
    for i, t in enumerate(range(T, 0, -1)):
        t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        t_emb = time_embed(t_tensor)
        mu, sigma = denoiser(x_t, t_emb, c)
        if t > 1:
            if replay:
                eps_t = step_noises[i]
            elif training:
                eps_t = torch.randn(
                    mu.shape, device=mu.device, dtype=torch.float32, generator=generator
                )
            else:
                eps_t = None
            if eps_t is not None:
                scale = alpha_bar[t - 1].mul(-1).add(1).clamp(min=0).sqrt() if use_schedule_scaling else 1.0
                x_t = mu + (scale * sigma * eps_t.float()).to(mu.dtype)
                used_noises.append(eps_t)
            else:
                x_t = mu
    return mu, sigma, used_noises
