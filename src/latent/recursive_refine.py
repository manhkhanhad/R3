"""Recursively refined R3 thought (TRM, Jolicoeur-Martineau 2025, arXiv 2510.04871).

R3's pooled thought ``r`` is the question x.  A tiny shared network keeps an
answer ``y`` (in the centred r3title item space of diffusion_refine.py) and a
latent scratchpad ``z``; one *cycle* is

    z <- net(z + c(r) + Y(y))     n_latent times     (latent reasoning, sees x)
    y <- y + out(net(z + Y(y)))   once                (answer update, no x)

Deep supervision: y is scored after every cycle (MSE + catalogue CE against the
target item's vector, as in diffusion_refine.py) and (y, z) are detached before
the next cycle, so each cycle gets its own gradient and memory stays one cycle
deep.  The thought slot carries only the final answer:

    r' = g * normalize(P(y_C))

and the next-token NLL backpropagates through the last cycle into c(r) and r.
At test time the number of cycles can differ from training
(``x0_override``-free path, ``eval_cycles``); every cycle's y is kept in
``last_cycles`` for the per-cycle nearest-item readout.

``cycles = 1, n_latent = 1`` is the one-shot control with the same network.
Reuses DiffusionRefineModel (item table, slot projection, aux scaling, eval
overrides); new modules are fp32 (prefix ``diffuse.``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_refine import DiffusionRefineModel


class RecursiveCore(nn.Module):
    """Tiny shared network (TRM uses 2 layers) plus the y/x/z interfaces and the slot projection."""

    def __init__(self, hidden_size, dim, width, depth):
        super().__init__()
        self.dim = dim
        self.cond = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.y_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(width)
        self.y_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        self.y_init = nn.Parameter(torch.zeros(dim))
        self.z_init = nn.Parameter(torch.zeros(width))
        # P: answer -> thought slot (hidden size); g: its norm (same as diffusion_refine.Denoiser)
        self.proj = nn.Linear(dim, hidden_size)
        self.gain = nn.Parameter(torch.tensor(30.0))

    def net(self, h):
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)

    def cycle(self, y, z, c, n_latent):
        for _ in range(n_latent):
            z = self.net(z + c + self.y_in(y))
        y = y + self.y_out(self.net(z + self.y_in(y)))
        return y, z


class RecursiveRefineModel(DiffusionRefineModel):
    def __init__(self, config):
        super().__init__(config)
        spec = self.spec
        self.diffuse = RecursiveCore(config.hidden_size, int(spec["dimension"]),
                                     int(spec.get("width", 2048)), int(spec.get("depth", 2)))
        self.eval_cycles = None   # test-time override of the number of cycles
        self.last_cycles = None   # [C, B, d] answers of every cycle (last forward)

    def reset_refiner(self):
        for m in self.diffuse.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.reset_parameters()
        with torch.no_grad():
            self.diffuse.y_init.zero_()
            self.diffuse.z_init.zero_()
            self.diffuse.gain.fill_(30.0)

    def recurse(self, c, cycles):
        core = self.diffuse
        n_latent = int(self.spec.get("n_latent", 6))
        y = core.y_init.expand(c.size(0), -1)
        z = core.z_init.expand(c.size(0), -1)
        answers = []
        for _ in range(cycles):
            y, z = core.cycle(y, z, c, n_latent)
            answers.append(y)
            y, z = y.detach(), z.detach()   # deep supervision: gradient within a cycle only
        return answers

    def sample(self, c, grad_last=False):
        cycles = self.eval_cycles or int(self.spec.get("cycles", 8))
        with torch.set_grad_enabled(grad_last and torch.is_grad_enabled()):
            answers = self.recurse(c, cycles)
        self.last_cycles = torch.stack([a.detach() for a in answers])
        return answers[-1]

    def _training_x0(self, c, target):
        n = self.item_x0.shape[0]
        ok = (target < n) & self.item_valid[target.clamp(max=n - 1)]
        safe = target.clamp(max=n - 1)
        x0 = self.item_x0[safe]
        answers = self.recurse(c, int(self.spec.get("cycles", 8)))
        okf = ok.float()
        denom = okf.sum().clamp(min=1)
        item = F.normalize(self.item_x0, dim=-1)
        tau = float(self.spec.get("tau", 0.05))
        mses, ces = [], []
        for y in answers:
            mse = ((y - x0) ** 2).mean(-1)
            logits = (F.normalize(y, dim=-1) @ item.T / tau).masked_fill(~self.item_valid[None], float("-inf"))
            ce = F.cross_entropy(logits, safe, reduction="none")
            mses.append(torch.where(ok, mse, 0.0).sum() / denom)   # where, not *: invalid targets give inf CE
            ces.append(torch.where(ok, ce, 0.0).sum() / denom)
        mse, ce = torch.stack(mses).mean(), torch.stack(ces).mean()
        self.last_aux = mse + float(self.spec.get("ce_weight", 1.0)) * ce
        self.last_aux_parts = {"mse": mse.detach(), "ce": ce.detach(),
                               "ce_first": ces[0].detach(), "ce_last": ces[-1].detach()}
        self.last_cycles = torch.stack([a.detach() for a in answers])
        return answers[-1]
