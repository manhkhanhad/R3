"""WIDER spectral regulariser (arXiv 2609.07406) adapted to R3 with one thought (K = 1).

WIDER penalises the m latent states of one example for lying along their own
stop-gradient mean direction:  L_spec = (1/m) sum_i (z_i . q)^2,  z_i = h_i/||h_i||,
q = stopgrad(normalize(sum_i z_i)),  L = L_base + lambda * L_spec  (inference unchanged).

R3 has one latent per example, so the within-example form is a constant (= 1, no
gradient).  Here the "states" are the pooled latents r of different users and q is
the shared direction across users (R3's collapse is across users: on Toys r is the
same vector for everyone).  With a micro-batch of 4, q is an exponential moving
average of the batch mean unit direction (stop-gradient):

    z_b = r_b / ||r_b||,   q = normalize(ema),   L_spec = mean_b (z_b . q)^2
    ema <- m * ema + (1 - m) * mean_b z_b          (after the step, no grad)

Applied to the pooled r before it is spliced in (training only; eval loss and
inference are plain R3).  Config: ``config.latent_wider = {"weight": lambda,
"momentum": m}``.
"""

import torch
import torch.nn.functional as F


class WiderState:
    def __init__(self, weight, momentum=0.99):
        self.weight, self.momentum = float(weight), float(momentum)
        self.ema = None
        self.latent = None          # pooled r of the current training forward (with grad)
        self.calls = 0
        self.last = {}

    def penalty(self):
        r = self.latent
        self.latent = None
        if r is None:
            return None
        z = F.normalize(r.float(), dim=-1)
        batch_mean = z.mean(0).detach()
        if self.ema is None:
            self.ema = batch_mean.clone()
        q = F.normalize(self.ema, dim=0)
        cos = z @ q
        pen = (cos ** 2).mean()
        self.ema = self.momentum * self.ema + (1 - self.momentum) * batch_mean
        self.calls += 1
        self.last = {"wider_penalty": float(pen.detach()), "wider_cos_mean": float(cos.detach().mean()),
                     "wider_ema_norm": float(self.ema.norm())}
        if self.calls % 100 == 1:
            print(f"[wider] call {self.calls}: penalty {self.last['wider_penalty']:.4f}, "
                  f"mean cos to shared direction {self.last['wider_cos_mean']:.4f}, "
                  f"|ema| {self.last['wider_ema_norm']:.4f}", flush=True)
        return pen


def wider_state(model):
    """The model's WiderState (created on first use from config.latent_wider), or None."""
    spec = getattr(model.config, "latent_wider", None)
    if not spec or float(spec.get("weight", 0)) <= 0:
        return None
    state = getattr(model, "_wider", None)
    if state is None:
        state = WiderState(spec["weight"], spec.get("momentum", 0.99))
        model._wider = state
    return state
