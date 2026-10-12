"""HiLaR stage 2: hierarchical latent alignment fine-tuning on R3 (document/hilar.md).

Re-implements "Hierarchical Latent Alignment Fine-tuning" of HiLaR (Hu et al., arXiv
2607.27760, Sec. 3.2):

    L_SFT = L_rec + lambda_align * (1/K) sum_k ||Proj(tau_k) - e_{u,k}||^2,  lambda_align = 0.1

* tau_k, the k-th continuous latent state, comes from the K-step latent machinery of
  ``LaRecLatentModel``: tau_k = head(state before slot k), and slot k's input is its soft
  token, so every step conditions on the history and all earlier steps (HiLaR's
  mu(H_u, tau^{<k}); the paper does not say how states are fed back).
* e_{u,k} is the frozen stage-1 quantizer's cumulative code embedding for the row
  (``src/latent_distill/hilar_quantizer.py``), in its standardised space.
* Proj is a learned linear map.  CHOICE: the squared error is averaged over dimensions
  (summing over 1,536 dimensions would make lambda = 0.1 dominate the CE); the alignment
  term is added in training only and weighted by the micro-batch token share.
* LaRec's own auxiliary losses are off (``latent_larec`` alpha = gamma = 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from larec_latent import LaRecLatentModel
from layers import LatentModel


class HiLaRLatentModel(LaRecLatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_hilar", None)
        if not spec:
            raise ValueError("latent_hilar config is required")
        self.align_weight = float(spec.get("lambda_align", 0.1))
        self.align_proj_q = nn.Linear(config.hidden_size, int(spec["target_dim"]))
        self.targets = None  # [N, K, D] standardised cumulative code embeddings, CPU
        # Stage 3 (hilar_grpo.py) controls, all off for SFT and evaluation:
        self.rollout_sigma = None  # [K]: sample tau_k = mu_k + sigma_k * eps_k
        self.rollout_eps = None  # [N, K, H] noise for the next prompt pass
        self.replay_tau = None  # [N, K, H]: feed these latents; mu_k recomputed (grad only via the head)
        self.keep_levels = None  # int: slots after the first `keep_levels` get the plain thought embedding
        self.last_chain = None

    def generate_embs(self, input_ids, attention_mask):
        """LaRec's K-step chain with HiLaR's stochastic policy: tau_k = mu_k + sigma_k eps_k,
        mu_k = head(state before slot k); slot k's input is the soft token of tau_k."""
        pos = self.slot_positions(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        embeds = self.model.embed_tokens(input_ids)
        empty = embeds[rows, pos[:, 0]].clone()
        mus, taus, states, inputs = [], [], [], []
        for k in range(self.k):
            if self.keep_levels is not None and k >= self.keep_levels:
                embeds = embeds.index_put((rows, pos[:, k]), empty)
                continue
            upto = int(pos[:, k].max())
            if self.replay_tau is not None:
                with torch.no_grad():
                    hidden = self.model(inputs_embeds=embeds[:, :upto], attention_mask=attention_mask[:, :upto]).last_hidden_state
            else:
                hidden = self.model(inputs_embeds=embeds[:, :upto], attention_mask=attention_mask[:, :upto]).last_hidden_state
            state = hidden[rows, pos[:, k] - 1]
            if k > 0:
                states.append(state)
            mu = self.head(state)
            inputs.append(state)
            if self.replay_tau is not None:
                tau = self.replay_tau[:, k].to(mu.dtype)
            elif self.rollout_sigma is not None:
                tau = mu + float(self.rollout_sigma[k]) * self.rollout_eps[:, k].to(mu.dtype)
            else:
                tau = mu
            z = self.soft_token(tau)
            embeds = embeds.index_put((rows, pos[:, k]), z.to(embeds.dtype))
            mus.append(mu)
            taus.append(tau)
        if taus:
            heads = torch.stack(taus, 1)
            if self.swap_shift:
                swapped = torch.stack([self.soft_token(t) for t in taus], 1).roll(self.swap_shift, dims=0)
                for k in range(len(taus)):
                    embeds = embeds.index_put((rows, pos[:, k]), swapped[:, k].to(embeds.dtype))
            self._trace = {"pos": pos, "heads": heads, "mus": torch.stack(mus, 1), "states": states,
                           "head_inputs": torch.stack(inputs, 1)}
            self.last_chain = self._trace  # survives later forward calls (e.g. decoding steps of generate)
        return embeds

    def set_targets(self, e):
        if e.shape[1] != self.k:
            raise ValueError(f"targets have {e.shape[1]} levels, config says {self.k}")
        self.targets = e

    def reset_projection(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            # CHOICE: near-zero init. tau has unnormalised final-layer scale (a few huge dimensions);
            # a standard init gave an initial alignment MSE of ~22 against unit-variance targets.
            w = torch.empty(self.align_proj_q.weight.shape).normal_(0, 1e-3 * self.align_proj_q.in_features ** -0.5, generator=g)
            self.align_proj_q.weight.copy_(w.to(self.align_proj_q.weight))
            self.align_proj_q.bias.zero_()

    def forward(self, larec_row=None, **kwargs):
        self._trace = None
        output = LatentModel.forward(self, **kwargs)  # K-step latents via generate_embs; no LaRec losses
        if not (self.training and larec_row is not None and kwargs.get("labels") is not None and self._trace):
            return output
        if self.targets is None:
            raise ValueError("alignment targets have not been set")
        tau = self._trace["heads"]  # [B, K, H]
        target = self.targets[larec_row.long().cpu()].to(tau.device).float()
        pred = self.align_proj_q(tau.to(self.align_proj_q.weight.dtype)).float()
        loss_align = F.mse_loss(pred, target)
        output.loss = output.loss + self.align_weight * self._accumulation_share(kwargs) * loss_align
        self._log_align(loss_align.detach())
        return output

    def _log_align(self, value, every=200):
        self._stats.append(value.float().cpu())
        if len(self._stats) >= every:
            print(f"[hilar] align_mse={torch.stack(self._stats).mean().item():.4f} (target variance ~1)", flush=True)
            self._stats = []
