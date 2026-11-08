"""Variational latent for R3 (option A): a conditional VAE over the thought slot.

Plain R3 is a CVAE whose "posterior" never sees the target and whose KL is
identically zero: ``r = f(x)`` is a deterministic function of the history ``x``
the decoder also reads, so ``I(y; r | x) = 0`` by construction and the decoder
correctly ignores it.  Here ``r`` is given information the decoder's own path does
not have:

  prior      p(z | x)    = N(mu_p, sigma_p^2)   from the existing pooled r (first pass)
  posterior  q(z | x, y) = N(mu_q, sigma_q^2)   from the first-pass hidden states, query =
                                                 the last target token (training rows
                                                 already contain the target; no extra pass)
  decoder    p(y | x, up(z)),  z ~ q in training (a fraction of rows use the prior path),
                               z = mu_p at inference (or a seeded sample, CVAE_MODE=sample)

  loss = CE + beta_t * max(KL(q || p), free_nats) + w_aux * aux(z_q) + w_aux_p * aux(mu_p)

``aux`` is the bag-of-words-style auxiliary of Zhao et al. 2017 / McCarthy et al.
2020: predict the target item *from the latent alone* (cosine softmax over the
Qwen3 item-embedding table), so the latent must carry target information even when
the decoder could get it elsewhere.  ``beta_t`` follows a linear or cyclical warm-up
(Bowman 2016; Fu et al. 2019); ``free_nats`` is a row-level free-bits threshold.

Module layout (all new modules fp32; the pooled r stays the bf16 R3 pooling):
  model.attention = PriorLatentLayer(pool)   -> up(mu_p) at inference; experiment.py's
                                                extract / mean / shuffle interventions
                                                therefore act on the slot vector as before
  model.posterior = PosteriorEncoder
  model.aux       = AuxItemHead (table loaded from config at training time only)

Config: ``config.latent_cvae = {"dim": d, "posterior_dim": 256, "logvar_init": -2.0,
"token_norm": ..., "aux": {"file": ..., "emb_dim": 1536, "tau": 0.05} | None}``.
Training-time scalars (beta, free nats, prior_prob, aux weights) are attributes of
``CVAELatentModel`` set by the training script.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from layers import LatentModel
from reasoning_dataset import LatentRDataset

LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) per row (sum over dimensions), nats."""
    return 0.5 * ((logvar_p - logvar_q) + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1.0).sum(-1)


class PriorLatentLayer(nn.Module):
    """Wraps R3's pooling: pooled r -> LayerNorm -> (mu_p, logvar_p) -> slot = up(z)."""

    def __init__(self, pool, hidden_size, spec):
        super().__init__()
        self.pool = pool
        d = int(spec["dim"])
        self.norm = nn.LayerNorm(hidden_size)
        self.mu = nn.Linear(hidden_size, d)
        self.logvar = nn.Linear(hidden_size, d)
        self.up = nn.Linear(d, hidden_size)
        self.sample = False  # inference: mean (False) or a sample (True; set model.attention.generator)
        self.generator = None
        self.last = None  # (mu_p, logvar_p) of the last call

    @property
    def end_k(self):
        return self.pool.end_k

    @end_k.setter
    def end_k(self, value):
        self.pool.end_k = value

    def reset_parameters(self, logvar_init, token_norm):
        self.norm.reset_parameters()
        self.mu.reset_parameters()
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_init))
        # E||up(z)|| ~ token_norm for z ~ N(0, I): entries ~ N(0, token_norm^2 / (H d))
        nn.init.normal_(self.up.weight, std=float(token_norm) / math.sqrt(self.up.weight.numel()))
        nn.init.zeros_(self.up.bias)

    def heads(self, pooled):
        x = self.norm(pooled.float())
        return self.mu(x), self.logvar(x).clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        pooled = self.pool(hidden_states, attention_mask, thought_id_idx)
        mu, logvar = self.heads(pooled)
        self.last = (mu.detach(), logvar.detach())
        z = mu
        if self.sample:
            eps = torch.randn(mu.shape, generator=self.generator, device=mu.device, dtype=mu.dtype)
            z = mu + (0.5 * logvar).exp() * eps
        return self.up(z)


class PosteriorEncoder(nn.Module):
    """q(z | x, y): one attention head over the first-pass hidden states (history +
    target), query from the last valid (target) token; fp32."""

    def __init__(self, hidden_size, dim, d_latent):
        super().__init__()
        self.query = nn.Linear(hidden_size, dim)
        self.key = nn.Linear(hidden_size, dim)
        self.value = nn.Linear(hidden_size, dim)
        self.norm = nn.LayerNorm(dim)
        self.mu = nn.Linear(dim, d_latent)
        self.logvar = nn.Linear(dim, d_latent)
        self.scale = 1.0 / math.sqrt(dim)

    def reset_parameters(self, logvar_init):
        for layer in (self.query, self.key, self.value, self.mu):
            layer.reset_parameters()
        self.norm.reset_parameters()
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_init))

    def forward(self, hidden_states, attention_mask, last_idx):
        h = hidden_states.float()
        rows = torch.arange(h.size(0), device=h.device)
        q = self.query(h[rows, last_idx])  # [B, dim]
        k, v = self.key(h), self.value(h)  # [B, T, dim]
        scores = torch.einsum("bd,btd->bt", q, k) * self.scale
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bt,btd->bd", w, v)
        x = self.norm(pooled)
        return self.mu(x), self.logvar(x).clamp(LOGVAR_MIN, LOGVAR_MAX)


class AuxItemHead(nn.Module):
    """Predict the target item from the latent alone: cosine softmax over an item table."""

    def __init__(self, d_latent, emb_dim, tau=0.05):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_latent, 512), nn.GELU(), nn.Linear(512, emb_dim))
        self.tau = float(tau)
        self.register_buffer("table", torch.zeros(0, emb_dim), persistent=False)

    @property
    def ready(self):
        return self.table.numel() > 0

    def set_table(self, embeddings):
        self.table = F.normalize(embeddings.float().to(self.table.device), dim=-1)

    def logits(self, z):
        return F.normalize(self.net(z.float()), dim=-1) @ self.table.T / self.tau

    def forward(self, z, target):
        """(cross-entropy, hit@1, hit@10) for rows with a valid target index."""
        logits = self.logits(z)
        ok = (target >= 0) & (target < self.table.size(0))
        if not ok.any():
            zero = z.new_zeros(())
            return zero, zero, zero
        logits, target = logits[ok], target[ok]
        ce = F.cross_entropy(logits, target)
        top = logits.topk(min(10, logits.size(-1)), dim=-1).indices
        hit1 = (top[:, 0] == target).float().mean()
        hit10 = (top == target[:, None]).any(1).float().mean()
        return ce, hit1.detach(), hit10.detach()


def attach_cvae(model, spec):
    """Build the variational modules on a LatentModel (any class) from config.latent_cvae."""
    hidden = model.config.hidden_size
    model.attention = PriorLatentLayer(model.attention, hidden, spec)
    model.posterior = PosteriorEncoder(hidden, int(spec.get("posterior_dim", 256)), int(spec["dim"]))
    aux = spec.get("aux")
    model.aux = AuxItemHead(int(spec["dim"]), int(aux["emb_dim"]), float(aux.get("tau", 0.05))) if aux else None


def cvae_modules(model):
    mods = [model.attention.norm, model.attention.mu, model.attention.logvar, model.attention.up, model.posterior]
    if getattr(model, "aux", None) is not None:
        mods.append(model.aux)
    return mods


def init_cvae(model, spec):
    """Fresh fp32 initialisation (training from a base model)."""
    for m in cvae_modules(model):
        m.float()
    model.attention.reset_parameters(spec.get("logvar_init", -2.0), spec["token_norm"])
    model.posterior.reset_parameters(spec.get("logvar_init", -2.0))
    if getattr(model, "aux", None) is not None:
        for layer in model.aux.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def load_fp32_cvae_params(model, checkpoint):
    """``from_pretrained(torch_dtype=bf16)`` rounds every weight; reload the fp32 modules."""
    from safetensors import safe_open

    for m in cvae_modules(model):
        m.float()
    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    params = dict(model.named_parameters())
    prefixes = ("attention.norm.", "attention.mu.", "attention.logvar.", "attention.up.", "posterior.", "aux.")
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith(prefixes) and key in params:
                    with torch.no_grad():
                        params[key].copy_(f.get_tensor(key).float())
                    loaded += 1
    if files and loaded == 0:
        raise RuntimeError(f"no variational-latent parameters found in {checkpoint}")


def per_example_nll(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1),
                                ignore_index=-100, reduction="none").view(shift_labels.size())
    mask = (shift_labels != -100).float()
    return (token_nll * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class CVAELatentModel(LatentModel):
    """Training class: ELBO-style objective in the prefill-with-labels branch; every
    other call (evaluation loss, ranking, generation) is the plain LatentModel path
    with the slot = up(mu_p)."""

    def __init__(self, config):
        super().__init__(config)
        self.kl_beta = 1.0
        self.kl_free_nats = 0.0
        self.prior_prob = 0.0
        self.aux_weight = 0.0
        self.aux_prior_weight = 0.0
        self.grad_accum_steps = 1  # see margin_loss_attention.py: forward has **kwargs
        self.last_stats = {}

    def _thought_positions(self, input_ids):
        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_token_id)
        if where.size(0) != input_ids.size(0):
            raise ValueError("variational latent expects exactly one thought token per row")
        return where[:, 1]

    def encode(self, input_ids, attention_mask):
        """First pass: (mu_p, logvar_p, mu_q, logvar_q, thought_idx)."""
        h = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        thought_idx = self._thought_positions(input_ids)
        pooled = self.attention.pool(h, self.pooling_mask(input_ids, attention_mask), thought_idx)
        mu_p, logvar_p = self.attention.heads(pooled)
        last_idx = attention_mask.long().sum(1) - 1  # right padding in training
        mu_q, logvar_q = self.posterior(h, attention_mask, last_idx)
        return mu_p, logvar_p, mu_q, logvar_q, thought_idx

    def decode_nll(self, input_ids, attention_mask, labels, thought_idx, r):
        """Per-row target NLL with ``r`` in the slot (bottleneck / history dropout applied)."""
        embs = self.model.embed_tokens(input_ids).clone()
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        embs[rows, thought_idx] = r.to(embs.dtype)
        out = LatentModel.forward(self, inputs_embeds=embs, attention_mask=attention_mask, use_cache=False,
                                  latent_bottleneck_ids=input_ids, return_dict=True)
        return per_example_nll(out.logits, labels), out

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        target_item = kwargs.pop("target_item", None)
        is_prefill = (input_ids is not None and kwargs.get("inputs_embeds") is None and attention_mask is not None
                      and input_ids.size() == attention_mask.size())
        if not (is_prefill and self.training and labels is not None):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        kwargs.pop("num_items_in_batch", None)

        mu_p, logvar_p, mu_q, logvar_q, thought_idx = self.encode(input_ids, attention_mask)
        eps = torch.randn_like(mu_q)
        z_q = mu_q + (0.5 * logvar_q).exp() * eps
        z_p = mu_p + (0.5 * logvar_p).exp() * eps
        use_prior = torch.rand(mu_q.size(0), device=mu_q.device) < self.prior_prob
        z = torch.where(use_prior[:, None], z_p, z_q)
        nll, out = self.decode_nll(input_ids, attention_mask, labels, thought_idx, self.attention.up(z))
        ce = nll.mean()

        kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        kl_loss = kl.clamp(min=float(self.kl_free_nats)).mean()
        total = ce + self.kl_beta * kl_loss
        stats = {"ce": ce.detach(), "kl": kl.detach().mean(), "kl_beta": torch.tensor(float(self.kl_beta)),
                 "sigma_q": (0.5 * logvar_q).exp().detach().mean(), "sigma_p": (0.5 * logvar_p).exp().detach().mean(),
                 "mu_gap": (mu_q - mu_p).detach().norm(dim=-1).mean()}
        if getattr(self, "aux", None) is not None and self.aux.ready and target_item is not None:
            if self.aux_weight > 0:
                aux_q, h1, h10 = self.aux(z_q, target_item)
                total = total + self.aux_weight * aux_q
                stats.update(aux_q=aux_q.detach(), aux_q_hit10=h10)
            if self.aux_prior_weight > 0:
                aux_p, _, h10p = self.aux(mu_p, target_item)
                total = total + self.aux_prior_weight * aux_p
                stats.update(aux_p=aux_p.detach(), aux_p_hit10=h10p)
        self.last_stats = stats
        return CausalLMOutputWithPast(loss=total / self.grad_accum_steps, logits=out.logits)


@torch.no_grad()
def cvae_probe(model, batch):
    """Rate / oracle-gap diagnostics on one right-padded training-format batch:
    KL(q||p), ||mu_q - mu_p||, teacher-forced NLL with z = mu_q (posterior oracle),
    z = mu_p (inference), z = 0 (mask), and aux hit@10 from mu_q / mu_p."""
    was_training = model.training
    model.eval()
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    mu_p, logvar_p, mu_q, logvar_q, thought_idx = model.encode(ids, mask)
    out = {"kl": gaussian_kl(mu_q, logvar_q, mu_p, logvar_p).mean().item(),
           "mu_gap": (mu_q - mu_p).norm(dim=-1).mean().item(),
           "sigma_q": (0.5 * logvar_q).exp().mean().item(), "sigma_p": (0.5 * logvar_p).exp().mean().item()}
    for name, z in (("nll_posterior", mu_q), ("nll_prior", mu_p), ("nll_mask", None)):
        r = model.attention.up(z) if z is not None else torch.zeros(ids.size(0), model.config.hidden_size, device=ids.device)
        out[name] = model.decode_nll(ids, mask, labels, thought_idx, r)[0].mean().item()
    out["nll_shuffle"] = model.decode_nll(ids, mask, labels, thought_idx, model.attention.up(mu_p.roll(1, 0)))[0].mean().item()
    if getattr(model, "aux", None) is not None and model.aux.ready and "target_item" in batch:
        _, _, out["aux_hit10_posterior"] = (x.item() if torch.is_tensor(x) else x for x in model.aux(mu_q, batch["target_item"]))
        _, _, out["aux_hit10_prior"] = (x.item() if torch.is_tensor(x) else x for x in model.aux(mu_p, batch["target_item"]))
    model.train(was_training)
    return out


class KLSchedule(TrainerCallback):
    """beta_t = beta * min(1, phase / warmup); linear (one phase over training) or cyclical."""

    def __init__(self, model, beta, warmup, schedule="linear", cycles=4):
        self.model, self.beta, self.warmup, self.schedule, self.cycles = model, float(beta), float(warmup), schedule, int(cycles)

    def factor(self, progress):
        if self.schedule == "cyclical":
            progress = (progress * self.cycles) % 1.0
        elif self.schedule != "linear":
            raise ValueError(f"unknown kl schedule {self.schedule}")
        return 1.0 if self.warmup <= 0 else min(1.0, progress / self.warmup)

    def on_step_begin(self, args, state, control, **kwargs):
        progress = state.global_step / max(1, state.max_steps)
        self.model.kl_beta = self.beta * self.factor(progress)


class CVAELogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return
        target = model.module if hasattr(model, "module") else model
        for k, v in getattr(target, "last_stats", {}).items():
            logs[f"cvae_{k}"] = float(v)


class CVAEDiagnostics(TrainerCallback):
    """cvae_probe on fixed validation rows at every evaluation; appended to
    <output_dir>/cvae_diagnostics.json."""

    def __init__(self, rows, collator, output_dir, batch_size=8):
        self.rows, self.collator, self.path, self.bs = rows, collator, os.path.join(output_dir, "cvae_diagnostics.json"), batch_size

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        import json
        target = model.module if hasattr(model, "module") else model
        device = next(target.parameters()).device
        sums, n = {}, 0
        for s in range(0, len(self.rows), self.bs):
            batch = self.collator(self.rows[s:s + self.bs])
            batch = {k: v.to(device) for k, v in batch.items()}
            stats = cvae_probe(target, batch)
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
        record = {"epoch": state.epoch, "step": state.global_step, **{k: v / n for k, v in sums.items()}}
        print("[cvae] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        history = json.load(open(self.path)) if os.path.exists(self.path) else []
        history.append(record)
        with open(self.path, "w") as f:
            json.dump(history, f, indent=2)


class TargetRowsDataset(LatentRDataset):
    """Plain R3 rows plus ``target_item`` (dataset item id; index into the aux table)."""

    def pre(self, idx):
        item = super().pre(idx)
        item["target_item"] = int(self.data.iloc[idx]["item_id"])
        return item


class TargetCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        targets = [int(f.pop("target_item")) for f in features]
        batch = self.base(features)
        batch["target_item"] = torch.tensor(targets, dtype=torch.long)
        return batch
