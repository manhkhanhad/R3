"""Item-transition reading latent (prompt unchanged, one thought slot).

R3's pooled latent r reads the row's item-transition neighbours (transition_prep.py:
items other users bought right after the row's last history items, time-causal)
and the result is added to r:

    q      = W_q r                                     (d = 256)
    z_n    = standardise_n(q . W_k normalize(e_n) / sqrt(d))   (over the row's neighbours)
    s_n    = tau * z_n + beta * log(1 + c_n) + b[src_n]
    a      = softmax_n(s_n)            (masked over the row's neighbours)
    m      = sum_n a_n e_n
    r'     = r + g * normalize(P m)                    (zero when the set is empty)

e_n is R3's title representation of item n (candidate_title_vectors.py), c_n the
transition count and src_n the source position (0 = last history item).  All
new parameters are fp32.  r decides which neighbours are read, so the read
depends on the latent.

Evaluation interventions (``trans_mode``): ``raw`` (own neighbours), ``none``
(no neighbours: r' = r), ``shuffle`` (another user's neighbour set), ``shuffle_r``
(own neighbours, then the final r' is replaced by another prompt's r' from the
same batch).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset


class TransitionReader(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        d = int(spec.get("dim", 256))
        self.d = d
        self.query = nn.Linear(hidden_size, d)
        self.key = nn.Linear(hidden_size, d)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.src_bias = nn.Parameter(torch.zeros(int(spec["L"])))
        self.beta = nn.Parameter(torch.ones(()))
        self.tau = nn.Parameter(torch.ones(()))
        self.gain = nn.Parameter(torch.tensor(float(spec.get("init_gain", 30.0))))
        self.init_gain = float(spec.get("init_gain", 30.0))
        self.register_buffer("table", torch.empty(0), persistent=False)

    def reset_parameters(self):
        for m in (self.query, self.key, self.proj):
            m.reset_parameters()
        with torch.no_grad():
            self.src_bias.zero_()
            self.beta.fill_(1.0)
            self.tau.fill_(1.0)
            self.gain.fill_(self.init_gain)

    def forward(self, r, items, src, count):
        """r [B, H]; items/src [B, N] (items -1 = pad); count [B, N] -> component [B, H], weights [B, N]."""
        valid = items >= 0
        e = self.table[items.clamp_min(0)].float()                               # [B, N, H]
        q = self.query(r.float())                                                # [B, d]
        # v2: keys from unit-normalised title vectors, and the content score standardised
        # over the row's neighbour set times a learned temperature.  v1 used raw scores:
        # with title-vector norms ~65 the softmax was one-hot from the start and the
        # reader received no gradient (356391).
        k = self.key(F.normalize(e, dim=-1))                                     # [B, N, d]
        z = (k @ q.unsqueeze(-1)).squeeze(-1) / self.d ** 0.5
        vf = valid.float()
        n = vf.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (z * vf).sum(-1, keepdim=True) / n
        std = ((((z - mean) ** 2 * vf).sum(-1, keepdim=True) / n) + 1e-8).sqrt()  # eps inside: no inf grad at 1 neighbour
        z = (z - mean) / (std + 1e-4)
        s = self.tau * z + self.beta * torch.log1p(count.float()) + self.src_bias[src.clamp(0, len(self.src_bias) - 1)]
        s = s.masked_fill(~valid, float("-inf"))
        has = valid.any(-1)
        w = torch.where(has.unsqueeze(-1), F.softmax(s.masked_fill(~has.unsqueeze(-1), 0.0), -1),
                        torch.zeros_like(s))
        m = (w.unsqueeze(1) @ e).squeeze(1)                                      # [B, H]
        comp = F.normalize(self.proj(m), dim=-1) * self.gain
        return comp * has.unsqueeze(-1).float(), w


class TransitionLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.trans = TransitionReader(config.hidden_size, config.latent_transition)
        self._trans = None            # (items, src, count) of the next training prefill
        self.trans_override = None    # same, set by evaluation before generate
        self.trans_disabled = False   # "none" intervention
        self.roll_latent = False      # "shuffle_r" intervention
        self.last_trans_weights = None

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        model = super().from_pretrained(path, *args, **kwargs)
        spec = model.config.latent_transition
        model.trans.float()
        if not load_fp32_trans_params(model, str(path)):
            model.trans.reset_parameters()  # from the base model
        table = torch.load(spec["title_vectors"], map_location="cpu", weights_only=False)["vectors"]
        model.trans.table = table.float().to(model.model.embed_tokens.weight.device)
        return model

    def forward(self, *args, trans_items=None, trans_src=None, trans_count=None, **kwargs):
        if trans_items is not None:
            self._trans = (trans_items, trans_src, trans_count)
        return super().forward(*args, **kwargs)

    def observe_reasoning_latent(self, latent):
        batch = self.trans_override if self.trans_override is not None else self._trans
        if self.training:
            self._trans = None  # do not leak into the next batch
        if self.trans_disabled:
            self.last_trans_weights = None
            out = latent
        else:
            if batch is None:
                raise RuntimeError("transition latent: no neighbours for this batch")
            reps = latent.size(0) // batch[0].size(0)
            items, src, count = (x.to(latent.device).repeat_interleave(reps, 0) for x in batch)
            comp, w = self.trans(latent, items, src, count)
            self.last_trans_weights = w.detach()
            out = (latent.float() + comp).to(latent.dtype)
        if self.roll_latent and batch is not None:
            reps = out.size(0) // batch[0].size(0)
            out = torch.roll(out, shifts=reps, dims=0)  # every prompt gets the next prompt's r'
        return out


def load_fp32_trans_params(model, checkpoint):
    """Reload the fp32 reader parameters (from_pretrained rounds them to bf16); False if absent."""
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    params = dict(model.trans.named_parameters())
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith("trans."):
                    with torch.no_grad():
                        params[key[len("trans."):]].copy_(f.get_tensor(key).float())
                    loaded += 1
    return loaded > 0


class TransitionRowsDataset(LatentRDataset):
    """Unchanged R3 rows plus the row's transition neighbours (indexed by the CSV row index).

    ``prep``: transition_prep.py output for this split; ``shuffle_seed`` gives every row
    the neighbour set of a random row of a different user.
    """

    def __init__(self, *args, prep=None, shuffle_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        source = self.data.index.to_numpy()
        if shuffle_seed is not None:
            from candidate_pooling import other_user_rows
            source = other_user_rows(self.data, shuffle_seed)
        for item, row in zip(self.inputs, source):
            item["trans_items"] = prep["items"][row].tolist()
            item["trans_src"] = prep["src"][row].tolist()
            item["trans_count"] = prep["count"][row].tolist()


class TransitionCollator:
    """Wrap DataCollatorForSeq2Seq: stack the neighbour tensors."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        features = [dict(f) for f in features]
        extra = {k: [f.pop(k) for f in features] for k in ("trans_items", "trans_src", "trans_count")}
        batch = self.base(features)
        batch["trans_items"] = torch.tensor(extra["trans_items"], dtype=torch.long)
        batch["trans_src"] = torch.tensor(extra["trans_src"], dtype=torch.long)
        batch["trans_count"] = torch.tensor(extra["trans_count"], dtype=torch.float32)
        return batch
