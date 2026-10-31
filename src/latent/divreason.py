"""DivReason (An et al., SIGIR 2026, doi 10.1145/3805712.3809732) on top of R3.

Divergent reasoning path generation (training-free uncertainty, Sec. 4.1.2):
  * MC-Dropout: dropout on every Transformer block's FFN output stays active in
    the passes that produce the reasoning paths (Eq. 4-5);
  * Directional perturbation: k-means centroids of layer-l hidden states,
    v_k = mu_k / ||mu_k|| (Eq. 8-9), added as h_l + alpha * v_k (Eq. 10), one
    direction per path.
Each of the N paths is one first pass under its own dropout mask and direction,
pooled by R3's latent attention into r_n (Eq. 2-3, 11).

Reasoning path aggregation (Sec. 4.2):
  s_n  = MLP(r_n)                                  (Eq. 12)
  C_ij = cos(r_i, r_j);  s'_n = s_n - beta * sum_{j!=n} C_nj   (Eq. 13-14)
  prune by a threshold, alpha_n = softmax(s')_n,  r_agg = sum alpha_n r_n  (Eq. 15-16)
r_agg replaces R3's latent in the <|Thought|> slot; the decoder still reads the
history, exactly as in R3.

Details the paper does not specify (our choices, all in the config):
  dropout rate, layer l, alpha (here relative to the median token norm at layer l),
  how hidden states are summarised for k-means (mean over prompt tokens), the
  pruning threshold (applied to the softmax weights, at evaluation only, keeping
  the top path), and whether centroids are centred before normalising
  (``center``: the paper normalises raw centroids, which share a large common
  component).  Directions come from the base model before training.

Config: ``config.latent_divreason = {"n_paths", "dropout", "layer", "alpha",
"beta", "prune", "router_dim"}``; directions are a buffer saved with the model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


class DivergentPaths(nn.Module):
    def __init__(self, hidden_size: int, spec: dict):
        super().__init__()
        self.n = int(spec["n_paths"])
        self.dropout = float(spec.get("dropout", 0.1))
        self.layer = int(spec.get("layer", 14))
        self.alpha = float(spec.get("alpha", 0.0))  # absolute, set from the relative value at init
        self.beta = float(spec.get("beta", 0.5))
        self.prune = float(spec.get("prune", 0.05))
        # Path-diversity test (user hypothesis): per-path CE and a hinge on the paths'
        # output divergence, history identical across paths.  0 = paper DivReason.
        self.path_ce_weight = float(spec.get("path_ce_weight", 0.0))
        self.path_div_weight = float(spec.get("path_div_weight", 0.0))
        self.path_div_margin = float(spec.get("path_div_margin", 0.2))
        self.path_div_metric = spec.get("path_div_metric", "js")  # js | sqrt_js
        self.last_paths: Optional[torch.Tensor] = None
        self.last_train_stats: dict = {}
        dim = int(spec.get("router_dim", 256))
        self.router = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim), nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer("directions", torch.zeros(self.n, hidden_size))
        self.float()
        # Runtime state, read by the hooks during path passes.
        self.active = False
        self.row_alpha: Optional[torch.Tensor] = None  # [rows, H] perturbation added at layer l
        self.mode = "paths"  # evaluation: "paths" (method) | "single" (one plain R3 path)
        self.override = None  # evaluation intervention on r_agg: callable [B, H] -> [B, H]
        self.last_r_agg: Optional[torch.Tensor] = None
        self.last_stats: dict = {}

    def attach(self, lm) -> None:
        """Register the MC-dropout and perturbation hooks on ``lm``'s decoder layers."""
        for i, layer in enumerate(lm.model.layers):
            layer.mlp.register_forward_hook(self._dropout_hook)
            if i == self.layer:
                layer.register_forward_hook(self._perturb_hook)

    def _dropout_hook(self, module, inputs, output):
        if self.active and self.dropout > 0:
            return F.dropout(output, p=self.dropout, training=True)
        return output

    def _perturb_hook(self, module, inputs, output):
        if not self.active or self.row_alpha is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + self.row_alpha.unsqueeze(1).to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def forward(self, lm, input_ids, attention_mask, thought_idx):
        """r_agg ``[B, H]`` (fp32) for the prompts in ``input_ids``."""
        # Beam search repeats each prompt; compute the paths once per distinct prompt.
        unique, inverse = torch.unique(torch.cat([input_ids, attention_mask], 1), dim=0, return_inverse=True)
        if unique.size(0) < input_ids.size(0):
            rows = torch.arange(input_ids.size(0), device=input_ids.device)
            first = torch.full((unique.size(0),), input_ids.size(0), dtype=torch.long, device=input_ids.device)
            first.scatter_reduce_(0, inverse, rows, reduce="amin")  # first row of each distinct prompt
            r_agg = self._aggregate(lm, input_ids[first], attention_mask[first], thought_idx[first])
            r_agg = r_agg[inverse]
        else:
            r_agg = self._aggregate(lm, input_ids, attention_mask, thought_idx)
        if self.override is not None:
            r_agg = self.override(r_agg)
        self.last_r_agg = r_agg.detach()
        return r_agg

    def _aggregate(self, lm, input_ids, attention_mask, thought_idx):
        batch = input_ids.size(0)
        self.last_paths = None
        if self.mode.startswith("path:"):  # evaluation: one perturbed path k alone
            r = self._paths(lm, input_ids, attention_mask, thought_idx, self.n, True)
            self.last_stats = {}
            return r[:, int(self.mode.split(":")[1])]
        if self.mode == "single":
            self.last_stats = {}
            return self._paths(lm, input_ids, attention_mask, thought_idx, 1, False)[:, 0]
        n = self.n
        r = self._paths(lm, input_ids, attention_mask, thought_idx, n, True)
        if self.training:
            self.last_paths = r  # with graph, for the per-path losses (path_losses)
        s = self.router(r).squeeze(-1)  # [B, N]
        unit = F.normalize(r, dim=-1)
        cos = unit @ unit.transpose(1, 2)
        off_diag = cos.sum(-1) - cos.diagonal(dim1=1, dim2=2)
        s_prime = s - self.beta * off_diag
        weights = torch.softmax(s_prime, dim=-1)
        if not self.training and self.prune > 0:
            keep = weights >= self.prune
            keep.scatter_(1, weights.argmax(1, keepdim=True), True)
            weights = weights * keep
            weights = weights / weights.sum(1, keepdim=True)
        pair = cos.masked_select(~torch.eye(n, dtype=torch.bool, device=cos.device).unsqueeze(0).expand_as(cos))
        offset = F.normalize(r - r.mean(1, keepdim=True), dim=-1)
        offset_cos = (offset @ offset.transpose(1, 2)).masked_select(
            ~torch.eye(n, dtype=torch.bool, device=cos.device).unsqueeze(0).expand_as(cos))
        self.last_stats = {
            "path_cos": float(pair.mean()),
            "path_offset_cos": float(offset_cos.mean()),
            "path_spread": float(((r - r.mean(1, keepdim=True)).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).mean()),
            "weight_max": float(weights.max(1).values.mean()),
            "weight_entropy": float(-(weights.clamp_min(1e-12).log() * weights).sum(1).mean()),
        }
        return (weights.unsqueeze(-1) * r).sum(1)

    def _paths(self, lm, input_ids, attention_mask, thought_idx, n, noise):
        """``[B, n, H]`` fp32 path latents; with ``noise``, row b*n + k gets MC-dropout and direction k."""
        batch = input_ids.size(0)
        rows_alpha = (self.alpha * self.directions.float()).repeat(batch, 1) if noise else None
        ids = input_ids.repeat_interleave(n, 0)
        mask = attention_mask.repeat_interleave(n, 0)
        self.active, self.row_alpha = noise, rows_alpha
        try:
            hidden = lm.model(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            self.active, self.row_alpha = False, None
        r = lm.attention(hidden_states=hidden, attention_mask=mask, thought_id_idx=thought_idx.repeat_interleave(n, 0))
        return r.float().view(batch, n, -1)


def target_logprobs(lm, input_ids, attention_mask, labels, latent, thought_pos):
    """Decoder pass with ``latent`` [B, H] in the thought slot; log-probs ``[T, V]`` (fp32) at
    the target positions (shifted, labels != -100) and the target ids ``[T]``."""
    embeds = lm.model.embed_tokens(input_ids)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    embeds = embeds.index_put((rows, thought_pos), latent.to(embeds.dtype))
    hidden = lm.model(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
    target = labels[:, 1:]
    keep = target.ne(-100)
    logits = lm.lm_head(hidden[:, :-1][keep]).float()
    return torch.log_softmax(logits, -1), target[keep]


def path_losses(lm, input_ids, attention_mask, labels, paths, thought_pos, margin, metric="js"):
    """Per-path decoder passes (history identical, only the latent differs).

    Returns (sum over target tokens of the path-mean NLL, the hinge
    max(0, margin - D) between the paths' next-token distributions at the target
    positions, number of target tokens, stats).  ``metric="js"``: D = mean pairwise
    JS (zero gradient when the paths agree, since JS is quadratic there);
    ``"sqrt_js"``: D = mean per-token Jensen-Shannon distance sqrt(JS), first order
    at agreement."""
    n = paths.size(1)
    logps, nll_sum = [], 0.0
    for k in range(n):
        logp, target = target_logprobs(lm, input_ids, attention_mask, labels, paths[:, k], thought_pos)
        nll_sum = nll_sum + F.nll_loss(logp, target, reduction="sum") / n
        logps.append(logp)
    js_pairs, dist_pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p, q = logps[i].exp(), logps[j].exp()
            log_m = torch.log((p + q) / 2 + 1e-12)
            js = 0.5 * (p * (logps[i] - log_m)).sum(-1) + 0.5 * (q * (logps[j] - log_m)).sum(-1)
            js_pairs.append(js.mean())
            dist_pairs.append(torch.sqrt(js.clamp(min=0) + 1e-8).mean())
    js_mean = torch.stack(js_pairs).mean()
    dist_mean = torch.stack(dist_pairs).mean()
    if metric == "js":
        divergence = js_mean
    elif metric == "sqrt_js":
        divergence = dist_mean
    else:
        raise ValueError(f"unknown path divergence metric {metric}")
    hinge = F.relu(margin - divergence)
    tokens = logps[0].size(0)
    return nll_sum, hinge, tokens, {"path_js": float(js_mean), "path_js_distance": float(dist_mean),
                                    "path_nll": float(nll_sum) / max(tokens, 1)}


@torch.no_grad()
def compute_directions(lm, batches, layer: int, k: int, center: bool = False, seed: int = 0):
    """k-means centroids of per-prompt mean layer-``layer`` hidden states (prompt tokens
    only), normalised (Eq. 8-9).  Returns ``(directions [k, H], median token norm at
    the layer, diagnostics)``.  ``batches`` yields (input_ids, attention_mask, prompt_mask)."""
    from sklearn.cluster import KMeans

    feats, norms = [], []
    for input_ids, attention_mask, prompt_mask in batches:
        out = lm.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[layer + 1].float()  # hidden_states[0] = embeddings; [i+1] = output of layer i
        m = prompt_mask.float().unsqueeze(-1)
        feats.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
        norms.append(h.norm(dim=-1)[prompt_mask.bool()].cpu())
    x = torch.cat(feats).numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    mu = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    if center:
        mu = mu - torch.tensor(x.mean(0), dtype=torch.float32)
    v = F.normalize(mu, dim=-1)
    cos = v @ v.T
    diag = {"n_rows": len(x), "cluster_sizes": torch.bincount(torch.tensor(km.labels_), minlength=k).tolist(),
            "direction_cos_mean": float(cos[~torch.eye(k, dtype=torch.bool)].mean()) if k > 1 else 1.0}
    return v, float(torch.cat(norms).median()), diag


def load_fp32_divreason_params(model, checkpoint: str) -> None:
    """Reload router and directions in fp32 (``from_pretrained(torch_dtype=bf16)`` rounds them)."""
    import os
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    model.divreason.float()
    state = dict(model.divreason.named_parameters())
    state.update(dict(model.divreason.named_buffers()))
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith("divreason."):
                    with torch.no_grad():
                        state[key[len("divreason."):]].copy_(f.get_tensor(key).float())
                    loaded += 1
    if files and loaded == 0:
        raise RuntimeError(f"no DivReason parameters found in {checkpoint}")


def directions_from_dataset(lm, rows, collator, layer, k, center, n_rows=4096, batch=16):
    """Directions from the first ``n_rows`` training rows (prompt tokens = labels == -100)."""
    device = next(lm.parameters()).device
    n_rows = min(n_rows, len(rows))

    def batches():
        for i in range(0, n_rows, batch):
            b = collator([rows[j] for j in range(i, min(i + batch, n_rows))])
            prompt = (b["labels"] == -100) & b["attention_mask"].bool()
            yield b["input_ids"].to(device), b["attention_mask"].to(device), prompt.to(device)

    return compute_directions(lm, batches(), layer, k, center)


class DivReasonDiagnostics(TrainerCallback):
    """Probe at every evaluation on fixed validation rows: does the target NLL depend on
    the sample's own r_agg?  Logs raw NLL and its change when r_agg is replaced by another
    row's (shuffle), the probe mean (mean), or the single unperturbed R3 path (single),
    plus the aggregation statistics.  Appends to ``<output_dir>/divreason_diagnostics.jsonl``."""

    def __init__(self, rows, collator, output_dir):
        self.batch = collator(rows)
        self.path = f"{output_dir}/divreason_diagnostics.jsonl"

    @staticmethod
    def _nll(model, batch):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        labels = batch["labels"][:, 1:]
        loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        valid = labels.ne(-100)
        return (loss * valid).sum(1) / valid.sum(1).clamp(min=1)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        dr = getattr(model, "divreason", None)
        if dr is not None and dr.last_train_stats and state.global_step % 10 == 0:
            import json
            print("divreason train:", json.dumps({"step": state.global_step, **dr.last_train_stats}), flush=True)

    @staticmethod
    def _js(logp_a, logp_b):
        p, q = logp_a.exp(), logp_b.exp()
        log_m = torch.log((p + q) / 2 + 1e-12)
        return float((0.5 * (p * (logp_a - log_m)).sum(-1) + 0.5 * (q * (logp_b - log_m)).sum(-1)).mean())

    def _path_identity_vs_content(self, model, chunks, step):
        """Same user, path 0 vs path 1 (path identity) and own path 0 vs another
        user's path 0 (user content): JS between target distributions and NLL change."""
        dr = model.divreason
        thought_id = model.model.embed_tokens.num_embeddings - 1
        bank = []
        for c in chunks:
            pos = torch.nonzero(c["input_ids"] == thought_id)[:, 1]
            bank.append(dr._paths(model, c["input_ids"], c["attention_mask"], pos, dr.n, True))
        bank = torch.cat(bank)
        other = bank.roll(1, 0)
        js_id, js_user, d_id, d_user = [], [], [], []
        for i, c in enumerate(chunks):
            pos = torch.nonzero(c["input_ids"] == thought_id)[:, 1]
            own = bank[i * step:(i + 1) * step]
            args_ = (model, c["input_ids"], c["attention_mask"], c["labels"])
            lp00, tgt = target_logprobs(*args_, own[:, 0], pos)
            lp01, _ = target_logprobs(*args_, own[:, 1], pos)
            lpx0, _ = target_logprobs(*args_, other[i * step:(i + 1) * step, 0], pos)
            nll = lambda lp: float(F.nll_loss(lp, tgt))
            js_id.append(self._js(lp00, lp01)); js_user.append(self._js(lp00, lpx0))
            d_id.append(nll(lp01) - nll(lp00)); d_user.append(nll(lpx0) - nll(lp00))
        mean = lambda x: sum(x) / len(x)
        return {"js_same_user_other_path": mean(js_id), "js_other_user_same_path": mean(js_user),
                "dnll_same_user_other_path": mean(d_id), "dnll_other_user_same_path": mean(d_user)}

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        import json
        if model is None or not hasattr(model, "divreason"):
            return
        was_training = model.training
        model.eval()
        dr = model.divreason
        device = next(model.parameters()).device
        n, step = self.batch["input_ids"].size(0), 4  # full-vocabulary logits: keep chunks small
        chunks = [{k: self.batch[k][i:i + step].to(device) for k in ("input_ids", "attention_mask", "labels")}
                  for i in range(0, n, step)]
        with torch.no_grad():
            raw, bank, stats = [], [], []
            for c in chunks:
                raw.append(self._nll(model, c))
                bank.append(dr.last_r_agg)
                stats.append(dr.last_stats)
            raw, bank = torch.cat(raw), torch.cat(bank)
            out = {}
            for name, rep in (("shuffle", bank.roll(1, 0)), ("mean", bank.mean(0, keepdim=True).expand_as(bank))):
                vals = []
                for i, c in enumerate(chunks):
                    dr.override = lambda x, r=rep[i * step:(i + 1) * step]: r.to(x)
                    try:
                        vals.append(self._nll(model, c))
                    finally:
                        dr.override = None
                out[name] = torch.cat(vals)
            dr.mode = "single"
            try:
                out["single"] = torch.cat([self._nll(model, c) for c in chunks])
            finally:
                dr.mode = "paths"
            identity = self._path_identity_vs_content(model, chunks, step) if dr.n > 1 else {}
        share = ((bank - bank.mean(0)).norm(dim=1) / bank.norm(dim=1).clamp_min(1e-12)).mean()
        record = {"step": state.global_step, "epoch": state.epoch, "raw_nll": float(raw.mean()),
                  "r_agg_residual_share": float(share)}
        for name, v in out.items():
            record[f"{name}_dnll"] = float((v - raw).mean())
        for key in stats[0]:
            record[key] = sum(s[key] for s in stats) / len(stats)
        record.update(identity)
        print("divreason diagnostics:", json.dumps(record), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if was_training:
            model.train()
