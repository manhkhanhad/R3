"""SASRec-informed pooling of the R3 latent (prompt unchanged).

R3 pools the last hidden states H before ``<|Thought|>`` into r_R3 with one
attention query (layers.SelfAttentionLayer).  Here R3's pooling also reads the
row's SASRec top-C candidate items (sasrec_candidates.py, out of fold for
training rows) in a separate branch:

    s_j    = tau * standardise_j(q · Wk c_j / sqrt(d)) + alpha * logp_j
    r_cand = softmax_j(s_j) · Wv c_j
    r      = r_R3 + g * r_cand

with R3's own Wq / Wk / Wv (unrotated: candidates have no prompt position).
A separate softmax is needed because R3's pooling puts a logit of ~100-360 on
one template token, so candidates in the same softmax get exactly zero mass
and gradient.  ``logp_j`` is SASRec's log-prob (its trust signal);
standardising q·k over the C candidates makes the selection scale-free.
g, tau, alpha are fp32, init 1.

Candidate vector c_j (``kind``):
* ``title``: R3's last-layer representation of the item title
  (candidate_title_vectors.py); same space as H, no new projection.
* ``sasrec``: the frozen SASRec item embedding through an fp32 projection,
  rescaled to the mean title-vector norm.

The prompt, the single thought slot and the decoder are unchanged; the decoder
sees candidates only through r.  Interventions (evaluation): ``shuffle``
(another user's candidates), ``drop`` (no candidate branch: exactly plain R3).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from layers import LatentModel, SelfAttentionLayer
from reasoning_dataset import LatentRDataset


class CandidatePool(nn.Module):
    """fp32 parameters of the candidate branch (gate, tau, alpha, optional SASRec projection)."""

    def __init__(self, hidden_size, spec):
        super().__init__()
        self.kind = spec["kind"]
        self.gate = nn.Parameter(torch.ones(()))
        self.tau = nn.Parameter(torch.ones(()))
        self.alpha = nn.Parameter(torch.ones(()))
        if self.kind == "sasrec":
            self.proj = nn.Linear(int(spec["sasrec_dim"]), hidden_size)
        elif self.kind != "title":
            raise ValueError(f"unknown candidate kind {self.kind}")
        self.register_buffer("table", torch.empty(0), persistent=False)
        self.register_buffer("scale", torch.tensor(float(spec.get("title_norm", 1.0))), persistent=False)

    def reset_parameters(self):
        with torch.no_grad():
            for p in (self.gate, self.tau, self.alpha):
                p.fill_(1.0)
        if self.kind == "sasrec":
            self.proj.reset_parameters()

    def vectors(self, items):
        v = self.table[items.to(self.table.device)].float()
        if self.kind == "sasrec":
            v = self.scale * F.normalize(self.proj(v), dim=-1)
        return v


class CandidatePoolingModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.cand_pool = CandidatePool(config.hidden_size, config.latent_candidate_pooling)
        self._cand = None           # (items [B, C], logp [B, C]) for the next prefill (training)
        self.cand_override = None   # same, set by evaluation before generate
        self.cand_disabled = False  # "drop" intervention: plain R3 pooling
        self.last_cand_weights = None  # [B, C] candidate selection weights (prefill)

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        model = super().from_pretrained(path, *args, **kwargs)
        spec = model.config.latent_candidate_pooling
        model.cand_pool.float()
        if not load_fp32_pool_params(model, str(path)):
            model.cand_pool.reset_parameters()  # starting from a plain R3 checkpoint
        device = model.model.embed_tokens.weight.device
        if spec["kind"] == "title":
            table = torch.load(spec["title_vectors"], map_location="cpu", weights_only=False)["vectors"]
        else:
            table = torch.load(spec["candidates_file"], map_location="cpu", weights_only=False)["item_embeddings"]
        model.cand_pool.table = table.float().to(device)
        model.cand_pool.scale.fill_(float(spec.get("title_norm", 1.0)))
        return model

    def forward(self, *args, cand_items=None, cand_logp=None, **kwargs):
        if cand_items is not None:
            self._cand = (cand_items, cand_logp)
        return super().forward(*args, **kwargs)

    def pool(self, hidden, attention_mask, thought_idx, items, logp):
        """R3's SelfAttentionLayer output plus g * the candidate branch."""
        att: SelfAttentionLayer = self.attention
        r = att(hidden_states=hidden, attention_mask=attention_mask, thought_id_idx=thought_idx)
        if self.cand_disabled:
            self.last_cand_weights = None
            return r
        rows = torch.arange(hidden.size(0), device=hidden.device)
        q = att.query(hidden[rows, thought_idx - 1]).float()                 # [B, H], unrotated
        c = self.cand_pool.vectors(items)                                    # [B, C, H] fp32
        kc = att.key(c.to(att.key.weight.dtype)).float()
        vc = att.value(c.to(att.value.weight.dtype)).float()
        z = (kc @ q.unsqueeze(-1)).squeeze(-1) * att.scale_score             # [B, C]
        z = (z - z.mean(-1, keepdim=True)) / (z.std(-1, keepdim=True) + 1e-4)
        p = self.cand_pool
        w = F.softmax(p.tau * z + p.alpha * logp.to(z.device).float(), -1)
        self.last_cand_weights = w.detach()
        r_cand = (w.unsqueeze(1) @ vc).squeeze(1)
        return (r.float() + p.gate * r_cand).to(r.dtype)

    def generate_embs(self, input_ids, attention_mask):
        cand = self.cand_override if self.cand_override is not None else self._cand
        if cand is None and not self.cand_disabled:
            raise RuntimeError("candidate pooling: no candidates for this batch")
        items = logp = None
        if cand is not None:
            reps = input_ids.size(0) // cand[0].size(0)
            items = cand[0].to(input_ids.device).repeat_interleave(reps, 0)
            logp = cand[1].to(input_ids.device).repeat_interleave(reps, 0)
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        thought_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_id)
        r = self.pool(hidden, attention_mask, where[:, 1], items, logp)
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[where[:, 0], where[:, 1]] = r.to(input_embs.dtype)
        if self.training:
            self._cand = None  # do not leak into the next batch
        return input_embs


def load_fp32_pool_params(model, checkpoint):
    """Reload the fp32 candidate-pool parameters (from_pretrained rounds them to bf16); False if absent."""
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    params = dict(model.cand_pool.named_parameters())
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith("cand_pool."):
                    with torch.no_grad():
                        params[key[len("cand_pool."):]].copy_(f.get_tensor(key).float())
                    loaded += 1
    return loaded > 0


class CandidateRowsDataset(LatentRDataset):
    """Unchanged R3 rows plus the row's SASRec candidates (``cand_items``, ``cand_logp``).

    ``candidates`` / ``logp``: [N_split, >= C] in CSV row order (indexed by the
    frame index, so sampling keeps the alignment); ``shuffle_seed`` gives every
    row the candidates of a random row of a different user.
    """

    def __init__(self, *args, candidates=None, logp=None, n_candidates=10, shuffle_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        items, scores = candidates[:, :n_candidates], logp[:, :n_candidates]
        source = self.data.index.to_numpy()
        if shuffle_seed is not None:
            source = other_user_rows(self.data, shuffle_seed)
        for item, row in zip(self.inputs, source):
            item["cand_items"] = items[row].tolist()
            item["cand_logp"] = scores[row].tolist()


def other_user_rows(frame, seed):
    """For every row of ``frame``, the CSV index of a random row of a different user."""
    users = frame["user_id"].astype(str).to_numpy()
    index = frame.index.to_numpy()
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(len(index)):
        j = int(torch.randint(len(index), (1,), generator=g))
        while users[j] == users[i]:
            j = int(torch.randint(len(index), (1,), generator=g))
        out.append(index[j])
    return out


class CandidateCollator:
    """Wrap DataCollatorForSeq2Seq: stack ``cand_items`` / ``cand_logp``."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        features = [dict(f) for f in features]
        items = [f.pop("cand_items") for f in features]
        logp = [f.pop("cand_logp") for f in features]
        batch = self.base(features)
        batch["cand_items"] = torch.tensor(items, dtype=torch.long)
        batch["cand_logp"] = torch.tensor(logp, dtype=torch.float32)
        return batch
