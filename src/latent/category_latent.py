"""Category-conditioned R3 latents (prompt unchanged), Toys level-3 categories.

Two ways to put a category into the latent (inputs from category_prep.py):

* ``additive`` (side_info_latent.SideInfoLatentModel, kind ``oracle_category``):
  r' = r + g * normalize(P(e_c)), e_c the gte embedding of one category.
  Trained with the target's category (privileged, training only); evaluated
  with the oracle category, another user's, the predicted top-1, or a mixture
  over the predicted top-K (one beam search per category, merged by
  category score + item sequence score).

* ``inject`` (IntuRec-style, ``CategoryInjectModel``): R3's pooled r is replaced
  by IntuRec's intuition encoder (/common/home/users/m/mk.ngohuu.2024/research/
  IntuRec/model/IntuRecF2.py, InjectNet): each of K candidate categories is
  encoded from its path tokens by a self-attention pooling (last-token query,
  RoPE), and the last-layer hidden state at the thought position cross-attends
  over the K category vectors.  Training list: the target category plus K-1
  out-of-fold predicted categories, in random order; evaluated with the
  predicted top-K (contains the target in ~40% of test rows), the oracle-
  inserted list, or another user's list.  Next-token loss only (no L_intu).

New trainable modules are fp32.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from layers import LatentModel, VanillaRoPE
from reasoning_dataset import LatentRDataset


class InjectNet(nn.Module):
    """IntuRec's InjectNet: per-candidate self-attention pooling, then cross-attention."""

    def __init__(self, hidden_size):
        super().__init__()
        self.W_q, self.W_k, self.W_v = (nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.W_q_2, self.W_k_2, self.W_v_2 = (nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.rope = VanillaRoPE(hidden_size)

    def encode(self, x, mask):
        """x [M, L, H] candidate token embeddings (left-padded), mask [M, L] -> [M, H]."""
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        cos, sin = self.rope(x, position_ids=torch.arange(x.size(1), device=x.device).unsqueeze(0))
        Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0).float(), sin.squeeze(0).float(), unsqueeze_dim=0)
        scores = (Q[:, -1:] @ K.transpose(-2, -1)) / x.size(-1) ** 0.5
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        return (F.softmax(scores, -1) @ V).squeeze(1)

    def forward(self, query, x, mask):
        """query [B, H]; x [B, N, L, H]; mask [B, N, L] -> r [B, H], weights [B, N]."""
        B, N, L, H = x.shape
        items = self.encode(x.reshape(B * N, L, H), mask.reshape(B * N, L)).reshape(B, N, H)
        q = self.W_q_2(query).unsqueeze(1)
        k, v = self.W_k_2(items), self.W_v_2(items)
        w = F.softmax((q @ k.transpose(-2, -1)) / H ** 0.5, -1)
        return (w @ v).squeeze(1), w.squeeze(1)


class CategoryInjectModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_category_inject
        self.injectnet = InjectNet(config.hidden_size)
        tokens = spec["token_ids"] + [spec["unknown_token_ids"]]
        width = max(map(len, tokens))
        pad = int(spec["pad_id"])
        # left-padded so the last position is a real token (IntuRec's last-token query)
        self.register_buffer("cat_tokens", torch.tensor([[pad] * (width - len(t)) + t for t in tokens]), persistent=False)
        self.register_buffer("cat_mask", torch.tensor([[0] * (width - len(t)) + [1] * len(t) for t in tokens]), persistent=False)
        self._cat = None
        self.cat_override = None
        self.last_cat_weights = None

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        model = super().from_pretrained(path, *args, **kwargs)
        model.injectnet.float()
        if not load_fp32_params(model, str(path), "injectnet."):
            for m in model.injectnet.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
        return model

    def forward(self, *args, cat_list=None, **kwargs):
        if cat_list is not None:
            self._cat = cat_list
        return super().forward(*args, **kwargs)

    def generate_embs(self, input_ids, attention_mask):
        cats = self.cat_override if self.cat_override is not None else self._cat
        if cats is None:
            raise RuntimeError("category inject: no category list for this batch")
        cats = cats.to(input_ids.device).repeat_interleave(input_ids.size(0) // cats.size(0), 0)
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        where = torch.nonzero(input_ids == self.model.embed_tokens.num_embeddings - 1)
        query = hidden[where[:, 0], where[:, 1]].float()                          # IntuRec: state at the thought
        tok = self.cat_tokens.to(cats.device)[cats]                                 # [B, N, L]
        mask = self.cat_mask.to(cats.device)[cats]
        x = self.model.embed_tokens(tok).float()
        r, w = self.injectnet(query, x, mask)
        self.last_cat_weights = w.detach()
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[where[:, 0], where[:, 1]] = r.to(input_embs.dtype)
        if self.training:
            self._cat = None
        return input_embs


def load_fp32_params(model, checkpoint, prefix):
    """Reload parameters under ``prefix`` in fp32 (from_pretrained rounds them to bf16); False if absent."""
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    params = dict(model.named_parameters())
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith(prefix) and key in params:
                    with torch.no_grad():
                        params[key].copy_(f.get_tensor(key).float())
                    loaded += 1
    return loaded > 0


class CategoryRowsDataset(LatentRDataset):
    """Unchanged R3 rows plus per-row categories (indexed by the frame index).

    ``target`` [N_split] target category index (``unknown`` = n_cats);
    ``lists`` [N_split, K] predicted categories.  Modes:
      additive: ``oracle`` (target), ``shuffle`` (target of another user's row),
                ``pred:<j>`` (j-th predicted category)  -> ``side_index``
      inject:   ``oracle`` (target + K-1 predicted non-target, random order),
                ``pred`` (predicted top-K), ``shuffle`` (another user's predicted list)
                -> ``cat_list``
    """

    def __init__(self, *args, target=None, lists=None, injection="additive", mode="oracle", unknown=None,
                 seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        rows = self.data.index.to_numpy()
        users = self.data["user_id"].astype(str).to_numpy()
        rng = np.random.default_rng(seed)
        if mode == "shuffle":
            source = []
            for i in range(len(rows)):
                j = int(rng.integers(len(rows)))
                while users[j] == users[i]:
                    j = int(rng.integers(len(rows)))
                source.append(rows[j])
        else:
            source = rows
        for item, row, src in zip(self.inputs, rows, source):
            if injection == "additive":
                if mode in ("oracle", "shuffle"):
                    item["side_index"] = int(target[src])
                elif mode.startswith("pred:"):
                    item["side_index"] = int(lists[row][int(mode.split(":")[1])])
                else:
                    raise ValueError(f"unknown additive mode {mode}")
            else:
                k = lists.shape[1]
                if mode == "oracle" and int(target[row]) != unknown:
                    others = [int(c) for c in lists[row] if int(c) != int(target[row])][:k - 1]
                    cats = [int(target[row])] + others
                    rng_row = np.random.default_rng([seed, int(row)])
                    cats = [cats[i] for i in rng_row.permutation(len(cats))]
                elif mode in ("oracle", "pred"):
                    cats = [int(c) for c in lists[row]]
                elif mode == "shuffle":
                    cats = [int(c) for c in lists[src]]
                else:
                    raise ValueError(f"unknown inject mode {mode}")
                item["cat_list"] = cats


class CategoryCollator:
    """Wrap DataCollatorForSeq2Seq: stack ``cat_list`` / ``side_index``."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        features = [dict(f) for f in features]
        extra = {k: [f.pop(k) for f in features] for k in ("cat_list", "side_index") if k in features[0]}
        batch = self.base(features)
        for k, v in extra.items():
            batch[k] = torch.tensor(v, dtype=torch.long)
        return batch
