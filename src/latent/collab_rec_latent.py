"""CollabRec (IntuRec repo, model/CollabRecF2.py) ported onto R3's latent model.

The R3 thought slot ``<|Thought|>`` receives a vector built from collaboratively
retrieved *supports* instead of R3's history self-attention:

    q = last-layer hidden state at the <|Thought|> position  (first pass over the input)
    text mode:  r = InjectNet(q, support token embeddings)     (IntuRec's InjectNet)
    emb mode:   r = W_r q + softmax(W_q q (W_k P s)^T / sqrt(d)) W_v P s
                with s = precomputed sentence embeddings of the supports, P a learned projection
    no supports (k = 0):  text r = Linear(q);  emb r = W_r q

Only the producer of r changes; the R3 prompt, the target format and the NTP loss
are untouched. Optional (``rationale_beta`` > 0, text mode, training only): CollabRec's
rationale-alignment InfoNCE -- the thought vs the row's own train rationale (positive) and
``aux_negatives`` random train rationales, all pooled by InjectNet.pool over token embeddings,
temperature ``aux_temperature``; loss = NTP + beta * InfoNCE. Validation loss stays pure NTP
(IntuRec selects its checkpoint by valid NTP). The contrastive support loss (gamma) is not ported.

Supports are looked up from a table held as a non-persistent buffer (rebuilt by
the train/eval scripts from ``config.collab_rec``); each example carries the
indices of its supports, ``support_ids`` [k] (padding entries are -1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel


class InjectRoPE(nn.Module):
    """IntuRec's VanillaRoPE (model/Layers.py), base 1000."""

    def __init__(self, hidden_size):
        super().__init__()
        theta = 1.0 / (1000 ** (torch.arange(0, hidden_size, 2, dtype=torch.int64).float() / hidden_size))
        self.register_buffer("theta", theta, persistent=False)

    @torch.no_grad()
    def forward(self, pos):
        # pos: [bsz, seqlen] -> cos, sin: [bsz, seqlen, dim]
        theta = self.theta[None, :, None].float().expand(pos.shape[0], -1, 1).to(pos.device)
        freqs = (theta @ pos[:, None, :].float()).permute(0, 2, 1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos(), freqs.sin()


class InjectNet(nn.Module):
    """IntuRec's InjectNet (model/IntuRecF2.py), unchanged.

    Level 1: each support's token sequence is pooled by RoPE attention with the last
    token as query. Level 2: the thought query attends over the pooled supports.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.rope = InjectRoPE(hidden_size)
        self.W_q_2 = nn.Linear(hidden_size, hidden_size)
        self.W_k_2 = nn.Linear(hidden_size, hidden_size)
        self.W_v_2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def apply_rotary_pos_emb(Q, K, cos, sin):
        half = Q.shape[-1] // 2
        Q_rot = torch.cat([-Q[..., half:], Q[..., :half]], dim=-1)
        K_rot = torch.cat([-K[..., half:], K[..., :half]], dim=-1)
        return Q * cos + Q_rot * sin, K * cos + K_rot * sin

    def pool(self, x, mask):
        # x: [n, seqlen, dim] (left-padded), mask: [n, seqlen] -> [n, dim]
        n, seqlen, dim = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        cos, sin = self.rope(torch.arange(seqlen, device=x.device)[None, :].float().expand(n, -1))
        Q, K = self.apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(Q.dtype))
        scores = Q[:, -1:, :] @ K.transpose(-2, -1) / (dim ** 0.5)  # [n, 1, seqlen]
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), V).squeeze(1)

    def forward(self, query, x, mask, support_valid=None):
        # query: [bsz, 1, dim], x: [bsz, N, seqlen, dim], mask: [bsz, N, seqlen], support_valid: [bsz, N]
        bsz, N, seqlen, dim = x.shape
        pooled = self.pool(x.reshape(-1, seqlen, dim), mask.reshape(-1, seqlen)).reshape(bsz, N, dim)
        scores = self.W_q_2(query) @ self.W_k_2(pooled).transpose(-2, -1) / (dim ** 0.5)  # [bsz, 1, N]
        if support_valid is not None:
            scores = scores.masked_fill(~support_valid.unsqueeze(1), float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), self.W_v_2(pooled)).squeeze(1)


class SupportAttention(nn.Module):
    """CollabRec emb mode: thought = W_r(q) + attention over projected support vectors."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, supports=None, support_valid=None):
        res = self.W_r(query).squeeze(1)
        if supports is None:
            return res
        scores = self.W_q(query) @ self.W_k(supports).transpose(-2, -1) / (self.hidden_size ** 0.5)
        if support_valid is not None:
            scores = scores.masked_fill(~support_valid.unsqueeze(1), float("-inf"))
        return res + torch.matmul(torch.softmax(scores, dim=-1), self.W_v(supports)).squeeze(1)


class CollabLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "collab_rec", None)
        if not spec:
            raise ValueError("config.collab_rec is required")
        self.support_mode = spec["support_mode"]
        hidden = config.hidden_size
        if self.support_mode == "text":
            self.injectnet = InjectNet(hidden)
            self.no_support_proj = nn.Linear(hidden, hidden)  # k = 0 ablation
        elif self.support_mode == "emb":
            self.support_proj = nn.Linear(int(spec["rationale_dim"]), hidden)
            self.support_attn = SupportAttention(hidden)
        else:
            raise ValueError(f"unknown support_mode {self.support_mode!r}")
        # Table rows: text -> token ids [n, L] + mask [n, L]; emb -> vectors [n, d].
        self.register_buffer("support_table", torch.empty(0, 1), persistent=False)
        self.register_buffer("support_mask", torch.empty(0, 1, dtype=torch.bool), persistent=False)
        self._support_ids = None
        # rationale InfoNCE (set by the training script; off by default)
        self.rationale_beta = 0.0
        self.aux_temperature = 0.05
        self.aux_negatives = 63
        self.register_buffer("rationale_table", torch.empty(0, 1, dtype=torch.long), persistent=False)
        self.register_buffer("rationale_mask", torch.empty(0, 1, dtype=torch.bool), persistent=False)
        self._own_rows = None
        self._thought = None
        self.aux_log = []  # recent InfoNCE values (training)

    # ------------------------------------------------------------------ tables
    def set_support_table(self, table, mask=None):
        device = self.model.embed_tokens.weight.device
        table = torch.as_tensor(table)
        self.support_table = table.to(device)
        if mask is not None:
            self.support_mask = torch.as_tensor(mask).bool().to(device)

    def set_rationale_table(self, tokens, mask):
        """Train-row rationales [n_train, L] (InfoNCE positives by row, negatives sampled)."""
        device = self.model.embed_tokens.weight.device
        self.rationale_table = torch.as_tensor(tokens).long().to(device)
        self.rationale_mask = torch.as_tensor(mask).bool().to(device)

    def pool_rationales(self, ids):
        # ids: [...] rows of the rationale table -> [..., dim] (IntuRec's convertseq2emb)
        shape = ids.shape
        tokens, mask = self.rationale_table[ids.reshape(-1)], self.rationale_mask[ids.reshape(-1)]
        return self.injectnet.pool(self.model.embed_tokens(tokens), mask).reshape(*shape, -1)

    def rationale_infonce(self, thought, own_rows):
        own_rows = own_rows.to(self.rationale_table.device).long()
        negs = torch.randint(0, self.rationale_table.shape[0], (own_rows.shape[0], self.aux_negatives),
                             device=own_rows.device)
        z = F.normalize(thought.float(), dim=-1)
        pos = (z * F.normalize(self.pool_rationales(own_rows).float(), dim=-1)).sum(-1, keepdim=True)
        neg = torch.einsum("bd,bnd->bn", z, F.normalize(self.pool_rationales(negs).float(), dim=-1))
        logits = torch.cat([pos, neg], dim=-1) / self.aux_temperature
        return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))

    def set_support_ids(self, ids):
        """Per-example support indices for the next forward pass (generation)."""
        self._support_ids = None if ids is None else torch.as_tensor(ids).long()

    # ------------------------------------------------------------------ thought
    def thought_vector(self, query, support_ids):
        """query: [bsz, 1, dim]; support_ids: [bsz, k] (-1 = padding) or None -> [bsz, dim]."""
        if support_ids is not None and support_ids.shape[-1] == 0:
            support_ids = None
        if support_ids is not None and self.support_table.numel() == 0:
            raise ValueError("support_table has not been set")
        if support_ids is not None:
            support_ids = support_ids.to(self.support_table.device)
            valid = support_ids >= 0
            # a row with no valid support would give an all -inf softmax
            valid[:, 0] |= ~valid.any(dim=1)
            ids = support_ids.clamp(min=0)
        if self.support_mode == "text":
            if support_ids is None:
                return self.no_support_proj(query).squeeze(1)
            tokens = self.support_table[ids]  # [bsz, k, L]
            mask = self.support_mask[ids]
            embeds = self.model.embed_tokens(tokens)  # [bsz, k, L, dim]
            return self.injectnet(query, embeds, mask, valid)
        supports = None
        if support_ids is not None:
            vecs = self.support_table[ids].to(self.support_proj.weight.dtype)
            supports = self.support_proj(vecs)
        return self.support_attn(query, supports, None if support_ids is None else valid)

    def generate_embs(self, input_ids, attention_mask):
        output_mid = super(LatentModel, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        thought_id = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_id)
        if where.shape[0] != input_ids.shape[0]:
            raise ValueError("each example must contain exactly one <|Thought|> token")
        rows, cols = where[:, 0], where[:, 1]
        query = output_mid["hidden_states"][-1][rows, cols].unsqueeze(1)  # [bsz, 1, dim]

        ids = self._support_ids
        if ids is not None and ids.shape[0] != input_ids.shape[0]:
            # beam search expands each prompt num_beams times, consecutively
            ids = ids.repeat_interleave(input_ids.shape[0] // ids.shape[0], dim=0)
        if ids is not None:
            ids = ids.to(input_ids.device)[rows]
        latent = self.thought_vector(query, ids)
        if self._own_rows is not None:
            if rows.tolist() != list(range(input_ids.shape[0])):
                raise ValueError("InfoNCE expects one thought per row in order")
            self._thought = latent
        latent = self.observe_reasoning_latent(latent)
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[rows, cols] = latent.to(input_embs.dtype)
        return input_embs

    def forward(self, support_ids=None, own_row=None, **kwargs):
        if support_ids is not None:
            self._support_ids = support_ids
        use_aux = (self.training and self.rationale_beta > 0 and own_row is not None
                   and self.support_mode == "text" and self.rationale_table.numel() > 0)
        self._own_rows = own_row if use_aux else None
        try:
            out = super().forward(**kwargs)
            if use_aux and out.loss is not None:
                aux = self.rationale_infonce(self._thought, own_row)
                out.loss = out.loss + self.rationale_beta * aux
                self.aux_log.append(aux.item())
            return out
        finally:
            self._own_rows = self._thought = None
            if support_ids is not None:
                self._support_ids = None
