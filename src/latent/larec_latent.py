"""LaRec-style latent pre-training on top of R3 (document/latent_distillation.md).

Re-implements stage 1 ("Latent Reasoning Pre-training") of LaRec (Xia et al., arXiv
2607.24617) in this codebase, with R3 v1's hyperparameters:

* K latent reasoning steps instead of R3's single pooled thought.  The prompt ends in
  K ``<|Thought|>`` slots; slot k's input is z_k = head(s_{k-1}), where s_{k-1} is the
  final-layer state at the position just before slot k (the last prompt token for k = 1,
  slot k-1 otherwise), so each step conditions on the history and all earlier steps
  (Coconut-style feedback).  LaRec maps latents into the input-embedding space without
  specifying the head; here it is a soft token (Soft Thinking): a zero-initialised residual
  MLP on the state, then z = softmax(lm_head(.)) @ E, the expected input embedding of the
  predicted next token.  This keeps z on the input manifold: feeding back the raw state
  rescaled to token norm raised the initial target CE from 4.0 to 15.3.
* Step-level alignment (LaRec Eq. 4): InfoNCE between the student state after reading
  z_k (final layer, slot k) and the teacher state h_k after CoT step k
  (``teacher_states.py``), cosine similarity, temperature ``tau``.  Negatives are the
  same step's teacher states of ``negatives`` random other training rows (LaRec uses the
  mini-batch; with R3's micro-batch of 2 that would be a single negative).
  Final-layer states are highly anisotropic (random rows' teacher states have cosine
  0.94-0.99), which left the InfoNCE at chance.  So, as is standard for contrastive
  alignment (LaRec does not specify), teacher states are standardised per step and per
  dimension over the training rows, and student states are standardised per dimension by
  running (EMA) mean and variance and passed through a learned linear projection
  (identity at init) before the cosine.  Without the per-dimension scaling, a few
  massive-activation dimensions dominate the cosine and the InfoNCE stays at chance,
  although a linear probe from the history-only state reaches 3.7 vs chance 5.5
  (alignment_probe.py).
* Process direction alignment (LaRec Eq. 5-6): 1 - cos(z_k - z_{k-1}, e_tgt) for
  k = 2..K, with e_tgt the mean input embedding of the target title's tokens.
* L = L_rec + alpha * L_SA + gamma * L_PA (alpha = gamma = 1 in LaRec).  The auxiliary
  terms are only added in training mode (eval loss stays the plain CE R3 selects on),
  and are weighted by the micro-batch's share of label tokens under gradient
  accumulation (as in verified_evidence_latent.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset


class ReasoningHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up = nn.Linear(hidden, hidden)
        self.down = nn.Linear(hidden, hidden)

    def reset(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            w = torch.empty(self.up.weight.shape).normal_(0, 1 / math.sqrt(self.up.in_features), generator=g)
            self.up.weight.copy_(w.to(self.up.weight))
            self.up.bias.zero_()
            self.down.weight.zero_()  # starts as the identity map of the (rescaled) state
            self.down.bias.zero_()

    def forward(self, state):
        return state + self.down(F.gelu(self.up(state)))


class LaRecLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_larec", None)
        if not spec:
            raise ValueError("latent_larec config is required")
        self.k = int(spec["k"])
        self.alpha = float(spec.get("alpha", 1.0))
        self.gamma = float(spec.get("gamma", 1.0))
        self.tau = float(spec.get("tau", 0.1))
        self.negatives = int(spec.get("negatives", 255))
        self.token_norm = float(spec.get("token_norm", 1.0))
        self.head = ReasoningHead(config.hidden_size)
        self.align_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer("student_mean", torch.zeros(self.k, config.hidden_size), persistent=False)
        self.register_buffer("student_var", torch.ones(self.k, config.hidden_size), persistent=False)
        self.register_buffer("student_mean_ready", torch.zeros((), dtype=torch.bool), persistent=False)
        self.mean_momentum = float(spec.get("mean_momentum", 0.99))
        self.teacher_states = None  # [N, K, H] on CPU, set by set_teacher_states
        self.teacher_valid = None
        self._rows = None
        self._trace = None
        self._stats = []
        self.swap_shift = 0  # eval control: give each row the latent chain of the row `swap_shift` later

    def reset_alignment(self):
        with torch.no_grad():
            self.align_proj.weight.copy_(torch.eye(self.align_proj.in_features, dtype=self.align_proj.weight.dtype))
            self.align_proj.bias.zero_()

    def set_teacher_states(self, states, valid):
        """Standardise each step's states (per dimension) over the valid rows."""
        if states.shape[1] != self.k:
            raise ValueError(f"teacher states have {states.shape[1]} steps, config says {self.k}")
        valid = valid.bool()
        x = states[valid].float()
        mean, std = x.mean(0), x.std(0).clamp_min(1e-3)
        self.teacher_states = ((states.float() - mean) / std).to(torch.bfloat16)
        self.teacher_valid = valid

    def soft_token(self, state):
        """Expected input embedding under the next-token distribution (thought token excluded)."""
        logits = self.lm_head(state).float()
        logits[..., -1] = float("-inf")
        return logits.softmax(-1).to(self.model.embed_tokens.weight.dtype) @ self.model.embed_tokens.weight

    def slot_positions(self, input_ids):
        thought = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought)
        if where.shape[0] != input_ids.shape[0] * self.k:
            raise ValueError(f"every sequence needs exactly {self.k} thought tokens")
        pos = where[:, 1].view(input_ids.shape[0], self.k)
        if (pos[:, 1:] - pos[:, :-1] != 1).any():
            raise ValueError("thought tokens must be consecutive")
        return pos

    def generate_embs(self, input_ids, attention_mask):
        """Fill the K slots step by step: z_k = head(state just before slot k)."""
        pos = self.slot_positions(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        embeds = self.model.embed_tokens(input_ids)
        latents, states, heads = [], [], []
        for k in range(self.k):
            upto = int(pos[:, k].max())
            hidden = self.model(inputs_embeds=embeds[:, :upto], attention_mask=attention_mask[:, :upto]).last_hidden_state
            state = hidden[rows, pos[:, k] - 1]
            if k > 0:
                states.append(state)  # state after reading z_k-1
            head = self.head(state)
            heads.append(head)
            z = self.soft_token(head)
            embeds = embeds.index_put((rows, pos[:, k]), z.to(embeds.dtype))
            latents.append(z)
        chain = torch.stack(latents, 1)
        if self.swap_shift:
            swapped = chain.roll(self.swap_shift, dims=0)
            for k in range(self.k):
                embeds = embeds.index_put((rows, pos[:, k]), swapped[:, k].to(embeds.dtype))
        self._trace = {"pos": pos, "latents": chain, "states": states, "heads": torch.stack(heads, 1)}
        return embeds

    def forward(self, larec_row=None, **kwargs):
        self._rows = larec_row
        self._trace = None
        need_aux = self.training and larec_row is not None and kwargs.get("labels") is not None
        if need_aux:
            kwargs["output_hidden_states"] = True
        output = super().forward(**kwargs)
        if not need_aux or self._trace is None:
            return output
        trace = self._trace
        rows = torch.arange(trace["pos"].shape[0], device=trace["pos"].device)
        last_state = output.hidden_states[-1][rows, trace["pos"][:, -1]]  # after reading z_K
        student = torch.stack(trace["states"] + [last_state], 1).float()  # [B, K, H]
        output.hidden_states = None

        loss_sa = self.step_alignment(student, larec_row)
        loss_pa = self.direction_alignment(trace["latents"].float(), kwargs["labels"])
        share = self._accumulation_share(kwargs)
        output.loss = output.loss + share * (self.alpha * loss_sa + self.gamma * loss_pa)
        self._log(loss_sa.detach(), loss_pa.detach(), output.loss.detach())
        return output

    def step_alignment(self, student, row_ids):
        if self.teacher_states is None:
            raise ValueError("teacher states have not been set")
        ids = row_ids.long().cpu()
        ok = self.teacher_valid[ids]
        if not ok.any():
            return student.new_zeros(())
        pool = torch.nonzero(self.teacher_valid).squeeze(1)
        neg_ids = pool[torch.randint(len(pool), (self.negatives,))]
        if self.training:
            with torch.no_grad():
                batch_mean = student.detach().mean(0)
                batch_var = (student.detach() - self.student_mean).square().mean(0)
                if not bool(self.student_mean_ready):
                    self.student_mean.copy_(batch_mean)
                    self.student_var.copy_(student.detach().var(0, unbiased=False).clamp_min(1e-2)
                                           if student.shape[0] > 1 else torch.ones_like(batch_mean))
                    self.student_mean_ready.fill_(True)
                else:
                    self.student_mean.mul_(self.mean_momentum).add_(batch_mean, alpha=1 - self.mean_momentum)
                    self.student_var.mul_(self.mean_momentum).add_(batch_var, alpha=1 - self.mean_momentum)
        normed = (student - self.student_mean) * torch.rsqrt(self.student_var.clamp_min(1e-2))
        student = self.align_proj(normed.to(self.align_proj.weight.dtype)).float()
        pos = self.teacher_states[ids[ok]].to(student.device).float()  # [b, K, H]
        neg = self.teacher_states[neg_ids].to(student.device).float()  # [N, K, H]
        s = F.normalize(student[ok.to(student.device)], dim=-1)
        sim_pos = (s * F.normalize(pos, dim=-1)).sum(-1, keepdim=True)  # [b, K, 1]
        sim_neg = torch.einsum("bkh,nkh->bkn", s, F.normalize(neg, dim=-1))  # [b, K, N]
        logits = torch.cat([sim_pos, sim_neg], -1) / self.tau
        return F.cross_entropy(logits.flatten(0, 1), logits.new_zeros(logits.shape[0] * logits.shape[1], dtype=torch.long))

    def direction_alignment(self, latents, labels):
        """1 - cos(z_k - z_{k-1}, mean embedding of the target title), k = 2..K."""
        if self.k < 2:
            return latents.new_zeros(())
        eos = self.config.eos_token_id
        tgt = (labels != -100) & (labels != eos) if eos is not None else labels != -100
        emb = self.model.embed_tokens(labels.clamp_min(0)).float().detach()
        e_tgt = (emb * tgt.unsqueeze(-1)).sum(1) / tgt.sum(1, keepdim=True).clamp_min(1)
        steps = latents[:, 1:] - latents[:, :-1]
        return (1 - F.cosine_similarity(steps, e_tgt.unsqueeze(1), dim=-1)).mean()

    @staticmethod
    def _accumulation_share(kwargs):
        total = kwargs.get("num_items_in_batch")
        labels = kwargs.get("labels")
        if total is None or labels is None:
            return 1.0
        tokens = (labels[..., 1:] != -100).sum()
        return tokens.to(torch.float32) / torch.as_tensor(total, device=tokens.device, dtype=torch.float32)

    def _log(self, sa, pa, total, every=200):
        self._stats.append(torch.stack([sa.float(), pa.float()]).cpu())
        if len(self._stats) >= every:
            s = torch.stack(self._stats).mean(0).tolist()
            print(f"[larec] step_align={s[0]:.3f} (chance {math.log(self.negatives + 1):.3f}) direction={s[1]:.3f}", flush=True)
            self._stats = []


class LaRecLatentRDataset(LatentRDataset):
    """R3 prompt with K consecutive thought slots; ``larec_row`` indexes the teacher states."""

    def __init__(self, *args, k=6, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def row_id(self, idx):
        if "Unnamed: 0" in self.data.columns:
            return int(self.data.iloc[idx]["Unnamed: 0"])
        return int(self.data.index[idx])

    def pre(self, idx):
        example = super().pre(idx)
        ids = example["input_ids"]
        thought = self.tokenizer.encode("<|Thought|>", bos=False, eos=False)[0]
        at = ids.index(thought) + 1
        extra = [thought] * (self.k - 1)
        example["input_ids"] = ids[:at] + extra + ids[at:]
        example["attention_mask"] = [1] * len(example["input_ids"])
        if "labels" in example:
            example["labels"] = example["labels"][:at] + [-100] * (self.k - 1) + example["labels"][at:]
        example["larec_row"] = self.row_id(idx)
        return example
