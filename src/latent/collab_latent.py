"""Collaborative candidates for R3 (document/collaborative_reasoning_experiments.md).

Two ways to give the decoder the top-K items of a CF scorer (build_candidates.py):

* ``text`` (idea 2, CF text control): the candidates' titles are written into the
  prompt.  Text is a channel the decoder certainly reads, so this is the ceiling
  for what the candidates can add.
* ``slots`` (ideas 3/4, candidate slots): K ``<|Cand|>`` placeholder tokens are
  written into the prompt and their input embeddings are replaced by a learned
  projection of each candidate's frozen SASRec item embedding,
  ``e_k = normalize(P(v_item_k)) * token_norm``.  One slot per item, rather than
  one pooled vector, lets the decoder copy an item: the item-level CF oracle
  showed it can map a SASRec item embedding to its title.

The slots exist in both R3 passes: the first pass (which pools the thought ``r``)
and the second pass (which decodes) both see the candidate embeddings.

Rows without enough candidates are padded with -1 (a zero slot / omitted title).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset

CAND_TOKEN = "<|Cand|>"
CAND_LINE = "Items bought by users with similar purchase histories: {items}"


def candidate_line(mode, items, titles):
    """The prompt line for one row (``items``: item ids, -1 = none)."""
    if mode == "text":
        shown = [f'"{titles[i]}"' for i in items if i >= 0]
        return CAND_LINE.format(items=", ".join(shown) if shown else "none")
    if mode == "slots":
        return CAND_LINE.format(items=CAND_TOKEN * len(items))
    raise ValueError(f"unknown candidate mode {mode}")


class CollabCandidateLatentRDataset(LatentRDataset):
    """R3 rows with a candidate line; ``cand_items`` is added for slot models."""

    def __init__(self, *args, candidates=None, titles=None, mode="text", **kwargs):
        self.candidates = candidates  # LongTensor [rows of this split, K]
        self.titles = titles
        self.mode = mode
        super().__init__(*args, **kwargs)

    def get_history(self, row):
        out = super().get_history(row)
        items = self.candidates[int(row.name)].tolist()  # row.name: position in the full CSV
        out["input"] += "\n" + candidate_line(self.mode, items, self.titles)
        return out

    def pre(self, idx):
        example = super().pre(idx)
        if self.mode == "slots":
            example["cand_items"] = self.candidates[int(self.data.index[idx])].tolist()
        return example


class CandidateSlotLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_candidate_slots", None)
        if not spec:
            raise ValueError("config.latent_candidate_slots is required")
        self.cand_k = int(spec["k"])
        self.cand_token_id = int(spec["cand_token_id"])
        self.cand_token_norm = float(spec.get("token_norm", 1.0))
        dimension = int(spec["dimension"])
        self.cand_projection = nn.Linear(dimension, config.hidden_size)
        self.register_buffer("cand_table", torch.empty(0, dimension), persistent=False)
        self.register_buffer("cand_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self._cand_items = None
        self.last_thought = None

    def set_cand_table(self, table, valid):
        device = self.model.embed_tokens.weight.device
        self.cand_table = torch.as_tensor(table).float().to(device)
        self.cand_valid = torch.as_tensor(valid).bool().to(device)

    def set_cand_items(self, items):
        """Per-example candidate ids for the next forward pass (evaluation/generation)."""
        self._cand_items = None if items is None else torch.as_tensor(items).long()

    def candidate_embeddings(self, items):
        """[B, K] item ids -> [B, K, H] slot embeddings (zero where the id is -1 or unknown)."""
        if self.cand_table.numel() == 0:
            raise ValueError("cand_table has not been set")
        items = items.to(self.cand_table.device)
        safe = items.clamp(min=0)
        valid = (items >= 0) & self.cand_valid[safe]
        vec = self.cand_table[safe].to(self.cand_projection.weight.dtype)
        emb = F.normalize(self.cand_projection(vec).float(), dim=-1) * self.cand_token_norm
        return emb * valid.unsqueeze(-1).float()

    def embed_with_candidates(self, input_ids):
        embeds = self.model.embed_tokens(input_ids)
        where = input_ids == self.cand_token_id
        if not where.any():
            return embeds
        if self._cand_items is None:
            raise ValueError("candidate slots in the prompt but no cand_items were given")
        items = self._cand_items
        if items.shape[0] != input_ids.shape[0]:
            # beam search expands each prompt num_beams times, consecutively
            items = items.repeat_interleave(input_ids.shape[0] // items.shape[0], dim=0)
        counts = where.sum(1)
        if not bool((counts == items.shape[1]).all()):
            raise ValueError(f"expected {items.shape[1]} candidate slots per row, found {counts.tolist()}")
        embeds = embeds.clone()
        embeds[where] = self.candidate_embeddings(items).reshape(-1, embeds.shape[-1]).to(embeds.dtype)
        return embeds

    def generate_embs(self, input_ids, attention_mask):
        thought_id = self.model.embed_tokens.num_embeddings - 1
        where_thought = torch.nonzero(input_ids == thought_id)
        input_embs = self.embed_with_candidates(input_ids)
        hidden = self.model(inputs_embeds=input_embs, attention_mask=attention_mask, use_cache=False).last_hidden_state
        latent = self.attention(hidden_states=hidden, attention_mask=attention_mask, thought_id_idx=where_thought[:, 1])
        latent = self.project_reasoning_latent(self.observe_reasoning_latent(latent))
        self.last_thought = latent.detach()
        input_embs = input_embs.clone()
        input_embs[where_thought[:, 0], where_thought[:, 1]] = latent.to(input_embs.dtype)
        return input_embs

    def forward(self, cand_items=None, **kwargs):
        if cand_items is not None:
            self._cand_items = cand_items
        try:
            return super().forward(**kwargs)
        finally:
            if cand_items is not None:
                self._cand_items = None
