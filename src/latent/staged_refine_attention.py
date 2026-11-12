"""Staged refinement of the R3 thought with a SASRec-cluster head.

Two poolers read the history hidden states (one ``SelfAttentionLayer`` each):

* ``r_item``: the R3 thought, spliced into the <|Thought|> slot (generation loss);
* ``r_cl``:   feeds a cluster head ``LayerNorm -> Linear(K)`` that predicts the
              target item's SASRec cluster (cross-entropy).

Stage 1 (``stage`` buffer = 1): the thought is ``r_item``; the cluster head trains
independently. Stage 2 (``stage`` = 2) refines the thought with the cluster
prediction ``p`` (detached) and its confidence ``c = max_j p_j`` (detached):

* ``mode="vrec"``     (VRec, arXiv 2603.07725, Eq. 9-10): the prototype is the
  head's own weight columns, ``g = sum_j p_j W_{:,j}``, and
  ``r' = (1 - c) r_item + c g``. No normalization, no external embedding.
* ``mode="centroid"`` (same form as the oracle_raw side-information run):
  ``v = sum_j p_j mu_j`` over the SASRec centroids,
  ``r' = r_item + c * gain * normalize(P v)``.

During training, ``oracle_alpha`` > 0 replaces ``p`` by the true cluster (one-hot,
c = 1) for a random fraction alpha of the rows (warm-up). ``force_oracle`` does
the same for every row at evaluation (oracle-at-test ceiling); ``override_p`` /
``override_conf`` inject another example's prediction (swap tests).

Cluster codes, centroids and the stage are persistent buffers, so a checkpoint
carries everything its evaluation needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SelfAttentionLayer

MODES = ("vrec", "centroid")


class StagedRefineAttention(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.mode = spec["mode"]
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.num_clusters = int(spec["num_clusters"])
        self.item_pooler = SelfAttentionLayer(hidden_size)
        self.cluster_pooler = SelfAttentionLayer(hidden_size)
        self.cluster_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.num_clusters))
        dim = int(spec["centroid_dim"])
        self.register_buffer("cluster_codes", torch.full((int(spec["n_items"]),), -1, dtype=torch.long))
        self.register_buffer("centroids", torch.zeros(self.num_clusters, dim))
        self.register_buffer("stage", torch.tensor(1, dtype=torch.long))
        if self.mode == "centroid":
            self.projection = nn.Linear(dim, hidden_size)
            self.gain = nn.Parameter(torch.tensor(float(spec.get("init_norm", 30.0))))
        # runtime controls (not saved)
        self.oracle_alpha = 0.0
        self.oracle_codes = None
        self.force_oracle = False
        self.override_p = None
        self.override_conf = None
        self.transform = None  # evaluation hook: (p, conf) -> (p, conf), applied last
        self.last = {}

    @property
    def end_k(self):
        return self.item_pooler.end_k

    @end_k.setter
    def end_k(self, value):
        self.item_pooler.end_k = value
        self.cluster_pooler.end_k = value

    @staticmethod
    def _expand(x, n):
        # beam search expands each prompt num_beams times, consecutively
        return x if x is None or x.shape[0] == n else x.repeat_interleave(n // x.shape[0], 0)

    def refinement_inputs(self, logits):
        """Cluster distribution and confidence used by the refinement (after oracle/override)."""
        n = logits.shape[0]
        p = F.softmax(logits.detach().float(), -1)
        conf = p.max(-1).values
        codes = self._expand(self.oracle_codes, n)
        if codes is not None and (self.force_oracle or (self.training and self.oracle_alpha > 0)):
            codes = codes.to(p.device)
            use = codes >= 0
            if not self.force_oracle:
                use &= torch.rand(n, device=p.device) < self.oracle_alpha
            if use.any():
                p = torch.where(use[:, None], F.one_hot(codes.clamp_min(0), self.num_clusters).float(), p)
                conf = torch.where(use, torch.ones_like(conf), conf)
        if self.override_p is not None:
            p = self._expand(self.override_p, n).to(p.device).float()
            conf = self._expand(self.override_conf, n).to(p.device).float()
        if self.transform is not None:
            p, conf = self.transform(p, conf)
        return p, conf

    def refine(self, r_item, p, conf):
        r = r_item.float()
        if self.mode == "vrec":
            proto = p @ self.cluster_head[1].weight.detach().float()  # [B, H]
            out, comp = (1 - conf)[:, None] * r + conf[:, None] * proto, conf[:, None] * proto
        else:
            v = (p @ self.centroids.float()).to(self.projection.weight.dtype)
            comp = conf[:, None] * F.normalize(self.projection(v).float(), dim=-1) * self.gain.float()
            out = r + comp
        return out, comp

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        r_item = self.item_pooler(hidden_states, attention_mask, thought_id_idx)
        r_cl = self.cluster_pooler(hidden_states, attention_mask, thought_id_idx)
        head = self.cluster_head
        logits = head(r_cl.to(head[1].weight.dtype)).float()
        self.last = {"logits": logits, "r_item": r_item}
        if int(self.stage) < 2:
            return r_item
        p, conf = self.refinement_inputs(logits)
        out, comp = self.refine(r_item, p, conf)
        self.last.update(p=p, conf=conf, component=comp.detach(), refined=out.detach())
        return out.to(r_item.dtype)


def build_staged_refine_attention(hidden_size, spec):
    return StagedRefineAttention(hidden_size, spec)
