"""VRec-style verifier on the R3 thought vector with coarse group prototypes.

Follows VRec (arXiv 2603.07725) more closely than verified_evidence_latent.py:

* group labels: k ~ 20 coarse CF groups (spherical k-means of SASRec item embeddings,
  src/collab_reasoning/build_group_table.py);
* the verifier is a linear classifier on the (RMS-normalized) thought ``r``; its weight
  rows are the group prototypes;
* the reasoning is nudged, not replaced: R3's own ``r`` is kept when the verifier is
  uncertain,

      p  = softmax(W x + b),        x = r / rms(r)
      c  = 1 - H(p) / log k
      g  = ||r|| * normalize(sum_j p_j W_j)
      r* = (1 - c) * r + c * g

* trained jointly from a pretrained R3 recommender:  L = L_rec + weight * CE(p, group(target)).

The single reasoning step of R3 leaves out VRec's monotonicity loss across steps.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from verified_evidence_latent import evidence_component, VerifiedEvidenceLatentModel


class VRecGroupLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_vrec", None)
        if not spec:
            raise ValueError("latent_vrec config is required")
        self.evidence_k = int(spec["k"])
        self.evidence_weight = float(spec.get("weight", 0.5))
        self.evidence_gate = spec.get("gate", "entropy")
        self.evidence_label_smoothing = float(spec.get("label_smoothing", 0.1))
        self.verifier = nn.Linear(config.hidden_size, self.evidence_k)
        self.register_buffer("item_cluster", torch.empty(0, dtype=torch.long), persistent=False)
        self.evidence_override = None  # None | "zero": keep r unchanged (ablation)
        self._verifier_logits = None
        self._evidence_items = None
        self._seen = 0
        self._stats = []
        self.last_evidence = None

    def reset_verifier(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        w = torch.empty(self.verifier.weight.shape).normal_(0, 1 / math.sqrt(self.verifier.in_features), generator=g)
        with torch.no_grad():
            self.verifier.weight.copy_(w.to(self.verifier.weight))
            self.verifier.bias.zero_()

    def set_group_table(self, table):
        cid = torch.as_tensor(table["cluster_id"]).long()
        valid = torch.as_tensor(table["valid"]).bool() & (cid >= 0)
        cid = torch.where(valid, cid, torch.full_like(cid, -1))
        if int(cid.max()) + 1 != self.evidence_k:
            raise ValueError(f"table has {int(cid.max()) + 1} groups, config says {self.evidence_k}")
        self.item_cluster = cid.to(self.model.embed_tokens.weight.device)

    def load_tables_from_config(self):
        self.set_group_table(torch.load(self.config.latent_vrec["source"], map_location="cpu", weights_only=False))

    def verifier_logits(self, latent):
        x = latent.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return self.verifier(x.to(self.verifier.weight.dtype)).float()

    def slot_from_probs(self, latent, probs):
        """r* = (1 - c) r + c ||r|| normalize(p W); ``probs=None`` keeps r."""
        if probs is None:
            return latent
        r = latent.float()
        probs = probs.to(r.device).float()
        proto, conf = evidence_component(probs, self.verifier.weight.float(), 1.0, self.evidence_gate)
        g = F.normalize(proto, dim=-1) * r.norm(dim=-1, keepdim=True)
        c = conf.unsqueeze(-1)
        return ((1 - c) * r + c * g).to(latent.dtype)

    def observe_reasoning_latent(self, latent):
        logits = self.verifier_logits(latent)
        probs = logits.softmax(-1)
        self._verifier_logits = logits
        _, conf = evidence_component(probs.detach(), torch.zeros(self.evidence_k, 1, device=probs.device), 1.0, self.evidence_gate)
        self.last_evidence = {"probs": probs.detach(), "conf": conf}
        if self.evidence_override == "zero":
            return latent
        return self.slot_from_probs(latent, probs)

    def forward(self, evidence_item_id=None, **kwargs):
        self._verifier_logits = None
        output = super().forward(**kwargs)
        if evidence_item_id is None or output.loss is None or self._verifier_logits is None:
            return output
        if self.training:
            self._seen += len(evidence_item_id)
        y = self.item_cluster[evidence_item_id.long().to(self.item_cluster.device)]
        ok = y >= 0
        logits = self._verifier_logits
        if ok.any():
            loss_v = F.cross_entropy(logits[ok], y[ok], label_smoothing=self.evidence_label_smoothing)
            if self.training:
                self._log(logits[ok].detach(), y[ok], loss_v.detach())
        else:
            loss_v = output.loss.new_zeros(())
        share = VerifiedEvidenceLatentModel._accumulation_share(kwargs)
        output.loss = output.loss + self.evidence_weight * loss_v * share
        return output

    def _log(self, logits, y, loss_v, every=200):
        probs = logits.softmax(-1)
        _, conf = evidence_component(probs, torch.zeros(probs.shape[1], 1, device=probs.device), 1.0, "entropy")
        right = (probs.argmax(-1) == y).float()
        self._stats.append(torch.stack([loss_v.float(), right.mean(), conf.mean(), probs.max(-1).values.mean()]).cpu())
        if len(self._stats) >= every:
            s = torch.stack(self._stats).mean(0).tolist()
            print(f"[verifier] seen={self._seen} ce={s[0]:.3f} top1={s[1]:.3f} conf={s[2]:.3f} maxp={s[3]:.3f}", flush=True)
            self._stats = []
