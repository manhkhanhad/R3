"""Trained verifier over a fixed evidence corpus for the R3 thought slot (VRec-style).

Instead of trusting the thought vector ``r`` the backbone pools from the prompt, a
verifier reads ``r`` and predicts which piece of evidence from a fixed corpus
supports the recommendation.  The corpus is the k = 150 CF clusters of
``Musical_Instruments_sasrec_cluster150.pt``; evidence j is the cluster centroid
read through the side projection of the ``oracle_cfcluster150`` checkpoint, whose
decoder already reads that slot (document/collaborative_reasoning_experiments.md,
"Trained evidence verifier"):

    p      = softmax(verifier(r))                       distribution over evidence
    c      = 1 - H(p) / log k                           confidence (entropy gate)
    r'     = r + g * c * sum_j p_j * normalize(P(e_j))

A confident verifier puts one evidence vector in the slot; an uncertain one puts in
a short blend that the decoder learns to ignore, falling back on the history.  The
recommendation loss reaches the verifier through p and c, so it learns where its
evidence helps the decoder, not only which cluster the target is in:

    L = L_rec + weight * CE(p, cluster(target))          (label smoothing)

Curriculum: at the start of training a fraction of rows gets the true cluster as a
one-hot p (starting at ``oracle_mix_start``, decaying linearly to 0 over
``oracle_mix_examples`` examples), so the decoder keeps reading the slot while the
verifier is still untrained.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from side_info_latent import SideInfoLatentModel

GATES = ("entropy", "none")


def evidence_component(probs, evidence, gain, gate="entropy"):
    """``gain * c(p) * probs @ evidence`` and the confidence ``c`` (1 when ungated)."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    mix = probs @ evidence
    if gate == "entropy":
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        conf = (1 - entropy / math.log(probs.shape[-1])).clamp(0, 1)
    else:
        conf = torch.ones(probs.shape[0], device=probs.device, dtype=probs.dtype)
    return gain * conf.unsqueeze(-1) * mix, conf


def oracle_mix_rate(start, seen, total):
    """Fraction of rows given the true evidence after ``seen`` training examples."""
    if total <= 0:
        return 0.0
    return start * max(0.0, 1.0 - seen / total)


class VerifiedEvidenceLatentModel(SideInfoLatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_evidence", None)
        if not spec:
            raise ValueError("latent_evidence config is required")
        if self.side_kind != "oracle_category":
            raise ValueError("evidence is read through an oracle_category side projection")
        self.evidence_k = int(spec["k"])
        self.evidence_weight = float(spec.get("weight", 0.5))
        self.evidence_gate = spec.get("gate", "entropy")
        self.evidence_label_smoothing = float(spec.get("label_smoothing", 0.1))
        self.evidence_mix_start = float(spec.get("oracle_mix_start", 0.0))
        self.evidence_mix_examples = int(spec.get("oracle_mix_examples", 0))
        width = int(spec.get("verifier_width", 512))
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_size, width), nn.GELU(), nn.Linear(width, self.evidence_k)
        )
        dimension = int(getattr(config, "latent_side_info")["dimension"])
        self.register_buffer("evidence_centroids", torch.empty(0, dimension), persistent=False)
        self.register_buffer("item_cluster", torch.empty(0, dtype=torch.long), persistent=False)
        self.evidence_override = None  # None | "zero": drop the evidence component (ablation)
        self._evidence_items = None
        self._verifier_logits = None
        self._seen = 0
        self._stats = []
        self.last_evidence = None

    def reset_verifier(self, seed=0):
        """Explicit init: the verifier is never in the starting checkpoint."""
        g = torch.Generator().manual_seed(seed)
        for layer in self.verifier:
            if isinstance(layer, nn.Linear):
                w = torch.empty(layer.weight.shape).normal_(0, 1 / math.sqrt(layer.in_features), generator=g)
                with torch.no_grad():
                    layer.weight.copy_(w.to(layer.weight))
                    layer.bias.zero_()

    def set_evidence_table(self, table):
        """``table``: the oracle side table with per-item ``embeddings`` and ``cluster_id``."""
        emb = torch.as_tensor(table["embeddings"]).float()
        cid = torch.as_tensor(table["cluster_id"]).long()
        valid = torch.as_tensor(table["valid"]).bool() & (cid >= 0)
        cid = torch.where(valid, cid, torch.full_like(cid, -1))
        k = self.evidence_k
        if int(cid.max()) + 1 != k:
            raise ValueError(f"table has {int(cid.max()) + 1} clusters, config says {k}")
        sums = torch.zeros(k, emb.shape[1]).index_add_(0, cid[valid], emb[valid])
        counts = torch.bincount(cid[valid], minlength=k).clamp_min(1).unsqueeze(-1)
        device = self.model.embed_tokens.weight.device
        self.set_side_table(emb, valid)
        self.evidence_centroids = (sums / counts).to(device)
        self.item_cluster = cid.to(device)

    def evidence_vectors(self):
        """Unit evidence directions normalize(P(e_j)), shape [k, hidden]."""
        vec = self.evidence_centroids.to(self.side_projection.weight.dtype)
        return F.normalize(self.side_projection(vec).float(), dim=-1)

    def verifier_logits(self, latent):
        x = latent.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)  # r's norm varies 20-180
        return self.verifier(x.to(self.verifier[0].weight.dtype)).float()

    def observe_reasoning_latent(self, latent):
        if self.evidence_centroids.numel() == 0:
            raise ValueError("evidence table has not been set")
        logits = self.verifier_logits(latent)
        probs = logits.softmax(-1)
        used = probs
        items = self._evidence_items
        if self.training and items is not None and len(items) == latent.shape[0]:
            q = oracle_mix_rate(self.evidence_mix_start, self._seen, self.evidence_mix_examples)
            if q > 0:
                y = self.item_cluster[items]
                pick = (torch.rand(len(y), device=y.device) < q) & (y >= 0)
                onehot = F.one_hot(y.clamp_min(0), self.evidence_k).float()
                used = torch.where(pick.unsqueeze(-1), onehot, probs)
        component, conf = evidence_component(used, self.evidence_vectors(), self.side_gain.float(), self.evidence_gate)
        if self.evidence_override == "zero":
            component = torch.zeros_like(component)
        self._verifier_logits = logits
        self.last_evidence = {"probs": probs.detach(), "conf": conf.detach(), "component": component.detach()}
        return self.combine(latent, component)

    def forward(self, evidence_item_id=None, **kwargs):
        self._evidence_items = None if evidence_item_id is None else evidence_item_id.long().to(self.item_cluster.device)
        self._verifier_logits = None
        try:
            output = LatentModel.forward(self, **kwargs)
        finally:
            self._evidence_items = None
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
        output.loss = output.loss + self.evidence_weight * loss_v * self._accumulation_share(kwargs)
        return output

    @staticmethod
    def _accumulation_share(kwargs):
        """This micro-batch's share of the accumulated batch.

        With ``num_items_in_batch`` (passed by the HF Trainer under gradient
        accumulation) the LM loss is summed over micro-batches, not averaged, so a
        per-micro-batch mean must be weighted by its share of the label tokens.
        """
        total = kwargs.get("num_items_in_batch")
        labels = kwargs.get("labels")
        if total is None or labels is None:
            return 1.0
        tokens = (labels[..., 1:] != -100).sum()
        return tokens.to(torch.float32) / torch.as_tensor(total, device=tokens.device, dtype=torch.float32)

    def _log(self, logits, y, loss_v, every=200):
        probs = logits.softmax(-1)
        _, conf = evidence_component(probs, torch.zeros(probs.shape[1], 1, device=probs.device), 1.0, "entropy")
        right = (probs.argmax(-1) == y).float()
        self._stats.append(torch.stack([loss_v.float(), right.mean(), conf.mean(), probs.max(-1).values.mean()]).cpu())
        if len(self._stats) >= every:
            s = torch.stack(self._stats).mean(0).tolist()
            q = oracle_mix_rate(self.evidence_mix_start, self._seen, self.evidence_mix_examples)
            print(f"[verifier] seen={self._seen} ce={s[0]:.3f} top1={s[1]:.3f} conf={s[2]:.3f} "
                  f"maxp={s[3]:.3f} oracle_mix={q:.3f}", flush=True)
            self._stats = []


class EvidenceLatentRDataset(LatentRDataset):
    """Adds ``evidence_item_id`` (the target item) for the verifier's training label."""

    def pre(self, idx):
        example = super().pre(idx)
        example["evidence_item_id"] = int(self.data.iloc[idx]["item_id"])
        return example
