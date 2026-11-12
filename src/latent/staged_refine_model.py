"""Training model and dataset for staged-refinement R3 (staged_refine_attention.py).

loss = CE_generation(decoder | thought) + w * CE(cluster_head(r_cl), cluster(target))

In evaluation mode the model accumulates per-head validation statistics
(generation loss, cluster loss, top-1/top-5 accuracy, confidence) that
``StagedRefineCallback`` writes after every evaluation.
"""

import torch
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset


class StagedRefineLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.cluster_loss_weight = float(config.latent_staged_refine.get("loss_weight", 0.1))
        # Trainer passes num_items_in_batch (forward takes **kwargs) and does not divide
        # by gradient accumulation; the generation CE normalizes itself, the cluster
        # loss must be scaled here (as in multi_aspect_model.py).
        self.grad_accum_steps = 1
        self.last_stats = None
        self.reset_eval_stats()

    def reset_eval_stats(self):
        self.eval_sums = {"n_batches": 0, "generation_loss": 0.0, "cluster_loss": 0.0,
                          "n": 0, "top1": 0, "top5": 0, "confidence": 0.0}

    def eval_summary(self):
        s = self.eval_sums
        if not s["n"]:
            return {}
        return {"valid_generation_loss": s["generation_loss"] / max(s["n_batches"], 1),
                "valid_cluster_loss": s["cluster_loss"] / s["n"],
                "valid_cluster_top1": s["top1"] / s["n"], "valid_cluster_top5": s["top5"] / s["n"],
                "valid_cluster_confidence": s["confidence"] / s["n"]}

    def forward(self, aspect_item_id=None, **kwargs):
        att = self.attention
        accumulating = self.training and kwargs.get("num_items_in_batch") is not None
        scale = 1.0 / self.grad_accum_steps if accumulating else 1.0
        target = None
        if aspect_item_id is not None:
            target = att.cluster_codes[aspect_item_id.to(att.cluster_codes.device).long()]
            att.oracle_codes = target
        try:
            output = super().forward(**kwargs)
        finally:
            att.oracle_codes = None
        if target is None or output.loss is None:
            return output
        logits = att.last["logits"]
        valid = target >= 0
        loss_cl = F.cross_entropy(logits[valid], target[valid]) if valid.any() else logits.new_zeros(())
        generation = output.loss.detach() / scale
        with torch.no_grad():
            top5 = logits[valid].topk(5, -1).indices
            hit1 = (top5[:, 0] == target[valid]).float()
            hit5 = (top5 == target[valid][:, None]).any(-1).float()
            conf = F.softmax(logits[valid], -1).max(-1).values
        if self.training:
            stats = {"loss/generation": generation, "cluster/loss": loss_cl.detach(),
                     "cluster/top1": hit1.mean(), "cluster/top5": hit5.mean(), "cluster/confidence": conf.mean(),
                     "refine/stage": float(att.stage), "refine/oracle_alpha": float(att.oracle_alpha),
                     "norm/r_item": att.last["r_item"].float().norm(dim=-1).mean()}
            if "component" in att.last:
                stats["norm/component"] = att.last["component"].norm(dim=-1).mean()
                stats["norm/refined"] = att.last["refined"].norm(dim=-1).mean()
                stats["refine/confidence_used"] = att.last["conf"].mean()
            if getattr(att, "gain", None) is not None:
                stats["refine/gain"] = att.gain.detach().float()
            self.last_stats = stats
        else:
            s, k = self.eval_sums, int(valid.sum())
            s["n_batches"] += 1
            s["generation_loss"] += float(generation)
            s["cluster_loss"] += float(loss_cl) * k
            s["n"] += k
            s["top1"] += float(hit1.sum())
            s["top5"] += float(hit5.sum())
            s["confidence"] += float(conf.sum())
        output.loss = output.loss + scale * self.cluster_loss_weight * loss_cl
        return output


class StagedRefineLatentRDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        example["aspect_item_id"] = int(self.data.iloc[idx]["item_id"])
        return example
