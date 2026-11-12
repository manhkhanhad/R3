"""Training model and dataset for multi-aspect R3.

loss = CE_recommendation(decoder | fused r) + sum_k w_k * CE_k(head_k(r_k), code_k(target))

The aspect codes of the target come from the ``aspect_codes`` buffer of the
multi-aspect attention (see multi_aspect_attention.py); unknown codes (-1)
are ignored.
"""

import torch
import torch.nn.functional as F

from layers import LatentModel
from reasoning_dataset import LatentRDataset


class MultiAspectLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = config.latent_multi_aspect
        self.aspect_loss_weights = [float(w) for w in spec.get("loss_weights", [1.0] * len(spec["names"]))]
        self.last_aspect_stats = None
        # Trainer passes num_items_in_batch (forward takes **kwargs), so it does not
        # divide the loss by gradient accumulation; the base CE normalizes itself but
        # the aspect loss must be scaled here (same fix as margin_loss_attention.py).
        self.grad_accum_steps = 1

    def forward(self, aspect_item_id=None, **kwargs):
        accumulating = self.training and kwargs.get("num_items_in_batch") is not None
        scale = 1.0 / self.grad_accum_steps if accumulating else 1.0
        output = super().forward(**kwargs)
        if aspect_item_id is None or output.loss is None:
            return output
        att = self.attention
        ids = aspect_item_id.to(att.aspect_codes.device).long()
        stats, total = {}, output.loss.new_zeros(())
        for k, (name, logits) in enumerate(zip(att.names, att.last_logits)):
            target = att.aspect_codes[k, ids]
            valid = target >= 0
            if valid.any():
                loss_k = F.cross_entropy(logits[valid], target[valid])
                total = total + self.aspect_loss_weights[k] * loss_k
                stats[f"aspect_loss/{name}"] = loss_k.detach()
                stats[f"aspect_acc/{name}"] = (logits[valid].argmax(-1) == target[valid]).float().mean().detach()
            stats[f"gate/{name}"] = att.last_gate[:, k].mean().detach()
            stats[f"confidence/{name}"] = att.last_confidence[:, k].mean().detach()
            stats[f"gate_bias/{name}"] = att.gate_bias[k].detach()
        stats["gate/scale"] = att.gate_scale().detach()
        stats["loss/recommendation"] = output.loss.detach() / scale
        self.last_aspect_stats = stats
        output.loss = output.loss + scale * total
        return output


class MultiAspectLatentRDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        example["aspect_item_id"] = int(self.data.iloc[idx]["item_id"])
        return example
