"""Optional category-path alignment loss on the original R3 latent."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel


class CategoryAlignedLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "category_alignment", None)
        if not spec:
            raise ValueError("category_alignment config is required")
        dimension = int(spec["dimension"])
        self.category_projection = nn.Linear(config.hidden_size, dimension)
        self.category_alignment_weight = float(spec["weight"])
        self.register_buffer("category_embeddings", torch.empty(0, dimension), persistent=False)
        self.register_buffer("category_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self.last_align_loss = None
        self._alignment_latent = None
        self._capture_alignment_latent = False

    def observe_reasoning_latent(self, latent):
        if self._capture_alignment_latent:
            self._alignment_latent = latent
        return latent

    def set_category_targets(self, embeddings, valid):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.category_projection.out_features:
            raise ValueError("Category embedding dimension does not match projection")
        if valid.ndim != 1 or valid.shape[0] != embeddings.shape[0]:
            raise ValueError("Category validity mask has the wrong shape")
        self.category_embeddings = embeddings.float()
        self.category_valid = valid.bool()

    def forward(self, category_item_id=None, **kwargs):
        self._alignment_latent = None
        self._capture_alignment_latent = category_item_id is not None and kwargs.get("labels") is not None
        try:
            output = super().forward(**kwargs)
        finally:
            self._capture_alignment_latent = False
        if category_item_id is None or output.loss is None:
            return output
        if self.category_embeddings.numel() == 0:
            raise ValueError("Category targets have not been loaded")
        if self._alignment_latent is None:
            raise ValueError("No generated latent was available for alignment")
        ids = category_item_id.to(self.category_valid.device).long()
        if (ids < 0).any() or (ids >= self.category_valid.numel()).any():
            raise IndexError("category_item_id is outside the category embedding table")
        valid = self.category_valid[ids]
        if valid.any():
            latent = self._alignment_latent[valid]
            projected = self.category_projection(latent.to(self.category_projection.weight.dtype)).float()
            projected = F.normalize(projected, dim=-1)
            target = self.category_embeddings[ids[valid]].to(projected.device)
            align_loss = (projected - target).square().sum(dim=-1).mean()
        else:
            align_loss = output.loss.new_zeros(())
        self.last_align_loss = align_loss.detach()
        output.loss = output.loss + self.category_alignment_weight * align_loss
        return output
