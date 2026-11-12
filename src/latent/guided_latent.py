"""Item R3 guided by a frozen, precomputed latent from another model (e.g. category-R3's r_cat).

Same mechanism as the Experiment 1 side-information tests
(analysis/document/decoder_insensitivity_to_latent.md): the spliced thought becomes

    r' = r + g * normalize(P(v_i))

where v_i is the precomputed guide vector of example i (row i of a table set by
the caller), P is a learned linear map and g a learned scalar (init 30). The
table is a non-persistent buffer; training and evaluation code load it from the
files recorded in config.latent_guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel


def load_guide(path):
    return torch.load(path, map_location="cpu", weights_only=True)["latents"].float()


class GuidedLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_guidance", None)
        if not spec:
            raise ValueError("config.latent_guidance is required")
        dimension = int(spec["dimension"])
        self.guide_projection = nn.Linear(dimension, config.hidden_size)
        self.guide_gain = nn.Parameter(torch.tensor(float(spec.get("init_norm", 30.0))))
        self.register_buffer("guide_table", torch.empty(0, dimension), persistent=False)
        self._guide_index = None
        self._guide_override = None

    def set_guide_table(self, table):
        self.guide_table = torch.as_tensor(table).float().to(self.model.embed_tokens.weight.device)

    def set_guide_index(self, index):
        """Rows of the guide table for the next forward/generate call (None to clear)."""
        self._guide_index = None if index is None else torch.as_tensor(index).long()

    def guide_component(self, index):
        vec = self.guide_table[index.to(self.guide_table.device)].to(self.guide_projection.weight.dtype)
        return F.normalize(self.guide_projection(vec).float(), dim=-1) * self.guide_gain.float()

    def observe_reasoning_latent(self, latent):
        index = self._guide_index
        if index is None:
            raise ValueError("guide index is required (dataset column guide_index or set_guide_index)")
        if latent.shape[0] != len(index):  # beam search repeats each prompt num_beams times, consecutively
            index = index.repeat_interleave(latent.shape[0] // len(index))
        return (latent.float() + self.guide_component(index)).to(latent.dtype)

    def forward(self, guide_index=None, **kwargs):
        if guide_index is not None:
            self._guide_index = guide_index
        try:
            return super().forward(**kwargs)
        finally:
            if guide_index is not None:
                self._guide_index = None
