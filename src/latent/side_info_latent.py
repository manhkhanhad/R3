"""Side-information injection into the R3 thought slot (decoder-insensitivity tests).

The thought vector becomes

    r' = r + g * normalize(P(v))            (optionally rescaled to token norm)

where ``v`` is a per-example side vector looked up from a caller-provided table:

* ``oracle_category``: the gte embedding of the *target* item's category path.
  Deliberately leaky; it tests whether the decoder reads the latent slot when
  the slot carries information the history lacks.
* ``random_category``: the category of a random item (null control, no
  information about the example).
* ``sasrec_sequence``: the frozen SASRec sequence embedding of the row's
  history (collaborative knowledge absent from the title text).
* ``none``: no side vector; with ``token_scale`` this is plain R3 with the
  thought rescaled to input-embedding norm (the H3 cell).

``token_scale`` rescales the spliced vector to the mean input-embedding norm.
The table is a non-persistent buffer: training and evaluation code rebuild it
from the source recorded in ``config.latent_side_info``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LatentModel

KINDS = ("oracle_category", "random_category", "sasrec_sequence", "none")


class SideInfoLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "latent_side_info", None)
        if not spec or spec.get("kind") not in KINDS:
            raise ValueError(f"latent_side_info.kind must be one of {KINDS}")
        self.side_kind = spec["kind"]
        self.side_token_scale = bool(spec.get("token_scale", False))
        self.side_token_norm = float(spec.get("token_norm", 1.0))
        dimension = int(spec.get("dimension", 1))
        if self.side_kind != "none":
            self.side_projection = nn.Linear(dimension, config.hidden_size)
            self.side_gain = nn.Parameter(torch.tensor(float(spec.get("init_norm", 30.0))))
        self.register_buffer("side_table", torch.empty(0, dimension), persistent=False)
        self.register_buffer("side_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self._side_index = None
        self.last_side_component = None

    def set_side_table(self, table, valid=None):
        table = torch.as_tensor(table).float()
        if valid is None:
            valid = torch.ones(table.shape[0], dtype=torch.bool)
        device = self.model.embed_tokens.weight.device
        self.side_table = table.to(device)
        self.side_valid = torch.as_tensor(valid).bool().to(device)

    def set_side_index(self, index):
        """Per-example table rows for the next forward pass (evaluation/generation)."""
        self._side_index = None if index is None else torch.as_tensor(index).long()

    def side_component(self, index, batch_size=None):
        """g * normalize(P(v)) for the given table rows; zero rows where invalid."""
        if self.side_kind == "none":
            return None
        if self.side_table.numel() == 0:
            raise ValueError("side_table has not been set")
        device = self.side_table.device
        if self.side_kind == "random_category":
            n = batch_size if batch_size is not None else len(index)
            pool = torch.nonzero(self.side_valid).squeeze(-1)
            index = pool[torch.randint(len(pool), (n,), device=device)]
        index = index.to(device)
        valid = self.side_valid[index]
        vec = self.side_table[index].to(self.side_projection.weight.dtype)
        comp = F.normalize(self.side_projection(vec).float(), dim=-1) * self.side_gain.float()
        return comp * valid.unsqueeze(-1).float()

    def combine(self, latent, component):
        out = latent.float() if component is None else latent.float() + component
        if self.side_token_scale:
            out = F.normalize(out, dim=-1) * self.side_token_norm
        return out.to(latent.dtype)

    def observe_reasoning_latent(self, latent):
        component = None
        if self.side_kind != "none":
            index = self._side_index
            if index is None and self.side_kind != "random_category":
                raise ValueError("side_index is required for this side-information kind")
            if index is not None and latent.shape[0] != len(index):
                # beam search expands each prompt num_beams times, consecutively
                repeat = latent.shape[0] // len(index)
                index = index.repeat_interleave(repeat)
            component = self.side_component(index, batch_size=latent.shape[0])
        self.last_side_component = None if component is None else component.detach()
        return self.combine(latent, component)

    def forward(self, side_index=None, **kwargs):
        if side_index is not None:
            self._side_index = side_index
        try:
            return super().forward(**kwargs)
        finally:
            if side_index is not None:
                self._side_index = None
