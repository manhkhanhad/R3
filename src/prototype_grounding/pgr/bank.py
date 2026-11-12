"""Prototype bank loading for prototype-guided reasoning (Phase 2).

Loads either the raw RALR evidence bank or the augmented one produced by
``augment_bank.py`` and exposes every row that has an embedding, tagged by
verification tier.

Two file shapes this loader has to handle:

- The **raw** RALR bank: 514 rows, one per scored cluster. ``status`` is
  ``"ok"`` (embedding present) or ``"discarded_low_quality"`` (``embedding:
  null``). No ``tier`` field. Loading this file keeps only the 164 ``ok``
  rows -- the 350 discarded ones have nothing to embed.
- The **augmented** bank (``augment_bank.py``): all 514 rows re-embedded
  uniformly from one model (currently Alibaba-NLP/gte-large-en-v1.5, 1024-dim
  -- deliberately *not* matched to whatever RALR's original pipeline used, since
  the goal is internal self-consistency for our own retrieval, not continuity
  with RALR's clustering-time embedding space), each tagged ``tier: "verified"``
  (``status == "ok"``, 164 rows) or ``"unverified"`` (350 rows, quality_score
  0.0-0.53 -- failed RALR's own verification gate, included anyway on the bet
  that quality-weighted MMR + the quality_score feature fed into proj_up can
  do that suppression job instead of a hard offline cutoff).

Either way: a row is usable here iff it has a non-null ``embedding``. `tier`
defaults to `"verified"` when the field is absent (true by construction for
the raw file, since every embedded row there passed `status == "ok"`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Checked against the actual augmented CDs_and_Vinyl bank: 8/514 rationales contain
# CJK characters, all 8 in the unverified tier (0/164 verified) -- plausibly part of
# why those specific rows failed RALR's verification gate in the first place (a
# Chinese-language rationale would likely confuse an English-context ranking check).
# gte-large-en-v1.5 is English-optimized, so these 8 rows' embeddings are less
# trustworthy than the rest of the bank regardless of quality_score.
_CJK_RE = re.compile(r"[一-鿿]")


@dataclass
class PrototypeBank:
    ids: list[str]
    facets: list[str]
    anchors: list[dict[str, Any]]
    rationales: list[str]
    tiers: list[str]             # "verified" | "unverified"
    support: torch.Tensor        # [N] long
    quality_score: torch.Tensor  # [N] float
    embeddings: torch.Tensor     # [N, D] float

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return self.embeddings.shape[-1]

    def filtered(
        self,
        min_quality: float | None = None,
        require_anchor: bool = False,
        require_verified: bool = False,
        exclude_non_english: bool = False,
    ) -> "PrototypeBank":
        """Return a new bank restricted to entries passing the given eligibility filter.

        Applied *before* retrieval, not as a retrieval-time weighting -- this
        controls which prototypes are even candidates for MMR selection.
        `require_verified=True` restricts to the `tier == "verified"` subset,
        i.e. falls back to RALR's own offline verification gate instead of
        trusting quality-weighted MMR to suppress the unverified tier.
        `exclude_non_english=True` drops rationales containing CJK characters --
        see the module-level note; only ever removes rows already scored low by
        quality_score, but their embeddings are unreliable for a different reason
        (wrong-language input to an English-only embedding model) that quality
        weighting alone doesn't account for.
        """
        keep = [
            i for i in range(len(self))
            if (min_quality is None or self.quality_score[i].item() >= min_quality)
            and (not require_anchor or bool(self.anchors[i]))
            and (not require_verified or self.tiers[i] == "verified")
            and (not exclude_non_english or not _CJK_RE.search(self.rationales[i]))
        ]
        if len(keep) == len(self):
            return self
        idx = torch.tensor(keep, dtype=torch.long)
        return PrototypeBank(
            ids=[self.ids[i] for i in keep],
            facets=[self.facets[i] for i in keep],
            anchors=[self.anchors[i] for i in keep],
            rationales=[self.rationales[i] for i in keep],
            tiers=[self.tiers[i] for i in keep],
            support=self.support[idx],
            quality_score=self.quality_score[idx],
            embeddings=self.embeddings[idx],
        )


def load_prototype_bank(path: str | Path) -> PrototypeBank:
    """Load a RALR evidence-bank JSONL file (raw or augmented), keeping rows with an embedding."""
    path = Path(path)
    ids: list[str] = []
    facets: list[str] = []
    anchors: list[dict[str, Any]] = []
    rationales: list[str] = []
    tiers: list[str] = []
    support: list[int] = []
    quality_score: list[float] = []
    embeddings: list[list[float]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("embedding") is None:
                continue
            ids.append(row["id"])
            facets.append(row.get("facet", "general"))
            anchors.append(row.get("anchor") or {})
            rationales.append(row["rationale"])
            tiers.append(row.get("tier", "verified"))
            support.append(int(row.get("support", 0)))
            quality_score.append(float(row["quality_score"]))
            embeddings.append(row["embedding"])

    if not ids:
        raise ValueError(f"no rows with an embedding found in {path}")

    dims = {len(e) for e in embeddings}
    if len(dims) > 1:
        raise ValueError(
            f"mixed embedding dimensions in {path}: {dims} -- rows came from different "
            f"embedding models/spaces and cannot be compared by cosine similarity together"
        )

    return PrototypeBank(
        ids=ids,
        facets=facets,
        anchors=anchors,
        rationales=rationales,
        tiers=tiers,
        support=torch.tensor(support, dtype=torch.long),
        quality_score=torch.tensor(quality_score, dtype=torch.float32),
        embeddings=torch.tensor(embeddings, dtype=torch.float32),
    )
