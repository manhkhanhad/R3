"""Trainer subclass adding the Phase 3 auxiliary losses on top of CE (Training curriculum,
Phase 3): L = L_E1 + mu*L_recons + nu*L_commit + xi*L_diversity + zeta*L_balance, kept
active throughout joint fine-tuning (not dropped after Phase 2) so gradient drifting
the mapping doesn't undo Phase 2's reconstruction pretraining -- see the design doc's
Cross-space projection collapse risk.

L_diversity added after the head-collapse failure mode was confirmed in production
(see losses.py's module docstring): without it, nothing stops Stage 3's continued
training from re-collapsing query_proj even if Stage 2's warm-start fixed it, since CE
gradient never reaches query_proj at all (MMR's hard argmax blocks it) and L_commit
alone has no term rewarding input-sensitivity. L_balance added after L_diversity was
confirmed (three real trials) to converge at the loss level without fixing practical
retrieval diversity -- see losses.py::usage_balance_loss for the mechanism. Both are
kept, not just the newer one: L_diversity's smoothness in continuous query-vector space
may still help optimization even though it alone doesn't solve population-level
collapse.

L_balance here is weaker than in Stage 2's dedicated warm-start (train_3stage.py):
it only sees this step's training micro_batch (typically 4, vs. Stage 2's ~128) as its
reference pool for "how often does each item win" -- a much noisier statistic at that
size. Stage 2 is where the heavy lifting for fixing collapse happens; this is meant to
stop Stage 3's joint training from *re*-collapsing an already-fixed retriever, not to
fix collapse from scratch on 4-example batches.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import Trainer

from losses import commitment_loss_from_retrieval, query_diversity_loss, usage_balance_loss
from projection import reconstruction_loss


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class PrototypeGuidedTrainer(Trainer):
    def __init__(
        self, *args, recon_weight: float = 1.0, commit_weight: float = 0.1,
        diversity_weight: float = 1.0, balance_weight: float = 1.0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.recon_weight = recon_weight
        self.commit_weight = commit_weight
        self.diversity_weight = diversity_weight
        self.balance_weight = balance_weight

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        pgr_model = _unwrap(model)
        if getattr(pgr_model, "prototype_bank", None) is not None:
            recon = reconstruction_loss(pgr_model.proto_projector, pgr_model.prototype_bank)
            loss = loss + self.recon_weight * recon
            log = {"loss_recon": recon.item()}

            if pgr_model._last_retrieval is not None:
                commit = commitment_loss_from_retrieval(
                    pgr_model._last_retrieval, pgr_model.prototype_bank,
                    device=next(pgr_model.proto_projector.parameters()).device,
                )
                loss = loss + self.commit_weight * commit
                log["loss_commit"] = commit.item()

                queries = torch.stack([r.queries for r in pgr_model._last_retrieval])
                diversity = query_diversity_loss(queries)
                loss = loss + self.diversity_weight * diversity
                log["loss_diversity"] = diversity.item()

                balance = usage_balance_loss(queries, pgr_model.prototype_bank.embeddings)
                loss = loss + self.balance_weight * balance
                log["loss_balance"] = balance.item()

            log["loss_ce"] = outputs.loss.item()
            self.log(log)

        return (loss, outputs) if return_outputs else loss
