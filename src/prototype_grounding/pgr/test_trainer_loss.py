"""Checks PrototypeGuidedTrainer.compute_loss's actual logic (loss combination,
gradient reaching all three loss-relevant submodules) against the tiny synthetic
model from test_model_integration.py, without constructing a full Trainer (which
needs a TrainingArguments + dataset just to instantiate) -- a minimal duck-typed
stub standing in for `self` is enough since compute_loss only touches
self.recon_weight / self.commit_weight / self.log.

Run: python3 test_trainer_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from test_model_integration import build_tiny_bank, build_tiny_config, build_batch
from model import PrototypeGuidedLatentModel
from trainer import PrototypeGuidedTrainer


class _StubTrainer:
    """Duck-typed stand-in for transformers.Trainer -- only what compute_loss touches."""
    def __init__(self, recon_weight, commit_weight, diversity_weight, balance_weight):
        self.recon_weight = recon_weight
        self.commit_weight = commit_weight
        self.diversity_weight = diversity_weight
        self.balance_weight = balance_weight
        self.logged = []

    def log(self, d):
        self.logged.append(dict(d))


def main() -> None:
    model = PrototypeGuidedLatentModel(build_tiny_config())
    model.set_prototype_bank(build_tiny_bank())
    model.train()

    input_ids, attention_mask, labels = build_batch()  # B=2, so query_diversity_loss is non-trivial
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    stub = _StubTrainer(recon_weight=2.0, commit_weight=0.5, diversity_weight=1.5, balance_weight=0.8)
    loss = PrototypeGuidedTrainer.compute_loss(stub, model, inputs)

    assert torch.isfinite(loss), f"combined loss not finite: {loss}"
    assert len(stub.logged) == 1
    log = stub.logged[0]
    assert set(log.keys()) == {"loss_recon", "loss_commit", "loss_diversity", "loss_balance", "loss_ce"}
    print(f"combined loss = {loss.item():.4f}")
    print(f"logged: {log}")

    # sanity: combined loss should equal ce + weighted sum of the auxiliary terms,
    # reconstructed from the logged components
    expected = (
        log["loss_ce"]
        + stub.recon_weight * log["loss_recon"]
        + stub.commit_weight * log["loss_commit"]
        + stub.diversity_weight * log["loss_diversity"]
        + stub.balance_weight * log["loss_balance"]
    )
    assert abs(loss.item() - expected) < 1e-4, f"combined loss {loss.item()} != expected {expected}"
    print("loss = ce + recon_weight*recon + commit_weight*commit + diversity_weight*diversity "
          "+ balance_weight*balance: OK")

    loss.backward()
    for name, param in [
        ("proto_projector.proj_up", next(model.proto_projector.proj_up.parameters())),
        ("proto_projector.proj_down", next(model.proto_projector.proj_down.parameters())),
        ("retriever.query_proj", next(model.retriever.query_proj.parameters())),
    ]:
        grad = param.grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} got no gradient from the combined loss"
        print(f"{name}.grad nonzero: OK")

    print("\nall trainer-loss checks passed")


if __name__ == "__main__":
    main()
