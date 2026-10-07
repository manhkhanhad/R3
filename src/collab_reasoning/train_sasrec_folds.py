#!/usr/bin/env python3
"""Train user-fold SASRec models for leakage-free training-row candidates.

The baseline SASRec (``data.SASREC_FULL``) was trained on every training user,
so on a *training* row it has already fit that row's own target.  Candidates or
side vectors built from it are far more accurate on training rows than on test
rows, and the LLM learns to trust them more than it should.  Here each fold
model is trained on the training users outside one fold (same Cornac config as
the baseline) and is used only for the training rows of the held-out users.
Validation and test rows keep using the baseline model, which never saw them.

Usage: python train_sasrec_folds.py --folds 4 --out_dir <dir> [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

import data as D

BASELINE = Path("/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/sasrec.py")


def user_fold(user: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(user.encode()).digest()[:4], "big") % folds


def load_baseline():
    spec = importlib.util.spec_from_file_location("sasrec_baseline", BASELINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look the module up while being defined
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cornac-matplotlib")
    import torch
    from cornac.eval_methods import NextItemEvaluation
    from cornac.metrics import NDCG, Recall
    from cornac.models import SASRec

    base = load_baseline()
    examples = {s: base.load_examples(D.split_file(s)) for s in D.SPLITS}
    val_data = base.evaluation_sessions(examples["valid"], "valid")
    test_data = base.evaluation_sessions(examples["test"], "test")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"folds": a.folds, "fold_rule": "sha256(user_id)[:4] % folds", "models": {}}
    for fold in range(a.folds):
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        kept = [e for e in examples["train"] if user_fold(e.user_id, a.folds) != fold]
        train_data = base.training_sessions(kept)
        print(f"fold {fold}: {len(kept)} training rows, {len({r[0] for r in train_data})} users", flush=True)
        method = NextItemEvaluation.from_splits(train_data=train_data, val_data=val_data, test_data=test_data,
                                                fmt="USIT", exclude_unknowns=True, mode="last", seed=a.seed, verbose=False)
        # same configuration as the baseline (its .pkl.meta)
        model = SASRec(embedding_dim=128, batch_size=512, learning_rate=1e-3, n_sample=2048, n_epochs=a.epochs,
                       max_len=10, num_blocks=2, num_heads=1, dropout=0.2, device=a.device, seed=a.seed,
                       model_selection="best", val_eval_every=20, val_k=20, val_metric="ndcg", verbose=False)
        test_result, val_result = method.evaluate(model=model, metrics=[Recall(10), NDCG(10)], user_based=False,
                                                  show_validation=True)
        path = model.save(str(a.out_dir / f"fold{fold}"), metadata={"fold": fold, "folds": a.folds})
        manifest["models"][str(fold)] = {
            "path": str(Path(path).resolve()), "train_rows": len(kept),
            "validation": {k: float(v) for k, v in val_result.metric_avg_results.items()},
            "test": {k: float(v) for k, v in test_result.metric_avg_results.items()},
        }
        print(fold, manifest["models"][str(fold)], flush=True)
        with (a.out_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
