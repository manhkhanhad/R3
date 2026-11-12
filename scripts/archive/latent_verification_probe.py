"""Cross-fit a target-free selector for the v4 candidate-thought experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def features_for_candidate(own, candidate, mean, history_length, source):
    own_norm = np.linalg.norm(own, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    mean_norm = np.linalg.norm(mean)
    own_mean = np.sum(own * mean, axis=1) / (own_norm * mean_norm).clip(1e-8)
    candidate_mean = np.sum(candidate * mean, axis=1) / (candidate_norm * mean_norm).clip(1e-8)
    own_candidate = np.sum(own * candidate, axis=1) / (own_norm * candidate_norm).clip(1e-8)
    source_flags = np.zeros((len(own), 3), dtype=np.float32)
    source_flags[:, {"mean": 0, "same_user": 1, "random": 2}[source]] = 1
    return np.column_stack((source_flags, own_norm, candidate_norm,
                            np.linalg.norm(own - candidate, axis=1), own_mean,
                            candidate_mean, own_candidate, np.log1p(history_length)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="latent_v4_checkpoint_1152")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evidence = ROOT / "analysis/evidence/CDs_and_Vinyl" / args.tag
    result = ROOT / "analysis/results/latent_verification" / f"{args.tag}_candidates_n{args.limit}_s{args.seed}"
    stored = np.load(result.with_suffix(".npz"))
    losses = stored["losses"]
    names = stored["candidate_names"].tolist()
    own = torch.load(evidence / "valid/thoughts.pt", map_location="cpu", weights_only=True).float()[:args.limit].numpy()
    train = torch.load(evidence / "train/thoughts.pt", map_location="cpu", weights_only=True).float().numpy()
    mean = train.mean(axis=0)
    meta = pd.read_csv(evidence / "valid/metadata.csv").iloc[:args.limit]
    history_length = meta.history_len.to_numpy()
    groups = meta.user_id.to_numpy()

    candidate_columns = [index for index, name in enumerate(names)
                         if name == "mean" or name.startswith("same_user_") or name.startswith("random_")]
    features = []
    for column in candidate_columns:
        name = names[column]
        source = "mean" if name == "mean" else name.rsplit("_", 1)[0]
        if source == "mean":
            vectors = np.broadcast_to(mean, own.shape)
        else:
            selected = stored[f"{source}_indices"][:, int(name.rsplit("_", 1)[1])]
            vectors = train[np.maximum(selected, 0)]
        features.append(features_for_candidate(own, vectors, mean, history_length, source))
    features = np.stack(features, axis=1)
    candidate_losses = losses[:, candidate_columns]
    gain = losses[:, [0]] - candidate_losses
    available = np.isfinite(candidate_losses)
    predictions = np.full_like(gain, -np.inf)
    fold_ids = np.full(args.limit, -1)

    cross_validation = GroupKFold(n_splits=5)
    for fold, (train_rows, test_rows) in enumerate(cross_validation.split(own, groups=groups)):
        training_features = features[train_rows].reshape(-1, features.shape[-1])
        training_gain = gain[train_rows].reshape(-1)
        valid_training = available[train_rows].reshape(-1)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(training_features[valid_training], training_gain[valid_training])
        test_predictions = model.predict(features[test_rows].reshape(-1, features.shape[-1])).reshape(
            len(test_rows), -1)
        test_predictions[~available[test_rows]] = -np.inf
        predictions[test_rows] = test_predictions
        fold_ids[test_rows] = fold

    report = {"n": args.limit, "folds": 5, "grouped_by": "user_id", "model": "standardized Ridge(alpha=10)",
              "candidate_names": [names[column] for column in candidate_columns],
              "features": ["source", "own_norm", "candidate_norm", "distance", "own_mean_cosine",
                           "candidate_mean_cosine", "own_candidate_cosine", "log_history_length"],
              "selection": {}}
    rng = np.random.default_rng(args.seed)
    best_column = np.argmax(predictions, axis=1)
    best_prediction = np.max(predictions, axis=1)
    for threshold in (0.0, 0.01):
        use = best_prediction > threshold
        chosen_gain = np.where(use, gain[np.arange(args.limit), best_column], 0.0)
        replicates = np.array([chosen_gain[rng.integers(args.limit, size=args.limit)].mean()
                               for _ in range(2000)])
        report["selection"][str(threshold)] = {
            "selected": int(use.sum()), "mean_gain_vs_raw": float(chosen_gain.mean()),
            "ci95": np.quantile(replicates, [.025, .975]).tolist(),
            "helped_gt_0.01": int((chosen_gain > .01).sum()),
            "hurt_lt_minus_0.01": int((chosen_gain < -.01).sum()),
        }
    oracle = np.maximum(0, np.max(np.where(available, gain, -np.inf), axis=1))
    report["target_aware_oracle_deployable_pool_gain"] = float(oracle.mean())
    report["raw_mean_nll"] = float(losses[:, 0].mean())
    report["note"] = "Only mean, same-user, and random candidates are selectable; same-target candidates are excluded. The oracle uses target NLL and is diagnostic."
    np.savez_compressed(result.with_name(result.name + "_probe.npz"), predictions=predictions,
                        gain=gain, available=available, fold_ids=fold_ids, candidate_columns=candidate_columns)
    result.with_name(result.name + "_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
