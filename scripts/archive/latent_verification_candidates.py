"""Score checkpoint-matched candidate thoughts on fixed validation histories."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "prototype_grounding"))
import experiment as experiment


def candidate_indices(train_meta, valid_meta, count, seed):
    generator = np.random.default_rng(seed)
    user_groups = train_meta.groupby("user_id").indices
    target_groups = train_meta.groupby("target_item").indices
    choices = {}
    for name, groups, key in (
        ("same_user", user_groups, "user_id"),
        ("same_target", target_groups, "target_item"),
    ):
        indices = np.full((len(valid_meta), count), -1, dtype=np.int64)
        for row_index, value in enumerate(valid_meta[key]):
            pool = groups.get(value)
            if pool is not None:
                indices[row_index] = generator.choice(pool, size=count, replace=len(pool) < count)
        choices[name] = indices
    choices["random"] = generator.integers(len(train_meta), size=(len(valid_meta), count))
    return choices


def summarize(losses, candidate_names, groups):
    own = losses[:, 0]
    result = {"raw_mean_nll": float(own.mean()), "groups": {}}
    for name, columns in groups.items():
        options = losses[:, columns]
        best = np.min(options, axis=1)
        available = np.isfinite(best)
        gain = own[available] - best[available]
        result["groups"][name] = {
            "available": int(available.sum()),
            "oracle_mean_nll": float(best[available].mean()) if available.any() else None,
            "oracle_gain_vs_raw": float(gain.mean()) if available.any() else None,
            "gain_gt_0.01": int((gain > 0.01).sum()),
            "gain_gt_0.05": int((gain > 0.05).sum()),
            "gain_lt_minus_0.01": int((gain < -0.01).sum()),
        }
    best_all = np.min(losses, axis=1)
    result["oracle_all"] = {
        "mean_nll": float(best_all.mean()),
        "gain_vs_raw": float((own - best_all).mean()),
        "best_candidate_counts": {
            name: int((np.argmin(losses, axis=1) == column).sum())
            for column, name in enumerate(candidate_names)
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", default="latent_v4_checkpoint_1152")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = ROOT / "analysis/evidence/CDs_and_Vinyl" / args.tag
    train_summary = json.loads((base / "train/summary.json").read_text())
    valid_summary = json.loads((base / "valid/summary.json").read_text())
    expected_hash = experiment.sha256(args.checkpoint / "model.safetensors")
    if any(summary["checkpoint_sha256"] != expected_hash for summary in (train_summary, valid_summary)):
        raise ValueError("Checkpoint and cached thought hashes do not match")
    if valid_summary["n"] < args.limit:
        raise ValueError("Validation thought cache is shorter than --limit")

    train_vectors = torch.load(base / "train/thoughts.pt", map_location="cpu", weights_only=True).float()
    valid_vectors = torch.load(base / "valid/thoughts.pt", map_location="cpu", weights_only=True).float()[:args.limit]
    train_meta = pd.read_csv(base / "train/metadata.csv")
    valid_meta = pd.read_csv(base / "valid/metadata.csv").iloc[:args.limit]
    indices = candidate_indices(train_meta, valid_meta, args.count, args.seed)
    mean_vector = train_vectors.mean(0)

    candidate_names = ["raw", "mean"]
    groups = {"mean": [1]}
    for group_name in ("same_user", "same_target", "random"):
        groups[group_name] = []
        for position in range(args.count):
            groups[group_name].append(len(candidate_names))
            candidate_names.append(f"{group_name}_{position}")
    groups["own_or_mean"] = [0, 1]
    groups["own_or_user"] = [0, *groups["same_user"]]
    groups["own_or_target"] = [0, *groups["same_target"]]
    groups["own_or_random"] = [0, *groups["random"]]

    model, tokenizer, device = experiment.load_model(args.checkpoint)
    dataset = experiment.PromptDataset("valid", tokenizer, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda rows: experiment.collate(rows, tokenizer.pad_token_id))
    all_losses = []
    offset = 0
    for batch in loader:
        ids, mask, labels, positions = [batch[key].to(device) for key in
                                       ("input_ids", "attention_mask", "labels", "thought_pos")]
        count = len(ids)
        with torch.no_grad():
            embeddings = model.model.embed_tokens(ids)
        batch_losses = []
        for candidate_name in candidate_names:
            if candidate_name == "raw":
                vectors = valid_vectors[offset:offset + count]
                available = np.ones(count, dtype=bool)
            elif candidate_name == "mean":
                vectors = mean_vector.expand(count, -1)
                available = np.ones(count, dtype=bool)
            else:
                group_name, position = candidate_name.rsplit("_", 1)
                selected = indices[group_name][offset:offset + count, int(position)]
                available = selected >= 0
                vectors = train_vectors[np.maximum(selected, 0)]
            with torch.no_grad():
                loss = experiment.per_example_nll(
                    model, experiment.substitute(embeddings, positions, vectors.to(device)), mask, labels
                ).cpu().numpy()
            loss[~available] = np.inf
            batch_losses.append(loss)
        all_losses.append(np.stack(batch_losses, axis=1))
        offset += count
        if offset % 64 == 0 or offset == len(dataset):
            print(f"scored {offset}/{len(dataset)}", flush=True)

    losses = np.concatenate(all_losses)
    output = ROOT / "analysis/results/latent_verification" / f"{args.tag}_candidates_n{args.limit}_s{args.seed}"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), losses=losses, candidate_names=candidate_names,
                        same_user_indices=indices["same_user"], same_target_indices=indices["same_target"],
                        random_indices=indices["random"])
    summary = summarize(losses, candidate_names, groups)
    summary.update({"checkpoint": str(args.checkpoint), "checkpoint_sha256": expected_hash,
                    "split": "valid", "n": args.limit, "count_per_group": args.count,
                    "seed": args.seed, "target_aware_group": "same_target",
                    "note": "All oracle choices use the known target and are diagnostic, not inference-time policies."})
    experiment.atomic_json(output.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
