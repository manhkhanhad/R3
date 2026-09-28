"""Test whether a token-accurate same-user thought improves frozen R3 ranking."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "prototype_grounding"))
import experiment


def token_accuracies(model, dataset, indices, device, pad_id):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        collate_fn=lambda rows: experiment.collate(rows, pad_id),
    )
    accuracies = {}
    for position, batch in enumerate(loader):
        ids, mask, labels, thought_pos = [batch[key].to(device) for key in
                                          ("input_ids", "attention_mask", "labels", "thought_pos")]
        with torch.inference_mode():
            _, embeddings = experiment.thought_embeds(model, ids, mask, thought_pos)
            logits = model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
        targets = labels[:, 1:]
        predicted = logits[:, :-1].argmax(-1)
        included = targets.ne(-100)
        accuracies[indices[position]] = float(
            ((predicted == targets) & included).sum() / included.sum().clamp_min(1)
        )
        if (position + 1) % 100 == 0:
            print(f"checked {position + 1}/{len(indices)} train candidates", flush=True)
    return accuracies


def choose_pairs(indices, train_meta, valid_meta, train_vectors, valid_vectors,
                 accuracies, threshold, limit, seed, selection):
    eligible = np.array(sorted(index for index, value in accuracies.items() if value >= threshold))
    rng = np.random.default_rng(seed)
    pairs = []
    for row_index, candidates in enumerate(indices):
        matched = [int(index) for index in candidates if index in accuracies
                   and accuracies[int(index)] >= threshold]
        if not matched:
            continue
        if selection == "similarity":
            similarities = torch.nn.functional.cosine_similarity(
                train_vectors[matched], valid_vectors[row_index].unsqueeze(0), dim=1)
            positive = matched[int(similarities.argmax())]
        else:
            positive = max(matched, key=lambda index: (
                accuracies[index], -float(train_meta.iloc[index].raw_nll)))
        other = eligible[train_meta.iloc[eligible].user_id.to_numpy() != valid_meta.iloc[row_index].user_id]
        if not len(other):
            continue
        random_positive = int(rng.choice(other))
        pairs.append((row_index, positive, random_positive))
        if len(pairs) == limit:
            break
    return pairs


def generate_rank(model, tokenizer, row, vector, prefix_data, generation_config):
    prompt = row["input_ids"][:row["prompt_len"]]
    ids = torch.tensor([prompt], device=model.device)
    mask = torch.ones_like(ids)

    def generate_embs(input_ids, attention_mask):
        thought_id = model.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_id)
        embeddings = model.model.embed_tokens(input_ids).clone()
        embeddings[where[:, 0], where[:, 1]] = vector.to(embeddings.dtype)
        return embeddings

    def allowed(batch_id, input_ids):
        return prefix_data.get(experiment.get_hash(input_ids), [])

    model.generate_embs = generate_embs
    processor = experiment.CFEnhancedLogitsProcessor(
        guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=allowed,
        cf_dict=None, unconditional_ids=None, model=model, tokenizer=tokenizer, num_beams=10,
    )
    with torch.inference_mode():
        generated = model.generate(
            ids, attention_mask=mask, generation_config=generation_config,
            return_dict_in_generate=True, output_scores=True,
            logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), processor]),
        )
    decoded = tokenizer.batch_decode(generated.sequences[:, len(prompt):], skip_special_tokens=True)
    predictions = [value.split("Response:")[-1].strip(' \n"') for value in decoded]
    target = row["target_title"].strip(' \n"')
    return predictions.index(target) + 1 if target in predictions else 10**9


def summarize(records):
    modes = records[0]["ranks"]
    summary = {mode: experiment.recommendation_metrics(
        np.array([record["ranks"][mode] for record in records])) for mode in modes}
    raw = np.array([record["ranks"]["raw"] for record in records])
    for mode in modes:
        if mode == "raw":
            continue
        ranks = np.array([record["ranks"][mode] for record in records])
        summary[mode]["raw_only_hits_at_10"] = int(((raw <= 10) & (ranks > 10)).sum())
        summary[mode]["mode_only_hits_at_10"] = int(((raw > 10) & (ranks <= 10)).sum())
    best = np.minimum.reduce([np.array([record["ranks"][mode] for record in records])
                              for mode in modes if mode != "random_positive"])
    return summary, experiment.recommendation_metrics(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", default="latent_v4_checkpoint_1152")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--selection", choices=("accuracy", "similarity"), default="similarity")
    args = parser.parse_args()

    base = ROOT / "analysis/evidence/CDs_and_Vinyl" / args.tag
    candidate_path = ROOT / "analysis/results/latent_verification" / f"{args.tag}_candidates_n256_s{args.seed}.npz"
    candidate_data = np.load(candidate_path)
    indices = candidate_data["same_user_indices"]
    expected_hash = experiment.sha256(args.checkpoint / "model.safetensors")
    for split in ("train", "valid"):
        summary = json.loads((base / split / "summary.json").read_text())
        if summary["checkpoint_sha256"] != expected_hash:
            raise ValueError(f"{split} thought cache does not match checkpoint")

    train_meta = pd.read_csv(base / "train/metadata.csv")
    valid_meta = pd.read_csv(base / "valid/metadata.csv").iloc[:len(indices)]
    train_vectors = torch.load(base / "train/thoughts.pt", map_location="cpu", weights_only=True).float()
    valid_vectors = torch.load(base / "valid/thoughts.pt", map_location="cpu", weights_only=True).float()
    candidate_ids = sorted(set(int(index) for index in indices.flat if index >= 0))
    model, tokenizer, device = experiment.load_model(args.checkpoint)
    train_dataset = experiment.PromptDataset("train", tokenizer)
    accuracies = token_accuracies(model, train_dataset, candidate_ids, device, tokenizer.pad_token_id)
    pairs = choose_pairs(indices, train_meta, valid_meta, train_vectors, valid_vectors,
                         accuracies, args.min_accuracy, args.limit, args.seed, args.selection)
    if not pairs:
        raise ValueError("No same-user candidate passed the correctness gate")
    print(f"selected {len(pairs)} histories from {len(indices)} after gate", flush=True)

    tokenizer.padding_side = "left"
    valid_dataset = experiment.PromptDataset("valid", tokenizer, len(indices))
    prefix_data = experiment.get_prefix_data(str(experiment.INFO_FILE), tokenizer)
    generation_config = GenerationConfig(
        num_beams=10, num_return_sequences=10, length_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
    )
    mean_vector = train_vectors.mean(0).to(device)
    records = []
    selection_tag = "" if args.selection == "accuracy" else "_similar"
    output = ROOT / "analysis/results/latent_verification" / (
        f"{args.tag}_refinement_oracle{selection_tag}_n{args.limit}_s{args.seed}.json")
    for row_index, positive_index, random_index in pairs:
        row = valid_dataset.rows[row_index]
        own = valid_vectors[row_index].to(device)
        positive = train_vectors[positive_index].to(device)
        blend_quarter = own + 0.25 * (positive - own)
        blend_half = own + 0.50 * (positive - own)
        blend_quarter = blend_quarter * (own.norm() / blend_quarter.norm().clamp_min(1e-8))
        blend_half = blend_half * (own.norm() / blend_half.norm().clamp_min(1e-8))
        vectors = {
            "raw": own,
            "same_user_positive": positive,
            "blend_0.25": blend_quarter,
            "blend_0.50": blend_half,
            "random_positive": train_vectors[random_index].to(device),
            "mean": mean_vector,
        }
        ranks = {mode: generate_rank(model, tokenizer, row, vector, prefix_data, generation_config)
                 for mode, vector in vectors.items()}
        records.append({"sample_id": row["sample_id"], "user_id": row["user_id"],
                        "positive_train_id": positive_index, "positive_token_accuracy": accuracies[positive_index],
                        "random_train_id": random_index, "ranks": ranks})
        if len(records) % 8 == 0 or len(records) == len(pairs):
            print(f"ranked {len(records)}/{len(pairs)} histories", flush=True)
            summary, oracle = summarize(records)
            experiment.atomic_json(output, {
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": expected_hash,
                "split": "valid", "candidate_source": "train", "gate": "teacher-forced token accuracy",
                "min_accuracy": args.min_accuracy, "candidate_histories": len(indices),
                "selected_histories": len(pairs), "completed_histories": len(records),
                "selection": f"highest-{args.selection} same-user train latent among four pre-sampled candidates",
                "blend_norm": "matched to own thought norm",
                "oracle_note": "Best-per-history rank uses the validation target and is not an inference policy.",
                "metrics": summary, "answer_aware_oracle": oracle, "records": records,
            })


if __name__ == "__main__":
    main()
