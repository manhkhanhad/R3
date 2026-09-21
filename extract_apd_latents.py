#!/usr/bin/env python3
"""Resumably extract all train/validation/test reasoning latents for APD.

This is the user-provided hook-based extractor, adapted only so its two APD
helper imports are self-contained in the R3 repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import fire
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from latent.layers import LatentModel
from latent.reasoning_dataset import LatentRDataset


CATEGORY_NAMES = {"CDs_and_Vinyl_full": "musics", "CDs_and_Vinyl": "musics"}


def _left_pad(features: list[dict], pad_token_id: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(feature["input_ids"]) for feature in features)
    input_ids, attention_mask = [], []
    for feature in features:
        padding = width - len(feature["input_ids"])
        input_ids.append([pad_token_id] * padding + list(feature["input_ids"]))
        attention_mask.append([0] * padding + list(feature["attention_mask"]))
    return (torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attention_mask, dtype=torch.long, device=device))


class ItemCatalog:
    """Token trie and reverse lookup for constrained catalog generation."""

    def __init__(self, sequences_to_ids: dict[tuple[int, ...], set[int]], eos_token_id: int):
        self.sequences_to_ids = sequences_to_ids
        self.eos_token_id = eos_token_id
        self.prefix_to_next_tokens: dict[tuple[int, ...], list[int]] = {}
        for sequence in sequences_to_ids:
            for stop, token in enumerate(sequence):
                self.prefix_to_next_tokens.setdefault(sequence[:stop], []).append(token)
        self.prefix_to_next_tokens = {
            prefix: sorted(set(tokens)) for prefix, tokens in self.prefix_to_next_tokens.items()
        }
        self.max_new_tokens = max(map(len, sequences_to_ids))

    @classmethod
    def from_info_file(cls, info_file: str, tokenizer: AutoTokenizer) -> "ItemCatalog":
        sequences_to_ids: dict[tuple[int, ...], set[int]] = {}
        with Path(info_file).open(encoding="utf-8") as handle:
            for line in handle:
                title, item_id = line.rstrip("\n").rsplit("\t", 1)
                tokens = tokenizer.encode(f'"{title.strip()}"\n', add_special_tokens=False)
                while tokens and tokens[0] == tokenizer.bos_token_id:
                    tokens.pop(0)
                while tokens and tokens[-1] == tokenizer.eos_token_id:
                    tokens.pop()
                sequence = tuple([*tokens, int(tokenizer.eos_token_id)])
                sequences_to_ids.setdefault(sequence, set()).add(int(item_id))
        return cls(sequences_to_ids, int(tokenizer.eos_token_id))

    def item_ids_for_generated_tokens(self, tokens: list[int]) -> set[int]:
        try:
            end = tokens.index(self.eos_token_id)
            sequence = tuple(tokens[:end + 1])
        except ValueError:
            sequence = tuple(tokens)
        return self.sequences_to_ids.get(sequence, set())


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_text_write(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _target_tokens(tokenizer, title: str) -> list[int]:
    tokens = tokenizer.encode(f'"{title}"\n')
    while tokens and tokens[0] == tokenizer.bos_token_id:
        tokens = tokens[1:]
    while tokens and tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]
    return [*tokens, int(tokenizer.eos_token_id)]


def _per_sample_loss(model: LatentModel, context_features: list[dict],
                     target_rows: list[list[int]], pad_token_id: int,
                     device: torch.device) -> list[float]:
    full_features, context_lengths = [], []
    for feature, target in zip(context_features, target_rows):
        context = list(feature["input_ids"])
        full_features.append({"input_ids": context + target,
                              "attention_mask": [1] * (len(context) + len(target))})
        context_lengths.append(len(context))
    input_ids, attention_mask = _left_pad(full_features, pad_token_id, device)
    labels = torch.full_like(input_ids, -100)
    for row, (feature, target) in enumerate(zip(full_features, target_rows)):
        left_padding = input_ids.shape[1] - len(feature["input_ids"])
        start = left_padding + context_lengths[row]
        labels[row, start:start + len(target)] = torch.tensor(target, device=device)
    output = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False, return_dict=True)
    shifted_logits = output.logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1),
        ignore_index=-100, reduction="none").view_as(shifted_labels)
    valid = shifted_labels.ne(-100)
    return ((token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)).cpu().tolist()


def _completed_shards(shard_dir: Path) -> tuple[list[Path], int]:
    paths = sorted(shard_dir.glob("latents-*.pt"))
    completed = 0
    for expected, path in enumerate(paths):
        if path.name != f"latents-{expected:06d}.pt":
            raise ValueError(f"non-contiguous extraction shard sequence at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.ndim != 2:
            raise ValueError(f"invalid latent shard shape in {path}: {tuple(payload.shape)}")
        metadata_path = shard_dir / f"metadata-{expected:06d}.jsonl"
        metadata_count = sum(1 for line in metadata_path.open() if line.strip())
        if metadata_count != len(payload):
            raise ValueError(f"latent/metadata count mismatch in shard {expected}")
        completed += len(payload)
    return paths, completed


def _finalize(shard_dir: Path, output_dir: Path) -> tuple[torch.Tensor, list[dict]]:
    shard_paths, _ = _completed_shards(shard_dir)
    latent_parts = [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]
    latents = torch.cat(latent_parts, dim=0)
    records = []
    for index in range(len(shard_paths)):
        metadata_path = shard_dir / f"metadata-{index:06d}.jsonl"
        records.extend(json.loads(line) for line in metadata_path.open() if line.strip())
    _atomic_torch_save(latents, output_dir / "latents.pt")
    _atomic_text_write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                       output_dir / "metadata.jsonl")
    return latents, records


@torch.inference_mode()
def main(base_model: str, data_file: str, info_file: str, output_dir: str,
         category: str = "CDs_and_Vinyl_full", split: str = "train",
         batch_size: int = 8, cutoff_len: int = 512, top_n: int = 5,
         sample: int = -1, seed: int = 0, shard_size: int = 256,
         include_recommendation_loss: bool = True, start_index: int = 0,
         stop_index: int = -1) -> None:
    if category not in CATEGORY_NAMES:
        raise ValueError(f"unknown category {category!r}")
    if batch_size < 1 or top_n < 1 or shard_size < batch_size:
        raise ValueError("batch_size/top_n must be positive and shard_size >= batch_size")
    output = Path(output_dir)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    thought_ids = tokenizer("<|Thought|>", add_special_tokens=False)["input_ids"]
    if len(thought_ids) != 1 or int(thought_ids[0]) != len(tokenizer) - 1:
        raise ValueError("base_model must be a pretrained R3 latent checkpoint")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not torch.cuda.is_available():
        raise RuntimeError("full APD latent extraction requires a CUDA GPU")
    device = torch.device("cuda")
    model = LatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    dataset = LatentRDataset(train_file=data_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample, test=True, seed=seed,
                             category=CATEGORY_NAMES[category])
    effective_stop = len(dataset) if stop_index < 0 else min(stop_index, len(dataset))
    if start_index < 0 or start_index >= effective_stop:
        raise ValueError("start_index must be non-negative and smaller than stop_index")
    partition_size = effective_stop - start_index
    catalog = ItemCatalog.from_info_file(info_file, tokenizer)
    generation_config = GenerationConfig(
        do_sample=False, num_beams=top_n, num_return_sequences=top_n,
        max_new_tokens=catalog.max_new_tokens, early_stopping=top_n > 1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True)

    existing_shards, completed = _completed_shards(shard_dir)
    if completed > partition_size:
        raise ValueError("extraction shards contain more samples than the selected partition")
    if completed:
        print(f"[APD extraction] resuming after {completed}/{partition_size} partition samples")
    dataset_tag = Path(data_file).stem
    shard_index = len(existing_shards)
    pending_latents: list[torch.Tensor] = []
    pending_metadata: list[dict] = []

    def flush() -> None:
        nonlocal shard_index, pending_latents, pending_metadata
        if not pending_latents:
            return
        _atomic_torch_save(torch.cat(pending_latents), shard_dir / f"latents-{shard_index:06d}.pt")
        _atomic_text_write(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pending_metadata),
            shard_dir / f"metadata-{shard_index:06d}.jsonl")
        shard_index += 1
        pending_latents, pending_metadata = [], []

    starts = range(start_index + completed, effective_stop, batch_size)
    for start in tqdm(starts, desc=f"extract {split} APD latents"):
        stop = min(start + batch_size, len(dataset))
        features = [dataset[index] for index in range(start, stop)]
        input_ids, attention_mask = _left_pad(features, tokenizer.pad_token_id, device)
        prompt_width = input_ids.shape[1]
        captured: list[torch.Tensor] = []

        def capture_latent(_module, _inputs, result):
            if result.ndim == 2 and result.shape[1] == model.config.hidden_size:
                captured.append(result.detach())

        hook = model.attention.register_forward_hook(capture_latent)
        try:
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=lambda _batch_id, sequence:
                    catalog.prefix_to_next_tokens.get(
                        tuple(int(token) for token in sequence[prompt_width:].tolist()),
                        [catalog.eos_token_id]))
        finally:
            hook.remove()
        if not captured:
            raise RuntimeError("failed to capture latent reasoning during generation")
        batch_latent = captured[0]
        if batch_latent.shape[0] == len(features) * top_n:
            batch_latent = batch_latent.reshape(len(features), top_n, -1)[:, 0]
        elif batch_latent.shape[0] != len(features):
            raise RuntimeError(f"unexpected captured latent shape: {tuple(batch_latent.shape)}")
        batch_latent = batch_latent.float().cpu().to(torch.float16)

        target_rows = [_target_tokens(tokenizer, str(dataset.data.iloc[p]["item_title"]))
                       for p in range(start, stop)]
        losses = (_per_sample_loss(model, features, target_rows,
                                   tokenizer.pad_token_id, device)
                  if include_recommendation_loss else [None] * len(features))
        suffixes = generated[:, prompt_width:].reshape(len(features), top_n, -1)
        for offset, position in enumerate(range(start, stop)):
            row = dataset.data.iloc[position]
            target_item_id = int(row["item_id"])
            predicted_ids = [sorted(catalog.item_ids_for_generated_tokens(seq.tolist()))
                             for seq in suffixes[offset]]
            rank = next((rank for rank, ids in enumerate(predicted_ids, 1)
                         if target_item_id in ids), None)
            original_index = int(dataset.data.index[position])
            pending_metadata.append({
                "sample_id": f"{dataset_tag}-{split}-{original_index:06d}",
                "row_index": original_index, "target_item_id": target_item_id,
                "prediction_item_ids": predicted_ids, "top5_rank": rank,
                "is_success": rank is not None, "recommendation_loss": losses[offset]})
        pending_latents.append(batch_latent)
        if sum(len(part) for part in pending_latents) >= shard_size:
            flush()
    flush()

    latents, records = _finalize(shard_dir, output)
    if len(latents) != partition_size or len(records) != partition_size:
        raise RuntimeError("final extraction count does not match selected partition")
    norms = latents.float().norm(dim=-1).numpy()
    hashes = [hashlib.sha1(row.numpy().tobytes()).digest() for row in latents]
    _, counts = np.unique(np.asarray(hashes), return_counts=True, axis=0)
    success_count = sum(bool(row["is_success"]) for row in records)
    summary = {
        "base_model": os.path.abspath(base_model), "data_file": os.path.abspath(data_file),
        "info_file": os.path.abspath(info_file), "split": split,
        "sample_count": len(records), "successful_count": success_count,
        "failed_count": len(records) - success_count,
        "success_rate_at_5": success_count / max(1, len(records)),
        "latent_dim": int(latents.shape[1]), "latent_dtype": str(latents.dtype),
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_std": float(norms.std(ddof=1)),
        "exact_duplicate_rate": float(1.0 - len(counts) / len(records)),
        "maximum_exact_duplicate_frequency": int(counts.max()),
        "top_n": top_n, "seed": seed,
        "start_index": start_index, "stop_index": effective_stop,
        "includes_recommendation_loss": include_recommendation_loss,
        "latents_path": str((output / "latents.pt").resolve()),
        "metadata_path": str((output / "metadata.jsonl").resolve())}
    _atomic_text_write(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       output / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    fire.Fire(main)
