#!/usr/bin/env python3
"""Run the single-thought prototype-grounding experiment pipeline.

The script deliberately keeps all generated state below ``analysis/`` and treats
the supplied Stage-1 checkpoint as immutable.  Expensive model passes are cached
per split; clustering and grounding sweeps can then be repeated cheaply.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

# CODE: where this source tree lives (may be a frozen copy under output_dir/).
# ROOT: repository holding data/, analysis/, output_dir/ (override with R3_ROOT).
CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("R3_ROOT", CODE)).resolve()
sys.path.insert(0, str(CODE / "src" / "grpo_attention_tuning"))
from model import LatentModel  # noqa: E402
from latent_grpo_dataset import get_prefix_data  # noqa: E402
from latent_grpo_processor import CFEnhancedLogitsProcessor, get_hash  # noqa: E402


CHECKPOINT = ROOT / "output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5"
ANALYSIS = ROOT / "analysis"
THOUGHT = "<|Thought|>"

# Human-readable category noun used in the training-time prompt template
# (matches the `category_dict` in src/grpo_attention_tuning/{train_noise_grpo,noise_eval}.py).
CATEGORY_PROMPT_MAP = {
    "Office_Products": "office products", "Books": "books", "steam": "games",
    "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
    "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
    "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
    "yelp": "resturant",
}


def _glob_one(pattern: Path) -> Path:
    matches = sorted(pattern.parent.glob(pattern.name))
    if not matches:
        raise FileNotFoundError(f"no file matches {pattern}")
    return matches[0]


def configure_category(category: str) -> None:
    """Point every dataset-derived global at `category` (default: CDs_and_Vinyl)."""
    global CATEGORY, CATEGORY_PROMPT, DATA_ROOT, SPLIT_FILES, INFO_FILE
    CATEGORY = category
    CATEGORY_PROMPT = CATEGORY_PROMPT_MAP[category]
    DATA_ROOT = ROOT / "data" / category
    SPLIT_FILES = {
        split: _glob_one(DATA_ROOT / split / f"{category}*.csv")
        for split in ("train", "valid", "test")
    }
    INFO_FILE = _glob_one(DATA_ROOT / "info" / f"{category}*.txt")


configure_category("CDs_and_Vinyl")
SEEDS = (13, 42, 73)
KS = (32, 64, 128, 256)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptDataset(Dataset):
    def __init__(self, split: str, tokenizer: AutoTokenizer, limit: int = -1, history_mode: str = "raw"):
        self.split = split
        self.frame = pd.read_csv(SPLIT_FILES[split])
        if limit > 0:
            self.frame = self.frame.iloc[:limit].copy()
        self.tokenizer = tokenizer
        self.history_mode = history_mode
        self.catalog_titles = [line.split("\t")[0].strip(' \n"') for line in INFO_FILE.open()]
        self.rows = [self._encode(i, row) for i, row in self.frame.iterrows()]

    def _strip_edges(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        while ids and ids[0] == self.tokenizer.bos_token_id:
            ids.pop(0)
        while ids and ids[-1] == self.tokenizer.eos_token_id:
            ids.pop()
        return ids

    def _encode(self, sample_id: int, row: pd.Series) -> dict[str, Any]:
        history = ast.literal_eval(row.history_item_title)
        if self.history_mode == "truncate":
            history = history[-max(1, len(history) // 2):]
        elif self.history_mode == "noise":
            candidates = [self.catalog_titles[(sample_id * 104729 + j * 7919) % len(self.catalog_titles)] for j in range(3)]
            history = history + [x for x in candidates if x not in history and x != str(row.item_title)]
        titles = ", ".join(f'"{x}"' for x in history)
        instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request. \n\n### Instruction:\n"
            f"Given a list of {CATEGORY_PROMPT} the user recetenly enjoy, please write a new {CATEGORY_PROMPT} that the user may bought\n"
        )
        prompt = (
            f"### User Input: \nThe user has palyed the following {CATEGORY_PROMPT}s before: {titles}"
            f"\n\n### Response: \n{THOUGHT}"
        )
        target = f'"{str(row.item_title)}"\n'
        ids = self._strip_edges(self.tokenizer.encode(instruction))
        if self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        ids += self._strip_edges(self.tokenizer.encode(prompt))
        prompt_len = len(ids)
        ids += self._strip_edges(self.tokenizer.encode(target)) + [self.tokenizer.eos_token_id]
        if len(ids) > 2560:
            removed = len(ids) - 2560
            ids = ids[-2560:]
            prompt_len = max(1, prompt_len - removed)
        history_ids = ast.literal_eval(row.history_item_id)
        return {
            "sample_id": int(sample_id), "input_ids": ids, "prompt_len": prompt_len,
            "user_id": str(row.user_id), "target_item": int(row.item_id),
            "history_len": len(history), "history_item_ids": history_ids,
            "target_title": str(row.item_title),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, masks, labels, thought_pos = [], [], [], []
    meta = []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        ids.append(x["input_ids"] + [pad_id] * pad)
        masks.append([1] * n + [0] * pad)
        labels.append([-100] * x["prompt_len"] + x["input_ids"][x["prompt_len"]:] + [-100] * pad)
        thought_pos.append(x["prompt_len"] - 1)
        meta.append({k: v for k, v in x.items() if k not in {"input_ids", "prompt_len"}})
    return {
        "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
        "labels": torch.tensor(labels), "thought_pos": torch.tensor(thought_pos), "meta": meta,
    }


def load_model(checkpoint: Path = CHECKPOINT) -> tuple[LatentModel, AutoTokenizer, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model-backed experiments")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(checkpoint)
    if (getattr(config, "latent_neighbor_refinement", None) or getattr(config, "latent_bottleneck", None)
            or getattr(config, "latent_pool_history", None) or getattr(config, "latent_attention_share", None)):
        # These live in the SFT model class.  The original GRPO class silently
        # drops the neighbor wrapper (random-initialising the plain attention
        # layer) and does not apply the L7 latent-only attention mask.
        sys.path.insert(0, str(CODE / "src" / "latent"))
        from layers import LatentModel as CorrectedLatentModel
        model, loading = CorrectedLatentModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, output_loading_info=True,
        )
        if loading["missing_keys"] or loading["unexpected_keys"]:
            raise RuntimeError(f"Corrected checkpoint did not load exactly: {loading}")
    else:
        model = LatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, device


def _share_positions(model, ids: torch.Tensor) -> None:
    """Attention-share checkpoints (attention_share.py): record the thought positions of this batch."""
    if getattr(model, "_latent_share", None) is not None:
        from attention_share import set_share_positions
        set_share_positions(model, ids, model.model.embed_tokens.num_embeddings - 1)


def thought_embeds(model: LatentModel, ids: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor):
    with torch.no_grad():
        _share_positions(model, ids)
        first = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=False, use_cache=False)
        pool_mask = model.pooling_mask(ids, mask) if hasattr(model, "pooling_mask") else mask
        thought = model.attention(first.last_hidden_state, pool_mask, pos)
        embeds = model.model.embed_tokens(ids).clone()
        embeds[torch.arange(ids.size(0), device=ids.device), pos] = thought.to(embeds.dtype)
    return thought.float(), embeds


def per_example_nll(model: LatentModel, embeds: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
                    ids: torch.Tensor | None = None) -> torch.Tensor:
    # A latent-bottleneck (L7) checkpoint needs the prompt ids to build its mask.
    extra = {"latent_bottleneck_ids": ids} if getattr(model.config, "latent_bottleneck", None) else {}
    if extra and ids is None:
        raise ValueError("latent-bottleneck checkpoints need ids for teacher-forced scoring")
    with torch.no_grad():
        logits = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, **extra).logits[:, :-1].float()
        shifted = labels[:, 1:]
        loss = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
        valid = shifted.ne(-100)
        return (loss * valid).sum(1) / valid.sum(1).clamp_min(1)


def substitute(embeds: torch.Tensor, pos: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    out = embeds.clone()
    out[torch.arange(out.size(0), device=out.device), pos] = vectors.to(out.dtype)
    return out


def ranks_from_existing_predictions(checkpoint: Path = CHECKPOINT) -> dict[int, int]:
    path = checkpoint / "final_result.json"
    if not path.exists():
        return {}
    records = json.load(path.open())
    ranks = {}
    for i, x in enumerate(records):
        target = x["output"].strip(' \n"')
        pred = [p.strip(' \n"') for p in x.get("predict", [])]
        ranks[i] = pred.index(target) + 1 if target in pred else 10**9
    return ranks


def extract(split: str, batch_size: int, limit: int = -1,
            checkpoint: Path = CHECKPOINT, tag: str = "") -> None:
    model, tokenizer, device = load_model(checkpoint)
    ds = PromptDataset(split, tokenizer, limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    vectors, z_vectors, raw_nll, mask_nll, metadata = [], [], [], [], []
    start = time.time()
    for step, batch in enumerate(loader):
        ids, mask, labels, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        thought, embeds = thought_embeds(model, ids, mask, pos)
        # For mu/z-decomposed checkpoints (Attempts 7/8+), last_z is set as a
        # side effect of the model.attention(...) call inside thought_embeds --
        # free to capture, no extra forward pass. mu's norm is architecturally
        # fixed (see VAEDecomposedSelfAttentionLayer), so any norm/direction
        # variance in the combined thought vector is a property of z, not mu;
        # geometry run on the combined vector alone can't distinguish "z is
        # genuinely diverse" from "mu's fixed norm is just large relative to a
        # collapsed z". Saving z separately lets latent-distribution measure it
        # directly instead of inferring it from the sum.
        last_z = getattr(model.attention, "last_z", None)
        if last_z is not None:
            z_vectors.append(last_z.detach().cpu().half())
        nll_raw = per_example_nll(model, embeds, mask, labels, ids)
        nll_mask = per_example_nll(model, substitute(embeds, pos, torch.zeros_like(thought)), mask, labels, ids)
        vectors.append(thought.cpu().half())
        raw_nll.append(nll_raw.cpu())
        mask_nll.append(nll_mask.cpu())
        metadata.extend(batch["meta"])
        if step % 100 == 0:
            print(f"extract {split}: {min((step+1)*batch_size,len(ds))}/{len(ds)}", flush=True)
    vec = torch.cat(vectors)
    raw = torch.cat(raw_nll).numpy()
    masked = torch.cat(mask_nll).numpy()
    utility = masked - raw
    ranks = ranks_from_existing_predictions(checkpoint) if split == "test" else {}
    for i, m in enumerate(metadata):
        m.update({"raw_nll": float(raw[i]), "masked_nll": float(masked[i]),
                  "utility": float(utility[i]), "raw_rank": int(ranks.get(i, -1)),
                  "hit_at_1": bool(ranks.get(i, 10**9) <= 1),
                  "hit_at_3": bool(ranks.get(i, 10**9) <= 3),
                  "hit_at_5": bool(ranks.get(i, 10**9) <= 5),
                  "hit_at_10": bool(ranks.get(i, 10**9) <= 10)})
    out = ANALYSIS / f"evidence/{CATEGORY}"
    if tag:
        out = out / tag
    out = out / split
    out.mkdir(parents=True, exist_ok=True)
    torch.save(vec, out / "thoughts.pt")
    if z_vectors:
        torch.save(torch.cat(z_vectors), out / "z_only.pt")
    pd.DataFrame(metadata).drop(columns=["history_item_ids"]).to_csv(out / "metadata.csv", index=False)
    atomic_json(out / "summary.json", {
        "split": split, "n": len(ds), "shape": list(vec.shape), "dtype": str(vec.dtype),
        "raw_nll": float(raw.mean()), "masked_nll": float(masked.mean()),
        "mean_utility": float(utility.mean()), "median_utility": float(np.median(utility)),
        "positive_fraction": float((utility > 0).mean()),
        "elapsed_seconds": time.time() - start, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "split_sha256": sha256(SPLIT_FILES[split]), "limit": limit,
    })
    print(json.dumps(json.load((out / "summary.json").open()), indent=2), flush=True)


def load_evidence(split: str, tag: str = "") -> tuple[torch.Tensor, pd.DataFrame]:
    base = ANALYSIS / f"evidence/{CATEGORY}"
    if tag:
        base = base / tag
    base = base / split
    return torch.load(base / "thoughts.pt", map_location="cpu", weights_only=True).float(), pd.read_csv(base / "metadata.csv")


def bank_mask(meta: pd.DataFrame, bank: str) -> np.ndarray:
    utility = meta.utility.to_numpy()
    if bank == "all": return np.ones(len(meta), dtype=bool)
    if bank == "positive": return utility > 0
    if bank == "top": return utility >= np.quantile(utility, .75)
    if bank == "harmful": return utility < 0
    if bank == "correct": return meta.raw_nll.to_numpy() <= np.quantile(meta.raw_nll, .25)
    raise ValueError(bank)


def spherical_kmeans(x: torch.Tensor, weights: torch.Tensor, k: int, seed: int,
                     iterations: int = 40, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    seed_all(seed)
    work_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = F.normalize(x.float().to(work_device), dim=1)
    weights = weights.to(work_device)
    # weighted k-means++ initialization
    first = torch.multinomial(weights / weights.sum(), 1).item()
    centers = [x[first]]
    closest = 1 - x @ centers[0]
    for _ in range(1, k):
        probs = weights * closest.clamp_min(1e-6).square()
        idx = torch.multinomial(probs / probs.sum(), 1).item()
        centers.append(x[idx])
        closest = torch.minimum(closest, 1 - x @ centers[-1])
    c = torch.stack(centers)
    assign = torch.zeros(len(x), dtype=torch.long, device=work_device)
    for _ in range(iterations):
        for start in range(0, len(x), chunk):
            assign[start:start+chunk] = (x[start:start+chunk] @ c.T).argmax(1)
        sums = torch.zeros(k, x.size(1), device=work_device)
        sums.index_add_(0, assign, x * weights[:, None])
        counts = torch.zeros(k, device=work_device).index_add_(0, assign, weights)
        empty = counts == 0
        if empty.any():
            sums[empty] = x[torch.randperm(len(x))[:int(empty.sum())]]
        new_c = F.normalize(sums, dim=1)
        if (new_c - c).norm(dim=1).max() < 1e-4:
            c = new_c; break
        c = new_c
    return c.half().cpu(), assign.cpu()


def mine(bank: str, k: int, seed: int) -> None:
    vectors, meta = load_evidence("train")
    keep = bank_mask(meta, bank)
    x, sub = vectors[keep], meta.loc[keep].reset_index(drop=True)
    utility = torch.tensor(sub.utility.to_numpy(), dtype=torch.float32)
    if bank in {"positive", "top"}:
        weights = utility.clamp(min=0, max=max(float(utility.quantile(.95)), 1e-6)) + 1e-4
    else:
        weights = torch.ones(len(sub))
    user_counts = sub.user_id.value_counts()
    weights /= torch.tensor(sub.user_id.map(user_counts).to_numpy(), dtype=torch.float32)
    weights *= len(weights) / weights.sum()
    retrieval_centers, assign = spherical_kmeans(x, weights, k, seed)
    raw_sums = torch.zeros(k, x.size(1)).index_add_(0, assign, x.float() * weights[:, None])
    raw_weights = torch.zeros(k).index_add_(0, assign, weights)
    raw_centers = raw_sums / raw_weights[:, None].clamp_min(1e-8)
    empty = raw_weights.eq(0)
    if empty.any():
        raw_centers[empty] = retrieval_centers[empty].float() * x.norm(dim=1).median()
    rows = []
    for j in range(k):
        idx = np.flatnonzero(assign.numpy() == j)
        users = sub.iloc[idx].user_id
        util = sub.iloc[idx].utility
        counts = users.value_counts()
        sims = F.normalize(x[idx].float(), dim=1) @ retrieval_centers[j].float() if len(idx) else torch.tensor([])
        nearest_sims = F.normalize(retrieval_centers.float(), dim=1) @ F.normalize(retrieval_centers[j].float(), dim=0)
        nearest_other = float(torch.cat([nearest_sims[:j], nearest_sims[j+1:]]).max())
        rows.append({"prototype": j, "interactions": len(idx), "distinct_users": users.nunique(),
                     "mean_utility": util.mean(), "utility_std": util.std(),
                     "mean_similarity": float(sims.mean()) if len(sims) else float("nan"),
                     "dominant_user_fraction": float(counts.iloc[0] / len(idx)) if len(idx) else 0,
                     "nearest_prototype_similarity": nearest_other})
    out = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(raw_centers.half(), out / "centroids.pt")
    torch.save(retrieval_centers, out / "retrieval_centroids.pt")
    np.save(out / "assignments.npy", assign.numpy())
    np.save(out / "evidence_indices.npy", np.flatnonzero(keep))
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)
    atomic_json(out / "manifest.json", {"bank": bank, "k": k, "seed": seed, "n_evidence": int(keep.sum()),
                "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"),
                "train_split_sha256": sha256(SPLIT_FILES["train"]), "centroid_sha256": sha256(out / "centroids.pt")})
    print(f"mined bank={bank} k={k} seed={seed} n={keep.sum()}", flush=True)


def retrieve(q: torch.Tensor, centers: torch.Tensor, topk: int, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qn, cn = F.normalize(q.float(), dim=1), F.normalize(centers.float(), dim=1)
    values, indices = (qn @ cn.T).topk(topk, dim=1)
    attn = F.softmax(values / temperature, dim=1)
    p = (attn[..., None] * centers.float()[indices]).sum(1)
    entropy = -(attn * attn.clamp_min(1e-9).log()).sum(1)
    margin = values[:, 0] - (values[:, 1] if topk > 1 else 0)
    return p, {"similarity": values[:, 0], "margin": margin, "entropy": entropy, "indices": indices[:, 0]}


def bank_components(bank: str, k: int, seed: int):
    vectors, meta = load_evidence("train")
    path = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}"
    evidence_idx = np.load(path / "evidence_indices.npy")
    assignments = torch.from_numpy(np.load(path / "assignments.npy")).long()
    sub = meta.iloc[evidence_idx].reset_index(drop=True)
    x = vectors[evidence_idx].float()
    utility = torch.tensor(sub.utility.to_numpy(), dtype=torch.float32)
    if bank in {"positive", "top"}:
        weights = utility.clamp(min=0, max=max(float(utility.quantile(.95)), 1e-6)) + 1e-4
    else:
        weights = torch.ones(len(sub))
    user_counts = sub.user_id.value_counts()
    weights /= torch.tensor(sub.user_id.map(user_counts).to_numpy(), dtype=torch.float32)
    weights *= len(weights) / weights.sum()
    sums = torch.zeros(k, x.size(1)).index_add_(0, assignments, x * weights[:, None])
    counts = torch.zeros(k).index_add_(0, assignments, weights)
    distinct = sub.assign(cluster=assignments.numpy()).groupby("cluster").user_id.nunique().reindex(range(k), fill_value=0).to_numpy()
    return x, sub, assignments, weights, sums, counts, distinct


def exclusion_retrieve(q: torch.Tensor, query_meta: list[dict[str, Any]], components,
                       exclusion: str, topk: int, temperature: float, min_users: int = 1):
    x, evidence_meta, assignments, weights, base_sums, base_counts, distinct = components
    grounded, diagnostics = [], []
    users_np = evidence_meta.user_id.to_numpy()
    targets_np = evidence_meta.target_item.to_numpy()
    for row, query in zip(q, query_meta):
        remove = np.zeros(len(evidence_meta), dtype=bool)
        if exclusion in {"user", "both"}: remove |= users_np == query["user_id"]
        if exclusion in {"target", "both"}: remove |= targets_np == query["target_item"]
        sums, counts = base_sums.clone(), base_counts.clone()
        idx = np.flatnonzero(remove)
        if len(idx):
            ti = torch.from_numpy(idx)
            sums.index_add_(0, assignments[ti], -(x[ti] * weights[ti, None]))
            counts.index_add_(0, assignments[ti], -weights[ti])
        centers = (sums / counts[:, None].clamp_min(1e-8)).to(row.device)
        center_directions = F.normalize(centers, dim=1)
        sim = F.normalize(row.float(), dim=0) @ center_directions.T
        valid = torch.from_numpy(distinct >= min_users).to(row.device) & counts.to(row.device).gt(1e-6)
        sim = sim.masked_fill(~valid, -torch.inf)
        vals, ids = sim.topk(topk)
        attn = F.softmax(vals / temperature, dim=0)
        grounded.append((attn[:, None] * centers[ids]).sum(0))
        diagnostics.append(torch.tensor([vals[0], vals[0] - (vals[1] if topk > 1 else 0),
                                          -(attn * attn.clamp_min(1e-9).log()).sum()], device=row.device))
    return torch.stack(grounded), torch.stack(diagnostics)


def collaborative_score(split: str, bank: str, k: int, seed: int, topk: int,
                        temperature: float, alpha: float, exclusion: str, min_users: int,
                        batch_size: int, limit: int = -1) -> None:
    model, tokenizer, device = load_model()
    ds = PromptDataset(split, tokenizer, limit)
    cached, _ = load_evidence(split)
    if limit > 0: cached = cached[:limit]
    components = bank_components(bank, k, seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    raw_all, grounded_all, diag_all = [], [], []
    offset = 0
    for batch in loader:
        ids, mask, labels, pos = [batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos")]
        _, embeds = thought_embeds(model, ids, mask, pos)
        q = cached[offset:offset+len(ids)].to(device)
        p, diag = exclusion_retrieve(q, batch["meta"], components, exclusion, topk, temperature, min_users)
        grounded = q + alpha * (p - q)
        raw_all.append(per_example_nll(model, embeds, mask, labels, ids).cpu())
        grounded_all.append(per_example_nll(model, substitute(embeds, pos, grounded), mask, labels, ids).cpu())
        diag_all.append(diag.cpu()); offset += len(ids)
    raw, grounded, diag = torch.cat(raw_all).numpy(), torch.cat(grounded_all).numpy(), torch.cat(diag_all).numpy()
    delta = raw - grounded
    result = {"split": split, "bank": bank, "k": k, "seed": seed, "topk": topk,
              "temperature": temperature, "alpha": alpha, "exclusion": exclusion, "min_users": min_users,
              "n": len(raw), "raw_nll": float(raw.mean()), "grounded_nll": float(grounded.mean()),
              "nll_improvement": float(delta.mean()), "helped_fraction": float((delta > 0).mean()),
              "bootstrap": bootstrap_delta(grounded, raw, seed),
              "mean_similarity": float(diag[:, 0].mean()), "mean_margin": float(diag[:, 1].mean()),
              "mean_entropy": float(diag[:, 2].mean()), "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"),
              "split_sha256": sha256(SPLIT_FILES[split])}
    key = f"{split}_{bank}_k{k}_s{seed}_{exclusion}_m{min_users}"
    atomic_json(ANALYSIS / f"results/cross_user/{key}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def oracle_score(split: str, bank: str, k: int, seed: int, candidate_k: int,
                 batch_size: int, limit: int = -1) -> None:
    model, tokenizer, device = load_model()
    ds = PromptDataset(split, tokenizer, limit)
    cached, _ = load_evidence(split)
    if limit > 0: cached = cached[:limit]
    centers_path = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}/centroids.pt"
    centers = torch.load(centers_path, map_location=device, weights_only=True).float()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    raw_out, sim_out, oracle_out = [], [], []
    offset = 0
    for batch in loader:
        ids, mask, labels, pos = [batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos")]
        _, embeds = thought_embeds(model, ids, mask, pos)
        q = cached[offset:offset+len(ids)].to(device)
        _, diag = retrieve(q, centers, candidate_k, .1)
        ids_top = (F.normalize(q, dim=1) @ F.normalize(centers, dim=1).T).topk(candidate_k, dim=1).indices
        raw = per_example_nll(model, embeds, mask, labels, ids)
        candidate_losses = []
        for j in range(candidate_k):
            candidate_losses.append(per_example_nll(model, substitute(embeds, pos, centers[ids_top[:, j]]), mask, labels, ids))
        losses = torch.stack(candidate_losses, 1)
        raw_out.append(raw.cpu()); sim_out.append(losses[:, 0].cpu()); oracle_out.append(losses.min(1).values.cpu())
        offset += len(ids)
    raw, sim, oracle = [torch.cat(x).numpy() for x in (raw_out, sim_out, oracle_out)]
    result = {"split": split, "bank": bank, "k": k, "seed": seed, "candidate_k": candidate_k, "n": len(raw),
              "raw_nll": float(raw.mean()), "similarity_nll": float(sim.mean()), "oracle_nll": float(oracle.mean()),
              "oracle_improvement": float((raw-oracle).mean()), "oracle_helped_fraction": float((oracle < raw).mean()),
              "similarity_improvement": float((raw-sim).mean()), "deployable": False,
              "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"), "split_sha256": sha256(SPLIT_FILES[split])}
    atomic_json(ANALYSIS / f"results/oracle/{split}_{bank}_k{k}_s{seed}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def score_configuration(split: str, bank: str, k: int, seed: int, topk: int,
                        temperature: float, alpha: float, mode: str, batch_size: int,
                        limit: int = -1) -> dict[str, Any]:
    model, tokenizer, device = load_model()
    ds = PromptDataset(split, tokenizer, limit)
    cached_vectors, cached_meta = load_evidence(split)
    if limit > 0: cached_vectors, cached_meta = cached_vectors[:limit], cached_meta.iloc[:limit]
    proto_path = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}/centroids.pt"
    centers = torch.load(proto_path, map_location="cpu", weights_only=True).float().to(device)
    global_mean = load_evidence("train")[0].mean(0).to(device) if mode == "mean" else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    raw_all, grounded_all, diag_all = [], [], []
    offset = 0
    rng = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        ids, mask, labels, pos = [batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos")]
        _, embeds = thought_embeds(model, ids, mask, pos)
        q = cached_vectors[offset:offset+len(ids)].to(device)
        p, diag = retrieve(q, centers, topk, temperature)
        if mode == "raw": grounded = q
        elif mode == "hard": grounded = centers[diag["indices"]]
        elif mode == "residual": grounded = q + alpha * (p - q)
        elif mode == "mean": grounded = global_mean.expand_as(q)
        elif mode == "shuffled": grounded = p[torch.randperm(len(p), generator=rng, device=device)]
        elif mode == "random": grounded = torch.randn(q.shape, generator=rng, device=device) * q.std()
        else: raise ValueError(mode)
        raw_all.append(per_example_nll(model, embeds, mask, labels, ids).cpu())
        grounded_all.append(per_example_nll(model, substitute(embeds, pos, grounded), mask, labels, ids).cpu())
        diag_all.append(torch.stack([diag["similarity"], diag["margin"], diag["entropy"]], 1).cpu())
        offset += len(ids)
    raw, grounded, diag = torch.cat(raw_all).numpy(), torch.cat(grounded_all).numpy(), torch.cat(diag_all).numpy()
    delta = raw - grounded
    result = {"split": split, "bank": bank, "k": k, "seed": seed, "topk": topk,
              "temperature": temperature, "alpha": alpha, "mode": mode, "n": len(raw),
              "raw_nll": float(raw.mean()), "grounded_nll": float(grounded.mean()),
              "nll_improvement": float(delta.mean()), "helped_fraction": float((delta > 0).mean()),
              "mean_similarity": float(diag[:, 0].mean()), "mean_margin": float(diag[:, 1].mean()),
              "mean_entropy": float(diag[:, 2].mean()), "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"),
              "split_sha256": sha256(SPLIT_FILES[split]), "prototype_sha256": sha256(proto_path)}
    key = f"{split}_{bank}_k{k}_s{seed}_{mode}_t{topk}_temp{temperature}_a{alpha}"
    result_path = ANALYSIS / f"results/frozen_grounding/{key}.json"
    atomic_json(result_path, result)
    np.savez_compressed(result_path.with_suffix(".npz"), raw_nll=raw, grounded_nll=grounded,
                        similarity=diag[:, 0], margin=diag[:, 1], entropy=diag[:, 2])
    print(json.dumps(result, indent=2), flush=True)
    return result


def recommendation_metrics(ranks: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        out[f"HR@{k}"] = float((ranks <= k).mean())
        out[f"NDCG@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    return out


def generate_rankings(split: str, bank: str, k: int, seed: int, topk: int,
                      temperature: float, alpha: float, mode: str, batch_size: int,
                      limit: int = -1, checkpoint: Path = CHECKPOINT,
                      tag: str = "") -> None:
    """Run the repository's constrained 10-beam catalog evaluator with grounding."""
    model, tokenizer, device = load_model(checkpoint)
    tokenizer.padding_side = "left"
    ds = PromptDataset(split, tokenizer, limit)
    needs_prototypes = mode in {"hard", "residual", "shuffled"}
    centers_path = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}/centroids.pt"
    centers = (torch.load(centers_path, map_location=device, weights_only=True).float()
               if needs_prototypes else None)
    train_vectors, train_meta = load_evidence("train", tag)
    train_mean = train_vectors.mean(0).to(device)
    train_norm_mean = train_vectors.norm(dim=-1).mean().to(device)
    prefix_data = get_prefix_data(str(INFO_FILE), tokenizer)
    rng = torch.Generator(device=device).manual_seed(seed)
    current_rows: list[dict[str, Any]] = []

    def allowed(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        return prefix_data.get(get_hash(input_ids), [])

    # Generation calls this only for the uncached prompt pass. The dynamic query
    # preserves the original left-padding/position behavior of noise_eval.py.
    def grounded_generate_embs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _share_positions(model, input_ids)
        first = model.model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True, use_cache=False)
        thought_id = model.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought_id)
        pos = where[:, 1]
        # History-only pooling checkpoints restrict the pooling keys (same mask as training / ranking).
        pool_mask = model.pooling_mask(input_ids, attention_mask) if hasattr(model, "pooling_mask") else attention_mask
        q = model.attention(first.last_hidden_state, pool_mask, pos).float()
        if needs_prototypes:
            assert centers is not None
            p, diag = retrieve(q, centers, topk, temperature)
        if mode == "raw": grounded = q
        elif mode == "mask": grounded = torch.zeros_like(q)
        elif mode == "mu_only":
            fixed_mu = getattr(model.attention, "fixed_mu", None)
            initialized = getattr(model.attention, "fixed_mu_initialized", False)
            if fixed_mu is None or not bool(initialized):
                raise ValueError("mu_only requires an Attempt-9 residual-correction checkpoint")
            grounded = fixed_mu.to(device).expand_as(q)
        elif mode == "mean": grounded = train_mean.expand_as(q)
        elif mode == "hard": grounded = centers[diag["indices"]]
        elif mode == "residual": grounded = q + alpha * (p - q)
        elif mode == "shuffled": grounded = p[torch.randperm(len(p), generator=rng, device=device)]
        elif mode in {"shuffle_user", "shuffle_target"}:
            picked = []
            expanded_rows = [current_rows[min(i * len(current_rows) // len(q), len(current_rows) - 1)] for i in range(len(q))]
            for row in expanded_rows:
                idx = (int(row["sample_id"]) * 104729 + seed * 7919) % len(train_vectors)
                key = "user_id" if mode == "shuffle_user" else "target_item"
                while str(train_meta.iloc[idx][key]) == str(row[key]):
                    idx = (idx + 1) % len(train_vectors)
                picked.append(idx)
            grounded = train_vectors[picked].to(device)
        elif mode == "z_alone":
            # mu-free residual: z_i = q - mu, re-normed to the typical raw-r norm so a
            # magnitude mismatch (mu's norm dominates q) doesn't get read as a content result.
            z = q - train_mean
            grounded = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * train_norm_mean
        elif mode == "noise_matched":
            # Null control for z_alone: same fixed norm, but a direction uncorrelated with q,
            # so any z_alone vs. noise_matched gap isolates z_i's content, not its magnitude.
            noise = torch.randn(q.shape, generator=rng, device=device)
            grounded = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8) * train_norm_mean
        elif mode.startswith("noise_"):
            scale = float(mode.split("_", 1)[1])
            grounded = q + torch.randn(q.shape, generator=rng, device=device) * train_vectors.std().to(device) * scale
        elif mode == "random": grounded = torch.randn(q.shape, generator=rng, device=device) * train_vectors.std().to(device)
        else: raise ValueError(mode)
        embs = model.model.embed_tokens(input_ids).clone()
        embs[where[:, 0], where[:, 1]] = grounded.to(embs.dtype)
        return embs

    model.generate_embs = grounded_generate_embs  # type: ignore[method-assign]
    gen_config = GenerationConfig(num_beams=10, num_return_sequences=10, length_penalty=1.0,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  max_new_tokens=128)
    records: list[dict[str, Any]] = []
    catalog = {line.split("\t")[0].strip(' \n"') for line in INFO_FILE.open()}
    violations = 0
    for start in range(0, len(ds), batch_size):
        rows = ds.rows[start:start+batch_size]
        current_rows = rows
        prompts = [r["input_ids"][:r["prompt_len"]] for r in rows]
        max_len, min_len = max(map(len, prompts)), min(map(len, prompts))
        ids = torch.tensor([[tokenizer.pad_token_id] * (max_len-len(x)) + x for x in prompts], device=device)
        masks = torch.tensor([[0] * (max_len-len(x)) + [1] * len(x) for x in prompts], device=device)
        processor = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None,
                    prefix_allowed_tokens_fn=allowed, cf_dict=None, unconditional_ids=None,
                    model=model, tokenizer=tokenizer, num_beams=10)
        with torch.no_grad():
            generated = model.generate(ids, attention_mask=masks, generation_config=gen_config,
                        return_dict_in_generate=True, output_scores=True,
                        logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), processor]))
        decoded = tokenizer.batch_decode(generated.sequences[:, min_len:], skip_special_tokens=True)
        decoded = [x.split("Response:")[-1].strip(' \n"') for x in decoded]
        for i, row in enumerate(rows):
            pred = decoded[i*10:(i+1)*10]
            violations += sum(x not in catalog for x in pred)
            target = row["target_title"].strip(' \n"')
            rank = pred.index(target) + 1 if target in pred else 10**9
            records.append({"sample_id": row["sample_id"], "user_id": row["user_id"],
                            "target_item": row["target_item"], "target": target,
                            "prediction": pred, "rank": rank})
        if start % (batch_size * 25) == 0:
            print(f"generate {mode}: {min(start+batch_size,len(ds))}/{len(ds)}", flush=True)
    ranks = np.array([x["rank"] for x in records])
    result = {"split": split, "bank": bank, "k": k, "seed": seed, "topk": topk,
              "temperature": temperature, "alpha": alpha, "mode": mode, "n": len(records),
              "catalog_violations": violations, **recommendation_metrics(ranks),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
              "split_sha256": sha256(SPLIT_FILES[split]),
              "prototype_sha256": sha256(centers_path) if needs_prototypes else None}
    key = f"{split}_{bank}_k{k}_s{seed}_{mode}_t{topk}_temp{temperature}_a{alpha}"
    out_dir = ANALYSIS / "results/frozen_grounding"
    if tag:
        out_dir = ANALYSIS / "results/intervention" / tag
    out = out_dir / f"generation_{key}.json"
    atomic_json(out, {"metrics": result, "records": records})
    print(json.dumps(result, indent=2), flush=True)


def bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int = 42, reps: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = a - b
    means = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(reps)])
    return {"mean": float(delta.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean()))}


def intervention(split: str, batch_size: int, limit: int = -1,
                 checkpoint: Path = CHECKPOINT, tag: str = "") -> None:
    model, tokenizer, device = load_model(checkpoint)
    ds = PromptDataset(split, tokenizer, limit)
    cached, _ = load_evidence(split, tag)
    if limit > 0: cached = cached[:limit]
    train_vec, train_meta = load_evidence("train", tag)
    mean = train_vec.mean(0).to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    names = ["raw", "mask", "mean", "shuffle_user", "shuffle_target", "noise_0.25", "noise_0.5", "noise_1.0"]
    fixed_mu = getattr(model.attention, "fixed_mu", None)
    fixed_mu_initialized = getattr(model.attention, "fixed_mu_initialized", False)
    if fixed_mu is not None and bool(fixed_mu_initialized):
        names.insert(2, "mu_only")
    values = {x: [] for x in names}
    offset = 0
    for batch_i, batch in enumerate(loader):
        ids, mask, labels, pos = [batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos")]
        _, embeds = thought_embeds(model, ids, mask, pos)
        q = cached[offset:offset+len(ids)].to(device)
        gen = torch.Generator(device=device).manual_seed(42 + batch_i)
        other = torch.randint(len(train_vec), (len(ids),), generator=torch.Generator().manual_seed(42 + batch_i))
        variants = {"raw": q, "mask": torch.zeros_like(q), "mean": mean.expand_as(q),
                    "shuffle_user": train_vec[other].to(device),
                    "shuffle_target": train_vec[(other + len(train_vec)//2) % len(train_vec)].to(device)}
        if "mu_only" in names:
            variants["mu_only"] = fixed_mu.to(device).expand_as(q)
        std = train_vec.std().to(device)
        for scale in (.25, .5, 1.0):
            variants[f"noise_{scale}"] = q + torch.randn(q.shape, generator=gen, device=device) * std * scale
        for name, vec in variants.items():
            values[name].append(per_example_nll(model, substitute(embeds, pos, vec), mask, labels, ids).cpu())
        offset += len(ids)
        if batch_i % 50 == 0: print(f"intervention {split}: {offset}/{len(ds)}", flush=True)
    arrays = {k: torch.cat(v).numpy() for k, v in values.items()}
    raw = arrays["raw"]
    summary = {"split": split, "n": len(raw), "variants": {},
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
               "split_sha256": sha256(SPLIT_FILES[split])}
    for name, arr in arrays.items():
        stats = bootstrap_delta(raw, arr)
        summary["variants"][name] = {"mean_nll": float(arr.mean()), "delta_nll_vs_raw": float(arr.mean()-raw.mean()),
                                              "raw_minus_variant": stats}
    out_dir = ANALYSIS / "results/intervention"
    if tag:
        out_dir = out_dir / tag
    out = out_dir / f"{split}_summary.json"
    atomic_json(out, summary)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(json.dumps(summary, indent=2), flush=True)


def manifest() -> None:
    config = {
        "dataset": "CDs_and_Vinyl", "checkpoint": str(CHECKPOINT), "architecture": "LatentModel",
        "hidden_size": 1536, "thought_steps": 1, "seeds": list(SEEDS), "prototype_counts": list(KS),
        "metrics": ["HR@1", "HR@3", "HR@5", "HR@10", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "target_NLL"],
        "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"),
        "splits": {k: {"path": str(v), "sha256": sha256(v), "rows": sum(1 for _ in v.open()) - 1} for k, v in SPLIT_FILES.items()},
        "artifact_root": str(ANALYSIS), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(ANALYSIS / "config/prototype_grounding.json", config)
    baseline = json.load((CHECKPOINT / "final_result.json").open())
    ranks = np.array(list(ranks_from_existing_predictions().values()))
    metrics = {"n": len(baseline), "catalog_violations": 60, "checkpoint_sha256": config["checkpoint_sha256"],
               "split_sha256": config["splits"]["test"]["sha256"]}
    for k in (1, 3, 5, 10):
        metrics[f"HR@{k}"] = float((ranks <= k).mean())
        metrics[f"NDCG@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    atomic_json(ANALYSIS / "results/baseline_1T/metrics.json", metrics)
    print(json.dumps(config, indent=2)); print(json.dumps(metrics, indent=2))


def build_cf_clusters(n_clusters: int = 64, rank: int = 32, seed: int = 42) -> None:
    """Derive leakage-safe item CF clusters from train user-item co-occurrence."""
    from scipy.cluster.vq import kmeans2
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    frame = pd.read_csv(SPLIT_FILES["train"])
    users = {u: i for i, u in enumerate(frame.user_id.unique())}
    n_items = sum(1 for _ in INFO_FILE.open())
    edges: set[tuple[int, int]] = set()
    for row in frame.itertuples():
        uid = users[row.user_id]
        for item in ast.literal_eval(row.history_item_id) + [int(row.item_id)]:
            if 0 <= int(item) < n_items:
                edges.add((uid, int(item)))
    rr = np.fromiter((x[0] for x in edges), dtype=np.int32)
    cc = np.fromiter((x[1] for x in edges), dtype=np.int32)
    mat = coo_matrix((np.ones(len(edges), dtype=np.float32), (rr, cc)),
                     shape=(len(users), n_items)).tocsr()
    _, singular, vt = svds(mat, k=rank, random_state=seed)
    factors = vt.T[:, ::-1] * singular[::-1]
    factors /= np.linalg.norm(factors, axis=1, keepdims=True).clip(1e-8)
    _, clusters = kmeans2(factors, n_clusters, iter=50, minit="++", seed=seed)
    out = ANALYSIS / f"cache/{CATEGORY}_cf_item_clusters.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": np.arange(n_items), "cf_cluster": clusters}).to_csv(out, index=False)
    mapping = dict(enumerate(clusters.tolist()))
    for split in SPLIT_FILES:
        meta_path = ANALYSIS / f"evidence/{CATEGORY}/{split}/metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            meta["target_category"] = CATEGORY
            meta["cf_cluster"] = meta.target_item.map(mapping).fillna(-1).astype(int)
            meta.to_csv(meta_path, index=False)
    atomic_json(ANALYSIS / f"cache/{CATEGORY}_cf_item_clusters.json", {
        "method": "train-only binary user-item matrix truncated SVD followed by k-means",
        "rank": rank, "n_clusters": n_clusters, "seed": seed, "users": len(users),
        "items": n_items, "edges": len(edges), "train_split_sha256": sha256(SPLIT_FILES["train"]),
        "artifact_sha256": sha256(out)})
    print(f"CF clusters: users={len(users)} items={n_items} edges={len(edges)}", flush=True)


def evidence_report() -> None:
    summaries: dict[str, Any] = {}
    train = pd.read_csv(ANALYSIS / "evidence/CDs_and_Vinyl/train/metadata.csv")
    subset_dir = ANALYSIS / "evidence/CDs_and_Vinyl/train/subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_counts = {}
    for bank in ("all", "positive", "top", "harmful"):
        indices = np.flatnonzero(bank_mask(train, bank))
        np.save(subset_dir / f"{bank}.npy", indices)
        subset_counts[bank] = len(indices)
    user_activity = train.groupby("user_id").size()
    item_popularity = train.groupby("target_item").size()
    for split in ("train", "valid", "test"):
        path = ANALYSIS / f"evidence/CDs_and_Vinyl/{split}/metadata.csv"
        if not path.exists(): continue
        meta = pd.read_csv(path)
        meta["user_activity"] = meta.user_id.map(user_activity).fillna(0)
        meta["item_popularity"] = meta.target_item.map(item_popularity).fillna(0)
        meta["user_group"] = pd.cut(meta.user_activity, [-1, 1, 4, np.inf], labels=["sparse", "medium", "active"])
        q1, q2 = train.target_item.map(item_popularity).quantile([.33, .67])
        meta["item_group"] = pd.cut(meta.item_popularity, [-1, q1, q2, np.inf], labels=["tail", "torso", "head"], duplicates="drop")
        grouped = {}
        for column in ("user_group", "item_group"):
            grouped[column] = {str(key): {"n": len(g), "mean_utility": float(g.utility.mean()),
                                                  "positive_fraction": float((g.utility > 0).mean())}
                               for key, g in meta.groupby(column, observed=True)}
        quantiles = {str(q): float(meta.utility.quantile(q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}
        summaries[split] = {"n": len(meta), "mean_utility": float(meta.utility.mean()),
                            "positive_fraction": float((meta.utility > 0).mean()),
                            "utility_quantiles": quantiles, "groups": grouped}
    summaries["subset_counts"] = subset_counts
    summaries["correct_bank_status"] = "unavailable: no train-split constrained rankings exist in the supplied checkpoint"
    atomic_json(ANALYSIS / "results/intervention/evidence_utility_summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def prototype_report() -> None:
    from scipy.optimize import linear_sum_assignment
    report: dict[str, Any] = {"configurations": {}, "unavailable": {
        "positive": "identical to all because all train utilities are positive",
        "harmful": "empty because no train utility is negative",
        "correct": "no constrained train rankings supplied"}}
    for bank in ("all", "top"):
        for k in KS:
            seed_rows, centers_by_seed = [], {}
            for seed in SEEDS:
                base = ANALYSIS / f"prototypes/CDs_and_Vinyl/{bank}/{k}/seed_{seed}"
                diag = pd.read_csv(base / "diagnostics.csv")
                assignments = np.load(base / "assignments.npy")
                counts = np.bincount(assignments, minlength=k)
                probs = counts / counts.sum()
                entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
                seed_rows.append({"seed": seed, "effective_prototypes": float(np.exp(entropy)),
                    "min_interactions": int(counts.min()), "max_interactions": int(counts.max()),
                    "median_distinct_users": float(diag.distinct_users.median()),
                    "max_dominant_user_fraction": float(diag.dominant_user_fraction.max()),
                    "mean_assignment_similarity": float(diag.mean_similarity.mean()),
                    "mean_nearest_prototype_similarity": float(diag.nearest_prototype_similarity.mean())})
                centers_by_seed[seed] = F.normalize(torch.load(base / "centroids.pt", map_location="cpu", weights_only=True).float(), dim=1)
            stability = []
            for a, b in ((13, 42), (13, 73), (42, 73)):
                sim = (centers_by_seed[a] @ centers_by_seed[b].T).numpy()
                row, col = linear_sum_assignment(-sim)
                stability.append({"seeds": [a, b], "matched_cosine": float(sim[row, col].mean())})
            report["configurations"][f"{bank}_k{k}"] = {"seeds": seed_rows, "stability": stability}
    atomic_json(ANALYSIS / "results/frozen_grounding/prototype_diagnostics_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def latent_distribution_report(seed: int = 42, pair_samples: int = 100_000,
                               pca_samples: int = 20_000, tag: str = "",
                               checkpoint: Path = CHECKPOINT, component: str = "r") -> None:
    """Measure and visualize diversity/collapse in extracted train thoughts.

    component="r" (default): the combined thought vector spliced into the
    decoder, r_i = mu_i + z_i for mu/z-decomposed checkpoints. component="z":
    the residual alone, saved by `extract` as z_only.pt when the checkpoint
    exposes model.attention.last_z. Since mu's norm is architecturally fixed
    (VAEDecomposedSelfAttentionLayer), running this on r conflates mu's fixed
    contribution with z's own variance; component="z" measures z directly.
    """
    seed_all(seed)
    rng = np.random.default_rng(seed)
    base = ANALYSIS / f"evidence/{CATEGORY}"
    if tag:
        base = base / tag
    base = base / "train"
    vector_file = "thoughts.pt" if component == "r" else "z_only.pt"
    if component == "z" and not (base / vector_file).exists():
        raise FileNotFoundError(
            f"{base / vector_file} not found -- re-run `extract` for this checkpoint/tag "
            "(z_only.pt is only saved for mu/z-decomposed checkpoints, and only after this fix)."
        )
    stored_vectors = torch.load(base / vector_file, map_location="cpu", weights_only=True)
    vectors = stored_vectors.float()
    meta = pd.read_csv(base / "metadata.csv")
    if "cf_cluster" not in meta.columns:
        cf_path = ANALYSIS / f"cache/{CATEGORY}_cf_item_clusters.csv"
        if not cf_path.exists():
            raise FileNotFoundError(f"CF-cluster mapping is required; run `cf-clusters --category {CATEGORY}` first")
        cf = pd.read_csv(cf_path).set_index("item_id").cf_cluster
        meta["cf_cluster"] = meta.target_item.map(cf).fillna(-1).astype(int)
    if len(vectors) != len(meta):
        raise ValueError(f"vector/metadata mismatch: {len(vectors)} != {len(meta)}")
    n, dim = vectors.shape
    unit = F.normalize(vectors, dim=1)
    mean = vectors.mean(0)
    mean_norm = mean.norm()
    norms = vectors.norm(dim=1)
    centered_norms = (vectors - mean).norm(dim=1)
    cosine_to_mean = unit @ F.normalize(mean, dim=0)
    stored_array = np.ascontiguousarray(stored_vectors.numpy())
    packed_rows = stored_array.view(np.dtype(
        (np.void, stored_array.dtype.itemsize * stored_array.shape[1]))).reshape(-1)
    _, exact_codes, exact_counts = np.unique(packed_rows, return_inverse=True, return_counts=True)
    exact_counts = np.sort(exact_counts)[::-1]

    def describe(values: torch.Tensor | np.ndarray) -> dict[str, float]:
        x = torch.as_tensor(values).float().cpu()
        q = torch.quantile(x, torch.tensor([0., .01, .05, .25, .5, .75, .95, .99, 1.]))
        return {
            "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
            **{f"q{int(p * 100):02d}": float(v) for p, v in zip(
                (0, .01, .05, .25, .5, .75, .95, .99, 1), q)},
        }

    def random_pairs() -> tuple[np.ndarray, np.ndarray]:
        left = rng.integers(0, n, pair_samples)
        right = rng.integers(0, n, pair_samples)
        right[right == left] = (right[right == left] + 1) % n
        return left, right

    def same_group_pairs(column: str) -> tuple[np.ndarray, np.ndarray]:
        codes, _ = pd.factorize(meta[column], sort=False)
        counts = np.bincount(codes)
        eligible = np.flatnonzero(counts[codes] > 1)
        left = rng.choice(eligible, size=pair_samples, replace=True)
        members = {g: np.flatnonzero(codes == g) for g in np.unique(codes[left])}
        right = np.empty(pair_samples, dtype=np.int64)
        for j, i in enumerate(left):
            choices = members[codes[i]]
            draw = choices[rng.integers(0, len(choices) - 1)]
            if draw == i:
                draw = choices[-1]
            right[j] = draw
        return left, right

    pair_stats: dict[str, Any] = {}
    pair_values: dict[str, np.ndarray] = {}
    for label, pair in {
        "random": random_pairs(),
        "same_user": same_group_pairs("user_id"),
        "same_target": same_group_pairs("target_item"),
        "same_cf_cluster": same_group_pairs("cf_cluster"),
    }.items():
        left, right = pair
        values = (unit[left] * unit[right]).sum(1).numpy()
        pair_values[label] = values
        pair_stats[label] = describe(values)

    # Exact eigenspectrum of a reproducible sample covariance. This quantifies
    # centered diversity; pairwise cosine mostly quantifies the shared offset.
    pca_n = min(pca_samples, n)
    pca_idx = rng.choice(n, pca_n, replace=False)
    x = vectors[pca_idx]
    pca_center = x.mean(0)
    x = x - pca_center
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_device = x.to(device)
    covariance = x_device.T @ x_device / max(1, pca_n - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0).cpu()
    eigenvectors = eigenvectors.flip(1)[:, :2].cpu()
    variance_fraction = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    cumulative = variance_fraction.cumsum(0)
    positive = variance_fraction[variance_fraction > 0]
    entropy_rank = float(torch.exp(-(positive * positive.log()).sum()))
    participation_rank = float(1.0 / variance_fraction.square().sum())

    def components_for(threshold: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(threshold)).item() + 1)

    # Group-centroid variance fractions, corrected by permutations with the
    # same group sizes to expose signal beyond finite-sample overfitting.
    x_cpu = x
    pca_meta = meta.iloc[pca_idx].reset_index(drop=True)
    total_ss = float(x_cpu.square().sum())

    def group_eta(column: str) -> dict[str, float | int]:
        codes_np, uniques = pd.factorize(pca_meta[column], sort=False)
        codes = torch.from_numpy(codes_np).long()
        groups = len(uniques)

        def eta_for(group_codes: torch.Tensor) -> float:
            sums = torch.zeros((groups, dim), dtype=x_cpu.dtype)
            sums.index_add_(0, group_codes, x_cpu)
            counts = torch.bincount(group_codes, minlength=groups).float().clamp_min(1)
            return float((sums.square().sum(1) / counts).sum() / total_ss)

        observed = eta_for(codes)
        null = []
        generator = torch.Generator().manual_seed(seed + len(column))
        for _ in range(10):
            null.append(eta_for(codes[torch.randperm(pca_n, generator=generator)]))
        return {"groups": groups, "observed_eta_squared": observed,
                "permutation_mean": float(np.mean(null)),
                "adjusted_eta_squared": observed - float(np.mean(null))}

    group_variance = {c: group_eta(c) for c in ("user_id", "target_item", "cf_cluster")}

    scalar_std = float(vectors.std())
    expected_noise_norm = {str(scale): scalar_std * math.sqrt(dim) * scale
                           for scale in (.25, .5, 1.)}
    sl, sr = random_pairs()
    shuffled_distance = (vectors[sl] - vectors[sr]).norm(dim=1)
    from scipy.stats import pearsonr, spearmanr
    utility_values = meta.utility.to_numpy()

    def utility_association(values: np.ndarray) -> dict[str, float]:
        return {"pearson": float(pearsonr(values, utility_values).statistic),
                "spearman": float(spearmanr(values, utility_values).statistic)}

    utility_frame = pd.DataFrame({"template": exact_codes, "utility": utility_values})
    template_utility = utility_frame.groupby("template").utility.agg(["count", "mean"])
    utility_ss = float(np.square(utility_values - utility_values.mean()).sum())
    template_between_ss = float((template_utility["count"] *
                                 np.square(template_utility["mean"] - utility_values.mean())).sum())

    report = {
        "dataset": CATEGORY, "split": "train", "n": n, "dimension": dim,
        "seed": seed, "pair_samples": pair_samples, "pca_samples": pca_n,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "split_sha256": sha256(SPLIT_FILES["train"]),
        "norm": describe(norms), "mean_vector_norm": float(mean_norm),
        "mean_norm_over_mean_sample_norm": float(mean_norm / norms.mean()),
        "centered_norm": describe(centered_norms),
        "centered_norm_over_sample_norm_mean": float((centered_norms / norms).mean()),
        "cosine_to_global_mean": describe(cosine_to_mean),
        "exact_templates_at_stored_precision": {
            "stored_dtype": str(stored_vectors.dtype),
            "unique_vectors": int(len(exact_counts)),
            "unique_fraction": float(len(exact_counts) / n),
            "samples_in_repeated_templates": int(exact_counts[exact_counts > 1].sum()),
            "repeated_template_count": int((exact_counts > 1).sum()),
            "largest_template_count": int(exact_counts[0]),
            "top_1_sample_fraction": float(exact_counts[:1].sum() / n),
            "top_5_sample_fraction": float(exact_counts[:5].sum() / n),
            "top_10_sample_fraction": float(exact_counts[:10].sum() / n),
            "top_20_sample_fraction": float(exact_counts[:20].sum() / n),
            "top_20_counts": exact_counts[:20].tolist(),
        },
        "pairwise_cosine": pair_stats,
        "shuffled_vector_distance": describe(shuffled_distance),
        "expected_gaussian_noise_l2": expected_noise_norm,
        "utility_association": {
            "norm": utility_association(norms.numpy()),
            "centered_norm": utility_association(centered_norms.numpy()),
            "cosine_to_global_mean": utility_association(cosine_to_mean.numpy()),
            "exact_template_eta_squared": template_between_ss / utility_ss,
        },
        "pca": {
            "entropy_effective_rank": entropy_rank,
            "participation_ratio": participation_rank,
            "top_eigenvalue_fraction": float(variance_fraction[0]),
            "top_2_fraction": float(cumulative[1]),
            "top_10_fraction": float(cumulative[9]),
            "top_32_fraction": float(cumulative[31]),
            "components_for_50pct": components_for(.5),
            "components_for_90pct": components_for(.9),
            "components_for_95pct": components_for(.95),
            "components_for_99pct": components_for(.99),
        },
        "group_centroid_variance": group_variance,
    }

    out_dir = ANALYSIS / "results/latent_distribution"
    if tag:
        out_dir = out_dir / tag
    fig_dir = ANALYSIS / "figures"
    prefix = f"{tag}_" if tag else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_stem = "train_latent_distribution" if component == "r" else f"train_latent_distribution_{component}"
    report["component"] = component
    atomic_json(out_dir / f"{out_stem}.json", report)
    np.savez_compressed(out_dir / f"{out_stem}_arrays.npz",
                        eigenvalues=eigenvalues.numpy(),
                        variance_fraction=variance_fraction.numpy(),
                        pca_indices=pca_idx,
                        random_pair_cosine=pair_values["random"],
                        same_user_cosine=pair_values["same_user"],
                        same_target_cosine=pair_values["same_target"],
                        same_cf_cluster_cosine=pair_values["same_cf_cluster"])

    os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS / "cache/matplotlib"))
    os.environ.setdefault("TMPDIR", str(ANALYSIS / "cache/tmp"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(norms.numpy(), bins=80, alpha=.8, label="vector norm")
    axes[0, 0].axvline(float(mean_norm), color="crimson", linestyle="--", label="global mean norm")
    axes[0, 0].set(title="Latent-vector norms", xlabel="L2 norm", ylabel="count")
    axes[0, 0].legend()
    for label, values in pair_values.items():
        axes[0, 1].hist(values, bins=100, density=True, histtype="step", linewidth=1.5,
                        label=label.replace("_", " "))
    axes[0, 1].set(title="Pairwise directional similarity", xlabel="cosine", ylabel="density")
    axes[0, 1].legend()
    axes[1, 0].plot(np.arange(1, dim + 1), cumulative.numpy(), linewidth=2)
    for threshold in (.5, .9, .95, .99):
        axes[1, 0].axhline(threshold, color="grey", linewidth=.6, linestyle="--")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(title="Centered PCA cumulative variance", xlabel="components (log scale)", ylabel="variance explained", ylim=(0, 1.01))
    axes[1, 1].hist(cosine_to_mean.numpy(), bins=80, color="tab:green", alpha=.85)
    axes[1, 1].set(title="Alignment with global mean", xlabel="cosine to mean", ylabel="count")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}latent_distribution_overview.png", dpi=180)
    plt.close(fig)

    projections = ((vectors - pca_center) @ eigenvectors).numpy()
    utility = meta.utility.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    density = axes[0].hexbin(projections[:, 0], projections[:, 1], gridsize=75,
                             bins="log", mincnt=1, cmap="magma")
    fig.colorbar(density, ax=axes[0], label="examples per hexagon (log color scale)")
    axes[0].set(title="Exact PCA coordinates (density reveals overlap)",
                xlabel="PC1", ylabel="PC2")
    plot_rng = np.random.default_rng(seed + 10_000)
    x_scale = max(float(np.ptp(projections[:, 0])), 1e-8)
    y_scale = max(float(np.ptp(projections[:, 1])), 1e-8)
    jittered = projections + np.column_stack((
        plot_rng.normal(0, .004 * x_scale, n),
        plot_rng.normal(0, .004 * y_scale, n)))
    axes[1].scatter(jittered[:, 0], jittered[:, 1], s=2, alpha=.10,
                    color="tab:blue", rasterized=True, linewidths=0)
    axes[1].set(title="All rows with small display-only jitter",
                xlabel="PC1", ylabel="PC2")
    fig.suptitle(f"All {n:,} train thoughts in the first two centered PCs")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}latent_pca_projection.png", dpi=180)
    fig.savefig(fig_dir / f"{prefix}latent_pca_all_scatter.png", dpi=180)
    plt.close(fig)

    template_n = np.bincount(exact_codes)
    template_projection = np.zeros((len(template_n), 2), dtype=np.float64)
    template_utility_mean = np.zeros(len(template_n), dtype=np.float64)
    np.add.at(template_projection, exact_codes, projections)
    np.add.at(template_utility_mean, exact_codes, utility)
    template_projection /= template_n[:, None]
    template_utility_mean /= template_n
    marker_size = 12 + 650 * np.sqrt(template_n / template_n.max())
    fig, ax = plt.subplots(figsize=(10, 8))
    points = ax.scatter(template_projection[:, 0], template_projection[:, 1],
                        c=template_utility_mean, s=marker_size, alpha=.65,
                        cmap="viridis", edgecolors="white", linewidths=.25)
    fig.colorbar(points, ax=ax, label="mean target-NLL gain versus zero vector")
    for count in (100, 1_000, 5_000):
        size = 12 + 650 * math.sqrt(count / template_n.max())
        ax.scatter([], [], s=size, color="grey", alpha=.65, edgecolors="white",
                   linewidths=.25, label=f"n={count:,}")
    ax.legend(title="template frequency", loc="lower left", frameon=True)
    ax.set(title="All train thoughts as frequency-weighted exact-template PCA points",
           xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}latent_pca_weighted_templates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(1, 21), exact_counts[:20], color="tab:purple")
    ax.set(title="Frequency of the 20 most common exact latent templates (stored precision)",
           xlabel="template frequency rank", ylabel="training examples", xticks=np.arange(1, 21))
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}latent_template_frequency.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    x = sub.add_parser("cf-clusters"); x.add_argument("--category", default="CDs_and_Vinyl")
    sub.add_parser("evidence-report")
    sub.add_parser("prototype-report")
    x = sub.add_parser("latent-distribution"); x.add_argument("--seed", type=int, default=42); x.add_argument("--pair-samples", type=int, default=100_000); x.add_argument("--pca-samples", type=int, default=20_000); x.add_argument("--tag", default=""); x.add_argument("--checkpoint", type=Path, default=CHECKPOINT); x.add_argument("--category", default="CDs_and_Vinyl"); x.add_argument("--component", choices=("r", "z"), default="r")
    x = sub.add_parser("extract"); x.add_argument("--split", choices=("train", "valid", "test"), required=True); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1); x.add_argument("--tag", default=""); x.add_argument("--checkpoint", type=Path, default=CHECKPOINT); x.add_argument("--category", default="CDs_and_Vinyl")
    x = sub.add_parser("intervention"); x.add_argument("--split", choices=("valid", "test"), default="valid"); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1); x.add_argument("--tag", default=""); x.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    x = sub.add_parser("mine"); x.add_argument("--bank", choices=("all", "correct", "positive", "top", "harmful"), required=True); x.add_argument("--k", type=int, required=True); x.add_argument("--seed", type=int, required=True)
    x = sub.add_parser("score"); x.add_argument("--split", choices=("valid", "test"), required=True); x.add_argument("--bank", required=True); x.add_argument("--k", type=int, required=True); x.add_argument("--seed", type=int, required=True); x.add_argument("--topk", type=int, default=4); x.add_argument("--temperature", type=float, default=.1); x.add_argument("--alpha", type=float, default=.5); x.add_argument("--mode", choices=("raw", "hard", "residual", "mean", "shuffled", "random"), default="residual"); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1)
    x = sub.add_parser("generate"); x.add_argument("--split", choices=("valid", "test"), required=True); x.add_argument("--bank", required=True); x.add_argument("--k", type=int, required=True); x.add_argument("--seed", type=int, required=True); x.add_argument("--topk", type=int, default=4); x.add_argument("--temperature", type=float, default=.1); x.add_argument("--alpha", type=float, default=.5); x.add_argument("--mode", choices=("raw", "mask", "mu_only", "hard", "residual", "mean", "shuffled", "shuffle_user", "shuffle_target", "noise_0.25", "noise_0.5", "noise_1.0", "random", "z_alone", "noise_matched"), default="residual"); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1); x.add_argument("--tag", default=""); x.add_argument("--checkpoint", type=Path, default=CHECKPOINT); x.add_argument("--category", default="CDs_and_Vinyl")
    x = sub.add_parser("collaborative"); x.add_argument("--split", choices=("valid", "test"), required=True); x.add_argument("--bank", required=True); x.add_argument("--k", type=int, required=True); x.add_argument("--seed", type=int, required=True); x.add_argument("--topk", type=int, default=4); x.add_argument("--temperature", type=float, default=.1); x.add_argument("--alpha", type=float, default=.5); x.add_argument("--exclusion", choices=("none", "user", "target", "both"), required=True); x.add_argument("--min-users", type=int, default=1); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1)
    x = sub.add_parser("oracle"); x.add_argument("--split", choices=("valid", "test"), required=True); x.add_argument("--bank", required=True); x.add_argument("--k", type=int, required=True); x.add_argument("--seed", type=int, required=True); x.add_argument("--candidate-k", type=int, default=8); x.add_argument("--batch-size", type=int, default=8); x.add_argument("--limit", type=int, default=-1)
    args = p.parse_args()
    if getattr(args, "category", None):
        configure_category(args.category)
    if args.command == "manifest": manifest()
    elif args.command == "cf-clusters": build_cf_clusters()
    elif args.command == "evidence-report": evidence_report()
    elif args.command == "prototype-report": prototype_report()
    elif args.command == "latent-distribution": latent_distribution_report(args.seed, args.pair_samples, args.pca_samples, args.tag, args.checkpoint, args.component)
    elif args.command == "extract": extract(args.split, args.batch_size, args.limit, args.checkpoint, args.tag)
    elif args.command == "intervention": intervention(args.split, args.batch_size, args.limit, args.checkpoint, args.tag)
    elif args.command == "mine": mine(args.bank, args.k, args.seed)
    elif args.command == "score": score_configuration(args.split, args.bank, args.k, args.seed, args.topk, args.temperature, args.alpha, args.mode, args.batch_size, args.limit)
    elif args.command == "generate": generate_rankings(args.split, args.bank, args.k, args.seed, args.topk, args.temperature, args.alpha, args.mode, args.batch_size, args.limit, args.checkpoint, args.tag)
    elif args.command == "collaborative": collaborative_score(args.split, args.bank, args.k, args.seed, args.topk, args.temperature, args.alpha, args.exclusion, args.min_users, args.batch_size, args.limit)
    elif args.command == "oracle": oracle_score(args.split, args.bank, args.k, args.seed, args.candidate_k, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
