"""Split loading, item maps and per-user timelines for the collaborative-reasoning experiments.

Every row of a split CSV is one next-item example: a history of up to 10 items
and a target.  Rows are globally chronological across splits (train targets <
valid targets < test targets), so a user's full timeline is the first row's
history followed by the targets of all of that user's rows in order.  The full
timeline lets a source read interactions older than the 10-item prompt window
(idea 6 in document/collaborative_reasoning_experiments.md).
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "latent"))  # sasrec_pair_features
CATEGORY = "Musical_Instruments"
PREFIX = "Musical_Instruments_5_2015-1-2018-11"
SPLITS = ("train", "valid", "test")
DATA = ROOT / "data" / CATEGORY
INFO_FILE = DATA / "info" / f"{PREFIX}.txt"
SASREC_FULL = Path(
    "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
    "Musical_Instruments_5_2015-1-2018-11/models/SASRec/2026-09-26_00-07-12-933295.pkl"
)
V1_PREDICTIONS = Path("/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1")
RESULTS = Path("/common/home/users/m/mk.ngohuu.2024/research/R3/analysis/results/collab_reasoning")


def split_file(split: str) -> Path:
    return DATA / split / f"{PREFIX}.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_title(x: str) -> str:
    return str(x).strip(' \n"')


@dataclass
class Rows:
    """Parsed rows of one split (arrays aligned with the CSV order)."""

    split: str
    user: np.ndarray            # str user ids
    history: list               # list[list[int]] dataset item ids (<= 10)
    history_ts: list            # list[list[int]]
    target: np.ndarray          # int item ids
    target_ts: np.ndarray       # int
    cutoff: np.ndarray          # latest observed history timestamp (recommendation time)
    full_past: list | None = None  # list[list[int]]: the user's whole timeline before the target

    def __len__(self) -> int:
        return len(self.target)


def load_rows(split: str, limit: int = -1) -> Rows:
    frame = pd.read_csv(split_file(split))
    if limit > 0:
        frame = frame.iloc[:limit]
    history = [list(map(int, ast.literal_eval(x))) for x in frame.history_item_id]
    history_ts = [list(map(int, ast.literal_eval(x))) for x in frame.history_timestamp]
    return Rows(
        split=split,
        user=frame.user_id.astype(str).to_numpy(),
        history=history,
        history_ts=history_ts,
        target=frame.item_id.to_numpy(dtype=np.int64),
        target_ts=frame.timestamp.to_numpy(dtype=np.int64),
        cutoff=np.array([max(t) for t in history_ts], dtype=np.int64),
    )


def load_all(limit: int = -1) -> dict[str, Rows]:
    rows = {s: load_rows(s, limit if s != "train" else -1) for s in SPLITS}
    attach_full_past(rows)
    return rows


def attach_full_past(rows: dict[str, Rows]) -> None:
    """Reconstruct each user's timeline from all splits and attach every row's full past.

    The CSVs keep per-user order within and across splits; each row's 10-item
    history must be a suffix of the reconstructed past (checked).
    """
    timeline: dict[str, list[int]] = {}
    for split in SPLITS:
        r = rows[split]
        past = []
        for u, h, t in zip(r.user, r.history, r.target):
            seq = timeline.get(u)
            if seq is None:
                seq = list(h)
                timeline[u] = seq
            n = min(len(seq), len(h))
            if seq[-n:] != h[-n:]:
                raise ValueError(f"timeline mismatch for user {u} in {split}")
            past.append(list(seq))
            seq.append(int(t))
        r.full_past = past


def n_items() -> int:
    return sum(1 for _ in INFO_FILE.open())


def item_titles() -> list[str]:
    titles = {}
    for line in INFO_FILE.open():
        t, i = line.rstrip("\n").rsplit("\t", 1)
        titles[int(i)] = clean_title(t)
    return [titles.get(i, "") for i in range(max(titles) + 1)]


def item_asin_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for split in SPLITS:
        frame = pd.read_csv(split_file(split), usecols=["item_id", "item_asin", "history_item_id", "item_asins"])
        for i, a in zip(frame.item_id, frame.item_asin):
            mapping.setdefault(int(i), a)
        for ids, asins in zip(frame.history_item_id, frame.item_asins):
            for i, a in zip(ast.literal_eval(ids), ast.literal_eval(asins)):
                mapping.setdefault(int(i), a)
    return mapping


def load_r3_predictions(directory: Path, split_rows: Rows) -> list[list[int]]:
    """R3 top-10 predictions as item ids (-1 for a generated title not in the catalog).

    ``directory`` holds either the standard evaluator's shards (0.json, 1.json, ...)
    or a single ``predictions.json`` written by collab_eval.py ``--predict_only``.
    """
    directory = Path(directory)
    title_to_id = {}
    for i, t in enumerate(item_titles()):
        title_to_id.setdefault(t, i)
    single = directory / "predictions.json"
    if single.exists():
        records = json.load(single.open())
        preds = [[title_to_id.get(clean_title(x), -1) for x in r["top10"]] for r in records]
    else:
        records = []
        for k in range(16):
            shard = directory / f"{k}.json"
            if not shard.exists():
                break
            records += json.load(shard.open())
        preds = [[title_to_id.get(clean_title(x), -1) for x in r["predict"]] for r in records]
    if len(preds) != len(split_rows):
        raise ValueError(f"{directory}: {len(preds)} predictions for {len(split_rows)} rows")
    return preds


def hits_at(topk: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Boolean hit@k for an [n, >=k] array of ranked item ids."""
    return (topk[:, :k] == target[:, None]).any(1)


def ndcg_at(topk: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    match = topk[:, :k] == target[:, None]
    pos = match.argmax(1)
    return np.where(match.any(1), 1.0 / np.log2(pos + 2.0), 0.0)
