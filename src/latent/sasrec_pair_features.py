"""Training-only SASRec collaborative signals for Attempt 11.

A Cornac SASRec checkpoint trained on the same split supplies two frozen CF
geometries:

* a sequence embedding per CSV row: SASRec's final hidden state after reading
  the row's history (the representation SASRec scores the next item with);
* an item embedding per catalog item, looked up for the row's target.

Both are L2-normalized so that dot products are cosine similarities.  They are
used only to choose and weight contrastive pairs; they never enter inference.
"""

import ast
import io
import os
import pickle

import numpy as np
import pandas as pd
import torch


def load_sasrec(checkpoint):
    """Load a pickled Cornac SASRec onto the CPU, even if saved from CUDA."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cornac-matplotlib")
    import torch.storage as torch_storage

    original = torch_storage._load_from_bytes
    torch_storage._load_from_bytes = lambda payload: torch.load(
        io.BytesIO(payload), weights_only=False, map_location="cpu"
    )
    try:
        with open(checkpoint, "rb") as handle:
            recommender = pickle.load(handle)
    finally:
        torch_storage._load_from_bytes = original
    recommender.model.to("cpu").eval()
    recommender.model.dev = "cpu"
    recommender.device = "cpu"
    return recommender


def item_embeddings(recommender):
    """Return normalized real-item embeddings (padding row removed)."""
    weight = recommender.model.item_emb.weight.detach().float()[: recommender.num_items]
    return torch.nn.functional.normalize(weight, dim=-1).numpy().astype(np.float32)


@torch.no_grad()
def encode_rows(recommender, frame, batch_size=4096):
    """Return (normalized sequence embeddings, target item index) per row.

    Histories are ASIN lists; unknown history items are dropped as in the
    baseline's scoring path.  An unknown target gets index -1.
    """
    iid_map = recommender.iid_map
    num_items = recommender.num_items
    max_len = recommender.max_len
    pad = recommender.pad_idx

    sequences = []
    for value in frame["item_asins"]:
        history = [iid_map[a] for a in ast.literal_eval(value) if iid_map.get(a, num_items) < num_items]
        history = history[-max_len:]
        sequences.append([pad] * (max_len - len(history)) + history)
    sequences = torch.as_tensor(sequences, dtype=torch.long)

    outputs = []
    for start in range(0, len(sequences), batch_size):
        outputs.append(recommender.model._encode(sequences[start:start + batch_size]).float())
    sequence_vectors = torch.nn.functional.normalize(torch.cat(outputs), dim=-1).numpy()

    targets = np.asarray(
        [iid_map[a] if iid_map.get(a, num_items) < num_items else -1 for a in frame["item_asin"]],
        dtype=np.int64,
    )
    return sequence_vectors.astype(np.float32), targets


def build_sasrec_features(checkpoint, csv_files):
    """Encode each CSV and return the shared item table plus per-file rows."""
    recommender = load_sasrec(checkpoint)
    items = item_embeddings(recommender)
    rows = {}
    for path in csv_files:
        frame = pd.read_csv(path, usecols=["item_asins", "item_asin"])
        rows[path] = encode_rows(recommender, frame)
    return items, rows


def similarity_report(items, sequence_vectors, targets, sample=4000, bank=10_000, topk=16, seed=0):
    """Summarize random-pair and top-k cosine similarity in both spaces."""
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(targets), size=min(sample, len(targets)), replace=False)
    pool = rng.choice(len(targets), size=min(bank, len(targets)), replace=False)
    known = targets >= 0

    seq_sim = sequence_vectors[anchors] @ sequence_vectors[pool].T
    seq_sim[anchors[:, None] == pool[None, :]] = np.nan
    a, p = anchors[known[anchors]], pool[known[pool]]
    item_sim = items[targets[a]] @ items[targets[p]].T
    item_sim[a[:, None] == p[None, :]] = np.nan
    report = {}
    for name, values in (("sequence", seq_sim), ("item", item_sim)):
        top = -np.sort(-np.nan_to_num(values, nan=-np.inf), axis=1)[:, :topk]
        q = np.nanquantile(values, [0.5, 0.9, 0.99])
        report[name] = {
            "random_mean": float(np.nanmean(values)),
            "random_p50": float(q[0]), "random_p90": float(q[1]), "random_p99": float(q[2]),
            "top1_mean": float(top[:, 0].mean()), f"top{topk}_mean": float(top[:, -1].mean()),
        }
    report["unknown_target_rows"] = int((~known).sum())
    return report


def main(
    checkpoint: str = "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/Musical_Instruments_5_2015-1-2018-11/models/SASRec/2026-09-26_00-07-12-933295.pkl",
    train_file: str = "data/Musical_Instruments/train/Musical_Instruments_5_2015-1-2018-11.csv",
):
    items, rows = build_sasrec_features(checkpoint, [train_file])
    sequence_vectors, targets = rows[train_file]
    print({"rows": len(targets), "catalog_items": len(items), "dim": items.shape[1]})
    for key, value in similarity_report(items, sequence_vectors, targets).items():
        print(key, value)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
