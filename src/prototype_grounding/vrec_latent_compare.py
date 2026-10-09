#!/usr/bin/env python3
"""Compare the reasoning-latent distribution of R3 and VRec checkpoints.

Reads what vrec_latent_extract.py wrote (analysis/evidence/<category>/<tag>/{train,test}/)
for each model and reports, for the pooled latent r and for the slot fed to the decoder:

* geometry: norm, anisotropy (|mean| / mean |r|), cosine to the mean, random-pair
  cosine, sample-specific share |r - mean| / |r|, effective rank and participation
  ratio of the centered spectrum, top-1 / top-10 explained variance, unique fp16
  vectors and top-10 template coverage (train split);
* distinguishability: eta^2 of VRec's group labels (15 categories, 25 CF clusters) and
  of the target item, and a linear probe trained on train latents, scored on test;
* decoder use (test): NLL increase when the slot is zeroed (utility) or replaced by
  another row's slot (content utility), with bootstrap CIs;
* VRec verifier (stages 2-3): entropy weight w, router weights, prototype usage,
  verifier accuracy, and how w relates to whether the row is a hit;
* ranking metrics from each model's evaluation output.

Usage:
  python vrec_latent_compare.py --model "R3 baseline=r3_tag:/path/final_result.json" \
      --model "VRec stage 1=stage1_tag:/path/result.json" ... --out analysis/results/vrec_compare/full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import experiment as E

LABELS = Path("/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/data/Musical_Instruments/info")
LABEL_FILES = {"category15": "Musical_Instruments_asin2cat_qwen_cluster_25.npy",
               "cf25": "Musical_Instruments_asin2cat_cf_cluster_25.npy"}


def ci(x, seed=42, reps=2000):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975))}


def geometry(stored: torch.Tensor, seed=42, pairs=100_000, pca_rows=20_000) -> dict:
    x = stored.float()
    n = len(x)
    rng = np.random.default_rng(seed)
    mean = x.mean(0)
    norms = x.norm(dim=1)
    unit = F.normalize(x, dim=1)
    left, right = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = left != right
    pair_cos = (unit[left[keep]] * unit[right[keep]]).sum(1)
    centered = x - mean
    idx = rng.choice(n, size=min(pca_rows, n), replace=False)
    eig = torch.linalg.svdvals(centered[idx]).square()
    share = eig / eig.sum()
    entropy = -(share[share > 0] * share[share > 0].log()).sum()
    arr = np.ascontiguousarray(stored.numpy())
    packed = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).reshape(-1)
    _, counts = np.unique(packed, return_counts=True)
    counts = np.sort(counts)[::-1]
    return {
        "n": n, "mean_norm": float(norms.mean()), "anisotropy": float(mean.norm() / norms.mean()),
        "cosine_to_mean": float((unit @ F.normalize(mean, dim=0)).mean()),
        "random_pair_cosine": float(pair_cos.mean()),
        "sample_specific_share": float((centered.norm(dim=1) / norms.clamp_min(1e-6)).mean()),
        "effective_rank": float(entropy.exp()), "participation_ratio": float(eig.sum() ** 2 / eig.square().sum()),
        "pc1_share": float(share[0]), "top10_pc_share": float(share[:10].sum()),
        "unique_fp16_fraction": float(len(counts) / n), "top10_template_coverage": float(counts[:10].sum() / n),
    }


def eta2(x: torch.Tensor, codes: np.ndarray) -> float:
    """Between-group share of the centered variance (rows with code < 0 excluded)."""
    keep = torch.from_numpy(codes >= 0)
    x = x[keep].float()
    c = torch.from_numpy(codes[codes >= 0]).long()
    x = x - x.mean(0)
    k = int(c.max()) + 1
    sums = torch.zeros(k, x.shape[1]).index_add_(0, c, x)
    counts = torch.bincount(c, minlength=k).float().clamp_min(1)
    return float((sums.square().sum(1) / counts).sum() / x.square().sum())


def probe(train_x, train_y, test_x, test_y, steps=3000, seed=0):
    """Multinomial logistic regression on standardized latents; test accuracy and majority baseline."""
    torch.manual_seed(seed)
    tr, te = train_y >= 0, test_y >= 0
    xtr, ytr = train_x[torch.from_numpy(tr)].float(), torch.from_numpy(train_y[tr]).long()
    xte, yte = test_x[torch.from_numpy(te)].float(), torch.from_numpy(test_y[te]).long()
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    k = int(max(ytr.max(), yte.max())) + 1
    layer = torch.nn.Linear(xtr.shape[1], k)
    opt = torch.optim.AdamW(layer.parameters(), lr=3e-3, weight_decay=1e-4)
    done = 0
    while done < steps:
        for b in torch.randperm(len(xtr)).split(1024):
            opt.zero_grad()
            F.cross_entropy(layer(xtr[b]), ytr[b]).backward()
            opt.step()
            done += 1
            if done >= steps:
                break
    with torch.no_grad():
        acc = float((layer(xte).argmax(1) == yte).float().mean())
    majority = float((yte == torch.bincount(ytr, minlength=k).argmax()).float().mean())
    return {"test_acc": acc, "majority_baseline": majority, "classes": k}


def labels_for(split: str) -> dict[str, np.ndarray]:
    frame = pd.read_csv(E.SPLIT_FILES[split])
    out = {"target_item": frame.item_id.to_numpy()}
    for name, f in LABEL_FILES.items():
        mapping = np.load(LABELS / f, allow_pickle=True).item()
        out[name] = frame.item_asin.map(lambda a: mapping.get(a, -1)).fillna(-1).astype(int).to_numpy()
    return out


def ranking(result_file: Path, frame: pd.DataFrame) -> tuple[dict, np.ndarray | None]:
    if not result_file or not result_file.exists():
        return {}, None
    records = json.load(result_file.open())
    if len(records) != len(frame):
        return {"warning": f"{len(records)} predictions for {len(frame)} rows"}, None
    ranks = []
    for rec, target in zip(records, frame.target_title):
        preds = [str(x).strip(' \n"') for x in rec["predict"]]
        t = str(target).strip(' \n"')
        ranks.append(preds.index(t) + 1 if t in preds else 10 ** 9)
    ranks = np.array(ranks)
    return E.recommendation_metrics(ranks), ranks <= 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help='"name=tag[:result_json]"')
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    lab = {s: labels_for(s) for s in ("train", "test")}
    report = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        tag, _, result = rest.partition(":")
        base = E.ANALYSIS / "evidence" / E.CATEGORY / tag
        entry = {"tag": tag}
        data = {}
        for split in ("train", "test"):
            d = base / split
            if not (d / "thoughts.pt").exists():
                continue
            data[split] = {"r": torch.load(d / "thoughts.pt", weights_only=True),
                           "slot": torch.load(d / "slot.pt", weights_only=True),
                           "meta": pd.read_csv(d / "metadata.csv"),
                           "verifier": torch.load(d / "verifier.pt", weights_only=True) if (d / "verifier.pt").exists() else None}
            ids = data[split]["meta"].sample_id.to_numpy()  # extraction may cover a prefix (--limit)
            data[split]["labels"] = {k: v[ids] for k, v in lab[split].items()}
        same_slot = all(torch.equal(v["r"], v["slot"]) for v in data.values())
        entry["slot_equals_r"] = same_slot
        for comp in (("r",) if same_slot else ("r", "slot")):
            res = {}
            if "train" in data:
                res["geometry_train"] = geometry(data["train"][comp])
                res["eta2_train"] = {k: eta2(data["train"][comp], v) for k, v in data["train"]["labels"].items()}
            if "train" in data and "test" in data:
                res["probe"] = {k: probe(data["train"][comp], data["train"]["labels"][k], data["test"][comp], data["test"]["labels"][k])
                                for k in LABEL_FILES}
            entry[comp] = res
        if "test" in data:
            meta = data["test"]["meta"]
            entry["decoder_use_test"] = {"raw_nll": float(meta.raw_nll.mean()), "utility_zero_slot": ci(meta.utility),
                                         "content_utility_swap_slot": ci(meta.content_utility),
                                         "swap_changes_nll_>0.01": float((meta.content_utility.abs() > 0.01).mean())}
            rank, hit = ranking(Path(result) if result else None, meta)
            entry["ranking"] = rank
            v = data["test"]["verifier"]
            if v is not None:
                w = v["weight"].numpy()
                ver = {"w_mean": float(w.mean()), "w_quantiles": np.quantile(w, [0.1, 0.5, 0.9]).tolist(),
                       "w_gt_0.5": float((w > 0.5).mean()), "gate_mean": v["gate"].mean(0).tolist(),
                       "slot_shift": float(((data["test"]["slot"].float() - data["test"]["r"].float()).norm(dim=1)
                                            / data["test"]["r"].float().norm(dim=1).clamp_min(1e-6)).mean())}
                for j, k in enumerate(LABEL_FILES):
                    y = data["test"]["labels"][k]
                    arg = v["argmax"][:, j].numpy()
                    ver[f"verifier_acc_{k}"] = float((arg == y)[y >= 0].mean())
                    ver[f"prototype_top1_share_{k}"] = float(np.bincount(arg).max() / len(arg))
                    ver[f"prototypes_used_{k}"] = int((np.bincount(arg) > 0).sum())
                if hit is not None:
                    ver["w_mean_on_hits"] = float(w[hit].mean())
                    ver["w_mean_on_misses"] = float(w[~hit].mean())
                entry["verifier_test"] = ver
        report[name] = entry
        print(name, json.dumps(entry, default=float)[:2000], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    E.atomic_json(a.out / "comparison.json", report)
    print(f"wrote {a.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
