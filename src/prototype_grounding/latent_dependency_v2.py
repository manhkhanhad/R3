#!/usr/bin/env python3
"""Latent dependency test v2 (intervention v2): per-sample, forward-only.

For each test sample x (agreed with the user 2026-11-05):
  1. y_hat = the model's own top-1 recommendation with its own latent r
     (read from a ranking file of the same checkpoint: eval_raw.json /
     final_result.json, or an experiment.py raw generation file).
  2. Teacher-force y_hat with r replaced by K latents drawn at random from a bank
     of latents the same checkpoint produced before (experiment.py extract
     --split train; rows of the same user excluded), plus the zero vector.
  3. Per sample: D_k = log p(y_hat | x, r) - log p(y_hat | x, r_k) (summed over
     the title tokens), KL_k = sum over title tokens of KL(p(.|x, r) || p(.|x, r_k)),
     the share of title positions whose argmax changes, and own_rank = 1 + number
     of donors under which y_hat is at least as likely as under the own r.

Caveat: y_hat is selected with the own r, so D is biased upward a little even for
a model that ignores r; compare against plain R3 as the null reference.
No gradients, no generation: (K + 2) teacher-forced forward passes per sample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as E  # noqa: E402


def load_predictions(path: Path) -> list[str]:
    obj = json.load(path.open())
    if isinstance(obj, dict) and "records" in obj:  # experiment.py generation file
        return [str(r["prediction"][0]).strip(' \n"') for r in obj["records"]]
    return [str(r["predict"][0]).strip(' \n"') for r in obj]  # latent_attention_eval output


def load_bank(tag: str, checkpoint: Path, verify: bool) -> tuple[torch.Tensor, np.ndarray]:
    base = E.ANALYSIS / f"evidence/{E.CATEGORY}" / tag / "train"
    summary = json.load((base / "summary.json").open())
    if verify:
        want = summary.get("checkpoint_sha256")
        got = E.sha256(checkpoint / "model.safetensors")
        if want != got:
            raise RuntimeError(f"bank {base} was extracted from another checkpoint ({want[:12]} != {got[:12]})")
    vectors, meta = E.load_evidence("train", tag)
    return vectors, meta.user_id.astype(str).to_numpy()


def donors(sample_id: int, user: str, bank_users: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, sample_id])
    picked: list[int] = []
    while len(picked) < k:
        for i in rng.choice(len(bank_users), size=4 * k, replace=False):
            if bank_users[i] != user and i not in picked:
                picked.append(int(i))
                if len(picked) == k:
                    break
    return np.array(picked)


def strip_edges(tokenizer, ids: list[int]) -> list[int]:
    ids = list(ids)
    while ids and ids[0] == tokenizer.bos_token_id:
        ids.pop(0)
    while ids and ids[-1] == tokenizer.eos_token_id:
        ids.pop()
    return ids


def score_rows(model, tokenizer, device, rows, preds, bank, bank_users, k, seed):
    """Returns one dict per row; variants = [own, zero, donor_1..donor_K]."""
    items = []
    for row, y in zip(rows, preds):
        prompt = row["input_ids"][:row["prompt_len"]]
        target = strip_edges(tokenizer, tokenizer.encode(f'"{y}"\n')) + [tokenizer.eos_token_id]
        items.append({**row, "input_ids": prompt + target, "prompt_len": len(prompt)})
    batch = E.collate(items, tokenizer.pad_token_id)
    ids, mask, labels, pos = [batch[x].to(device) for x in ("input_ids", "attention_mask", "labels", "thought_pos")]
    own, embeds = E.thought_embeds(model, ids, mask, pos)  # own r (history-only pooling mask if configured)
    n, v = ids.size(0), k + 2
    picks = [donors(r["sample_id"], r["user_id"], bank_users, k, seed) for r in rows]
    repl = torch.stack([torch.cat([own[i:i + 1], torch.zeros_like(own[i:i + 1]), bank[picks[i]].to(device)])
                        for i in range(n)])  # [n, v, d]
    rep = lambda t: t.repeat_interleave(v, 0)
    all_embeds = E.substitute(rep(embeds), rep(pos), repl.flatten(0, 1))
    extra = {"latent_bottleneck_ids": rep(ids)} if getattr(model.config, "latent_bottleneck", None) else {}
    with torch.no_grad():
        logits = model(inputs_embeds=all_embeds, attention_mask=rep(mask), use_cache=False, **extra).logits
    shifted = labels[:, 1:]
    out = []
    for i in range(n):
        t = shifted[i].ne(-100).nonzero().flatten()  # logit positions predicting title tokens
        lg = logits[i * v:(i + 1) * v, t].float().log_softmax(-1)  # [v, L, V]
        tok = shifted[i, t]
        lp = lg.gather(-1, tok[None, :, None].expand(v, -1, 1)).squeeze(-1).sum(1)  # [v]
        p_own = lg[0].exp()
        kl = (p_own[None] * (lg[0][None] - lg[1:])).sum(-1).sum(-1)  # [v-1]: zero, donors
        arg = lg.argmax(-1)
        change = (arg[1:] != arg[:1]).float().mean(1)
        d = (lp[0] - lp[2:]).cpu().numpy()
        out.append({
            "n_tokens": int(t.numel()), "logp_own": float(lp[0]), "logp_zero": float(lp[1]),
            "logp_donor_mean": float(lp[2:].mean()),
            "D_mean": float(d.mean()), "D_std": float(d.std(ddof=1)), "D_min": float(d.min()), "D_max": float(d.max()),
            "D_zero": float(lp[0] - lp[1]), "KL_mean": float(kl[1:].mean()), "KL_zero": float(kl[0]),
            "argmax_change_mean": float(change[1:].mean()), "argmax_change_zero": float(change[0]),
            "own_rank": int(1 + (lp[2:] >= lp[0]).sum()),
        })
    del logits
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True, help="evidence tag of the train-latent bank (experiment.py extract --tag)")
    p.add_argument("--predictions", type=Path, required=True, help="ranking file of the same checkpoint (top-1 = y_hat)")
    p.add_argument("--category", default="Toys_and_Games")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows-per-batch", type=int, default=2)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--shuffle-file", type=Path, default=None,
                   help="optional experiment.py shuffle_user generation file: checks D against hits it breaks")
    p.add_argument("--no-verify-bank", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()

    E.configure_category(a.category)
    model, tokenizer, device = E.load_model(a.checkpoint)
    bank, bank_users = load_bank(a.tag, a.checkpoint, not a.no_verify_bank)
    ds = E.PromptDataset("test", tokenizer, a.limit)
    preds = load_predictions(a.predictions)
    if a.limit > 0:
        preds = preds[:a.limit]
    if len(preds) != len(ds):
        raise RuntimeError(f"{len(preds)} predictions for {len(ds)} test rows")
    out_dir = a.output or (E.ANALYSIS / "results/intervention_v2" / a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, start = [], time.time()
    for s in range(0, len(ds), a.rows_per_batch):
        rows, ys = ds.rows[s:s + a.rows_per_batch], preds[s:s + a.rows_per_batch]
        for row, y, sc in zip(rows, ys, score_rows(model, tokenizer, device, rows, ys, bank, bank_users, a.k, a.seed)):
            records.append({"sample_id": row["sample_id"], "user_id": row["user_id"], "target": row["target_title"],
                            "y_hat": y, "hit_at_1": y == row["target_title"].strip(' \n"'), **sc})
        if s % (a.rows_per_batch * 250) == 0:
            print(f"v2: {min(s + a.rows_per_batch, len(ds))}/{len(ds)} ({time.time() - start:.0f}s)", flush=True)
    df = pd.DataFrame(records)

    summary = {"checkpoint": str(a.checkpoint), "tag": a.tag, "predictions": str(a.predictions), "k": a.k,
               "seed": a.seed, "n": len(df), "elapsed_seconds": time.time() - start}
    for col in ("D_mean", "D_zero", "KL_mean", "KL_zero", "argmax_change_mean", "argmax_change_zero"):
        x = df[col].to_numpy()
        summary[col] = {"mean": float(x.mean()), "median": float(np.median(x)),
                        "p90": float(np.quantile(x, .9)), "share_gt_1": float((x > 1).mean())}
    summary["own_rank_1_share"] = float((df.own_rank == 1).mean())  # null (exchangeable): 1/(K+1)
    summary["own_percentile_mean"] = float(((a.k + 1 - df.own_rank) / a.k).mean())  # null ~0.5
    summary["D_mean_by_hit_at_1"] = {str(h): float(g.D_mean.mean()) for h, g in df.groupby("hit_at_1")}
    if a.shuffle_file:
        sh = json.load(a.shuffle_file.open())["records"]
        if len(sh) >= len(df):
            sh_top1 = np.array([str(r["prediction"][0]).strip(' \n"') for r in sh[:len(df)]])
            changed = sh_top1 != df.y_hat.to_numpy()
            summary["shuffle_top1_changed_share"] = float(changed.mean())
            from sklearn.metrics import roc_auc_score
            if 0 < changed.sum() < len(changed):
                summary["auroc_D_mean_for_shuffle_top1_change"] = float(roc_auc_score(changed, df.D_mean))
    df.to_csv(out_dir / "per_sample.csv", index=False)
    E.atomic_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
