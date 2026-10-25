"""Queue staleness: how far a stored latent drifts from its current value.

A queue entry is r_t(x) (sample x encoded at step t) but is used at step t+D, where the
current latent would be r_{t+D}(x).  For every pair of checkpoints (t, t+D) this reports,
on a fixed set of training samples:

* raw_cos          cos(r_t(x), r_{t+D}(x)), also after centring by the fresh mean;
* user/target_cos  the same through the step-(t+D) projection heads, i.e. what InfoNCE sees;
* rank_corr        per anchor (fresh, step t+D), Spearman correlation between its similarities
                   to the other samples computed with stale vs fresh candidates (user head);
* top16_overlap    overlap of the 16 nearest candidates, stale vs fresh;
* gap              at step t+D (fresh): same-user minus random cosine through the user head,
                   the scale a stale positive must stay within to still count as positive.

Usage: python scripts/latent_drift_analysis.py --run_dir <dir with checkpoint-*> \
       --train_file <csv> --category <name> --out <json>
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "latent"))
from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel  # noqa: E402
from collaborative_contrastive_dataset import CollaborativeContrastiveDataset  # noqa: E402
from transformers import AutoTokenizer, DataCollatorForSeq2Seq  # noqa: E402

CATEGORY_TEXT = {"CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games",
                 "Musical_Instruments": "music instruments"}


def pick_rows(train, n_rows, per_user, seed):
    """Rows from users with >= 2 samples (so same-user pairs exist), up to per_user each."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in train.groupby("user_id").groups.items() if len(g) >= 2]
    rng.shuffle(groups)
    rows = []
    for g in groups:
        rows += list(rng.permutation(np.asarray(g))[:per_user])
        if len(rows) >= n_rows:
            break
    return sorted(int(i) for i in rows[:n_rows])


@torch.no_grad()
def encode(checkpoint, items, collator, batch_size):
    model = CollaborativeContrastiveLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    out = []
    for start in range(0, len(items), batch_size):
        batch = collator(items[start:start + batch_size])
        ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
        embeds = model.generate_embs(ids, mask)
        pos = model._thought_positions(ids)
        out.append(embeds[torch.arange(ids.size(0)), pos].float().cpu())
    heads = {name: {k: v.float().cpu() for k, v in getattr(model, name).state_dict().items()}
             for name in ("user_projection", "target_projection")}
    del model
    torch.cuda.empty_cache()
    return torch.cat(out), heads


def project(r, head):
    """LayerNorm -> Linear (no bias), as in the collaborative model; then L2-normalise."""
    x = F.layer_norm(r, (r.size(1),), head["0.weight"], head["0.bias"])
    return F.normalize(x @ head["1.weight"].T, dim=-1)


def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_rows", type=int, default=2000)
    parser.add_argument("--per_user", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint-*")),
                         key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    steps = [int(re.search(r"checkpoint-(\d+)", p).group(1)) for p in checkpoints]
    if len(checkpoints) < 2:
        raise SystemExit(f"need >= 2 checkpoints in {args.run_dir}, found {len(checkpoints)}")
    print(f"checkpoints: {steps}", flush=True)

    train = pd.read_csv(args.train_file)
    rows = pick_rows(train, args.n_rows, args.per_user, args.seed)
    users = train.user_id.to_numpy()[rows]
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
    data = CollaborativeContrastiveDataset(train_file=args.train_file, tokenizer=tokenizer, max_len=512,
                                           category=CATEGORY_TEXT[args.category], K=0)
    items = [{k: data[i][k] for k in ("input_ids", "attention_mask", "labels")} for i in rows]
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    print(f"{len(rows)} samples from {len(set(users))} users", flush=True)

    latents, heads = {}, {}
    for step, path in zip(steps, checkpoints):
        latents[step], heads[step] = encode(path, items, collator, args.batch_size)
        print(f"encoded checkpoint-{step}", flush=True)

    same_user = torch.as_tensor(users[:, None] == users[None, :]) & ~torch.eye(len(rows), dtype=torch.bool)
    off_diag = ~torch.eye(len(rows), dtype=torch.bool)
    per_checkpoint, pairs = {}, []
    for j in steps:
        u = project(latents[j], heads[j]["user_projection"])
        sim = u @ u.T
        per_checkpoint[j] = {"same_user_cos": sim[same_user].mean().item(),
                             "random_cos": sim[off_diag].mean().item()}
        per_checkpoint[j]["gap"] = per_checkpoint[j]["same_user_cos"] - per_checkpoint[j]["random_cos"]
        for i in [s for s in steps if s < j]:
            stale, fresh = latents[i], latents[j]
            mu = fresh.mean(0, keepdim=True)
            row = {"t": i, "t_plus_d": j, "delta": j - i,
                   "raw_cos": F.cosine_similarity(stale, fresh).mean().item(),
                   "centred_cos": F.cosine_similarity(stale - mu, fresh - mu).mean().item()}
            for name in ("user_projection", "target_projection"):
                row[name.split("_")[0] + "_cos"] = (project(stale, heads[j][name])
                                                   * project(fresh, heads[j][name])).sum(-1).mean().item()
            us = project(stale, heads[j]["user_projection"])
            s_fresh, s_stale = u @ u.T, u @ us.T
            s_fresh.fill_diagonal_(-2), s_stale.fill_diagonal_(-2)
            row["rank_corr"] = spearman_rows(s_fresh, s_stale)
            top_f, top_s = s_fresh.topk(16, 1).indices, s_stale.topk(16, 1).indices
            row["top16_overlap"] = np.mean([len(set(a.tolist()) & set(b.tolist())) / 16
                                            for a, b in zip(top_f, top_s)])
            row["gap_at_t_plus_d"] = per_checkpoint[j]["gap"]
            pairs.append(row)

    print("\n| t | t+D | D | raw cos | centred cos | user-head cos | target-head cos | rank corr | top-16 overlap | gap (same-user − random) at t+D |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(pairs, key=lambda r: (r["delta"], r["t"])):
        print(f"| {r['t']} | {r['t_plus_d']} | {r['delta']} | {r['raw_cos']:.3f} | {r['centred_cos']:.3f} | "
              f"{r['user_cos']:.3f} | {r['target_cos']:.3f} | {r['rank_corr']:.3f} | {r['top16_overlap']:.3f} | "
              f"{r['gap_at_t_plus_d']:.3f} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"run_dir": args.run_dir, "steps": steps, "n_samples": len(rows), "n_users": int(len(set(users))),
               "per_checkpoint": per_checkpoint, "pairs": pairs}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
