"""MW-R step 1: reference-model predictability of each row's target.

Scores NLL_ref(y | x) for every row with a *frozen, untrained-on-these-rows* reference LM (the training base
model, e.g. Qwen/Qwen2.5-1.5B): the exact R3 prompt (instruction + history + response header) with the
<|Thought|> slot removed, target = the R3 label tokens.  Low NLL_ref = the history already explains the target
(e.g. same-day near-copies), high NLL_ref = the target carries information the history lacks.

Writes <out_dir>/<split>.pt  {"row_id", "nll_mean", "nll_sum", "n_tokens", "is_copy"} and prints the AUROC of
-nll for the copy rule (the rule is used for this diagnostic only; MW-R never sees it).
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_attention_train_grounded_margin import CATEGORY_TEXT
from motive_contrastive import MotiveDataset


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count 1/2 (rank formulation)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = score.argsort()
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def score(base_model: str = "Qwen/Qwen2.5-1.5B", file: str = "", split: str = "train", out_dir: str = "",
          category: str = "Toys_and_Games", batch_size: int = 16, cutoff_len: int = 512, sample: int = -1,
          latent_model: bool = False):
    """latent_model=True: base_model is a trained R3-family latent checkpoint (e.g. MW-3b, whose predictions ignore
    the latent content); the <|Thought|> slot is kept and filled by its own pooled latent (LatentModel eval forward)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.pt")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    special = list(tokenizer.additional_special_tokens)
    if "<|Thought|>" not in special:
        special.append("<|Thought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    thought_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    data = MotiveDataset(train_file=file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=0,
                         category=CATEGORY_TEXT[category], K=0)
    if latent_model:
        from motive_contrastive import MotiveMarginLatentModel
        model = MotiveMarginLatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16).cuda().eval()

    rows = []
    for i in range(len(data)):
        e = data[i]
        keep = [latent_model or t != thought_id for t in e["input_ids"]]
        ids = [t for t, k in zip(e["input_ids"], keep) if k]
        labels = [l for l, k in zip(e["labels"], keep) if k]
        rows.append((e["row_id"], e["is_copy"], ids, labels))
    print(f"{split}: {len(rows)} rows; thought removed from {sum(1 for r in rows)} rows", flush=True)

    row_ids, nll_mean, nll_sum, n_tok, is_copy = [], [], [], [], []
    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        L = max(len(r[2]) for r in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (_, _, x, y) in enumerate(chunk):           # right padding
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        shift_logits, shift_labels = logits[:, :-1], lab[:, 1:].cuda()
        tok_nll = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                                    shift_labels.reshape(-1), ignore_index=-100,
                                                    reduction="none").view(shift_labels.shape)
        mask = (shift_labels != -100).float()
        s, n = (tok_nll * mask).sum(1), mask.sum(1)
        for j, (rid, c, _, _) in enumerate(chunk):
            row_ids.append(rid); is_copy.append(c)
            nll_sum.append(float(s[j])); n_tok.append(int(n[j])); nll_mean.append(float(s[j] / n[j].clamp(min=1)))
        if (b // batch_size) % 200 == 0:
            print(f"  {b + len(chunk)}/{len(rows)}", flush=True)

    res = {"row_id": torch.tensor(row_ids), "nll_mean": torch.tensor(nll_mean), "nll_sum": torch.tensor(nll_sum),
           "n_tokens": torch.tensor(n_tok), "is_copy": torch.tensor(is_copy), "base_model": base_model,
           "latent_model": latent_model}
    torch.save(res, out_path)
    c = np.array(is_copy, bool)
    m, s_ = np.array(nll_mean), np.array(nll_sum)
    print(f"[{split}] rows {len(c)}, copy rate {c.mean():.3f}", flush=True)
    print(f"[{split}] nll_mean copy {m[c].mean():.3f} (median {np.median(m[c]):.3f}) vs non-copy {m[~c].mean():.3f} "
          f"(median {np.median(m[~c]):.3f}); AUROC(-nll_mean -> copy) {auroc(-m, c):.3f}", flush=True)
    print(f"[{split}] nll_sum  copy {s_[c].mean():.2f} vs non-copy {s_[~c].mean():.2f}; AUROC(-nll_sum -> copy) {auroc(-s_, c):.3f}",
          flush=True)
    for q in (0.1, 0.2, 0.3, 0.5):
        thr = np.quantile(m, q)
        low = m <= thr
        print(f"[{split}] lowest {int(q * 100)}% nll_mean: copy share {c[low].mean():.3f}, captures {c[low].sum() / c.sum():.3f} of copy rows",
              flush=True)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(score)
