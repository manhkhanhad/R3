"""History-swap gap per row: gap_i = NLL(y_i | x_i, r_j) - NLL(y_i | x_i, r_i), split by copy / non-copy rows.

Same quantity the history-swap hinge charges, [m - gap_i]+ (m = 0.2), and the same per-token mean NLL as training
(MarginLatentModel._per_example_nll).  r_j is the latent of a random OTHER row of the same split (fixed derangement,
seed 0).  Pass 1 pools every row's latent; pass 2 decodes each row with its own latent and with r_j.  bs = 1 (no
padding effects).  Writes <out_dir>/<tag>_<split>.npz (per-row pos/neg/gap/is_copy) and appends a summary to
<out_dir>/summary.jsonl.
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from latent_attention_train_grounded_margin import CATEGORY_TEXT
from layers import LatentModel
from margin_loss_attention import MarginLatentModel
from motive_contrastive import MotiveDataset


def summarize(gap, pos, neg, c, m=0.2):
    out = {}
    for name, sel in (("all", np.ones_like(c, bool)), ("copy", c), ("noncopy", ~c)):
        g = gap[sel]
        out[name] = {"n": int(sel.sum()), "pos_nll": float(pos[sel].mean()), "neg_nll": float(neg[sel].mean()),
                     "gap_mean": float(g.mean()), "gap_median": float(np.median(g)),
                     "gap_q10_25_75_90": [float(np.quantile(g, q)) for q in (0.1, 0.25, 0.75, 0.9)],
                     "frac_gap_ge_m": float((g >= m).mean()), "frac_gap_le_0": float((g <= 0).mean()),
                     "hinge_mean": float(np.maximum(0.0, m - g).mean())}
    return out


@torch.no_grad()
def run(model_dir: str = "", tag: str = "", file: str = "", split: str = "test", out_dir: str = "",
        category: str = "Toys_and_Games", sample: int = -1, seed: int = 0, cutoff_len: int = 512):
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{tag}_{split}.npz")
    if os.path.exists(npz):
        print(f"skip {tag} {split} (done)")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if "<|Thought|>" not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(tokenizer.additional_special_tokens) + ["<|Thought|>"]})
    data = MotiveDataset(train_file=file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                         category=CATEGORY_TEXT[category], K=0)
    model = LatentModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
    thought_id = model.model.embed_tokens.num_embeddings - 1
    n = len(data)
    rows = [data[i] for i in range(n)]
    # pass 1: every row's latent
    lat = []
    for e in rows:
        ids = torch.tensor([e["input_ids"]], device="cuda")
        emb = model.generate_embs(ids, torch.ones_like(ids))
        t = int((ids[0] == thought_id).nonzero()[0])
        lat.append(emb[0, t].float().cpu())
    lat = torch.stack(lat)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    partner[order] = np.roll(order, 1)           # derangement: partner[i] != i
    # pass 2: own vs swapped latent
    pos, neg, cp = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i, e in enumerate(rows):
        ids = torch.tensor([e["input_ids"]], device="cuda")
        lab = torch.tensor([e["labels"]], device="cuda")
        emb = model.generate_embs(ids, torch.ones_like(ids))
        t = int((ids[0] == thought_id).nonzero()[0])
        out_pos = Qwen2ForCausalLM.forward(model, inputs_embeds=emb, use_cache=False, return_dict=True)
        emb_neg = emb.clone()
        emb_neg[0, t] = lat[partner[i]].to(emb.dtype).cuda()
        out_neg = Qwen2ForCausalLM.forward(model, inputs_embeds=emb_neg, use_cache=False, return_dict=True)
        pos[i] = float(MarginLatentModel._per_example_nll(out_pos.logits, lab)[0])
        neg[i] = float(MarginLatentModel._per_example_nll(out_neg.logits, lab)[0])
        cp[i] = bool(e["is_copy"])
        if i % 1000 == 0:
            print(f"  {tag} {split} {i}/{n}", flush=True)
    gap = neg - pos
    np.savez(npz, pos=pos, neg=neg, gap=gap, is_copy=cp, partner=partner)
    s = summarize(gap, pos, neg, cp)
    with open(os.path.join(out_dir, "summary.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, "split": split, "model_dir": model_dir, **s}) + "\n")
    for k in ("copy", "noncopy"):
        v = s[k]
        print(f"[{tag} {split} {k}] n {v['n']} pos {v['pos_nll']:.3f} neg {v['neg_nll']:.3f} gap mean {v['gap_mean']:+.3f} "
              f"median {v['gap_median']:+.3f} q10/25/75/90 {[round(x, 3) for x in v['gap_q10_25_75_90']]} "
              f"frac>=0.2 {v['frac_gap_ge_m']:.3f} frac<=0 {v['frac_gap_le_0']:.3f}", flush=True)


if __name__ == "__main__":
    fire.Fire(run)
