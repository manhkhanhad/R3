"""Is LaRec's step alignment learnable from the history? (document/latent_distillation.md)

The student's latent steps see only the history; the teacher states h_k come from a
prompt that contains the target.  This fits a ridge map from the frozen backbone's
history-only state (final layer, last token of the R3 prompt) to each step's
standardised teacher state, and reports on held-out rows the InfoNCE (cosine, tau, the
same number of random negatives as training) and top-1 retrieval of the right row.
Chance InfoNCE is log(negatives + 1).

Usage: python src/latent_distill/alignment_probe.py --states <train_states.pt> --csv <train csv> --out <json>
"""

import json
import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "latent"))

import fire
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_dataset import LatentRDataset


@torch.no_grad()
def main(states: str, csv: str, out: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct", train_rows: int = 5000,
         test_rows: int = 1000, negatives: int = 255, tau: float = 0.1, ridge: float = 10.0, seed: int = 0):
    spec = torch.load(states, weights_only=False)
    valid = spec["valid"].bool()
    teacher = spec["states"].float()
    mean, std = teacher[valid].mean(0), teacher[valid].std(0).clamp_min(1e-3)
    teacher = (teacher - mean) / std
    g = torch.Generator().manual_seed(seed)
    rows = torch.nonzero(valid).squeeze(1)
    rows = rows[torch.randperm(len(rows), generator=g)[:train_rows + test_rows]]

    tok = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
    data = LatentRDataset(train_file=csv, tokenizer=tok, max_len=1024, sample=-1, category="music instruments", test=True)
    feats = []
    for i in rows.tolist():
        ids = data[i]["input_ids"][:-1]  # the R3 prompt without the thought token
        h = llm.model(input_ids=torch.tensor([ids]).cuda()).last_hidden_state[0, -1].float().cpu()
        feats.append(h)
    x = torch.stack(feats)
    x = (x - x[:train_rows].mean(0)) / x[:train_rows].std(0).clamp_min(1e-3)
    xtr, xte = x[:train_rows], x[train_rows:]
    a = xtr.T @ xtr + ridge * torch.eye(x.shape[1])

    result = {"rows": len(rows), "negatives": negatives, "tau": tau, "chance_infonce": math.log(negatives + 1),
              "chance_top1": 1 / (negatives + 1), "steps": []}
    for k in range(teacher.shape[1]):
        y = teacher[rows, k]
        w = torch.linalg.solve(a, xtr.T @ y[:train_rows])
        pred = F.normalize(xte @ w, dim=-1)
        tgt = F.normalize(y[train_rows:], dim=-1)
        losses, top1 = [], []
        for i in range(len(pred)):
            neg = torch.randperm(len(tgt), generator=g)[:negatives + 1]
            neg = neg[neg != i][:negatives]
            cand = torch.cat([tgt[i:i + 1], tgt[neg]])
            logits = cand @ pred[i] / tau
            losses.append(float(-logits.log_softmax(0)[0]))
            top1.append(float(logits.argmax() == 0))
        result["steps"].append({"step": k + 1, "infonce": sum(losses) / len(losses), "top1": sum(top1) / len(top1),
                                "cos_pred_target": float((pred * tgt).sum(-1).mean())})
        print(result["steps"][-1], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    fire.Fire(main)
