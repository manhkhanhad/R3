"""HiLaR stage 1: temporal-guided hierarchical quantization (document/hilar.md).

Re-implements "Temporal-Guided Hierarchical Quantization" of HiLaR (Hu et al., arXiv
2607.27760, Sec. 3.1); the paper's code is not released, so unstated details are
choices, marked CHOICE:

1. h_u = final-layer state of a frozen LLM at the last token of a time-aware prompt
   (history titles with purchase dates).  CHOICE: Qwen2.5-1.5B-Instruct (the R3 base),
   final normed layer; h_u is standardised per dimension over the training rows, because
   final-layer states are highly anisotropic (random rows have cosine ~0.98).
2. Residual quantization, K = 4 levels, 256 codes per level:
   c_k = argmin_c ||r_{k-1} - E_k(c)||^2,  r_0 = h_u,  r_k = r_{k-1} - q_k,  e_k = sum_{j<=k} q_j.
   CHOICE: per-level codebooks in the full 1536-d space (no encoder is described),
   k-means initialisation on the first batch's residuals.
3. Temporal supervision: the history is split into K chronological equal-count windows;
   level k < K reconstructs the items of windows k..K, level K the next item i+, via
   multi-label BCE with negative sampling and level-specific decoders Dec_k.
   CHOICE: Dec_k is linear over the item vocabulary; 100 uniform negatives per row;
   lambda_k = 1.  Assignments depend on h_u only, so this loss trains the codebook
   vectors and decoders: a level-K code becomes "what users with this code buy next",
   a collaborative summary, not a per-row copy of the target.
4. L_RQ = ||h_u - e_K||^2 + L_temp + L_VQ, L_VQ = ||sg(r_{k-1}) - q_k||^2 (codebook)
   + beta ||r_{k-1} - sg(q_k)||^2 (commitment; inert without an encoder).  CHOICE:
   beta = 0.25, squared errors averaged over dimensions.

Outputs (training split only; alignment targets are a training signal):
  <out>/hu.pt       {"h": [N, H] bf16 raw states}
  <out>/targets.pt  {"e": [N, K, H] bf16 cumulative code embeddings (standardised space),
                     "codes": [N, K], "valid": [N], "mean", "std", ...}
  <out>/stats.json  code usage per level, reconstruction R^2, held-out decoder HR@10.
"""

import ast
import json
import math
import os
import time

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_prompt(titles, stamps):
    lines = "\n".join(f"[{time.strftime('%Y-%m-%d', time.gmtime(int(s)))}] {t}" for t, s in zip(titles, stamps))
    return (f"A customer bought the following musical instruments and accessories, oldest first, with purchase "
            f"dates:\n{lines}\nThe customer's next purchase will be:")


@torch.no_grad()
def encode_users(frame, model, batch_size=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": time_prompt(ast.literal_eval(t), ast.literal_eval(s))}],
                                       tokenize=False, add_generation_prompt=False)
               for t, s in zip(frame.history_item_title, frame.history_timestamp)]
    ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    order = np.argsort([len(x) for x in ids])
    out = torch.zeros(len(ids), llm.config.hidden_size, dtype=torch.bfloat16)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        width = max(len(ids[i]) for i in batch)
        x = torch.full((len(batch), width), tok.pad_token_id or 0)
        m = torch.zeros(len(batch), width, dtype=torch.long)
        for j, i in enumerate(batch):
            x[j, :len(ids[i])] = torch.tensor(ids[i])
            m[j, :len(ids[i])] = 1
        last = llm.model(input_ids=x.cuda(), attention_mask=m.cuda()).last_hidden_state
        ends = m.sum(1) - 1
        out[torch.as_tensor(batch)] = last[torch.arange(len(batch)), ends.cuda()].to(torch.bfloat16).cpu()
        if start // batch_size % 300 == 0:
            print(f"encoded {start + len(batch)}/{len(ids)}", flush=True)
    return out


def window_targets(history_ids, target, k):
    """Level k < K: items of windows k..K (chronological, equal count); level K: {target}."""
    windows = np.array_split(np.asarray(history_ids), k)
    levels = [sorted({int(i) for w in windows[j:] for i in w}) for j in range(k - 1)]
    return levels + [[int(target)]]


class ResidualQuantizer(nn.Module):
    def __init__(self, dim, levels, codes, items):
        super().__init__()
        self.codebooks = nn.Parameter(torch.zeros(levels, codes, dim))
        self.decoders = nn.ModuleList(nn.Linear(dim, items) for _ in range(levels))

    @torch.no_grad()
    def kmeans_init(self, h, iters=20, seed=0):
        g = torch.Generator(device=h.device).manual_seed(seed)
        r = h.clone()
        for k in range(self.codebooks.shape[0]):
            c = r[torch.randperm(len(r), generator=g, device=h.device)[:self.codebooks.shape[1]]].clone()
            for _ in range(iters):
                a = torch.cdist(r, c).argmin(1)
                for j in range(len(c)):
                    sel = a == j
                    c[j] = r[sel].mean(0) if sel.any() else r[torch.randint(len(r), (1,), device=h.device)].squeeze(0)
            self.codebooks[k] = c
            r = r - c[torch.cdist(r, c).argmin(1)]

    def forward(self, h):
        r, e, codes, vq = h, torch.zeros_like(h), [], h.new_zeros(())
        cumulative = []
        for k in range(self.codebooks.shape[0]):
            book = self.codebooks[k]
            c = torch.cdist(r.detach(), book.detach()).argmin(1)
            q = book[c]
            vq = vq + F.mse_loss(q, r.detach()) + 0.25 * F.mse_loss(r, q.detach())
            e = e + q
            r = r - q.detach()
            codes.append(c)
            cumulative.append(e)
        return torch.stack(cumulative, 1), torch.stack(codes, 1), vq


def temporal_loss(model, e, targets, n_items, negatives, g):
    """Mean over levels of BCE on each row's positive items and `negatives` random items."""
    total = e.new_zeros(())
    levels = e.shape[1]
    for k in range(levels):
        logits = model.decoders[k](e[:, k])
        pos_rows, pos_cols = [], []
        for i, t in enumerate(targets):
            pos_rows += [i] * len(t[k])
            pos_cols += t[k]
        if not pos_rows:
            continue
        pos_rows = torch.tensor(pos_rows, device=e.device)
        pos_cols = torch.tensor(pos_cols, device=e.device)
        neg_cols = torch.randint(n_items, (len(targets), negatives), generator=g, device=e.device)
        pos = F.binary_cross_entropy_with_logits(logits[pos_rows, pos_cols], torch.ones(len(pos_rows), device=e.device))
        neg = F.binary_cross_entropy_with_logits(logits.gather(1, neg_cols), torch.zeros_like(neg_cols, dtype=logits.dtype))
        total = total + pos + neg
    return total / levels


def main(csv: str, out: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct", levels: int = 4, codes: int = 256,
         epochs: int = 20, batch_size: int = 512, lr: float = 1e-3, negatives: int = 100, holdout: float = 0.05,
         seed: int = 0, limit: int = -1, device: str = "cuda"):
    torch.manual_seed(seed)
    frame = pd.read_csv(csv, usecols=["history_item_id", "history_item_title", "history_timestamp", "item_id"])
    if limit > 0:
        frame = frame.iloc[:limit]
    os.makedirs(out, exist_ok=True)
    hu_path = os.path.join(out, "hu.pt")
    if os.path.exists(hu_path):
        h_raw = torch.load(hu_path)["h"]
    else:
        h_raw = encode_users(frame, model)
        torch.save({"h": h_raw, "model": model, "csv": os.path.abspath(csv)}, hu_path)
    h = h_raw.float()
    mean, std = h.mean(0), h.std(0).clamp_min(1e-3)
    h = ((h - mean) / std).to(device)
    n_items = int(max(frame.item_id.max(), max(max(ast.literal_eval(x)) for x in frame.history_item_id))) + 1
    targets = [window_targets(ast.literal_eval(hist), t, levels) for hist, t in zip(frame.history_item_id, frame.item_id)]

    g = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(len(h), generator=g, device=device)
    n_hold = int(holdout * len(h))
    hold, train = perm[:n_hold], perm[n_hold:]

    rq = ResidualQuantizer(h.shape[1], levels, codes, n_items).to(device)
    rq.kmeans_init(h[train[:8192]])
    opt = torch.optim.Adam(rq.parameters(), lr=lr)
    for epoch in range(epochs):
        order = train[torch.randperm(len(train), generator=g, device=device)]
        sums = np.zeros(3)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            e, _, vq = rq(h[idx])
            rec = F.mse_loss(e[:, -1], h[idx])
            temp = temporal_loss(rq, e, [targets[i] for i in idx.tolist()], n_items, negatives, g)
            loss = rec + temp + vq
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += [rec.item(), temp.item(), vq.item()]
        n = math.ceil(len(order) / batch_size)
        print(f"epoch {epoch + 1}: recon {sums[0] / n:.4f} temporal {sums[1] / n:.4f} vq {sums[2] / n:.4f}", flush=True)

    with torch.no_grad():
        e_all, codes_all = [], []
        for start in range(0, len(h), 4096):
            e, c, _ = rq(h[start:start + 4096])
            e_all.append(e.cpu())
            codes_all.append(c.cpu())
        e_all, codes_all = torch.cat(e_all), torch.cat(codes_all)
        stats = {"levels": levels, "codes": codes, "rows": len(h), "n_items": n_items}
        for k in range(levels):
            counts = torch.bincount(codes_all[:, k], minlength=codes).float()
            p = counts / counts.sum()
            stats[f"level{k + 1}_used_codes"] = int((counts > 0).sum())
            stats[f"level{k + 1}_perplexity"] = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
            resid = h.cpu() - e_all[:, k]
            stats[f"level{k + 1}_recon_r2"] = float(1 - resid.square().sum() / (h.cpu() - h.cpu().mean(0)).square().sum())
        # held-out rows: does the level-K code predict the next item better than popularity?
        logits = rq.decoders[-1](e_all[hold.cpu(), -1].to(device))
        top = logits.topk(10, dim=1).indices.cpu()
        tgt = torch.tensor([targets[i][-1][0] for i in hold.tolist()])
        stats["holdout_levelK_hr10"] = float((top == tgt[:, None]).any(1).float().mean())
        pop = torch.bincount(torch.tensor([targets[i][-1][0] for i in train.tolist()]), minlength=n_items).topk(10).indices
        stats["holdout_popularity_hr10"] = float(torch.isin(tgt, pop).float().mean())
    print(json.dumps(stats, indent=1))
    torch.save({"e": e_all.to(torch.bfloat16), "codes": codes_all, "valid": torch.ones(len(h), dtype=torch.bool),
                "mean": mean, "std": std, "levels": levels, "model": model, "csv": os.path.abspath(csv)},
               os.path.join(out, "targets.pt"))
    torch.save(rq.state_dict(), os.path.join(out, "quantizer.pt"))
    with open(os.path.join(out, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=1)


if __name__ == "__main__":
    fire.Fire(main)
