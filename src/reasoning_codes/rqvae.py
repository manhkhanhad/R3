"""Gate 1: residual VQ-VAE (RQ-VAE) codes for the TARGET part of GEARec v2 reasoning.

Input: mxbai embeddings of each training row's target part ("what they want next" sentences;
src/reasoning_geometry/split_parts.py + embed_parts.py). Model: MLP encoder -> L residual codebooks of K
codes (EMA updates, k-means init, dead-code reset) -> MLP decoder; loss = cosine reconstruction +
commitment. Trained on TRAIN rows only.

Reports on TEST rows:
  fidelity   cosine(decoded, original target part) using levels 1..l
  health     per-level codebook usage / perplexity; share of rows whose 4-code tuple is unique
  info       rank of the true next item (mxbai catalog, history masked) from history + decoded target part
             at levels 1..l, vs history alone and vs the uncompressed target part
  predict    linear classifier on the history-TEXT embedding (leak-free) -> level-1 code; and level l given
             the TRUE codes of levels < l (teacher-forced coarse-to-fine), top-1 / top-5
Writes codes for all splits to <out>/codes_<split>.pt ([N, L] long, -1 where no reasoning) + the model.
"""

import ast
import json
import os

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
GEOM = R3 + "/output_dir/reasoning_geometry"
ITEM_EMB = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/outputs/item_emb_mxbai.pt"


class RQVAE(nn.Module):
    def __init__(self, d_in=1024, d_lat=128, levels=4, k=256, decay=0.99, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, 512), nn.GELU(), nn.Linear(512, d_in))
        self.levels, self.k, self.decay, self.beta = levels, k, decay, beta
        self.register_buffer("codebooks", torch.randn(levels, k, d_lat) * 0.01)
        self.register_buffer("ema_count", torch.ones(levels, k))
        self.register_buffer("ema_sum", self.codebooks.clone())
        self.register_buffer("initialised", torch.zeros(levels, dtype=torch.bool))

    def quantize(self, z, update=False):
        residual, quant, codes = z, torch.zeros_like(z), []
        for l in range(self.levels):
            cb = self.codebooks[l]
            if update and not self.initialised[l]:
                self.codebooks[l] = kmeans_init(residual.detach(), self.k)
                self.ema_sum[l] = self.codebooks[l].clone()
                self.ema_count[l] = 1.0
                self.initialised[l] = True
                cb = self.codebooks[l]
            d = (residual.pow(2).sum(-1, keepdim=True) - 2 * residual @ cb.T + cb.pow(2).sum(-1)[None])
            idx = d.argmin(-1)
            q = cb[idx]
            if update:
                with torch.no_grad():
                    onehot = F.one_hot(idx, self.k).float()
                    self.ema_count[l].mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                    self.ema_sum[l].mul_(self.decay).add_(onehot.T @ residual.detach(), alpha=1 - self.decay)
                    n = self.ema_count[l].sum()
                    count = (self.ema_count[l] + 1e-5) / (n + self.k * 1e-5) * n
                    self.codebooks[l] = self.ema_sum[l] / count[:, None]
                    # reset rarely used codes (EMA usage < ~1 per 8 batches) to random current residuals
                    dead = self.ema_count[l] < 0.125
                    if dead.any():
                        pick = residual.detach()[torch.randint(len(residual), (int(dead.sum()),))]
                        self.codebooks[l][dead] = pick
                        self.ema_sum[l][dead] = pick
                        self.ema_count[l][dead] = 1.0
            quant = quant + q
            residual = residual - q.detach() if not update else residual - q.detach()
            codes.append(idx)
        return quant, torch.stack(codes, -1)

    def decode_codes(self, codes, upto=None):
        upto = upto or self.levels
        q = sum(self.codebooks[l][codes[:, l]] for l in range(upto))
        return F.normalize(self.dec(q), dim=-1)

    def forward(self, x, update=False):
        z = self.enc(x)
        q, codes = self.quantize(z, update)
        commit = F.mse_loss(z, q.detach())
        q_st = z + (q - z).detach()                       # straight-through
        xr = F.normalize(self.dec(q_st), dim=-1)
        recon = (1 - (xr * x).sum(-1)).mean()
        return recon + self.beta * commit, recon, codes


def kmeans_init(x, k, iters=15):
    c = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def train_rqvae(X, levels, k, epochs, seed, log):
    torch.manual_seed(seed)
    model = RQVAE(levels=levels, k=k)
    opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec.parameters()), lr=1e-3)
    n = len(X)
    # k-means init of every level on a large first batch
    with torch.no_grad():
        model.quantize(model.enc(X[torch.randperm(n)[:16384]]), update=True)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, 1024):
            b = X[perm[s:s + 1024]]
            loss, recon, _ = model(b, update=True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(recon) * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"   epoch {ep:3d}: recon (1 - cos) {tot / n:.4f}")
    return model.eval()


def fit_linear(Xtr, ytr, n_classes, epochs=40, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr)); hold, fit = idx[: len(idx) // 10], idx[len(idx) // 10:]
    m = nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.as_tensor(ytr)
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        rng.shuffle(fit)
        for s in range(0, len(fit), 1024):
            b = fit[s:s + 1024]
            loss = F.cross_entropy(m(Xtr[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            hl = float(F.cross_entropy(m(Xtr[hold]), y[hold]))
        if hl < best - 1e-4:
            best, state, bad = hl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    m.load_state_dict(state)
    return m.eval()


def main(out=f"{R3}/output_dir/reasoning_codes/rqvae_L4_K256", levels=4, k=256, epochs=60, seed=0):
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/gate1.txt", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    P = {s: torch.load(f"{GEOM}/parts/{s}_parts.pt") for s in ("train", "valid", "test")}
    H = {s: torch.load(f"{GEOM}/mxbai/{s}.pt")["history"].float() for s in ("train", "valid", "test")}
    ok = {s: np.where(P[s]["has"].numpy())[0] for s in P}
    raw = {s: F.normalize(P[s]["target_part"].float(), dim=-1) for s in P}
    # centre: all target parts share a large template direction (random-pair cos ~0.78); quantize what differs
    mu = raw["train"][ok["train"]].mean(0)
    scale = float((raw["train"][ok["train"]] - mu).norm(dim=-1).mean())
    T = {s: F.normalize(raw[s] - mu, dim=-1) for s in raw}
    uncentre = lambda v: F.normalize(mu + scale * v, dim=-1)   # back to mxbai space for the info test
    torch.save({"mu": mu, "scale": scale}, f"{out}/centering.pt")
    log(f"RQ-VAE on CENTRED target-part embeddings: levels {levels} x {k} codes; train rows {len(ok['train'])}; "
        f"mean residual norm {scale:.3f}")
    model = train_rqvae(T["train"][ok["train"]], levels, k, epochs, seed, log)
    torch.save(model.state_dict(), f"{out}/rqvae.pt")

    codes = {}
    with torch.no_grad():
        for s in P:
            c = torch.full((len(T[s]), levels), -1, dtype=torch.long)
            _, cc = model.quantize(model.enc(T[s][ok[s]]))
            c[ok[s]] = cc
            codes[s] = c
            torch.save(c, f"{out}/codes_{s}.pt")

    te = ok["test"]; ct = codes["test"][te]; tt = T["test"][te]
    # ---- fidelity + health
    log("\nFidelity and codebook health (test):")
    with torch.no_grad():
        for l in range(1, levels + 1):
            rec = model.decode_codes(ct, l)
            log(f"   levels 1..{l}: cos(decoded, original) centred {float((rec * tt).sum(-1).mean()):.3f} | "
                f"in mxbai space {float((uncentre(rec) * raw['test'][te]).sum(-1).mean()):.3f}")
    ctr = codes["train"][ok["train"]]
    for l in range(levels):
        cnt = torch.bincount(ctr[:, l], minlength=k).float()
        p = cnt / cnt.sum()
        perp = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
        log(f"   level {l + 1}: codes used {int((cnt > 0).sum())}/{k}, perplexity {perp:.1f}")
    tuples = [tuple(r) for r in ctr.tolist()]
    from collections import Counter
    tc = Counter(tuples)
    log(f"   train 4-code tuples: {len(tc)} distinct for {len(tuples)} rows; rows with a unique tuple {sum(v == 1 for v in tc.values()) / len(tuples):.1%}; "
        f"test tuples seen in train {np.mean([tuple(r) in tc for r in ct.tolist()]):.1%}")

    # ---- info, coarse to fine
    dte = pd.read_csv(f"{DATA}/test/{STEM}.csv")
    items = F.normalize(torch.load(ITEM_EMB).float(), dim=-1)
    hist_ids = [ast.literal_eval(h) for h in dte["history_item_id"]]
    target = dte["item_id"].to_numpy()

    def ranks(Q):
        S = F.normalize(Q, dim=-1) @ items.T
        out = []
        for r, i in enumerate(te):
            excl = [h for h in hist_ids[i] if h != target[i]]
            if excl:
                S[r, excl] = float("-inf")
            out.append(int((S[r] > S[r, target[i]]).sum()) + 1)
        return np.array(out)

    def summ(rk):
        return f"median {np.median(rk):5.0f}  HR@10 {np.mean(rk <= 10):5.1%}  HR@100 {np.mean(rk <= 100):5.1%}"

    h = F.normalize(H["test"][te], dim=-1)
    log("\nInformation about the true next item (mxbai catalog rank, test):")
    log(f"   history alone                               {summ(ranks(h.clone()))}")
    log(f"   history + UNCOMPRESSED target part           {summ(ranks(h + raw['test'][te]))}")
    with torch.no_grad():
        for l in range(1, levels + 1):
            log(f"   history + target part decoded from 1..{l}      {summ(ranks(h + uncentre(model.decode_codes(ct, l))))}")

    # ---- predictability, coarse to fine (leak-free input: history text embedding)
    log("\nPredicting the codes from the history-TEXT embedding (leak-free), test top-1 / top-5:")
    Xtr = F.normalize(H["train"][ok["train"]], dim=-1); Xte = h
    maj = torch.bincount(ctr[:, 0], minlength=k).argmax()
    log(f"   level 1 majority code: top-1 {float((ct[:, 0] == maj).float().mean()):.1%}")
    for l in range(levels):
        prev_tr = [F.one_hot(ctr[:, j], k).float() for j in range(l)]
        prev_te = [F.one_hot(ct[:, j], k).float() for j in range(l)]
        A = torch.cat([Xtr] + prev_tr, -1); B = torch.cat([Xte] + prev_te, -1)
        m = fit_linear(A, ctr[:, l].numpy(), k, seed=seed)
        with torch.no_grad():
            top = m(B).topk(5, -1).indices
        t1 = float((top[:, 0] == ct[:, l]).float().mean()); t5 = float((top == ct[:, l:l + 1]).any(-1).float().mean())
        cond = "" if l == 0 else f" given TRUE levels 1..{l}"
        log(f"   level {l + 1}{cond:24s} top-1 {t1:5.1%}  top-5 {t5:5.1%}")
    logf.close()


if __name__ == "__main__":
    fire.Fire(main)
