"""Gate 1 (Cornac-TIGER-faithful RQ-VAE) for the TARGET part of GEARec v2 reasoning.

Matches cornac/models/tiger (tiger.py RQVAE, recom_tiger.py _fit_rqvae) in:
  encoder/decoder  MLP in -> 512 -> 256 -> 128 -> 32 (ReLU between, none after the last), mirrored decoder
  codebooks        nn.Parameter, learned by GRADIENT: per level  mse(e, r.detach()) + beta * mse(r, e.detach())
  init             k-means++ + 10 Lloyd iterations on ALL training rows' encoder outputs, level by level
                   through the residual chain, before training
  reconstruction   mse(x_hat, x); straight-through z_q = z + (q - z).detach()
  training         AdamW lr 1e-3, weight decay 0, 200 epochs, batch 1024, constant LR, final weights kept
  dead codes       Cornac's rule: after each epoch, codes with zero uses that epoch are restarted from random
                   encoder residuals (up to 8192 rows), level by level
Kept from our gate 1 (not changed by request): input = CENTRED, L2-normalised mxbai target-part embedding;
4 levels x 256 codes, no dedup token. Same report as rqvae.py.
"""

import ast
import os
from collections import Counter

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rqvae import DATA, GEOM, ITEM_EMB, R3, STEM, fit_linear


def mlp(d_in, hidden, d_out):
    dims, layers = [d_in, *hidden, d_out], []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def kmeans_pp(x, k, n_iters=10, gen=None):
    """k-means++ seeding + Lloyd iterations; empty clusters re-seeded with random points."""
    n = len(x)
    c = x[torch.randint(n, (1,), generator=gen)].clone()
    d2 = torch.cdist(x, c).pow(2).squeeze(1)
    for _ in range(1, k):
        p = d2 / d2.sum()
        nxt = x[torch.multinomial(p, 1, generator=gen)]
        c = torch.cat([c, nxt])
        d2 = torch.minimum(d2, torch.cdist(x, nxt).pow(2).squeeze(1))
    for _ in range(n_iters):
        a = torch.cdist(x, c).argmin(1)
        newc = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        newc[~empty] /= cnt[~empty, None].float()
        if empty.any():
            newc[empty] = x[torch.randint(n, (int(empty.sum()),), generator=gen)]
        c = newc
    return c


class RQVAE(nn.Module):
    def __init__(self, d_in=1024, latent=32, levels=4, k=256, beta=0.25):
        super().__init__()
        self.enc = mlp(d_in, (512, 256, 128), latent)
        self.dec = mlp(latent, (128, 256, 512), d_in)
        self.codebooks = nn.Parameter(torch.randn(levels, k, latent) * 0.01)
        self.levels, self.k, self.beta = levels, k, beta

    def quantize(self, z):
        r, q, ids, loss = z, torch.zeros_like(z), [], 0.0
        for l in range(self.levels):
            idx = torch.cdist(r, self.codebooks[l]).argmin(1)
            e = self.codebooks[l][idx]
            loss = loss + F.mse_loss(e, r.detach()) + self.beta * F.mse_loss(r, e.detach())
            q = q + e
            r = r - e.detach()
            ids.append(idx)
        return q, torch.stack(ids, -1), loss

    def forward(self, x):
        z = self.enc(x)
        q, ids, loss_rq = self.quantize(z)
        z_q = z + (q - z).detach()
        x_hat = self.dec(z_q)
        return F.mse_loss(x_hat, x), loss_rq, ids

    @torch.no_grad()
    def encode(self, x):
        return self.quantize(self.enc(x))[1]

    @torch.no_grad()
    def decode_codes(self, codes, upto=None):
        """Decoded vector in the (standardised) input space, not normalised."""
        upto = upto or self.levels
        q = sum(self.codebooks[l][codes[:, l]] for l in range(upto))
        return self.dec(q)

    @torch.no_grad()
    def kmeans_init(self, x, gen):
        r = self.enc(x)
        for l in range(self.levels):
            c = kmeans_pp(r, self.k, 10, gen)
            self.codebooks.data[l] = c
            r = r - c[torch.cdist(r, c).argmin(1)]

    @torch.no_grad()
    def restart_dead(self, x_sample, used, gen):
        """used: [levels, k] bool of codes selected this epoch."""
        r = self.enc(x_sample)
        n = 0
        for l in range(self.levels):
            dead = ~used[l]
            if dead.any():
                pick = r[torch.randint(len(r), (int(dead.sum()),), generator=gen)]
                self.codebooks.data[l][dead] = pick
                n += int(dead.sum())
            e = self.codebooks[l][torch.cdist(r, self.codebooks[l]).argmin(1)]
            r = r - e
        return n


def train(X, levels, k, latent, epochs, seed, log):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RQVAE(d_in=X.shape[1], latent=latent, levels=levels, k=k)
    model.kmeans_init(X, gen)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        used = torch.zeros(levels, k, dtype=torch.bool)
        tot = rec_tot = 0.0
        for s in range(0, n, 1024):
            b = X[perm[s:s + 1024]]
            rec, rq, ids = model(b)
            loss = rec + rq
            opt.zero_grad(); loss.backward(); opt.step()
            used[torch.arange(levels)[None, :].expand_as(ids), ids] = True
            tot += float(loss) * len(b); rec_tot += float(rec) * len(b)
        restarted = model.restart_dead(X[torch.randperm(n, generator=gen)[:8192]], used, gen)
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                cos = float(F.cosine_similarity(model.decode_codes(model.encode(X[:8192])), X[:8192], dim=-1).mean())
            log(f"   epoch {ep:3d}: loss {tot / n:.5f}  recon mse {rec_tot / n:.6f}  cos(decoded, x) {cos:.3f}  "
                f"codes used/level {used.sum(1).tolist()}  restarted {restarted}")
    return model.eval()


def main(out=f"{R3}/output_dir/reasoning_codes/rqvae_cornac_L4_K256_d32", levels=4, k=256, latent=32, epochs=200, seed=0):
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/gate1.txt", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    P = {s: torch.load(f"{GEOM}/parts/{s}_parts.pt") for s in ("train", "valid", "test")}
    H = {s: torch.load(f"{GEOM}/mxbai/{s}.pt")["history"].float() for s in P}
    ok = {s: np.where(P[s]["has"].numpy())[0] for s in P}
    raw = {s: F.normalize(P[s]["target_part"].float(), dim=-1) for s in P}
    # per-dimension z-score on train (Cornac feature_standardize=True, as in its Paischer config): centres the
    # shared template direction AND gives MSE a usable scale (raw unit vectors give MSE ~1e-3 -> no learning)
    mu = raw["train"][ok["train"]].mean(0)
    sd = raw["train"][ok["train"]].std(0).clamp(min=1e-6)
    T = {s: (raw[s] - mu) / sd for s in raw}
    uncentre = lambda v: F.normalize(mu + sd * v, dim=-1)       # back to mxbai space for the info test
    torch.save({"mu": mu, "sd": sd}, f"{out}/standardize.pt")
    log(f"Cornac-faithful RQ-VAE on z-scored target parts: levels {levels} x {k}, latent {latent}, epochs {epochs}; "
        f"train rows {len(ok['train'])}")
    model = train(T["train"][ok["train"]], levels, k, latent, epochs, seed, log)
    torch.save(model.state_dict(), f"{out}/rqvae.pt")
    codes = {}
    for s in P:
        c = torch.full((len(T[s]), levels), -1, dtype=torch.long)
        c[ok[s]] = torch.cat([model.encode(T[s][ok[s]][i:i + 8192]) for i in range(0, len(ok[s]), 8192)])
        codes[s] = c
        torch.save(c, f"{out}/codes_{s}.pt")

    te = ok["test"]; ct = codes["test"][te]; tt = T["test"][te]; ctr = codes["train"][ok["train"]]
    log("\nFidelity and codebook health (test):")
    for l in range(1, levels + 1):
        rec = model.decode_codes(ct, l)
        log(f"   levels 1..{l}: cos(decoded, original) standardised {float(F.cosine_similarity(rec, tt, dim=-1).mean()):.3f} | "
            f"in mxbai space {float((uncentre(rec) * raw['test'][te]).sum(-1).mean()):.3f}")
    for l in range(levels):
        cnt = torch.bincount(ctr[:, l], minlength=k).float(); p = cnt / cnt.sum()
        log(f"   level {l + 1}: codes used {int((cnt > 0).sum())}/{k}, perplexity {float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())):.1f}")
    tc = Counter(tuple(r) for r in ctr.tolist())
    log(f"   train 4-code tuples: {len(tc)} distinct for {len(ctr)} rows; rows with a unique tuple "
        f"{sum(v == 1 for v in tc.values()) / len(ctr):.1%}; test tuples seen in train {np.mean([tuple(r) in tc for r in ct.tolist()]):.1%}")

    dte = pd.read_csv(f"{DATA}/test/{STEM}.csv")
    items = F.normalize(torch.load(ITEM_EMB).float(), dim=-1)
    hist_ids = [ast.literal_eval(h) for h in dte["history_item_id"]]
    target = dte["item_id"].to_numpy()

    def ranks(Q):
        S = F.normalize(Q, dim=-1) @ items.T
        out_ = []
        for r, i in enumerate(te):
            excl = [h for h in hist_ids[i] if h != target[i]]
            if excl:
                S[r, excl] = float("-inf")
            out_.append(int((S[r] > S[r, target[i]]).sum()) + 1)
        return np.array(out_)

    summ = lambda rk: f"median {np.median(rk):5.0f}  HR@10 {np.mean(rk <= 10):5.1%}  HR@100 {np.mean(rk <= 100):5.1%}"
    h = F.normalize(H["test"][te], dim=-1)
    log("\nInformation about the true next item (mxbai catalog rank, test):")
    log(f"   history alone                               {summ(ranks(h.clone()))}")
    log(f"   history + UNCOMPRESSED target part           {summ(ranks(h + raw['test'][te]))}")
    for l in range(1, levels + 1):
        log(f"   history + target part decoded from 1..{l}      {summ(ranks(h + uncentre(model.decode_codes(ct, l))))}")

    log("\nPredicting the codes from the history-TEXT embedding (leak-free), test top-1 / top-5:")
    Xtr = F.normalize(H["train"][ok["train"]], dim=-1)
    maj = torch.bincount(ctr[:, 0], minlength=k).argmax()
    log(f"   level 1 majority code: top-1 {float((ct[:, 0] == maj).float().mean()):.1%}")
    for l in range(levels):
        A = torch.cat([Xtr] + [F.one_hot(ctr[:, j], k).float() for j in range(l)], -1)
        B = torch.cat([h] + [F.one_hot(ct[:, j], k).float() for j in range(l)], -1)
        m = fit_linear(A, ctr[:, l].numpy(), k, seed=seed)
        with torch.no_grad():
            top = m(B).topk(5, -1).indices
        cond = "" if l == 0 else f" given TRUE levels 1..{l}"
        log(f"   level {l + 1}{cond:24s} top-1 {float((top[:, 0] == ct[:, l]).float().mean()):5.1%}  "
            f"top-5 {float((top == ct[:, l:l + 1]).any(-1).float().mean()):5.1%}")
    logf.close()


if __name__ == "__main__":
    fire.Fire(main)
