"""Fair predictability comparison between codebooks: bits of each code explained by the history (probe).

For level l: H(code_l | true earlier codes) is estimated with a probe that sees ONLY the true earlier codes
(empirical prior), and with a probe that also sees the history-text embedding; the difference in test
cross-entropy (bits) is how much the history tells about the code. Also reports the code entropy.
This is still a frozen linear PROBE, not the recommender.
"""

import fire
import numpy as np
import torch
import torch.nn.functional as F

from rqvae import GEOM, R3, fit_linear


def ce_bits(model, X, y):
    with torch.no_grad():
        return float(F.cross_entropy(model(X), torch.as_tensor(y))) / np.log(2)


def main(runs=("rqvae_L4_K256", "rqvae_cornac_L4_K256_d32"), seed=0):
    runs = list(runs) if isinstance(runs, (list, tuple)) else [runs]
    H = {s: torch.load(f"{GEOM}/mxbai/{s}.pt")["history"].float() for s in ("train", "test")}
    for run in runs:
        d = f"{R3}/output_dir/reasoning_codes/{run}"
        C = {s: torch.load(f"{d}/codes_{s}.pt") for s in ("train", "test")}
        ok = {s: np.where(C[s][:, 0] >= 0)[0] for s in C}
        ctr, cte = C["train"][ok["train"]], C["test"][ok["test"]]
        Xtr, Xte = F.normalize(H["train"][ok["train"]], dim=-1), F.normalize(H["test"][ok["test"]], dim=-1)
        k = int(max(ctr.max(), cte.max())) + 1
        print(f"\n{run}:  level | code entropy (bits) | CE without history | CE with history | bits explained by history")
        for l in range(ctr.shape[1]):
            p = torch.bincount(ctr[:, l], minlength=k).float(); p = p / p.sum()
            ent = float(-(p[p > 0] * p[p > 0].log2()).sum())
            prev_tr = [F.one_hot(ctr[:, j], k).float() for j in range(l)]
            prev_te = [F.one_hot(cte[:, j], k).float() for j in range(l)]
            if l == 0:   # prior only = marginal distribution (smoothed)
                ps = (torch.bincount(ctr[:, 0], minlength=k).float() + 0.5); ps = ps / ps.sum()
                ce0 = float(-ps[cte[:, 0]].log2().mean())
            else:
                m0 = fit_linear(torch.cat(prev_tr, -1), ctr[:, l].numpy(), k, seed=seed)
                ce0 = ce_bits(m0, torch.cat(prev_te, -1), cte[:, l].numpy())
            m1 = fit_linear(torch.cat([Xtr] + prev_tr, -1), ctr[:, l].numpy(), k, seed=seed)
            ce1 = ce_bits(m1, torch.cat([Xte] + prev_te, -1), cte[:, l].numpy())
            cond = "" if l == 0 else f" | true 1..{l}"
            print(f"   {l + 1}{cond:12s} {ent:8.2f} {ce0:20.2f} {ce1:17.2f} {ce0 - ce1:24.2f}")


if __name__ == "__main__":
    fire.Fire(main)
