"""Stage 2: cluster the TARGET part (N clusters) and HISTORY part (M clusters); measure information vs predictability.

Information (oracle = the row's own target-part cluster):
  purity w.r.t. the next item's category; rank of the true item (mxbai catalog space, history masked) when the
  query is history + cluster centroid, vs history alone.
Predictability of the target-part cluster on TEST (models fit on TRAIN):
  majority; M->N table P(N | history-part cluster); linear classifier on the history-part embedding;
  linear classifier on the history-text embedding (leak-free); CF model on history item ids (leak-free).
Usefulness proxy: rank with history + PREDICTED cluster centroid (top-1) vs history alone.

Note: the history PART was written with the target visible (stand-in until it is regenerated without it).
"""

import ast
import json

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
ITEM_EMB = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/outputs/item_emb_mxbai.pt"


def kmeans(X, k, seed=0):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, max_iter=200).fit(X)
    C = torch.from_numpy(km.cluster_centers_).float()
    return km, C / C.norm(dim=-1, keepdim=True)


def fit_classifier(make_inputs, n_in_fn, y_tr, n_classes, train_idx, epochs=30, seed=0):
    """Tiny softmax model with 10% held-out early stopping. make_inputs(rows) -> tensor/tuple."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    hold, fit = train_idx[perm[: len(perm) // 10]], train_idx[perm[len(perm) // 10:]]
    model = n_in_fn()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.as_tensor(y_tr)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for s in range(0, len(fit), 1024):
            b = fit[rng.permutation(len(fit))][: 1024] if False else fit[s:s + 1024]
            loss = nn.functional.cross_entropy(model(make_inputs(b)), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hl = float(nn.functional.cross_entropy(model(make_inputs(hold)), y[hold]))
        if hl < best - 1e-4:
            best, best_state, bad = hl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
        rng.shuffle(fit)
    model.load_state_dict(best_state)
    return model.eval()


def topk_acc(scores, y, ks=(1, 5)):
    order = torch.as_tensor(scores).topk(max(ks), dim=1).indices.numpy()
    y = np.asarray(y)
    return {k: float(np.mean([y[i] in order[i, :k] for i in range(len(y))])) for k in ks}, order


def main(Ns=(16, 32, 64, 128, 256), M=256, emb=f"{R3}/output_dir/reasoning_geometry", seed=0):
    Ns = [int(n) for n in (Ns if isinstance(Ns, (list, tuple)) else [Ns])]
    P = {s: torch.load(f"{emb}/parts/{s}_parts.pt") for s in ("train", "test")}
    H = {s: torch.load(f"{emb}/mxbai/{s}.pt")["history"].float() for s in ("train", "test")}
    dtr, dte = pd.read_csv(f"{DATA}/train/{STEM}.csv"), pd.read_csv(f"{DATA}/test/{STEM}.csv")
    tr_ok = np.where(P["train"]["has"].numpy())[0]
    te_ok = np.where(P["test"]["has"].numpy())[0]
    Ttr, Tte = P["train"]["target_part"].float(), P["test"]["target_part"].float()
    Htr_part, Hte_part = P["train"]["history_part"].float(), P["test"]["history_part"].float()
    # history part may be empty for one-sentence paragraphs: fall back to history text embedding
    for X, Hx in ((Htr_part, H["train"]), (Hte_part, H["test"])):
        empty = X.norm(dim=-1) == 0
        X[empty] = Hx[empty]

    need = set(dtr["item_asin"]) | set(dte["item_asin"])
    cat = {}
    for line in open(f"{DATA}/meta_Toys_and_Games.json"):
        m = json.loads(line)
        if m.get("asin") in need and m["asin"] not in cat:
            cat[m["asin"]] = " > ".join((m.get("category") or [])[1:]) or "unknown"
    tr_cat = np.array([cat.get(a, "unknown") for a in dtr["item_asin"]])
    te_cat = np.array([cat.get(a, "unknown") for a in dte["item_asin"]])
    items = torch.load(ITEM_EMB).float(); items = items / items.norm(dim=-1, keepdim=True)
    hist_ids = [ast.literal_eval(h) for h in dte["history_item_id"]]
    target = dte["item_id"].to_numpy()

    def ranks(Q, rows):
        Q = Q / Q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = Q @ items.T
        out = []
        for r, i in enumerate(rows):
            excl = [h for h in hist_ids[i] if h != target[i]]
            if excl:
                S[r, excl] = float("-inf")
            out.append(int((S[r] > S[r, target[i]]).sum()) + 1)
        return np.array(out)

    def summary(rk):
        return f"median {np.median(rk):5.0f}  HR@10 {np.mean(rk <= 10):5.1%}  HR@100 {np.mean(rk <= 100):5.1%}"

    norm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    base = ranks(H["test"][te_ok].clone(), te_ok)
    print(f"test rows with a target part: {len(te_ok)}")
    print(f"baseline  history text alone:                         {summary(base)}\n")

    # history-part clusters (M) for the M->N mapping
    kmH, CH = kmeans(Htr_part[tr_ok].numpy(), M, seed)
    mtr = kmH.predict(Htr_part[tr_ok].numpy()); mte = kmH.predict(Hte_part[te_ok].numpy())

    # CF model inputs: history item ids (+ last item)
    n_items = items.shape[0]
    tr_hist = [ast.literal_eval(h) for h in dtr["history_item_id"]]

    def bag(rows, hists):
        ids, offs = [], []
        for i in rows:
            offs.append(len(ids)); ids += hists[i]
        return torch.tensor(ids), torch.tensor(offs), torch.tensor([hists[i][-1] for i in rows])

    class CF(nn.Module):
        def __init__(self, n_out, d=128):
            super().__init__()
            self.bag = nn.EmbeddingBag(n_items, d, mode="mean"); self.last = nn.Embedding(n_items, d)
            self.out = nn.Sequential(nn.ReLU(), nn.Linear(2 * d, n_out))

        def forward(self, x):
            ids, offs, last = x
            return self.out(torch.cat([self.bag(ids, offs), self.last(last)], -1))

    for N in Ns:
        kmT, CT = kmeans(Ttr[tr_ok].numpy(), N, seed)
        ytr = np.full(len(dtr), -1); ytr[tr_ok] = kmT.predict(Ttr[tr_ok].numpy())
        yte = kmT.predict(Tte[te_ok].numpy())
        # purity w.r.t. next-item category
        purity = 0
        for c in range(N):
            cats = tr_cat[tr_ok][ytr[tr_ok] == c]
            if len(cats):
                purity += pd.Series(cats).value_counts().iloc[0]
        purity /= len(tr_ok)
        print(f"===== N = {N} target-part clusters (train sizes: min {np.bincount(ytr[tr_ok]).min()}, max {np.bincount(ytr[tr_ok]).max()}); "
              f"category purity {purity:.1%}")
        oracle = ranks(norm(H["test"][te_ok]) + norm(CT[yte]), te_ok)
        print(f"   INFORMATION  history + TRUE cluster centroid:          {summary(oracle)}")

        # predictors
        preds = {}
        maj = np.bincount(ytr[tr_ok]).argmax()
        preds["majority cluster"] = np.tile(np.bincount(ytr[tr_ok], minlength=N), (len(te_ok), 1)).astype(float)
        table = np.ones((M, N)) * 0.1
        np.add.at(table, (mtr, ytr[tr_ok]), 1)
        preds[f"M->N table (history-part cluster, M={M})"] = table[mte]
        lin = lambda: nn.Linear(1024, N)
        mdl = fit_classifier(lambda b: Htr_part[b], lin, ytr, N, tr_ok, seed=seed)
        with torch.no_grad():
            preds["linear on history-PART embedding"] = mdl(Hte_part[te_ok]).numpy()
        mdl = fit_classifier(lambda b: H["train"][b], lin, ytr, N, tr_ok, seed=seed)
        with torch.no_grad():
            preds["linear on history TEXT embedding (leak-free)"] = mdl(H["test"][te_ok]).numpy()
        mdl = fit_classifier(lambda b: bag(b, tr_hist), lambda: CF(N), ytr, N, tr_ok, seed=seed)
        with torch.no_grad():
            preds["CF on history item ids (leak-free)"] = mdl(bag(te_ok, hist_ids)).numpy()
        for name, sc in preds.items():
            acc, order = topk_acc(sc, yte)
            use = ranks(norm(H["test"][te_ok]) + norm(CT[torch.from_numpy(order[:, 0])]), te_ok)
            print(f"   PREDICT {name:46s} top-1 {acc[1]:5.1%}  top-5 {acc[5]:5.1%} | history + predicted centroid: {summary(use)}")


if __name__ == "__main__":
    fire.Fire(main)
