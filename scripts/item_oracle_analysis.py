"""Item-oracle analysis: seen/unseen split, copy-the-given-vector rates (nn, shuffle donor), copy/other rows."""
import ast, glob, json, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
R = "/common/home/users/m/mk.ngohuu.2024/research/R3"
kind = sys.argv[1]
res = f"{R}/output_dir/item_oracle/Toys_and_Games_item_oracle_{kind}"
prep = torch.load(f"{R}/output_dir/item_oracle/prep/Toys_item_{kind}.pt", weights_only=False)
E = prep["embeddings"].float(); n = len(E); tgt = prep["target"]["test"].numpy()
valid = prep["valid"]
title_id = {}
for l in open(glob.glob(f"{R}/data/Toys_and_Games/info/*.txt")[0]):
    t, i = l.rstrip("\n").rsplit("\t", 1); title_id.setdefault(t.strip(), int(i))
te = pd.read_csv(glob.glob(f"{R}/data/Toys_and_Games/test/*.csv")[0])
y = te.item_id.astype(int).to_numpy()
seen = prep["seen_target"]["test"].numpy()
fw = lambda t: (str(t).lower().split() or [""])[0]
copy = np.array([fw(x) in {fw(h) for h in ast.literal_eval(hs)} for x, hs in zip(te.item_title, te.history_item_title)])
cl = lambda s: s.strip().strip('"')
def load(m):
    rows = json.load(open(f"{res}/eval_{m}.json"))
    return [[title_id.get(cl(x), -1) for x in r["predict"][:10]] for r in rows]
# what vector each mode gave: nn targets / shuffle donors (replicate eval logic)
unit = F.normalize(E, dim=-1); unit[~valid] = 0
def nn_item(k):
    out = tgt.copy(); ok = np.where(tgt < n)[0]
    for s in range(0, len(ok), 1024):
        r = ok[s:s+1024]; sim = unit[tgt[r]] @ unit.T
        sim[np.arange(len(r)), tgt[r]] = -2; sim[:, ~valid] = -2
        out[r] = sim.topk(k, dim=-1).indices[:, -1].numpy()
    return out
users = te.user_id.astype(str).to_numpy(); rng = np.random.default_rng(0); donor = []  # eval passes seed=0 to CategoryRowsDataset
for i in range(len(te)):
    j = int(rng.integers(len(te)))
    while users[j] == users[i]: j = int(rng.integers(len(te)))
    donor.append(j)
given = {"oracle": tgt, "nn_1": nn_item(1), "nn_5": nn_item(5), "nn_20": nn_item(20), "shuffle": tgt[np.array(donor)],
         "pred1": prep["lists"]["test"][:, 0].numpy()}
print(f"{kind}: n={len(te)}, unseen targets {(~seen).sum()}")
print("| mode | HR@10 seen | HR@10 unseen | given item in top-1 | given item in top-10 | true target in top-10 |")
print("|---|---:|---:|---:|---:|---:|")
for m in ["oracle", "swaphist", "noise_0.5", "noise_0.75", "noise_0.9", "nn_1", "nn_5", "nn_20", "pred1", "shuffle"]:
    L = load(m); hit = np.array([y[r] in L[r] for r in range(len(L))])
    g = given.get(m, tgt)
    g1 = np.mean([L[r][0] == g[r] for r in range(len(L))]); g10 = np.mean([g[r] in L[r] for r in range(len(L))])
    print(f"| {m} | {hit[seen].mean():.4f} | {hit[~seen].mean():.4f} | {g1:.4f} | {g10:.4f} | {hit.mean():.4f} |")
L = load("oracle"); hit = np.array([y[r] in L[r] for r in range(len(L))])
print(f"oracle HR@10 copy rows {hit[copy].mean():.4f}, other rows {hit[~copy].mean():.4f}")
