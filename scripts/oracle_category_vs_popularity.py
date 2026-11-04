"""Does the additive oracle-category model just return the category's popular items?
Per row (target has a level-3 category): compare the model's top-10 with the
history-free category-popularity top-10 (training interaction counts, history excluded)."""
import ast, json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "output_dir/category_code/code_a72b0d9/scripts")
import category_prediction_data as cpd
paths = cpd.category_map("data/Toys_and_Games/meta_Toys_and_Games.json", 3)
prep = torch.load("output_dir/category_latent/prep/Toys_category_prep.pt", weights_only=False)
catalog = prep["catalog"]; unk = len(catalog); cidx = {c: i for i, c in enumerate(catalog)}
title_id = {}
for l in open("data/Toys_and_Games/info/Toys_and_Games_5_2017-1-2018-11.txt"):
    t, i = l.rstrip("\n").rsplit("\t", 1); title_id.setdefault(t.strip(), int(i))
n = max(title_id.values()) + 1
id_asin = {}
for s in ("train", "valid", "test"):
    f = pd.read_csv(f"data/Toys_and_Games/{s}/Toys_and_Games_5_2017-1-2018-11.csv")
    id_asin.update(zip(f.item_id.astype(int), f.item_asin))
    for a_s, i_s in zip(f.item_asins, f.history_item_id): id_asin.update(zip(ast.literal_eval(i_s), ast.literal_eval(a_s)))
icat = np.array([cidx.get(paths.get(id_asin.get(i)), unk) for i in range(n)])
tr = pd.read_csv("data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv")
cnt = np.zeros(n)
for i in tr.item_id.astype(int): cnt[i] += 1
for h in tr.history_item_id:
    for i in ast.literal_eval(h): cnt[i] += 1
te = pd.read_csv("data/Toys_and_Games/test/Toys_and_Games_5_2017-1-2018-11.csv")
hist = [set(ast.literal_eval(h)) for h in te.history_item_id]
fw = lambda t: (str(t).lower().split() or [""])[0]
copy = np.array([fw(x) in {fw(h) for h in ast.literal_eval(hs)} for x, hs in zip(te.item_title, te.history_item_title)])
target = prep["target"]["test"].numpy(); known = target != unk
y = te.item_id.astype(int).to_numpy()
members = {c: np.where(icat == c)[0] for c in range(unk)}
def catpop(c, h):
    m = [i for i in members[c][np.argsort(-cnt[members[c]], kind="stable")] if i not in h]
    return m
# within-category popularity percentile (0 = most popular) for every item
pct = np.full(n, np.nan)
for c, m in members.items():
    o = m[np.argsort(-cnt[m], kind="stable")]; pct[o] = np.arange(len(o)) / max(len(o) - 1, 1)
cl = lambda s: s.strip().strip('"')
runs = {"oracle model, own history": "output_dir/category_latent/Toys_and_Games_cat_latent_additive/eval_oracle.json",
        "oracle model, other user's history": "output_dir/category_latent/Toys_and_Games_cat_latent_additive/eval_oracle_swaphist.json"}
cp = [catpop(target[r], hist[r]) if known[r] else [] for r in range(len(te))]
rng = np.random.default_rng(0)
print("rows", known.sum(), "| median category size", int(np.median([len(members[target[r]]) for r in np.where(known)[0]])))
hit_cp = np.array([y[r] in cp[r][:10] for r in range(len(te))])
print(f"category popularity HR@10 all/copy/other: {hit_cp[known].mean():.4f} / {hit_cp[known&copy].mean():.4f} / {hit_cp[known&~copy].mean():.4f}")
for name, p in runs.items():
    rows = json.load(open(p))
    L = [[title_id.get(cl(x), -1) for x in r["predict"][:10]] for r in rows]
    hit = np.array([y[r] in L[r] for r in range(len(L))])
    ov = np.array([len(set(L[r]) & set(cp[r][:10])) / 10 for r in range(len(L))])
    incat = np.array([[i for i in L[r] if i >= 0 and icat[i] == target[r]] for r in range(len(L))], dtype=object)
    pc = np.array([np.mean(pct[np.array(x, int)]) if len(x) else np.nan for x in incat])
    # random-in-category expectation of overlap: 10/|category minus history|
    exp = np.array([min(10, len(cp[r])) / max(len(cp[r]), 1) for r in range(len(L))])
    both = (hit & hit_cp); only_m = hit & ~hit_cp; only_c = ~hit & hit_cp
    for sub, msk in (("all", known), ("copy", known & copy), ("other", known & ~copy)):
        print(f"{name:36s} {sub:5s} HR@10 {hit[msk].mean():.4f} | overlap with catpop top-10 {ov[msk].mean():.3f} (random-in-category {exp[msk].mean():.3f}) "
              f"| within-category popularity percentile of in-category recs {np.nanmean(pc[msk]):.3f} (random 0.5)")
    print(f"  hits: both {both[known].sum()}, model only {only_m[known].sum()}, catpop only {only_c[known].sum()}")
