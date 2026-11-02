"""History-free baseline given the oracle target category (Musical): top-10 most frequent training targets in that category."""
import ast, glob, collections, math, torch, pandas as pd
D = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments"
paths = torch.load(f"{D}/info/Musical_Instruments_gte_large_en_v1_5_categories.pt", weights_only=False)["paths"]
tr = pd.read_csv(glob.glob(f"{D}/train/*.csv")[0]); te = pd.read_csv(glob.glob(f"{D}/test/*.csv")[0])
cnt = collections.Counter(tr.item_id.astype(int))
bycat = collections.defaultdict(list)
for i in range(len(paths)): bycat[paths[i]].append(i)
for c in bycat: bycat[c].sort(key=lambda i: -cnt.get(i, 0))
def metrics(lists, ys):
    out = {}
    for k in (1, 3, 5, 10):
        hr = nd = 0
        for L, y in zip(lists, ys):
            if y in L[:k]:
                hr += 1; nd += 1 / math.log2(L.index(y) + 2)
        out[f"HR@{k}"] = round(hr / len(ys), 4); out[f"NDCG@{k}"] = round(nd / len(ys), 4)
    return out
ys = te.item_id.astype(int).tolist()
hist = [set(ast.literal_eval(h)) for h in te.history_item_id]
a = [bycat[paths[y]][:10] for y in ys]
b = [[i for i in bycat[paths[y]] if i not in h][:10] for y, h in zip(ys, hist)]
sizes = [len(bycat[paths[y]]) for y in ys]
print("n", len(ys), "categories", len(bycat), "median target-category size", sorted(sizes)[len(sizes)//2])
print("catpop (oracle category, no history):           ", metrics(a, ys))
print("catpop minus history items (history used only to exclude):", metrics(b, ys))
print("rows with category size <= 10:", sum(s <= 10 for s in sizes) / len(sizes))
