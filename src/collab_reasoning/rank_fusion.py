"""Untuned reciprocal-rank fusion of saved top-10 lists on the test split (no training)."""
import json, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
import torch
import data as D
R = '/common/home/users/m/mk.ngohuu.2024/research/R3/analysis/results/collab_reasoning'
titles = D.item_titles()
rows = D.load_rows('test')
tgt = [titles[t] for t in rows.target]
S = D.V1_PREDICTIONS
v1 = [[D.clean_title(q) for q in x['predict']] for x in json.load(open(S / '0.json')) + json.load(open(S / '1.json'))]
lists = {'v1': v1}
for v in ('cf_text', 'residual_text', 'cf_slots'):
    lists[v] = [r['top10'] for r in json.load(open(f'{R}/eval/full_{v}/full_predictions.json'))]
fused = torch.load(str(D.DATA) + '/info/collab_reasoning/candidates_fused_full.pt', weights_only=False)['test'].numpy()
lists['fused_cf'] = [[titles[i] for i in row if i >= 0] for row in fused]


def hr(L):
    return np.mean([t in l[:10] for t, l in zip(tgt, L)])


def rrf(*Ls, k=60, w=None):
    w = w or [1.0] * len(Ls)
    out = []
    for parts in zip(*Ls):
        s = {}
        for wi, l in zip(w, parts):
            for r, x in enumerate(l[:10]):
                s[x] = s.get(x, 0) + wi / (k + r + 1)
        out.append(sorted(s, key=lambda x: -s[x])[:10])
    return out


for name, L in lists.items():
    print(f'{name:14s} HR@10 {hr(L):.4f}')
for a in ('cf_text', 'residual_text', 'fused_cf'):
    print(f'RRF v1+{a:12s} HR@10 {hr(rrf(v1, lists[a])):.4f}   union ceiling {np.mean([t in x[:10] or t in y[:10] for t, x, y in zip(tgt, v1, lists[a])]):.4f}')
# paired test v1 vs cf_text
a = np.array([t in l for t, l in zip(tgt, v1)]); b = np.array([t in l for t, l in zip(tgt, lists['cf_text'])])
n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
from scipy.stats import binomtest
print('v1 vs cf_text: gained', n01, 'lost', n10, 'exact McNemar p =', round(binomtest(n01, n01 + n10).pvalue, 3))
c = np.array([t in l for t, l in zip(tgt, rrf(v1, lists['fused_cf']))])
n01, n10 = int((~a & c).sum()), int((a & ~c).sum())
print('v1 vs RRF(v1+fused): gained', n01, 'lost', n10, 'exact McNemar p =', binomtest(n01, n01 + n10).pvalue)
for k in (10, 30, 120):
    print('k', k, 'HR@10', round(hr(rrf(v1, lists['fused_cf'], k=k)), 4))
