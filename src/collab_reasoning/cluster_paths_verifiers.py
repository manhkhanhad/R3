import json, sys
import numpy as np
import torch
sys.path.insert(0, 'src/collab_reasoning')
import data as D

recs = json.load(open('/common/home/users/m/mk.ngohuu.2024/research/R3/analysis/results/collab_reasoning/cluster_paths/full/paths.json'))
test = recs[1000:]
titles = D.item_titles()
fused = torch.load('data/Musical_Instruments/info/collab_reasoning/candidates_fused_full.pt', weights_only=False)['test'].numpy()
src = {s: np.load(f'/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/collab_reasoning/sources/full/test_{s}.npz')['ids']
       for s in ('sasrec', 'itemknn', 'donor', 'markov')}
for r in test:
    i = r['row']
    r['cf10'] = {titles[x] for x in fused[i] if x >= 0}
    r['cf'] = {}
    for ids in src.values():
        for rank, x in enumerate(ids[i]):
            r['cf'][titles[x]] = r['cf'].get(titles[x], 0) + 1 / (rank + 1)


def clusters(r):
    return [c for c in r['paths'] if c != 'r3']


def r3hit(r):
    return r['target'] in r['paths']['r3']['titles']


def sel(rs, key):
    ok = [r for r in rs if str(r['target_cluster']) in r['paths']]
    acc = np.mean([max(clusters(r), key=lambda c: key(r, c)) == str(r['target_cluster']) for r in ok])
    return acc, len(ok), np.mean([1 / len(clusters(r)) for r in ok])


mass = lambda r, c: sum(r['cf'].get(t, 0) for t in r['paths'][c]['titles'])
feats = {
    'p(c|x) mix': lambda r, c: r['probs']['mix'].get(c, 0),
    'overlap with fused CF top-10': lambda r, c: len(set(r['paths'][c]['titles']) & r['cf10']),
    'CF reciprocal-rank mass (4 sources)': mass,
    'p(c|x) mix * (1 + CF mass)': lambda r, c: r['probs']['mix'].get(c, 0) * (1 + mass(r, c)),
}
for label, rs in (('R3-miss rows', [r for r in test if not r3hit(r)]), ('all rows', test)):
    print(label)
    for n, f in feats.items():
        acc, cnt, chance = sel(rs, f)
        print(f'  {n:38s} acc {acc:.3f}  (n={cnt}, chance {chance:.3f})')
