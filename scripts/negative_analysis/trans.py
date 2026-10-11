import ast, collections, numpy as np, pandas as pd, torch
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
cat=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
D=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl')
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
test=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
tr=pd.read_csv(R+'train/Musical_Instruments_5_2015-1-2018-11.csv')
leaf=lambda i: cat[i]
# category transitions: last-item leaf -> target leaf; any-history leaf -> target leaf (train)
T1=collections.defaultdict(collections.Counter); TA=collections.defaultdict(collections.Counter); glob=collections.Counter()
I1=collections.defaultdict(collections.Counter)  # item-level: last item -> target item
for h,t in zip(tr.history_item_id,tr.item_id):
    h=ast.literal_eval(h); glob[leaf(t)]+=1
    if not h: continue
    T1[leaf(h[-1])][leaf(t)]+=1; I1[h[-1]][t]+=1
    for x in set(map(leaf,h)): TA[x][leaf(t)]+=1
pop=collections.Counter(tr.item_id)
leafpop=collections.defaultdict(list)
for i,c in pop.items(): leafpop[leaf(i)].append((c,i))
for v in leafpop.values(): v.sort(reverse=True)
def topk(counter,k): return [c for c,_ in counter.most_common(k)]
out=[]
for (_,r),hit in zip(test.iterrows(),D.hit):
    h=ast.literal_eval(r.history_item_id); t=int(r.item_id); tl=leaf(t)
    last=T1.get(leaf(h[-1]),collections.Counter()); anyc=collections.Counter()
    for x in set(map(leaf,h)): anyc.update(TA.get(x,{}))
    rank_in_leaf=next((k for k,(c,i) in enumerate(leafpop[tl]) if i==t),None)
    nxt=I1.get(h[-1],collections.Counter())
    out.append(dict(hit=hit, eq_last=tl==leaf(h[-1]),
        last_top5=tl in topk(last,5), any_top5=tl in topk(anyc,5), glob_top5=tl in topk(glob,5),
        any_top10=tl in topk(anyc,10),
        leaf_rank=rank_in_leaf, leaf_size=len(leafpop[tl]),
        cooc_top10=t in topk(nxt,10), cooc_seen=len(nxt)>0))
O=pd.DataFrame(out); M=O[~O.hit]; H=O[O.hit]
print('category-transition recall of target leaf (from train statistics)')
for c in ['last_top5','any_top5','any_top10','glob_top5','cooc_seen','cooc_top10']:
    print(f'  {c:12s} hits {H[c].mean():.3f}  misses {M[c].mean():.3f}  misses w/ leaf!=last {M[~M.eq_last][c].mean():.3f}')
print('target popularity rank within its leaf (0 = most popular), misses:')
print(M.leaf_rank.describe(percentiles=[.25,.5,.75]).round(1).to_string())
print('misses whose target is top-3 in its leaf: %.3f ; leaf size median %d'%((M.leaf_rank<3).mean(), M.leaf_size.median()))
# best achievable: predict R3 + "leaf transition then top-popular item" ? -> oracle-ish upper count
fix=(~O.hit)&O.any_top5&(O.leaf_rank<3)
print('misses fixable by "top-5 transitioned leaf x top-3 popular item" (15 candidates): %d (%.1f%% of misses, +%.4f HR if perfectly merged)'%(fix.sum(),100*fix.sum()/len(M),fix.sum()/len(O)))
