import ast, json, collections, numpy as np, pandas as pd, torch
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
paths=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
a2i={}
for s in ('train','valid','test'):
    f=pd.read_csv(f'{R}{s}/Musical_Instruments_5_2015-1-2018-11.csv',usecols=['item_asin','item_id','item_asins','history_item_id'])
    a2i.update(zip(f.item_asin,f.item_id))
    for a,h in zip(f.item_asins,f.history_item_id): a2i.update(zip(ast.literal_eval(a),ast.literal_eval(h)))
print('catalog asins mapped',len(a2i))
ab={};av={}
with open(R+'meta_Musical_Instruments.json') as fh:
    for line in fh:
        if '"asin"' not in line: continue
        d=json.loads(line); i=a2i.get(d.get('asin'))
        if i is None: continue
        ab[i]=[a2i[x] for x in d.get('also_buy',[]) if x in a2i]; av[i]=[a2i[x] for x in d.get('also_view',[]) if x in a2i]
print('catalog items with meta',len(ab),' mean in-catalog also_buy %.1f also_view %.1f'%(np.mean([len(v) for v in ab.values()]),np.mean([len(v) for v in av.values()])))
D=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl')
te=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
H=[ast.literal_eval(h) for h in te.history_item_id]; T=te.item_id.values
trans=np.array([paths[t]!=paths[h[-1]] for t,h in zip(T,H)]); hit=D.hit.values
def score(h,graph,decay=0.7):
    c=collections.Counter()
    for k,x in enumerate(h):
        w=decay**(len(h)-1-k)
        for pos,y in enumerate(graph.get(x,[])): c[y]+=w/(1+0.05*pos)   # list order ~ strength
    for x in h: c.pop(x,None)
    return c
out={}
for name,g in [('also_buy',ab),('also_view',av),('buy+view',{i:ab.get(i,[])+av.get(i,[]) for i in set(ab)|set(av)})]:
    r10=[];r50=[];cov=[];leaf1=[]
    for h,t in zip(H,T):
        c=score(h,g); ranked=[x for x,_ in c.most_common(50)]
        cov.append(t in c); r10.append(t in ranked[:10]); r50.append(t in ranked)
        lc=collections.Counter()
        for x,v in c.items(): lc[paths[x]]+=v
        leaf1.append(bool(lc) and lc.most_common(1)[0][0]==paths[t])
    r10,r50,cov,leaf1=map(np.array,(r10,r50,cov,leaf1))
    for gname,m in [('all',np.ones_like(hit)),('R3 hits',hit),('R3 misses',~hit),('transition misses',(~hit)&trans)]:
        print(f'{name:9s} {gname:18s} n={m.sum():5d} target reachable {cov[m].mean():.3f}  item@10 {r10[m].mean():.3f}  @50 {r50[m].mean():.3f}  leaf@1 {leaf1[m].mean():.3f}')
    out[name]=r10
# complementarity: union with R3
for name,r10 in out.items():
    print(f'{name}: perfect union with R3 HR@10 = {np.mean(hit|r10):.4f} (R3 {hit.mean():.4f})')
