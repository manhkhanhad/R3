import ast, numpy as np, pandas as pd, torch
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
paths=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
D=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl')
te=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
ts=te.timestamp.values; hts=[ast.literal_eval(x) for x in te.history_timestamp]; H=[ast.literal_eval(x) for x in te.history_item_id]
print('timestamp granularity: share divisible by 86400 = %.3f' % np.mean(ts%86400==0))
D['gap_days']=(ts-np.array([h[-1] for h in hts]))/86400
D['transition']=[paths[t]!=paths[h[-1]] for t,h in zip(te.item_id,H)]
# same-day purchases inside history: how often items are bought in the same "basket"
D['same_day_last']=D.gap_days==0
print('\nGap (days) last history item -> target')
for name,m in [('R3 hits',D.hit),('R3 misses',~D.hit),('continuation (same leaf)',~D.transition),('transition (leaf switch)',D.transition),('transition misses',D.transition&~D.hit)]:
    g=D.gap_days[m]; print(f'  {name:26s} n={m.sum():5d} same-day {np.mean(g==0):.3f}  <=7d {np.mean(g<=7):.3f}  median {g.median():6.1f}  mean {g.mean():6.1f}')
b=pd.cut(D.gap_days,[-1,0,1,7,30,90,365,1e9],labels=['same day','1 day','2-7d','8-30d','1-3mo','3-12mo','>1y'])
t=D.groupby(b,observed=True).agg(n=('hit','size'),share=('hit',lambda x: len(x)/len(D)),HR=('hit','mean'),transition_rate=('transition','mean'))
t['HR_transition']=D[D.transition].groupby(b[D.transition],observed=True).hit.mean()
t['HR_continuation']=D[~D.transition].groupby(b[~D.transition],observed=True).hit.mean()
print('\nBy gap bucket'); print(t.round(3).to_string())
# within-history: do transitions also happen between history items on same day (basket) vs different days?
same=[];diff=[]
for h,tt in zip(H,hts):
    for a,b2,ta,tb in zip(h[:-1],h[1:],tt[:-1],tt[1:]):
        (same if tb==ta else diff).append(paths[a]!=paths[b2])
print('\nWithin histories: leaf-switch rate between consecutive items  same day %.3f (n=%d)  different days %.3f (n=%d)'%(np.mean(same),len(same),np.mean(diff),len(diff)))
# target bought same day as several history items => basket
D['n_same_day_hist']=[sum(x==t for x in tt) for tt,t in zip(hts,ts)]
print('target shares its day with >=1 history item: %.3f overall; hits %.3f; misses %.3f'%((D.n_same_day_hist>0).mean(),(D.n_same_day_hist[D.hit]>0).mean(),(D.n_same_day_hist[~D.hit]>0).mean()))
