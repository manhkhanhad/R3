import ast, json, collections, numpy as np, pandas as pd, torch
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
title2id={}
for line in open(R+'info/Musical_Instruments_5_2015-1-2018-11.txt'):
    t,i=line.rstrip('\n').rsplit('\t',1); title2id.setdefault(t.strip(),int(i))
cat=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
def lvl(i,k):
    p=cat[i].split(' > ') if i is not None and 0<=i<len(cat) and cat[i] else []
    return ' > '.join(p[:k]) if len(p)>=k else (' > '.join(p) or None)
brand=lambda t: t.split()[0].lower().strip(',') if t else None
train=pd.read_csv(R+'train/Musical_Instruments_5_2015-1-2018-11.csv')
tgt_pop=collections.Counter(train.item_id)
any_pop=collections.Counter(train.item_id); [any_pop.update(ast.literal_eval(h)) for h in train.history_item_id]
rows=[]
for s in (0,1):
    df=pd.read_csv(P+f'{s}.csv'); js=json.load(open(P+f'{s}.json'))
    assert len(df)==len(js)
    for (_,r),j in zip(df.iterrows(),js):
        preds=[p.strip().strip('"').strip() for p in j['predict']]
        pid=[title2id.get(p) for p in preds]
        tgt=int(r.item_id); hist=ast.literal_eval(r.history_item_id); htit=ast.literal_eval(r.history_item_title)
        rank=next((k for k,x in enumerate(pid) if x==tgt),None)
        d=dict(hit=rank is not None, hist_len=len(hist), repeat=tgt in hist,
               tgt_pop=tgt_pop[tgt], tgt_cold=tgt_pop[tgt]==0, tgt_anypop=any_pop[tgt],
               pred_pop=np.mean([tgt_pop[x] for x in pid if x is not None]) if any(x is not None for x in pid) else np.nan,
               unmatched=sum(x is None for x in pid), pred_in_hist=sum(x in hist for x in pid if x is not None))
        for k,name in ((1,'top'),(2,'l2'),(3,'l3'),(99,'leaf')):
            tc=lvl(tgt,k); hc=[lvl(h,k) for h in hist]; pc=[lvl(x,k) for x in pid if x is not None]
            d[f'{name}_tgt']=tc; d[f'{name}_eq_last']=tc==hc[-1]; d[f'{name}_in_hist']=tc in hc
            d[f'{name}_pred_any']=tc in pc; d[f'{name}_pred_frac']=np.mean([c==tc for c in pc]) if pc else 0
        tb=brand(r.item_title); hb=[brand(t) for t in htit]
        d['brand']=tb; d['brand_in_hist']=tb in hb
        d['brand_pred_any']=any(brand(p)==tb for p in preds)
        d['brand_leaf_pred']=any(brand(p)==tb and lvl(x,99)==lvl(tgt,99) for p,x in zip(preds,pid) if x is not None)
        rows.append(d)
D=pd.DataFrame(rows); D.to_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl')
print('n',len(D),'HR@10 %.4f'%D.hit.mean(),'unmatched preds/row %.2f'%D.unmatched.mean())
