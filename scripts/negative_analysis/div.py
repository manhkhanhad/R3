import ast, json, numpy as np, pandas as pd, torch
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
paths=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
t2i={}
for l in open(R+'info/Musical_Instruments_5_2015-1-2018-11.txt'):
    t,i=l.rstrip('\n').rsplit('\t',1); t2i.setdefault(t.strip(),int(i))
D=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl')
te=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
js=json.load(open(P+'0.json'))+json.load(open(P+'1.json'))
nd=[];lastshare=[];histshare=[]
for j,h in zip(js,te.history_item_id):
    h=ast.literal_eval(h); pl=[paths[t2i[p.strip().strip('"').strip()]] for p in j['predict']]
    hl={paths[x] for x in h}
    nd.append(len(set(pl))); lastshare.append(np.mean([p==paths[h[-1]] for p in pl])); histshare.append(np.mean([p in hl for p in pl]))
nd=np.array(nd); print('distinct leaves in top-10: mean %.2f median %d'%(nd.mean(),np.median(nd)))
print('share of top-10 slots in last item leaf %.3f ; in any history leaf %.3f'%(np.mean(lastshare),np.mean(histshare)))
print('target leaf is a history leaf: %.3f of rows'%D.leaf_in_hist.mean())
