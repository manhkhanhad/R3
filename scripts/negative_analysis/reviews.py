import ast, json, collections, numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
paths=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
te=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
D=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl'); hit=D.hit.values
H=[ast.literal_eval(h) for h in te.history_item_id]; HA=[ast.literal_eval(a) for a in te.item_asins]; T=te.item_id.values
HT=[ast.literal_eval(t) for t in te.history_timestamp]
trans=np.array([paths[t]!=paths[h[-1]] for t,h in zip(T,H)])
users=set(te.user_id); need={(u,a) for u,aa in zip(te.user_id,HA) for a in aa}
rev={}
with open(R+'Musical_Instruments.json') as fh:
    for line in fh:
        d=json.loads(line); u=d.get('reviewerID')
        if u in users and (u,d.get('asin')) in need:
            rev[(u,d['asin'],d.get('unixReviewTime'))]=(d.get('summary','')+'. '+d.get('reviewText',''))
byua=collections.defaultdict(list)
for (u,a,t),txt in rev.items(): byua[(u,a)].append((t,txt))
# item documents: title + category + description
a2i={};title={}
for line in open(R+'info/Musical_Instruments_5_2015-1-2018-11.txt'):
    t,i=line.rstrip('\n').rsplit('\t',1); title[int(i)]=t
for s in ('train','valid','test'):
    f=pd.read_csv(f'{R}{s}/Musical_Instruments_5_2015-1-2018-11.csv',usecols=['item_asin','item_id']); a2i.update(zip(f.item_asin,f.item_id))
N=len(paths); docs=[title.get(i,'')+' '+(paths[i] or '') for i in range(N)]
qt=[];qr=[];has=[];rlen=[]
for u,h,aa,tt in zip(te.user_id,H,HA,HT):
    titles=' '.join(title.get(x,'') for x in h)
    rv=[]
    for a,t in zip(aa,tt):
        for (rt,txt) in byua.get((u,a),[]):
            if rt is None or rt<=t+86400: rv.append(txt)   # review written by the history item's date
    has.append(len(rv)>0); rlen.append(sum(len(x.split()) for x in rv))
    qt.append(titles); qr.append(' '.join(rv))
print('rows with >=1 history review: %.3f ; mean review words per row %.0f'%(np.mean(has),np.mean(rlen)))
vec=TfidfVectorizer(stop_words='english',sublinear_tf=True,min_df=2).fit(docs+qr[:3000])
Dm=vec.transform(docs)
def evalq(Q,name):
    S=(vec.transform(Q)@Dm.T).toarray()
    for k,h in enumerate(H): S[k,h]=-1
    st=S[np.arange(len(T)),T][:,None]; rank=(S>st).sum(1)+((S==st).sum(1)-1)//2+(st[:,0]<=0)*N   # ties: middle; zero score: never a hit
    for g,m in [('all',np.ones_like(hit)),('R3 hits',hit),('R3 misses',~hit),('transition misses',(~hit)&trans)]:
        print(f'{name:16s} {g:18s} item@10 {np.mean(rank[m]<10):.3f} @50 {np.mean(rank[m]<50):.3f} @200 {np.mean(rank[m]<200):.3f}')
    return rank<10
a=evalq(qt,'titles only'); b=evalq(qr,'reviews only'); c=evalq([x+' '+y for x,y in zip(qt,qr)],'titles+reviews')
print('union with R3 HR@10: reviews-only %.4f titles+reviews %.4f (R3 %.4f)'%(np.mean(hit|b),np.mean(hit|c),hit.mean()))
