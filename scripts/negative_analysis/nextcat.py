import ast, collections, numpy as np, pandas as pd, torch, scipy.sparse as sp
from scipy.sparse.linalg import svds
torch.manual_seed(0); np.random.seed(0)
R='/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments/'
P='/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/'
paths=torch.load(R+'info/Musical_Instruments_gte_large_en_v1_5_categories.pt',weights_only=False)['paths']
N=len(paths); leaves=sorted({p or 'NONE' for p in paths}); L={c:k for k,c in enumerate(leaves)}
leaf=np.array([L[p or 'NONE'] for p in paths]); C=len(leaves)
tr=pd.read_csv(R+'train/Musical_Instruments_5_2015-1-2018-11.csv')
te=pd.concat([pd.read_csv(P+'0.csv'),pd.read_csv(P+'1.csv')],ignore_index=True)
hit=pd.read_pickle('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/rows.pkl').hit.values
trH=[ast.literal_eval(h) for h in tr.history_item_id]; trT=tr.item_id.values
teH=[ast.literal_eval(h) for h in te.history_item_id]; teT=te.item_id.values
def rw(h,decay=0.7):  # recency weights, last item weight 1
    return [decay**(len(h)-1-k) for k in range(len(h))]
# --- statistics from train ---
pop=np.bincount(trT,minlength=N).astype(float); leafpop=np.bincount(leaf[trT],minlength=C).astype(float)
M1=np.zeros((C,C))
for h,t in zip(trH,trT):
    for x,w in zip(h,rw(h)): M1[leaf[x],leaf[t]]+=w
M1n=M1/ (M1.sum(1,keepdims=True)+1)
# item transition graph: history item -> target, recency weighted
rows,cols,vals=[],[],[]
for h,t in zip(trH,trT):
    for x,w in zip(h,rw(h)): rows.append(x); cols.append(t); vals.append(w)
G=sp.csr_matrix((vals,(rows,cols)),shape=(N,N)); G.sum_duplicates()
deg=np.asarray(G.sum(1)).ravel(); Gn=sp.diags(1/np.maximum(deg,1))@G
G2=Gn@Gn
# PureSVD over user x item (train sequences)
users={u:k for k,u in enumerate(tr.user_id.unique())}
ui=collections.defaultdict(set)
for u,h,t in zip(tr.user_id,trH,trT): ui[users[u]].update(h); ui[users[u]].add(t)
r_,c_=zip(*[(u,i) for u,s in ui.items() for i in s])
X=sp.csr_matrix((np.ones(len(r_)),(r_,c_)),shape=(len(users),N))
X=X@sp.diags(1/np.sqrt(np.maximum(np.asarray(X.sum(0)).ravel(),1)))
_,_,Vt=svds(X.astype(float),k=64); V=Vt.T
S=torch.load(R+'info/Musical_Instruments_sasrec_item.pt',weights_only=False)['embeddings'].float().numpy()
S/=np.linalg.norm(S,axis=1,keepdims=True)+1e-9
# learned next-leaf classifier: softmax regression on [recency bag of leaves, last leaf one-hot, bag of items]
def feat(H):
    r,c,v=[],[],[]
    for k,h in enumerate(H):
        for x,w in zip(h,rw(h)): r+= [k,k]; c+=[leaf[x], 2*C+x]; v+=[w,w]
        r.append(k); c.append(C+leaf[h[-1]]); v.append(1.)
    m=sp.coo_matrix((v,(r,c)),shape=(len(H),2*C+N)).tocsr(); return m
Xtr=feat(trH); Xte=feat(teH)
def tens(m):
    m=m.tocoo(); return torch.sparse_coo_tensor(np.vstack([m.row,m.col]),m.data,m.shape,dtype=torch.float32).coalesce()
W=torch.nn.Linear(Xtr.shape[1],C); opt=torch.optim.Adam(W.parameters(),lr=3e-3,weight_decay=1e-6)
yt=torch.tensor(leaf[trT])
for ep in range(8):
    perm=np.random.permutation(len(trT))
    for b in range(0,len(perm),1024):
        idx=perm[b:b+1024]; loss=torch.nn.functional.cross_entropy(W(tens(Xtr[idx]).to_dense()),yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): clf_leaf=torch.cat([W(tens(Xte[b:b+2048]).to_dense()) for b in range(0,len(teT),2048)]).numpy()
# --- per test row scores ---
def item_scores(name,h):
    w=np.array(rw(h))
    if name=='pop': return pop
    if name=='graph1': return (sp.csr_matrix((w,(np.zeros(len(h)),h)),shape=(1,N))@Gn).toarray().ravel()
    if name=='graph2':
        q=sp.csr_matrix((w,(np.zeros(len(h)),h)),shape=(1,N)); return (q@Gn+0.5*(q@G2)).toarray().ravel()
    if name=='puresvd': return (w@V[h])@V.T
    if name=='sasrec_mean': return S@(w@S[h])
def leaf_from_items(s,h):
    s=s.copy(); s[h]=-np.inf; top=np.argsort(-s)[:200]; v=np.maximum(s[top],0)+1e-9
    return np.bincount(leaf[top],weights=v,minlength=C)
methods=['pop','leafMarkov','graph1','graph2','puresvd','sasrec_mean','classifier']
res={m:[] for m in methods}
for k,(h,t) in enumerate(zip(teH,teT)):
    tl=leaf[t]
    for m in methods:
        if m=='leafMarkov': ls=sum(w*M1n[leaf[x]] for x,w in zip(h,rw(h)))+1e-6*leafpop; it=None
        elif m=='classifier': ls=clf_leaf[k]; it=None
        else:
            it=item_scores(m,h).astype(float); ls=leaf_from_items(it,h)
        lr=(ls>ls[tl]).sum()
        ir=np.nan
        if it is not None: s=it.copy(); s[h]=-np.inf; ir=(s>s[t]).sum()
        res[m].append((lr,ir))
trans=leaf[teT]!=np.array([leaf[h[-1]] for h in teH])
groups={'all':np.ones(len(teT),bool),'R3 hits':hit,'R3 misses':~hit,'transition misses':(~hit)&trans}
print('leaf recall@1/@5/@10 | item recall@10/@50   (items already in history excluded)')
for m in methods:
    a=np.array(res[m],dtype=float)
    line=f'{m:12s}'
    for g,msk in groups.items():
        lr=a[msk,0]; ir=a[msk,1]
        line+=f' | {g}: {np.mean(lr<1):.3f}/{np.mean(lr<5):.3f}/{np.mean(lr<10):.3f}'
        if not np.isnan(ir).all(): line+=f' it {np.mean(ir<10):.3f}/{np.mean(ir<50):.3f}'
    print(line)
print('groups n:',{g:int(m.sum()) for g,m in groups.items()}, 'leaves',C)
np.save('/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/nextcat_res.npy',{m:np.array(res[m],dtype=float) for m in methods},allow_pickle=True)
