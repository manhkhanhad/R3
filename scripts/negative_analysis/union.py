import numpy as np, pandas as pd
T='/common/home/users/m/mk.ngohuu.2024/.claude/jobs/f205c9d9/tmp/miss/'
hit=pd.read_pickle(T+'rows.pkl').hit.values
r=np.load(T+'nextcat_res.npy',allow_pickle=True).item()
u=hit.copy()
for m in ['pop','graph1','graph2','puresvd','sasrec_mean']:
    h=r[m][:,1]<10
    print(f'{m:12s} HR@10 {h.mean():.4f}  union-with-R3 {np.mean(hit|h):.4f}')
    u|=h
print(f'R3 HR@10 {hit.mean():.4f} ; oracle union of R3 + 5 CF lists (60 slots) {u.mean():.4f}')
for m in ['graph1','puresvd']:
    print(f'{m} alone with 50 slots: HR@50 {np.mean(r[m][:,1]<50):.4f}')
