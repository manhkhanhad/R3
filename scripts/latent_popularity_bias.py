import ast, glob, json, math, collections, sys
import numpy as np, pandas as pd
R="/common/home/users/m/mk.ngohuu.2024/research/R3"; S="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir"
D=f"{R}/data/Toys_and_Games"
info=glob.glob(f"{D}/info/Toys_and_Games_5_*.txt")[0]
t2i={}
for l in open(info):
    t,i=l.rstrip("\n").rsplit("\t",1); t2i.setdefault(t.strip(),int(i))
n=max(t2i.values())+1
tr=pd.read_csv(glob.glob(f"{D}/train/*.csv")[0])
cnt=np.zeros(n)
for i in tr.item_id.astype(int): cnt[i]+=1
for h in tr.history_item_id:
    for i in ast.literal_eval(h): cnt[i]+=1
order=np.argsort(-cnt); rank=np.empty(n,int); rank[order]=np.arange(n)
top1=rank< n*0.01; top10=rank< n*0.10
def norm(s): return s.strip().strip('"').strip()
def lists(path):
    d=json.load(open(path)); out=[]
    for r in d:
        p=r["predict"]; p=ast.literal_eval(p) if isinstance(p,str) else p
        out.append(([t2i.get(norm(x),-1) for x in p][:10], t2i.get(norm(r["output"]),-1)))
    return out
runs=dict(
 R3=f"{R}/../R3_ori/output_dir/Toys_and_Games/latent/final_result.json",
 no_latent=f"{R}/output_dir/r3_no_latent/Toys_and_Games/eval_no_latent.json",
 T2_const_raw=f"{S}/continuous_thoughts_results/constant_k1/eval_raw.json",
 T2_const_mask=f"{S}/continuous_thoughts_results/constant_k1/eval_mask.json",
 T3_const_nohist_raw=f"{S}/continuous_thoughts_results/constant_k1_bottleneck/eval_raw.json",
 L7_k1_raw=f"{S}/continuous_thoughts_results/hidden_k1_bottleneck/eval_raw.json",
 L7_k1_mean=f"{S}/continuous_thoughts_results/hidden_k1_bottleneck/eval_mean.json",
 L7_k1_shuffle=f"{S}/continuous_thoughts_results/hidden_k1_bottleneck/eval_shuffle.json",
 cand_sasrec_raw=f"{R}/output_dir/candidate_pooling/Toys_and_Games_cand_pool_sasrec_c10/eval_raw.json",
 cand_sasrec_drop=f"{R}/output_dir/candidate_pooling/Toys_and_Games_cand_pool_sasrec_c10/eval_drop.json",
)
res={}
tgt=None
for k,p in runs.items():
    L=lists(p); items=np.array([[i for i in l if i>=0] for l,_ in L],dtype=object)
    flat=np.concatenate([np.array(x,int) for x in items])
    per=[np.mean(np.log1p(cnt[np.array(x,int)])) for x in items if len(x)]
    ys=np.array([y for _,y in L]); 
    if tgt is None: tgt=ys
    hits=np.array([y in l for l,y in L])
    head=top10[ys]
    res[k]=dict(n=len(L),HR10=hits.mean(),mean_log_pop=np.mean(per),median_pop_rank=float(np.median(rank[flat])),
        share_top1pct=top1[flat].mean(),share_top10pct=top10[flat].mean(),coverage=len(set(flat.tolist()))/n,
        HR10_head_targets=hits[head].mean(),HR10_tail_targets=hits[~head].mean(), unmapped=float(np.mean([sum(i<0 for i in l) for l,_ in L])))
    res[k]["per_row"]=per
ys=tgt; ys=ys[ys>=0]
print(f"targets: mean_log_pop {np.mean(np.log1p(cnt[ys])):.3f} median_rank {np.median(rank[ys]):.0f} top1% {top1[ys].mean():.3f} top10% {top10[ys].mean():.3f}; catalog n={n}, head share of targets {top10[ys].mean():.3f}")
print(f"{'run':22s} {'HR@10':>6s} {'meanlogpop':>10s} {'med_rank':>8s} {'top1%':>6s} {'top10%':>6s} {'cover':>6s} {'HRhead':>6s} {'HRtail':>6s}")
for k,v in res.items():
    print(f"{k:22s} {v['HR10']:.4f} {v['mean_log_pop']:10.3f} {v['median_pop_rank']:8.0f} {v['share_top1pct']:.3f} {v['share_top10pct']:.3f} {v['coverage']:.3f} {v['HR10_head_targets']:.4f} {v['HR10_tail_targets']:.4f}  unm={v['unmapped']:.2f}")
rng=np.random.default_rng(42)
def boot(a,b):
    a=np.array(a);b=np.array(b);d=a-b;bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
    return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
for a,b in [("R3","no_latent"),("T2_const_raw","T2_const_mask"),("T2_const_raw","no_latent"),("L7_k1_raw","L7_k1_mean"),("L7_k1_raw","L7_k1_shuffle"),("cand_sasrec_raw","cand_sasrec_drop"),("T3_const_nohist_raw","no_latent")]:
    if len(res[a]["per_row"])==len(res[b]["per_row"]):
        m,l,h=boot(res[a]["per_row"],res[b]["per_row"]); print(f"Δ mean log-pop {a} − {b}: {m:+.3f} [{l:+.3f}, {h:+.3f}]")
pop10=set(order[:10].tolist())
L=lists(runs["R3"]); print("popularity top-10 baseline HR@10", np.mean([y in pop10 for _,y in L]))
for k in ("T3_const_nohist_raw","L7_k1_mean"):
    L=lists(runs[k]); c=collections.Counter(i for l,_ in L for i in l)
    print(k, "distinct", len(c), "most common:", [(int(i), int(rank[i]), m) for i,m in c.most_common(10)])
