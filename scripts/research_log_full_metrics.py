"""Print HR/NDCG@1,3,5,10,20 as markdown rows using the repo's ranking metric code."""
import json
import sys

sys.path.insert(0, sys.argv[1])  # worktree src/
from utils.ranking_metrics import calculate_ranking_metrics

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
INFO = {
    "Musical": f"{R3}/data/Musical_Instruments/info/Musical_Instruments_5_2015-1-2018-11.txt",
    "CDs": f"{R3}/data/CDs_and_Vinyl/info/CDs_and_Vinyl_5_2015-10-2018-11.txt",
}
RUNS = [
    ("L1", "Musical", "two-phase (final)", f"{R3}/output_dir/Musical_Instruments_R3_collab_margin/0.json"),
    ("L2", "CDs", "two-phase (final)", f"{R3}/output_dir/CDs_and_Vinyl_R3_collab_margin/0.json"),
] + [
    ("L5", "CDs", mode, f"{R3}/output_dir/CDs_and_Vinyl_R3_grounded/eval_{mode}.json")
    for mode in ("knn", "off", "shuffle_outcome", "latent_only", "random")
] + [
    ("L4", "Musical", "knn", f"{R3}/output_dir/Musical_Instruments_R3_grounded/eval_knn.json"),
]
K = (1, 3, 5, 10, 20)
print("| run | data | mode | n | " + " | ".join(f"HR@{k}" for k in K) + " | " + " | ".join(f"NDCG@{k}" for k in K) + " |")
print("|---|---|---|---:|" + "---:|" * (2 * len(K)))
for run, data, mode, path in RUNS:
    m = calculate_ranking_metrics(json.load(open(path)), INFO[data])
    hr = " | ".join(f"{m[f'hr@{k}']:.4f}" for k in K)
    nd = " | ".join(f"{m[f'ndcg@{k}']:.4f}" for k in K)
    print(f"| {run} | {data} | {mode} | {m['samples']} | {hr} | {nd} |")
