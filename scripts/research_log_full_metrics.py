"""Print HR/NDCG@1,3,5,10,20 as markdown rows using the repo's ranking metric code."""
import json
import os
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
]
# Every finished control/mode file of the grounded and intent runs.
for run, data, folder, modes in (
    ("L4", "Musical", "Musical_Instruments_R3_grounded", ("knn", "off", "shuffle_outcome", "latent_only", "random")),
    ("L5", "CDs", "CDs_and_Vinyl_R3_grounded", ("knn", "off", "shuffle_outcome", "latent_only", "random")),
    ("L6a", "Musical", "Musical_Instruments_R3_intent", ("mixture", "map", "oracle", "random", "r0")),
    ("L6b", "Musical", "Musical_Instruments_R3_intent_margin", ("mixture", "map", "oracle", "random", "r0")),
    ("L6c", "Musical", "Musical_Instruments_R3_intent_fixes", ("mixture", "map", "oracle", "random", "r0")),
    ("L6d-valid", "Musical", "Musical_Instruments_R3_intent_router_valid", ("aggregate", "prior_blend", "aggregate_uniform")),
    ("L6d-train", "Musical", "Musical_Instruments_R3_intent_router_train", ("aggregate", "prior_blend", "aggregate_uniform")),
):
    for mode in modes:
        path = f"{R3}/output_dir/{folder}/eval_{mode}.json"
        if os.path.exists(path):
            RUNS.append((run, data, mode, path))
K = (1, 3, 5, 10, 20)
for dataset in INFO:
    rows = [r for r in RUNS if r[1] == dataset]
    if not rows:
        continue
    print(f"\n### {dataset}\n")
    print("| run | mode | n | " + " | ".join(f"HR@{k}" for k in K) + " | " + " | ".join(f"NDCG@{k}" for k in K) + " |")
    print("|---|---|---:|" + "---:|" * (2 * len(K)))
    for run, _, mode, path in rows:
        m = calculate_ranking_metrics(json.load(open(path)), INFO[dataset])
        hr = " | ".join(f"{m[f'hr@{k}']:.4f}" for k in K)
        nd = " | ".join(f"{m[f'ndcg@{k}']:.4f}" for k in K)
        print(f"| {run} | {mode} | {m['samples']} | {hr} | {nd} |")
