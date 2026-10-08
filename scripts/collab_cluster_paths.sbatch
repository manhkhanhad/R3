#!/usr/bin/env bash
#SBATCH --job-name=R3_cluster_paths
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=36:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Parallel cluster-hypothesis paths + marginalization (document/collaborative_reasoning_experiments.md, idea 2 follow-up).
# Usage: sbatch scripts/collab_cluster_paths.sbatch [full|smoke]
source scripts/collab_env.sh
mode="${1:-full}"
gen_batch=8; (( is_3090 )) && gen_batch=4
rows=3000; tune=1000
[[ "$mode" == smoke ]] && { rows=16; tune=8; }
out=${results}/cluster_paths/${mode}
(
  cd src/prototype_grounding
  python cluster_paths_decode.py --cluster_model "$cluster_oracle" --r3_model "$v1" --rows "$rows" --topk 5 \
    --predictors fused,grufull,mix --gen_batch_size "$gen_batch" --out "$out"
)
python src/collab_reasoning/cluster_paths_aggregate.py --paths "$out/paths.json" --tune_rows "$tune"
echo "[$(date --iso-8601=seconds)] done: $out/aggregate.md"
