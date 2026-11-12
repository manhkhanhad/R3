#!/usr/bin/env bash
#SBATCH --job-name=R3_cluster_posterior
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# E2-diag: exact decoder posterior over CF-cluster evidence (refine-logs/EXPERIMENT_PLAN.md).
# Test rows (diagnostic, matched to the cluster-path decodes) + all validation rows (E3 training data).
# Usage: sbatch scripts/collab_posterior.sbatch [full|smoke]
source scripts/collab_env.sh
mode="${1:-full}"
fwd=32; (( is_3090 )) && fwd=16
test_rows=3000; valid_rows=-1
[[ "$mode" == smoke ]] && { test_rows=16; valid_rows=16; }
out=${scratch}/collab_reasoning/posterior/${mode}
paths=${results}/cluster_paths/${PATHS_TAG:-full_retrain}/paths.json
(
  cd src/prototype_grounding
  python cluster_posterior.py --cluster_model "$cluster_oracle" --r3_model "$v1" --split test --rows "$test_rows" \
    --r3_hits "$paths" --fwd_batch "$fwd" --out "$out"
  python cluster_posterior.py --cluster_model "$cluster_oracle" --r3_model "$v1" --split valid --rows "$valid_rows" \
    --r3_hits "${v1_valid_preds}/predictions.json" --fwd_batch "$fwd" --out "$out"
)
python src/collab_reasoning/cluster_posterior_report.py --posterior "$out/test.pt" --paths "$paths" \
  --out "${results}/posterior/${mode}/e2_diag.md"
echo "[$(date --iso-8601=seconds)] done: ${results}/posterior/${mode}/e2_diag.md"
