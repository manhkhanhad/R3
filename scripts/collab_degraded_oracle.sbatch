#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_degraded
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Degraded CF-cluster oracle dose-response + plug-in group predictor (idea 5's accuracy threshold).
# Usage: sbatch scripts/collab_degraded_oracle.sbatch [full|smoke]
source scripts/collab_env.sh
mode="${1:-full}"
gen_batch=8; (( is_3090 )) && gen_batch=4
limit=2000; ps="0,0.1,0.2,0.3,0.5,0.7,1.0"
[[ "$mode" == smoke ]] && { limit=16; ps="0,1.0"; }
cd src/prototype_grounding
python degraded_oracle_eval.py --checkpoint "$cluster_oracle" --tag "${mode}_cfcluster150" --ps "$ps" \
  --plugin "../../${tables}/group_pred150_${build_tag}.pt" --limit "$limit" --gen_batch_size "$gen_batch"
echo "[$(date --iso-8601=seconds)] done"
