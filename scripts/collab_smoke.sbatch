#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=6:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# End-to-end smoke test of every GPU path with tiny limits, on the padded smoke tables
# (BUILD_TAG=smoke_padded). The full jobs depend on this one (afterok). Smoke numbers are not results.
export BUILD_TAG=smoke_padded
source scripts/collab_env.sh
(
  cd src/prototype_grounding
  gen_batch=8; (( is_3090 )) && gen_batch=4
  python collab_eval.py --checkpoint "$v1" --tag smoke_v1_valid_predictions --mode plain --split valid \
    --predict_only --predict_limit 16 --gen_batch_size "$gen_batch"
)
for variant in cf_slots cf_text group_pred residual_slots; do
  bash scripts/collab_train.sbatch "$variant" smoke
done
bash scripts/collab_degraded_oracle.sbatch smoke
echo "[$(date --iso-8601=seconds)] smoke passed"
