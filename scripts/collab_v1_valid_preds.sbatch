#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_v1valid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Top-10 predictions of plain R3 v1 on the validation split: the residual scorer is trained on its misses.
source scripts/collab_env.sh
gen_batch=8; (( is_3090 )) && gen_batch=4
cd src/prototype_grounding
python collab_eval.py --checkpoint "$v1" --tag v1_valid_predictions --mode plain --split valid --predict_only \
  --gen_batch_size "$gen_batch"
echo "[$(date --iso-8601=seconds)] done: $v1_valid_preds/predictions.json"
