#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_folds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# User-fold SASRec models for leakage-free training-row candidates (src/collab_reasoning/train_sasrec_folds.py).
source scripts/collab_env.sh
python src/collab_reasoning/train_sasrec_folds.py --folds 4 --out_dir "$fold_dir" --device cuda --epochs "${EPOCHS:-500}"
echo "[$(date --iso-8601=seconds)] done: $fold_dir/manifest.json"
