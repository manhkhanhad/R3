#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=16:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Complementarity screen (idea 1), fused/residual CF scorers (idea 4), group predictor (idea 5)
# and the candidate tables used by scripts/collab_train.sbatch.
source scripts/collab_env.sh
export COLLAB_VERBOSE=1
python src/collab_reasoning/build_candidates.py --fold_manifest "$fold_dir/manifest.json" \
  --valid_preds "$v1_valid_preds" --tag "$build_tag"
echo "[$(date --iso-8601=seconds)] done: ${results}/${build_tag}/screen.md"
