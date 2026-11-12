#!/usr/bin/env bash
#SBATCH --job-name=R3_router_ceiling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/multiAspectR3
#SBATCH --output=run_logs/%x.%j.out
# Step 0 of the aspect router (src/prototype_grounding/staged_refine_router_ceiling.py).
# Usage: sbatch scripts/staged_refine_router_ceiling.sbatch <run dir containing latent/> <tag> [extra args]
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
run="$1"; tag="$2"; shift 2
cd src/prototype_grounding
python staged_refine_router_ceiling.py --checkpoint "$run/latent" --tag "$tag" "$@"
