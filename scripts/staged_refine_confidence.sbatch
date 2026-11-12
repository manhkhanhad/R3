#!/usr/bin/env bash
#SBATCH --job-name=R3_refine_conf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=02:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/multiAspectR3
#SBATCH --output=run_logs/%x.%j.out
# Usage: sbatch scripts/staged_refine_confidence.sbatch <run name, e.g. refine_centroid_w1.0>
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
d=/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/Musical_Instruments_R3_$1
cd src/prototype_grounding
python staged_refine_confidence.py --checkpoint "$d/latent" --test_json "$d/0.json" --tag "full_$1"
