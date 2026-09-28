#!/usr/bin/env bash
#SBATCH --job-name=R3_side_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Usage: sbatch scripts/side_info_eval_only.sbatch <checkpoint> <tag> [extra side_info_eval.py args...]
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
ckpt="$1"; tag="$2"; shift 2
cd src/prototype_grounding
python side_info_eval.py --checkpoint "$ckpt" --tag "$tag" "$@"
