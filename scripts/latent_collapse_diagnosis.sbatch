#!/usr/bin/env bash
#SBATCH --job-name=R3_collapse_diag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Usage: CODE_ROOT=<frozen copy> sbatch scripts/latent_collapse_diagnosis.sbatch <checkpoint> <tag> [category] [limit]
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export R3_ROOT=/common/home/users/m/mk.ngohuu.2024/research/R3

checkpoint="$1"; tag="$2"; category="${3:-Toys_and_Games}"; limit="${4:-2000}"
code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
cd "$code/src/prototype_grounding"
python latent_collapse_diagnosis.py --checkpoint "$checkpoint" --tag "$tag" --category "$category" --limit "$limit"
