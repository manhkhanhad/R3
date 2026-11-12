#!/usr/bin/env bash
#SBATCH --job-name=R3_category_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=02:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Usage: sbatch scripts/category_probe_musical.sbatch <checkpoint> <tag> [limit]
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
cd src/prototype_grounding
python category_probe.py --checkpoint "$1" --tag "$2" --limit "${3:-2000}" \
  --category_embeddings ../../data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt
