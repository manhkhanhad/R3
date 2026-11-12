#!/usr/bin/env bash
#SBATCH --job-name=R3_decoder_sens
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=08:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Usage: sbatch scripts/decoder_sensitivity_musical.sbatch <checkpoint> <tag> [limit]
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1

checkpoint="$1"; tag="$2"; limit="${3:-2000}"
echo "[$(date --iso-8601=seconds)] decoder sensitivity: $checkpoint -> $tag (limit $limit)"
cd src/prototype_grounding
python decoder_sensitivity.py --checkpoint "$checkpoint" --tag "$tag" \
  --category Musical_Instruments --split test --limit "$limit" --batch-size 4
echo "[$(date --iso-8601=seconds)] done"
