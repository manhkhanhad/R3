#!/usr/bin/env bash
#SBATCH --job-name=R3_pair_sim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Usage: CODE_ROOT=<frozen copy> sbatch scripts/latent_pair_similarity.sbatch <checkpoint> <tag> [category] [limit]
# Extracts r / h_last / h_mean for train and test (skipped if already saved), then reports.
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export R3_ROOT=/common/home/users/m/mk.ngohuu.2024/research/R3

checkpoint="$1"; tag="$2"; category="${3:-Toys_and_Games}"; limit="${4:--1}"
code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
echo "[$(date --iso-8601=seconds)] pair similarity: $checkpoint -> $tag ($category, limit $limit, code $code)"
cd "$code/src/prototype_grounding"
for split in test train; do
  if [[ -f "$R3_ROOT/output_dir/latent_pair_similarity/$tag/$split/summary.json" ]]; then
    echo "skip extract $split (exists)"; continue
  fi
  python latent_pair_similarity.py extract --checkpoint "$checkpoint" --tag "$tag" \
    --category "$category" --split "$split" --limit "$limit" --batch-size 8
done
python latent_pair_similarity.py report --tag "$tag" --category "$category" --splits train test
echo "[$(date --iso-8601=seconds)] done"
