#!/usr/bin/env bash
#SBATCH --job-name=R3_E4_rank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=08:00:00
#SBATCH --output=analysis/logs/E4_ranking_slurm.%j.log

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1

for seed in 13 42 73; do
  python src/prototype_grounding/experiment.py generate \
    --split test --bank top --k 32 --seed "$seed" --mode residual \
    --topk 1 --temperature 0.1 --alpha 0.75 --batch-size 16
done
