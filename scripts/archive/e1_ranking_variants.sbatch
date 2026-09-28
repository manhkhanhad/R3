#!/usr/bin/env bash
#SBATCH --job-name=R3_E1_rank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=12:00:00
#SBATCH --output=analysis/logs/E1_ranking_slurm.%j.log

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1

for mode in mean shuffle_user shuffle_target noise_0.25 noise_0.5 noise_1.0; do
  python src/prototype_grounding/experiment.py generate \
    --split test --bank all --k 32 --seed 42 --mode "$mode" \
    --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16
done
