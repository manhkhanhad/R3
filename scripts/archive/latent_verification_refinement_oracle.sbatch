#!/usr/bin/env bash
#SBATCH --job-name=R3_refine_oracle
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/latent_verification_refinement_oracle.py \
  --checkpoint "$HOME/scratchDirectory/R3/output_dir/CDs_and_Vinyl_v4/latent/checkpoint-1152" \
  --limit 256 --seed 42 --min-accuracy 0.8 --selection similarity
