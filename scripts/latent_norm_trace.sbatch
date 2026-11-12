#!/usr/bin/env bash
#SBATCH --job-name=R3_norm_trace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=01:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
cd src/prototype_grounding
python latent_norm_trace.py --limit 200 --out ../../analysis/results/decoder_sensitivity/latent_norm_trace.json --checkpoints \
  v0=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v0/latent/checkpoint-1557 \
  base_fresh_init=base \
  v0_ckpt519=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v0/latent/checkpoint-519 \
  v1=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_v1/latent \
  collaborative_infonce=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_R3_collaborative_infonce/latent \
  category_align=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_R3_category_align_gte_v1/latent
