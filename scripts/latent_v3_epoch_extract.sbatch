#!/usr/bin/env bash
#SBATCH --job-name=R3_v6_latent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=analysis/logs/latent_v6_epochs/extract.%A_%a.log

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

steps=(384 768 1152)
step="${steps[$SLURM_ARRAY_TASK_ID]}"
checkpoint="output_dir/CDs_and_Vinyl_v6/latent/checkpoint-$step"
tag="latent_v6_checkpoint_$step"

python src/prototype_grounding/experiment.py extract \
  --split train --batch-size 32 --checkpoint "$checkpoint" --tag "$tag"

python visualize_reasoning.py --latent_path analysis/evidence/CDs_and_Vinyl/$tag/train/thoughts.pt --metadata_path analysis/evidence/CDs_and_Vinyl/$tag/train/metadata.csv --output_path analysis/figures/$tag.html
