#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_geometry
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|h100nvl"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=analysis/logs/R3_collaborative_infonce_geometry/job.%j.out
#SBATCH --error=analysis/logs/R3_collaborative_infonce_geometry/job.%j.err

# Submit with --dependency=afterok:<training_job_id>. This job analyzes only
# the final, deployed plain-R3 latent r; training-only InfoNCE projection heads
# and the non-persistent FIFO corpus are intentionally not part of extraction.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

echo "[$(date --iso-8601=seconds)] Starting collaborative-InfoNCE geometry analysis (job ${SLURM_JOB_ID:-local})"

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="CDs_and_Vinyl"
checkpoint="output_dir/CDs_and_Vinyl_R3_collaborative_infonce/latent"
tag="latent_collaborative_infonce"
evidence="analysis/evidence/${category}/${tag}/train/thoughts.pt"
result="analysis/results/latent_distribution/${tag}/train_latent_distribution.json"
log_dir="analysis/logs/R3_collaborative_infonce_geometry"

mkdir -p "$log_dir" analysis/figures

if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "Missing completed checkpoint: $checkpoint/model.safetensors" >&2
  exit 1
fi

if [[ ! -s "$evidence" ]]; then
  echo "[$(date --iso-8601=seconds)] Extracting train latents for $tag"
  python src/prototype_grounding/experiment.py extract \
    --split train --batch-size 32 \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    >"$log_dir/extract_train.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Existing train evidence found; skipping extraction"
fi

if [[ ! -s "$result" ]]; then
  echo "[$(date --iso-8601=seconds)] Computing latent geometry for $tag"
  python src/prototype_grounding/experiment.py latent-distribution \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    --component r \
    >"$log_dir/latent_distribution.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Existing geometry result found; skipping analysis"
fi

echo "[$(date --iso-8601=seconds)] Geometry analysis complete: $result"
