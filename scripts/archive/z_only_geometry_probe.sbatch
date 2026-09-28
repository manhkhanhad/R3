#!/usr/bin/env bash
#SBATCH --job-name=R3_z_only_geometry
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=02:00:00
#SBATCH --output=run_logs/%x.%j.out

# Re-extract (adds z_only.pt as a side effect -- see experiment.py::extract,
# free since model.attention.last_z is already computed inside
# thought_embeds()) and run the geometry report on z alone for the mu/z
# checkpoints still on disk. mu's norm is architecturally fixed
# (VAEDecomposedSelfAttentionLayer), so geometry run on the combined r vector
# conflates mu's fixed contribution with z's own (possibly collapsed)
# variance; this isolates z directly. Read-only w.r.t. the checkpoints;
# thoughts.pt gets rewritten with identical (deterministic, eval-mode)
# values, harmless.

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A checkpoints=(
  [latent_vae7b_nokl_checkpoint_1152]="output_dir/CDs_and_Vinyl_R3_vae7b_nokl/latent/checkpoint-1152"
  [latent_vae7b_a09_checkpoint_1152]="output_dir/CDs_and_Vinyl_R3_vae7b_a09/latent/checkpoint-1152"
  [latent_vae7b_nocurriculum_checkpoint_1152]="output_dir/CDs_and_Vinyl_R3_vae7b_nocurriculum/latent/checkpoint-1152"
)

for tag in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$tag]}"
  log_dir="analysis/logs/z_only_geometry/$tag"
  mkdir -p "$log_dir"

  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    echo "Missing checkpoint: $checkpoint/model.safetensors, skipping $tag" >&2
    continue
  fi

  echo "[$(date --iso-8601=seconds)] Re-extracting (with z_only.pt) for $tag"
  python src/prototype_grounding/experiment.py extract \
    --split train --batch-size 32 --checkpoint "$checkpoint" --tag "$tag" \
    >"$log_dir/extract_train.log" 2>&1

  echo "[$(date --iso-8601=seconds)] Running z-only latent-distribution for $tag"
  python src/prototype_grounding/experiment.py latent-distribution \
    --checkpoint "$checkpoint" --tag "$tag" --component z \
    >"$log_dir/latent_distribution_z.log" 2>&1
  echo "[$(date --iso-8601=seconds)] Done $tag"
done

echo "[$(date --iso-8601=seconds)] z-only geometry probe complete"
