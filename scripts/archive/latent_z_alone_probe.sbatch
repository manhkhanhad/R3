#!/usr/bin/env bash
#SBATCH --job-name=R3_z_alone_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=03:00:00
#SBATCH --output=run_logs/%x.%j.out

# mu/z decomposition probe: r_i = mu + z_i.
# Adds two new E1 intervention modes not in the original 8-mode suite:
#   z_alone       - the mu-free residual z_i = r_i - mu, re-normed to the typical
#                   raw-r norm so magnitude mismatch isn't misread as a content effect.
#   noise_matched - null control: same fixed norm as z_alone, but a direction
#                   uncorrelated with r_i. The z_alone vs. noise_matched gap isolates
#                   whether z_i's *direction* carries decodable content, controlling
#                   for the "missing mu scaffold" penalty that should hit both equally.
#
# Run against the three checkpoints currently on disk that span the collapse
# spectrum (margin=0.2 and the v4 baseline checkpoints have since been deleted
# from output_dir/ and are not available for this probe without retraining):
#   - scheduled_alpha075 checkpoint-1152: the cleanest working checkpoint (genuine
#     content-dependence, no activation-dependence confound)
#   - causal (L_Zdrop) checkpoint-768: activation-dependence WITHOUT content-dependence
#   - gradmask checkpoint-1152: the most severe geometric collapse measured (cosine 0.9998)
#
# Reuses each tag's already-extracted train evidence; only runs the two new modes.

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
  [latent_scheduled_alpha075_checkpoint_1152]="output_dir/CDs_and_Vinyl_scheduled_alpha075/latent/checkpoint-1152"
  [latent_causal_checkpoint_768]="output_dir/CDs_and_Vinyl_R3_causal/latent/checkpoint-768"
  [latent_gradmask_checkpoint_1152]="output_dir/CDs_and_Vinyl_R3_gradmask/latent/checkpoint-1152"
)

artifact_is_complete() {
  local artifact="$1" checkpoint="$2"
  [[ -s "$artifact" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics", {}); raise SystemExit(0 if m.get("n") == 6157 and m.get("checkpoint") == sys.argv[2] else 1)' \
    "$artifact" "$checkpoint"
}

for tag in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$tag]}"
  result_dir="analysis/results/intervention/$tag"
  log_dir="analysis/logs/z_alone_probe/$tag"
  mkdir -p "$result_dir" "$log_dir"

  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    echo "Missing checkpoint: $checkpoint/model.safetensors" >&2
    exit 1
  fi

  for mode in z_alone noise_matched; do
    output="$result_dir/generation_test_all_k32_s42_${mode}_t1_temp0.1_a1.0.json"
    if artifact_is_complete "$output" "$checkpoint"; then
      echo "[$(date --iso-8601=seconds)] Skipping completed $tag / $mode"
      continue
    fi

    echo "[$(date --iso-8601=seconds)] Starting $tag / $mode"
    python src/prototype_grounding/experiment.py generate \
      --split test --bank all --k 32 --seed 42 --mode "$mode" \
      --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
      --checkpoint "$checkpoint" --tag "$tag" \
      >"$log_dir/ranking_${mode}.log" 2>&1
    echo "[$(date --iso-8601=seconds)] Completed $tag / $mode"
  done
done

echo "[$(date --iso-8601=seconds)] z_alone probe complete for all tags"
