#!/usr/bin/env bash
#SBATCH --job-name=R3_vae7c_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --output=analysis/logs/R3_vae7c_verified_analysis/job.%j.out
#SBATCH --error=analysis/logs/R3_vae7c_verified_analysis/job.%j.err

# Submit with --dependency=afterok:<7c-training-job>. Runs geometry for both
# the deployed r=mu+z and z alone, followed by the standard eight-mode causal
# ranking intervention on the final normal-size student checkpoint.

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="CDs_and_Vinyl"
checkpoint="output_dir/${category}_R3_vae7c_verified/latent"
tag="latent_vae7c_verified"
evidence="analysis/evidence/${category}/${tag}/train/thoughts.pt"
z_evidence="analysis/evidence/${category}/${tag}/train/z_only.pt"
r_geometry="analysis/results/latent_distribution/${tag}/train_latent_distribution.json"
z_geometry="analysis/results/latent_distribution/${tag}/train_latent_distribution_z.json"
result_dir="analysis/results/intervention/${tag}"
log_dir="analysis/logs/R3_vae7c_verified_analysis"

mkdir -p "$result_dir" "$log_dir" analysis/figures

if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "Missing completed 7c checkpoint: $checkpoint/model.safetensors" >&2
  exit 1
fi

if [[ ! -s "$evidence" || ! -s "$z_evidence" ]]; then
  echo "[$(date --iso-8601=seconds)] Extracting train r and z for $tag"
  python src/prototype_grounding/experiment.py extract \
    --split train --batch-size 32 \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    >"$log_dir/extract_train.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Existing r/z evidence found; skipping extraction"
fi

if [[ ! -s "$r_geometry" ]]; then
  echo "[$(date --iso-8601=seconds)] Computing combined-r geometry"
  python src/prototype_grounding/experiment.py latent-distribution \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    --component r \
    >"$log_dir/latent_distribution_r.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Existing r geometry found; skipping"
fi

if [[ ! -s "$z_geometry" ]]; then
  echo "[$(date --iso-8601=seconds)] Computing z-only geometry"
  python src/prototype_grounding/experiment.py latent-distribution \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    --component z \
    >"$log_dir/latent_distribution_z.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Existing z geometry found; skipping"
fi

python visualize_reasoning.py \
  --latent_path "$evidence" \
  --metadata_path "analysis/evidence/${category}/${tag}/train/metadata.csv" \
  --output_path "analysis/figures/${tag}.html" \
  >"$log_dir/visualize.log" 2>&1 \
  || echo "visualize_reasoning.py failed (non-fatal), see $log_dir/visualize.log"

artifact_is_complete() {
  local artifact="$1"
  [[ -s "$artifact" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics", {}); raise SystemExit(0 if m.get("n") == 6157 and m.get("checkpoint") == sys.argv[2] else 1)' \
    "$artifact" "$checkpoint"
}

modes=(raw mask mean shuffle_user shuffle_target noise_0.25 noise_0.5 noise_1.0)
for mode in "${modes[@]}"; do
  output="$result_dir/generation_test_all_k32_s42_${mode}_t1_temp0.1_a1.0.json"
  if artifact_is_complete "$output"; then
    echo "[$(date --iso-8601=seconds)] Skipping completed $tag / $mode"
    continue
  fi

  echo "[$(date --iso-8601=seconds)] Starting $tag / $mode"
  python src/prototype_grounding/experiment.py generate \
    --split test --bank all --k 32 --seed 42 --mode "$mode" \
    --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" \
    >"$log_dir/ranking_${mode}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $tag / $mode"
done

date --iso-8601=seconds >"$result_dir/RANKING_COMPLETE"
echo "[$(date --iso-8601=seconds)] 7c r/z geometry and causal intervention complete"
