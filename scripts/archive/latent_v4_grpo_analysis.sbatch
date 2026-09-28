#!/usr/bin/env bash
#SBATCH --job-name=R3_v4_grpo_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --output=analysis/logs/latent_v4_grpo/job.%j.log

# Latent-collapse geometry + E1 causal-intervention ranking for the post-GRPO
# attention-only fine-tune (nomul_attn_lr_5e-5), to extend
# analysis/document/latent_v4_epoch_collapse_report.md with the RL stage.

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

checkpoint="output_dir/CDs_and_Vinyl_v4/nomul_attn_lr_5e-5"
tag="latent_v4_grpo_lr5e-5"
result_dir="analysis/results/intervention/$tag"
log_dir="analysis/logs/latent_v4_grpo"

mkdir -p "$result_dir" "$log_dir"

if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "Missing checkpoint: $checkpoint/model.safetensors" >&2
  exit 1
fi

if [[ ! -s "analysis/evidence/CDs_and_Vinyl/$tag/train/thoughts.pt" ]]; then
  echo "[$(date --iso-8601=seconds)] Extracting train evidence for $tag"
  python src/prototype_grounding/experiment.py extract \
    --split train --batch-size 32 --checkpoint "$checkpoint" --tag "$tag" \
    >"$log_dir/extract_train.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Train evidence already present for $tag, skipping extract"
fi

echo "[$(date --iso-8601=seconds)] Computing latent-distribution geometry for $tag"
python src/prototype_grounding/experiment.py latent-distribution \
  --tag "$tag" --checkpoint "$checkpoint" \
  >"$log_dir/latent_distribution.log" 2>&1

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
    --checkpoint "$checkpoint" --tag "$tag" \
    >"$log_dir/ranking_${mode}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $tag / $mode"
done

date --iso-8601=seconds >"$result_dir/RANKING_COMPLETE"
echo "[$(date --iso-8601=seconds)] E1 ranking + geometry complete for $tag"
