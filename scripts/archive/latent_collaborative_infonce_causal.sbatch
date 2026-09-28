#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_causal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=analysis/logs/R3_collaborative_infonce_causal/job.%j.out
#SBATCH --error=analysis/logs/R3_collaborative_infonce_causal/job.%j.err

# Causal ranking interventions for the completed plain-R3 collaborative
# InfoNCE checkpoint. Projection heads and the training-only FIFO are ignored;
# interventions act on the deployed raw reasoning latent r.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

echo "[$(date --iso-8601=seconds)] Starting collaborative-InfoNCE causal analysis (job ${SLURM_JOB_ID:-local})"

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
result_dir="analysis/results/intervention/${tag}"
log_dir="analysis/logs/R3_collaborative_infonce_causal"

mkdir -p "$result_dir" "$log_dir"

if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "Missing completed checkpoint: $checkpoint/model.safetensors" >&2
  exit 1
fi

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
echo "[$(date --iso-8601=seconds)] Collaborative-InfoNCE causal interventions complete"
