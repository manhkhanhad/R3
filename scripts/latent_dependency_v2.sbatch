#!/usr/bin/env bash
#SBATCH --job-name=R3_dep_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Latent dependency test v2 (intervention v2, user design 2026-11-05): per-sample,
# forward-only. y_hat = the checkpoint's own top-1; teacher-force it with r replaced by
# K latents drawn from a bank of the same checkpoint's train latents (+ zero vector).
# Extracts the bank first if it is missing.
# Env: CODE_ROOT (required), CHECKPOINT (required), TAG (required), PREDICTIONS (required),
#      CATEGORY, K, SEED, LIMIT, SHUFFLE_FILE, EXTRACT_LIMIT.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD"

code_root="${CODE_ROOT:?}"; ckpt="${CHECKPOINT:?}"; tag="${TAG:?}"; preds="${PREDICTIONS:?}"
category="${CATEGORY:-Toys_and_Games}"
echo "[$(date --iso-8601=seconds)] v2 tag=$tag ckpt=$ckpt GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ ! -f "analysis/evidence/${category}/${tag}/train/thoughts.pt" ]]; then
  echo "[$(date --iso-8601=seconds)] extracting train latent bank"
  python "$code_root/src/prototype_grounding/experiment.py" extract --split train --batch-size 16 \
    --limit "${EXTRACT_LIMIT:--1}" --checkpoint "$ckpt" --tag "$tag" --category "$category"
fi
extra=()
[[ -n "${SHUFFLE_FILE:-}" ]] && extra+=(--shuffle-file "$SHUFFLE_FILE")
[[ -n "${EXTRACT_LIMIT:-}" ]] && extra+=(--no-verify-bank)
python "$code_root/src/prototype_grounding/latent_dependency_v2.py" --checkpoint "$ckpt" --tag "$tag" \
  --predictions "$preds" --category "$category" --k "${K:-16}" --seed "${SEED:-42}" \
  --limit "${LIMIT:--1}" ${OUTPUT:+--output "$OUTPUT"} "${extra[@]}"
echo "[$(date --iso-8601=seconds)] v2 $tag complete"
