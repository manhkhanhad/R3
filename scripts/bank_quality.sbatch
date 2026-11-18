#!/usr/bin/env bash
#SBATCH --job-name=R3_bank_quality
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|l40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Latent-bank quality (src/prototype_grounding/bank_quality.py): coverage, geometry and swap-in
# usefulness of same-target / same-user / random / mean latents on a trained checkpoint.
# Env: CODE_ROOT (required), CHECKPOINT (required), TAG (required), CATEGORY, LIMIT, N_ANCHORS, K_POS, TAU.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD"
code_root="${CODE_ROOT:?}"
echo "[$(date --iso-8601=seconds)] bank quality: ckpt=${CHECKPOINT:?} tag=${TAG:?} GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python "$code_root/src/prototype_grounding/bank_quality.py" --checkpoint "$CHECKPOINT" --tag "$TAG" \
  --category "${CATEGORY:-Toys_and_Games}" --limit "${LIMIT:-10000}" --n_anchors "${N_ANCHORS:-800}" \
  --k_pos "${K_POS:-4}" --tau "${TAU:-0.1}" --batch_size 8
echo "[$(date --iso-8601=seconds)] done"
