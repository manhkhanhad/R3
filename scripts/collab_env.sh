#!/usr/bin/env bash
# Shared environment and paths for the collaborative-reasoning jobs
# (document/collaborative_reasoning_experiments.md). Sourced by scripts/collab_*.sbatch.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/cornac-matplotlib-$USER
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"

category=Musical_Instruments
data=data/${category}
train_file=${data}/train/Musical_Instruments_5_2015-1-2018-11.csv
eval_file=${data}/valid/Musical_Instruments_5_2015-1-2018-11.csv
scratch=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3
results=/common/home/users/m/mk.ngohuu.2024/research/R3/analysis/results/collab_reasoning
fold_dir=${scratch}/collab_reasoning/sasrec_folds
v1=${scratch}/output_dir/Musical_Instruments_v1/latent
v1_valid_preds=${results}/eval/v1_valid_predictions
tables=${data}/info/collab_reasoning
build_tag="${BUILD_TAG:-full}"
cluster_oracle=${scratch}/output_dir/Musical_Instruments_R3_side_oracle_cfcluster150/latent

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
is_3090=0; [[ "$gpu_name" == *"3090"* ]] && is_3090=1
echo "[$(date --iso-8601=seconds)] host=$(hostname) GPU=${gpu_name:-none} job=${SLURM_JOB_ID:-local}"
