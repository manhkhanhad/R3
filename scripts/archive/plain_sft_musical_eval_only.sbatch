#!/usr/bin/env bash
#SBATCH --job-name=R3_plain_SFT_MI_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_plain_SFT_Musical}"
export WANDB_TAGS="${WANDB_TAGS:-plain-sft,Musical_Instruments,evaluation-only}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
eval_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
  eval_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; evaluation batch size: $eval_batch_size"

category="Musical_Instruments"
base_data_dir="data/${category}"
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_plain_sft${OUTPUT_SUFFIX:-}"
model_dir="${MODEL_DIR:-${task_dir}/model}"
if [[ ! -s "$model_dir/model.safetensors" ]]; then
  echo "Missing trained model: $model_dir/model.safetensors" >&2
  exit 1
fi

python src/latent/plain_sft_eval.py \
  --base_model "$model_dir" \
  --info_file "${base_data_dir}/info/Musical_Instruments_5_2015-1-2018-11.txt" \
  --test_data_path "${base_data_dir}/test/Musical_Instruments_5_2015-1-2018-11.csv" \
  --result_json_data "${RESULT_JSON_DATA:-${task_dir}/final_result.json}" \
  --sample "${EVAL_SAMPLE:--1}" --batch_size "$eval_batch_size"
