#!/usr/bin/env bash
#SBATCH --job-name=R3_intent_router
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Phase 2 (L6d): DivReason-style path aggregation on a phase-1 intent checkpoint.
# Router trained through the frozen decoder's loss; then ranking under aggregation modes.
# Env: PHASE1 (checkpoint dir), FIT_SPLIT=valid|train, FIT_SAMPLE, CATEGORY, CODE_ROOT,
#      OUTPUT_SUFFIX, NUM_EPOCHS, TAU, BETA, PRUNE, TEST_SAMPLE, MODES.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
category="${CATEGORY:-Musical_Instruments}"
fit_split="${FIT_SPLIT:-valid}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_intent_router_${category}_${fit_split}${OUTPUT_SUFFIX:-}}"
export WANDB_TAGS="${WANDB_TAGS:-intent-reasoning,path-aggregation,divreason}"

code_root="${CODE_ROOT:-.}"
phase1="${PHASE1:?set PHASE1 to a phase-1 intent checkpoint}"
echo "[$(date --iso-8601=seconds)] code root: $code_root; phase1: $phase1; fit split: $fit_split"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

data_dir="data/${category}"
fit_file=$(ls ${data_dir}/${fit_split}/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="output_dir/${category}_R3_intent_router_${fit_split}${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
mkdir -p "$out"

echo "[$(date --iso-8601=seconds)] Training router on ${fit_file}"
python "$code_root/src/latent/latent_attention_train_intent_router.py" \
  --base_model "$phase1" --fit_file "$fit_file" --output_dir "$out" --category "$category" \
  --fit_sample "${FIT_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 1e-3 \
  --tau "${TAU:-0.5}" --beta "${BETA:-0.5}" --prune "${PRUNE:-0.05}"

echo "[$(date --iso-8601=seconds)] Router trained; ranking under aggregation modes"
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-aggregate prior_blend aggregate_uniform}; do
  echo "[$(date --iso-8601=seconds)] mode=$mode"
  INTENT_MODE="$mode" CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
echo "[$(date --iso-8601=seconds)] ${category} intent router run complete"
