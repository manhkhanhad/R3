#!/usr/bin/env bash
#SBATCH --job-name=R3_histpool
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# History-only pooling (user design 2026-11-04): plain R3 whose r pools only over the
# history title tokens (--pool_history_only, src/latent/history_pooling.py); prompt,
# first pass and recommendation pass (full attention) unchanged. Recipe = R3_ori Toys
# (base Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs, early stopping, best epoch).
# Then: pooling check (weight on non-title tokens = 0), constrained 10-beam ranking,
# r interventions raw / mask / mean / shuffle_user (experiment.py), paired bootstrap.
# Env: CODE_ROOT (required), CATEGORY, NUM_EPOCHS, OUTPUT_ROOT, OUTPUT_NAME, TRAIN_SAMPLE,
#      VALID_SAMPLE, TEST_SAMPLE, EXTRACT_LIMIT, INTERVENTION_LIMIT, MODES, SKIP_TRAIN.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
category="${CATEGORY:-Toys_and_Games}"
num_epochs="${NUM_EPOCHS:-3}"
name="${OUTPUT_NAME:-${category}_history_pooling}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,history-pooling}"
export R3_ROOT="$PWD"

code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] $name: history-only pooling, GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-output_dir}/${name}"
out="${task_dir}/latent"
tag="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
interv_dir="analysis/results/intervention/${tag}"
mkdir -p "$out"

if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then  # smoke runs: small validation file
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  echo "[$(date --iso-8601=seconds)] training"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "$num_epochs" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model False --save_total_limit 3 \
    --pool_history_only True "${resume[@]}"
fi
echo "[$(date --iso-8601=seconds)] pooling check"
python "$code_root/scripts/history_pooling_check.py" --checkpoint "$out" --test_file "$test_file" --category "$category" --rows 64

echo "[$(date --iso-8601=seconds)] standard ranking (constrained 10-beam)"
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/eval_raw.json"
cp "$task_dir/eval_raw.json" "$out/final_result.json"
python "$code_root/src/utils/calc.py" --path "$task_dir/eval_raw.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] r interventions: extract train latents (mean / shuffle bank)"
python "$code_root/src/prototype_grounding/experiment.py" extract --split train --batch-size 16 \
  --limit "${EXTRACT_LIMIT:--1}" --checkpoint "$out" --tag "$tag" --category "$category"
modes="${MODES:-raw mask mean shuffle_user}"
for mode in $modes; do
  echo "[$(date --iso-8601=seconds)] intervention mode=$mode"
  python "$code_root/src/prototype_grounding/experiment.py" generate --split test \
    --bank all --k 32 --seed 42 --mode "$mode" --topk 1 \
    --temperature 0.1 --alpha 1.0 --batch-size 16 --limit "${INTERVENTION_LIMIT:--1}" \
    --checkpoint "$out" --tag "$tag" --category "$category"
done
python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw $(echo "$modes" | sed 's/\braw\b//')
echo "[$(date --iso-8601=seconds)] $name complete"
