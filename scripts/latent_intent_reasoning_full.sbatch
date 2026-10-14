#!/usr/bin/env bash
#SBATCH --job-name=R3_intent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Disentangled-interest R3: K reasoning paths per history; the decoder is trained on the
# path the item-aware posterior credits. Ranking under intent modes on one checkpoint.
# Env: CATEGORY, CODE_ROOT, OUTPUT_SUFFIX, N_INTENTS, MARGIN_WEIGHT, TRAIN_SAMPLE,
#      EVAL_SAMPLE, TEST_SAMPLE, NUM_EPOCHS, MODES.

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
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_intent_${category}${OUTPUT_SUFFIX:-}}"
export WANDB_TAGS="${WANDB_TAGS:-intent-reasoning,disentangled-interest}"

code_root="${CODE_ROOT:-.}"
echo "[$(date --iso-8601=seconds)] code root: $code_root"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
item_emb_file="${data_dir}/info/${category}_qwen3emb_1536.pt"
task_dir="output_dir/${category}_R3_intent${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
mkdir -p "$out"

echo "[$(date --iso-8601=seconds)] Starting ${category} intent reasoning training"
python "$code_root/src/latent/latent_attention_train_intent.py" \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" --item_emb_file "$item_emb_file" \
  --output_dir "$out" --category "$category" \
  --sample "${TRAIN_SAMPLE:--1}" --eval_sample "${EVAL_SAMPLE:--1}" \
  --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs "${NUM_EPOCHS:-3}" --cutoff_len 512 \
  --n_intents "${N_INTENTS:-4}" --margin_weight "${MARGIN_WEIGHT:-0}" \
  --mixture_weight 0.2 --prior_weight 0.1 --balance_weight 0.01 --explore_epochs 0.1

echo "[$(date --iso-8601=seconds)] Training finished; ranking under intent modes"
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-mixture map oracle random r0}; do
  echo "[$(date --iso-8601=seconds)] mode=$mode"
  INTENT_MODE="$mode" CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
echo "[$(date --iso-8601=seconds)] ${category} intent reasoning run complete"
