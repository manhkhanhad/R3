#!/usr/bin/env bash
#SBATCH --job-name=R3_MI_from_SFT519_e2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
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
export WANDB_NAME="${WANDB_NAME:-R3_Musical_from_SFT519_e2}"
export WANDB_TAGS="${WANDB_TAGS:-plain-r3,Musical_Instruments,sft-init}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
eval_batch_size=8
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
  eval_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size; evaluation batch size: $eval_batch_size"

category="Musical_Instruments"
base_data_dir="data/${category}"
base_checkpoint="/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/Musical_Instruments_plain_sft/model/checkpoint-519"
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3_from_SFT519_e2${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
train_file="${base_data_dir}/train/Musical_Instruments_5_2015-1-2018-11.csv"
valid_file="${base_data_dir}/valid/Musical_Instruments_5_2015-1-2018-11.csv"
test_file="${base_data_dir}/test/Musical_Instruments_5_2015-1-2018-11.csv"
info_file="${base_data_dir}/info/Musical_Instruments_5_2015-1-2018-11.txt"

if [[ ! -s "$base_checkpoint/model.safetensors" || ! -s "$base_checkpoint/tokenizer.json" ]]; then
  echo "Missing SFT initialization checkpoint: $base_checkpoint" >&2
  exit 1
fi
mkdir -p "$out"

echo "[$(date --iso-8601=seconds)] Starting two new R3 epochs from $base_checkpoint"
python src/latent/latent_attention_train.py \
  --base_model "$base_checkpoint" \
  --train_file "$train_file" --eval_file "$valid_file" \
  --output_dir "$out" --category "$category" \
  --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs 2 --cutoff_len 512 --learning_rate 3e-4

echo "[$(date --iso-8601=seconds)] Training finished; starting ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python src/latent/latent_attention_eval.py \
  --batch_size "$eval_batch_size" --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
  --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] Musical Instruments R3 from SFT-519 run complete"
