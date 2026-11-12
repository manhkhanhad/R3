#!/usr/bin/env bash
#SBATCH --job-name=R3_cat_align_MI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40|a40|a100|h100|h100nvl|3090"
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
export WANDB_NAME="${WANDB_NAME:-R3_cat_align_gte_Musical}"
export WANDB_TAGS="${WANDB_TAGS:-category-alignment,gte-large-en-v1.5,Musical_Instruments}"

category=Musical_Instruments
base_data_dir="data/${category}"
train_file="${base_data_dir}/train/Musical_Instruments_5_2015-1-2018-11.csv"
eval_file="${base_data_dir}/valid/Musical_Instruments_5_2015-1-2018-11.csv"
test_file="${base_data_dir}/test/Musical_Instruments_5_2015-1-2018-11.csv"
info_file="${base_data_dir}/info/Musical_Instruments_5_2015-1-2018-11.txt"
category_embeddings_file="${base_data_dir}/info/Musical_Instruments_gte_large_en_v1_5_categories.pt"
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_R3_category_align_gte_v1"
model_dir="${task_dir}/latent"

test -s "$category_embeddings_file"
mkdir -p "$model_dir"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
eval_batch_size=8
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
  eval_batch_size=4
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$model_dir" --category "$category" \
  --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs 3 --cutoff_len 512 \
  --category_embeddings_file "$category_embeddings_file" \
  --category_alignment_weight 0.1

echo "[$(date --iso-8601=seconds)] Training finished; starting standard ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
python src/latent/latent_attention_eval.py \
  --batch_size "$eval_batch_size" --base_model "$model_dir" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
  --output_path "$model_dir/final_result.json" --nn 1
python src/utils/calc.py --path "$model_dir/final_result.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] Musical Instruments category-alignment run complete"
