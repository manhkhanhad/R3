#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_margin_Musical
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=36:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Epoch 1: CE + margin (Attempt 2), FIFO corpus filling but no InfoNCE.
# Epochs 2-3: CE + margin + collaborative user/target InfoNCE (Attempt 10).
# CODE_ROOT lets the job run a frozen code snapshot; defaults to this checkout.

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
export WANDB_NAME="${WANDB_NAME:-R3_collab_margin_Musical}"
export WANDB_TAGS="${WANDB_TAGS:-collaborative-infonce,margin,two-phase,cross-dataset}"

code_root="${CODE_ROOT:-.}"
echo "[$(date --iso-8601=seconds)] code root: $code_root"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

category="Musical_Instruments"
base_data_dir="data/${category}"
train_file="${base_data_dir}/train/Musical_Instruments_5_2015-1-2018-11.csv"
eval_file="${base_data_dir}/valid/Musical_Instruments_5_2015-1-2018-11.csv"
test_file="${base_data_dir}/test/Musical_Instruments_5_2015-1-2018-11.csv"
info_file="${base_data_dir}/info/Musical_Instruments_5_2015-1-2018-11.txt"
task_dir="output_dir/${category}_R3_collab_margin${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
mkdir -p "$out"

echo "[$(date --iso-8601=seconds)] Starting ${category} margin -> collaborative InfoNCE training"
python "$code_root/src/latent/latent_attention_train_collaborative_margin.py" \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs 3 --cutoff_len 512 \
  --margin_only_epochs "${MARGIN_ONLY_EPOCHS:-1}" \
  --margin 0.2 --margin_weight 1.0 \
  --phase2_margin_weight "${PHASE2_MARGIN_WEIGHT:-1.0}" \
  --projection_dim 256 --temperature 0.1 \
  --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
  --bank_size 10000 --max_user_positives 16 \
  --max_target_positives 16 --num_negatives 256 \
  --positive_min_token_accuracy "${POSITIVE_MIN_TOKEN_ACCURACY:-1.0}"

echo "[$(date --iso-8601=seconds)] Training finished; starting standard ranking evaluation"
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/0.json"
python "$code_root/src/utils/merge.py" --input_path "$task_dir/" \
  --output_path "$out/final_result.json" --nn 1
python "$code_root/src/utils/calc.py" --path "$out/final_result.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] ${category} margin -> collaborative InfoNCE run complete"
