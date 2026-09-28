#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_MI_title_cf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
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
export WANDB_NAME="${WANDB_NAME:-R3_collab_MI_title_cf_from_v0_e2}"
export WANDB_TAGS="${WANDB_TAGS:-collaborative-title-cf,musical-instruments,v0-epoch2}"

category="Musical_Instruments"
source_checkpoint="$HOME/scratchDirectory/R3/output_dir/Musical_Instruments_v0/latent/checkpoint-1038"
task_dir="$HOME/scratchDirectory/R3/output_dir/Musical_Instruments_R3_collaborative_title_cf_from_v0_e2"
out="$task_dir/latent"
base_data_dir="data/$category"
train_file="$base_data_dir/train/Musical_Instruments_5_2015-1-2018-11.csv"
eval_file="$base_data_dir/valid/Musical_Instruments_5_2015-1-2018-11.csv"
test_file="$base_data_dir/test/Musical_Instruments_5_2015-1-2018-11.csv"
info_file="$base_data_dir/info/Musical_Instruments_5_2015-1-2018-11.txt"

test -s "$source_checkpoint/model.safetensors"
mkdir -p "$out"
sha256sum \
  "$source_checkpoint/model.safetensors" \
  src/latent/collaborative_pair_features.py \
  src/latent/collaborative_contrastive_dataset.py \
  src/latent/collaborative_contrastive_attention.py \
  src/latent/latent_attention_train_collaborative.py \
  scripts/latent_collaborative_title_cf_musical_full.sbatch \
  >"$task_dir/run_inputs.sha256"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
prefill_batch_size=8
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
  prefill_batch_size=4
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch: $micro_batch_size; prefill batch: $prefill_batch_size"

echo "[$(date --iso-8601=seconds)] Starting $category title+CF collaborative run from $source_checkpoint"
python src/latent/latent_attention_train_collaborative.py \
  --base_model "$source_checkpoint" \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs 1 --learning_rate 5e-5 --cutoff_len 512 \
  --projection_dim 256 --temperature 0.1 \
  --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
  --cf_contrastive_weight 0.1 --title_cf_positives True \
  --title_positive_floor 0.25 --cf_topk 32 \
  --cf_min_common 2 --cf_min_cosine 0.1 \
  --bank_size 10000 --max_user_positives 16 \
  --max_target_positives 16 --num_negatives 256 \
  --positive_min_token_accuracy 0.8 \
  --prefill_queue True --prefill_batch_size "$prefill_batch_size"

echo "[$(date --iso-8601=seconds)] Training finished; starting standard ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python src/latent/latent_attention_eval.py \
  --batch_size 8 --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
  --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] $category title+CF collaborative run complete"
