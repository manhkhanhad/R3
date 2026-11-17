#!/usr/bin/env bash
#SBATCH --job-name=R3_Goodreads
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out
# Plain R3 (stage-1 latent SFT + ranking eval) on Goodreads.
# Runs from frozen code: CODE_DIR (git archive of a commit). Outputs go to scratch.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

REPO=/common/home/users/m/mk.ngohuu.2024/research/R3
CODE_DIR="${CODE_DIR:?set CODE_DIR to the frozen code dir}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}"
cd "$CODE_DIR"
echo "[$(date --iso-8601=seconds)] code: $CODE_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_Goodreads}"
export WANDB_TAGS="${WANDB_TAGS:-r3-baseline,goodreads}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

category="Goodreads"
name="Goodreads_5_2017-1-2017-12"
data="${REPO}/data/${category}"
train_file="${data}/train/${name}.csv"
eval_file="${data}/valid/${name}.csv"
test_file="${data}/test/${name}.csv"
info_file="${data}/info/${name}.txt"
task_dir="${OUTPUT_ROOT}/${category}_R3${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ -n "${SKIP_TRAIN:-}" ]]; then
  test -f "$out/model.safetensors" || { echo "SKIP_TRAIN set but no model in $out" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] SKIP_TRAIN: evaluating existing model in $out"
else
echo "[$(date --iso-8601=seconds)] Starting ${category} R3 training"
python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs 3 --cutoff_len 512 \
  --save_total_limit 1
fi

echo "[$(date --iso-8601=seconds)] Training finished; starting ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python src/latent/latent_attention_eval.py \
  --batch_size 8 --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
  --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file" \
  | tee "$out/metric.log"

mkdir -p "${REPO}/output_dir/${category}_R3${OUTPUT_SUFFIX:-}"
cp "$out/final_result.json" "$out/metric.log" "${REPO}/output_dir/${category}_R3${OUTPUT_SUFFIX:-}/"
echo "[$(date --iso-8601=seconds)] ${category} R3 run complete"
