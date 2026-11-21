#!/usr/bin/env bash
#SBATCH --job-name=R3_eval_exhist
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Ranking evaluation of a trained R3-family latent model with each user's history titles excluded from the
# constrained beam search (latent_attention_eval.py --exclude_history True). Read-only on the model dir.
# Env: CODE_ROOT (required), CATEGORY (required), MODEL_DIR (required, the latent/ dir), TAG (required).
# Output: output_dir/exclude_history/<CATEGORY>/<TAG>.json (+ metric log).
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD" WANDB_MODE="${WANDB_MODE:-offline}"
code="${CODE_ROOT:?}"; category="${CATEGORY:?}"; model="${MODEL_DIR:?}"; tag="${TAG:?}"
test_file=$(ls data/${category}/test/${category}_5_*.csv); info_file=$(ls data/${category}/info/${category}_5_*.txt)
out_dir="output_dir/exclude_history/${category}"; mkdir -p "$out_dir"
echo "[$(date --iso-8601=seconds)] $tag: $model (exclude_history)"
CUDA_VISIBLE_DEVICES=0 python "$code/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$model" --sample -1 --info_file "$info_file" --category "$category" \
  --test_data_path "$test_file" --result_json_data "$out_dir/${tag}.json" --exclude_history True
python "$code/src/utils/calc.py" --path "$out_dir/${tag}.json" --item_path "$info_file" | tee "$out_dir/${tag}.metric.log"
echo "[$(date --iso-8601=seconds)] done"
