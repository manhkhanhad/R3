#!/usr/bin/env bash
#SBATCH --job-name=R3_sasrec_category
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# SASRec-style category prediction (src/latent/sasrec_category.py), item IDs only
# and item IDs + history categories, then category-vs-item analysis against the
# R3 / T2 item evals.  Data from scripts/category_prediction_data.py.
# Usage: CODE_ROOT=<frozen copy> sbatch scripts/sasrec_category.sbatch
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1

code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
level="${LEVEL:-3}"
data="output_dir/category_prediction/Toys_and_Games_level${level}/data"
results="output_dir/category_prediction/Toys_and_Games_sasrec_category_level${level}"
mkdir -p "$results"
echo "[$(date --iso-8601=seconds)] code=$code GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

for variant in ids ids_cats; do
  flag=(); [[ $variant == ids_cats ]] && flag=(--history_categories)
  echo "[$(date --iso-8601=seconds)] variant $variant"
  python "$code/src/latent/sasrec_category.py" --data_dir "$data" --level "$level" \
    --output "$results/eval_${variant}.json" "${flag[@]}"
  python "$code/scripts/category_vs_item_analysis.py" --level "$level" \
    --category_eval "$results/eval_${variant}.json" \
    --item_eval R3=/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/Toys_and_Games/latent/final_result.json \
    --item_eval T2=output_dir/history_subset_probe/Toys_and_Games_T2/eval_full.json \
    --out "$results/report_${variant}.md" > /dev/null
done
echo "[$(date --iso-8601=seconds)] done"
