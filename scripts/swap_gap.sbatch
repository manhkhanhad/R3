#!/usr/bin/env bash
#SBATCH --job-name=R3_swapgap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|l40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# History-swap gap NLL(y|x,r_other) - NLL(y|x,r_own) per row, copy vs non-copy (src/latent/swap_gap.py),
# on test (all rows) and a 5,000-row train sample, for each MODELS entry "tag=model_dir". Read-only on model dirs.
# Env: CODE_ROOT (required), MODELS (required), CATEGORY, TRAIN_SAMPLE (5000), OUT_DIR.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD"
code="${CODE_ROOT:?}"
category="${CATEGORY:-Toys_and_Games}"
out_dir="${OUT_DIR:-output_dir/margin_wider_collab/swap_gap/${category}}"
test_file=$(ls data/${category}/test/${category}_5_*.csv); train_file=$(ls data/${category}/train/${category}_5_*.csv)
for pair in ${MODELS:?}; do
  tag="${pair%%=*}"; dir="${pair#*=}"
  echo "[$(date --iso-8601=seconds)] $tag <- $dir"
  python "$code/src/latent/swap_gap.py" --model_dir "$dir" --tag "$tag" --file "$test_file" --split test \
    --out_dir "$out_dir" --category "$category"
  python "$code/src/latent/swap_gap.py" --model_dir "$dir" --tag "$tag" --file "$train_file" --split train \
    --out_dir "$out_dir" --category "$category" --sample "${TRAIN_SAMPLE:-5000}"
done
echo "[$(date --iso-8601=seconds)] swap gap complete"
