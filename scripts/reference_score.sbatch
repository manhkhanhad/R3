#!/usr/bin/env bash
#SBATCH --job-name=R3_refscore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|l40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# MW-R step 1: reference-model NLL of each row's target (src/latent/reference_score.py) on train / valid / test.
# Env: CODE_ROOT (required), BASE_MODEL (default Qwen/Qwen2.5-1.5B), CATEGORY, SAMPLE (-1 = all), OUT_DIR.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD"
code_root="${CODE_ROOT:?}"
category="${CATEGORY:-Toys_and_Games}"
out_dir="${OUT_DIR:-output_dir/margin_wider_collab/reference_nll/${category}}"
for split in train valid test; do
  if [[ -f "$out_dir/${split}.pt" ]]; then echo "skip $split (done)"; continue; fi
  file=$(ls data/${category}/${split}/${category}_5_*.csv)
  echo "[$(date --iso-8601=seconds)] scoring $split <- $file"
  python "$code_root/src/latent/reference_score.py" --base_model "${BASE_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --file "$file" --split "$split" --out_dir "$out_dir" --category "$category" --sample "${SAMPLE:--1}" ${LATENT_MODEL:+--latent_model True}
done
echo "[$(date --iso-8601=seconds)] reference scoring complete"
