#!/usr/bin/env bash
#SBATCH --job-name=R3_hist_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%A_%a.out
# Step 0 for B1: rank the Toys test set with a fixed checkpoint (default: T2,
# constant slot, full history in training) while the prompt keeps only k
# history items chosen by recency / random / target-similarity oracle /
# anti-oracle (scripts/history_subset_probe.py builds the CSVs).
# Env: CODE_ROOT (required), CHECKPOINT, PROBE_DIR, CATEGORY.
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
category="${CATEGORY:-Toys_and_Games}"
checkpoint="${CHECKPOINT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Toys_and_Games_thoughts_constant_k1/latent}"
probe_dir="${PROBE_DIR:-output_dir/history_subset_probe/${category}_T2}"
info_file=$(ls data/${category}/info/${category}_5_*.txt)
variants=(recent_k1 random_k1 oracle_k1 antioracle_k1 recent_k3 random_k3 oracle_k3 antioracle_k3)
v="${variants[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date --iso-8601=seconds)] $v checkpoint=$checkpoint code=$code_root GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python "$code_root/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$checkpoint" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$probe_dir/inputs/${v}.csv" \
  --result_json_data "$probe_dir/eval_${v}.json"
python "$code_root/src/utils/calc.py" --path "$probe_dir/eval_${v}.json" --item_path "$info_file"
echo "[$(date --iso-8601=seconds)] $v complete"
