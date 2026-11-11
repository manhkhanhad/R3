#!/usr/bin/env bash
#SBATCH --job-name=R3_rationale_eval_files
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out
# Evaluate one trained rationale-SFT model with several test rationale files (mode own = "use this file's
# text for each row"). Each file is evaluated once; finished results (eval_<name>.json) are skipped.
# Env: CODE, BASE_MODEL (trained model dir; results are written there), RAT_FILES (space-separated paths).
set -Eeuo pipefail
CODE=${CODE:?}; BASE_MODEL=${BASE_MODEL:?}; RAT_FILES=${RAT_FILES:?}
R3=/common/home/users/m/mk.ngohuu.2024/research/R3
DATA=$R3/data/Toys_and_Games; stem=Toys_and_Games_5_2017-1-2018-11
module purge; module load CUDA/12.4.0
eval "$(conda shell.bash hook)"; conda activate rrec
export HF_HUB_OFFLINE=1 WANDB_PROJECT=R3
cd "$CODE"
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} model=$BASE_MODEL"
for f in $RAT_FILES; do
    name=$(basename "$f" .json)
    res="$BASE_MODEL/eval_${name}.json"
    if [[ -f $res ]]; then echo "[$(date -Is)] $name done already"; continue; fi
    echo "[$(date -Is)] eval file=$name"
    python -u src/latent/rationale_sft_eval.py --base_model "$BASE_MODEL" --category Toys_and_Games \
        --info_file "$DATA/info/$stem.txt" --test_data_path "$DATA/test/$stem.csv" \
        --rationale_file "$f" --mode own --result_json_data "$res.tmp" --batch_size 8
    mv "$res.tmp" "$res"
done
echo "[$(date -Is)] done"
