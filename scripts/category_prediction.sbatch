#!/usr/bin/env bash
#SBATCH --job-name=R3_category_pred
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out

# Next-item category prediction: plain Qwen SFT (src/latent/plain_sft_*.py) on
# the R3 prompt with the instruction asking for the category, constrained beam
# search over the category catalog, then category-vs-item analysis against the
# R3 / T2 item evals on the same test rows.
# Usage: CODE_ROOT=<frozen copy> [SAMPLE=-1] [EPOCHS=3] sbatch scripts/category_prediction.sbatch
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"

code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
name="${NAME:-Toys_and_Games_category_level3}"
level="${LEVEL:-3}"
sample="${SAMPLE:--1}"
epochs="${EPOCHS:-3}"
data="output_dir/category_prediction/Toys_and_Games_level${level}/data"
out="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${name}"
results="output_dir/category_prediction/${name}"
split=Toys_and_Games_5_2017-1-2018-11.csv
instruction="Given a list of toys and games the user recently enjoyed, please predict the category of the next toys and games item the user may buy."

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
mbs=4; [[ "$gpu_name" == *"3090"* ]] && mbs=2
echo "[$(date --iso-8601=seconds)] $name: level=$level sample=$sample epochs=$epochs GPU=$gpu_name mbs=$mbs code=$code"
mkdir -p "$out/model" "$results"

if [[ ! -f "$data/categories.txt" ]]; then
  python "$code/scripts/category_prediction_data.py" --data_dir data/Toys_and_Games --output_dir "$data" --level "$level"
fi

if [[ ! -f "$out/model/config.json" ]]; then
  resume=()
  last="$(ls -d "$out"/model/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$last" ]] && resume=(--resume_from_checkpoint "$last")
  python "$code/src/latent/plain_sft_train.py" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$data/train/$split" --eval_file "$data/valid/$split" \
    --output_dir "$out/model" --sample "$sample" \
    --micro_batch_size "$mbs" --batch_size 128 \
    --num_epochs "$epochs" --cutoff_len 512 --learning_rate 2e-5 \
    --category "toys and games" --dataset_name Toys_and_Games --instruction "$instruction" "${resume[@]}"
fi

echo "[$(date --iso-8601=seconds)] evaluating"
python "$code/src/latent/plain_sft_eval.py" \
  --base_model "$out/model" --info_file "$data/categories.txt" \
  --test_data_path "$data/test/$split" --sample "$sample" \
  --result_json_data "$out/eval_category.json" --batch_size "$mbs" \
  --category "toys and games" --dataset_name Toys_and_Games --instruction "$instruction"
cp "$out/eval_category.json" "$results/"

if [[ "$sample" == "-1" ]]; then
  python "$code/scripts/category_vs_item_analysis.py" --level "$level" \
    --category_eval "$results/eval_category.json" \
    --item_eval R3=/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/Toys_and_Games/latent/final_result.json \
    --item_eval T2=output_dir/history_subset_probe/Toys_and_Games_T2/eval_full.json \
    --out "$results/report.md"
fi
echo "[$(date --iso-8601=seconds)] done"
