#!/usr/bin/env bash
#SBATCH --job-name=R3_category_Musical
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=20:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/category-r3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# R3 trained to predict the target item's category path instead of its title (same prompt, latent, recipe).
# Usage: sbatch scripts/category_r3_musical.sbatch [full|smoke]
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

mode="${1:-full}"
DATA=/common/home/users/m/mk.ngohuu.2024/research/R3/data/Musical_Instruments
cat_file=$DATA/info/Musical_Instruments_gte_large_en_v1_5_categories.pt
train_file=$DATA/train/Musical_Instruments_5_2015-1-2018-11.csv
eval_file=$DATA/valid/Musical_Instruments_5_2015-1-2018-11.csv

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch=4; eval_batch=8
if [[ "$gpu_name" == *"3090"* ]]; then micro_batch=2; eval_batch=4; fi
if [[ "$mode" == smoke ]]; then
  out=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/smoke_category_r3; sample=256; epochs=1; limit=16
else
  out=/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/Musical_Instruments_R3_category; sample=-1; epochs=3; limit=-1
fi
echo "[$(date --iso-8601=seconds)] mode=$mode GPU=$gpu_name micro_batch=$micro_batch out=$out"
mkdir -p "$out/latent"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out/latent" --category Musical_Instruments \
  --sample "$sample" --micro_batch_size "$micro_batch" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 512 \
  --task category --category_file "$cat_file" \
  --save_only_model True --save_total_limit 1
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

python src/latent/category_eval.py --base_model "$out/latent" --out_dir "$out/eval" \
  --batch_size "$eval_batch" --limit "$limit"
echo "[$(date --iso-8601=seconds)] done ($mode)"
