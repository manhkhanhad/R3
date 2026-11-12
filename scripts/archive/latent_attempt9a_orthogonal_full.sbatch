#!/usr/bin/env bash
#SBATCH --job-name=R3_attempt9a_orthogonal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=08:00:00
#SBATCH --output=run_logs/%x.%j.out

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="CDs_and_Vinyl"
base_data_dir="data/${category}"
train_file="${base_data_dir}/train/${category}_5_2015-10-2018-11.csv"
eval_file="${base_data_dir}/valid/${category}_5_2015-10-2018-11.csv"
test_file=$(find "${base_data_dir}/test" -maxdepth 1 -name "${category}*.csv" -print -quit)
info_file=$(find "${base_data_dir}/info" -maxdepth 1 -name "${category}*.txt" -print -quit)

base_checkpoint="output_dir/CDs_and_Vinyl_R3_vae7b_nocurriculum/latent/checkpoint-1152"
mean_thoughts="analysis/evidence/CDs_and_Vinyl/latent_vae7b_nocurriculum_checkpoint_1152/train/thoughts.pt"
task_dir="output_dir/CDs_and_Vinyl_R3_attempt9a_orthogonal"
out="${task_dir}/latent"
mkdir -p "$out"

# Single-variable change from Attempt 9: project every learned residual onto
# the orthogonal complement of the fixed train-mean direction.
python src/latent/latent_attention_train_residual.py \
    --base_model "$base_checkpoint" \
    --mean_thoughts_path "$mean_thoughts" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" --sample -1 \
    --micro_batch_size 4 --batch_size 128 --num_epochs 3 \
    --cutoff_len 512 --learning_rate 3e-4 \
    --bottleneck_size 512 --residual_scale 1.0 \
    --orthogonal_to_mu=True

echo "[$(date --iso-8601=seconds)] Attempt 9a training finished; starting standard ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
python src/latent/latent_attention_eval.py --batch_size 8 \
    --base_model "$out" --sample -1 \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" \
    --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
    --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file"
