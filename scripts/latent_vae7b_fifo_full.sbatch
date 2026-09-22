#!/usr/bin/env bash
#SBATCH --job-name=R3_vae7b_fifo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Attempt 7b FIFO startup variant.  The detached-z queue is filled to 256 rows
# before aggregate KL is enabled.  New rows are then appended and the oldest
# rows evicted.  All other 7b settings are held fixed.  This job performs only
# the standard recommendation ranking evaluation after training (no E1 or
# latent-distribution analysis).
category="CDs_and_Vinyl"
base_data_dir="data/${category}"
train_file="${base_data_dir}/train/${category}_5_2015-10-2018-11.csv"
eval_file="${base_data_dir}/valid/${category}_5_2015-10-2018-11.csv"
test_file=$(find "${base_data_dir}/test" -maxdepth 1 -name "${category}*.csv" -print -quit)
info_file=$(find "${base_data_dir}/info" -maxdepth 1 -name "${category}*.txt" -print -quit)
task_dir="output_dir/CDs_and_Vinyl_R3_vae7b_fifo"
out="${task_dir}/latent"
mkdir -p "$out"

python src/latent/latent_attention_train_vae.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --ramp_epochs 3 --alpha_max 0.75 \
    --rho_mu 1.0 --ema_momentum 0.99 \
    --kl_weight 1.0 --mu_consistency_weight 0.5 \
    --z_queue_size 256 --kl_min_queue_size 256

echo "[$(date --iso-8601=seconds)] FIFO training finished; starting standard ranking evaluation"
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
python src/latent/latent_attention_eval.py --batch_size 8 \
    --base_model "$out" --sample -1 \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" \
    --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" \
    --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file"
