#!/usr/bin/env bash
#SBATCH --job-name=R3_cmp_baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3VRecLatentCompare
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# R3 baseline for the R3-vs-VRec latent comparison (document/vrec_vs_r3_latent_comparison.md).
# Same configuration as VRec stage 1 (Verifiable-Rec/code/scripts/run_musical_instruments.sh with
# N_THOUGHT=1): Qwen2.5-1.5B base, identical Musical Instruments split, lr 3e-4, warmup 20,
# batch 128 = micro 2 x 64 accumulation on 1 GPU, 3 epochs with early stopping, cutoff 512, seed 0, bf16.
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category=Musical_Instruments
data=/common/home/users/m/mk.ngohuu.2024/research/R3/data/${category}
stem=Musical_Instruments_5_2015-1-2018-11
base=/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/Qwen2.5-1.5B
task_dir=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/vrec_compare/Musical_Instruments_r3_baseline
out=${task_dir}/latent
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] host=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"

python src/latent/latent_attention_train.py \
  --base_model "$base" \
  --train_file "${data}/train/${stem}.csv" --eval_file "${data}/valid/${stem}.csv" \
  --output_dir "$out" --category "$category" \
  --sample -1 --seed 0 --micro_batch_size 2 --batch_size 128 \
  --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

python src/utils/split.py --input_path "${data}/test/${stem}.csv" --output_path "$task_dir" --nn 1
python src/latent/latent_attention_eval.py \
  --batch_size 4 --base_model "$out" --sample -1 \
  --info_file "${data}/info/${stem}.txt" --category "$category" \
  --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "${data}/info/${stem}.txt"
echo "[$(date --iso-8601=seconds)] done"
