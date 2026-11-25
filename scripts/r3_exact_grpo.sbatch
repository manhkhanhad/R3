#!/bin/bash
#SBATCH --job-name=R3_exact_grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Exact original R3 stage-2 GRPO (R3_ori git HEAD c558b44, scripts/attention_grpo.sh), unmodified code
# except the dead `from noisy_layer import ...` line (file never committed; caller commented out).
# Settings copied from attention_grpo.sh: torchrun 2 GPUs, num_generations 8, sample -1, 1 epoch,
# beta 0.00, batch 256, micro 8, seed 42, num_iterations 1, epsilon 0.2 / 0.28; LR (default 1e-5).
# Eval = R3's noise_eval.py on 2 shards (one per GPU) + merge, as in attention_grpo.sh.
# Afterwards (read-only on the saved model): r interventions with our experiment.py (ANALYSIS_CODE).
set -euo pipefail
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export R3_ROOT="$PWD"  # experiment.py resolves data/ and output_dir/ from R3_ROOT (371452 interventions failed without it)
# smoke 370692 hung 30 min in the first NCCL all-gather (DDP param check) on album: GPU peer-to-peer
# transport hang. Route NCCL through host memory instead; launch environment only, numerics unchanged.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
code="${CODE_ROOT:?frozen R3 code copy}"
analysis_code="${ANALYSIS_CODE:?frozen analysis code copy (experiment.py)}"
base_model="${BASE_MODEL:?stage-1 checkpoint}"
tag="${TAG:?tag}"
lr="${LR:-1e-5}"
seed="${SEED:-42}"
smoke="${SMOKE:-0}"
category="${CATEGORY:-Toys_and_Games}"
data="data/${category}"
train_file=$(ls ${data}/train/${category}_5_*.csv)
eval_file=$(ls ${data}/valid/${category}_5_*.csv)
test_file=$(ls ${data}/test/${category}_5_*.csv)
info_file=$(ls ${data}/info/${category}_5_*.txt)
if [[ "$smoke" == 1 ]]; then sample=256; batch=64; test_sample=64; else sample=-1; batch=256; test_sample=-1; fi
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3exact_${tag}"
out="${task_dir}/model"
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' '); base ${base_model}; code ${code}; lr ${lr}; smoke ${smoke}; sample ${sample}; batch ${batch}"

if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  torchrun --nnodes=1 --nproc_per_node=2 --master_port=$((RANDOM % 9999 + 10000)) \
    "$code/src/grpo_attention_tuning/train_noise_grpo.py" --category "$category" \
    --num_generations 8 --sample "$sample" --num_epochs 1 --info_file "$info_file" \
    --base_model "$base_model" --output_dir "$out" --beta 0.00 \
    --batch_size "$batch" --micro_batch_size 8 --lr "$lr" --seed "$seed" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28
fi

results="output_dir/r3_exact_grpo/${tag}"
mkdir -p "$results"
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
echo "[$(date --iso-8601=seconds)] R3 noise_eval (2 shards)"
python "$code/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 2
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python "$code/src/grpo_attention_tuning/noise_eval.py" \
    --batch_size 12 --base_model "$out" --sample "$test_sample" \
    --info_file "$info_file" --category "$category" --end_k -1 \
    --test_data_path "$task_dir/${i}.csv" --result_json_data "$task_dir/${i}.json" &
done
wait
python "$code/src/utils/merge.py" --input_path "$task_dir/" --output_path "$out/final_result.json" --nn 2
python "$code/src/utils/calc.py" --path "$out/final_result.json" --item_path "$info_file"
cp "$out/final_result.json" "$results/"
[[ "$smoke" == 1 ]] && { echo "[$(date --iso-8601=seconds)] smoke complete"; exit 0; }
fi

if [[ "${SKIP_INTERV:-0}" == 1 ]]; then echo "[$(date --iso-8601=seconds)] ${tag} complete (ranking only)"; exit 0; fi
echo "[$(date --iso-8601=seconds)] r interventions (analysis only)"
log_dir="analysis/logs/R3exact_${tag}"; mkdir -p "$log_dir"
python "$analysis_code/src/prototype_grounding/experiment.py" extract \
  --split train --batch-size 32 --checkpoint "$out" --tag "R3exact_${tag}" --category "$category" \
  >"$log_dir/extract_train.log" 2>&1
for mode in raw mask mean shuffle_user; do
  python "$analysis_code/src/prototype_grounding/experiment.py" generate \
    --split test --bank all --k 32 --seed 42 --mode "$mode" \
    --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
    --checkpoint "$out" --tag "R3exact_${tag}" --category "$category" \
    >"$log_dir/ranking_${mode}.log" 2>&1
done
cp analysis/results/intervention/R3exact_${tag}/*.json "$results/" 2>/dev/null || true
echo "[$(date --iso-8601=seconds)] ${tag} complete"
