#!/bin/bash
#SBATCH --job-name=R3_beam_grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="l40s|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# R3 stage-2 GRPO (original recipe: noise N(0,1.5^2) on r, 8 copies per row, noise-free first copy, advantage vs the noise-free
# reward, teacher-forced target log-prob in the loss, beta 0, 1 iteration) with TWO changes requested by the user:
#   (1) reward = NDCG@10 of the target in a 10-beam constrained beam-search list generated from the noisy r (instead of -PPL)
#   (2) full-model training (decoder + embeddings + lm_head + attention), fp32 master weights with bf16 autocast, gradient checkpointing.
# Also (needed by 2): fewer rows (SAMPLE, beam search on every copy is slow), smaller batch (BATCH) so there are enough updates,
# warm-up WARMUP steps instead of 200 (the run only has SAMPLE/BATCH steps); evaluation in fp32 (bf16 rounding would erase tiny updates).
# BATCH counts generations (8 per row): BATCH=64 is 8 rows per step. ADV_BASELINE group = reward minus the same row's noise-free reward; batch = stock R3.
# Env: CODE_ROOT BASE_MODEL TAG LR [SAMPLE=8000 BATCH=64 WARMUP=25 SEED=42 ADV_BASELINE=group SMOKE=1 CATEGORY]
#      EVAL_ONLY_MODEL=<dir>: skip training, evaluate that model in fp32 (used for the start model under the same dtype).
set -euo pipefail
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export R3_ROOT="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
code="${CODE_ROOT:?frozen code copy}"
base_model="${BASE_MODEL:?start checkpoint}"
tag="${TAG:?tag}"
lr="${LR:-1e-6}"
seed="${SEED:-42}"
smoke="${SMOKE:-0}"
category="${CATEGORY:-Toys_and_Games}"
data="data/${category}"
train_file=$(ls ${data}/train/${category}_5_*.csv)
eval_file=$(ls ${data}/valid/${category}_5_*.csv)
test_file=$(ls ${data}/test/${category}_5_*.csv)
info_file=$(ls ${data}/info/${category}_5_*.txt)
sample="${SAMPLE:-8000}"; batch="${BATCH:-64}"; warmup="${WARMUP:-25}"; adv_baseline="${ADV_BASELINE:-group}"; test_sample=-1
if [[ "$smoke" == 1 ]]; then sample=64; batch=32; warmup=1; test_sample=48; fi
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3beam_${tag}"
out="${task_dir}/model"
mkdir -p "$task_dir"
echo "[$(date --iso-8601=seconds)] GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ' '); base ${base_model}; code ${code}; lr ${lr}; sample ${sample}; batch ${batch}; warmup ${warmup}; seed ${seed}; adv_baseline ${adv_baseline}; smoke ${smoke}"

if [[ -n "${EVAL_ONLY_MODEL:-}" ]]; then
  out="$EVAL_ONLY_MODEL"
else
  mkdir -p "$out"
  torchrun --nnodes=1 --nproc_per_node=2 --master_port=$((RANDOM % 9999 + 10000)) \
    "$code/src/grpo_attention_tuning/train_noise_grpo.py" --category "$category" \
    --num_generations 8 --sample "$sample" --num_epochs 1 --info_file "$info_file" \
    --base_model "$base_model" --output_dir "$out" --beta 0.00 \
    --batch_size "$batch" --micro_batch_size 8 --lr "$lr" --seed "$seed" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
    --reward_type beam_ndcg --rank_beams 10 --train_full True --model_fp32 True --gradient_checkpointing True \
    --warmup_steps "$warmup" --save_steps 1000000 --adv_baseline "$adv_baseline"
  # one checkpoint per run: keep the final model in $out, drop the trainer checkpoint (optimizer state, ~18 GB)
  rm -rf "$out"/checkpoint-*
fi

results="output_dir/r3_beam_grpo/${tag}"
mkdir -p "$results"
echo "[$(date --iso-8601=seconds)] R3 noise_eval fp32 (2 shards)"
python "$code/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 2
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python "$code/src/grpo_attention_tuning/noise_eval.py" \
    --batch_size 12 --base_model "$out" --sample "$test_sample" --dtype fp32 \
    --info_file "$info_file" --category "$category" --end_k -1 \
    --test_data_path "$task_dir/${i}.csv" --result_json_data "$task_dir/${i}.json" &
done
wait
python "$code/src/utils/merge.py" --input_path "$task_dir/" --output_path "$task_dir/final_result.json" --nn 2
python "$code/src/utils/calc.py" --path "$task_dir/final_result.json" --item_path "$info_file"
cp "$task_dir/final_result.json" "$results/"
cp "$out/trainer_state.json" "$results/" 2>/dev/null || true
echo "[$(date --iso-8601=seconds)] ${tag} complete"
