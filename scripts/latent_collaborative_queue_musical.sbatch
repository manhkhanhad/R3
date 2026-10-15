#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_queue_Musical
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Attempt 10 (collaborative multi-positive InfoNCE, plain R3, Musical) with a new
# queue admission policy.  Identical to Attempt 10 except:
#   - no token-accuracy gate; instead a relative NLL gate: a sample enters the queue
#     when its target NLL is within the best 25% of the last 1,024 training samples;
#   - VARIANT=diverse adds diversity-aware insertion: per-user cap 2, per-target cap 4,
#     a same-user latent with cosine >= 0.9 refreshes the existing entry.
# Queue stays 10k slots, no age limit.  Then: standard ranking, train-latent geometry,
# r interventions (raw / mask / mean / shuffle_user / shuffle_target).
# Env: VARIANT (gate|diverse), CODE_ROOT (frozen code copy), SKIP_TRAIN.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R3_ROOT="$PWD"
code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
variant="${VARIANT:?set VARIANT to gate or diverse}"

queue_args=(--positive_min_token_accuracy 0.0 --nll_gate_quantile 0.25 --nll_gate_window 1024)
case "$variant" in
  gate) ;;
  diverse) queue_args+=(--bank_max_per_user 2 --bank_max_per_target 4 --bank_dedup_cos 0.9) ;;
  *) echo "unknown VARIANT=$variant" >&2; exit 1 ;;
esac

export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_collab_queue_${variant}_Musical}"
export WANDB_TAGS="${WANDB_TAGS:-collaborative-infonce,queue-policy}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch: $micro_batch_size; variant: $variant; code: $code"

category="Musical_Instruments"
base_data_dir="data/${category}"
train_file="${base_data_dir}/train/Musical_Instruments_5_2015-1-2018-11.csv"
eval_file="${base_data_dir}/valid/Musical_Instruments_5_2015-1-2018-11.csv"
test_file="${base_data_dir}/test/Musical_Instruments_5_2015-1-2018-11.csv"
info_file="${base_data_dir}/info/Musical_Instruments_5_2015-1-2018-11.txt"
tag="musical_instruments_collab_queue_${variant}"
task_dir="output_dir/${category}_R3_collab_queue_${variant}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] training"
  python "$code/src/latent/latent_attention_train_collaborative.py" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --projection_dim 256 --temperature 0.1 \
    --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
    --bank_size 10000 --max_user_positives 16 \
    --max_target_positives 16 --num_negatives 256 \
    "${queue_args[@]}"
fi

echo "[$(date --iso-8601=seconds)] standard ranking"
python "$code/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python "$code/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/0.json"
python "$code/src/utils/merge.py" --input_path "$task_dir/" --output_path "$out/final_result.json" --nn 1
python "$code/src/utils/calc.py" --path "$out/final_result.json" --item_path "$info_file"

log_dir="analysis/logs/R3_collab_queue_${variant}_musical"
result_dir="analysis/results/intervention/${tag}"
mkdir -p "$log_dir" "$result_dir"
echo "[$(date --iso-8601=seconds)] train-latent extraction and geometry"
python "$code/src/prototype_grounding/experiment.py" extract \
  --split train --batch-size 32 --checkpoint "$out" --tag "$tag" --category "$category" \
  >"$log_dir/extract_train.log" 2>&1
python "$code/src/prototype_grounding/experiment.py" latent-distribution \
  --checkpoint "$out" --tag "$tag" --category "$category" --component r \
  >"$log_dir/latent_distribution.log" 2>&1

for mode in raw mask mean shuffle_user shuffle_target; do
  echo "[$(date --iso-8601=seconds)] intervention ${mode}"
  python "$code/src/prototype_grounding/experiment.py" generate \
    --split test --bank all --k 32 --seed 42 --mode "$mode" \
    --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
    --checkpoint "$out" --tag "$tag" --category "$category" \
    >"$log_dir/ranking_${mode}.log" 2>&1
done
date --iso-8601=seconds >"$result_dir/RANKING_COMPLETE"
echo "[$(date --iso-8601=seconds)] ${variant} complete"
