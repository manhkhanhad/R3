#!/usr/bin/env bash
#SBATCH --job-name=R3_divreason
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# DivReason reimplementation, SFT stage (src/latent/divreason.py): N divergent R3
# paths (MC-dropout + k-means directional perturbation) aggregated by a router with
# redundancy suppression into r_agg; decoder reads history + r_agg as in R3.
# PATHS=0 trains plain R3 with the identical recipe (baseline).  Recipe matched to
# the T1-T3 Toys runs (base Qwen/Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs,
# early stopping).  Then constrained 10-beam ranking under r_agg interventions
# (raw / single path / mean / shuffle) and paired bootstrap.
# Env: CODE_ROOT (required), OUTPUT_ROOT (default scratch), CATEGORY, PATHS, DROPOUT, LAYER, ALPHA,
#      BETA, CENTER (True/False), OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, NUM_EPOCHS, TEST_SAMPLE, MODES, SKIP_TRAIN.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="${CATEGORY:-Toys_and_Games}"
paths="${PATHS:-4}"
center="${CENTER:-False}"
if [[ $paths -gt 0 ]]; then
  suffix=""
  [[ $center == True ]] && suffix=_centered
  name="${OUTPUT_NAME:-${category}_divreason_n${paths}${suffix}}"
else
  name="${OUTPUT_NAME:-${category}_r3_plain_matched}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-divreason}"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
[[ $paths -gt 0 && "$gpu_name" == *"3090"* ]] && micro_batch_size=1  # N first passes per row
echo "[$(date --iso-8601=seconds)] $name: paths=$paths center=$center GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  echo "[$(date --iso-8601=seconds)] training"
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  div=()
  [[ $paths -gt 0 ]] && div=(--divreason_paths "$paths" --divreason_dropout "${DROPOUT:-0.1}"
    --divreason_layer "${LAYER:-14}" --divreason_alpha "${ALPHA:-0.1}" --divreason_beta "${BETA:-0.5}"
    --divreason_center "$center")
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model False --save_total_limit 2 "${div[@]}" "${resume[@]}"
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
default_modes="raw single mean shuffle"
[[ $paths -gt 0 ]] || default_modes="raw"
for mode in ${MODES:-$default_modes}; do
  echo "[$(date --iso-8601=seconds)] ranking, thought_mode=$mode"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 4 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --thought_mode "$mode" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
cp "$task_dir/eval_raw.json" "$out/final_result.json"
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
