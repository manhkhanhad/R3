#!/usr/bin/env bash
#SBATCH --job-name=R3_steer
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
# B1 steered reading (src/latent/steered_reading.py): a K=1 continuous thought
# selects which k history items the recommendation pass reads (dropped items are
# masked as if deleted); selector trained toward the oracle items (target-similar
# titles, training only).  Recipe matched to the T1-T3 Toys runs (base
# Qwen/Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs, early stopping).  Then:
# self-checks, constrained 10-beam ranking under thought interventions
# (raw / mean / shuffle) and selection interventions (recent / all), paired bootstrap.
# Env: CODE_ROOT (required), OUTPUT_ROOT (default scratch), CATEGORY, STEER_K, RULE (learned|recent),
#      AUX_WEIGHT, THOUGHT_SOURCE (hidden|constant), OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, NUM_EPOCHS, TEST_SAMPLE, MODES, SKIP_TRAIN.
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
k="${STEER_K:-2}"
rule="${RULE:-learned}"
aux="${AUX_WEIGHT:-0.2}"
source="${THOUGHT_SOURCE:-hidden}"  # constant = recency-split baseline (one learned token, fixed rule)
suffix=""; [[ "$source" != hidden ]] && suffix="_${source}"
name="${OUTPUT_NAME:-${category}_steer_${rule}_k${k}${suffix}}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,steered-reading}"
export R3_ROOT="$PWD"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] $name: k=$k rule=$rule aux=$aux source=$source GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
emb_file="${data_dir}/info/${category}_qwen3emb_1536.pt"
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] self-checks (assignment, mask = deletion, KV-cache decoding)"
  (cd "$task_dir" && python "$code_root/src/latent/steered_reading_check.py" --test_file "$R3_ROOT/$test_file" --k "$k")
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  echo "[$(date --iso-8601=seconds)] training"
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model False --save_total_limit 2 \
    --thought_source "$source" --n_thoughts 1 \
    --steered_k "$k" --steered_rule "$rule" --steered_aux_weight "$aux" \
    --item_embeddings_file "$emb_file" "${resume[@]}"
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-raw mean shuffle sel_recent sel_all}; do
  thought=raw; selection=learned
  case "$mode" in
    mean|shuffle) thought="$mode" ;;
    sel_recent) selection=recent ;;
    sel_all) selection=all ;;
  esac
  [[ "$rule" == recent && "$mode" == sel_recent ]] && continue  # identical to raw
  [[ "$rule" == recent && "$selection" == learned ]] && selection=recent
  echo "[$(date --iso-8601=seconds)] ranking $mode (thought=$thought selection=$selection)"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --thought_mode "$thought" \
    --selection_mode "$selection" --item_embeddings_file "$emb_file" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
cp "$task_dir/eval_raw.json" "$out/final_result.json"
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
