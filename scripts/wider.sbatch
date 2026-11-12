#!/usr/bin/env bash
#SBATCH --job-name=R3_wider
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# WIDER spectral regulariser (arXiv 2609.07406) on plain R3, K = 1 (src/latent/wider_reg.py):
# training adds WEIGHT * mean_b (z_b . q)^2, z_b = r_b/||r_b||, q = EMA of the batch mean
# direction (stop-grad); eval loss and inference are plain R3. Recipe = R3 (base
# Qwen2.5-1.5B, lr 3e-4, batch 128, <= 3 epochs, early stopping). Then constrained 10-beam
# ranking, r interventions raw / mask / mean / shuffle_user (experiment.py), paired
# bootstrap, latent geometry (scripts/latent_rank.py), operator trace (n = 300).
# Env: CODE_ROOT (required), WEIGHT (default 0.1), MOMENTUM, CATEGORY (default Toys_and_Games),
#      NUM_EPOCHS, OUTPUT_ROOT, OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, TEST_SAMPLE,
#      EXTRACT_LIMIT, INTERVENTION_LIMIT, MODES, TRACE_N.
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
weight="${WEIGHT:-0.1}"
case "$category" in CDs_and_Vinyl) prompt_category=musics ;; Toys_and_Games) prompt_category="toys and games" ;; Musical_Instruments) prompt_category="music instruments" ;; *) echo "no prompt category for $category" >&2; exit 2 ;; esac
name="${OUTPUT_NAME:-${category}_wider_${weight}}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,wider}"
export R3_ROOT="$PWD"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
echo "[$(date --iso-8601=seconds)] $name: wider weight=$weight GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
tag="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
interv_dir="analysis/results/intervention/${tag}"
results="output_dir/wider/${name}"
mkdir -p "$out" "$results"

if [[ ! -f "$out/config.json" ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  echo "[$(date --iso-8601=seconds)] training"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model True --save_total_limit 1 \
    --wider_weight "$weight" --wider_momentum "${MOMENTUM:-0.99}" "${resume[@]}"
fi
[[ -f "$out/config.json" ]] || { echo "no trained model in $out" >&2; exit 1; }

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
if [[ ! -f "$task_dir/eval_raw.json" ]]; then
  echo "[$(date --iso-8601=seconds)] standard ranking (constrained 10-beam)"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/eval_raw.json"
fi
python "$code_root/src/utils/calc.py" --path "$task_dir/eval_raw.json" --item_path "$info_file"
cp "$task_dir/eval_raw.json" "$results/"
ls -d "$out"/checkpoint-* >/dev/null 2>&1 && cp "$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$results/" || true

echo "[$(date --iso-8601=seconds)] r interventions: extract train latents (mean / shuffle bank)"
python "$code_root/src/prototype_grounding/experiment.py" extract --split train --batch-size 16 \
  --limit "${EXTRACT_LIMIT:-10000}" --checkpoint "$out" --tag "$tag" --category "$category"
python "$code_root/scripts/latent_rank.py" "analysis/evidence/${category}/${tag}/train/thoughts.pt" "$results/latent_rank.json"
modes="${MODES:-raw mask mean shuffle_user}"
for m in $modes; do
  echo "[$(date --iso-8601=seconds)] intervention mode=$m"
  python "$code_root/src/prototype_grounding/experiment.py" generate --split test \
    --bank all --k 32 --seed 42 --mode "$m" --topk 1 \
    --temperature 0.1 --alpha 1.0 --batch-size 8 --limit "${INTERVENTION_LIMIT:--1}" \
    --checkpoint "$out" --tag "$tag" --category "$category"
done
python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw $(echo "$modes" | sed 's/\braw\b//')
cp "$interv_dir"/*.json "$results/" 2>/dev/null || true

echo "[$(date --iso-8601=seconds)] operator trace"
python "$code_root/src/prototype_grounding/latent_operator_trace.py" --checkpoint "$out" \
  --test_file "$test_file" --category "$prompt_category" --n "${TRACE_N:-300}" \
  --output "analysis/results/latent_operator_trace/${tag}.json"
echo "[$(date --iso-8601=seconds)] $name complete"
