#!/usr/bin/env bash
#SBATCH --job-name=R3_plain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Plain R3 (latent_attention_train.py: stage-1 latent SFT, lr 3e-4, batch 128, <= 3 epochs, early stopping,
# best epoch kept) for any CATEGORY, then the same evaluation as margin_wider_collab.sbatch: ranking (eval_raw.json)
# and r interventions raw / mask / mean / shuffle_user + paired bootstrap.
# Env: CODE_ROOT (required), CATEGORY (required), BASE_MODEL (Qwen/Qwen2.5-1.5B), OUTPUT_NAME (<CATEGORY>_R3_base),
#      INTERVENTIONS (1), EXTRACT_LIMIT (10000).
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R3_ROOT="$PWD"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
category="${CATEGORY:?set CATEGORY}"
name="${OUTPUT_NAME:-${category}_R3_base}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}" WANDB_NAME="${WANDB_NAME:-$name}" WANDB_TAGS="${WANDB_TAGS:-r3-baseline,$category}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
mbs=4; [[ "$gpu_name" == *"3090"* ]] && mbs=2
data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv); eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv); info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
tag="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
results="output_dir/r3_plain/${name}"
mkdir -p "$out" "$results"
echo "[$(date --iso-8601=seconds)] $name: base=${BASE_MODEL:-Qwen/Qwen2.5-1.5B} GPU=$gpu_name mbs=$mbs code=$code_root"
if [[ ! -f "$out/model.safetensors" ]]; then
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model "${BASE_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --train_file "$train_file" --eval_file "$eval_file" --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size "$mbs" --batch_size 128 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
    --save_total_limit 1 "${resume[@]}"
  rm -rf "$out"/checkpoint-*   # best model is saved at $out; keep one model only
fi
[[ -f "$out/trainer_state.json" ]] && cp "$out/trainer_state.json" "$results/" || true
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
if [[ ! -f "$task_dir/eval_raw.json" ]]; then
  echo "[$(date --iso-8601=seconds)] ranking"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample -1 --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/eval_raw.json"
fi
python "$code_root/src/utils/calc.py" --path "$task_dir/eval_raw.json" --item_path "$info_file" | tee "$results/metric.log"
cp "$task_dir/eval_raw.json" "$results/"
if [[ "${INTERVENTIONS:-1}" == 1 ]]; then
  interv_dir="analysis/results/intervention/${tag}"
  echo "[$(date --iso-8601=seconds)] r interventions: extract train latents (mean / shuffle bank)"
  python "$code_root/src/prototype_grounding/experiment.py" extract --split train --batch-size 16 \
    --limit "${EXTRACT_LIMIT:-10000}" --checkpoint "$out" --tag "$tag" --category "$category"
  for m in raw mask mean shuffle_user; do
    echo "[$(date --iso-8601=seconds)] intervention mode=$m"
    python "$code_root/src/prototype_grounding/experiment.py" generate --split test \
      --bank all --k 32 --seed 42 --mode "$m" --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 8 --limit -1 \
      --checkpoint "$out" --tag "$tag" --category "$category"
  done
  python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw mask mean shuffle_user
  cp "$interv_dir"/*.json "$results/" 2>/dev/null || true
fi
echo "[$(date --iso-8601=seconds)] $name complete"
