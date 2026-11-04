#!/usr/bin/env bash
#SBATCH --job-name=R3_thoughts
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

# Latent-utilization redesign: Coconut-style K-step continuous thoughts
# (src/latent/continuous_thoughts.py) with or without the L7 bottleneck, recipe
# matched to R3_ori Toys (base Qwen/Qwen2.5-1.5B, lr 3e-4, batch 128, 3 epochs,
# early stopping). Then: self-check, bottleneck leak check, constrained 10-beam
# ranking, thought interventions (mask / mean / shuffle), paired bootstrap.
# GIST=1 (with SOURCE=constant LATENT_ONLY=1): gist bottleneck, K = latents + 1 readout; the latents read the history at every
# layer and the target reads it only through them (latent_bottleneck.py); interventions raw / delete.
# Env: CODE_ROOT (required), OUTPUT_ROOT (default output_dir), CATEGORY, SOURCE (hidden|constant), K, LATENT_ONLY (1/0), GIST,
#      HISTORY_BIAS (soft bottleneck, e.g. -4), OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, NUM_EPOCHS, TEST_SAMPLE, MODES, SKIP_TRAIN.

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
source="${SOURCE:-hidden}"
k="${K:-4}"
latent_only="${LATENT_ONLY:-1}"
gist="${GIST:-0}"
hbias="${HISTORY_BIAS:-}"  # soft bottleneck: finite negative attention bias on the history (needs LATENT_ONLY=1)
name="${OUTPUT_NAME:-${category}_thoughts_${source}_k${k}$([[ $latent_only == 1 ]] && echo _bottleneck)$([[ $gist == 1 ]] && echo _gist)$([[ -n $hbias ]] && echo _hb${hbias#-})}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,continuous-thoughts}"
export R3_ROOT="$PWD"

code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] $name: source=$source k=$k latent_only=$latent_only history_bias=${hbias:-none} GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-output_dir}/${name}"  # OUTPUT_ROOT: e.g. scratch (home quota is ~1 TB)
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] self-check on the base model"
  python "$code_root/src/latent/continuous_thoughts_check.py" --test_file "$test_file" --k "$k"
  [[ $gist == 1 ]] && python "$code_root/src/latent/latent_gist_check.py" --test_file "$test_file" --k "$k"
  if [[ -n "${VALID_SAMPLE:-}" ]]; then  # smoke runs: small validation file
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  echo "[$(date --iso-8601=seconds)] training"
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  latent_flag=()
  [[ $latent_only == 1 ]] && latent_flag=(--latent_only True)
  [[ $gist == 1 ]] && latent_flag+=(--latent_gist True)
  [[ -n $hbias ]] && latent_flag+=(--history_bias="$hbias")
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model False --save_total_limit 2 \
    --thought_source "$source" --n_thoughts "$k" "${latent_flag[@]}" "${resume[@]}"
fi

if [[ $latent_only == 1 && $gist != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] bottleneck leak check (fp32)"
  python "$code_root/src/latent/latent_bottleneck_check.py" \
    --checkpoint "$out" --test_file "$test_file" --category "$category" --pairs 16
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-raw mask mean shuffle}; do
  [[ "$source" == constant && "$mode" == shuffle ]] && continue  # identical to raw by construction
  [[ "$mode" == delete && $gist != 1 ]] && continue
  echo "[$(date --iso-8601=seconds)] ranking, thought_mode=$mode"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --thought_mode "$mode" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
cp "$task_dir/eval_raw.json" "$out/final_result.json"
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' | grep -v thought_bank || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
