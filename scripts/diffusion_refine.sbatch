#!/usr/bin/env bash
#SBATCH --job-name=R3_diffusion_refine
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
# Diffusion-refined R3 thought (src/latent/diffusion_refine.py): r' = g*normalize(P(diffuse(r))),
# denoiser target = the target item's R3 title vector (item_oracle_prep.py r3title table).
# Toys, prompt unchanged, trained from the base model with R3's recipe; then ranking with
#   raw       sampled x0_hat (the method)
#   oracle    target item's vector in place of x0_hat (ceiling of this decoder)
#   shuffle   another user's x0_hat
#   mean      mean x0_hat
#   swaphist  raw, with another user's history (scripts/swap_history.py)
# Every eval also writes diffusion_x0.json (nearest-item retrieval of x0_hat).
# Evals already written are skipped (requeue-safe).
# Env: CODE_ROOT (required), MODE (diffusion|regress), OUTPUT_NAME, NUM_EPOCHS, LR, TRAIN_SAMPLE,
#      EVAL_SAMPLE, TEST_SAMPLE, MODES, OUTPUT_ROOT, PREP, STEPS, WEIGHT, CE_WEIGHT, CHAIN_PROB.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category=Toys_and_Games
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
mode="${MODE:-diffusion}"
prep="${PREP:-output_dir/item_oracle/prep/Toys_item_r3title.pt}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"

name="${OUTPUT_NAME:-${category}_diffusion_refine_${mode}}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,diffusion-refine}"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] $name: mode=$mode GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ! -f "$out/config.json" ]]; then
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  eval_sample=(); [[ -n "${EVAL_SAMPLE:-}" ]] && eval_sample=(--eval_sample "$EVAL_SAMPLE")
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" "${eval_sample[@]}" \
    --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate "${LR:-3e-4}" --cutoff_len 512 \
    --save_only_model True --save_total_limit 2 \
    --diffusion_refine_file "$prep" --diffusion_mode "$mode" \
    --diffusion_steps "${STEPS:-32}" --diffusion_weight "${WEIGHT:-1.0}" \
    --diffusion_ce_weight "${CE_WEIGHT:-1.0}" --diffusion_chain_prob "${CHAIN_PROB:-0.5}" "${resume[@]}"
fi
[[ -f "$out/config.json" ]] || { echo "no trained model in $out" >&2; exit 1; }

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for emode in ${MODES:-raw oracle shuffle mean swaphist}; do
  result="$task_dir/eval_${emode}.json"
  [[ -f "$result" ]] && { echo "skip $emode (done)"; continue; }
  test_csv="$task_dir/0.csv"; dmode="$emode"
  if [[ "$emode" == swaphist ]]; then
    python "$code_root/scripts/swap_history.py" --input "$task_dir/0.csv" --output "$task_dir/0_swaphist.csv"
    test_csv="$task_dir/0_swaphist.csv"; dmode=raw
  fi
  echo "[$(date --iso-8601=seconds)] ranking diffusion_eval_mode=$emode"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$test_csv" --diffusion_eval_mode "$dmode" \
    --result_json_data "$result"
  [[ "$emode" == raw ]] && cp "$task_dir/diffusion_x0.json" "$task_dir/diffusion_x0_raw.json"
  python "$code_root/src/utils/calc.py" --path "$result" --item_path "$info_file"
done
results="output_dir/diffusion_refine/${name}"
mkdir -p "$results"
cp "$task_dir"/eval_*.json "$task_dir"/diffusion_x0_raw.json "$results/"
ls -d "$out"/checkpoint-* >/dev/null 2>&1 && cp "$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$results/" || true
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
