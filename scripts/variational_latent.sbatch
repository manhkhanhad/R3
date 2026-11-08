#!/usr/bin/env bash
#SBATCH --job-name=R3_variational
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
# Variational line (research log "Latent utilization", variational framing 2026-11-08):
#   C  history dropout      DROPOUT_P=0.5 [DROPOUT_UNIT=item|token] [DROPOUT_EVAL_P=0]
#   A  variational latent   CVAE=1 [CVAE_DIM=64 KL_BETA=1 KL_FREE=0 KL_WARMUP=0.3 KL_SCHEDULE=linear
#                           PRIOR_PROB=0 AUX_WEIGHT=0 AUX_PRIOR_WEIGHT=0 AUX_TAU=0.05]
# Recipe = plain R3 Toys stage 1 (Qwen2.5-1.5B, lr 3e-4, batch 128, <= 3 epochs, early stopping),
# then constrained 10-beam ranking, r interventions raw / mask / mean / shuffle_user (experiment.py),
# paired bootstrap, latent rank.  C also ranks with the training dropout rate (EVAL_DROPOUT_RATES);
# A also ranks with a sampled prior latent (CVAE_MODE=sample) when CVAE_SAMPLE_EVAL=1.
# Outputs under OUTPUT_ROOT (default R3/output_dir/variational: scratch quota is full), results copied
# to output_dir/variational/<name>/.
# Env: CODE_ROOT (required), OUTPUT_NAME, CATEGORY (Toys_and_Games), BASE_MODEL, NUM_EPOCHS, TRAIN_SAMPLE,
#      VALID_SAMPLE, TEST_SAMPLE, EXTRACT_LIMIT, INTERVENTION_LIMIT, MODES, EVAL_DROPOUT_RATES, MICRO_BATCH.
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
case "$category" in CDs_and_Vinyl) prompt_category=musics ;; Toys_and_Games) prompt_category="toys and games" ;; Musical_Instruments) prompt_category="music instruments" ;; *) echo "no prompt category for $category" >&2; exit 2 ;; esac
dropout_p="${DROPOUT_P:-0}"; dropout_unit="${DROPOUT_UNIT:-item}"; dropout_eval_p="${DROPOUT_EVAL_P:-0}"
cvae="${CVAE:-0}"
tagparts="${category}"
[[ "$dropout_p" != "0" ]] && tagparts="${tagparts}_drop${dropout_unit}${dropout_p}"
[[ "$cvae" == "1" ]] && tagparts="${tagparts}_cvae${CVAE_DIM:-64}_b${KL_BETA:-1}_f${KL_FREE:-0}_aux${AUX_WEIGHT:-0}_pp${PRIOR_PROB:-0}"
name="${OUTPUT_NAME:-$tagparts}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,variational}"
export R3_ROOT="$PWD"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size="${MICRO_BATCH:-4}"
echo "[$(date --iso-8601=seconds)] $name: dropout p=$dropout_p unit=$dropout_unit eval_p=$dropout_eval_p cvae=$cvae GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
item_emb=$(ls ${data_dir}/info/${category}_qwen3emb_1536.pt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/variational}/${name}"
out="${task_dir}/latent"
tag="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
interv_dir="analysis/results/intervention/${tag}"
results="output_dir/variational/${name}"
# task_dir already lives under output_dir/variational by default: keep the summary files beside it
[[ "$(realpath -m "$results")" == "$(realpath -m "$task_dir")" ]] && results="${task_dir}/summary"
mkdir -p "$out" "$results"

cvae_args=()
if [[ "$cvae" == "1" ]]; then
  cvae_args=(--cvae --cvae_dim "${CVAE_DIM:-64}" --kl_beta "${KL_BETA:-1}" --kl_free_nats "${KL_FREE:-0}"
             --kl_warmup "${KL_WARMUP:-0.3}" --kl_schedule "${KL_SCHEDULE:-linear}" --prior_prob "${PRIOR_PROB:-0}"
             --aux_weight "${AUX_WEIGHT:-0}" --aux_prior_weight "${AUX_PRIOR_WEIGHT:-0}" --aux_tau "${AUX_TAU:-0.05}"
             --item_embeddings_file "$item_emb")
fi
if [[ ! -f "$out/config.json" ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" && -f "$last_ckpt/trainer_state.json" && -f "$last_ckpt/optimizer.pt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  echo "[$(date --iso-8601=seconds)] training"
  python "$code_root/src/latent/latent_attention_train_variational.py" \
    --base_model "${BASE_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size "${BATCH_SIZE:-128}" \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512 \
    --save_only_model --save_total_limit 1 \
    --history_dropout_p "$dropout_p" --history_dropout_unit "$dropout_unit" --history_dropout_eval_p "$dropout_eval_p" \
    "${cvae_args[@]}" "${resume[@]}"
fi
[[ -f "$out/config.json" ]] || { echo "no trained model in $out" >&2; exit 1; }
[[ -f "$out/cvae_diagnostics.json" ]] && cp "$out/cvae_diagnostics.json" "$results/" || true

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
rank() {  # rank <suffix> [env assignments...]
  local suffix="$1"; shift
  if [[ ! -f "$task_dir/eval_${suffix}.json" ]]; then
    echo "[$(date --iso-8601=seconds)] ranking ${suffix} ($*)"
    env "$@" CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
      --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
      --info_file "$info_file" --category "$category" \
      --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/eval_${suffix}.json"
  fi
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${suffix}.json" --item_path "$info_file" | tee "$results/metrics_${suffix}.txt"
  cp "$task_dir/eval_${suffix}.json" "$results/"
}
rank raw
for p in ${EVAL_DROPOUT_RATES:-}; do rank "drop${p}" HISTORY_DROPOUT_EVAL_P="$p"; done
[[ "$cvae" == "1" && "${CVAE_SAMPLE_EVAL:-1}" == "1" ]] && rank sample CVAE_MODE=sample
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
echo "[$(date --iso-8601=seconds)] $name complete"
