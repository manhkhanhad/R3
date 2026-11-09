#!/usr/bin/env bash
#SBATCH --job-name=R3_mw_collab
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
# Margin + WIDER (356322 recipe: Qwen2.5-1.5B-Instruct, lr 3e-4, batch 128, 3 epochs, margin 0.2 x1.0,
# WIDER 0.1) + collaborative reasoning, from base. METHOD:
#   infonce  : two-phase L1 model (collaborative_margin_attention.py): epoch 1 margin (+WIDER), epochs 2-3
#              + user/target multi-positive InfoNCE (tau 0.1, 0.1/0.1, queue 10k, token-acc gate GATE=0.5).
#              Eval = 356322 pipeline: ranking, experiment.py interventions raw/mask/mean/shuffle_user,
#              paired bootstrap, latent geometry, operator trace.
#   grounded : grounded_margin.py: r1 = r0 + s*||r0||*dir(donor evidence), 32 time-causal donors (other
#              users' r0 + observed next item, Qwen3 item embeddings), margin on r1, WIDER on r1.
#              Eval: donor controls knn / off / shuffle_outcome and slot interventions (SLOT_MODE on r1)
#              mask / mean / shuffle_user; paired bootstrap vs knn; slot-bank geometry.
# Env: CODE_ROOT (required), METHOD (infonce|grounded), CATEGORY (Toys_and_Games), WEIGHT (0.1), MARGIN,
#      MARGIN_WEIGHT, GATE, NUM_EPOCHS, OUTPUT_ROOT, OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, TEST_SAMPLE,
#      EXTRACT_LIMIT, INTERVENTION_LIMIT, MODES, CONTROLS, SLOT_MODES, TRACE_N, USER_W, TARGET_W, HARD_NEG.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
method="${METHOD:?METHOD=infonce|grounded}"
category="${CATEGORY:-Toys_and_Games}"
weight="${WEIGHT:-0.1}"
case "$category" in CDs_and_Vinyl) prompt_category=musics ;; Toys_and_Games) prompt_category="toys and games" ;; Musical_Instruments) prompt_category="music instruments" ;; *) echo "no prompt category for $category" >&2; exit 2 ;; esac
margin="${MARGIN:-0.2}"; margin_weight="${MARGIN_WEIGHT:-1.0}"
name="${OUTPUT_NAME:-${category}_margin_${margin}_wider_${weight}_${method}}"
# infonce variant B (MWC-I-B): USER_W=0 HARD_NEG=16 (same-user different-target hard negatives for the target head)
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,margin,wider,collaborative,$method}"
export R3_ROOT="$PWD"
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
echo "[$(date --iso-8601=seconds)] $name: method=$method margin=$margin x$margin_weight wider=$weight GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
tag="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
results="output_dir/margin_wider_collab/${name}"
mkdir -p "$out" "$results"

if [[ ! -f "$out/config.json" ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  echo "[$(date --iso-8601=seconds)] training"
  common=(--base_model "${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --train_file "$train_file" --eval_file "$eval_file" --output_dir "$out" --category "$category"
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate 3e-4 --cutoff_len 512
    --margin "$margin" --margin_weight "$margin_weight"
    --wider_weight "$weight" --wider_momentum "${MOMENTUM:-0.99}")
  if [[ "$method" == infonce ]]; then
    python "$code_root/src/latent/latent_attention_train_collaborative_margin.py" "${common[@]}" \
      --margin_only_epochs 1 --phase2_margin_weight "$margin_weight" \
      --projection_dim 256 --temperature 0.1 \
      --user_contrastive_weight "${USER_W:-0.1}" --target_contrastive_weight "${TARGET_W:-0.1}" \
      --hard_negatives "${HARD_NEG:-0}" \
      --bank_size 10000 --max_user_positives 16 --max_target_positives 16 --num_negatives 256 \
      --positive_min_token_accuracy "${GATE:-0.5}" --early_stopping True --eval_sample -1 "${resume[@]}"
  else
    python "$code_root/src/latent/latent_attention_train_grounded_margin.py" "${common[@]}" \
      --item_emb_file "${data_dir}/info/${category}_qwen3emb_1536.pt" --eval_sample -1 \
      --topn 32 --explore 8 --queue_size 20000 --init_scale 0.5 \
      --tau_item 0.07 --tau_util 0.5 --item_weight 0.2 --retrieval_weight 0.1 \
      --slot_bank_rows "${EXTRACT_LIMIT:-10000}" "${resume[@]}"
  fi
fi
[[ -f "$out/config.json" ]] || { echo "no trained model in $out" >&2; exit 1; }
ls -d "$out"/checkpoint-* >/dev/null 2>&1 && cp "$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$results/" || true

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
rank_eval() {  # $1 = output label; env passed through
  if [[ ! -f "$task_dir/eval_$1.json" ]]; then
    echo "[$(date --iso-8601=seconds)] ranking $1"
    CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
      --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
      --info_file "$info_file" --category "$category" \
      --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/eval_$1.json"
  fi
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_$1.json" --item_path "$info_file"
  cp "$task_dir/eval_$1.json" "$results/"
}

if [[ "$method" == infonce ]]; then
  rank_eval raw
  interv_dir="analysis/results/intervention/${tag}"
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
else
  python "$code_root/scripts/latent_rank.py" "$out/slot_bank.pt" "$results/latent_rank.json"
  for c in ${CONTROLS:-knn off shuffle_outcome}; do GROUNDING_MODE="$c" rank_eval "$c"; done
  for s in ${SLOT_MODES:-mask mean shuffle_user}; do GROUNDING_MODE=knn SLOT_MODE="$s" rank_eval "slot_$s"; done
  others=""
  for c in ${CONTROLS:-knn off shuffle_outcome}; do if [[ $c != knn ]]; then others+="$c "; fi; done
  for s in ${SLOT_MODES:-mask mean shuffle_user}; do others+="slot_$s "; done
  python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" knn $others | tee "$results/paired_bootstrap.txt"
fi
echo "[$(date --iso-8601=seconds)] $name complete"
