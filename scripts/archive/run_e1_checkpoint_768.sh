#!/usr/bin/env bash

# Complete, resumable E1 run for latent_dev/checkpoint-768.
# Launch safely across terminal disconnects with:
#   nohup bash scripts/run_e1_checkpoint_768.sh >/dev/null 2>&1 &

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

checkpoint="output_dir/CDs_and_Vinyl/latent_dev/checkpoint-768"
tag="latent_dev_checkpoint_768"
result_dir="analysis/results/intervention/$tag"
evidence_dir="analysis/evidence/CDs_and_Vinyl/$tag"
log_dir="analysis/logs/E1_checkpoint_768"
master_log="$log_dir/run.log"

mkdir -p "$result_dir" "$log_dir"
exec >>"$master_log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %z'; }
log() { echo "[$(timestamp)] $*"; }
trap 'status=$?; log "FAILED at line $LINENO (exit $status)"; exit "$status"' ERR

if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  log "Missing checkpoint: $checkpoint/model.safetensors"
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < 1 )); then
  log "No CUDA GPU is visible"
  exit 1
fi
log "Starting/resuming E1 with $gpu_count visible GPU(s)"

artifact_is_full() {
  local summary="$1"
  local expected_n="$2"
  [[ -s "$summary" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("limit") == -1 and d.get("n") == int(sys.argv[2]) else 1)' \
    "$summary" "$expected_n"
}

extract_split() {
  local split="$1"
  local expected_n="$2"
  local gpu="$3"
  local summary="$evidence_dir/$split/summary.json"
  if artifact_is_full "$summary" "$expected_n"; then
    log "Skipping completed $split extraction"
    return
  fi
  # Reuse an extraction started before this resumable driver was launched.
  local process_pattern="src/prototype_grounding/[e]xperiment.py extract --split $split"
  while pgrep -u "$USER" -f "$process_pattern" >/dev/null; do
    log "Waiting for an existing $split extraction"
    sleep 30
  done
  if artifact_is_full "$summary" "$expected_n"; then
    log "Reusing completed $split extraction"
    return
  fi
  log "Extracting $split on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python src/prototype_grounding/experiment.py extract \
      --split "$split" --batch-size 32 \
      --checkpoint "$checkpoint" --tag "$tag" \
      >"$log_dir/extract_${split}.log" 2>&1
  log "Completed $split extraction"
}

# Train and validation extraction are independent. Use both GPUs when available.
if (( gpu_count >= 2 )); then
  extract_split train 49251 0 & train_pid=$!
  extract_split valid 6156 1 & valid_pid=$!
  extract_status=0
  wait "$train_pid" || extract_status=$?
  wait "$valid_pid" || extract_status=$?
  if (( extract_status != 0 )); then
    log "At least one extraction failed; inspect $log_dir/extract_*.log"
    exit "$extract_status"
  fi
else
  extract_split train 49251 0
  extract_split valid 6156 0
fi

nll_summary="$result_dir/valid_summary.json"
if [[ -s "$nll_summary" ]] && python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("n") == 6156 else 1)' \
  "$nll_summary"; then
  log "Skipping completed validation NLL intervention"
else
  log "Running paired validation NLL intervention on GPU 0"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python src/prototype_grounding/experiment.py intervention \
      --split valid --batch-size 32 \
      --checkpoint "$checkpoint" --tag "$tag" \
      >"$log_dir/intervention_valid.log" 2>&1
  log "Completed validation NLL intervention"
fi

modes=(raw mask mean shuffle_user shuffle_target noise_0.25 noise_0.5 noise_1.0)
active_pids=()
active_modes=()

reap_rankings() {
  local next_pids=()
  local next_modes=()
  local i pid mode
  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    mode="${active_modes[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      next_pids+=("$pid")
      next_modes+=("$mode")
    else
      if wait "$pid"; then
        log "Completed test ranking: $mode"
      else
        log "Test ranking failed: $mode (see $log_dir/ranking_${mode}.log)"
        return 1
      fi
    fi
  done
  active_pids=("${next_pids[@]}")
  active_modes=("${next_modes[@]}")
}

for mode in "${modes[@]}"; do
  output="$result_dir/generation_test_all_k32_s42_${mode}_t1_temp0.1_a1.0.json"
  if [[ -s "$output" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("metrics", {}).get("n") == 6157 else 1)' \
    "$output"; then
    log "Skipping completed test ranking: $mode"
    continue
  fi

  while (( ${#active_pids[@]} >= gpu_count )); do
    sleep 10
    reap_rankings
  done
  gpu=$(( ${#active_pids[@]} % gpu_count ))
  log "Starting test ranking on GPU $gpu: $mode"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python src/prototype_grounding/experiment.py generate \
      --split test --bank all --k 32 --seed 42 --mode "$mode" \
      --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
      --checkpoint "$checkpoint" --tag "$tag" \
      >"$log_dir/ranking_${mode}.log" 2>&1 &
  active_pids+=("$!")
  active_modes+=("$mode")
done

while (( ${#active_pids[@]} > 0 )); do
  sleep 10
  reap_rankings
done

date --iso-8601=seconds >"$result_dir/COMPLETE"
log "E1 COMPLETE: results are in $result_dir"
