#!/usr/bin/env bash
#SBATCH --job-name=R3_rationale_valtest
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Rationale units for Toys valid/test histories, with the same generator, prompts and
# decoding as the train bank (results/toys_full_qwen35_9b). The job serves its own
# Qwen3.5-9B with vLLM on this GPU (the shared sodag server is down) and talks to it
# on a FIXED localhost URL: base_url is part of the saved generation config, so a
# requeued job must use the identical string for --resume to accept earlier records.
# Targets are never sent to the model (pipeline.build_inputs).
#
# Env: CODE (frozen external_rational copy, required), SPLITS ("valid test"),
#      LIMIT (0 = all rows), OUT_ROOT (R3/external_rational/results), CONC (24).
set -euo pipefail
R3=/common/home/users/m/mk.ngohuu.2024/research/R3
CODE=${CODE:?set CODE to the frozen external_rational directory}
SPLITS=${SPLITS:-"valid test"}
LIMIT=${LIMIT:-0}
OUT_ROOT=${OUT_ROOT:-$R3/external_rational/results}
CONC=${CONC:-24}
PORT=18431
MODEL=Qwen/Qwen3.5-9B
BASE_URL=http://127.0.0.1:${PORT}/v1
ENV=/common/home/users/m/mk.ngohuu.2024/.conda/envs/racer   # vLLM 0.17.1, cu128
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}         # GLIBCXX_3.4.30 for libzmq
export HF_HUB_OFFLINE=1
DATA=$R3/data/Toys_and_Games
log() { echo "[$(date -Is)] $*"; }

log "host=$(hostname) job=${SLURM_JOB_ID:-local} code=$CODE splits='$SPLITS' limit=$LIMIT out=$OUT_ROOT conc=$CONC"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

SERVER_LOG=$R3/run_logs/R3_rationale_vllm.${SLURM_JOB_ID:-local}.log
"$ENV/bin/vllm" serve "$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --language-model-only \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > "$SERVER_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
    if ! kill -0 $SERVER 2>/dev/null; then log "vLLM exited during startup"; tail -40 "$SERVER_LOG"; exit 1; fi
    curl -s --noproxy '*' -m 5 "$BASE_URL/models" | grep -q "$MODEL" && break
    sleep 10
done
curl -s --noproxy '*' -m 5 "$BASE_URL/models" | grep -q "$MODEL" || { log "vLLM not ready after 30 min"; tail -40 "$SERVER_LOG"; exit 1; }
log "vLLM ready at $BASE_URL (log $SERVER_LOG)"

status=0
for split in $SPLITS; do
    out=$OUT_ROOT/toys_${split}_qwen35_9b
    # Repeating the command retries failed records; two passes per split.
    for pass in 1 2; do
        log "split=$split pass=$pass -> $out"
        set +e
        "$ENV/bin/python" -u "$CODE/run_full_dataset.py" \
            --output-dir "$out" --split "$split" \
            --train "$DATA/$split/Toys_and_Games_5_2017-1-2018-11.csv" \
            --metadata "$DATA/meta_Toys_and_Games.json" \
            --base-url "$BASE_URL" --model "$MODEL" \
            --selection all-prefixes --limit "$LIMIT" --min-history 1 --max-history 20 \
            --concurrency "$CONC" --resume 2>&1
        rc=$?
        set -e
        log "split=$split pass=$pass rc=$rc"
        [[ $rc -eq 0 ]] && break
    done
    # rc=1 also means "some records failed validation" (~8% on train): only a missing
    # summary counts as a job failure.
    if [[ -f $out/generation/summary.json ]]; then cat "$out/generation/summary.json"; else status=1; fi
done
log "done status=$status"
exit $status
