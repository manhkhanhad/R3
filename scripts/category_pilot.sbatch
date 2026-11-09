#!/usr/bin/env bash
#SBATCH --job-name=R3_category_pilot
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# LLM next-category pilot: Qwen3.5-9B (self-hosted vLLM) predicts the next item's category for 500
# validation rows under variants A (history + taxonomy), B (+ neighbour statistics), B_think (B with
# thinking). Inputs from src/category_pilot/prepare.py; scored on CPU with score.py. Resumable.
# Env: CODE (frozen code root, required), PILOT_DIR, VARIANTS.
set -euo pipefail
CODE=${CODE:?set CODE}
PILOT_DIR=${PILOT_DIR:-/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/category_pilot/valid500}
VARIANTS=${VARIANTS:-A,B,B_think}
PORT=18432
MODEL=Qwen/Qwen3.5-9B
ENV=/common/home/users/m/mk.ngohuu.2024/.conda/envs/racer
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
log() { echo "[$(date -Is)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} code=$CODE pilot=$PILOT_DIR variants=$VARIANTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SERVER_LOG=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/R3_category_pilot_vllm.${SLURM_JOB_ID:-local}.log
"$ENV/bin/vllm" serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --language-model-only --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching > "$SERVER_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    kill -0 $SERVER 2>/dev/null || { log "vLLM exited"; tail -40 "$SERVER_LOG"; exit 1; }
    curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL" && break
    sleep 10
done
curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL" || { log "vLLM not ready"; exit 1; }
log "vLLM ready"

"$ENV/bin/python" -u "$CODE/src/category_pilot/query.py" --pilot-dir "$PILOT_DIR" --variants "$VARIANTS" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --concurrency 32
log "queries done"
/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin/python "$CODE/src/category_pilot/score.py" \
    --pilot_dir "$PILOT_DIR" --variants "$VARIANTS"
log "done"
