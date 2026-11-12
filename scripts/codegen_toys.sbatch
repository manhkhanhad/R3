#!/usr/bin/env bash
#SBATCH --job-name=R3_codegen
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out
# Gate 3: generate 4 reasoning codes then the item (src/reasoning_codes/codegen.py). Single best model copy
# (BestModelCallback); training is skipped only when TRAINING_DONE exists; evals: gen / oracle / shuffle.
# Env: CODE (frozen code root), OUT, CODES_DIR, EVAL_MODES.
set -Eeuo pipefail
CODE=${CODE:?}
OUT=${OUT:-/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/reasoning_codes/runs/Toys_codegen_cornac_L4K256}
CODES_DIR=${CODES_DIR:-/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/reasoning_codes/rqvae_cornac_L4_K256_d32}
EVAL_MODES=${EVAL_MODES:-"gen oracle shuffle"}
module purge; module load CUDA/12.4.0
eval "$(conda shell.bash hook)"; conda activate rrec
export HF_HUB_OFFLINE=1 WANDB_PROJECT=R3
cd "$CODE"
log() { echo "[$(date -Is)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} code=$CODE out=$OUT codes=$CODES_DIR eval='$EVAL_MODES'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ ${SMOKE:-0} == 1 ]]; then
    # untrained base model + code tokens; all eval modes on 16 test rows (pipeline check only)
    python -u src/reasoning_codes/codegen.py init_model --output_dir "$OUT"
    for m in gen oracle shuffle; do
        python -u src/reasoning_codes/codegen.py evaluate --model_dir "$OUT" --codes_dir "$CODES_DIR" --mode "$m" \
            --result_json "$OUT/eval_${m}.json" --limit 16
    done
    log "smoke done"; exit 0
fi
if [[ -f $OUT/TRAINING_DONE ]]; then
    log "training finished already, skipping training"
else
    log "training"
    python -u src/reasoning_codes/codegen.py train --output_dir "$OUT" --codes_dir "$CODES_DIR"
fi
for m in $EVAL_MODES; do
    res="$OUT/eval_${m}.json"
    if [[ -f $res ]]; then log "eval $m done already"; continue; fi
    log "eval mode=$m"
    python -u src/reasoning_codes/codegen.py evaluate --model_dir "$OUT" --codes_dir "$CODES_DIR" --mode "$m" \
        --result_json "$res.tmp"
    mv "$res.tmp" "$res"
done
log "done"
