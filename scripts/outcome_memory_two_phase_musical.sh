#!/usr/bin/env bash
# Two epochs of Attempt-2 margin training, fresh donor bank, then two memory epochs.
set -Eeuo pipefail
cd /common/home/users/m/mk.ngohuu.2024/research/R3
python_bin=${PYTHON_BIN:-/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin/python}
category=Musical_Instruments
data_dir=data/${category}
run_tag=${RUN_TAG:-musical_margin2_memory2}
run_root=${RUN_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${run_tag}}
phase1=$run_root/phase1_margin
phase2=$run_root/phase2_memory
bank=analysis/evidence/$category/$run_tag/train
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export NEIGHBOR_REFINEMENT_MODE=knn OUTCOME_MEMORY_VALUE_MODE=outcome
if [[ ${SKIP_PHASE1:-0} != 1 ]]; then
  "$python_bin" src/latent/latent_attention_train_margin.py \
    --base_model "${PRETRAINED_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --train_file "$data_dir/train/${category}_5_2015-1-2018-11.csv" \
    --eval_file "$data_dir/valid/${category}_5_2015-1-2018-11.csv" \
    --output_dir "$phase1" --category "$category" \
    --num_epochs 2 --margin 0.2 --margin_weight 1.0 \
    --batch_size 128 --micro_batch_size "${PHASE1_MICRO_BATCH_SIZE:-2}" \
    --gradient_checkpointing True --early_stopping_patience 0 \
    --load_best_model_at_end False --save_only_model True \
    --sample "${TRAIN_SAMPLE:--1}" --eval_sample "${EVAL_SAMPLE:--1}" --cutoff_len 512
fi
[[ -s "$phase1/model.safetensors" ]] || { echo "Missing final phase-1 checkpoint" >&2; exit 1; }
# Measure dependence and diversity before attaching memory. Use validation,
# so this diagnostic does not influence choices using test targets.
if [[ ${SKIP_PHASE1_DIAGNOSTIC:-0} != 1 ]]; then
  "$python_bin" src/prototype_grounding/decoder_sensitivity.py \
    --checkpoint "$phase1" --tag "${run_tag}_phase1" --category "$category" \
    --split valid --limit "${DIAGNOSTIC_LIMIT:-2000}" --batch-size "${DIAGNOSTIC_BATCH_SIZE:-1}"
fi
if [[ ${SKIP_BANK_EXTRACTION:-0} != 1 ]]; then
  "$python_bin" src/prototype_grounding/experiment.py extract --split train \
    --checkpoint "$phase1" --tag "$run_tag" --category "$category" \
    --batch-size "${EXTRACT_BATCH_SIZE:-4}"
fi
# Margin checkpoints lack Attempt-10's collaborative projection heads.
# Retrieve by centered history-latent cosine, not randomly initialized heads.
BASE_MODEL="$phase1" BANK_DIR="$bank" OUTPUT_DIR="$phase2" RETRIEVAL_SPACE=history \
  NUM_EPOCHS=2 MEMORY_VALUE_MODE=outcome bash scripts/outcome_memory_musical.sh
