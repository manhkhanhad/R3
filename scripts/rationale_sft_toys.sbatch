#!/usr/bin/env bash
#SBATCH --job-name=R3_rationale_sft
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Does a text rationale improve recommendation? Latent-free SFT on the R3 prompt (Toys), with
# R3's recipe (lr 3e-4, batch 128, 3 epochs, early stopping, best validation-loss checkpoint).
#   MODE=own : "<rational> {row's own rationale} <end_rational>" where R3 puts <|Thought|>
#   MODE=off : no rationale block (latent-free control)
# Rationales: Qwen3.5-9B units from external_rational (target-free, history + metadata only),
# rendered per CSV row by src/latent/rationale_text.py. Then 10-beam catalog-constrained ranking
# on the full test split under EVAL_MODES (own model: own / shuffle / empty).
# Env: CODE (frozen code root, required), MODE, EVAL_MODES, RUN_TAG.
set -Eeuo pipefail
CODE=${CODE:?set CODE to the frozen code root}
MODE=${MODE:-own}
if [[ $MODE == own ]]; then EVAL_MODES=${EVAL_MODES:-"own shuffle empty"}; else EVAL_MODES=${EVAL_MODES:-off}; fi
RUN_TAG=${RUN_TAG:-qwen35_9b}
R3=/common/home/users/m/mk.ngohuu.2024/research/R3
SCRATCH=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3
category=Toys_and_Games
DATA=$R3/data/$category
stem=Toys_and_Games_5_2017-1-2018-11
GEN=$R3/external_rational/results
RAT_DIR=$SCRATCH/rationale_sft/toys_${RUN_TAG}
OUT=$SCRATCH/output_dir/${category}_rationale_sft_${MODE}_${RUN_TAG}
log() { echo "[$(date -Is)] $*"; }

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export HF_HUB_OFFLINE=1 WANDB_PROJECT=R3
cd "$CODE"
log "host=$(hostname) job=${SLURM_JOB_ID:-local} code=$CODE mode=$MODE eval='$EVAL_MODES' out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ $MODE != off ]]; then
    declare -A gen=([train]=$GEN/toys_full_qwen35_9b [valid]=$GEN/toys_valid_qwen35_9b [test]=$GEN/toys_test_qwen35_9b)
    for split in train valid test; do
        python src/latent/rationale_text.py build --generations "${gen[$split]}/generation/generations" \
            --csv "$DATA/$split/$stem.csv" --out "$RAT_DIR/$split.json"
    done
fi

if [[ -f $OUT/model.safetensors ]]; then
    log "model exists, skipping training"
else
    log "training"
    python -u src/latent/rationale_sft_train.py --category $category \
        --train_file "$DATA/train/$stem.csv" --eval_file "$DATA/valid/$stem.csv" \
        --output_dir "$OUT" --rationale_dir "$RAT_DIR" --mode "$MODE" \
        --batch_size 128 --micro_batch_size 2 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 1536
fi

for em in $EVAL_MODES; do
    log "eval mode=$em"
    python -u src/latent/rationale_sft_eval.py --base_model "$OUT" --category $category \
        --info_file "$DATA/info/$stem.txt" --test_data_path "$DATA/test/$stem.csv" \
        --rationale_file "$RAT_DIR/test.json" --mode "$em" \
        --result_json_data "$OUT/eval_${em}.json" --batch_size 8
done
log "done"
