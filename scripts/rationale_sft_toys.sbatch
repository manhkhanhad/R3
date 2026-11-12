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
# Env: CODE (frozen code root, required), MODE, EVAL_MODES, RUN_TAG, OUT_BASE.
set -Eeuo pipefail
CODE=${CODE:?set CODE to the frozen code root}
MODE=${MODE:-own}
# HINT=gold: ORACLE ablation — the target's category path is appended inside the rationale block
# at train and valid time. EVAL_MODES entries are "<mode>[:<hint>]" with hint none|gold|shuffle.
HINT=${HINT:-none}
if [[ $HINT == gold ]]; then
    EVAL_MODES=${EVAL_MODES:-"own:gold own:shuffle own:none empty:gold"}
elif [[ $MODE == own ]]; then EVAL_MODES=${EVAL_MODES:-"own shuffle empty"}; else EVAL_MODES=${EVAL_MODES:-off}; fi
RUN_TAG=${RUN_TAG:-qwen35_9b}
R3=/common/home/users/m/mk.ngohuu.2024/research/R3
# OUT_BASE: where rationale texts and models go (default scratch; R3/output_dir when scratch is full).
SCRATCH=${OUT_BASE:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3}
category=Toys_and_Games
DATA=$R3/data/$category
stem=Toys_and_Games_5_2017-1-2018-11
GEN=$R3/external_rational/results
RAT_DIR=$SCRATCH/rationale_sft/toys_${RUN_TAG}
# RAT_PREBUILT: a directory that already holds train/valid/test.json (e.g. src/latent/v2_reasoning.py);
# the Qwen-bank build step is then skipped.
RAT_DIR=${RAT_PREBUILT:-$RAT_DIR}
OUT=$SCRATCH/output_dir/${category}_rationale_sft_${MODE}_${RUN_TAG}
[[ $HINT == gold ]] && OUT=${OUT}_goldcat
log() { echo "[$(date -Is)] $*"; }

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export HF_HUB_OFFLINE=1 WANDB_PROJECT=R3
cd "$CODE"
log "host=$(hostname) job=${SLURM_JOB_ID:-local} code=$CODE mode=$MODE hint=$HINT eval='$EVAL_MODES' out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ $MODE != off && -n ${RAT_PREBUILT:-} ]]; then
    for split in train valid test; do
        [[ -f $RAT_DIR/$split.json ]] || { log "missing $RAT_DIR/$split.json"; exit 1; }
    done
    log "using prebuilt rationales in $RAT_DIR"
elif [[ $MODE != off ]]; then
    declare -A gen=([train]=$GEN/toys_full_qwen35_9b [valid]=$GEN/toys_valid_qwen35_9b [test]=$GEN/toys_test_qwen35_9b)
    for split in train valid test; do
        python src/latent/rationale_text.py build --generations "${gen[$split]}/generation/generations" \
            --csv "$DATA/$split/$stem.csv" --out "$RAT_DIR/$split.json"
    done
fi
hint_args=()
if [[ $HINT == gold ]]; then
    for split in train valid test; do
        python src/latent/rationale_text.py categories --csv "$DATA/$split/$stem.csv" \
            --meta "$DATA/meta_Toys_and_Games.json" --out "$RAT_DIR/${split}_category.json"
    done
    hint_args=(--hint_dir "$RAT_DIR" --hint_mode gold)
fi

# TRAINING_DONE marks a finished run (the best model is saved in place during training, so
# model.safetensors alone does not mean training finished). Older runs without the marker but with a
# promoted final model can set SKIP_TRAINING=1.
if [[ -f $OUT/TRAINING_DONE || ${SKIP_TRAINING:-0} == 1 ]]; then
    log "training finished already, skipping training"
else
    log "training"
    python -u src/latent/rationale_sft_train.py --category $category \
        --train_file "$DATA/train/$stem.csv" --eval_file "$DATA/valid/$stem.csv" \
        --output_dir "$OUT" --rationale_dir "$RAT_DIR" --mode "$MODE" \
        --batch_size 128 --micro_batch_size 2 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 1536 \
        "${hint_args[@]}"
fi

for spec in $EVAL_MODES; do
    em=${spec%%:*}
    hm=none; [[ $spec == *:* ]] && hm=${spec#*:}
    name=$em; [[ $hm != none || $HINT == gold ]] && name=${em}_hint-${hm}
    log "eval mode=$em hint=$hm"
    python -u src/latent/rationale_sft_eval.py --base_model "$OUT" --category $category \
        --info_file "$DATA/info/$stem.txt" --test_data_path "$DATA/test/$stem.csv" \
        --rationale_file "$RAT_DIR/test.json" --mode "$em" \
        --hint_file "$RAT_DIR/test_category.json" --hint_mode "$hm" \
        --result_json_data "$OUT/eval_${name}.json" --batch_size 8
done
log "done"
