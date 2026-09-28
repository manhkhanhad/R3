#!/usr/bin/env bash
#SBATCH --job-name=R3_sideinfo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=16:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Decoder-insensitivity side-information tests (analysis/document/decoder_insensitivity_to_latent.md).
# Usage: sbatch --job-name=R3_side_<variant> scripts/side_info_musical.sbatch <variant> [full|smoke]
# variants: oracle_raw oracle_token null_raw text_control sasrec_raw plain_token
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"

variant="$1"; mode="${2:-full}"
category=Musical_Instruments
data=data/${category}
train_file=${data}/train/Musical_Instruments_5_2015-1-2018-11.csv
eval_file=${data}/valid/Musical_Instruments_5_2015-1-2018-11.csv
cat_file=${data}/info/Musical_Instruments_gte_large_en_v1_5_categories.pt
sasrec=/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/Musical_Instruments_5_2015-1-2018-11/models/SASRec/2026-09-26_00-07-12-933295.pkl
export WANDB_NAME="R3_side_${variant}_${mode}"

eval_flags=()
case "$variant" in
  oracle_raw)   train_flags=(--side_info_kind oracle_category --side_info_file "$cat_file") ;;
  oracle_token) train_flags=(--side_info_kind oracle_category --side_info_file "$cat_file" --side_token_scale True) ;;
  null_raw)     train_flags=(--side_info_kind random_category --side_info_file "$cat_file") ;;
  sasrec_raw)   train_flags=(--side_info_kind sasrec_sequence --side_info_file "$sasrec") ;;
  plain_token)  train_flags=(--side_info_kind none --side_token_scale True) ;;
  text_control) train_flags=(--category_text_file "$cat_file"); eval_flags=(--category_text) ;;
  *) echo "unknown variant $variant" >&2; exit 2 ;;
esac

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4; gen_batch_size=8
# 24 GB 3090: smaller micro-batch (MICRO_BATCH_3090, default 2; 1 if the CE-loss logits OOM); effective batch stays 128
if [[ "$gpu_name" == *"3090"* ]]; then micro_batch_size="${MICRO_BATCH_3090:-2}"; gen_batch_size=4; fi
echo "[$(date --iso-8601=seconds)] variant=$variant mode=$mode GPU=$gpu_name micro_batch=$micro_batch_size"
if [[ "$mode" == smoke ]]; then
  out=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/smoke_side_${variant}/latent
  sample=-1; epochs=0.002; limit=16; gen_limit=8; full=()
  [[ "$variant" != sasrec_raw ]] && { sample=256; epochs=1; }
  export WANDB_MODE=disabled
else
  # OUT_ROOT: where checkpoints go (scratch by default; set to the repo output_dir when scratch is full)
  out_root="${OUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}"
  out=${out_root}/Musical_Instruments_R3_side_${variant}/latent
  sample=-1; epochs=3; limit=2000; gen_limit=1000; full=(--full_ranking)
fi
mkdir -p "$out"
# weights-only checkpoints, keep best + latest (disk quota); resume from a complete earlier checkpoint if present
# (RESUME=0 disables resuming. Note: resuming with a different micro-batch than the checkpoint was saved with
#  made HF Trainer 4.48 skip all remaining steps, see decoder_insensitivity_to_latent.md, Experiment 1 run history)
save_flags=(--save_only_model True --save_total_limit 1)
resume=()
last=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [[ "${RESUME:-1}" == 1 && -n "$last" && -f "$last/trainer_state.json" && -f "$last/model.safetensors" && -f "$last/optimizer.pt" ]]; then
  resume=(--resume_from_checkpoint "$last"); echo "resuming from $last"
fi

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 512 "${train_flags[@]}" "${save_flags[@]}" "${resume[@]}"
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

cd src/prototype_grounding
python side_info_eval.py --checkpoint "$out" --tag "${mode}_${variant}" \
  --limit "$limit" --gen_limit "$gen_limit" --gen_batch_size "$gen_batch_size" "${full[@]}" "${eval_flags[@]}"
echo "[$(date --iso-8601=seconds)] done: $variant ($mode)"
