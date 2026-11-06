#!/bin/bash
#SBATCH --job-name=R3_collabrec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --exclude=lunar,analog
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# CollabRec on R3 (src/latent/collab_rec_*.py): thought = attention over collaborative supports.
#   SOURCE=peer|motive|oracle_motive|intuition|none  MODE=text|emb  K=10  CATEGORY=Toys_and_Games|CDs_and_Vinyl
#   CODE_ROOT (default: this worktree)  TRAIN_SAMPLE / EVAL_SAMPLE / TEST_SAMPLE (smoke runs)
#   EVAL_OVERRIDES="none shuffle" adds ranking evals with the supports dropped / taken from another row.
#   IntuRec-matching: MICRO=8  LOSS_AVERAGE=micro_batch  TARGET_FORMAT=end_predict
#   MAX_GRAD_NORM=0 (no clipping, IntuRec; default 1.0)  BETA=0.1 (rationale InfoNCE, needs LOSS_AVERAGE=micro_batch)
#   IntuRec stage 2 (stage 1 bypassed): SOURCE=intuition K=15 ALPHA=1.0 (intuition loss)
#   STOP_AFTER_STEPS=N: loss check run (full LR schedule, stops after N optimizer steps, no eval/save/ranking).
# Example: sbatch --export=ALL,SOURCE=oracle_motive scripts/collab_rec.sbatch   (run from any dir)
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}

category=${CATEGORY:-Toys_and_Games}
source=${SOURCE:-peer}
mode=${MODE:-text}
k=${K:-10}
code_root=${CODE_ROOT:-/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollabRec}
name=${OUTPUT_NAME:-${category}_collabrec_${source}_${mode}_k${k}}
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-collabrec}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=${MICRO:-4}
echo "[$(date --iso-8601=seconds)] $name host=$(hostname) GPU=$gpu_name mbs=$micro_batch_size code=$code_root"
git -C "$code_root" log -1 --format='code commit %h %s'

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ! -f "$out/config.json" ]]; then
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [[ -n "$last_ckpt" && -f "$last_ckpt/optimizer.pt" && -f "$last_ckpt/scheduler.pt" ]]; then
    resume=(--resume_from_checkpoint "$last_ckpt"); echo "resuming from $last_ckpt"
  elif [[ -n "$last_ckpt" ]]; then
    # model-only checkpoints (save_only_model): HF would silently restart AdamW and the LR schedule -> start fresh
    echo "no optimizer/scheduler state in $last_ckpt: removing partial checkpoints, training from step 0"
    rm -rf "$out"/checkpoint-*
  fi
  eval_sample=(); [[ -n "${EVAL_SAMPLE:-}" ]] && eval_sample=(--eval_sample "$EVAL_SAMPLE")
  python "$code_root/src/latent/collab_rec_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" "${eval_sample[@]}" \
    --support_source "$source" --support_mode "$mode" --support_k "$k" \
    --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate "${LR:-3e-4}" --cutoff_len 512 \
    --save_only_model True --save_total_limit 2 --wandb_run_name "$name" \
    --target_format "${TARGET_FORMAT:-r3}" --loss_average "${LOSS_AVERAGE:-batch}" \
    --stop_after_steps "${STOP_AFTER_STEPS:-0}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" --rationale_beta "${BETA:-0}" \
    --intuition_alpha "${ALPHA:-0}" \
    "${resume[@]}" || exit 1
fi
if [[ "${STOP_AFTER_STEPS:-0}" -gt 0 ]]; then
  echo "[$(date --iso-8601=seconds)] $name check run done"; exit 0
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for override in raw ${EVAL_OVERRIDES:-}; do
  ov=(); [[ "$override" != raw ]] && ov=(--support_override "$override")
  echo "[$(date --iso-8601=seconds)] ranking supports=$override"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/collab_rec_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --support_split test "${ov[@]}" \
    --result_json_data "$task_dir/eval_${override}.json" || exit 1
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${override}.json" --item_path "$info_file"
done
results="output_dir/collab_rec/${name}"
mkdir -p "$results"
cp "$task_dir"/eval_*.json "$results/"
ls -d "$out"/checkpoint-* >/dev/null 2>&1 && cp "$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$results/" || true
echo "[$(date --iso-8601=seconds)] $name complete"
