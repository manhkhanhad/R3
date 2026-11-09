#!/bin/bash
#SBATCH --job-name=R3_collabrec_grpo
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
# R3 stage 2 (stock noise-GRPO: lr 1e-5, beta 0.01, 8 generations, eps 0.2/0.28, batch 256, 1 epoch)
# on a CollabRec stage-1 checkpoint (src/latent/collab_rec_grpo.py); only InjectNet is trained.
# Then ranking with the supports raw / dropped / shuffled (collab_rec_eval.py), as for stage 1.
#   STAGE1 (default: C4 checkpoint)  NAME  CATEGORY  CODE_ROOT  LR
#   SMOKE=1: 256 train prompts, batch 64, save every 2 steps, 64 test rows, raw ranking only.
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}

scratch=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir
category=${CATEGORY:-Toys_and_Games}
stage1=${STAGE1:-${scratch}/${category}_collabrec_motive_text_k10_C4/latent}
name=${NAME:-${category}_collabrec_motive_text_k10_C4_grpo}; [[ "${SMOKE:-0}" == 1 ]] && name="smoke_${name}"
code_root=${CODE_ROOT:-/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollabRec}
smoke=${SMOKE:-0}
if [[ "$smoke" == 1 ]]; then sample=256; batch=64; save_steps=2; test_sample=64; overrides=""
else sample=-1; batch=256; save_steps=200; test_sample=-1; overrides=${EVAL_OVERRIDES-none shuffle}; fi
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-collabrec,grpo-stage2}"
echo "[$(date --iso-8601=seconds)] $name host=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) stage1=$stage1 smoke=$smoke"
git -C "$code_root" log -1 --format='code commit %h %s'
[[ -f "$stage1/config.json" ]] || { echo "missing stage-1 checkpoint $stage1"; exit 1; }

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="${scratch}/${name}"
out="${task_dir}/model"
mkdir -p "$out"

if [[ ! -f "$out/config.json" ]]; then
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [[ -n "$last_ckpt" && -f "$last_ckpt/optimizer.pt" && -f "$last_ckpt/scheduler.pt" ]]; then
    resume=(--resume_from_checkpoint "$last_ckpt"); echo "resuming from $last_ckpt"
  elif [[ -n "$last_ckpt" ]]; then
    echo "no optimizer/scheduler state in $last_ckpt: removing partial checkpoints, training from step 0"
    rm -rf "$out"/checkpoint-*
  fi
  python "$code_root/src/latent/collab_rec_grpo.py" \
    --base_model "$stage1" --train_file "$train_file" --output_dir "$out" --category "$category" \
    --sample "$sample" --batch_size "$batch" --micro_batch_size 8 --lr "${LR:-1e-5}" \
    --num_generations 8 --beta 0.01 --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
    --num_epochs 1 --seed 42 --save_steps "$save_steps" --wandb_run_name "$name" \
    "${resume[@]}" || exit 1
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for override in raw $overrides; do
  ov=(); [[ "$override" != raw ]] && ov=(--support_override "$override")
  echo "[$(date --iso-8601=seconds)] ranking supports=$override"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/collab_rec_eval.py" \
    --batch_size 8 --base_model "$out" --sample "$test_sample" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --support_split test "${ov[@]}" \
    --result_json_data "$task_dir/eval_${override}.json" || exit 1
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${override}.json" --item_path "$info_file"
done
results="output_dir/collab_rec/${name}"
mkdir -p "$results"
cp "$task_dir"/eval_*.json "$results/"
echo "[$(date --iso-8601=seconds)] $name complete"
