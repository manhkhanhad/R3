#!/usr/bin/env bash
#SBATCH --job-name=R3_guidance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Candidates as guidance vs content (src/latent/candidate_pooling.py), trained
# from the base model with R3's recipe (Qwen2.5-1.5B, lr 3e-4, batch 128, <= 3
# epochs, early stopping), Toys:
#   A  plain R3 (retrained, same recipe)
#   B  + SASRec top-10 candidates (out of fold) in a separate pooling branch
#   C  + 10 random non-history items (guidance without content)
#   D  as B, branch annealed to 0 over the middle third of training; evaluated as plain R3
# Ranking: constrained 10-beam search; modes per run (B: raw shuffle drop;
# C: raw real drop; D: raw guided; A: raw), paired bootstrap vs raw.
# Env: CODE_ROOT (required), RUN (A|B|C|D), OUTPUT_NAME, NUM_EPOCHS, LR, TRAIN_SAMPLE,
#      EVAL_SAMPLE, TEST_SAMPLE, MODES, OUTPUT_ROOT, CANDIDATES.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category=Toys_and_Games
code_root="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
run="${RUN:?set RUN=A|B|C|D}"
candidates="${CANDIDATES:-output_dir/candidate_pooling/prep/Toys_sasrec_candidates.pt}"
name="${OUTPUT_NAME:-${category}_guidance_${run}}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,guidance-scaffold}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2

cand=(--candidate_kind sasrec --candidates_file "$candidates" --n_candidates 10)
case "$run" in
  A) cand=(); default_modes="raw" ;;
  B) default_modes="raw shuffle drop" ;;
  C) cand+=(--candidate_source random); default_modes="raw real drop" ;;
  D) cand+=(--candidate_anneal "0.3333,0.6667"); default_modes="raw guided" ;;
  *) echo "unknown RUN $run" >&2; exit 2 ;;
esac
echo "[$(date --iso-8601=seconds)] $name: run=$run GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

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
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  eval_sample=(); [[ -n "${EVAL_SAMPLE:-}" ]] && eval_sample=(--eval_sample "$EVAL_SAMPLE")
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" "${eval_sample[@]}" \
    --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --learning_rate "${LR:-3e-4}" --cutoff_len 512 \
    --save_only_model True --save_total_limit 2 --pool_diagnostics True \
    "${cand[@]}" "${resume[@]}"
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-$default_modes}; do
  echo "[$(date --iso-8601=seconds)] ranking candidate_mode=$mode"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --candidate_mode "$mode" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
results="output_dir/guidance_scaffold/${name}"
mkdir -p "$results"
cp "$task_dir"/eval_*.json "$results/"
cp "$out"/pool_diagnostics.jsonl "$out"/trainer_state.json "$results/" 2>/dev/null || true
ls -d "$out"/checkpoint-* >/dev/null 2>&1 && cp "$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$results/" || true
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
