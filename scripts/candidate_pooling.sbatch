#!/usr/bin/env bash
#SBATCH --job-name=R3_cand_pool
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
# SASRec-informed pooling of the R3 latent (src/latent/candidate_pooling.py):
# r's pooling also attends to the row's SASRec top-C candidates (out of fold for
# training rows); prompt unchanged.  Fine-tuned from the trained R3 Toys checkpoint.
# STAGE=prep: SASRec candidates + R3 title vectors (shared, written once).
# STAGE=run (default): train KIND (title | sasrec), then constrained 10-beam
# ranking with candidate_mode raw / shuffle / drop and paired bootstrap.
# Env: CODE_ROOT (required), STAGE, KIND, N_CAND, LR, NUM_EPOCHS, SHUFFLE_SEED (control),
#      OUTPUT_NAME, TRAIN_SAMPLE, VALID_SAMPLE, TEST_SAMPLE, MODES, OUTPUT_ROOT, PREP_DIR.
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
base=/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/Toys_and_Games/latent
prep="${PREP_DIR:-output_dir/candidate_pooling/prep}"
candidates="$prep/Toys_sasrec_candidates.pt"
titles="$prep/Toys_r3_title_vectors.pt"
data_dir="data/${category}"
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"

if [[ "${STAGE:-run}" == prep ]]; then
  mkdir -p "$prep"
  echo "[$(date --iso-8601=seconds)] prep: SASRec candidates (GPU=$gpu_name, code=$code_root)"
  [[ -f "$candidates" ]] || python "$code_root/src/latent/sasrec_candidates.py" --data_dir "$data_dir" --output "$candidates"
  echo "[$(date --iso-8601=seconds)] prep: R3 title vectors"
  [[ -f "$titles" ]] || python "$code_root/src/latent/candidate_title_vectors.py" --base_model "$base" \
    --info_file "$info_file" --output "$titles"
  echo "[$(date --iso-8601=seconds)] prep done"
  exit 0
fi

kind="${KIND:?set KIND=title|sasrec}"
n_cand="${N_CAND:-10}"
name="${OUTPUT_NAME:-${category}_cand_pool_${kind}_c${n_cand}}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-$name}"
export WANDB_TAGS="${WANDB_TAGS:-latent-utilization,candidate-pooling}"
micro_batch_size=4
[[ "$gpu_name" == *"3090"* ]] && micro_batch_size=2
shuffle=(); [[ -n "${SHUFFLE_SEED:-}" ]] && shuffle=(--candidate_shuffle_seed "$SHUFFLE_SEED")
echo "[$(date --iso-8601=seconds)] $name: kind=$kind C=$n_cand shuffle=${SHUFFLE_SEED:-none} GPU=$gpu_name mbs=$micro_batch_size code=$code_root"

train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${name}"
out="${task_dir}/latent"
mkdir -p "$out"

if [[ ! -f "$out/config.json" ]]; then
  if [[ -n "${VALID_SAMPLE:-}" ]]; then
    head -n $((VALID_SAMPLE + 1)) "$eval_file" > "$task_dir/valid_sample.csv"; eval_file="$task_dir/valid_sample.csv"
  fi
  resume=()
  last_ckpt=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [[ -n "$last_ckpt" ]] && resume=(--resume_from_checkpoint "$last_ckpt") && echo "resuming from $last_ckpt"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model "$base" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-2}" --learning_rate "${LR:-1e-4}" --cutoff_len 512 \
    --save_only_model True --save_total_limit 1 \
    --candidate_kind "$kind" --candidates_file "$candidates" --title_vectors_file "$titles" \
    --n_candidates "$n_cand" "${shuffle[@]}" "${resume[@]}"
fi

python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
for mode in ${MODES:-raw shuffle drop}; do
  echo "[$(date --iso-8601=seconds)] ranking candidate_mode=$mode"
  CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
    --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$task_dir/0.csv" --candidate_mode "$mode" \
    --result_json_data "$task_dir/eval_${mode}.json"
  python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${mode}.json" --item_path "$info_file"
done
results="output_dir/candidate_pooling/${name}"
mkdir -p "$results"
cp "$task_dir"/eval_*.json "$results/"
others=$(ls "$task_dir"/eval_*.json | xargs -n1 basename | sed 's/^eval_//; s/\.json$//' | grep -v '^raw$' || true)
[[ -n "$others" ]] && python "$code_root/scripts/paired_bootstrap_eval.py" "$task_dir" raw $others
echo "[$(date --iso-8601=seconds)] $name complete"
