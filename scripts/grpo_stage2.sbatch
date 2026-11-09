#!/usr/bin/env bash
#SBATCH --job-name=R3_grpo_stage2
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

# R3 stage 2 (noise-GRPO, attention-only, LLM frozen) on a stage-1 checkpoint, any category
# (CATEGORY, default Toys_and_Games); outputs on scratch, results copied to output_dir/grpo_stage2/<tag>.
# Known issue (stock R3): reference log-probs use the policy's r, so the KL term is ~0.
#   KL_ANCHOR=1: reference log-probs from the reference model's own (frozen stage-1) r, same
#   noise -> a real KL trust region (beta = BETA, default 0.01).
#   VARIANT=collab  (L9B): + collaborative InfoNCE with the L8b queue (NLL gate 25%,
#                    per-user cap 2, per-target cap 4, dedup cos 0.9), lambda auto-
#                    calibrated over 20 steps to 30% of the GRPO gradient norm.
#   VARIANT=stock   (L9C): stock stage 2 (control).
# Settings: lr 1e-5, beta 0.01, 8 generations, epsilon 0.2/0.28, batch 256, 1 epoch,
# one GPU.  Then: ranking (noise_eval), train-latent geometry, r interventions.
# Env: VARIANT, BASE_MODEL, CODE_ROOT, TAG, SMOKE (1 = 256 prompts, batch 64, test 64 rows,
#      ranking only), SKIP_TRAIN, LR (default 1e-5), EVAL (full | ranking: stop after
#      standard ranking; geometry and interventions can be run later with SKIP_TRAIN=1).

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R3_ROOT="$PWD"
code="${CODE_ROOT:?set CODE_ROOT to a frozen code copy}"
variant="${VARIANT:?set VARIANT to collab or stock}"
base_model="${BASE_MODEL:?set BASE_MODEL to the stage-1 checkpoint}"
tag="${TAG:?set TAG, e.g. musical_l9b_collab}"
smoke="${SMOKE:-0}"
lr="${LR:-1e-5}"
eval_mode="${EVAL:-full}"
beta="${BETA:-0.01}"
kl_args=()
[[ "${KL_ANCHOR:-0}" == 1 ]] && kl_args=(--kl_anchor True)

case "$variant" in
  collab) collab_args=(--collab True) ;;
  stock) collab_args=() ;;
  *) echo "unknown VARIANT=$variant" >&2; exit 1 ;;
esac
if [[ "$smoke" == 1 ]]; then
  sample=256; batch=64; test_sample=64; collab_args+=(--collab_calibration_steps 2)
else
  sample=-1; batch=256; test_sample=-1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_grpo_${tag}}"
export WANDB_TAGS="${WANDB_TAGS:-grpo-stage2,${variant}}"

category="${CATEGORY:-Toys_and_Games}"
data="data/${category}"
train_file=$(ls ${data}/train/${category}_5_*.csv)
eval_file=$(ls ${data}/valid/${category}_5_*.csv)
test_file=$(ls ${data}/test/${category}_5_*.csv)
info_file=$(ls ${data}/info/${category}_5_*.txt)
task_dir="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}/${category}_R3_grpo_${tag}"
out="${task_dir}/model"
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); variant ${variant}; base ${base_model}; code ${code}; smoke ${smoke}; lr ${lr}; beta ${beta}; kl_anchor ${KL_ANCHOR:-0}; eval ${eval_mode}"

if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] GRPO stage 2"
  python "$code/src/grpo_attention_tuning/train_noise_grpo.py" \
    --category "$category" --base_model "$base_model" --output_dir "$out" \
    --train_file "$train_file" --eval_file "$eval_file" --info_file "$info_file" \
    --num_generations 8 --sample "$sample" --num_epochs 1 \
    --batch_size "$batch" --micro_batch_size 8 --lr "$lr" --beta "$beta" --seed 42 \
    --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
    "${collab_args[@]}" "${kl_args[@]}"
fi

echo "[$(date --iso-8601=seconds)] ranking"
python "$code/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
python "$code/src/grpo_attention_tuning/noise_eval.py" \
  --batch_size 8 --base_model "$out" --sample "$test_sample" \
  --info_file "$info_file" --category "$category" --end_k -1 \
  --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/0.json"
python "$code/src/utils/merge.py" --input_path "$task_dir/" --output_path "$out/final_result.json" --nn 1
python "$code/src/utils/calc.py" --path "$out/final_result.json" --item_path "$info_file"
[[ "$smoke" == 1 ]] && { mkdir -p "output_dir/grpo_stage2/${tag}"; cp "$out/final_result.json" "output_dir/grpo_stage2/${tag}/"; echo "[$(date --iso-8601=seconds)] smoke complete"; exit 0; }
[[ "$eval_mode" == ranking ]] && { echo "[$(date --iso-8601=seconds)] ranking-only run complete"; exit 0; }

log_dir="analysis/logs/R3_grpo_${tag}"
result_dir="analysis/results/intervention/${tag}"
mkdir -p "$log_dir" "$result_dir"
echo "[$(date --iso-8601=seconds)] train-latent extraction and geometry"
python "$code/src/prototype_grounding/experiment.py" extract \
  --split train --batch-size 32 --checkpoint "$out" --tag "$tag" --category "$category" \
  >"$log_dir/extract_train.log" 2>&1
python "$code/src/prototype_grounding/experiment.py" latent-distribution \
  --checkpoint "$out" --tag "$tag" --category "$category" --component r \
  >"$log_dir/latent_distribution.log" 2>&1
for mode in raw mask mean shuffle_user; do
  echo "[$(date --iso-8601=seconds)] intervention ${mode}"
  python "$code/src/prototype_grounding/experiment.py" generate \
    --split test --bank all --k 32 --seed 42 --mode "$mode" \
    --topk 1 --temperature 0.1 --alpha 1.0 --batch-size 16 \
    --checkpoint "$out" --tag "$tag" --category "$category" \
    >"$log_dir/ranking_${mode}.log" 2>&1
done
interv_dir="analysis/results/intervention/${tag}"
python "$code/scripts/paired_bootstrap_eval.py" "$interv_dir" raw mask mean shuffle_user
results="output_dir/grpo_stage2/${tag}"
mkdir -p "$results"
cp "$task_dir/0.json" "$out/final_result.json" "$results/" 2>/dev/null || true
cp "$interv_dir"/*.json "$results/" 2>/dev/null || true
date --iso-8601=seconds >"$result_dir/RANKING_COMPLETE"
echo "[$(date --iso-8601=seconds)] ${tag} complete"
