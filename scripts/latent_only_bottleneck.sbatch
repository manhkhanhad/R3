#!/usr/bin/env bash
#SBATCH --job-name=R3_latent_only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# L7: plain R3 with a structural latent-only bottleneck (RGLT-style). Target tokens
# attend only to the fixed instruction prefix, the thought token r and earlier
# target tokens; the history is hidden in training AND generation
# (src/latent/latent_bottleneck.py, flag --latent_only). Recipe = plain R3 v1.
# Then: mask check, standard constrained 10-beam ranking, r interventions
# (raw / mask / mean / shuffle_user / shuffle_target) on the same checkpoint,
# paired bootstrap raw vs each intervention.
# Env: CATEGORY, CODE_ROOT, OUTPUT_SUFFIX, TRAIN_SAMPLE, NUM_EPOCHS, TEST_SAMPLE,
#      EXTRACT_LIMIT, INTERVENTION_LIMIT, MODES, SKIP_TRAIN.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
category="${CATEGORY:-Musical_Instruments}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_latent_only_${category}${OUTPUT_SUFFIX:-}}"
export WANDB_TAGS="${WANDB_TAGS:-latent-only,bottleneck,rglt}"
# experiment.py resolves data/ and analysis/ against R3_ROOT, code against CODE_ROOT
export R3_ROOT="$PWD"

code_root="${CODE_ROOT:-.}"
echo "[$(date --iso-8601=seconds)] code root: $code_root"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
  micro_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

data_dir="data/${category}"
train_file=$(ls ${data_dir}/train/${category}_5_*.csv)
eval_file=$(ls ${data_dir}/valid/${category}_5_*.csv)
test_file=$(ls ${data_dir}/test/${category}_5_*.csv)
info_file=$(ls ${data_dir}/info/${category}_5_*.txt)
task_dir="output_dir/${category}_R3_latent_only${OUTPUT_SUFFIX:-}"
out="${task_dir}/latent"
tag="$(echo "${category}" | tr '[:upper:]' '[:lower:]')_latent_only${OUTPUT_SUFFIX:-}"
interv_dir="analysis/results/intervention/${tag}"
mkdir -p "$out"

if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
  echo "[$(date --iso-8601=seconds)] Starting ${category} latent-only R3 training"
  python "$code_root/src/latent/latent_attention_train.py" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$out" --category "$category" \
    --sample "${TRAIN_SAMPLE:--1}" --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs "${NUM_EPOCHS:-3}" --cutoff_len 512 \
    --save_only_model True --save_total_limit 2 \
    --latent_only True
fi

echo "[$(date --iso-8601=seconds)] Mask check on the trained checkpoint (fp32)"
python "$code_root/src/latent/latent_bottleneck_check.py" \
  --checkpoint "$out" --test_file "$test_file" --category "$category" --pairs 32

echo "[$(date --iso-8601=seconds)] Standard ranking evaluation (constrained 10-beam)"
python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$out" --sample "${TEST_SAMPLE:--1}" \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" \
  --result_json_data "$task_dir/eval_raw.json"
cp "$task_dir/eval_raw.json" "$out/final_result.json"
python "$code_root/src/utils/calc.py" --path "$task_dir/eval_raw.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] Interventions on r: extract train latents (for mean / shuffle)"
python "$code_root/src/prototype_grounding/experiment.py" extract --split train --batch-size 16 \
  --limit "${EXTRACT_LIMIT:--1}" --checkpoint "$out" --tag "$tag" --category "$category"
modes="${MODES:-raw mask mean shuffle_user shuffle_target}"
for mode in $modes; do
  echo "[$(date --iso-8601=seconds)] intervention mode=$mode"
  python "$code_root/src/prototype_grounding/experiment.py" generate --split test \
    --bank all --k 32 --seed 42 --mode "$mode" --topk 1 \
    --temperature 0.1 --alpha 1.0 --batch-size 16 --limit "${INTERVENTION_LIMIT:--1}" \
    --checkpoint "$out" --tag "$tag" --category "$category"
done

echo "[$(date --iso-8601=seconds)] Paired bootstrap: raw vs interventions"
python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw $(echo "$modes" | sed 's/\braw\b//')
echo "[$(date --iso-8601=seconds)] ${category} latent-only R3 run complete"
