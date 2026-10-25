#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_drift
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Queue-staleness (drift) measurement for collaborative InfoNCE (Attempt 10).
# Trains Attempt 10 with a model-only checkpoint every SAVE_STEPS optimizer steps (all kept),
# then re-encodes a fixed set of training samples with every checkpoint and measures how far
# a stored latent drifts from its current value (scripts/latent_drift_analysis.py).
# No ranking evaluation.  Env: CATEGORY, GATE (token-accuracy gate), EPOCHS, SAVE_STEPS,
# CODE_ROOT, TAG, SKIP_TRAIN.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="${CATEGORY:?set CATEGORY}"; code="${CODE_ROOT:?set CODE_ROOT}"; tag="${TAG:?set TAG}"
gate="${GATE:-0.5}"; epochs="${EPOCHS:-2}"; save_steps="${SAVE_STEPS:-100}"
export WANDB_PROJECT="${WANDB_PROJECT:-R3}" WANDB_NAME="${WANDB_NAME:-R3_collab_drift_${tag}}"
data="data/${category}"
train_file=$(ls ${data}/train/${category}_5_*.csv); eval_file=$(ls ${data}/valid/${category}_5_*.csv)
out="output_dir/${category}_R3_collab_drift_${tag}"
mkdir -p "$out"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro=4; [[ "$gpu_name" == *"3090"* ]] && micro=2
echo "[$(date --iso-8601=seconds)] GPU $gpu_name; micro $micro; gate $gate; epochs $epochs; save every $save_steps; code $code"

if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  python "$code/src/latent/latent_attention_train_collaborative.py" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$train_file" --eval_file "$eval_file" --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size "$micro" --batch_size 128 --num_epochs "$epochs" --cutoff_len 512 \
    --projection_dim 256 --temperature 0.1 \
    --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
    --bank_size 10000 --max_user_positives 16 --max_target_positives 16 --num_negatives 256 \
    --positive_min_token_accuracy "$gate" \
    --save_steps "$save_steps" --save_total_limit 0 --save_only_model True
fi

echo "[$(date --iso-8601=seconds)] drift analysis"
python "$code/scripts/latent_drift_analysis.py" --run_dir "$out" --train_file "$train_file" \
  --category "$category" --out "analysis/results/drift/${tag}.json"
echo "[$(date --iso-8601=seconds)] done"
