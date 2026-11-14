#!/usr/bin/env bash
#SBATCH --job-name=R3_nolatent_ref
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|l40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# MW-R-H: trained HISTORY-ONLY reference model for the MW-R hinge weights.
# 1) retrain R3-without-latent (the job-344012 recipe: plain_sft_train.py from the frozen no-latent code
#    SFT_CODE_ROOT, base Qwen2.5-1.5B, lr 3e-4, batch 128, <= 3 epochs, early stopping on eval loss) into a NEW dir;
# 2) its ranking eval (sanity: 344012 scored HR@10 0.1045);
# 3) reference_score.py (SCORER_ROOT) NLL_ref(y|x) per train/valid/test row -> reference_nll/<category>_nolatent.
# Env: SCORER_ROOT (required; also the SFT code unless SFT_CODE_ROOT), CATEGORY. plain_sft_train.py = code_24c71a5 recipe with save_total_limit=1.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}" R3_ROOT="$PWD"
sft="${SFT_CODE_ROOT:-$SCORER_ROOT}"
scorer="${SCORER_ROOT:?set SCORER_ROOT to a frozen copy with src/latent/reference_score.py}"
category="${CATEGORY:-Toys_and_Games}"
declare -A noun=([Toys_and_Games]="toys and games" [Musical_Instruments]="music instruments" [CDs_and_Vinyl]="musics")
data="data/${category}"
train_file=$(ls "$data"/train/*.csv); valid_file=$(ls "$data"/valid/*.csv)
test_file=$(ls "$data"/test/*.csv); info_file=$(ls "$data"/info/*.txt)
out="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3_no_latent_ref"
results="output_dir/margin_wider_collab/no_latent_ref/${category}"
ref_dir="output_dir/margin_wider_collab/reference_nll/${category}_nolatent"
mkdir -p "$out/model" "$results"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
mbs=4; [[ "$gpu_name" == *"3090"* ]] && mbs=2
echo "[$(date --iso-8601=seconds)] $category no-latent reference: GPU=$gpu_name mbs=$mbs sft=$sft scorer=$scorer"
if [[ ! -f "$out/model/config.json" ]]; then
  resume=()
  last="$(ls -d "$out"/model/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$last" ]] && resume=(--resume_from_checkpoint "$last")
  python "$sft/src/latent/plain_sft_train.py" \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_file "$train_file" --eval_file "$valid_file" \
    --output_dir "$out/model" --sample -1 \
    --micro_batch_size "$mbs" --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 --learning_rate 3e-4 --early_stopping True \
    --category "${noun[$category]}" --dataset_name "$category" "${resume[@]}"
  rm -rf "$out"/model/checkpoint-*   # final (best) model is saved at $out/model; keep one model only
fi
if [[ ! -f "$results/eval_no_latent.json" ]]; then
  echo "[$(date --iso-8601=seconds)] evaluating"
  python "$sft/src/latent/plain_sft_eval.py" \
    --base_model "$out/model" --info_file "$info_file" \
    --test_data_path "$test_file" --sample -1 \
    --result_json_data "$results/eval_no_latent.json" --batch_size "$mbs" \
    --category "${noun[$category]}" --dataset_name "$category"
fi
for split in train valid test; do
  if [[ -f "$ref_dir/${split}.pt" ]]; then echo "skip scoring $split (done)"; continue; fi
  file=$(ls "$data"/${split}/${category}_5_*.csv)
  echo "[$(date --iso-8601=seconds)] scoring $split"
  python "$scorer/src/latent/reference_score.py" --base_model "$out/model" \
    --file "$file" --split "$split" --out_dir "$ref_dir" --category "$category"
done
echo "[$(date --iso-8601=seconds)] no-latent reference complete"
