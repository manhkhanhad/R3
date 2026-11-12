#!/usr/bin/env bash
#SBATCH --job-name=R3_multiaspect
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/multiAspectR3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Multi-aspect R3 with gated fusion (src/latent/multi_aspect_attention.py).
# Usage: sbatch --job-name=R3_ma_<variant> scripts/multi_aspect_musical.sbatch <confidence|mean> [full|smoke]
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

fusion="$1"; mode="${2:-full}"
category=Musical_Instruments
data=data/${category}
train_file=${data}/train/Musical_Instruments_5_2015-1-2018-11.csv
eval_file=${data}/valid/Musical_Instruments_5_2015-1-2018-11.csv
test_file=${data}/test/Musical_Instruments_5_2015-1-2018-11.csv
info_file=${data}/info/Musical_Instruments_5_2015-1-2018-11.txt
aspect_file=${data}/info/Musical_Instruments_aspects.pt
export WANDB_NAME="R3_multiaspect_${fusion}_${mode}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4; eval_batch_size=8
if [[ "$gpu_name" == *"3090"* ]]; then micro_batch_size="${MICRO_BATCH_3090:-1}"; eval_batch_size=4; fi
echo "[$(date --iso-8601=seconds)] fusion=$fusion mode=$mode GPU=$gpu_name micro_batch=$micro_batch_size"

root=/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir
if [[ "$mode" == smoke ]]; then
  task_dir=${root}/smoke_multiaspect_${fusion}; sample=256; epochs=1; test_limit=32; export WANDB_MODE=disabled
  eval_args=(--limit 16 --valid_limit 8 --test_limit 8)
else
  task_dir=${root}/Musical_Instruments_R3_multiaspect_${fusion}; sample=-1; epochs=3; test_limit=-1
  eval_args=(--limit 2000 --valid_limit 2000)
fi
out=${task_dir}/latent
mkdir -p "$out"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 512 \
  --aspect_file "$aspect_file" --aspect_fusion "$fusion" --aspect_loss_weight 0.1 \
  --save_only_model True --save_total_limit 1

echo "[$(date --iso-8601=seconds)] training finished; standard ranking evaluation"
if [[ "$mode" == smoke ]]; then head -n $((test_limit + 1)) "$test_file" > "$task_dir/test_subset.csv"; test_file="$task_dir/test_subset.csv"; fi
python src/utils/split.py --input_path "$test_file" --output_path "$task_dir" --nn 1
python src/latent/latent_attention_eval.py --batch_size "$eval_batch_size" --base_model "$out" --sample -1 \
  --info_file "$info_file" --category "$category" \
  --test_data_path "$task_dir/0.csv" --result_json_data "$task_dir/0.json"
python src/utils/merge.py --input_path "$task_dir/" --output_path "$out/final_result.json" --nn 1
python src/utils/calc.py --path "$out/final_result.json" --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] multi-aspect evaluation (probes, sensitivity, beam re-scoring)"
cd src/prototype_grounding
python multi_aspect_eval.py --checkpoint "$out" --tag "${mode}_${fusion}" "${eval_args[@]}"
echo "[$(date --iso-8601=seconds)] done: $fusion ($mode)"
