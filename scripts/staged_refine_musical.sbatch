#!/usr/bin/env bash
#SBATCH --job-name=R3_refine
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

# Staged-refinement R3 with a SASRec-cluster head (src/latent/staged_refine_attention.py).
# Usage: sbatch --job-name=R3_refine_<mode> scripts/staged_refine_musical.sbatch <vrec|centroid> <warmup_epochs> [full|smoke]
#   epochs 1-2: R3 thought + independent cluster head; epochs 3-4: thought refined by the cluster prediction.
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

refine_mode="$1"; warmup="$2"; mode="${3:-full}"
# dataset: CATEGORY=Musical_Instruments (default) | Toys_and_Games
category="${CATEGORY:-Musical_Instruments}"
data=data/${category}
case "$category" in
  Musical_Instruments) prefix=Musical_Instruments_5_2015-1-2018-11 ;;
  Toys_and_Games) prefix=Toys_and_Games_5_2017-1-2018-11 ;;
  *) echo "unknown CATEGORY $category" >&2; exit 1 ;;
esac
train_file=${data}/train/${prefix}.csv
eval_file=${data}/valid/${prefix}.csv
test_file=${data}/test/${prefix}.csv
info_file=${data}/info/${prefix}.txt
cluster_file=${data}/info/${category}_sasrec_k60_refine.pt
name="refine_${refine_mode}_w${warmup}"
export WANDB_NAME="R3_${name}_${mode}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4; eval_batch_size=8
if [[ "$gpu_name" == *"3090"* ]]; then micro_batch_size="${MICRO_BATCH_3090:-1}"; eval_batch_size=4; fi
echo "[$(date --iso-8601=seconds)] refine=$refine_mode warmup=$warmup mode=$mode GPU=$gpu_name micro_batch=$micro_batch_size"

# home quota is nearly full: training outputs go to scratch (OUTPUT_ROOT overrides)
root="${OUTPUT_ROOT:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir}"
if [[ "$mode" == smoke ]]; then
  task_dir=${root}/smoke_${category}_${name}; sample=256; epochs=2; stage1=1; test_limit=16; export WANDB_MODE=disabled
  diag_args=(--limit 16 --gen_limit 8)
else
  task_dir=${root}/${category}_R3_${name}; sample=-1; epochs=4; stage1=2; test_limit=-1
  diag_args=(--limit 2000 --gen_limit 2000)
fi
out=${task_dir}/latent
mkdir -p "$out"
# checkpoints carry no optimizer state, so a requeued (preempted) job restarts training
# from scratch: drop partial checkpoints and the per-epoch log of the previous attempt
echo "[$(date --iso-8601=seconds)] restart count: ${SLURM_RESTART_COUNT:-0}"
rm -rf "$out"/checkpoint-* "$out/epoch_metrics.jsonl"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro_batch_size" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 512 \
  --refine_cluster_file "$cluster_file" --refine_mode "$refine_mode" \
  --refine_stage1_epochs "$stage1" --refine_warmup_epochs "$warmup" --refine_loss_weight 0.1 \
  --save_only_model True --save_total_limit "$((epochs - stage1 + 1))"

# end-of-stage-1 checkpoint (epoch == stage1)
stage1_ckpt=$(python - "$out" "$stage1" <<'EOF'
import glob, json, os, sys
out, stage1 = sys.argv[1], float(sys.argv[2])
for d in sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1])):
    # epoch is logged as e.g. 1.999 when the last step of an epoch is a partial accumulation
    if abs(json.load(open(os.path.join(d, "trainer_state.json")))["epoch"] - stage1) < 0.01:
        print(d); break
EOF
)
echo "[$(date --iso-8601=seconds)] stage-1 checkpoint: ${stage1_ckpt:-NOT FOUND}"

if [[ "$mode" == smoke ]]; then head -n $((test_limit + 1)) "$test_file" > "$task_dir/test_subset.csv"; test_file="$task_dir/test_subset.csv"; fi
standard_eval() {  # <model_dir> <result_dir>
  mkdir -p "$2"
  python src/utils/split.py --input_path "$test_file" --output_path "$2" --nn 1
  python src/latent/latent_attention_eval.py --batch_size "$eval_batch_size" --base_model "$1" --sample -1 \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$2/0.csv" --result_json_data "$2/0.json"
  python src/utils/calc.py --path "$2/0.json" --item_path "$info_file"
}
echo "[$(date --iso-8601=seconds)] standard evaluation: final model (stage 2)"
standard_eval "$out" "$task_dir"
if [[ -n "$stage1_ckpt" ]]; then
  echo "[$(date --iso-8601=seconds)] standard evaluation: end of stage 1"
  standard_eval "$stage1_ckpt" "$task_dir/stage1"
fi

# output dirs have been deleted externally before: keep the small result files in analysis/ too
keep=analysis/results/staged_refine/${mode}_${category}_${name}/run_files
mkdir -p "$keep/stage1"
cp "$task_dir/0.json" "$out/epoch_metrics.jsonl" "$keep/"
[[ -f "$task_dir/stage1/0.json" ]] && cp "$task_dir/stage1/0.json" "$keep/stage1/"

echo "[$(date --iso-8601=seconds)] diagnostics"
cd src/prototype_grounding
python staged_refine_eval.py --category "$category" --checkpoint "$out" --tag "${mode}_${category}_${name}" "${diag_args[@]}"
if [[ -n "$stage1_ckpt" ]]; then
  python staged_refine_eval.py --category "$category" --checkpoint "$stage1_ckpt" --tag "${mode}_${category}_${name}_stage1" "${diag_args[@]}" --skip generation
fi
echo "[$(date --iso-8601=seconds)] done: $name ($mode)"
