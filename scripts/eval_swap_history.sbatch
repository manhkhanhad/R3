#!/usr/bin/env bash
#SBATCH --job-name=R3_swaphist
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# History-swap intervention (scripts/swap_history.py): every test row gets another
# user's history; the row's own target (and so its oracle category) is kept.
# Ranking with constrained 10-beam search on a trained checkpoint.
# Env: CODE_ROOT (required), CKPT (checkpoint dir), TASK_DIR (dir holding split.py's 0.csv;
#      results go there), TAG (output name), CATEGORY_MODE (e.g. oracle; empty for plain R3).
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
code_root="${CODE_ROOT:?}"; ckpt="${CKPT:?}"; task_dir="${TASK_DIR:?}"; tag="${TAG:?}"
info_file=$(ls data/Toys_and_Games/info/Toys_and_Games_5_*.txt)
test_file=$(ls data/Toys_and_Games/test/Toys_and_Games_5_*.csv)
[[ -f "$task_dir/0.csv" ]] || python "$code_root/src/utils/split.py" --input_path "$test_file" --output_path "$task_dir" --nn 1
python "$code_root/scripts/swap_history.py" --input "$task_dir/0.csv" --output "$task_dir/0_swaphist.csv"
mode=(); [[ -n "${CATEGORY_MODE:-}" ]] && mode=(--category_mode "$CATEGORY_MODE")
echo "[$(date --iso-8601=seconds)] ranking $tag (ckpt=$ckpt ${CATEGORY_MODE:+category_mode=$CATEGORY_MODE})"
CUDA_VISIBLE_DEVICES=0 python "$code_root/src/latent/latent_attention_eval.py" \
  --batch_size 8 --base_model "$ckpt" --sample -1 \
  --info_file "$info_file" --category Toys_and_Games \
  --test_data_path "$task_dir/0_swaphist.csv" "${mode[@]}" \
  --result_json_data "$task_dir/eval_${tag}.json"
python "$code_root/src/utils/calc.py" --path "$task_dir/eval_${tag}.json" --item_path "$info_file"
echo "[$(date --iso-8601=seconds)] done"
