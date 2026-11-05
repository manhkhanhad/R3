#!/usr/bin/env bash
#SBATCH --job-name=R3_op_trace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Operator trace (src/prototype_grounding/latent_operator_trace.py): follow the
# user-specific difference of r (own vs another user's r) through pooling,
# RMSNorm, W_V, attention, W_O, residual and logits, with a random-direction control.
# Env: CODE_ROOT (required), CKPT, TEST_FILE, CATEGORY_NAME, N, OUT.
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1
code_root="${CODE_ROOT:?}"
ckpt="${CKPT:-/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/CDs_and_Vinyl/latent}"
test_file="${TEST_FILE:-data/CDs_and_Vinyl/test/CDs_and_Vinyl_5_2015-10-2018-11.csv}"
out="${OUT:-analysis/results/latent_operator_trace/cds_r3_ori.json}"
python "$code_root/src/prototype_grounding/latent_operator_trace.py" --checkpoint "$ckpt" \
  --test_file "$test_file" --category "${CATEGORY_NAME:-musics}" --n "${N:-300}" --output "$out"
echo "[$(date --iso-8601=seconds)] done"
