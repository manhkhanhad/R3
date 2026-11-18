#!/usr/bin/env bash
#SBATCH --job-name=R3_motive_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|l40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Motive-structure probe (src/prototype_grounding/motive_probe.py) on several checkpoints.
# Env: CODE_ROOT (required), CKPTS = "tag=checkpoint_dir tag2=dir2 ...", CATEGORY, LIMIT, T.
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 R3_ROOT="$PWD"
code_root="${CODE_ROOT:?}"
for pair in ${CKPTS:?}; do
  tag="${pair%%=*}"; ckpt="${pair#*=}"
  if [[ -f "analysis/results/motive_probe/${tag}.json" ]]; then echo "skip $tag (done)"; continue; fi
  echo "[$(date --iso-8601=seconds)] probe $tag <- $ckpt"
  python "$code_root/src/prototype_grounding/motive_probe.py" --checkpoint "$ckpt" --tag "$tag" \
    --category "${CATEGORY:-Toys_and_Games}" --limit "${LIMIT:-20000}" --t "${T:-0.8}"
done
echo "[$(date --iso-8601=seconds)] done"
