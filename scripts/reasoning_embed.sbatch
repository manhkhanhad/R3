#!/usr/bin/env bash
#SBATCH --job-name=R3_reasoning_embed
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out
# Embed GEARec v2 reasoning + history text with mxbai (src/reasoning_geometry/embed.py). Env: CODE, OUT.
set -euo pipefail
CODE=${CODE:?set CODE}
OUT=${OUT:-/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/reasoning_geometry/mxbai}
ENV=/common/home/users/m/mk.ngohuu.2024/.conda/envs/racer
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-} HF_HUB_OFFLINE=1
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} out=$OUT"
"$ENV/bin/python" -u "$CODE/src/reasoning_geometry/embed.py" --out "$OUT"
echo "[$(date -Is)] done"
