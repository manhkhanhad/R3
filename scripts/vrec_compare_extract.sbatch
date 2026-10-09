#!/usr/bin/env bash
#SBATCH --job-name=R3_cmp_extract
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3VRecLatentCompare
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Latent extraction for the R3-vs-VRec comparison (src/prototype_grounding/vrec_latent_extract.py).
# Usage: sbatch scripts/vrec_compare_extract.sbatch <limit> <splits> <kind:checkpoint:tag> [...]
#   limit -1 = all rows; splits e.g. train,test
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
limit="$1"; splits="$2"; shift 2
echo "[$(date --iso-8601=seconds)] host=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
cd src/prototype_grounding
for spec in "$@"; do
  IFS=: read -r kind checkpoint tag <<< "$spec"
  for split in ${splits//,/ }; do
    echo "[$(date --iso-8601=seconds)] $kind $tag $split"
    python vrec_latent_extract.py --kind "$kind" --checkpoint "$checkpoint" --tag "$tag" --split "$split" --limit "$limit"
  done
done
echo "[$(date --iso-8601=seconds)] done"
