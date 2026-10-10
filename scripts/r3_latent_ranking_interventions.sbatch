#!/usr/bin/env bash
#SBATCH --job-name=R3_cmp_e1
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

# E1 ranking interventions (as for v0, decoder_insensitivity_to_latent.md): constrained 10-beam ranking
# on the full test split with the latent kept (raw), zeroed (mask), replaced by the training mean (mean)
# or by another user's training latent (shuffle_user). Needs train latents extracted under <tag>.
# Usage: sbatch scripts/r3_latent_ranking_interventions.sbatch <checkpoint> <tag>
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
checkpoint="$1"; tag="$2"
gen_batch=8
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)" == *3090* ]] && gen_batch=4
cd src/prototype_grounding
for mode in raw mask mean shuffle_user; do
  echo "[$(date --iso-8601=seconds)] $tag $mode"
  python experiment.py generate --split test --bank all --k 1 --seed 42 --mode "$mode" \
    --batch-size "$gen_batch" --tag "$tag" --checkpoint "$checkpoint" --category Musical_Instruments
done
echo "[$(date --iso-8601=seconds)] done"
