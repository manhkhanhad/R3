#!/bin/bash
#SBATCH --job-name=R3_grad_coherence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# Test A (2026-11-22): alignment of per-row rank-GRPO gradients on the pooling attention,
# R3_ori vs margin+WIDER vs MW-R-H, same stratified Toys training rows (copy / non-copy).
# Read-only on all checkpoints; writes analysis/results/grad_coherence/<tag>/.
set -euo pipefail
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export PYTHONUNBUFFERED=1
code="${CODE_ROOT:?frozen code copy}"
tag="${TAG:?tag}"
S=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir
if [[ "${SMOKE:-0}" == 1 ]]; then n=4; else n=300; fi
python "$code/src/grpo_attention_tuning/grad_coherence.py" \
  --models "R3_ori=/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/Toys_and_Games/latent" \
           "MW=$S/Toys_and_Games_margin_0.2_wider_0.1/latent" \
           "MW-R-H=$S/Toys_and_Games_margin_0.2_wider_0.1_refweight_nolatent_rank/latent" \
  --category Toys_and_Games \
  --train_file data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv \
  --info_file data/Toys_and_Games/info/Toys_and_Games_5_2017-1-2018-11.txt \
  --out "analysis/results/grad_coherence/${tag}" --n_copy "$n" --n_other "$n"
echo "[$(date --iso-8601=seconds)] ${tag} complete"
