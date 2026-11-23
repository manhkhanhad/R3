#!/bin/bash
#SBATCH --job-name=R3_hilar_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --output=run_logs/%x.%j.out
# HiLaR-style GRPO signal probe (no update): Gaussian noise on the single latent r of MW-R-H / MW, antithetic
# rollouts, within-group reward spread per sigma (relative to the inter-user distance of r). Read-only on checkpoints.
set -euo pipefail
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec
export PYTHONUNBUFFERED=1
code="${CODE_ROOT:?frozen code copy}"
tag="${TAG:?tag}"
S=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir
if [[ "${SMOKE:-0}" == 1 ]]; then n=4; else n=128; fi
python "$code/src/grpo_attention_tuning/hilar_probe.py" \
  --models "MW-R-H=$S/Toys_and_Games_margin_0.2_wider_0.1_refweight_nolatent_rank/latent" \
           "MW=$S/Toys_and_Games_margin_0.2_wider_0.1/latent" \
  --category Toys_and_Games \
  --train_file data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv \
  --info_file data/Toys_and_Games/info/Toys_and_Games_5_2017-1-2018-11.txt \
  --out "analysis/results/hilar_probe/${tag}" --n_copy "$n" --n_other "$n" --G 12 --sigmas "${SIGMAS:-0.1,0.25,0.5,1.0}"
echo "[$(date --iso-8601=seconds)] ${tag} complete"
