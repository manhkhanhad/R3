#!/usr/bin/env bash
#SBATCH --job-name=R3_v3_compare
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=00:30:00
#SBATCH --output=analysis/logs/latent_v3_epochs/compare.%j.log

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/R3
export MPLBACKEND=Agg

python src/prototype_grounding/compare_latent_epochs.py
