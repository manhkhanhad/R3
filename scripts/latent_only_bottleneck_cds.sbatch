#!/usr/bin/env bash
#SBATCH --job-name=R3_latent_only_CDs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/R3_latent_only_CDs.%j.out

# L7 on CDs_and_Vinyl (prepared, not yet submitted). Same pipeline as
# latent_only_bottleneck.sbatch; submit with a frozen code copy, e.g.
#   sbatch --export=ALL,CODE_ROOT=output_dir/latent_only_code/code_<hash> \
#     output_dir/latent_only_code/code_<hash>/scripts/latent_only_bottleneck_cds.sbatch
export CATEGORY=CDs_and_Vinyl
exec bash "${CODE_ROOT:-.}/scripts/latent_only_bottleneck.sbatch"
