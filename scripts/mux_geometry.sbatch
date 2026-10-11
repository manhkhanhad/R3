#!/usr/bin/env bash
#SBATCH --job-name=R3_mux_geometry
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=04:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3MuxLatent
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Geometric collapse of r for the MUX grid and R3 v1 (document/mux_latent_experiments.md).
source scripts/collab_env.sh
ck=${scratch}/output_dir
cd src/prototype_grounding
python latent_geometry.py --limit 2000 --out "${results}/../mux_latent/geometry.json" --checkpoints \
  "r3_v1=${v1}" \
  "pool_none=${ck}/Musical_Instruments_R3_mux_pool_none/latent" \
  "pool_mux0.1=${ck}/Musical_Instruments_R3_mux_pool_mux0.1/latent" \
  "pool_mux0.5=${ck}/Musical_Instruments_R3_mux_pool_mux0.5/latent" \
  "codi_none=${ck}/Musical_Instruments_R3_mux_codi_none/latent" \
  "codi_mux0.1=${ck}/Musical_Instruments_R3_mux_codi_mux0.1/latent" \
  "codi_mux0.5=${ck}/Musical_Instruments_R3_mux_codi_mux0.5/latent"
echo "[$(date --iso-8601=seconds)] done"
