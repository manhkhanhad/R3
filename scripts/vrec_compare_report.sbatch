#!/usr/bin/env bash
#SBATCH --job-name=R3_cmp_report
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3VRecLatentCompare
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Comparison report for the R3-vs-VRec latent study (src/prototype_grounding/vrec_latent_compare.py).
set -Eeuo pipefail
module purge
eval "$(conda shell.bash hook)"
conda activate rrec
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
r3=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/vrec_compare/Musical_Instruments_r3_baseline/latent
vrec=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/vrec_compare/Musical_Instruments_1T
cd src/prototype_grounding
python vrec_latent_compare.py \
  --model "R3 baseline=cmp_r3_baseline:${r3}/final_result.json" \
  --model "VRec stage 1 (warm-up)=cmp_vrec_stage1:${vrec}/pretrain_1T/result.json" \
  --model "VRec stage 2 (verifier, LLM frozen)=cmp_vrec_stage2" \
  --model "VRec final (stage 3)=cmp_vrec_final:${vrec}/vrec_2v_1T/result.json" \
  --out /common/home/users/m/mk.ngohuu.2024/research/R3/analysis/results/vrec_compare/full
