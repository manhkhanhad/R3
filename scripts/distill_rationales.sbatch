#!/usr/bin/env bash
#SBATCH --job-name=R3_rationales
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/LatentDistillation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Latent distillation, stage 0 (document/latent_distillation.md): history-only rationales
# for every row of train/valid/test.
# Usage: sbatch scripts/distill_rationales.sbatch [full|smoke] [model] [tag]
source scripts/collab_env.sh
mode="${1:-full}"; model="${2:-Qwen/Qwen2.5-7B-Instruct}"; tag="${3:-qwen7b}"
source scripts/distill_paths.sh "$mode"
out=${rationales_root}/${tag}
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] mode=$mode model=$model out=$out"
for split in test valid train; do
  [[ -f "$out/$split.json" ]] && { echo "exists: $out/$split.json"; continue; }
  python src/latent_distill/generate_rationales.py --model "$model" \
    --csv "$(ls ${data_root}/${split}/*.csv)" --out "$out/$split.json"
done
echo "[$(date --iso-8601=seconds)] done"
