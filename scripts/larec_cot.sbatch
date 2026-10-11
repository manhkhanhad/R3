#!/usr/bin/env bash
#SBATCH --job-name=R3_larec_cot
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

# LaRec-style latent pre-training, data (document/latent_distillation.md): target-aware
# K-step CoT for the training split, then the frozen backbone's hidden state after each step.
# Usage: sbatch scripts/larec_cot.sbatch [full|smoke] [k]
source scripts/collab_env.sh
mode="${1:-full}"; k="${2:-6}"
source scripts/distill_paths.sh "$mode"
out=${distill_root}/larec_${mode}_k${k}
mkdir -p "$out"
train_csv=$(ls ${data_root}/train/*.csv)
echo "[$(date --iso-8601=seconds)] mode=$mode k=$k out=$out"
[[ -f "$out/train_cot.json" ]] || python src/latent_distill/generate_cot.py --model Qwen/Qwen2.5-7B-Instruct \
  --csv "$train_csv" --out "$out/train_cot.json" --k "$k"
[[ -f "$out/train_states.pt" ]] || python src/latent_distill/teacher_states.py \
  --cot "$out/train_cot.json" --csv "$train_csv" --out "$out/train_states.pt"
echo "[$(date --iso-8601=seconds)] done"
