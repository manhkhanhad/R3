#!/usr/bin/env bash
#SBATCH --job-name=R3_qwen3emb_steps
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=04:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/LatentDistillation
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Embed each Toys unit-CoT step text standalone with Qwen3-Embedding-8B (last-token
# pooling, truncated to 1536 = Qwen2.5-1.5B width, L2-normalised; the recipe of
# explanation_embeddings.py), then measure its geometry (cot_geometry.py).
# Uses the racer env for the embedding because Qwen3 needs transformers >= 4.51
# (rrec has 4.48); the geometry analysis runs in rrec.
set -Eeuo pipefail
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
unit=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/latent_distill/larec_unit_toys_k3
keep=/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/latent_distill
states=${unit}/train_states_qwen3emb.pt
conda activate racer
[[ -f "$states" ]] || python src/latent_distill/step_embeddings.py --cot "${unit}/train_cot.json" --out "$states" \
  --model Qwen/Qwen3-Embedding-8B --pooling last --dim 1536 --batch_size 64
mkdir -p "${keep}/states" "${keep}/geometry"
cp "$states" "${keep}/states/"
conda activate rrec
python src/latent_distill/cot_geometry.py --states "$states" \
  --csv data/Toys_and_Games/train/Toys_and_Games_5_2017-1-2018-11.csv --pairs 20000 \
  --out "${keep}/geometry/qwen3emb.json"
