#!/usr/bin/env bash
#SBATCH --job-name=R3_evalcheck
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=06:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/LatentDistillation
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Eval sanity check: score plain R3 on Toys (Toys_and_Games_r3_plain_matched, whose
# R3-eval final_result.json gives HR@10 0.1066) with rationale_eval.py --mode none, the
# script the LaRec runs are scored with.  Agreement means the LaRec gap is real.
source scripts/collab_env.sh
category=Toys_and_Games
data=data/${category}
model=${scratch}/output_dir/Toys_and_Games_r3_plain_matched/latent
out=${scratch}/latent_distill/eval_check_toys
mkdir -p "$out"
python src/latent/rationale_eval.py --base_model "$model" --info_file "$(ls ${data}/info/${category}*.txt)" \
  --mode none --category "$category" --test_data_path "$(ls ${data}/test/*.csv)" \
  --result_json_data "$out/eval_r3_plain.json" --batch_size 8 | tee "$out/eval_r3_plain.log"
