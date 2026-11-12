#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_pgr_full

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# Prototype-guided reasoning: full 3-stage curriculum + standard HR/NDCG ranking
# evaluation for both the Stage-1 baseline and the Stage-3 result, all in one job.
#
# Previously split across 3 dependency-chained jobs because of a time-budget estimate
# that assumed Stage 1 needed close to a full day. That estimate was wrong -- checked
# via `sacct` against the actual historical calibration job (258441, same base model +
# dataset) and via this job's own earlier run, Stage 1's real training loop takes
# ~2h47m (1152 steps @ ~8.7s/step). One job comfortably fits the whole pipeline within
# researchshort's 2-day cap:
#   Stage 1 training  ~3.5h  (2h47m loop + eval-during-training + checkpoint saves)
#   Stage 1 eval       up to ~4h  (beam search, num_beams=10, 6157 test rows -- no
#                                  direct calibration point for this part yet, budgeted
#                                  generously)
#   Stage 2            ~0.5h  (small modules, not the 1.5B backbone)
#   Stage 3 training   ~3.5-4.5h  (same step count as Stage 1 + retrieval/splice overhead)
#   Stage 3 eval        up to ~4h  (same caveat as Stage 1 eval, see below)
#   ------------------------------------------------------------------
#   ~16-17h worst case; 30h budget leaves comfortable margin without approaching the
#   48h partition cap.
#
# IMPORTANT CAVEAT on the Stage 3 eval specifically: latent_attention_eval.py hardcodes
# `from layers import LatentModel` (not PrototypeGuidedLatentModel) and evaluates via
# model.generate() -- beam search, incremental KV-cache decoding. Loading the Stage-3
# checkpoint this way drops the prototype-specific weights (retriever.*,
# proto_projector.*) as unexpected keys, and even if it didn't,
# PrototypeGuidedLatentModel.forward() only splices prototypes on the training-time
# "full pass" (input_ids.size() == attention_mask.size()) -- never during .generate()'s
# incremental steps (see model.py's class docstring: prototype-guided generation is a
# deliberate, documented scope cut, not implemented yet). So the Stage 3 eval numbers
# below measure the fine-tuned *backbone* only (which did train under the combined
# CE + L_recons + L_commit objective, so it's not a meaningless number) -- they are NOT
# a test of whether prototype retrieval improves ranking quality. That needs a
# generation-time-compatible splicing implementation that doesn't exist yet.

cd /common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/prototype-guided-reasoning
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=$(ls -f ${base_data_dir}/train/${category}*.csv)
eval_file=$(ls -f ${base_data_dir}/valid/${category}*.csv)
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

task_dir=./output_dir/${category}_R3_prototype_guided
mkdir -p "$task_dir"

run_ranking_eval() {
  # $1 = checkpoint dir, $2 = eval subdirectory name, $3 = label for the log banner
  local model_ckpt="$1"
  local eval_dir="${task_dir}/$2"
  local label="$3"
  mkdir -p "$eval_dir"

  local array=(0)
  local nn=1
  python src/utils/split.py --input_path "$test_file" \
          --output_path "$eval_dir" --nn ${nn}

  for (( i=0; i<nn; i++ ))
  do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo "${eval_dir}/${i}.csv"
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model "$model_ckpt" --sample -1 \
        --info_file "$info_file" --category "$category" \
        --test_data_path "${eval_dir}/${i}.csv" \
        --result_json_data "${eval_dir}/${i}.json" &
  done
  wait

  echo "PROTOTYPE-GUIDED REASONING -- ${label} EVALUATION"
  python src/utils/merge.py --input_path "${eval_dir}/" \
          --output_path "${eval_dir}/final_result.json" --nn ${nn}
  python src/utils/calc.py --path "${eval_dir}/final_result.json" \
          --item_path "$info_file"
}

# --- Stage 1: backbone as-is, no prototype machinery ---
echo "[$(date --iso-8601=seconds)] Starting Stage 1"
(
  cd src/prototype_grounding/pgr
  python3 train_3stage.py stage1 \
      --base_model "$BASE_MODEL" \
      --train_file "../../../${train_file}" \
      --eval_file "../../../${eval_file}" \
      --output_dir "../../../${task_dir}/stage1" \
      --category "$category" \
      --sample -1 \
      --micro_batch_size 4 --batch_size 128 \
      --num_epochs 3 --cutoff_len 512
)
echo "[$(date --iso-8601=seconds)] Stage 1 complete, running its ranking evaluation"
run_ranking_eval "${task_dir}/stage1" "stage1_eval" "STAGE 1 (BASELINE)"

# --- Stage 2: mapping pretrained alone (frozen Stage-1 backbone) ---
echo "[$(date --iso-8601=seconds)] Starting Stage 2"
(
  cd src/prototype_grounding/pgr
  python3 train_3stage.py stage2 \
      --stage1_checkpoint "../../../${task_dir}/stage1" \
      --train_file "../../../${train_file}" \
      --output_dir "../../../${task_dir}/stage2" \
      --category "$category" \
      --n_heads 3 --k_per_head 3 --lam 0.5 \
      --recon_steps 2000 --recon_lr 1e-3 \
      --commit_steps 1000 --commit_lr 1e-3 \
      --r_sample_size 4096 --r_batch_size 16 \
      --seed 0
)
echo "[$(date --iso-8601=seconds)] Stage 2 complete"

# --- Stage 3: joint fine-tune ---
echo "[$(date --iso-8601=seconds)] Starting Stage 3"
(
  cd src/prototype_grounding/pgr
  python3 train_3stage.py stage3 \
      --stage1_checkpoint "../../../${task_dir}/stage1" \
      --stage2_mapping "../../../${task_dir}/stage2/mapping.pt" \
      --train_file "../../../${train_file}" \
      --eval_file "../../../${eval_file}" \
      --output_dir "../../../${task_dir}/stage3" \
      --category "$category" \
      --sample -1 \
      --micro_batch_size 4 --batch_size 128 \
      --num_epochs 3 --cutoff_len 512 \
      --recon_weight 1.0 --commit_weight 0.1 \
      --seed 0
)
echo "[$(date --iso-8601=seconds)] Stage 3 complete, running its ranking evaluation (backbone-only, see caveat above)"
run_ranking_eval "${task_dir}/stage3" "stage3_eval" "STAGE 3 (BACKBONE-ONLY, PROTOTYPE WEIGHTS NOT EXERCISED -- SEE CAVEAT ABOVE)"

echo "[$(date --iso-8601=seconds)] 3-stage curriculum + evaluation complete"
