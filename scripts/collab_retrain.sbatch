#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_retrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Retrain a checkpoint lost in the ~2026-10-23 scratch purge, with its original recipe
# (Qwen2.5-1.5B-Instruct, 3 epochs, effective batch 128, lr 3e-4, best eval-loss checkpoint).
#   v1      plain R3 SFT                         -> $v1
#   oracle  oracle_cfcluster150 side-info SFT    -> $cluster_oracle
# Usage: sbatch scripts/collab_retrain.sbatch v1|oracle
source scripts/collab_env.sh
which="$1"
case "$which" in
  v1)     out="$v1"; flags=() ;;
  oracle) out="$cluster_oracle"; flags=(--side_info_kind oracle_category --side_info_file "${data}/info/Musical_Instruments_sasrec_cluster150.pt") ;;
  *) echo "unknown model $which" >&2; exit 2 ;;
esac
micro=4; (( is_3090 )) && micro=1
export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_collab_retrain_${which}"
mkdir -p "$out"
resume=()
# Newest complete checkpoint (a save interrupted mid-write has no trainer_state.json).
for ckpt in $(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -nr); do
  if [[ -f "$ckpt/trainer_state.json" && -f "$ckpt/model.safetensors" && -f "$ckpt/optimizer.pt" ]]; then
    resume=(--resume_from_checkpoint "$ckpt"); echo "resuming from $ckpt"; break
  fi
done
echo "[$(date --iso-8601=seconds)] model=$which out=$out micro_batch=$micro"
python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample -1 --micro_batch_size "$micro" --batch_size 128 --num_epochs 3 --cutoff_len 512 \
  --save_total_limit 1 "${flags[@]}" "${resume[@]}"
echo "[$(date --iso-8601=seconds)] done: $out"
