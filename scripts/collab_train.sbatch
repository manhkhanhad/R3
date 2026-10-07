#!/usr/bin/env bash
#SBATCH --job-name=R3_collab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3CollaborativeReasoning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Collaborative-reasoning LLM variants (document/collaborative_reasoning_experiments.md).
# Usage: sbatch --job-name=R3_collab_<variant> scripts/collab_train.sbatch <variant> [full|smoke]
# variants:
#   cf_text         idea 2: fused-CF top-10 titles written in the prompt
#   cf_slots        idea 3: fused-CF top-10 as SASRec item vectors in <|Cand|> slots
#   residual_text   idea 4: residual-CF top-10 titles in the prompt
#   residual_slots  idea 4: residual-CF top-10 in <|Cand|> slots
#   group_pred      idea 5: predicted CF-cluster distribution (k=150) added to the thought slot
source scripts/collab_env.sh

variant="$1"; mode="${2:-full}"
fused=${tables}/candidates_fused_${build_tag}.pt
residual=${tables}/candidates_residual_${build_tag}.pt
group=${tables}/group_pred150_${build_tag}.pt
item_table=${data}/info/Musical_Instruments_sasrec_item.pt
micro=2; gen_batch=8; cutoff=768
case "$variant" in
  cf_text)        train_flags=(--collab_candidates_file "$fused" --collab_mode text); cutoff=1024; ev=(collab text "$fused") ;;
  cf_slots)       train_flags=(--collab_candidates_file "$fused" --collab_mode slots --collab_item_table "$item_table"); ev=(collab slots "$fused") ;;
  residual_text)  train_flags=(--collab_candidates_file "$residual" --collab_mode text); cutoff=1024; ev=(collab text "$residual") ;;
  residual_slots) train_flags=(--collab_candidates_file "$residual" --collab_mode slots --collab_item_table "$item_table"); ev=(collab slots "$residual") ;;
  group_pred)     train_flags=(--side_info_kind row_table --side_info_file "$group"); ev=(side) ;;
  *) echo "unknown variant $variant" >&2; exit 2 ;;
esac
if (( is_3090 )); then
  micro=2; gen_batch=4
  [[ "$variant" == *_text ]] && micro=1
fi

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_collab_${variant}_${mode}"
if [[ "$mode" == smoke ]]; then
  out=${scratch}/output_dir/smoke_collab_${variant}/latent
  sample=256; epochs=1; limit=16; gen_limit=8; full=()
  [[ "$variant" == group_pred ]] && { sample=-1; epochs=0.002; }
  export WANDB_MODE=disabled
else
  out=${scratch}/output_dir/Musical_Instruments_R3_collab_${variant}/latent
  sample=-1; epochs=3; limit=2000; gen_limit=1000; full=(--full_ranking)
fi
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] variant=$variant mode=$mode micro_batch=$micro cutoff=$cutoff out=$out"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len "$cutoff" "${train_flags[@]}" \
  --save_only_model True --save_total_limit 1
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

cd src/prototype_grounding
if [[ "${ev[0]}" == collab ]]; then
  python collab_eval.py --checkpoint "$out" --tag "${mode}_${variant}" --mode "${ev[1]}" --candidates "../../${ev[2]}" \
    --limit "$limit" --gen_limit "$gen_limit" --gen_batch_size "$gen_batch" "${full[@]}"
else
  python side_info_eval.py --checkpoint "$out" --tag "collab_${mode}_${variant}" \
    --limit "$limit" --gen_limit "$gen_limit" --gen_batch_size "$gen_batch" "${full[@]}"
fi
echo "[$(date --iso-8601=seconds)] done: $variant ($mode)"
