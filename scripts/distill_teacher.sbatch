#!/usr/bin/env bash
#SBATCH --job-name=R3_teacher
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/LatentDistillation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# Latent distillation, stage 1 (document/latent_distillation.md): R3 trained with a
# history-only rationale, from Qwen2.5-1.5B-Instruct with the R3 v1 recipe.
#   latent: only the pooler reads the rationale; the decoder sees the plain R3 prompt + r
#   text:   the rationale is written into the decoder prompt (reference)
# Then constrained-beam ranking on the test set; for latent, also with shuffled / empty
# rationales (does the prediction depend on what r carries?).
# Usage: sbatch scripts/distill_teacher.sbatch [full|smoke] [latent|text] [rationale tag]
source scripts/collab_env.sh
mode="${1:-full}"; rmode="${2:-latent}"; tag="${3:-qwen7b}"
source scripts/distill_paths.sh "$mode"
rationales=${rationales_root}/${tag}
name=teacher_${rmode}_${tag}
base=Qwen/Qwen2.5-1.5B-Instruct

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_${name}_${mode}"
if [[ "$mode" == smoke ]]; then
  out=${distill_root}/smoke_${name}/latent
  epochs=1; limit=16; ctrl_limit=16
  export WANDB_MODE=disabled
else
  out=${distill_root}/Musical_Instruments_${name}/latent
  epochs=3; limit=-1; ctrl_limit=2000
fi
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] mode=$mode rationale_mode=$rmode rationales=$rationales out=$out"

if [[ ! -f "$out/model.safetensors" ]]; then
  python src/latent/latent_attention_train.py \
    --base_model "$base" --rationale_dir "$rationales" --rationale_mode "$rmode" \
    --train_file "$(ls ${data_root}/train/*.csv)" --eval_file "$(ls ${data_root}/valid/*.csv)" \
    --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size 2 --batch_size 128 --num_epochs "$epochs" --cutoff_len 1024 \
    --save_only_model True --save_total_limit 1
fi
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

test_csv=$(ls ${data_root}/test/*.csv)
evaluate() {  # tag, limit, extra args
  local t=$1 n=$2; shift 2
  [[ -f "$out/eval_$t.json" ]] && { echo "exists: eval_$t"; return; }
  python src/latent/rationale_eval.py --base_model "$out" --info_file "$info_file" \
    --test_data_path "$test_csv" --rationale_file "$rationales/test.json" --mode "$rmode" \
    --result_json_data "$out/eval_$t.json" --limit "$n" --batch_size 8 "$@" | tee "$out/eval_$t.log"
}
evaluate true "$limit"
if [[ "$rmode" == latent ]]; then
  evaluate shuffle "$ctrl_limit" --rationale_override shuffle
  evaluate empty "$ctrl_limit" --rationale_override empty
fi
echo "[$(date --iso-8601=seconds)] done: $name ($mode)"
