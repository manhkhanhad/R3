#!/usr/bin/env bash
#SBATCH --job-name=R3_vrec
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

# VRec-style verifier with coarse CF groups (document/collaborative_reasoning_experiments.md, "VRec-style
# coarse groups"): start from R3 v1, nudge r toward the verifier's group prototype, train end to end.
# Usage: sbatch scripts/collab_vrec.sbatch [full|smoke] [k] [weight]
source scripts/collab_env.sh
mode="${1:-full}"; k="${2:-20}"; weight="${3:-0.5}"
groups=${data}/info/Musical_Instruments_sasrec_cluster${k}.pt
[[ -f "$groups" ]] || python src/collab_reasoning/build_group_table.py --k "$k"
name=vrec_k${k}_w${weight}
micro=2; gen_batch=8
(( is_3090 )) && gen_batch=4

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_${name}_${mode}"
if [[ "$mode" == smoke ]]; then
  out=${scratch}/output_dir/smoke_${name}/latent
  sample=256; epochs=1; limit=16; gen_limit=16; ablate=8
  export WANDB_MODE=disabled
else
  out=${scratch}/output_dir/Musical_Instruments_R3_${name}/latent
  sample=-1; epochs=3; limit=2000; gen_limit=-1; ablate=2000
fi
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] mode=$mode k=$k weight=$weight base=$v1 out=$out"

python src/latent/latent_attention_train.py \
  --base_model "$v1" --vrec_groups_file "$groups" --evidence_weight "$weight" \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 768 \
  --save_only_model True --save_total_limit 1
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

cd src/prototype_grounding
python evidence_eval.py --checkpoint "$out" --tag "${mode}_${name}" --r3_predictions "$(dirname "$v1")" \
  --limit "$limit" --gen_limit "$gen_limit" --ablate_limit "$ablate" --gen_batch_size "$gen_batch"
echo "[$(date --iso-8601=seconds)] done: $name ($mode)"
