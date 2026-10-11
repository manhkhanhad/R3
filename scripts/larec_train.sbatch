#!/usr/bin/env bash
#SBATCH --job-name=R3_larec
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

# LaRec-style latent pre-training on R3 (document/latent_distillation.md), R3 v1 hyperparameters
# (Qwen2.5-1.5B-Instruct, lr 3e-4, batch 128, 3 epochs, early stopping), K latent steps,
# L = CE + alpha*InfoNCE(step states, teacher states) + gamma*direction loss.
# Then constrained-beam ranking on the test set, and with each user's latent chain swapped
# for another user's (first 2,000 rows).
# Usage: sbatch scripts/larec_train.sbatch [full|smoke] [k] [alpha] [gamma]
source scripts/collab_env.sh
mode="${1:-full}"; k="${2:-6}"; alpha="${3:-1.0}"; gamma="${4:-1.0}"
source scripts/distill_paths.sh "$mode"
states=${distill_root}/larec_${mode}_k${k}/train_states.pt
name=larec_k${k}_a${alpha}_g${gamma}

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
echo "[$(date --iso-8601=seconds)] mode=$mode k=$k alpha=$alpha gamma=$gamma states=$states out=$out"

if [[ ! -f "$out/model.safetensors" ]]; then
  python src/latent/latent_attention_train.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct --larec_states "$states" --larec_k "$k" \
    --larec_alpha "$alpha" --larec_gamma "$gamma" \
    --train_file "$(ls ${data_root}/train/*.csv)" --eval_file "$(ls ${data_root}/valid/*.csv)" \
    --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size 2 --batch_size 128 --num_epochs "$epochs" --cutoff_len 512 \
    --save_only_model True --save_total_limit 1
fi
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

test_csv=$(ls ${data_root}/test/*.csv)
evaluate() {  # tag, limit, extra args
  local t=$1 n=$2; shift 2
  [[ -f "$out/eval_$t.json" ]] && { echo "exists: eval_$t"; return; }
  python src/latent/rationale_eval.py --base_model "$out" --info_file "$info_file" --mode larec \
    --test_data_path "$test_csv" --result_json_data "$out/eval_$t.json" --limit "$n" --batch_size 8 "$@" \
    | tee "$out/eval_$t.log"
}
evaluate true "$limit"
evaluate swap "$ctrl_limit" --latent_override swap
echo "[$(date --iso-8601=seconds)] done: $name ($mode)"
