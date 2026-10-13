#!/usr/bin/env bash
#SBATCH --job-name=R3_hilar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/HiLaR
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# HiLaR re-implementation on R3 (document/hilar.md):
#   stage 1: temporal-guided hierarchical quantizer over frozen user states (training split)
#   stage 2: K = 4 latent steps aligned to the cumulative code embeddings (MSE, lambda 0.1),
#            R3 v1 hyperparameters (Qwen2.5-1.5B-Instruct, lr 3e-4, batch 128, 3 epochs, early stopping)
# then constrained-beam ranking on the test set, and with each user's latent chain swapped
# for another user's (first 2,000 rows).
# Usage: sbatch scripts/hilar.sbatch [full|smoke] [lambda]
source scripts/collab_env.sh
mode="${1:-full}"; lambda="${2:-0.1}"
source scripts/distill_paths.sh "$mode"
train_csv=$(ls ${data_root}/train/*.csv)
quant=${distill_root}/hilar_${mode}_quantizer
name=hilar_l${lambda}

if [[ "$mode" == smoke ]]; then
  out=${distill_root}/smoke_${name}/latent
  epochs=1; limit=16; ctrl_limit=16; qepochs=2
  export WANDB_MODE=disabled
else
  out=${distill_root}/Musical_Instruments_${name}/latent
  epochs=3; limit=-1; ctrl_limit=2000; qepochs=20
fi
mkdir -p "$out" "$quant"
echo "[$(date --iso-8601=seconds)] mode=$mode lambda=$lambda quantizer=$quant out=$out"

[[ -f "$quant/targets.pt" ]] || python src/latent_distill/hilar_quantizer.py --csv "$train_csv" --out "$quant" --epochs "$qepochs"
echo "[$(date --iso-8601=seconds)] stage 1 done"

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_${name}_${mode}"
if [[ ! -f "$out/model.safetensors" ]]; then
  # Preemption: resume from the newest epoch checkpoint (saved with optimizer state).
  resume=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  resume_args=(); [[ -n "$resume" ]] && { echo "resuming from $resume"; resume_args=(--resume_from_checkpoint "$resume"); }
  python src/latent/latent_attention_train.py "${resume_args[@]}" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct --hilar_targets "$quant/targets.pt" --hilar_lambda "$lambda" \
    --train_file "$train_csv" --eval_file "$(ls ${data_root}/valid/*.csv)" \
    --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size 2 --batch_size 128 --num_epochs "$epochs" --cutoff_len 512 \
    --save_only_model False --save_total_limit 1
fi
echo "[$(date --iso-8601=seconds)] stage 2 done; evaluating"

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
