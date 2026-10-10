#!/usr/bin/env bash
#SBATCH --job-name=R3_mux
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3MuxLatent
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# MUX-style category supervision of the R3 thought vector (document/mux_latent_experiments.md).
# Usage: sbatch --job-name=R3_mux_<variant> scripts/mux_train.sbatch <variant> [full|smoke]
# variants: pool_none pool_mux0.1 pool_mux0.5 codi_none codi_mux0.1 codi_mux0.5
#   pool = R3 attention pooling, codi = MLP+LayerNorm on h[thought-1]; muxB = KL weight B (none = 0)
source scripts/collab_env.sh

variant="$1"; mode="${2:-full}"
builder="${variant%%_*}"; sup="${variant#*_}"
case "$builder" in pool|codi) ;; *) echo "unknown builder in $variant" >&2; exit 2 ;; esac
case "$sup" in
  none)  weight=0 ;;
  mux*)  weight="${sup#mux}" ;;
  *) echo "unknown supervision in $variant" >&2; exit 2 ;;
esac
categories=${data}/info/Musical_Instruments_gte_large_en_v1_5_categories.pt
micro=2; gen_batch=8
(( is_3090 )) && gen_batch=4

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_mux_${variant}_${mode}"
if [[ "$mode" == smoke ]]; then
  out=${scratch}/output_dir/smoke_mux_${variant}/latent
  sample=256; epochs=1; limit=16; gen_limit=16
  export WANDB_MODE=disabled
else
  out=${scratch}/output_dir/Musical_Instruments_R3_mux_${variant}/latent
  sample=-1; epochs=3; limit=2000; gen_limit=-1
fi
mkdir -p "$out"
echo "[$(date --iso-8601=seconds)] variant=$variant builder=$builder weight=$weight mode=$mode out=$out"

python src/latent/latent_attention_train.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --latent_builder "$builder" --mux_category_file "$categories" --mux_weight "$weight" \
  --train_file "$train_file" --eval_file "$eval_file" \
  --output_dir "$out" --category "$category" \
  --sample "$sample" --micro_batch_size "$micro" --batch_size 128 \
  --num_epochs "$epochs" --cutoff_len 768 \
  --save_only_model True --save_total_limit 1
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

cd src/prototype_grounding
python mux_eval.py --checkpoint "$out" --tag "${mode}_${variant}" --r3_predictions "$(dirname "$v1")" \
  --limit "$limit" --gen_limit "$gen_limit" --gen_batch_size "$gen_batch"
echo "[$(date --iso-8601=seconds)] done: $variant ($mode)"
