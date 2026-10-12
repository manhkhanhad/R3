#!/usr/bin/env bash
#SBATCH --job-name=R3_hilar_grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/HiLaR
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# HiLaR stage 3 (document/hilar.md): GRPO on the latent policy of a stage-2 checkpoint.
#   probe: 50 batches without updates; reports the within-group reward spread (is there a signal?)
#   train: GRPO (paper: lr 5e-6, 4 users x 6 rollouts), then test ranking + latent-swap control
# Usage: sbatch scripts/hilar_grpo.sbatch [probe|train] [full|smoke] [lambda] [steps]
source scripts/collab_env.sh
action="${1:-probe}"; mode="${2:-full}"; lambda="${3:-0.1}"; steps="${4:-2000}"
source scripts/distill_paths.sh "$mode"
train_csv=$(ls ${data_root}/train/*.csv)
quant=${distill_root}/hilar_${mode}_quantizer
if [[ "$mode" == smoke ]]; then
  stage2=${distill_root}/smoke_hilar_l${lambda}/latent; limit=16; ctrl_limit=16; steps=4
else
  stage2=${distill_root}/Musical_Instruments_hilar_l${lambda}/latent; limit=-1; ctrl_limit=2000
fi
out=${stage2%/latent}_grpo/latent
echo "[$(date --iso-8601=seconds)] action=$action mode=$mode stage2=$stage2 out=$out steps=$steps"

if [[ "$action" == probe ]]; then
  python src/latent/hilar_grpo.py --checkpoint "$stage2" --targets "$quant/targets.pt" --train_csv "$train_csv" \
    --info_file "$info_file" --output_dir "${out%/latent}/probe" --probe_only True --steps "$([[ $mode == smoke ]] && echo 4 || echo 50)" --log_every 5
  exit 0
fi

mkdir -p "$out"
[[ -f "$out/model.safetensors" ]] || python src/latent/hilar_grpo.py --checkpoint "$stage2" --targets "$quant/targets.pt" \
  --train_csv "$train_csv" --info_file "$info_file" --output_dir "$out" --steps "$steps"
echo "[$(date --iso-8601=seconds)] GRPO done; evaluating"
test_csv=$(ls ${data_root}/test/*.csv)
evaluate() {
  local t=$1 n=$2; shift 2
  [[ -f "$out/eval_$t.json" ]] && { echo "exists: eval_$t"; return; }
  python src/latent/rationale_eval.py --base_model "$out" --info_file "$info_file" --mode larec \
    --test_data_path "$test_csv" --result_json_data "$out/eval_$t.json" --limit "$n" --batch_size 8 "$@" \
    | tee "$out/eval_$t.log"
}
evaluate true "$limit"
evaluate swap "$ctrl_limit" --latent_override swap
echo "[$(date --iso-8601=seconds)] done: GRPO ($mode)"
