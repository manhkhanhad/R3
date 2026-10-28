#!/usr/bin/env bash
#SBATCH --job-name=R3_cfg_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# P1 (latent-utilization line, plan v2): zero-training CFG contrast probe on the
# L7 latent-only checkpoint.  Decodes on s_null + w*(s_mode - s_null) with the
# null branch = train-mean latent in the thought slot (src/prototype_grounding/
# latent_contrast.py).  Measures (i) contrast magnitude, (ii) decoding effect of
# guidance on raw r, (iii) intervention behavior under guidance.  Promotion gate
# for C11 = (ii): some w with guided > plain, paired CI excluding 0.
# Env: LIMIT (rows, default 1500), WLIST (default "1.5 3 5"), CODE_ROOT.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R3_ROOT="$PWD"

category="Musical_Instruments"
code_root="${CODE_ROOT:?set CODE_ROOT to the frozen code copy}"
checkpoint="output_dir/${category}_R3_latent_only/latent"
src_tag="musical_instruments_latent_only"
tag="${src_tag}_cfgprobe"
interv_dir="analysis/results/intervention/${tag}"
limit="${LIMIT:-1500}"
wlist="${WLIST:-1.5 3 5}"

echo "[$(date --iso-8601=seconds)] code root: $code_root | limit: $limit | w: $wlist"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name"

# Reuse the L7 run's extracted train latents (train mean / shuffle donors).
evidence_dir="analysis/evidence/${category}"
if [[ ! -e "${evidence_dir}/${tag}" ]]; then
  ln -s "${src_tag}" "${evidence_dir}/${tag}"
fi

gen () { # gen <mode> <cfg_w> [limit]
  local mode="$1" w="$2" lim="${3:-$limit}"
  echo "[$(date --iso-8601=seconds)] generate mode=$mode cfg_w=$w limit=$lim"
  python "$code_root/src/prototype_grounding/experiment.py" generate --split test \
    --bank all --k 32 --seed 42 --mode "$mode" --topk 1 \
    --temperature 0.1 --alpha 1.0 --batch-size 8 --limit "$lim" \
    --checkpoint "$checkpoint" --tag "$tag" --category "$category" --cfg-w "$w"
}

echo "[$(date --iso-8601=seconds)] smoke (16 rows, w=3)"
gen raw 3 16
rm -f "$interv_dir"/generation_*_raw_cfg3_t*.json  # remove the smoke file before the real run

echo "[$(date --iso-8601=seconds)] subset baseline (w=0, same rows)"
gen raw 0
for w in $wlist; do
  gen raw "$w"
done
# interventions under guidance at the middle w: mean (doubles as the identity
# check: slot==null => guided==null) and shuffle_user
gen mean 3
gen shuffle_user 3

echo "[$(date --iso-8601=seconds)] paired bootstrap: guidance effect on raw"
wmodes=""
for w in $wlist; do wmodes="$wmodes raw_cfg${w}"; done
python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw $wmodes
echo "[$(date --iso-8601=seconds)] paired bootstrap: interventions under guidance (w=3)"
python "$code_root/scripts/paired_bootstrap_eval.py" "$interv_dir" raw_cfg3 mean_cfg3 shuffle_user_cfg3
echo "[$(date --iso-8601=seconds)] P1 CFG probe complete"
