#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_pgr_stage3_eval_v2

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# Re-evaluates the ALREADY-TRAINED Stage 3 checkpoint (job 262317) using
# prototype_eval.py instead of latent_attention_eval.py -- no retraining needed, this
# is purely a corrected evaluation of the existing checkpoint. The original
# stage3_eval/ results used plain LatentModel + generate(), silently dropping the
# prototype weights and never exercising retrieval (see the caveat comments in
# R3_prototype_guided_full.sbatch). prototype_eval.py loads PrototypeGuidedLatentModel,
# attaches the bank, and generate() is made prototype-aware via
# PrototypeGuidedLatentModel.prepare_inputs_for_generation (splices once on the prefill
# step; constrained decoding is unaffected since generate() is still called with real
# input_ids throughout -- see model.py's docstrings). Verified on a tiny synthetic
# model before this job: multi-step generation completes without cache/shape errors,
# real prompt tokens are preserved exactly where constrained decoding needs them, and
# retrieval genuinely fires (test_generation.py).
#
# Output goes to stage3_eval_v2/, separate from the original stage3_eval/, so both can
# be compared directly.

cd /common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/prototype-guided-reasoning
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

category="CDs_and_Vinyl"
base_data_dir=data/${category}
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)

task_dir=./output_dir/${category}_R3_prototype_guided
model_ckpt="${task_dir}/stage3"
eval_dir="${task_dir}/stage3_eval_v2"
mkdir -p "$eval_dir"

if [[ ! -s "${model_ckpt}/model.safetensors" ]]; then
  echo "Missing Stage-3 checkpoint at ${model_ckpt}/model.safetensors" >&2
  exit 1
fi

array=(0)
nn=1
python src/utils/split.py --input_path "$test_file" \
        --output_path "$eval_dir" --nn ${nn}

cd src/prototype_grounding/pgr

for (( i=0; i<nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo "../../../${eval_dir}/${i}.csv"
    python3 prototype_eval.py --batch_size 8 \
        --base_model "../../../${model_ckpt}" --sample -1 \
        --info_file "../../../${info_file}" --category "$category" \
        --test_data_path "../../../${eval_dir}/${i}.csv" \
        --result_json_data "../../../${eval_dir}/${i}.json" &
done
wait

cd ../../..

echo "PROTOTYPE-GUIDED REASONING -- STAGE 3, PROTOTYPE-AWARE EVALUATION (v2)"
python src/utils/merge.py --input_path "${eval_dir}/" \
        --output_path "${eval_dir}/final_result.json" --nn ${nn}
python src/utils/calc.py --path "${eval_dir}/final_result.json" \
        --item_path "$info_file"

echo "[$(date --iso-8601=seconds)] Stage 3 prototype-aware evaluation complete"
