#!/usr/bin/env bash
# Continue the pinned Musical Instruments Attempt-10 checkpoint, then run
# ordinary ranking and same-history latent interventions on the result.
set -Eeuo pipefail
module purge 2>/dev/null || true
module load CUDA/12.4.0 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/R3
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=${WANDB_MODE:-disabled}

category=Musical_Instruments
data=data/${category}
train_file=${data}/train/${category}_5_2015-1-2018-11.csv
valid_file=${data}/valid/${category}_5_2015-1-2018-11.csv
test_file=${data}/test/${category}_5_2015-1-2018-11.csv
info_file=$(ls -f ${data}/info/${category}*.txt)
base=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_R3_collaborative_infonce/latent
bank=analysis/evidence/${category}/musical_instruments_collaborative_infonce/train
item_emb=${data}/info/${category}_qwen3emb_1536.pt
root=/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3_projected_paired
model=${root}/model
tag=${EVAL_TAG:-musical_instruments_projected_paired}
mkdir -p "$root"
if [[ ${SKIP_TRAIN:-0} != 1 ]]; then
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if (( gpu_memory < 32000 )); then
        train_memory_args=(--micro_batch_size 1 --gradient_checkpointing True)
    else
        train_memory_args=(--micro_batch_size 4)
    fi
    echo "GPU memory: ${gpu_memory} MiB; training args: ${train_memory_args[*]}"
    python src/latent/latent_attention_train_projected_correction.py \
        --base_model "$base" --bank_dir "$bank" --train_file "$train_file" \
        --eval_file "$valid_file" --item_emb_file "$item_emb" --output_dir "$model" \
        --category "$category" --topn 32 --max_delta_ratio 0.1 \
        --pair_weight 0.3 --pair_margin 0.01 "${train_memory_args[@]}"
fi

if [[ ${SKIP_STANDARD_EVAL:-0} != 1 ]]; then
    python src/utils/split.py --input_path "$test_file" --output_path "$root/eval" --nn 1
    python src/latent/latent_attention_eval.py --batch_size 8 --base_model "$model" \
        --sample -1 --info_file "$info_file" --category "$category" \
        --test_data_path "$root/eval/0.csv" --result_json_data "$root/eval/0.json"
    python src/utils/merge.py --input_path "$root/eval/" --output_path "$root/eval/result.json" --nn 1
    python src/utils/calc.py --path "$root/eval/result.json" --item_path "$info_file"
fi

# The controls use the new checkpoint's own train-latent distribution.
if [[ ${SKIP_EXTRACT:-0} != 1 ]]; then
    python src/prototype_grounding/experiment.py extract --split train --batch-size 16 \
        --checkpoint "$model" --tag "$tag" --category "$category"
fi
for mode in raw mean shuffle_user shuffle_target mask; do
    result="analysis/results/intervention/${tag}/generation_test_all_k32_s42_${mode}_t1_temp0.1_a1.0.json"
    if [[ -s "$result" ]]; then
        echo "Skipping completed ${mode}: $result"
        continue
    fi
    python src/prototype_grounding/experiment.py generate --split test \
        --bank all --k 32 --seed 42 --mode "$mode" --topk 1 \
        --temperature 0.1 --alpha 1.0 --batch-size 16 \
        --checkpoint "$model" --tag "$tag" --category "$category"
done
