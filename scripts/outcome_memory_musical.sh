#!/usr/bin/env bash
# Outcome-grounded donor memory; run inside an allocated GPU job.
set -Eeuo pipefail
cd /common/home/users/m/mk.ngohuu.2024/research/R3
python_bin=${PYTHON_BIN:-/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin/python}
category=Musical_Instruments
data_dir=data/${category}
base=${BASE_MODEL:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_R3_collaborative_infonce/latent}
bank=${BANK_DIR:-analysis/evidence/${category}/musical_instruments_collaborative_infonce/train}
value_mode=${MEMORY_VALUE_MODE:-outcome}
output=${OUTPUT_DIR:-/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3_outcome_memory_${value_mode}}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONUNBUFFERED=1
# Ordinary CE first: paired margin previously raised sensitivity without ranking gains.
"$python_bin" src/latent/latent_attention_train_projected_correction.py \
  --base_model "$base" --bank_dir "$bank" \
  --train_file "$data_dir/train/${category}_5_2015-1-2018-11.csv" \
  --eval_file "$data_dir/valid/${category}_5_2015-1-2018-11.csv" \
  --item_emb_file "$data_dir/info/${category}_qwen3emb_1536.pt" \
  --output_dir "$output" --category "$category" \
  --outcome_grounded True --memory_value_mode "$value_mode" --retrieval_space "${RETRIEVAL_SPACE:-auto}" \
  --pair_weight 0 --topn 32 --max_delta_ratio 0.1 \
  --micro_batch_size "${MICRO_BATCH_SIZE:-1}" --gradient_checkpointing True \
  --num_epochs "${NUM_EPOCHS:-1}" --sample "${TRAIN_SAMPLE:--1}" --eval_sample "${EVAL_SAMPLE:--1}"
# Hold history/checkpoint fixed while changing memory values or donor relevance.
for control in outcome latent_only shuffle_outcome off random; do
  retrieval=knn
  values=$control
  if [[ $control == off || $control == random ]]; then
    retrieval=$control
    values=outcome
  fi
  NEIGHBOR_REFINEMENT_MODE=$retrieval OUTCOME_MEMORY_VALUE_MODE=$values \
    "$python_bin" src/latent/latent_attention_eval.py --base_model "$output" \
    --test_data_path "$data_dir/test/${category}_5_2015-1-2018-11.csv" \
    --info_file "$data_dir/info/${category}_5_2015-1-2018-11.txt" \
    --category "$category" --sample "${TEST_SAMPLE:--1}" --batch_size "${EVAL_BATCH_SIZE:-4}" \
    --result_json_data "$output/eval_${control}.json"
  "$python_bin" src/utils/calc.py --path "$output/eval_${control}.json" \
    --item_path "$data_dir/info/${category}_5_2015-1-2018-11.txt"
done
