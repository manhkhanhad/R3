#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="a40"
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_collab_InfoNCE

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# Plain R3 only. InfoNCE is active for all three epochs. The non-persistent
# FIFO holds at most 10k recent latents whose complete target-token prediction
# was correct; same-sample stale entries are never eligible as positives.
category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=data/${category}/train/${category}_5_2015-10-2018-11.csv
eval_file=data/${category}/valid/${category}_5_2015-10-2018-11.csv
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir=./output_dir/${category}_R3_collaborative_infonce
out=${task_dir}/latent
mkdir -p ${out}

BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

python src/latent/latent_attention_train_collaborative.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --projection_dim 256 --temperature 0.1 \
    --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
    --bank_size 10000 --max_user_positives 16 \
    --max_target_positives 16 --num_negatives 256 \
    --positive_min_token_accuracy 1.0

echo "[$(date --iso-8601=seconds)] Training finished, starting standard ranking evaluation"

array=(0)
nn=1
python src/utils/split.py --input_path $test_file \
    --output_path ${task_dir} --nn ${nn}

model_ckpt=${out}
for ((i=0; i<nn; i++)); do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ${task_dir}/${i}.csv \
        --result_json_data ${task_dir}/${i}.json &
done
wait

python src/utils/merge.py --input_path ${task_dir}/ \
    --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
    --item_path ${info_file}
