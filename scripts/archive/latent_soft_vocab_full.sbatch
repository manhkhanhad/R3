#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="l40s|a100|l40|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_soft_vocab

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

# Isolated ablation: plain R3 with only its raw pooled latent replaced by a
# top-64, temperature-1.0 mixture of pretrained vocabulary embeddings.
category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=$(ls -f ${base_data_dir}/train/${category}*.csv)
eval_file=$(ls -f ${base_data_dir}/valid/${category}*.csv)
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
port=$((RANDOM % 5000 + 10000))
task_dir=./output_dir/${category}_R3_soft_vocab_k64_t1
out=${task_dir}/latent
mkdir -p ${out}

BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train_soft_vocab.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --soft_vocab_top_k 64 --soft_vocab_temperature 1.0

echo "[$(date --iso-8601=seconds)] Training finished, starting standard ranking evaluation"

array=(0 1)
nn=2
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
