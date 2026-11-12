#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="a40"
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_vae7a_evalonly

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

# 7a's training (job 258440) completed successfully -- checkpoint-1152 and
# model.safetensors are valid -- but the chained standard eval crashed with a
# dtype mismatch (VAEDecomposedSelfAttentionLayer.mu_head/z_head are nn.Linear,
# and F.linear rejects a float32 input against bf16 weights; only surfaced at
# eval time since Trainer's bf16 autocast hid it during training). Fixed in
# src/latent/decomposed_attention.py. This re-runs only the standard eval
# (no retraining needed) against the already-trained checkpoint.

category="CDs_and_Vinyl"
base_data_dir=data/${category}
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir=./output_dir/CDs_and_Vinyl_R3_vae7a
out=${task_dir}/latent

array=(0)
nn=1
python src/utils/split.py --input_path $test_file \
        --output_path ${task_dir} --nn ${nn}

model_ckpt=${out}
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${task_dir}/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ${task_dir}/${i}.csv \
        --result_json_data ${task_dir}/${i}.json &
done
wait

echo "VAE-DECOMPOSED 7a (mu_consistency=0) EVALUATION (re-run after dtype-bug fix)"
python src/utils/merge.py --input_path ${task_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}
