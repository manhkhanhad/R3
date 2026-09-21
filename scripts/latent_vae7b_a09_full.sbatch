#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run_logs/%x.%j.out
#SBATCH --constraint="a40"
#SBATCH --partition=hadylauwresearch
#SBATCH --qos=hadylauwresearch-priority
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=R3_vae7b_a09

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# Run B: 7b with the attention schedule relaxed (alpha_max=0.9), everything else
# identical. Single-variable test of how much of 7b's gap is the schedule's tax
# (alpha=0.75 alone already costs ~10% vs the unrestricted baseline).
# Attempt 7a: VAE-shaped encoder, mu_i/z_i = attention(x), r_i = mu_i + z_i, no
# sampling. ||mu_i|| fixed (rho_mu * sqrt(H)); z regularized toward N(0, I) via
# an aggregate KL over a queue of recent z (mu_consistency_weight=0, pure prior,
# mu_consistency_weight=0.5 adds the identifiability asymmetry: mu_i is
# softly tied toward its own EMA across training, closing the loophole where CE
# routes all sample-specific content through the unconstrained mean channel). Single GPU: schedule draws, z queue, mu EMA are not
# DDP-synchronized (see vae_decomposed_attention.py).

category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=data/${category}/train/${category}_5_2015-10-2018-11.csv
eval_file=data/${category}/valid/${category}_5_2015-10-2018-11.csv
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir=./output_dir/CDs_and_Vinyl_R3_vae7b_a09
out=${task_dir}/latent
mkdir -p ${out}

BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

python src/latent/latent_attention_train_vae.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --ramp_epochs 3 --alpha_max 0.9 \
    --rho_mu 1.0 --ema_momentum 0.99 \
    --kl_weight 1.0 --mu_consistency_weight 0.5 --z_queue_size 256

echo "[$(date --iso-8601=seconds)] Training finished, starting standard ranking evaluation"

# --- Standard HR/NDCG ranking evaluation (same as launch_run.sh/latent_train.sh) ---
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

echo "7b-alpha0.9 (Run B) EVALUATION"
python src/utils/merge.py --input_path ${task_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}
