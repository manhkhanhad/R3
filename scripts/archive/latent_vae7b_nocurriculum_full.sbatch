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
#SBATCH --job-name=R3_vae7b_nocurriculum

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

# 7b with the attention-curriculum disabled: ramp_epochs~0 + alpha_max=1.0 means
# alpha reaches 1.0 at (essentially) step 1 and the schedule callback then always
# draws "both" (unrestricted history + latent) for the rest of training -- no
# reasoning_only/history_only mixing at all. Isolates the mu/z mechanism's own
# contribution to causal necessity with zero help from the schedule's x-shortcut
# guard. KL (1.0) and mu-consistency anchor (0.5) unchanged from 7b. Single GPU:
# z queue and mu EMA are not DDP-synchronized (see vae_decomposed_attention.py).

category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=data/${category}/train/${category}_5_2015-10-2018-11.csv
eval_file=data/${category}/valid/${category}_5_2015-10-2018-11.csv
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir=./output_dir/CDs_and_Vinyl_R3_vae7b_nocurriculum
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
    --ramp_epochs 0 --alpha_max 1.0 \
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

echo "VAE-DECOMPOSED 7b, CURRICULUM DISABLED (always both) EVALUATION"
python src/utils/merge.py --input_path ${task_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}
