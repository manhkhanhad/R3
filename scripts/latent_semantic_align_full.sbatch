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
#SBATCH --job-name=R3_semantic_align

set -Eeuo pipefail

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# Attempt 8: memory-bank semantic alignment for z, on top of the Attempt 7
# mu/z encoder. Positive = same user_id or same target_item (found via a
# 20000-entry memory bank of recent (z, user_hash, target_hash, own_nll)
# tuples), spliced onto the anchor's own mu and decoder-scored against the
# anchor's own target, weighted by stopgrad(exp(-candidate's own NLL)) so a
# low-quality candidate is down-weighted as a positive. Negative = different
# user_id AND different target_item, hinge-pushed up to a margin (reusing
# the Attempt 2 margin mechanism). Routed through the decoder throughout, to
# avoid the marginal-constraint gap that broke Attempt 7a. mu_consistency
# (7b's identifiability anchor) kept on; KL prior off by default to isolate
# this mechanism's own effect. Single GPU: memory bank and mu EMA are not
# DDP-synchronized (see semantic_alignment_attention.py).

category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=data/${category}/train/${category}_5_2015-10-2018-11.csv
eval_file=data/${category}/valid/${category}_5_2015-10-2018-11.csv
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir=./output_dir/CDs_and_Vinyl_R3_semantic_align
out=${task_dir}/latent
mkdir -p ${out}

BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

python src/latent/latent_attention_train_semantic.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --ramp_epochs 3 --alpha_max 0.75 \
    --rho_mu 1.0 --ema_momentum 0.99 \
    --mu_consistency_weight 0.5 --kl_weight 0.0 \
    --pos_weight 1.0 --neg_weight 1.0 --margin 2.0 --bank_size 20000

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

echo "SEMANTIC ALIGNMENT EVALUATION"
python src/utils/merge.py --input_path ${task_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}
