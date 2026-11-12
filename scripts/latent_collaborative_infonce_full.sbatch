#!/usr/bin/env bash
#SBATCH --job-name=R3_collab_CDs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=24:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-R3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
export WANDB_NAME="${WANDB_NAME:-R3_collab_CDs}"
export WANDB_TAGS="${WANDB_TAGS:-collaborative-infonce,cross-dataset}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
micro_batch_size=4
if [[ "$gpu_name" == *"3090"* ]]; then
    micro_batch_size=2
fi
echo "[$(date --iso-8601=seconds)] GPU: $gpu_name; micro batch size: $micro_batch_size"

# Plain R3 only. InfoNCE is active for all three epochs. The non-persistent
# FIFO holds at most 10k recent latents meeting the target-token accuracy gate;
# same-sample stale entries are never eligible as positives.
category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=data/${category}/train/${category}_5_2015-10-2018-11.csv
eval_file=data/${category}/valid/${category}_5_2015-10-2018-11.csv
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
task_dir="/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/${category}_R3_collaborative_infonce${OUTPUT_SUFFIX:-}"
out=${task_dir}/latent
mkdir -p ${out}

BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

python src/latent/latent_attention_train_collaborative.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size "$micro_batch_size" --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --projection_dim 256 --temperature 0.1 \
    --user_contrastive_weight 0.1 --target_contrastive_weight 0.1 \
    --bank_size 10000 --max_user_positives 16 \
    --max_target_positives 16 --num_negatives 256 \
    --positive_min_token_accuracy "${POSITIVE_MIN_TOKEN_ACCURACY:-1.0}" \
    "${resume_args[@]}"

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
