#!/bin/bash
#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=20           # Number of CPU to request for the job
#SBATCH --mem=64GB                 # How much memory does your job require?
#SBATCH --gres=gpu:2                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00         # How long to run the job for? Jobs exceed this time will be terminated
                                # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=run_logs/%x.%j.out          # Where should the log files go?
                                # You must provide an absolute path eg /common/home/module/username/
                                # If no paths are provided, the output file will be placed in your current working directory
# Remove if you do not want the workload scheduler to requeue your job after preemption --requeue
#SBATCH --constraint="3090"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=dev     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information


# Parameterized stage-1 SFT train+eval harness for fast hyperparameter iteration.
# All knobs are env vars with defaults matching the paper-exact config
# (scripts/diffureason_train.sh). Defaults use a subsampled train/eval set so a
# full experiment (train + eval + metrics) finishes in minutes instead of hours
# -- intended for relative comparisons across configs, not final reported
# numbers; pass TRAIN_SAMPLE=-1 (full train set) and EVAL_SAMPLE=10000 (larger
# than any test split, so it's clamped to the full split) for a final,
# full-scale run.
#
# Interactive usage:
#   RUN_NAME=exp1_baseline LATENT_R=1 DIFF_T=0 ALPHA=0.0 bash scripts/diffureason_experiment.sh
#
# Slurm usage (this script has an #SBATCH header above -- edit --gres/--time/
# --constraint there for your job, or override on the command line):
#   mkdir -p run_logs
#   sbatch --job-name=exp1_baseline \
#       --export=ALL,RUN_NAME=exp1_baseline,LATENT_R=1,DIFF_T=0,ALPHA=0.0 \
#       scripts/diffureason_experiment.sh
# NPROC/GPUS below should match #SBATCH --gres=gpu:N above (both default to 2).
set -e

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec

category="CDs_and_Vinyl"
base_data_dir=data/${category}
train_file=$(ls -f ${base_data_dir}/train/${category}*.csv)
eval_file=$(ls -f ${base_data_dir}/valid/${category}*.csv)
test_file=$(ls -f ${base_data_dir}/test/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
item_emb_file=${base_data_dir}/info/${category}_qwen3emb_1536.pt

RUN_NAME=${RUN_NAME:-exp_default}
LATENT_R=${LATENT_R:-3}
DIFF_T=${DIFF_T:-16}
DIFF_S=${DIFF_S:-0.008}
USE_SCHED_SCALE=${USE_SCHED_SCALE:-False}
EPS_INIT_SCALE=${EPS_INIT_SCALE:-1.0}
ALPHA=${ALPHA:-1.0}
DIFF_LOSS=${DIFF_LOSS:-mse}
LR=${LR:-3e-4}
EPOCHS=${EPOCHS:-1}
TRAIN_SAMPLE=${TRAIN_SAMPLE:-3000}
EVAL_SAMPLE=${EVAL_SAMPLE:-300}
MICRO_BS=${MICRO_BS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
SEED=${SEED:-42}
NPROC=${NPROC:-2}
GPUS=${GPUS:-0,1}

out=./output_dir/${category}/${RUN_NAME}
mkdir -p ${out}
echo "=== ${RUN_NAME} ===" | tee ${out}/config.txt
echo "LATENT_R=${LATENT_R} DIFF_T=${DIFF_T} DIFF_S=${DIFF_S} USE_SCHED_SCALE=${USE_SCHED_SCALE} EPS_INIT_SCALE=${EPS_INIT_SCALE} ALPHA=${ALPHA} DIFF_LOSS=${DIFF_LOSS} LR=${LR} EPOCHS=${EPOCHS} TRAIN_SAMPLE=${TRAIN_SAMPLE} EVAL_SAMPLE=${EVAL_SAMPLE} MICRO_BS=${MICRO_BS} BATCH_SIZE=${BATCH_SIZE} SEED=${SEED}" | tee -a ${out}/config.txt

port=$((RANDOM % 5000 + 10000))
# Under Slurm, CUDA_VISIBLE_DEVICES is already set correctly by --gres=gpu:N
# (remapped to 0..N-1 for the job's allocation) -- don't override it there.
# Outside Slurm (interactive/nohup runs), set it explicitly from GPUS.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPUS}
fi
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

ALPHA_ARGS=""
if [ "$(python3 -c "print(1 if float(${ALPHA})>0 else 0)")" = "1" ]; then
    ALPHA_ARGS="--item_emb_file ${item_emb_file}"
fi

torchrun --nproc_per_node=${NPROC} --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --micro_batch_size ${MICRO_BS} --batch_size ${BATCH_SIZE} \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample ${TRAIN_SAMPLE} --seed ${SEED} \
    --latent_R ${LATENT_R} --diff_T ${DIFF_T} --diff_s ${DIFF_S} \
    --use_schedule_scaling ${USE_SCHED_SCALE} --diff_eps_init_scale ${EPS_INIT_SCALE} \
    --alpha ${ALPHA} --diff_loss ${DIFF_LOSS} ${ALPHA_ARGS} \
    --num_epochs ${EPOCHS} --learning_rate ${LR} \
    > >(tee -a ${out}/train.log) 2> >(tee -a ${out}/train.err >&2)

array=(${GPUS//,/ })
nn=${NPROC}
split_dir=./output_dir/${category}_${RUN_NAME}
mkdir -p ${split_dir}
python src/utils/split.py --input_path $test_file \
        --output_path ${split_dir} --nn ${nn}

model_ckpt=${out}
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample ${EVAL_SAMPLE} \
        --info_file ${info_file} --category ${category} \
        --test_data_path ${split_dir}/${i}.csv \
        --result_json_data ${split_dir}/${i}.json &
done
wait
python src/utils/merge.py --input_path ${split_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} 2>&1 | tee ${out}/metric.log

echo "DONE ${RUN_NAME}" | tee -a ${out}/metric.log

# Disk hygiene: eval is done, the trained weights (HF Trainer checkpoint-*
# subdirs, ~9G each, plus the final ~3G root model.safetensors) are no longer
# needed -- keep only logs/config/metrics/predictions. This is a small-scale
# throwaway tuning run, not a final artifact; a config worth keeping should be
# re-run at full scale separately, not preserved from these sweeps.
rm -rf ${out}/checkpoint-*
rm -f ${out}/model.safetensors
echo "cleaned up checkpoint/model weight files for ${RUN_NAME}" | tee -a ${out}/metric.log
