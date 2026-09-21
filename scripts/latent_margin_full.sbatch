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
#SBATCH --constraint="l40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=R3_margin     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec

# CDs_and_Vinyl
category="CDs_and_Vinyl"
base_data_dir=data/${category}
# base_data_dir="data/CDs_and_Vinyl_3_2022-10-2023-10_randNeg"
tdir=test
train_file=$(ls -f ${base_data_dir}/train/${category}*.csv)
eval_file=$(ls -f ${base_data_dir}/valid/${category}*.csv)
test_file=$(ls -f ${base_data_dir}/${tdir}/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
echo ${train_file} ${test_file} ${info_file} ${eval_file}
port=$((RANDOM % 5000 + 10000))
export CUDA_VISIBLE_DEVICES=0,1
task_name="latent_${category}"
out=./output_dir/${category}_R3_margin_m0.5/latent
mkdir -p ${out}

# Replace with your actual base model path
# for example: Qwen/Qwen2.5-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train_margin.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} \
    --sample -1 \
    --micro_batch_size 4 --batch_size 128 \
    --num_epochs 3 --cutoff_len 512 \
    --margin 0.5 --margin_weight 1.0

echo "[$(date --iso-8601=seconds)] Training finished, starting standard ranking evaluation"

# --- Standard HR/NDCG ranking evaluation (same as launch_run.sh/latent_train.sh) ---
task_dir=./output_dir/${category}_R3_margin_m0.5
array=(0 1)
nn=2
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

echo "MARGIN EVALUATION"
python src/utils/merge.py --input_path ${task_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}
