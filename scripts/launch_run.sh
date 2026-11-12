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
#SBATCH --job-name=R3     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information




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
out=./output_dir/${category}_v7/latent
mkdir -p ${out}

# Replace with your actual base model path
# for example: Qwen/Qwen2.5-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --micro_batch_size 2 \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# available gpu
array=(0 1)

# total number of gpus
nn=2
mkdir -p ./output_dir/${category}_v7
python src/utils/split.py --input_path $test_file \
        --output_path ./output_dir/${category}_v7 --nn ${nn}

model_ckpt=./output_dir/${category}_v7/latent
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${category}/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ./output_dir/${category}_v7/${i}.csv \
        --result_json_data ./output_dir/${category}_v7/${i}.json &
done
wait

echo "SFT EVALUATION"
python src/utils/merge.py --input_path ./output_dir/${category}_v7/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}

#==========================================
# TRAINING WITH ATTENTION GRPO
#==========================================

for category in "CDs_and_Vinyl"; do

    path_to_data="data/CDs_and_Vinyl"
    train_file=$(ls -f ${path_to_data}/train/${category}*.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    for lr in 5e-5; do #1e-4 1e-5 5e-4 5e-5;
        export CUDA_VISIBLE_DEVICES=0,1
        task_name=nomul_attn_lr_${lr}
        base_model=output_dir/${category}_v7/latent/checkpoint-1152
        out=./output_dir/${category}_v7/${task_name}
        mkdir -p $out
        torchrun --nnodes=1 --nproc_per_node=2 --master_port=$((RANDOM % 9999 + 10000)) \
            src/grpo_attention_tuning/train_noise_grpo.py --category $category \
            --num_generations 8 --sample -1 --num_epochs 1 --info_file $info_file \
            --base_model $base_model --output_dir $out --beta 0.01 \
            --batch_size 256 --micro_batch_size 8 --lr $lr --seed 42 \
            --train_file $train_file --eval_file $eval_file \
            --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
            > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

        # available gpu
        array=(0 1)
        nn=2
        model_ckpt=${out}
        mkdir -p ./output_dir/${category}_v7
        python src/utils/split.py --input_path $test_file \
                --output_path ./output_dir/${category}_v7 --nn ${nn}

        # 遍历数组并输出内容
        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo output_dir/${category}_v7/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 12 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ./output_dir/${category}_v7/${i}.csv \
                    --result_json_data ./output_dir/${category}_v7/${i}.json &
            done
        wait
        echo "RL EVALUATION"
        python src/utils/merge.py --input_path ./output_dir/${category}_v7/ \
                --output_path ${model_ckpt}/final_result.json --nn ${nn}

        python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} > >(tee -a ${out}/metric.log)
    done
done

echo "EVALUATION COMPLETE"
# #==========================================
# # EVALUATION WITH ATTENTION GRPO
# #==========================================

# array=(0 1)
# nn=2

# for category in "CDs_and_Vinyl"; do
#     path_to_data=data
#     train_file=$(ls -f ${path_to_data}/${category}/train/${category}*11.csv)
#     eval_file=$(ls -f ${path_to_data}/${category}/valid/${category}*11.csv)
#     test_file=$(ls -f ${path_to_data}/${category}/test/${category}*11.csv)
#     info_file=$(ls -f ${path_to_data}/${category}/info/${category}*.txt)
#     base_model=output_dir/${category}_v7/latent
#     for lr in 5e-5; do
#         model_ckpt=output_dir/CDs_and_Vinyl/nomul_attn_lr_5e-5

#         mkdir -p ./output_dir/${category}_v7
#         python src/utils/split.py --input_path $test_file \
#                 --output_path ./output_dir/${category}_v7 --nn ${nn}

#         for (( i=0; i<$nn; i++ ))
#         do
#             export CUDA_VISIBLE_DEVICES=${array[$i]}
#             echo output_dir/${category}_v7/${i}.csv
#             python src/grpo_attention_tuning/noise_eval.py \
#                     --batch_size 8 --base_model $model_ckpt --sample -1 \
#                     --info_file ${info_file} --category ${category} --end_k -1 \
#                     --test_data_path ./output_dir/${category}_v7/${i}.csv \
#                     --result_json_data ./output_dir/${category}_v7/${i}.json &
#         done
#         wait
#         python src/utils/merge.py --input_path ./output_dir/${category}_v7/ \
#                 --output_path ${model_ckpt}/final_result.json --nn ${nn}


#         python src/utils/calc.py --path ${model_ckpt}/final_result.json \
#         --item_path ${info_file}
#     done

# done