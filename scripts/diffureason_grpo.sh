# DiffuReason stage-2: GRPO over the original LatentR3 loss/advantage/ratio
# scheme (token-level logprob ratio, swap_adjacent_blocks baseline, token KL).
# The only change vs. the original attention_grpo.sh is that the perturbed
# thought-slot candidate z_i = mu + sigma * eps_i is drawn from the learned
# diffusion Gaussian instead of a fixed isotropic noise (--use_diffusion True,
# the default; pass False to fall back to the original fixed-noise scheme).
# Requires a stage-1 checkpoint from scripts/diffureason_train.sh. No longer
# restricted to exactly 2 GPUs.

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec

for category in "CDs_and_Vinyl"; do

    path_to_data="data/CDs_and_Vinyl"
    train_file=$(ls -f ${path_to_data}/train/${category}*.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    for lr in 5e-5; do #1e-4 1e-5 5e-4 5e-5;
        export CUDA_VISIBLE_DEVICES=0,1
        task_name=diffureason_grpo_lr_${lr}
        base_model=output_dir/${category}/diffureason_latent
        out=./output_dir/${category}/${task_name}
        mkdir -p $out
        torchrun --nnodes=1 --nproc_per_node=2 --master_port=$((RANDOM % 9999 + 10000)) \
            src/grpo_attention_tuning/train_noise_grpo.py --category $category \
            --num_generations 8 --sample -1 --num_epochs 1 --info_file $info_file \
            --base_model $base_model --output_dir $out --beta 0.01 \
            --batch_size 256 --micro_batch_size 8 --lr $lr --seed 42 \
            --train_file $train_file --eval_file $eval_file \
            --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 --use_diffusion True \
            > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

        # available gpu
        array=(0 1)
        nn=2
        model_ckpt=${out}
        split_dir=./output_dir/${category}_diffureason
        mkdir -p ${split_dir}
        python src/utils/split.py --input_path $test_file \
                --output_path ${split_dir} --nn ${nn}

        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo ${split_dir}/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 12 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ${split_dir}/${i}.csv \
                    --result_json_data ${split_dir}/${i}.json &
            done
        wait
        python src/utils/merge.py --input_path ${split_dir}/ \
                --output_path ${model_ckpt}/final_result.json --nn ${nn}

        python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} > >(tee -a ${out}/metric.log)
    done

done
