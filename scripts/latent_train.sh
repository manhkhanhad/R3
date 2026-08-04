
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
out=./output_dir/${category}/latent
mkdir -p ${out}

# Replace with your actual base model path
# for example: Qwen/Qwen2.5-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --micro_batch_size 4 \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# available gpu
array=(0 1)

# total number of gpus
nn=2
mkdir -p ./output_dir/${category}_base
python src/utils/split.py --input_path $test_file \
        --output_path ./output_dir/${category}_base --nn ${nn}

model_ckpt=./output_dir/${category}/latent
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${category}_base/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ./output_dir/${category}_base/${i}.csv \
        --result_json_data ./output_dir/${category}_base/${i}.json &
done
wait
python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} > >(tee -a ${out}/metric.log)