# DiffuReason (Think-then-Diffuse) stage-1 SFT + eval
# Paper: arXiv 2602.09744. R=3 thinking steps, T=16 denoising steps, CE-only loss.
#
# Optional MSE anchor loss (off by default): first generate item embeddings
#   python src/utils/gen_item_embeddings.py --info_file ${info_file}
# then add to the train command:
#   --alpha 1.0 --item_emb_file ${base_data_dir}/info/${category}_qwen3emb_1536.pt

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate rrec

# CDs_and_Vinyl
category="CDs_and_Vinyl"
base_data_dir="data/CDs_and_Vinyl"
tdir=test
train_file=$(ls -f ${base_data_dir}/train/${category}*.csv)
eval_file=$(ls -f ${base_data_dir}/valid/${category}*.csv)
test_file=$(ls -f ${base_data_dir}/${tdir}/${category}*.csv)
info_file=$(ls -f ${base_data_dir}/info/${category}*.txt)
echo ${train_file} ${test_file} ${info_file} ${eval_file}
port=$((RANDOM % 5000 + 10000))
export CUDA_VISIBLE_DEVICES=0,1
task_name="diffureason_latent_${category}"
out=./output_dir/${category}/diffureason_latent
mkdir -p ${out}

# Replace with your actual base model path
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --micro_batch_size 8 \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    --latent_R 3 --diff_T 16 --diff_s 0.008 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# available gpu
array=(0 1)

# total number of gpus
nn=2
split_dir=./output_dir/${category}_diffureason
mkdir -p ${split_dir}
python src/utils/split.py --input_path $test_file \
        --output_path ${split_dir} --nn ${nn}

model_ckpt=${out}
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${split_dir}/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 8 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ${split_dir}/${i}.csv \
        --result_json_data ${split_dir}/${i}.json &
done
wait
python src/utils/merge.py --input_path ${split_dir}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} > >(tee -a ${out}/metric.log)
