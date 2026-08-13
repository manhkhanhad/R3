# DiffuReason (Think-then-Diffuse) stage-1 SFT + eval, matching the paper
# (arXiv 2602.09744) as closely as possible for this LLM/text-title backbone.
# Paper settings (Sec. 4.2/Table 5): R=3 thinking steps (Eq. 1), T=16 denoising
# steps with cosine schedule s=0.008 (Eq. 17, no forward process -- Appendix
# C.1.1), alpha=beta=1.0 loss weights, L_diff = ||mu - v_target||^2 verbatim
# (Eq. 7, --diff_loss mse). use_schedule_scaling defaults to False in the train
# script now (paper-exact: Algorithm 1 samples x_{t-1} ~ N(mu,sigma) directly,
# no extra alpha_bar-based noise annealing), so no flag is needed here. beta
# (the GRPO alignment loss weight) is stage-2-only; stage-2 stays as-is for now
# per current scope (SFT first, GRPO to be revisited separately).
#
# MSE anchor loss (Qwen3-Embedding-8B item title embeddings) is enabled below via
# --alpha 1.0 --item_emb_file. Embedding generation needs transformers>=4.51 (qwen3
# arch support), which the rrec env's 4.48.2 doesn't have -- so that one step runs
# under the khanhnhm env; training/eval stay on rrec as before.

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"

# CDs_and_Vinyl
category="CDs_and_Vinyl"
base_data_dir=data/${category}
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

item_emb_file=${base_data_dir}/info/${category}_qwen3emb_1536.pt
if [ ! -f ${item_emb_file} ]; then
    conda activate khanhnhm
    python src/utils/gen_item_embeddings.py --info_file ${info_file} \
        --output_file ${item_emb_file}
    conda deactivate
fi

conda activate rrec

# Replace with your actual base model path
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --micro_batch_size 4 \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    --latent_R 3 --diff_T 16 --diff_s 0.008 \
    --alpha 1.0 --item_emb_file ${item_emb_file} --diff_loss mse \
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
