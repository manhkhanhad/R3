

## Directory Structure

```plaintext
.
|____scripts
|____src
| |____latent
| | (SFT code)
| |____grpo_attention_tuning (RL code)
| | |____latent_grpo_processor.py
| | |____latent_grpo_dataset.py
| | |____model.py
| | |____grpo_trainer.py
| | |____train_noise_grpo.py
| | |____noise_eval.py
| |____utils (others)
|____data (example data and our processing code)
| |____process.py
|____environment.yaml
|____README.md
```

## Environment

```bash
conda env create -n your_env -f environment.yaml
```

## Data process

You can download Amazon 2018 dataset through official website and process them with the processing code we provide. Then, you can get the dataset we use in our paper.

```bash
# Take the book dataset as an example
# Download the dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz
wget wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz
# Unzip
gunzip Books.json.gz
gunzip meta_Books.json.gz
# Preprocess
python ./data/process.py --category "Books"
```

## Train and evaluation

### Weights & Biases logging

Log in once with `wandb login`, then give every process in one experiment the
same run ID. W&B shared mode allows training and evaluation to write to that
run concurrently:

```bash
export WANDB_PROJECT=R3
export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="my-experiment"

# Start the training and evaluation commands from the same environment.
# Both commands automatically read the variables above.
```

Training loss and validation loss are logged by the Transformers integration.
Inference logs per-shard ranking metrics under `evaluation_shard/*`, while
`src/utils/calc.py` logs the merged metrics under `evaluation/*`. Every run is
tagged with the original dataset category (for example, `CDs_and_Vinyl`). Set
`WANDB_DISABLED=true` to turn logging off.

```bash
conda activate your_env
# SFT
bash scripts/latent_train.sh
# modified GRPO (change the path to your sft model)
bash scripts/attention_grpo.sh
# for evaluation
bash scripts/grpo_eval.sh
```

> **Note:**
> - We strongly recommend that you use two GPUs to run GRPO. We did not test our code on more GPUs due to computational resources. This is related to the calculation of the advantage function.

## Hyperparameter Search

To achieve expected results, tune the learning rate in GRPO:

- `lr`: {1e-5, 1e-4, 5e-4}

You can tune more hyperparameters to get more impressive results.

> **Note:**
> - Results may vary across different devices even with the same hyperparameters, due to differences in computation precision. We conduct our experiments using 2 NVIDIA A100 GPUs.


If you find our paper or code helpful to your work, please cite it.

```
@article{latentr3,
  title={Reinforced latent reasoning for llm-based recommendation},
  author={Zhang, Yang and Xu, Wenxin and Zhao, Xiaoyan and Wang, Wenjie and Feng, Fuli and He, Xiangnan and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2505.19092},
  year={2025}
}
```
