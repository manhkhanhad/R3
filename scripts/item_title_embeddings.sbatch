#!/usr/bin/env bash
#SBATCH --job-name=R3_item_emb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=4:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Item-title embedding table <category>_qwen3emb_1536.pt (Qwen3-Embedding-8B, last token,
# 1536 dims), same recipe as the CDs/Musical tables; first re-embeds 256 CDs titles and
# refuses to write unless they match the stored CDs table (cosine >= 0.99).
# Env: CATEGORY, CODE_ROOT.  Uses the racer env (Qwen3 needs transformers >= 4.51).

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR
module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate racer

category="${CATEGORY:?set CATEGORY}"
code="${CODE_ROOT:?set CODE_ROOT}"
info=$(ls data/${category}/info/${category}_5_*.txt)
out="data/${category}/info/${category}_qwen3emb_1536.pt"
[[ -e "$out" ]] && { echo "$out already exists; not overwriting" >&2; exit 1; }
cd "$code/src/latent"
python item_title_embeddings.py --info_file "$OLDPWD/$info" --output "$OLDPWD/$out" \
  --check_table "$OLDPWD/data/CDs_and_Vinyl/info/CDs_and_Vinyl_qwen3emb_1536.pt" \
  --check_info "$OLDPWD/data/CDs_and_Vinyl/info/CDs_and_Vinyl_5_2015-10-2018-11.txt"
echo "[$(date --iso-8601=seconds)] done"
