#!/usr/bin/env bash
#SBATCH --job-name=R3_expl_emb_CDs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|l40|a40|3090"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=12:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=run_logs/%x.%j.out

# Embed the target-conditioned teacher explanations (CDs train split) with
# Qwen3-Embedding-8B @1536 (same recipe as <category>_qwen3emb_1536.pt) for latent
# distillation.  Views: full / post_evidence / feature.  Uses the racer env because
# Qwen3 needs transformers >= 4.51 (rrec has 4.48).  Refuses to run on a partial
# explanation file unless ALLOW_PARTIAL=1.
# Env: CODE_ROOT, EXPLANATIONS_DIR, OUTPUT, LIMIT, BATCH_SIZE, ALLOW_PARTIAL.

set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FAILED at line ${BASH_LINENO[0]} (exit $status)" >&2' ERR

module purge
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate racer

code=${CODE_ROOT:-/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/R3LatentCollaborativeReasoning}
expl=${EXPLANATIONS_DIR:-/common/home/users/m/mk.ngohuu.2024/research/RRec_coconut/Scoter_COT/results/description_reasoning_full_v3}
category=CDs_and_Vinyl
data=data/${category}
output=${OUTPUT:-${data}/info/${category}_explanation_qwen3emb_1536.pt}

python - "$expl/progress.json" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
done = p["processed"] >= p["total"]
print(f"explanations: processed {p['processed']}/{p['total']}, accepted {p['accepted']}, failed {p['failed']}")
if not done and os.environ.get("ALLOW_PARTIAL") != "1":
    sys.exit("explanation generation is not finished; set ALLOW_PARTIAL=1 to embed anyway")
PY

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
batch=${BATCH_SIZE:-$(( gpu_memory > 30000 ? 64 : 16 ))}
echo "[$(date --iso-8601=seconds)] host $(hostname), gpu ${gpu_memory} MiB, batch ${batch}, code ${code}"

python "${code}/src/latent/explanation_embeddings.py" \
    --explanations "${expl}/candidates.jsonl" \
    --train_csv "${data}/train/${category}_5_2015-10-2018-11.csv" \
    --output "${output}" \
    --model Qwen/Qwen3-Embedding-8B --dim 1536 --window 10 \
    --batch_size "${batch}" --limit "${LIMIT:-0}" \
    --item_table "${data}/info/${category}_qwen3emb_1536.pt" \
    --item_info "${data}/info/${category}_5_2015-10-2018-11.txt"
echo "[$(date --iso-8601=seconds)] done"
