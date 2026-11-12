#!/usr/bin/env bash
#SBATCH --job-name=R3_larec_unit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40|l40s|a100|h100"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --time=48:00:00
#SBATCH --chdir=/common/home/users/m/mk.ngohuu.2024/research/R3/.claude/worktrees/LatentDistillation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/R3/run_logs/%x.%j.out

# LaRec-style latent pre-training on Toys from external rationale units
# (document/latent_distillation.md): the target-free, metadata-grounded units of
# external_rational/results/toys_full_qwen35_9b become K = 3 teacher steps
# (evidence / interpretation / recommendation; unit_cot.py), the frozen backbone's
# mean-pooled state over each step's tokens is the alignment target
# (teacher_states.py), and the student is R3 with 3 latent slots and the v2
# paper-literal losses (larec_latent.py; teacher states standardised at load).
# Then constrained-beam ranking on the Toys test set, and the swap control.
# teacher: llm = frozen Qwen2.5-1.5B states mean-pooled per step (teacher_states.py);
#          gte = gte-large-en-v1.5 embedding of each step text (step_embeddings.py);
#          qwen3emb = Qwen3-Embedding-8B embedding of each step text, last-token pooling,
#          truncated to 1536 (scripts/qwen3emb_steps_toys.sbatch builds it; needs racer).
# Usage: sbatch scripts/larec_unit_toys.sbatch [full|smoke] [llm|gte|qwen3emb] [alpha] [gamma]
source scripts/collab_env.sh
mode="${1:-full}"; teacher="${2:-llm}"; alpha="${3:-1.0}"; gamma="${4:-1.0}"; k=3
case "$teacher" in llm) suffix=mean ;; gte) suffix=gte ;; qwen3emb) suffix=qwen3emb ;;
  *) echo "teacher must be llm, gte or qwen3emb" >&2; exit 1 ;; esac
category=Toys_and_Games
data=data/${category}
info_file=$(ls ${data}/info/${category}*.txt)
generations=/common/home/users/m/mk.ngohuu.2024/research/R3/external_rational/results/toys_full_qwen35_9b/generation/generations
distill_root=${scratch}/latent_distill
unit_root=${distill_root}/larec_unit_toys_k${k}
name=larec_unit_${teacher}_${mode}_k${k}_a${alpha}_g${gamma}

export WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
export WANDB_NAME="R3_toys_${name}"
if [[ "$mode" == smoke ]]; then
  data_root=${distill_root}/toys_smoke_data
  for split in train valid test; do
    mkdir -p "${data_root}/${split}"
    src=$(ls ${data}/${split}/*.csv)
    [[ -f "${data_root}/${split}/$(basename "$src")" ]] || head -n 513 "$src" > "${data_root}/${split}/$(basename "$src")"
  done
  out=${distill_root}/toys_smoke_${name}/latent
  states=${unit_root}/smoke_states_${suffix}.pt
  cot=${unit_root}/smoke_cot.json
  epochs=1; limit=16; ctrl_limit=16
  export WANDB_MODE=disabled
else
  data_root=${data}
  out=${distill_root}/Toys_and_Games_${name}/latent
  states=${unit_root}/train_states_${suffix}.pt
  cot=${unit_root}/train_cot.json
  epochs=3; limit=-1; ctrl_limit=2000
fi
train_csv=$(ls ${data_root}/train/*.csv)
mkdir -p "$out" "$unit_root"
echo "[$(date --iso-8601=seconds)] mode=$mode teacher=$teacher states=$states k=$k alpha=$alpha gamma=$gamma generations=$generations out=$out"

[[ -f "$cot" ]] || python src/latent_distill/unit_cot.py \
  --generations "$generations" --csv "$train_csv" --out "$cot"
if [[ ! -f "$states" ]]; then
  if [[ "$teacher" == llm ]]; then
    python src/latent_distill/teacher_states.py --cot "$cot" --csv "$train_csv" --out "$states" --pool mean
  elif [[ "$teacher" == qwen3emb ]]; then
    echo "missing $states: build it with scripts/qwen3emb_steps_toys.sbatch (racer env)" >&2; exit 1
  else
    python src/latent_distill/step_embeddings.py --cot "$cot" --out "$states"
  fi
fi
echo "[$(date --iso-8601=seconds)] teacher states ready; training"

if [[ ! -f "$out/model.safetensors" ]]; then
  python src/latent/latent_attention_train.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct --larec_states "$states" --larec_k "$k" \
    --larec_alpha "$alpha" --larec_gamma "$gamma" \
    --train_file "$train_csv" --eval_file "$(ls ${data_root}/valid/*.csv)" \
    --output_dir "$out" --category "$category" \
    --sample -1 --micro_batch_size 2 --batch_size 128 --num_epochs "$epochs" --cutoff_len 512 \
    --save_only_model True --save_total_limit 1
fi
echo "[$(date --iso-8601=seconds)] training finished; evaluating"

test_csv=$(ls ${data_root}/test/*.csv)
evaluate() {  # tag, limit, extra args
  local t=$1 n=$2; shift 2
  [[ -f "$out/eval_$t.json" ]] && { echo "exists: eval_$t"; return; }
  python src/latent/rationale_eval.py --base_model "$out" --info_file "$info_file" --mode larec \
    --category "$category" \
    --test_data_path "$test_csv" --result_json_data "$out/eval_$t.json" --limit "$n" --batch_size 8 "$@" \
    | tee "$out/eval_$t.log"
}
evaluate true "$limit"
evaluate swap "$ctrl_limit" --latent_override swap
# scratch is purged periodically: keep the final model and evals under home
keep=/common/home/users/m/mk.ngohuu.2024/research/R3/output_dir/latent_distill/$(basename "$(dirname "$out")")
mkdir -p "$keep"
rsync -a --exclude 'checkpoint-*' "$out" "$keep/"
echo "[$(date --iso-8601=seconds)] copied to $keep"
echo "[$(date --iso-8601=seconds)] done: $name"
