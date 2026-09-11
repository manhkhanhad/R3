#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
mkdir -p analysis/logs

python src/prototype_grounding/experiment.py manifest
for split in train valid test; do
  python src/prototype_grounding/experiment.py extract --split "$split" --batch-size 8
done
python src/prototype_grounding/experiment.py intervention --split valid --batch-size 8

for bank in all correct positive top harmful; do
  for k in 32 64 128 256; do
    for seed in 13 42 73; do
      python src/prototype_grounding/experiment.py mine --bank "$bank" --k "$k" --seed "$seed"
    done
  done
done
