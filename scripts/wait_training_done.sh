#!/usr/bin/env bash
# Exit when every given job has printed "training finished" (or any job fails/ends early).
# Usage: scripts/wait_training_done.sh <jobid> [<jobid> ...]
while true; do
  done_n=0
  for j in "$@"; do
    f=$(ls run_logs/*."$j".out 2>/dev/null)
    st=$(sacct -j "$j" -X -n -o State | tr -d ' ')
    if [[ -n "$f" ]] && grep -q "training finished" "$f"; then
      done_n=$((done_n + 1))
    elif [[ "$st" != RUNNING && "$st" != PENDING && -n "$st" ]]; then
      echo "job $j ended before training finished: $st"; exit 1
    fi
  done
  if [[ $done_n == $# ]]; then
    for j in "$@"; do
      f=$(ls run_logs/*."$j".out)
      echo "== $(basename "$f"): eval_loss per epoch:" $(tr '\r' '\n' < "$f" | grep -oE "'eval_loss': [0-9.]+" | grep -oE "[0-9.]+$")
    done
    exit 0
  fi
  sleep 300
done
