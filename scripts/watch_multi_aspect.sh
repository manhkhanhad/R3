#!/usr/bin/env bash
# Stream milestone events for the multi-aspect runs: epoch eval losses, training done,
# standard ranking result, evaluation done, and failures. Exits when all jobs end.
# Usage: scripts/watch_multi_aspect.sh <jobid> [<jobid> ...]
declare -A seen
while true; do
  running=0
  for j in "$@"; do
    f=$(ls run_logs/*."$j".out 2>/dev/null)
    st=$(sacct -j "$j" -X -n -o State | tr -d ' ')
    if [[ -n "$f" ]]; then
      name=$(basename "$f" ."$j".out)
      while IFS= read -r line; do
        key="$j:$line"
        [[ -n "${seen[$key]}" ]] && continue
        seen[$key]=1
        echo "[$name $j] $line"
      done < <(tr '\r' '\n' < "$f" | grep -oE "'eval_loss': [0-9.]+|'epoch': [0-9.]+}|training finished.*|\"hr@10\": [0-9.]+|chosen_lambda.*|done: .*|Traceback.*|[A-Za-z]*Error: .*|FAILED at line.*" \
                 | grep -vE "^'epoch'" | cut -c1-200)
    fi
    case "$st" in
      RUNNING|PENDING|"") running=1 ;;
      COMPLETED) [[ -z "${seen[$j:END]}" ]] && { seen[$j:END]=1; echo "[$j] job COMPLETED"; } ;;
      *) [[ -z "${seen[$j:END]}" ]] && { seen[$j:END]=1; echo "[$j] job ended: $st"; } ;;
    esac
  done
  [[ $running == 0 ]] && exit 0
  sleep 120
done
