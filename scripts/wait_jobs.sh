#!/usr/bin/env bash
# Block until all given SLURM jobs finish; exit early (printing states) if any ends badly.
# Usage: scripts/wait_jobs.sh <jobid>[,<jobid>...] [poll_seconds]
jobs="$1"; poll="${2:-300}"
n=$(echo "$jobs" | tr ',' '\n' | wc -l)
while true; do
  st=$(sacct -j "$jobs" -X -n -o JobID,JobName%24,State,Elapsed)
  if echo "$st" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF"; then echo "A JOB ENDED BADLY:"; echo "$st"; exit 1; fi
  if [ "$(echo "$st" | grep -c COMPLETED)" -ge "$n" ]; then echo "ALL COMPLETED:"; echo "$st"; exit 0; fi
  sleep "$poll"
done
