#!/bin/bash
# All RPO resort takes rendered with the OVRTX path tracer (one after another: 6 GB GPU).
LOG=${LOG:-/tmp/claude-1000/-home-ubuntu/15e3f6c1-b0e5-44da-9f75-2cd404c42d7b/scratchpad}
OUT=$HOME/rpo_tools/resort/results_rtx
cd ~/IsaacLab
X="uv run --extra ovrtx --extra video --with onnxruntime python"
for job in ${JOBS:-bump welcome delivery party}; do
  rm -rf $OUT/$job; mkdir -p $OUT/$job
  start=$(date +%s)
  if [ $job = party ]; then
    $X ~/rpo_tools/resort/rpo_party.py --rtx 1 --out $OUT/$job --viz newton_rtx > $LOG/rtx_$job.log 2>&1
  else
    $X ~/rpo_tools/resort/rpo_jobs.py --job $job --rtx 1 --out $OUT/$job --viz newton_rtx > $LOG/rtx_$job.log 2>&1
  fi
  echo "== $job ($(( $(date +%s) - start )) s)"; grep -E "RESULT\]|Traceback" $LOG/rtx_$job.log | cut -c1-200
done
