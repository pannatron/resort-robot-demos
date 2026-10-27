#!/bin/bash
# final RTX takes for the safety-first pitch
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results_rtx
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
X="uv run --extra ovrtx --extra video python"
for sc in 6 5; do
  NE=1; [ $sc = 5 ] && NE=2
  rm -rf $OUT/s$sc; mkdir -p $OUT/s$sc
  $X ~/dobot_tools/resort/patrol.py --rtx 1 --scenario $sc --out $OUT/s$sc --walk_policy $WALK --max_minutes 6 --checkpoint $R/model_2999.pt --num_envs $NE --viz newton_rtx > $S/final_s$sc.log 2>&1
  echo "== s$sc"; grep -E "EVENT|Traceback" $S/final_s$sc.log | cut -c1-110
done
for job in breakfast guide camera; do
  rm -rf $OUT/$job; mkdir -p $OUT/$job
  $X ~/dobot_tools/resort/services.py --rtx 1 --job $job --out $OUT/$job --walk_policy $WALK --checkpoint $R/model_2999.pt --num_envs 1 --viz newton_rtx > $S/final_$job.log 2>&1
  echo "== $job"; grep -E "RESULT\]|Traceback" $S/final_$job.log | cut -c1-110
done
