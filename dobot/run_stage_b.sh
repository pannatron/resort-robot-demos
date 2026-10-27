#!/bin/bash
# scenarios 2, 3 (video) and 4 (50 randomized laps), then the summary
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results

cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
for sc in 1 2 3; do
  mkdir -p $OUT/s$sc
  uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario $sc --out $OUT/s$sc --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/s$sc.log 2>&1
  echo "== s$sc"; grep -E "RESULT|fell" $S/s$sc.log
done
mkdir -p $OUT/s4
uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 4 --out $OUT/s4 --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 50 --viz newton > $S/s4.log 2>&1
echo "== s4"; grep -E "RESULT|fell" $S/s4.log
uv run python ~/dobot_tools/resort/summarize.py $OUT > $S/summary.txt 2>&1; tail -60 $S/summary.txt
