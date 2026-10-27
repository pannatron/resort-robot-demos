#!/bin/bash
# final takes for the pitch video (fixed scene: no lintel collision, merged floor), then 50-lap re-test
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
for job in breakfast guide camera; do
  rm -rf $OUT/$job; mkdir -p $OUT/$job
  uv run --extra video python ~/dobot_tools/resort/services.py --job $job --out $OUT/$job --walk_policy $WALK --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/$job.log 2>&1
  echo "== $job"; grep -E "RESULT\]" $S/$job.log | cut -c1-160
done
for sc in 2; do
  rm -rf $OUT/s$sc; mkdir -p $OUT/s$sc
  uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario $sc --out $OUT/s$sc --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/s${sc}f.log 2>&1
  echo "== s$sc"; grep -E "RESULT" $S/s${sc}f.log | head -3
done
rm -rf $OUT/s4; mkdir -p $OUT/s4
uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 4 --out $OUT/s4 --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 50 --viz newton > $S/s4f.log 2>&1
echo "== s4"; grep -E "RESULT" $S/s4f.log | head -12
