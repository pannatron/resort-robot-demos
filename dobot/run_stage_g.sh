#!/bin/bash
# final pitch takes with the new look (textures, sky, people, clean HUD)
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
for job in guide breakfast camera; do
  rm -rf $OUT/$job; mkdir -p $OUT/$job
  uv run --extra video python ~/dobot_tools/resort/services.py --job $job --out $OUT/$job --walk_policy $WALK --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/$job.log 2>&1
  echo "== $job"; grep -E "RESULT\]|Traceback" $S/$job.log | cut -c1-140
done
rm -rf $OUT/s2; mkdir -p $OUT/s2
uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 2 --out $OUT/s2 --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/s2g.log 2>&1
echo "== s2"; grep -E "RESULT|Traceback" $S/s2g.log | head -3
