#!/bin/bash
# re-record scenarios 1-2 with the fixed stair camera, then cut chat-size highlights
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results
while pgrep -f run_stage_b.sh >/dev/null; do sleep 10; done
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
for sc in 1 2; do
  rm -rf $OUT/s$sc; mkdir -p $OUT/s$sc
  uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario $sc --out $OUT/s$sc --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/s${sc}b.log 2>&1
  echo "== s$sc"; grep -E "RESULT" $S/s${sc}b.log
done
for sc in 1 2 3; do
  uv run --extra video python ~/dobot_tools/resort/highlight.py $OUT/s$sc/scenario${sc}_highlight.mp4 $OUT/s$sc/scenario$sc.mp4 2>&1 | grep -v Warn
done
uv run python ~/dobot_tools/resort/summarize.py $OUT > $S/summary.txt 2>&1; cat $S/summary.txt
