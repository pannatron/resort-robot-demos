#!/bin/bash
# robustness: 50 randomized laps (after stage C), then summary + highlight of env 0
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results

cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
rm -rf $OUT/s4; mkdir -p $OUT/s4
uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 4 --out $OUT/s4 --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 50 --viz newton > $S/s4b.log 2>&1
grep -E "RESULT" $S/s4b.log
uv run --extra video python ~/dobot_tools/resort/highlight.py $OUT/s4/scenario4_highlight.mp4 $OUT/s4/scenario4.mp4 2>&1 | grep -v -i warn
uv run python ~/dobot_tools/resort/summarize.py $OUT > $S/summary.txt 2>&1; cat $S/summary.txt
