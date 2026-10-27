#!/bin/bash
# all four pitch takes rendered with the OVRTX path tracer, then the pitch cut
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results_rtx
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
X="uv run --extra ovrtx --extra video python"
for job in guide breakfast camera; do
  rm -rf $OUT/$job; mkdir -p $OUT/$job
  $X ~/dobot_tools/resort/services.py --rtx 1 --job $job --out $OUT/$job --walk_policy $WALK --checkpoint $R/model_2999.pt --num_envs 1 --viz newton_rtx > $S/rtx_$job.log 2>&1
  echo "== $job"; grep -E "RESULT\]|Traceback" $S/rtx_$job.log | cut -c1-140
done
rm -rf $OUT/s2; mkdir -p $OUT/s2
$X ~/dobot_tools/resort/patrol.py --rtx 1 --scenario 2 --out $OUT/s2 --walk_policy $WALK --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton_rtx > $S/rtx_s2.log 2>&1
echo "== s2"; grep -E "RESULT|Traceback" $S/rtx_s2.log | head -3
$X ~/dobot_tools/resort/promo.py $OUT $HOME/dobot_tools/resort/clips/DOBOT_resort_pitch_RTX.mp4 $HOME/dobot_tools/resort/clips/facts.json 2>&1 | grep -E "pitch|Trace|Error"
