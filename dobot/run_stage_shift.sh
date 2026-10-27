#!/bin/bash
# shift-change scene (GL + RTX) and both pitch cuts; waits for the running RTX batch
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
while pgrep -f run_stage_rtx.sh >/dev/null; do sleep 10; done
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
for kind in gl rtx; do
  if [ $kind = rtx ]; then OUT=$HOME/dobot_tools/resort/results_rtx; FL="--rtx 1 --viz newton_rtx"; X="uv run --extra ovrtx --extra video python"; NAME=DOBOT_resort_pitch_RTX.mp4
  else OUT=$HOME/dobot_tools/resort/results; FL="--viz newton"; X="uv run --extra video python"; NAME=DOBOT_resort_pitch.mp4; fi
  rm -rf $OUT/s5; mkdir -p $OUT/s5
  $X ~/dobot_tools/resort/patrol.py --scenario 5 $FL --out $OUT/s5 --walk_policy $WALK --max_minutes 3 --checkpoint $R/model_2999.pt --num_envs 2 > $S/s5_$kind.log 2>&1
  echo "== s5 $kind"; grep -E "EVENT|Traceback" $S/s5_$kind.log | cut -c1-120
  $X ~/dobot_tools/resort/promo.py $OUT $HOME/dobot_tools/resort/clips/$NAME $HOME/dobot_tools/resort/clips/facts.json 2>&1 | grep -E "pitch|Trace|Error"
done
