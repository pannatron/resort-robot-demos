#!/bin/bash
# tablet-control scene + Camera Man re-take (RTX camera-switch fix)
OUT=$HOME/dobot_tools/resort/results_rtx
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06/model_2999.pt
WALK=logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/exported/policy.pt
X="uv run --extra ovrtx --extra video python"
$X ~/dobot_tools/resort/tablet.py --out $OUT/tablet --rtx 1 --checkpoint $R --num_envs 1 --viz newton_rtx > $OUT/tablet/run.log 2>&1
echo "== tablet $?"; grep -aE "RESULT|Traceback" $OUT/tablet/run.log | cut -c1-160
mv $OUT/camera $OUT/camera_swapbug_bak; mkdir -p $OUT/camera
$X ~/dobot_tools/resort/services.py --rtx 1 --job camera --out $OUT/camera --walk_policy $WALK --checkpoint $R --num_envs 1 --viz newton_rtx > $S/fix_camera.log 2>&1
echo "== camera $?"; grep -aE "RESULT\]|Traceback" $S/fix_camera.log | cut -c1-160
echo ALL DONE
