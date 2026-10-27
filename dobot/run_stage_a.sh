#!/bin/bash
# wait for walk training, export its policy, then run scenario 1 (full lap, video)
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results
while kill -0 44842 2>/dev/null; do sleep 20; done
cd ~/IsaacLab
WRUN=$(ls -td logs/rsl_rl/dobot_quad_wheel_walk/2026-* | head -1)
WM=$(ls $WRUN/model_*.pt | sort -V | tail -1)
echo "walk model $WM"
VX=0.4 FOOT=".*_wheel" uv run python ~/dobot_tools/gait_eval.py --task IsaacContrib-Velocity-Stairs-DobotQuadWheelWalk --num_envs 64 --checkpoint $WM > $S/walk_gait.log 2>&1
grep GAIT $S/walk_gait.log
R=logs/rsl_rl/dobot_quad_wheel_rough/2026-10-22_13-37-06
mkdir -p $OUT/s1
uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 1 --out $OUT/s1 --walk_policy $WRUN/exported/policy.pt --max_minutes 10 --checkpoint $R/model_2999.pt --num_envs 1 --viz newton > $S/s1.log 2>&1
grep -E "RESULT|fell|EVENT" $S/s1.log
echo "WALK=$WRUN/exported/policy.pt"
