#!/bin/bash
# party skills: wait for the wheel payload fine-tune, then dance -> rear-up -> walk payload
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/tasks/b0e0a09vd.output
until grep -q "wheel exit" $OUT; do sleep 20; done
pkill -f "run_payload_train_resume" ; sleep 2; pkill -f "DobotQuadWheelWalk-Payload"; sleep 10
cd ~/IsaacLab
for T in Dance RearUp; do
  N=1500; [ $T = RearUp ] && N=2000
  uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Party-Flat-DobotQuadWheel-$T --num_envs 4096 --max_iterations $N > $S/train_$T.log 2>&1
  echo "$T exit $?"; grep -E "Mean reward|error_height|error_tilt|up_pitch" $S/train_$T.log | tail -4
done
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Stairs-DobotQuadWheelWalk-Payload --num_envs 4096 --max_iterations 800 \
  --checkpoint $PWD/logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/model_4498.pt > $S/train_walk_payload.log 2>&1
echo "walk exit $?"
