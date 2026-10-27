#!/bin/bash
# fine-tune rolling + walking policies with a 0-5 kg payload
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
cd ~/IsaacLab
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Rough-DobotQuadWheel-Payload --num_envs 4096 --max_iterations 750 \
  --checkpoint $PWD/logs/rsl_rl/dobot_quad_wheel_rough_payload/2026-10-22_22-45-56/model_3050.pt > $S/train_wheel_payload2.log 2>&1
echo "wheel exit $?"; grep -E "terrain_levels|error_vel_xy|base_contact|flat_orientation" $S/train_wheel_payload2.log | tail -4
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Stairs-DobotQuadWheelWalk-Payload --num_envs 4096 --max_iterations 800 \
  --checkpoint $PWD/logs/rsl_rl/dobot_quad_wheel_walk/2026-10-22_16-49-33/model_4498.pt > $S/train_walk_payload.log 2>&1
echo "walk exit $?"; grep -E "terrain_levels|error_vel_xy|base_contact|flat_orientation" $S/train_walk_payload.log | tail -4
