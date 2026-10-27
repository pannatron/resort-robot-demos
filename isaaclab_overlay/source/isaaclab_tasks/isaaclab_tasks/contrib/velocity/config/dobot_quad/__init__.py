# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dobot quadruped velocity-tracking environments (point-foot and wheel-legged)."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_ENVS = {
    "IsaacContrib-Velocity-Flat-DobotQuad": ("flat_env_cfg:DobotQuadFlatEnvCfg", "DobotQuadFlatPPORunnerCfg"),
    "IsaacContrib-Velocity-Rough-DobotQuad": ("rough_env_cfg:DobotQuadRoughEnvCfg", "DobotQuadRoughPPORunnerCfg"),
    "IsaacContrib-Velocity-Flat-DobotQuadWheel": (
        "wheel_env_cfg:DobotQuadWheelFlatEnvCfg",
        "DobotQuadWheelFlatPPORunnerCfg",
    ),
    "IsaacContrib-Velocity-Rough-DobotQuadWheel": (
        "wheel_env_cfg:DobotQuadWheelRoughEnvCfg",
        "DobotQuadWheelRoughPPORunnerCfg",
    ),
    # walking mode for stairs (wheels braked), paired with the rolling policy above
    "IsaacContrib-Velocity-Stairs-DobotQuadWheelWalk": (
        "wheel_env_cfg:DobotQuadWheelStairWalkEnvCfg",
        "DobotQuadWheelStairWalkPPORunnerCfg",
    ),
    # payload fine-tunes (0-5 kg on the trunk, smooth and level)
    "IsaacContrib-Velocity-Rough-DobotQuadWheel-Payload": (
        "wheel_env_cfg:DobotQuadWheelRoughPayloadEnvCfg",
        "DobotQuadWheelRoughPayloadPPORunnerCfg",
    ),
    "IsaacContrib-Velocity-Stairs-DobotQuadWheelWalk-Payload": (
        "wheel_env_cfg:DobotQuadWheelStairWalkPayloadEnvCfg",
        "DobotQuadWheelStairWalkPayloadPPORunnerCfg",
    ),
    # party skills: body-pose dance and standing on the rear legs
    "IsaacContrib-Party-Flat-DobotQuadWheel-Dance": (
        "party_env_cfg:DobotQuadWheelDanceEnvCfg",
        "DobotQuadWheelDancePPORunnerCfg",
    ),
    "IsaacContrib-Party-Flat-DobotQuadWheel-RearUp": (
        "party_env_cfg:DobotQuadWheelRearUpEnvCfg",
        "DobotQuadWheelRearUpPPORunnerCfg",
    ),
    # hardware-deployable variants: no base linear velocity in the actor, sim-to-real randomization
    "IsaacContrib-Velocity-Flat-DobotQuad-Deploy": (
        "deploy_env_cfg:DobotQuadDeployFlatEnvCfg",
        "DobotQuadDeployFlatPPORunnerCfg",
    ),
    "IsaacContrib-Velocity-Rough-DobotQuad-Deploy": (
        "deploy_env_cfg:DobotQuadDeployRoughEnvCfg",
        "DobotQuadDeployRoughPPORunnerCfg",
    ),
    "IsaacContrib-Velocity-Flat-DobotQuadWheel-Deploy": (
        "deploy_env_cfg:DobotQuadWheelDeployFlatEnvCfg",
        "DobotQuadWheelDeployFlatPPORunnerCfg",
    ),
    "IsaacContrib-Velocity-Rough-DobotQuadWheel-Deploy": (
        "deploy_env_cfg:DobotQuadWheelDeployRoughEnvCfg",
        "DobotQuadWheelDeployRoughPPORunnerCfg",
    ),
}

for _task_id, (_env_cfg, _agent_cfg) in _ENVS.items():
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_agent_cfg}",
            "default_agent": "rsl_rl",
        },
    )
