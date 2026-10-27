# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the point-foot Dobot quadruped velocity-tracking environment on flat terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .rough_env_cfg import DobotQuadRoughEnvCfg


def make_flat(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Switches a rough-terrain velocity environment to a flat plane without height scanning."""
    newton_mjwarp = env_cfg.sim.physics.newton_mjwarp
    newton_mjwarp.solver_cfg.njmax = 80
    newton_mjwarp.solver_cfg.nconmax = 40
    env_cfg.sim.physics.default = newton_mjwarp
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.height_scanner = None
    env_cfg.observations.policy.height_scan = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.rewards.flat_orientation_l2.weight = -2.5


@configclass
class DobotQuadFlatEnvCfg(DobotQuadRoughEnvCfg):
    """Configuration for the point-foot Dobot quadruped on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        make_flat(self)
        self.rewards.feet_air_time.weight = 0.25
        # without a target the policy crouches to ~0.23 m; nominal stance is ~0.37 m
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2, weight=-30.0, params={"target_height": 0.33})
