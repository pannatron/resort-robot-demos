# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the point-foot Dobot quadruped velocity-tracking environment on rough terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.dobot import DOBOT_QUAD_CFG  # isort: skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BASE_BODY = "link_trunk"


def base_height_above_terrain_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes the squared error of the base height above the scanned terrain.

    Unlike :func:`isaaclab.envs.mdp.base_height_l2`, rays that miss the terrain (``inf`` hits at mesh seams)
    are ignored, so one missed ray cannot turn the reward non-finite and poison the training batch.

    Args:
        env: The environment.
        target_height: Desired base height above the mean terrain height under the scanner [m].
        sensor_cfg: The height-scanner ray caster.
        asset_cfg: The robot.

    Returns:
        Squared height error [m^2], shape ``(num_envs,)``. Zero for environments where every ray missed.
    """
    asset = env.scene[asset_cfg.name]
    hits_z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    valid = torch.isfinite(hits_z)
    num_valid = valid.sum(dim=1)
    ground = torch.where(valid, hits_z, 0.0).sum(dim=1) / num_valid.clamp(min=1)
    error = torch.square(asset.data.root_pos_w.torch[:, 2] - (target_height + ground))
    return torch.where(num_valid > 0, error, 0.0)


def configure_dobot_scene(env_cfg: LocomotionVelocityRoughEnvCfg, robot_cfg) -> None:
    """Applies the robot, sensor paths, and terrain scaling shared by both Dobot variants."""
    env_cfg.sim.use_newton_actuators = True
    env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # the converted USD nests the links under ``Robot/Geometry/link_trunk/...``
    env_cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
    if env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{BASE_BODY}"
    if env_cfg.scene.terrain.terrain_generator is not None:
        terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["random_rough"].noise_step = 0.01
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.12)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.12)
    # events and terminations reference the trunk instead of ``base``
    env_cfg.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
    env_cfg.events.base_com.params["asset_cfg"].body_names = BASE_BODY
    env_cfg.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
    env_cfg.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY


@configclass
class DobotQuadRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the point-foot Dobot quadruped on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        configure_dobot_scene(self, DOBOT_QUAD_CFG)
        # actions
        self.actions.joint_pos.scale = 0.25
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        # shins count too: without this the policy kneels on its rear calves
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(thigh|calf)"
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # gait shaping: keep the hips under the body and the feet planted while in contact
        self.rewards.abad_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_abad")},
        )
        # height above the scanned terrain; without a target the policy crouches and kneels
        self.rewards.base_height = RewTerm(
            func=base_height_above_terrain_l2,
            weight=-10.0,
            params={"target_height": 0.33, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
