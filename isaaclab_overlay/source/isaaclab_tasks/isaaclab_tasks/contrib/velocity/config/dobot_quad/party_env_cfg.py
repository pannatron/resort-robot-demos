# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Party skills for the wheel-legged Dobot quadruped: a body-pose "dance" policy and a rear-up (stand on the
hind legs) policy.

* Dance: on top of the velocity command, the policy tracks a body-pose command (trunk height, roll, pitch)
  that is resampled every ~1-2 s. A show controller then choreographs a dance by streaming poses to the beat
  (bounce, sway, nod, spin on the wheels, side slides).
* Rear-up: a binary command asks the robot to lift its front wheels and balance on the rear legs with the trunk
  pitched nose-up, then come back down to the normal stance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .wheel_env_cfg import LEG_JOINTS, DobotQuadWheelFlatEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

NOMINAL_HEIGHT = 0.40
FRONT_WHEELS = "link_front_.*_wheel"
REAR_WHEELS = "link_rear_.*_wheel"


# ------------------------------------------------------------------------------------------ commands


class BodyPoseCommand(CommandTerm):
    """Uniformly sampled trunk pose: height above the ground [m], roll [rad], pitch [rad]."""

    cfg: BodyPoseCommandCfg

    def __init__(self, cfg: BodyPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.pose = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose[:, 0] = NOMINAL_HEIGHT
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_tilt"] = torch.zeros(self.num_envs, device=self.device)
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or ["height", "roll", "pitch"]

    def __str__(self) -> str:
        return f"BodyPoseCommand:\n\tResampling time range: {self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        """Desired (height, roll, pitch), shape (num_envs, 3)."""
        return self.pose

    def _update_metrics(self):
        h = self.robot.data.root_pos_w.torch[:, 2]
        self.metrics["error_height"] = torch.abs(h - self.pose[:, 0])
        g = self.robot.data.projected_gravity_b.torch
        self.metrics["error_tilt"] = torch.linalg.norm(g - target_gravity_b(self.pose), dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        r = self.cfg.ranges
        u = torch.rand(n, 3, device=self.device)
        self.pose[env_ids, 0] = r.height[0] + (r.height[1] - r.height[0]) * u[:, 0]
        self.pose[env_ids, 1] = r.roll[0] + (r.roll[1] - r.roll[0]) * u[:, 1]
        self.pose[env_ids, 2] = r.pitch[0] + (r.pitch[1] - r.pitch[0]) * u[:, 2]
        neutral = torch.rand(n, device=self.device) < self.cfg.rel_neutral_envs
        ids = torch.as_tensor(env_ids, device=self.device)[neutral]
        self.pose[ids] = torch.tensor([NOMINAL_HEIGHT, 0.0, 0.0], device=self.device)

    def _update_command(self):
        pass


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    class_type: type = BodyPoseCommand
    asset_name: str = MISSING
    rel_neutral_envs: float = 0.15
    """Fraction of resamples that ask for the nominal stance."""

    @configclass
    class Ranges:
        height: tuple[float, float] = (0.28, 0.48)
        roll: tuple[float, float] = (-0.35, 0.35)
        pitch: tuple[float, float] = (-0.35, 0.35)

    ranges: Ranges = Ranges()


class RearUpCommand(CommandTerm):
    """Binary command: 1 = stand on the rear legs, 0 = normal four-wheel stance."""

    cfg: RearUpCommandCfg

    def __init__(self, cfg: RearUpCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.flag = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["up_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"RearUpCommand:\n\tResampling time range: {self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        return self.flag

    def _update_metrics(self):
        pitch_up = trunk_pitch_up(self.robot)
        self.metrics["up_pitch"] = torch.where(self.flag[:, 0] > 0.5, pitch_up, torch.zeros_like(pitch_up))

    def _resample_command(self, env_ids: Sequence[int]):
        self.flag[env_ids, 0] = (torch.rand(len(env_ids), device=self.device) < self.cfg.rel_up_envs).float()

    def _update_command(self):
        pass


@configclass
class RearUpCommandCfg(CommandTermCfg):
    class_type: type = RearUpCommand
    asset_name: str = MISSING
    rel_up_envs: float = 0.6


# ------------------------------------------------------------------------------------------ helpers / rewards


def target_gravity_b(pose: torch.Tensor) -> torch.Tensor:
    """Gravity direction in the trunk frame for the commanded roll and pitch (yaw-free)."""
    q = math_utils.quat_from_euler_xyz(pose[:, 1], pose[:, 2], torch.zeros_like(pose[:, 1]))
    down = torch.zeros(pose.shape[0], 3, device=pose.device)
    down[:, 2] = -1.0
    return math_utils.quat_apply_inverse(q, down)


def trunk_pitch_up(robot) -> torch.Tensor:
    """Nose-up angle of the trunk [rad] from the projected gravity (0 = level, pi/2 = vertical)."""
    # nose up tilts gravity toward the trunk's -x axis
    gx = robot.data.projected_gravity_b.torch[:, 0]
    return torch.asin(torch.clamp(-gx, -1.0, 1.0))


def track_body_height_exp(env: ManagerBasedRLEnv, std: float, command_name: str,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = robot.data.root_pos_w.torch[:, 2] - cmd[:, 0]
    return torch.exp(-torch.square(err) / std**2)


def track_body_tilt_exp(env: ManagerBasedRLEnv, std: float, command_name: str,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(robot.data.projected_gravity_b.torch - target_gravity_b(cmd)), dim=1)
    return torch.exp(-err / std**2)


def rear_up_pitch(env: ManagerBasedRLEnv, target: float, std: float, command_name: str,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Up: reward the nose-up angle (dense, linear up to the target) plus a bonus near the target.
    Down: reward a level trunk."""
    robot = env.scene[asset_cfg.name]
    up = env.command_manager.get_command(command_name)[:, 0] > 0.5
    pitch = trunk_pitch_up(robot)
    shaped = torch.clamp(pitch, 0.0, target) / target + torch.exp(-torch.square(pitch - target) / std**2)
    level = torch.exp(-torch.square(pitch) / std**2) * 2.0
    return torch.where(up, shaped, level)


def rear_up_contacts(env: ManagerBasedRLEnv, command_name: str, front_cfg: SceneEntityCfg,
                     rear_cfg: SceneEntityCfg) -> torch.Tensor:
    """Up: penalize front wheels on the ground and rear wheels in the air. Down: penalize any wheel in the air."""
    sensor = env.scene.sensors[front_cfg.name]
    f = sensor.data.net_forces_w.torch
    front = (torch.linalg.norm(f[:, front_cfg.body_ids], dim=-1) > 1.0).float().sum(dim=1)
    rear = (torch.linalg.norm(f[:, rear_cfg.body_ids], dim=-1) > 1.0).float().sum(dim=1)
    up = env.command_manager.get_command(command_name)[:, 0] > 0.5
    return torch.where(up, front + (2.0 - rear), (2.0 - front) + (2.0 - rear))


def rear_up_height(env: ManagerBasedRLEnv, command_name: str,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Up: reward a high trunk. Down: track the nominal height."""
    robot = env.scene[asset_cfg.name]
    h = robot.data.root_pos_w.torch[:, 2]
    up = env.command_manager.get_command(command_name)[:, 0] > 0.5
    return torch.where(up, torch.clamp(h - NOMINAL_HEIGHT, 0.0, 0.35) / 0.35,
                       torch.exp(-torch.square(h - NOMINAL_HEIGHT) / 0.05**2))


# ------------------------------------------------------------------------------------------ env cfgs


@configclass
class DobotQuadWheelDanceEnvCfg(DobotQuadWheelFlatEnvCfg):
    """Dance: track body height / roll / pitch while rolling or spinning on the wheels."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.body_pose = BodyPoseCommandCfg(asset_name="robot", resampling_time_range=(0.8, 2.5))
        vel = self.commands.base_velocity
        vel.resampling_time_range = (1.0, 3.0)
        vel.heading_command = False
        vel.rel_standing_envs = 0.3
        vel.ranges.lin_vel_x = (-0.6, 0.6)
        vel.ranges.lin_vel_y = (-0.4, 0.4)
        vel.ranges.ang_vel_z = (-2.5, 2.5)
        vel.ranges.heading = None
        self.observations.policy.pose_commands = ObsTerm(func=mdp.generated_commands,
                                                         params={"command_name": "body_pose"})
        # the pose command replaces the fixed height / level-trunk terms
        self.rewards.base_height = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.track_height = RewTerm(func=track_body_height_exp, weight=1.5,
                                            params={"std": 0.04, "command_name": "body_pose"})
        self.rewards.track_tilt = RewTerm(func=track_body_tilt_exp, weight=1.5,
                                          params={"std": 0.15, "command_name": "body_pose"})
        # dancing needs bent legs, vertical bounce and quick tilts
        self.rewards.leg_deviation.weight = -0.02
        self.rewards.abad_deviation.weight = -0.05
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.events.push_robot = None
        self.episode_length_s = 12.0


@configclass
class DobotQuadWheelRearUpEnvCfg(DobotQuadWheelFlatEnvCfg):
    """Rear-up: stand on the rear legs with the front wheels in the air, and come back down."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.rear_up = RearUpCommandCfg(asset_name="robot", resampling_time_range=(3.0, 6.0))
        vel = self.commands.base_velocity
        vel.heading_command = False
        vel.rel_standing_envs = 1.0
        vel.ranges.lin_vel_x = (0.0, 0.0)
        vel.ranges.lin_vel_y = (0.0, 0.0)
        vel.ranges.ang_vel_z = (0.0, 0.0)
        vel.ranges.heading = None
        self.observations.policy.rear_up_command = ObsTerm(func=mdp.generated_commands,
                                                           params={"command_name": "rear_up"})
        self.rewards.base_height = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.wheel_lift = None
        self.rewards.leg_deviation = None
        self.rewards.abad_deviation.weight = -0.05
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.005
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.rear_up_pitch = RewTerm(func=rear_up_pitch, weight=2.0,
                                             params={"target": math.radians(70.0), "std": 0.25,
                                                     "command_name": "rear_up"})
        self.rewards.rear_up_height = RewTerm(func=rear_up_height, weight=1.0, params={"command_name": "rear_up"})
        self.rewards.rear_up_contacts = RewTerm(
            func=rear_up_contacts, weight=-0.5,
            params={"command_name": "rear_up",
                    "front_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_WHEELS),
                    "rear_cfg": SceneEntityCfg("contact_forces", body_names=REAR_WHEELS)})
        # the rear calves may brush the ground while balancing; the trunk still ends the episode
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.events.push_robot = None
        self.episode_length_s = 15.0
