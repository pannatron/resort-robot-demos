# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-deployable variants of the Dobot quadruped velocity-tracking environments.

The robot's low-level DDS interface provides IMU and joint states but no base linear velocity, so the actor
observes only what the hardware measures. The critic keeps ``base_lin_vel`` as a privileged observation.
Command latency, PD gains, and ground friction are randomized to cover the gap to the real robot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .flat_env_cfg import DobotQuadFlatEnvCfg
from .rough_env_cfg import DobotQuadRoughEnvCfg
from .wheel_env_cfg import DobotQuadWheelFlatEnvCfg, DobotQuadWheelRoughEnvCfg

from isaaclab_assets.robots.dobot import DOBOT_LEG_ACTUATOR_CFG  # isort: skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

MAX_DELAY_STEPS = 4
"""Upper bound of the command delay in physics steps (4 x 5 ms = 20 ms)."""


def randomize_command_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_delay: int,
) -> None:
    """Samples one command delay per environment, shared by all of the robot's actuators.

    Newton's native delay component builds every environment with ``max_delay`` steps and ignores
    :attr:`~isaaclab.actuators.DelayedPDActuatorCfg.min_delay`, so the delay is re-sampled here on reset.

    Args:
        env: The environment.
        env_ids: Environments to update. Defaults to all.
        asset_cfg: The articulation whose actuators are delayed.
        max_delay: Largest delay to sample [physics steps], at most the actuators' ``max_delay``.
    """
    from isaaclab.actuators.newton import write_group_parameter  # noqa: PLC0415

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    delay = torch.randint(0, max_delay + 1, (len(env_ids), 1), device=asset.device, dtype=torch.int32)
    for name, actuator_cfg in asset.cfg.actuators.items():
        # Newton may merge groups into one actuator object, so size each group from its own joints
        num_joints = len(asset.find_joints(actuator_cfg.joint_names_expr)[0])
        write_group_parameter(
            asset.actuators, name, "delay", "delay_steps", values=delay.expand(-1, num_joints), env_ids=env_ids
        )


def _delayed(actuator_cfg) -> DelayedPDActuatorCfg:
    """Returns a delayed PD actuator with the same joints, limits, and gains as ``actuator_cfg``."""
    return DelayedPDActuatorCfg(
        joint_names_expr=actuator_cfg.joint_names_expr,
        actuator_effort_limit=actuator_cfg.actuator_effort_limit,
        actuator_velocity_limit=actuator_cfg.actuator_velocity_limit,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        armature=actuator_cfg.armature,
        friction=actuator_cfg.friction,
        min_delay=0,
        max_delay=MAX_DELAY_STEPS,
    )


def make_deployable(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Restricts the actor to hardware-measurable observations and adds sim-to-real randomization."""
    # observations: privileged critic, hardware-only actor
    env_cfg.observations.critic = env_cfg.observations.policy.replace(enable_corruption=False)
    env_cfg.observations.policy.base_lin_vel = None
    # actuators: random command latency
    robot = env_cfg.scene.robot
    robot.actuators = {name: _delayed(cfg) for name, cfg in robot.actuators.items()}
    env_cfg.events.command_delay = EventTerm(
        func=randomize_command_delay,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "max_delay": MAX_DELAY_STEPS},
    )
    # events: wider friction, per-reset PD gain error
    env_cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
    env_cfg.events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=DOBOT_LEG_ACTUATOR_CFG.joint_names_expr),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class DobotQuadDeployFlatEnvCfg(DobotQuadFlatEnvCfg):
    """Deployable point-foot Dobot quadruped on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        make_deployable(self)


@configclass
class DobotQuadDeployRoughEnvCfg(DobotQuadRoughEnvCfg):
    """Deployable point-foot Dobot quadruped on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        make_deployable(self)


@configclass
class DobotQuadWheelDeployFlatEnvCfg(DobotQuadWheelFlatEnvCfg):
    """Deployable wheel-legged Dobot quadruped on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        make_deployable(self)


@configclass
class DobotQuadWheelDeployRoughEnvCfg(DobotQuadWheelRoughEnvCfg):
    """Deployable wheel-legged Dobot quadruped on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        make_deployable(self)
