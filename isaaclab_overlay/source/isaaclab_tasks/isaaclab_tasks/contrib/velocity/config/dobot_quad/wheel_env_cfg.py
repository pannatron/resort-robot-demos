# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the wheel-legged Dobot quadruped velocity-tracking environments.

The legs are driven by joint-position targets and the wheels by joint-velocity targets.
Wheel angles are unbounded, so they are left out of the joint-position observation.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .flat_env_cfg import make_flat
from .rough_env_cfg import base_height_above_terrain_l2, configure_dobot_scene

from isaaclab_assets.robots.dobot import DOBOT_QUAD_WHEEL_CFG  # isort: skip

LEG_JOINTS = [".*_abad", ".*_thigh_pitch", ".*_calf_pitch"]
WHEEL_JOINTS = [".*_wheel"]

WHEEL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # baseline where rolling is cheapest
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        # hills: climb up to a summit platform, or descend into a valley
        "hill_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "hill_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(proportion=0.10, amplitude_range=(0.0, 0.15), num_waves=4),
        # stairs up and down; the tallest step is ~1.7x the wheel radius (0.0875 m)
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.175,
            step_height_range=(0.02, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=True,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.175,
            step_height_range=(0.02, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=True,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
            convert_to_heightfield=True,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)
"""All-terrain course for the wheel-legged robot: hills, waves, stairs, steps, and rough ground."""


@configclass
class DobotQuadWheelRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the wheel-legged Dobot quadruped on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        configure_dobot_scene(self, DOBOT_QUAD_WHEEL_CFG)
        # actions: leg position targets + wheel velocity targets [rad/s]
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True
        )
        self.actions.wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINTS, scale=10.0)
        # observations: wheel angle is meaningless, wheel speed is kept
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # rewards: roll rather than step
        self.rewards.feet_air_time = None
        # shins count too: without this the policy kneels on its rear calves
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(thigh|calf)"
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.leg_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        # same gait shaping as the point-foot task: keep the hips under the body
        self.rewards.abad_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_abad")},
        )
        self.rewards.wheel_lift = RewTerm(
            func=mdp.feet_air_time,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
                "command_name": "base_velocity",
                "threshold": 0.0,
            },
        )
        # all-terrain: stairs and steps need the wheels to leave the ground, so lifting is cheap here
        self.scene.terrain.terrain_generator = WHEEL_TERRAINS_CFG
        self.rewards.wheel_lift.weight = -0.05
        self.rewards.leg_deviation.weight = -0.05
        self.rewards.base_height = RewTerm(
            func=base_height_above_terrain_l2,
            weight=-10.0,
            params={"target_height": 0.40, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )


@configclass
class DobotQuadWheelFlatEnvCfg(DobotQuadWheelRoughEnvCfg):
    """Configuration for the wheel-legged Dobot quadruped on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        make_flat(self)
        # on flat ground the wheels should stay down and the legs near the stance
        self.rewards.wheel_lift.weight = -0.5
        self.rewards.leg_deviation.weight = -0.1
        # nominal stance is ~0.44 m; the target leaves room to bend the knees
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2, weight=-30.0, params={"target_height": 0.40})


STAIR_WALK_TERRAINS_CFG = WHEEL_TERRAINS_CFG.replace(
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "stairs_up": WHEEL_TERRAINS_CFG.sub_terrains["stairs_up"].replace(
            proportion=0.35, step_height_range=(0.04, 0.18)
        ),
        "stairs_down": WHEEL_TERRAINS_CFG.sub_terrains["stairs_down"].replace(
            proportion=0.35, step_height_range=(0.04, 0.18)
        ),
        "boxes": WHEEL_TERRAINS_CFG.sub_terrains["boxes"].replace(proportion=0.1),
        "hill_up": WHEEL_TERRAINS_CFG.sub_terrains["hill_up"].replace(proportion=0.1),
    }
)
"""Stair-heavy course for the walking mode; the tallest step (0.18 m) is above a resort stair (~0.15 m)."""


@configclass
class DobotQuadWheelStairWalkEnvCfg(DobotQuadWheelRoughEnvCfg):
    """Walking mode of the wheel-legged Dobot quadruped: wheels braked, the legs step over stairs.

    The wheel action is kept (scale 0) so the observation and action layouts match the rolling policy,
    which lets a patrol controller switch between the two policies on the same robot.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = STAIR_WALK_TERRAINS_CFG
        # wheels hold zero velocity: the wheel actuator's damping acts as a brake
        self.actions.wheel_vel.scale = 0.0
        # step instead of rolling: reward swing phases, penalize wheels moving while in contact
        self.rewards.wheel_lift = None
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
                "command_name": "base_velocity",
                "threshold": 0.4,
            },
        )
        self.rewards.wheel_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel"),
            },
        )
        # patrol speeds on stairs are slow; a narrower range gives the curriculum cleaner signal
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)


def make_payload(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Carry 0-5 kg on the trunk (first-aid kit, parcel) smoothly: level body, no jerky motion."""
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (0.01, 5.0)
    env_cfg.events.add_base_mass.params["operation"] = "add"
    env_cfg.events.add_base_mass.params["distribution"] = "uniform"
    env_cfg.rewards.flat_orientation_l2.weight = -2.5
    env_cfg.rewards.ang_vel_xy_l2.weight = -0.1
    env_cfg.rewards.action_rate_l2.weight = -0.02


@configclass
class DobotQuadWheelRoughPayloadEnvCfg(DobotQuadWheelRoughEnvCfg):
    """Rolling policy fine-tuned with a random 0-5 kg payload on the trunk."""

    def __post_init__(self):
        super().__post_init__()
        make_payload(self)


@configclass
class DobotQuadWheelStairWalkPayloadEnvCfg(DobotQuadWheelStairWalkEnvCfg):
    """Stair-walking policy fine-tuned with a random 0-5 kg payload on the trunk."""

    def __post_init__(self):
        super().__post_init__()
        make_payload(self)
