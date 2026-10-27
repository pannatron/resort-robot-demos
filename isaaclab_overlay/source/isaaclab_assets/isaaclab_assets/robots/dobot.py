# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Dobot quadruped robots.

The following configurations are available:

* :obj:`DOBOT_QUAD_CFG`: Dobot quadruped with point feet (12 DoF)
* :obj:`DOBOT_QUAD_WHEEL_CFG`: Dobot wheel-legged quadruped (12 leg DoF + 4 wheels)

The USD files were converted from the URDFs in the Dobot Quad SDK (MIT license) with
``fix_base=False`` and ``merge_fixed_joints=False`` so that the foot links stay separate bodies.

Reference: https://github.com/Dobot-Team/dobot_quad_sdk/tree/main/resources
"""

import os

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

DOBOT_DATA_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Dobot")

_RIGID_PROPS = PhysxRigidBodyCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
_ARTICULATION_PROPS = [
    PhysxArticulationCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    ),
    NewtonArticulationCfg(self_collision_enabled=False),
]

# Effort and velocity limits come from the URDF ``<limit>`` tags; armature from its ``<armature>`` tags.
# The gains follow the SDK's E9 low-level example (kp=30, kd=1.2).
DOBOT_LEG_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_abad", ".*_thigh_pitch", ".*_calf_pitch"],
    actuator_effort_limit={".*_abad": 23.0, ".*_thigh_pitch": 23.0, ".*_calf_pitch": 55.0},
    saturation_effort={".*_abad": 23.0, ".*_thigh_pitch": 23.0, ".*_calf_pitch": 55.0},
    actuator_velocity_limit=21.0,
    stiffness=30.0,
    damping=1.2,
    friction=0.0,
    armature={".*_abad": 0.01, ".*_thigh_pitch": 0.01, ".*_calf_pitch": 0.025},
)
"""Leg actuators shared by both Dobot variants."""

# Nominal stance: knees behind the hips, feet under the hips (verified with forward kinematics).
_LEG_JOINT_POS = {
    ".*_left_abad": 0.05,
    ".*_right_abad": -0.05,
    "joint_front_.*_thigh_pitch": 0.8,
    "joint_rear_.*_thigh_pitch": 0.9,
    ".*_calf_pitch": -1.5,
}

DOBOT_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(DOBOT_DATA_DIR, "dobot_quad", "usd", "dobot_quad", "dobot_quad.usda"),
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=_LEG_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"legs": DOBOT_LEG_ACTUATOR_CFG},
)
"""Configuration of the point-foot Dobot quadruped using a DC-motor actuator model."""

DOBOT_QUAD_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(DOBOT_DATA_DIR, "dobot_quad_wheel", "usd", "dobot_quad_wheel", "dobot_quad_wheel.usda"),
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={**_LEG_JOINT_POS, ".*_wheel": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DOBOT_LEG_ACTUATOR_CFG,
        # wheels are velocity-controlled: zero stiffness, damping acts as the velocity gain
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel"],
            actuator_effort_limit=20.0,
            saturation_effort=20.0,
            actuator_velocity_limit=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of the wheel-legged Dobot quadruped (legs: position control, wheels: velocity control)."""
