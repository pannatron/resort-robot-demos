"""Isaac Lab env for the resort patrol: the all-terrain wheel-leg task on a resort-sized terrain tile.

All environments share one resort (Newton keeps each env in its own collision world, so robots
never touch each other), which lets the robustness test run many randomized laps in parallel.
"""

from __future__ import annotations

import math

import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen  # noqa: F401
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.dobot_quad.wheel_env_cfg import DobotQuadWheelRoughEnvCfg

import resort_scene as rs

SCENE = rs.build()
X0, X1, Y0, Y1 = -12.0, 72.0, -80.0, 16.0  # extent of the collision mesh (design frame)
NUM_CHAIRS, NUM_PLANTERS = 2, 2
PARK = [(-10.0 + 1.5 * i, -77.0) for i in range(NUM_CHAIRS + NUM_PLANTERS)]  # off-route storage (beach)


def resort_terrain(difficulty: float, cfg: SubTerrainBaseCfg):
    mesh = SCENE.collision_mesh()
    mesh.apply_translation((-X0, -Y0, 0.0))
    dx, dy, _ = SCENE.dock
    return [mesh], np.array([dx - X0, dy - Y0, 0.0])


@configclass
class ResortTerrainCfg(SubTerrainBaseCfg):
    function = resort_terrain


def _obstacle(kind: str, idx: int) -> RigidObjectCfg:
    px, py = PARK[idx]
    if kind == "chair":
        spawn = sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.85),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=7.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.18)),
        )
        z = 0.43
    else:
        spawn = sim_utils.CylinderCfg(
            radius=0.32,
            height=0.75,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.52)),
        )
        z = 0.38
    # positions are relative to the env origin (the dock)
    dx, dy, _ = SCENE.dock
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle{idx}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(px - dx, py - dy, rs.L_BEACH + z)),
    )


@configclass
class ResortPatrolEnvCfg(DobotQuadWheelRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(X1 - X0, Y1 - Y0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=None,
            use_cache=False,
            curriculum=False,
            sub_terrains={"resort": ResortTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = None
        # floor friction is set per zone on the robot's wheels; contacts use max(robot, ground)
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max", restitution_combine_mode="multiply", static_friction=0.0,
            dynamic_friction=0.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        for i in range(NUM_CHAIRS + NUM_PLANTERS):
            kind = "chair" if i < NUM_CHAIRS else "planter"
            setattr(self.scene, f"obstacle{i}", _obstacle(kind, i))
        # the patrol controller owns the command and the episode
        self.episode_length_s = 1.0e5
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.debug_vis = False
        self.terminations.time_out = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
        # start exactly on the dock, facing east
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None  # the robustness test turns pushes on
        self.observations.policy.enable_corruption = True


PUSH_TERM_PARAMS = {"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}}
PUSH_INTERVAL_S = (8.0, 15.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
