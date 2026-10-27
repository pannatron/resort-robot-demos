"""Isaac Lab env for the RPO humanoid at the resort: Roboparty's RPO on the resort collision mesh.

Reuses the resort scene of the Dobot demo (~/dobot_tools/resort) and the RPO asset of ~/w7work/rpo_lab.
The skill runner (rpo_skills.py) computes every policy's inputs itself, so the env only needs the robot,
the terrain and the joint-position action term (POLICY_JOINTS order, standing pose + 0.25 x action).
Falls do not end the episode: the robot gets up by itself with the get-up policy.
"""

from __future__ import annotations

import os
import sys

import numpy as np

DOBOT = os.path.expanduser("~/dobot_tools/resort")
RPO_LAB = os.path.expanduser("~/w7work/rpo_lab/src")
for p in (DOBOT, RPO_LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg  # noqa: E402

import resort_scene as rs  # noqa: E402
from rpo_lab.robot import POLICY_JOINTS, RPO_CFG  # noqa: E402

SCENE = rs.build()
# Collision: Roboparty's MJCF (their sim2sim, where get-up was verified) gives every link a full-shell mesh
# collision; the URDF only has thin boxes / cylinders, so face-down falls sink in and the get-up fails.
# assets/rpo_full/urdf/rpo.urdf = the URDF with each primitive-only link's visual mesh as its collision.
# RPO_COLLISION=urdf selects the original asset.
FULL_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/usd_full/rpo/rpo.usda")
RPO_USD = FULL_USD if os.environ.get("RPO_COLLISION", "full") == "full" else RPO_CFG.spawn.usd_path
X0, X1, Y0, Y1 = -12.0, 72.0, -80.0, 16.0  # extent of the collision mesh (design frame), as in patrol_env_cfg


def resort_terrain(difficulty: float, cfg: SubTerrainBaseCfg):
    mesh = SCENE.collision_mesh()
    mesh.apply_translation((-X0, -Y0, 0.0))
    dx, dy, _ = SCENE.dock
    return [mesh], np.array([dx - X0, dy - Y0, 0.0])


@configclass
class ResortTerrainCfg(SubTerrainBaseCfg):
    function = resort_terrain


@configclass
class RpoResortEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # --- robot: RPO instead of G1 (same as rpo_lab use_rpo, without the observation manager inputs)
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                           spawn=RPO_CFG.spawn.replace(usd_path=RPO_USD))
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        self.scene.height_scanner = None
        self.actions.joint_pos.joint_names = POLICY_JOINTS
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.rewards = None
        self.curriculum = None
        for ev in ("add_base_mass", "base_external_force_torque", "push_robot", "physics_material", "base_com"):
            if hasattr(self.events, ev):
                setattr(self.events, ev, None)
        # --- resort terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(X1 - X0, Y1 - Y0), border_width=0.0, num_rows=1, num_cols=1, horizontal_scale=0.1,
            vertical_scale=0.005, slope_threshold=None, use_cache=False, curriculum=False,
            sub_terrains={"resort": ResortTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0,
            dynamic_friction=1.0,
        )
        # lying on the ground (get-up) touches the floor with many links at once
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 600
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 200
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        # the job controller owns the command and the episode; a fall is handled by the get-up skill
        self.episode_length_s = 1.0e6
        self.terminations.time_out = None
        self.terminations.base_contact = None
        if hasattr(self.terminations, "bad_orientation"):
            self.terminations.bad_orientation = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.debug_vis = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


TASK = "RPO-Resort"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:RpoResortEnvCfg"})
