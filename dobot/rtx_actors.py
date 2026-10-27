"""Animated people / parcel / gate as physics-model bodies, so the OVRTX path tracer can move them.

OVRTX only updates transforms of the simulation model's shapes; meshes logged straight into the viewer stay
where they were first drawn. Each body part therefore becomes a collision-free kinematic rigid object that the
script teleports every step. Parts mirror :func:`common.person_parts` (tags, sizes, colors).
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

SKIN_LIGHT, SKIN_TAN = (0.86, 0.68, 0.54), (0.72, 0.54, 0.42)
PANTS, SHOES, HAIR = (0.16, 0.17, 0.22), (0.12, 0.11, 0.11), (0.10, 0.08, 0.07)


def person_specs(clothes, skin, vest=None):
    """(tag, shape, dims, color) for every part produced by common.person_parts."""
    S = []
    for nm in ("l", "r"):
        S += [(f"thigh{nm}", "capsule", (0.072, 0.43), PANTS), (f"shin{nm}", "capsule", (0.058, 0.42), PANTS),
              (f"foot{nm}", "capsule", (0.045, 0.1603), SHOES), (f"uarm{nm}", "capsule", (0.048, 0.29), clothes),
              (f"farm{nm}", "capsule", (0.04, 0.26), skin), (f"hand{nm}", "sphere", (0.045,), skin)]
    S += [("pelvis", "capsule_y", (0.12, 0.16), PANTS), ("torso", "capsule", (0.165, 0.22), clothes),
          ("neck", "capsule", (0.05, 0.08), skin), ("head", "sphere", (0.105,), skin),
          ("hair", "sphere", (0.097,), HAIR), ("nose", "sphere", (0.022,), skin)]
    if vest is not None:
        S += [("vest", "cylinder", (0.172, 0.08), vest), ("cap", "cylinder", (0.107, 0.07), (0.10, 0.12, 0.22)),
              ("brim", "cylinder", (0.09, 0.012), (0.10, 0.12, 0.22))]
    return S


PARCEL_SPECS = [("box", "cuboid", (0.36, 0.26, 0.20), (0.80, 0.62, 0.38)),
                ("tag", "cuboid", (0.12, 0.27, 0.006), (0.15, 0.45, 0.85))]
KIT_SPECS = [("case", "cuboid", (0.32, 0.24, 0.15), (0.95, 0.95, 0.95)),
             ("cross1", "cuboid", (0.16, 0.05, 0.006), (0.85, 0.1, 0.1)),
             ("cross2", "cuboid", (0.05, 0.16, 0.006), (0.85, 0.1, 0.1))]
GATE_SPECS = [("leaf", "cuboid", (3.0, 0.08, 1.5), (0.35, 0.25, 0.16))]


def _spawn(shape, dims, color):
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )
    if shape == "sphere":
        return sim_utils.SphereCfg(radius=dims[0], **common)
    if shape == "capsule":
        return sim_utils.CapsuleCfg(radius=dims[0], height=dims[1], axis="Z", **common)
    if shape == "capsule_y":
        return sim_utils.CapsuleCfg(radius=dims[0], height=dims[1], axis="Y", **common)
    if shape == "cylinder":
        return sim_utils.CylinderCfg(radius=dims[0], height=dims[1], axis="Z", **common)
    return sim_utils.CuboidCfg(size=dims, **common)


def add_actor_objects(env_cfg, actors: dict) -> None:
    """actors: name -> specs list. Adds one rigid object per part, parked far below the resort."""
    k = 0
    for name, specs in actors.items():
        for tag, shape, dims, color in specs:
            setattr(env_cfg.scene, f"act_{name}_{tag}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Act_{name}_{tag}",
                spawn=_spawn(shape, dims, color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(float(k), 0.0, -40.0)),
            ))
            k += 1


class ActorDriver:
    """Teleports the part objects of env 0 to the poses computed by common.person_parts / place."""

    def __init__(self, uenv, actors: dict, offset: np.ndarray, model=None):
        self.objs = {}
        self.colors = {}
        for name, specs in actors.items():
            for tag, _shape, _dims, color in specs:
                self.objs[(name, tag)] = uenv.scene[f"act_{name}_{tag}"]
                self.colors[(name, tag)] = color
        self.offset = np.asarray(offset, float)
        self.device = uenv.device
        self.zero_v = torch.zeros(1, 6, device=self.device)
        self.env0 = torch.zeros(1, dtype=torch.int32, device=self.device)  # other envs keep their parked copies
        if model is not None:  # make sure the renderer gets the intended colors
            self._paint(model)

    def _paint(self, model):
        labels = list(model.body_label)
        body = model.shape_body.numpy()
        col = model.shape_color.numpy() if model.shape_color is not None else None
        if col is None:
            return
        for (name, tag), color in self.colors.items():
            key = f"Act_{name}_{tag}"
            for b, lab in enumerate(labels):
                if key in str(lab):
                    col[body == b] = color
        model.shape_color.assign(col)

    def set(self, name, parts) -> None:
        for tag, _mesh, _color, T in parts:
            obj = self.objs.get((name, tag))
            if obj is None:
                continue
            q = trimesh.transformations.quaternion_from_matrix(T)  # wxyz
            pose = torch.tensor([[*(T[:3, 3] + self.offset), q[1], q[2], q[3], q[0]]], dtype=torch.float32,
                                device=self.device)
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=self.env0)
            obj.write_root_velocity_to_sim_index(root_velocity=self.zero_v, env_ids=self.env0)

    def park(self, name) -> None:
        for (n, tag), obj in self.objs.items():
            if n == name:
                pose = torch.tensor([[0.0, 0.0, -40.0, 0, 0, 0, 1]], dtype=torch.float32, device=self.device)
                obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=self.env0)
