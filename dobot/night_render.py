"""Night look for the Newton GL viewer: dim moonlight, scene meshes, and post-process lamp glow.

The GL viewer has one directional light and no emissive materials, so lamps are faked twice:
warm translucent light pools on the ground (in the scene) and additive glow sprites (in the frame),
hidden when the collision mesh occludes the lamp from the camera.
"""

from __future__ import annotations

import math
import os

import numpy as np
import warp as wp

import resort_scene as rs

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# material -> (Poly Haven texture, tile size [m], tint); textures are CC0 (assets/LICENSE.txt)
MATERIALS = {
    "marble": ("marble_01", 2.5, (1.0, 1.0, 1.0)), "paver": ("square_brick_paving", 2.0, (0.95, 0.92, 0.9)),
    "carpet": ("dirty_carpet", 1.2, (0.95, 0.35, 0.33)), "tile": ("rectangular_paving", 2.5, (1.0, 0.97, 0.92)),
    "grass": ("leafy_grass", 2.5, (0.85, 1.0, 0.8)), "stone": ("precast_stone_paving", 1.8, (1.0, 1.0, 1.0)),
    "wall": ("painted_plaster_wall", 3.0, (1.0, 0.98, 0.94)), "wood": ("brown_planks_05", 1.2, (1.0, 1.0, 1.0)),
    "roof": ("clay_roof_tiles", 2.0, (1.0, 1.0, 1.0)), "sand": ("coast_sand_01", 3.0, (1.0, 1.0, 1.0)),
    "trunk": ("bark_brown_01", 1.0, (1.0, 1.0, 1.0)), "concrete": ("brushed_concrete", 2.0, (1.0, 1.0, 1.0)),
    "hedge": ("leafy_grass", 1.0, (0.45, 0.7, 0.45)), "leaf": ("leafy_grass", 1.2, (0.55, 0.85, 0.5)),
}
SKIES = {"morning": "qwantani_morning_puresky_4k.jpg", "day": "kloofendal_43d_clear_puresky_4k.jpg",
         "night": "qwantani_night_puresky_4k.jpg"}


def _load_tex(name):
    from PIL import Image

    return np.asarray(Image.open(os.path.join(ASSETS, "tex", name + ".jpg")).convert("RGB"))


def triplanar_uv(m, tile, smooth=False):
    """Unshares vertices and projects UVs along each face's dominant axis (box-mapping)."""
    import trimesh

    corner_n = np.asarray(m.vertex_normals)[m.faces.flatten()] if smooth else None
    m = m.copy()
    m.unmerge_vertices()
    fn = m.face_normals
    ax = np.abs(fn).argmax(1)
    v = m.vertices
    uv = np.zeros((len(v), 2))
    for i, f in enumerate(m.faces):
        a = ax[i]
        keep = [k for k in range(3) if k != a]
        uv[f] = v[f][:, keep] / tile
    if corner_n is None:
        corner_n = np.repeat(fn, 3, axis=0)
    out_n = np.zeros_like(v)
    out_n[m.faces.flatten()] = corner_n
    return trimesh.Trimesh(v, m.faces, process=False), uv.astype(np.float32), out_n.astype(np.float32)


def setup_night(viewer) -> None:
    r = viewer.renderer
    r.sky_upper = (0.015, 0.025, 0.06)
    r.sky_lower = (0.03, 0.035, 0.06)
    r.background_color = r.sky_upper
    r.ambient_sky = (0.07, 0.08, 0.14)
    r.ambient_ground = (0.035, 0.03, 0.03)
    r._light_color = (0.11, 0.13, 0.22)  # moonlight
    r.draw_sky = True
    r.draw_shadows = True
    r.spotlight_enabled = False
    d = np.array([-0.35, 0.45, 0.82])
    r._sun_direction = d / np.linalg.norm(d)
    r.shadow_extents = 30.0


def setup_day(viewer, morning: bool = True) -> None:
    r = viewer.renderer
    r.sky_upper = (0.42, 0.62, 0.88) if morning else (0.35, 0.58, 0.9)
    r.sky_lower = (0.95, 0.80, 0.62) if morning else (0.80, 0.85, 0.9)
    r.background_color = r.sky_upper
    r.ambient_sky = (0.55, 0.58, 0.66)
    r.ambient_ground = (0.32, 0.28, 0.24)
    r._light_color = (1.0, 0.86, 0.70) if morning else (1.0, 0.97, 0.92)
    r.draw_sky = True
    r.draw_shadows = True
    r.spotlight_enabled = False
    d = np.array([0.75, -0.25, 0.45]) if morning else np.array([0.3, -0.35, 0.88])
    r._sun_direction = d / np.linalg.norm(d)
    r.shadow_extents = 30.0


def _wp_mesh(m, device):
    pts = wp.array(np.asarray(m.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    idx = wp.array(np.asarray(m.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=device)
    return pts, idx


def _normals(m):
    return np.asarray(m.vertex_normals, dtype=np.float32)


class NightScene:
    """Logs the resort into a viewer (once) and adds lamp glow to captured frames."""

    def __init__(self, scene: rs.Scene, offset: np.ndarray, device: str = "cuda:0"):
        self.scene = scene
        self.offset = np.asarray(offset, dtype=np.float64)  # design frame -> world
        self.device = device
        self.collision = scene.collision_mesh()
        cm = self.collision.copy()
        cm.apply_translation(self.offset)
        self._occ_pts = wp.array(np.asarray(cm.vertices, np.float32), dtype=wp.vec3, device=device)
        self._occ_idx = wp.array(np.asarray(cm.faces, np.int32).flatten(), dtype=wp.int32, device=device)
        self._occ_mesh = wp.Mesh(points=self._occ_pts, indices=self._occ_idx)
        self.lamp_world = []
        self.lamp_kind = []
        for x, y, z, kind in scene.lamps:
            self.lamp_world.append(np.array([x, y, z]) + self.offset)
            self.lamp_kind.append(kind)
        self.lamp_world = np.array(self.lamp_world)

    def log_parts(self, viewer, actor: str, parts) -> None:
        """Draw rigid parts as single instances of cached meshes; per-frame motion goes through transforms."""
        import trimesh

        names = self.__dict__.setdefault("_meshname", {})
        for tag, mesh, color, T in parts:
            key = f"{actor}/{tag}"
            if key not in names:  # each part's mesh is registered once (hidden prototype), then only moved
                names[key] = f"/actormesh/{actor}_{tag}"
                pts = wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=self.device)
                idx = wp.array(np.asarray(mesh.faces, np.int32).flatten(), dtype=wp.int32, device=self.device)
                nrm = wp.array(np.asarray(mesh.vertex_normals, np.float32), dtype=wp.vec3, device=self.device)
                viewer.log_mesh(names[key], pts, idx, normals=nrm, color=tuple(color), hidden=True)
            q = trimesh.transformations.quaternion_from_matrix(T)  # wxyz
            pos = T[:3, 3] + self.offset
            xf = wp.array([wp.transform(wp.vec3(*pos.astype(np.float32)), wp.quat(q[1], q[2], q[3], q[0]))],
                          dtype=wp.transform, device=self.device)
            col = wp.array([wp.vec3(*color)], dtype=wp.vec3, device=self.device)
            mat = wp.array([wp.vec4(0.8, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
            viewer.log_instances(f"/actor/{key}", names[key], xf, None, col, mat)

    def ground_z(self, x: float, y: float) -> float:
        """Top surface height (design frame) under (x, y), ignoring overhangs above 2.5 m."""
        loc, _, _ = self.collision.ray.intersects_location([[x, y, 10.0]], [[0, 0, -1.0]])
        if len(loc) == 0:
            return rs.L_LOW
        zs = sorted(loc[:, 2], reverse=True)
        return float(zs[0])

    def _log(self, viewer, name, mesh, color, **kw):
        m = mesh.copy()
        m.apply_translation(self.offset)
        pts, idx = _wp_mesh(m, self.device)
        given = kw.pop("normals", None)
        nrm = wp.array(_normals(m) if given is None else given, dtype=wp.vec3, device=self.device)
        uvs = kw.pop("uvs", None)
        if uvs is not None:
            uvs = wp.array(np.asarray(uvs, np.float32), dtype=wp.vec2, device=self.device)
        textured = kw.get("texture") is not None
        viewer.log_mesh(name, pts, idx, normals=nrm, uvs=uvs, color=tuple(color), **kw)
        if textured and hasattr(type(viewer), "objects") or (textured and "objects" in vars(viewer)):
            # GL viewer: log_mesh uploads the image but leaves the material's texture flag off
            obj = viewer.objects[viewer._qualify(name)]
            r, m_, c, _ = obj.material
            obj.material = (r, m_, c, 1.0)

    def log_static(self, viewer, lamps_on: bool = True, textured: bool = True) -> None:
        import trimesh

        self.lamps_on = lamps_on
        groups = {}
        for p in self.scene.pieces:
            tex = MATERIALS.get(p.mat) if textured and p.opacity is None else None
            key = ("tex", p.mat) if tex else ("col", p.color, p.opacity, p.roughness, p.metallic)
            groups.setdefault(key, []).append(p)
        for gi, (key, ps) in enumerate(groups.items()):
            p0 = ps[0]
            kw = dict(roughness=p0.roughness, metallic=p0.metallic, backface_culling=False)
            if p0.opacity is not None:
                kw["opacity"] = p0.opacity
            if key[0] == "tex":
                name, tile, tint = MATERIALS[p0.mat]
                parts = [triplanar_uv(p.mesh, tile, smooth=p0.mat == "leaf") for p in ps]
                verts, faces, uvs, nrms, n0 = [], [], [], [], 0
                for m, uv, nr in parts:
                    verts.append(m.vertices)
                    faces.append(m.faces + n0)
                    uvs.append(uv)
                    nrms.append(nr)
                    n0 += len(m.vertices)
                mesh = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
                tex = np.clip(_load_tex(name).astype(np.float32) * np.array(tint)[None, None], 0, 255).astype(np.uint8)
                self._log(viewer, f"/resort/g{gi}", mesh, (1.0, 1.0, 1.0), texture=tex, uvs=np.vstack(uvs),
                          normals=np.vstack(nrms), **kw)
            else:
                mesh = trimesh.util.concatenate([p.mesh for p in ps])
                self._log(viewer, f"/resort/g{gi}", mesh, p0.color, **kw)
        # lamps: fixtures + warm light pools on the ground
        for i, (x, y, z, kind) in enumerate(self.scene.lamps):
            gz = self.ground_z(x, y) if kind != "wall" else self.ground_z(x, y - 0.6)
            if kind == "pole":
                self._log(viewer, f"/lamp/pole{i}", rs.cyl(x, y, 0.05, gz, z), rs.C["metal"])
                self._log(viewer, f"/lamp/head{i}", rs.sphere(x, y, z + 0.1, 0.18), rs.C["lamp"], roughness=0.2)
                r_pool = 3.2
            elif kind == "bollard":
                self._log(viewer, f"/lamp/pole{i}", rs.cyl(x, y, 0.08, gz, z), rs.C["metal"])
                self._log(viewer, f"/lamp/head{i}", rs.cyl(x, y, 0.09, z, z + 0.12), rs.C["lamp"])
                r_pool = 1.6
            elif kind == "ceiling":
                self._log(viewer, f"/lamp/head{i}", rs.cyl(x, y, 0.35, z, z + 0.05), rs.C["lamp"])
                r_pool = 3.0
            else:  # wall sconce
                self._log(viewer, f"/lamp/head{i}", rs.box(x - 0.12, x + 0.12, y - 0.06, y + 0.06, z - 0.15, z + 0.15),
                          rs.C["lamp"])
                r_pool = 1.8
            px, py = (x, y) if kind != "wall" else (x, y - 0.9)
            if not lamps_on:
                continue
            for j, (rr, op) in enumerate([(r_pool * f, 0.10) for f in (1.0, 0.82, 0.66, 0.5, 0.36, 0.22)]):
                self._log(viewer, f"/lamp/pool{i}_{j}", rs.disc(px, py, gz + 0.01 + 0.002 * j, rr),
                          rs.C["glow"], opacity=op, roughness=1.0)
        # pool closing-time sign (textured board)
        sx, sy = self.scene.sign_text_pos
        z0 = rs.L_LOW + 0.75
        b = rs.box(sx - 0.03, sx, sy - 0.45, sy + 0.45, z0, z0 + 0.6)
        self._log(viewer, "/resort/sign_back", b, rs.C["metal"])
        import trimesh

        v = np.array([[sx - 0.035, sy + 0.45, z0], [sx - 0.035, sy - 0.45, z0],
                      [sx - 0.035, sy - 0.45, z0 + 0.6], [sx - 0.035, sy + 0.45, z0 + 0.6]])
        face = trimesh.Trimesh(v, np.array([[0, 2, 1], [0, 3, 2]]), process=False)
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
        self._log(viewer, "/resort/sign_face", face, (1, 1, 1), texture=rs.sign_texture(), uvs=uv,
                  backface_culling=False, roughness=0.6)

    # ------------------------------------------------------------------------------------- post-process

    def project(self, viewer, pts_world: np.ndarray, width: int, height: int):
        """World points -> pixel coords (u, v), depth, and an in-front mask."""
        cam = viewer.camera
        view = np.array(cam.get_view_matrix(), dtype=np.float64).reshape(4, 4).T
        proj = np.array(cam.get_projection_matrix(), dtype=np.float64).reshape(4, 4).T
        p = np.concatenate([pts_world, np.ones((len(pts_world), 1))], 1)
        clip = (proj @ view @ p.T).T
        w = clip[:, 3]
        ndc = clip[:, :3] / np.where(np.abs(w) < 1e-9, 1e-9, w)[:, None]
        u = (ndc[:, 0] * 0.5 + 0.5) * width
        v = (1 - (ndc[:, 1] * 0.5 + 0.5)) * height
        return u, v, w, w > 0.05

    def visible(self, eye: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
        """True where the straight line eye -> point is not blocked by the collision mesh."""
        n = len(pts_world)
        if n == 0:
            return np.zeros(0, bool)
        d = pts_world - eye[None]
        dist = np.linalg.norm(d, axis=1)
        dirs = d / np.maximum(dist[:, None], 1e-6)
        out = wp.zeros(n, dtype=wp.int32, device=self.device)
        wp.launch(_occlusion_kernel, dim=n, device=self.device, inputs=[
            self._occ_mesh.id, wp.vec3(*eye.astype(np.float32)),
            wp.array(dirs.astype(np.float32), dtype=wp.vec3, device=self.device),
            wp.array((dist - 0.35).astype(np.float32), dtype=wp.float32, device=self.device), out])
        return out.numpy() == 0

    def add_glow(self, viewer, frame: np.ndarray) -> np.ndarray:
        if not getattr(self, "lamps_on", True):
            return frame
        h, w = frame.shape[:2]
        if len(self.lamp_world) == 0:
            return frame
        heads = self.lamp_world.copy()
        heads[:, 2] += np.where(np.array(self.lamp_kind) == "pole", 0.1, 0.05)
        u, v, depth, front = self.project(viewer, heads, w, h)
        eye = np.array(viewer.camera.pos, dtype=np.float64)
        vis = self.visible(eye, heads) & front & (u > -50) & (u < w + 50) & (v > -50) & (v < h + 50)
        out = frame.astype(np.float32)
        if getattr(self, "_grid_shape", None) != (h, w):
            self._yy, self._xx = np.mgrid[0:h, 0:w].astype(np.float32)
            self._grid_shape = (h, w)
        yy, xx = self._yy, self._xx
        f_px = h / (2 * math.tan(math.radians(viewer.camera.fov) / 2))
        for i in np.nonzero(vis)[0]:
            size = {"pole": 0.9, "bollard": 0.35, "ceiling": 0.8, "wall": 0.45}[self.lamp_kind[i]]
            rad = float(np.clip(size * f_px / depth[i], 3.0, 90.0))
            ext = 4.5 * rad
            x0, x1 = int(max(u[i] - ext, 0)), int(min(u[i] + ext, w))
            y0, y1 = int(max(v[i] - ext, 0)), int(min(v[i] + ext, h))
            if x1 <= x0 or y1 <= y0:
                continue
            r2 = (xx[y0:y1, x0:x1] - u[i]) ** 2 + (yy[y0:y1, x0:x1] - v[i]) ** 2
            g = np.exp(-r2 / (2 * (rad * 0.55) ** 2)) * 1.0 + np.exp(-r2 / (2 * (rad * 1.3) ** 2)) * 0.22
            out[y0:y1, x0:x1] += g[..., None] * np.array([255, 205, 140], np.float32)[None, None]
        return np.clip(out, 0, 255).astype(np.uint8)


@wp.kernel
def _occlusion_kernel(mesh: wp.uint64, eye: wp.vec3, dirs: wp.array(dtype=wp.vec3),
                      max_t: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, eye, dirs[i], max_t[i])
    if q.result:
        out[i] = 1


def sky_key(kind: str):
    """Horizon color of a sky photo: used as the renderer's clear/background color and keyed out later."""
    from PIL import Image

    img = np.asarray(Image.open(os.path.join(ASSETS, "sky", SKIES[kind])).convert("RGB"))
    if kind == "night":
        img = (img.astype(np.float32) * 0.55).astype(np.uint8)
    h = img.shape[0]
    horizon = img[int(h * 0.46): int(h * 0.5)].reshape(-1, 3).mean(0)
    return np.clip(np.round(horizon), 0, 254).astype(np.uint8)


class Sky:
    """Real sky photo (CC0 equirect) composited where the renderer cleared to a key color.

    The key and the fog color are the photo's horizon color, so anti-aliased edges and distant fog blend
    into the sky instead of fringing.
    """

    def __init__(self, viewer, kind: str, sun_dir: np.ndarray | None = None, rtx: bool = False):
        from PIL import Image

        self.tol = 4 if rtx else 0

        img = np.asarray(Image.open(os.path.join(ASSETS, "sky", SKIES[kind])).convert("RGB"))
        if kind == "night":
            img = (img.astype(np.float32) * 0.55).astype(np.uint8)
        self.img = img
        h, w = img.shape[:2]
        horizon = img[int(h * 0.46): int(h * 0.5)].reshape(-1, 3).mean(0)
        haze = img[int(h * 0.44): int(h * 0.48)].reshape(-1, 3).mean(0) * (0.45 if kind == "night" else 1.0)
        self.key = np.clip(np.round(horizon), 0, 254).astype(np.uint8)  # exact clear color in the frame
        self.az0 = 0.0
        if rtx:  # background color comes from the visualizer cfg (sky_key); no GL renderer to configure
            return
        r = viewer.renderer
        r.draw_sky = False
        r.sky_upper = tuple(float(c) / 255.0 for c in self.key)
        r.background_color = r.sky_upper
        r.sky_lower = tuple(float(c) / 255.0 for c in haze)
        # rotate the photo so its brightest spot (sun/moon) sits in the direction of the scene light
        self.az0 = 0.0
        if sun_dir is not None:
            lum = img[: h // 2].astype(np.float32).sum(-1)
            vy, ux = np.unravel_index(np.argmax(lum), lum.shape)
            az_img = (0.5 - ux / w) * 2 * math.pi
            self.az0 = math.atan2(sun_dir[1], sun_dir[0]) - az_img

    def composite(self, viewer, frame: np.ndarray) -> np.ndarray:
        if self.tol:  # RTX: tone mapping shifts the magenta background, so key on hue instead of exact value
            f16 = frame.astype(np.int16)
            mag = np.minimum(f16[..., 0], f16[..., 2]) - f16[..., 1]
            alpha = np.clip((mag - 40) / 70.0, 0.0, 1.0)
            mask = alpha > 0.0
            if mask.any():
                out = self._sky_pixels(viewer, frame, mask)
                a = alpha[..., None]
                spill = frame.copy()
                spill[..., 0] = np.minimum(frame[..., 0], frame[..., 1] + 20)  # remove magenta fringe
                spill[..., 2] = np.minimum(frame[..., 2], frame[..., 1] + 20)
                return (a * out + (1 - a) * np.where(mask[..., None], spill, frame)).astype(np.uint8)
            return frame
        else:
            mask = np.all(frame == self.key[None, None], axis=-1)
        if not mask.any():
            return frame
        return self._sky_pixels(viewer, frame, mask)

    def _sky_pixels(self, viewer, frame, mask):
        """Frame copy with the sky photo looked up along each masked pixel's view ray."""
        hgt, wid = frame.shape[:2]
        cam = viewer.camera
        f = np.array(list(cam.get_front()), float)
        rt = np.array(list(cam.get_right()), float)
        up = np.array(list(cam.get_up()), float)
        ty = math.tan(math.radians(cam.fov) / 2)
        tx = ty * wid / hgt
        vv, uu = np.nonzero(mask)
        x = (2 * (uu + 0.5) / wid - 1) * tx
        y = (1 - 2 * (vv + 0.5) / hgt) * ty
        d = f[None] + x[:, None] * rt[None] + y[:, None] * up[None]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        az = np.arctan2(d[:, 1], d[:, 0]) - self.az0
        el = np.arcsin(np.clip(d[:, 2], -1, 1))
        ih, iw = self.img.shape[:2]
        iu = ((0.5 - az / (2 * math.pi)) % 1.0 * iw).astype(int) % iw
        iv = np.clip(((0.5 - el / math.pi) * ih).astype(int), 0, ih - 1)
        out = frame.copy()
        out[vv, uu] = self.img[iv, iu]
        return out


INTENSITY_SCALE = float(os.environ.get("LAMP_SCALE", 25.0))


def setup_rtx_lights(viewer, look: str, lamps=None, offset=np.zeros(3)):
    """USD lights for the OVRTX viewer (must run before its first frame): HDRI dome + sun/moon + lamp lights."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    st = viewer.stage
    for p in ("/root/_RTXDomeLight", "/root/_RTXDistantLight"):
        if st.GetPrimAtPath(p):
            st.RemovePrim(p)
    dome = UsdLux.DomeLight.Define(st, "/root/ResortSky")
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(os.path.join(ASSETS, "sky", SKIES[look])))
    dome.CreateTextureFormatAttr().Set("latlong")
    dome.GetIntensityAttr().Set({"morning": 900.0, "day": 1000.0, "night": 120.0}[look])
    sun = UsdLux.DistantLight.Define(st, "/root/ResortSun")
    sun.GetIntensityAttr().Set({"morning": 2500.0, "day": 3500.0, "night": 25.0}[look])
    sun.GetColorAttr().Set(Gf.Vec3f(*{"morning": (1.0, 0.85, 0.68), "day": (1.0, 0.97, 0.92),
                                      "night": (0.6, 0.7, 1.0)}[look]))
    sun.GetAngleAttr().Set(0.53)
    xf = UsdGeom.Xform(sun.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*{"morning": (-62.0, 0.0, 70.0), "day": (-28.0, 0.0, 30.0),
                                       "night": (-40.0, 0.0, -20.0)}[look]))
    if look == "night" and lamps is not None:
        for i, (x, y, z, kind) in enumerate(lamps):
            lt = UsdLux.SphereLight.Define(st, f"/root/ResortLamp{i}")
            lt.GetRadiusAttr().Set(0.12 if kind != "ceiling" else 0.3)
            lt.GetIntensityAttr().Set(INTENSITY_SCALE * {"pole": 22000.0, "bollard": 3500.0, "ceiling": 6000.0,
                                                         "wall": 4000.0}[kind])
            lt.GetColorAttr().Set(Gf.Vec3f(1.0, 0.78, 0.5))
            lt.CreateTreatAsPointAttr().Set(False)
            # just outside the fixture mesh, otherwise the lamp head blocks its own light
            p = np.array([x, y, z + {"pole": -0.2, "bollard": 0.25, "ceiling": -0.15, "wall": 0.0}[kind]]) + offset
            if kind == "wall":
                p[1] -= 0.3
            UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*p.tolist()))
