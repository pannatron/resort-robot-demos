"""Night-time resort mock-up for the wheel-legged Dobot patrol demo.

Pure geometry, no simulator imports: the patrol script turns :func:`build` into
(1) one static collision mesh (the Isaac Lab terrain) and (2) colored visual meshes for the Newton viewer.

Design frame (meters): +x east, +y north, z up. Real-world proportions:
  - main level z = 0: guard booth + entrance plaza, lobby (marble), room corridor (carpet, 3 m wide)
  - lower grounds z = -1.2: pool deck, lawn; reached by stairs of 8 x 15 cm risers, 30 cm treads
  - villa plateau z = -0.9: reached by a 4.3 deg garden slope; beach gate in the south fence
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

L_MAIN, L_LOW, L_VILLA, L_BEACH = 0.0, -1.2, -0.9, -1.45
BOTTOM = -3.0
RISE, TREAD, N_RISERS = 0.15, 0.30, 8

# colors (sRGB 0..1), chosen to read correctly under the dim night lighting
C = dict(
    marble=(0.86, 0.85, 0.80), paver=(0.42, 0.40, 0.37), carpet=(0.50, 0.13, 0.14), tile=(0.80, 0.74, 0.64),
    grass=(0.16, 0.36, 0.15), stone=(0.58, 0.56, 0.52), wall=(0.93, 0.90, 0.84), wood=(0.42, 0.27, 0.15),
    roof=(0.45, 0.22, 0.14), sand=(0.86, 0.78, 0.60), sea=(0.04, 0.16, 0.32), water=(0.15, 0.60, 0.85),
    glass=(0.25, 0.45, 0.55), metal=(0.30, 0.31, 0.33), cushion=(0.92, 0.92, 0.95), lamp=(1.0, 0.86, 0.55),
    glow=(1.0, 0.70, 0.35), dock=(0.95, 0.75, 0.10), hedge=(0.12, 0.30, 0.12), leaf=(0.10, 0.33, 0.13),
    trunk=(0.33, 0.22, 0.13), concrete=(0.62, 0.61, 0.58), gold=(0.85, 0.68, 0.30), sign=(0.95, 0.95, 0.95),
)


@dataclass
class Piece:
    mesh: trimesh.Trimesh
    color: tuple
    physics: bool = True
    opacity: float | None = None
    roughness: float = 0.7
    metallic: float = 0.0
    texture: np.ndarray | None = None
    uvs: np.ndarray | None = None
    mat: str | None = None  # material name (textured in the renderer when a texture is mapped)


@dataclass
class Zone:
    """Axis-aligned floor region: name shown in the HUD, dry friction, locomotion mode."""

    name: str
    rect: tuple  # x0, x1, y0, y1
    mu: float
    mode: str = "wheel"
    wet_able: bool = False


@dataclass
class Waypoint:
    x: float
    y: float
    checkpoint: str | None = None
    speed: float = 1.0
    stop_s: float = 0.0
    look: tuple | None = None  # point to face while stopped


@dataclass
class Scene:
    pieces: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    route: list = field(default_factory=list)
    lamps: list = field(default_factory=list)
    obstacle_slots: list = field(default_factory=list)  # (x, y, yaw, kind) candidate positions
    dock: tuple = (-3.0, 2.6, 0.0)
    dock2: tuple = (-3.0, 0.8, 0.0)
    gate_hinge: tuple = (48.5, -60.0)
    gate_width: float = 3.0
    sign_text_pos: tuple = (62.0, -2.0)
    bounds: tuple = (-12.0, 72.0, -80.0, 16.0)
    # (x0, x1, y0, y1, v_max): doorways and floor-material changes are crossed slowly
    slow_zones: list = field(default_factory=lambda: [(23.5, 28.5, 1.0, 5.0, 0.35), (4.0, 8.0, 0.5, 5.5, 0.45)])

    def collision_mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate([p.mesh for p in self.pieces if p.physics])

    def zone_at(self, x: float, y: float) -> Zone | None:
        for z in self.zones:  # first match wins
            x0, x1, y0, y1 = z.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return z
        return None


# ---------------------------------------------------------------------------------------------- primitives


def box(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    m.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return m


def obox(cx, cy, cz, sx, sy, sz, yaw=0.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(sx, sy, sz))
    m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    m.apply_translation((cx, cy, cz))
    return m


def cyl(x, y, r, z0, z1, sections=16) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    m.apply_translation((x, y, (z0 + z1) / 2))
    return m


def sphere(x, y, z, r, subdiv=3) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    m.apply_translation((x, y, z))
    return m


def hull(points) -> trimesh.Trimesh:
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def ramp_y(x0, x1, y0, y1, z_at_y0, z_at_y1) -> trimesh.Trimesh:
    """Solid slope between y0 and y1 (top surface linear in y), down to BOTTOM."""
    pts = [(x, y, z) for x in (x0, x1) for y, z in ((y0, z_at_y0), (y1, z_at_y1))]
    pts += [(x, y, BOTTOM) for x in (x0, x1) for y in (y0, y1)]
    return hull(pts)


def quad(x0, x1, y0, y1, z) -> trimesh.Trimesh:
    """Upward-facing rectangle (a floor decal)."""
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float)
    return trimesh.Trimesh(v, np.array([[0, 1, 2], [0, 2, 3]]), process=False)


def slope_quad(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    """Rectangle whose height goes linearly from z0 (at y0) to z1 (at y1); a decal on a slope."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    f = [[0, 1, 2], [0, 2, 3]] if y1 > y0 else [[0, 2, 1], [0, 3, 2]]
    return trimesh.Trimesh(v, np.array(f), process=False)


def disc(x, y, z, r, n=24) -> trimesh.Trimesh:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    v = np.vstack([[x, y, z], np.stack([x + r * np.cos(a), y + r * np.sin(a), np.full(n, z)], 1)])
    f = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    return trimesh.Trimesh(v, np.array(f), process=False)


def stairs(x0, x1, y_top_edge, direction, z_top, n=N_RISERS) -> list[trimesh.Trimesh]:
    """Straight flight descending from ``z_top`` at ``y_top_edge`` along ``direction`` (+1 north, -1 south).

    Returns ``n - 1`` treads; the last riser lands on the lower floor, which the caller provides.
    """
    out = []
    for i in range(1, n):
        ya = y_top_edge + direction * TREAD * (i - 1)
        yb = y_top_edge + direction * TREAD * i
        out.append(box(x0, x1, min(ya, yb), max(ya, yb), BOTTOM, z_top - RISE * i))
    return out


# ---------------------------------------------------------------------------------------------- the resort


def build() -> Scene:
    s = Scene()
    P = s.pieces

    def add(mesh, color, physics=True, **kw):
        mat = color if isinstance(color, str) else None
        P.append(Piece(mesh, C[color] if isinstance(color, str) else color, physics, mat=mat, **kw))

    # --- lower grounds (z = -1.2), with the pool basin cut out -------------------------------------
    px0, px1, py0, py1 = 40.0, 56.0, -17.0, -7.0  # pool: 16 m x 10 m
    for r in [(-12, px0, -34, 1.3), (px1, 72, -34, 1.3), (px0, px1, py1, 1.3), (px0, px1, -34, py0)]:
        add(box(r[0], r[1], r[2], r[3], BOTTOM, L_LOW), "grass")
    add(box(px0, px1, py0, py1, BOTTOM, L_LOW - 1.4), "tile")  # basin floor, 1.4 m deep
    add(quad(px0, px1, py0, py1, L_LOW - 0.25), "water", physics=False, opacity=0.75, roughness=0.1)
    for r in [(px0 - 0.4, px1 + 0.4, py1, py1 + 0.4), (px0 - 0.4, px1 + 0.4, py0 - 0.4, py0),
              (px0 - 0.4, px0, py0, py1), (px1, px1 + 0.4, py0, py1)]:
        add(quad(r[0], r[1], r[2], r[3], L_LOW + 0.004), "stone", physics=False)  # coping
    # pool deck tiles
    for r in [(30, px0 - 0.4, -24, 1.3), (px1 + 0.4, 66, -24, 1.3), (px0 - 0.4, px1 + 0.4, py1 + 0.4, 1.3),
              (px0 - 0.4, px1 + 0.4, -24, py0 - 0.4)]:
        add(quad(r[0], r[1], r[2], r[3], L_LOW + 0.003), "tile", physics=False)

    # --- main level (z = 0): plaza, lobby, corridor, rooms --------------------------------------------
    # The patrol axis (y 1.3-4.6, plaza -> lobby -> corridor) is ONE collision box: a box seam across the
    # path is an internal mesh edge that wheels catch on (it stalled a loaded robot and tripped wet laps).
    add(box(-4, 62, 1.3, 4.6, BOTTOM, L_MAIN), "marble", roughness=0.25)
    add(box(-12, -4, -6, 16, BOTTOM, L_MAIN), "paver")  # entrance plaza, around the axis
    add(box(-4, 6, -6, 1.3, BOTTOM, L_MAIN), "paver")
    add(box(-4, 6, 4.6, 16, BOTTOM, L_MAIN), "paver")
    add(box(6, 26, -6, 1.3, BOTTOM, L_MAIN), "marble", roughness=0.25)  # lobby, around the axis
    add(box(6, 26, 4.6, 12, BOTTOM, L_MAIN), "marble", roughness=0.25)
    add(quad(-4, 6, 1.3, 4.6, L_MAIN + 0.003), "paver", physics=False)  # surface colors along the axis
    add(quad(26, 62, 1.3, 4.6, L_MAIN + 0.003), "carpet", physics=False)
    add(box(26, 62, 4.6, 12, BOTTOM, 3.2), "wall")  # guest-room block
    for k in range(8):  # room doors + number plates + wall lamps
        xd = 28.5 + 4.5 * k
        add(box(xd - 0.5, xd + 0.5, 4.56, 4.6, 0.0, 2.1), "wood", physics=False)
        add(box(xd + 0.6, xd + 0.8, 4.55, 4.6, 1.4, 1.6), "gold", physics=False, metallic=0.8)
        s.lamps.append((xd + 2.25, 4.45, 2.3, "wall"))
    add(box(26, 57, 1.15, 1.3, BOTTOM, 1.0), "wall")  # corridor balustrade
    add(box(60, 62, 1.15, 1.3, BOTTOM, 1.0), "wall")
    add(box(61.85, 62, 1.3, 4.6, L_MAIN, 3.2), "wall")  # corridor end wall
    # lobby walls (0.3 m thick, 4 m high) with doorways on the axis y = 3
    add(box(6, 26, -6, -5.7, 0, 4), "wall")
    add(box(6, 26, 11.7, 12, 0, 4), "wall")
    add(box(6, 6.3, -6, 1.0, 0, 4), "wall")
    add(box(6, 6.3, 5.0, 12, 0, 4), "wall")
    add(box(25.7, 26, -6, 1.3, 0, 4), "wall")
    add(box(25.7, 26, 4.6, 12, 0, 4), "wall")
    # door lintels are visual only: the height scanner casts from above and would read them as a 3 m step
    add(box(6, 6.3, 1.0, 5.0, 3.0, 4), "wall", physics=False)
    add(box(25.7, 26, 1.3, 4.6, 3.0, 4), "wall", physics=False)
    for cx, cy in [(11, -2), (11, 8), (21, -2), (21, 8)]:
        add(box(cx - 0.3, cx + 0.3, cy - 0.3, cy + 0.3, 0, 4), "marble", roughness=0.3)
    add(box(13.5, 19.5, 9.4, 10.2, 0, 1.1), "wood")  # reception counter
    add(box(13.5, 19.5, 9.3, 10.3, 1.1, 1.15), "stone", physics=False)
    for sx in (8.5, 18.5):  # sofas + coffee table
        add(box(sx, sx + 2.4, -5.2, -4.4, 0, 0.45), "cushion")
        add(box(sx, sx + 2.4, -5.6, -5.2, 0, 0.85), "cushion")
        add(box(sx + 0.6, sx + 1.8, -3.7, -3.1, 0, 0.4), "wood")
    for cx, cy in [(7.2, 0.3), (7.2, 5.7), (24.8, 0.6), (24.8, 5.4)]:  # planters by the doors
        add(cyl(cx, cy, 0.35, 0, 0.6), "concrete")
        add(sphere(cx, cy, 1.0, 0.55), "leaf", physics=False)
    for lx in (10.0, 16.0, 22.0):
        s.lamps.append((lx, 3.0, 3.6, "ceiling"))

    # guard booth + charging dock
    add(box(-4.6, -1.4, 4.0, 7.0, 0, 1.0), "wall")
    add(box(-4.6, -1.4, 4.0, 7.0, 1.0, 2.3), "glass", roughness=0.1, metallic=0.3)
    add(box(-4.6, -1.4, 4.0, 7.0, 2.3, 2.6), "wall")
    add(box(-5.0, -1.0, 3.6, 7.4, 2.6, 2.75), "roof", physics=False)
    add(quad(-3.6, -2.4, 2.0, 3.2, 0.004), "dock", physics=False)
    add(quad(-3.6, -2.4, 0.2, 1.4, 0.004), "dock", physics=False)  # second charging dock (2-robot shifts)
    s.lamps.append((-1.0, 3.6, 3.0, "pole"))
    s.lamps.append((3.0, -4.0, 3.0, "pole"))

    # --- stairs ------------------------------------------------------------------------------------
    # corridor -> pool deck: 3 m wide, descending south from the balustrade gap
    for m in stairs(57.0, 60.0, 1.3, -1, L_MAIN):
        add(m, "stone")
    add(box(56.7, 57.0, 1.3 - TREAD * 7, 1.3, BOTTOM, L_MAIN + 0.9), "wall")  # cheek walls / railings
    add(box(60.0, 60.3, 1.3 - TREAD * 7, 1.3, BOTTOM, L_MAIN + 0.9), "wall")
    # lawn -> plaza: 3 m wide, ascending north to the plaza edge at y = -6
    for m in stairs(-1.5, 1.5, -6.0, -1, L_MAIN):
        add(m, "stone")
    add(box(-1.8, -1.5, -6.0 - TREAD * 7, -6.0, BOTTOM, L_MAIN + 0.9), "wall")
    add(box(1.5, 1.8, -6.0 - TREAD * 7, -6.0, BOTTOM, L_MAIN + 0.9), "wall")

    # --- pool deck furniture and the closing-time sign -------------------------------------------------
    for k in range(5):
        lx = 42.5 + 2.6 * k
        for ly, back in ((-4.4, 1), (-19.6, -1)):  # loungers face the pool
            add(box(lx - 0.35, lx + 0.35, ly - 0.95, ly + 0.95, L_LOW, L_LOW + 0.35), "wood")
            add(box(lx - 0.33, lx + 0.33, ly - 0.9, ly + 0.9, L_LOW + 0.35, L_LOW + 0.42), "cushion", physics=False)
            yb = ly + back * 0.75
            add(obox(lx, yb, L_LOW + 0.6, 0.66, 0.08, 0.55), "cushion", physics=False)
    add(box(61.9, 62.0, -2.0, -1.9, L_LOW, L_LOW + 1.3), "metal", physics=False)  # sign post
    s.sign_text_pos = (62.0, -1.95)
    for lx, ly in [(37, -3), (37, -21), (59, -21), (63.5, -12), (48, -2), (48, -22)]:
        s.lamps.append((lx, ly, L_LOW + 3.0, "pole"))

    # --- garden: slope up to the villa plateau, stone path, trees ---------------------------------------
    add(ramp_y(-12, 72, -30, -34, L_LOW, L_VILLA), "grass")
    add(box(-12, 72, -60, -34, BOTTOM, L_VILLA), "grass")
    # stepping-stone path decals (x = 50) and the return path (y = -46, x = 0)
    path = [(55.0, 63.0, -26.0, -21.0), (48.6, 51.4, -34.0, -26.0)]
    for r in path:
        add(quad(r[0], r[1], r[2], r[3], L_LOW + 0.004), "stone", physics=False)
    add(slope_quad(48.6, 51.4, -30, -34, L_LOW + 0.004, L_VILLA + 0.004), "stone", physics=False)
    add(quad(48.6, 51.4, -57.5, -34.0, L_VILLA + 0.004), "stone", physics=False)
    add(quad(-1.4, 51.4, -47.4, -44.6, L_VILLA + 0.004), "stone", physics=False)
    add(quad(-1.4, 1.4, -34.0, -44.6, L_VILLA + 0.004), "stone", physics=False)
    add(quad(-1.4, 1.4, -30.0, -8.1, L_LOW + 0.004), "stone", physics=False)
    add(slope_quad(-1.4, 1.4, -30, -34, L_LOW + 0.004, L_VILLA + 0.004), "stone", physics=False)
    rng = np.random.default_rng(7)
    trees = [(44, -28), (57, -29), (42, -40), (58, -40), (20, -38), (10, -40), (30, -52), (-6, -50),
             (-6, -20), (8, -16), (18, -26), (25, -12), (-7, -36), (12, -54), (64, -36), (36, -30)]
    for tx, ty in trees:
        zb = L_VILLA if ty < -34 else L_LOW if ty > -30 else L_LOW + (L_VILLA - L_LOW) * (-30 - ty) / 4
        h = rng.uniform(3.0, 5.0)
        add(cyl(tx, ty, 0.18, zb, zb + h), "trunk")
        add(sphere(tx, ty, zb + h + 0.6, rng.uniform(1.2, 1.8)), "leaf", physics=False)
    for k in range(10):  # low hedges along the lawn side of the return path
        add(box(2.2, 3.0, -29 + 2.0 * k, -28 + 2.0 * k, L_LOW, L_LOW + 0.7), "hedge")
    for ly in range(-28, -8, 6):
        s.lamps.append((-2.2, ly, L_LOW + 0.8, "bollard"))
    for ly in (-38, -42, -50, -54):
        s.lamps.append((52.2, ly, L_VILLA + 0.8, "bollard"))
    for lx in range(6, 48, 8):
        s.lamps.append((lx, -43.8, L_VILLA + 3.0, "pole"))
    s.lamps.append((52.2, -31.5, L_LOW + 0.9, "bollard"))
    s.lamps.append((-2.2, -38, L_VILLA + 0.8, "bollard"))

    # --- villas -----------------------------------------------------------------------------------
    for vx0 in (14.0, 34.0, 56.0):
        vx1 = vx0 + 10.0
        add(box(vx0, vx1, -56, -48, L_VILLA, L_VILLA + 3.2), "wall")
        add(hull([(vx0 - 0.5, -56.5, L_VILLA + 3.2), (vx1 + 0.5, -56.5, L_VILLA + 3.2),
                  (vx0 - 0.5, -47.5, L_VILLA + 3.2), (vx1 + 0.5, -47.5, L_VILLA + 3.2),
                  ((vx0 + vx1) / 2, -52, L_VILLA + 5.2)]), "roof", physics=False)
        add(box(vx0 + 4.4, vx0 + 5.6, -48.02, -47.98, L_VILLA, L_VILLA + 2.1), "wood", physics=False)
        add(box(vx0 + 1.0, vx0 + 3.4, -48.02, -47.98, L_VILLA + 1.0, L_VILLA + 2.2), "glow", physics=False)
        s.lamps.append((vx0 + 6.2, -47.7, L_VILLA + 2.4, "wall"))

    # --- fence, beach gate, beach ------------------------------------------------------------------
    gx0, gx1 = s.gate_hinge[0], s.gate_hinge[0] + s.gate_width
    add(box(-12, gx0 - 0.15, -60.1, -59.9, L_VILLA, L_VILLA + 1.6), "wall")
    add(box(gx1 + 0.15, 72, -60.1, -59.9, L_VILLA, L_VILLA + 1.6), "wall")
    for gx in (gx0 - 0.15, gx1 + 0.15):
        add(box(gx - 0.2, gx + 0.2, -60.25, -59.75, L_VILLA, L_VILLA + 2.0), "concrete")
    s.lamps.append((gx0 - 0.15, -59.5, L_VILLA + 2.2, "wall"))
    s.lamps.append((gx1 + 0.15, -59.5, L_VILLA + 2.2, "wall"))
    add(box(-12, 72, -80, -60.1, BOTTOM, L_BEACH), "sand")
    # gentle ramp from the beach gate down to the sand (5 deg)
    z_at = lambda y: L_VILLA + (L_BEACH - L_VILLA) * (-59.95 - y) / 6.05  # noqa: E731
    add(ramp_y(48.6, 51.4, -59.95, -60.35, z_at(-59.95), z_at(-60.35)), "stone")
    add(ramp_y(46.0, 54.0, -60.35, -66.0, z_at(-60.35), z_at(-66.0)), "stone")
    # beach props: umbrellas + sunbeds + palms (visual), away from the walking line
    for ux, uy in ((58.0, -72.0), (66.0, -71.0), (40.0, -72.5)):
        add(cyl(ux, uy, 0.04, L_BEACH, L_BEACH + 2.3), "metal", physics=False)
        add(hull([(ux + 1.4 * math.cos(a), uy + 1.4 * math.sin(a), L_BEACH + 2.0) for a in
                  np.linspace(0, 2 * math.pi, 9)[:-1]] + [(ux, uy, L_BEACH + 2.5)]), (0.95, 0.55, 0.2), physics=False)
        add(box(ux - 1.2, ux - 0.5, uy - 1.0, uy + 1.0, L_BEACH, L_BEACH + 0.3), "cushion", physics=False)
        add(box(ux + 0.5, ux + 1.2, uy - 1.0, uy + 1.0, L_BEACH, L_BEACH + 0.3), "cushion", physics=False)
    for px_, py_ in ((44.0, -64.0), (56.5, -63.5), (70.0, -66.0), (34.0, -66.0)):
        add(cyl(px_, py_, 0.16, L_BEACH, L_BEACH + 5.5), "trunk")
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
            add(obox(px_ + 1.1 * math.cos(a), py_ + 1.1 * math.sin(a), L_BEACH + 5.4, 2.2, 0.5, 0.06, a), "leaf",
                physics=False)
    add(quad(-200, 260, -260, -70, L_BEACH - 0.1), "sea", physics=False, roughness=0.15)
    add(quad(-200, 260, 16, 200, L_LOW), "hedge", physics=False)  # dark surroundings
    add(quad(-200, -12, -70, 16, L_LOW), "hedge", physics=False)
    add(quad(72, 260, -70, 16, L_LOW), "hedge", physics=False)

    # --- zones (first match wins): HUD name, dry friction, locomotion mode ---------------------------
    s.zones = [
        Zone("บันไดลงสระ", (56.5, 60.5, -1.1, 1.8), 0.7, "walk"),
        Zone("บันไดขึ้นป้อม", (-2.0, 2.0, -8.5, -5.6), 0.7, "walk"),
        Zone("ป้อม รปภ.", (-12, 6, -6, 16), 0.8),
        Zone("ล็อบบี้", (6, 26, -6, 12), 0.6, wet_able=True),
        Zone("ทางเดินห้องพัก", (26, 62, 1.3, 4.6), 0.9),
        Zone("ลานสระว่ายน้ำ", (30, 66, -24, 1.3), 0.6, wet_able=True),
        Zone("ทางเดินในสวน", (-12, 72, -34, -24), 0.75),
        Zone("ทางเดินในสวน", (-12, 30, -24, -6), 0.75),
        Zone("ประตูชายหาด", (44, 56, -60, -54), 0.8),
        Zone("โซนวิลล่า", (-12, 72, -60, -34), 0.75),
    ]

    # --- patrol route --------------------------------------------------------------------------------
    s.route = [
        Waypoint(-3.0, 2.6, "ป้อม รปภ. (เริ่ม)", speed=0.6),
        Waypoint(1.5, 3.0, speed=0.8),
        Waypoint(6.2, 3.0, speed=0.8),
        Waypoint(16.0, 3.0, "ล็อบบี้", speed=0.9),
        Waypoint(25.8, 3.0, speed=0.8),
        Waypoint(44.0, 3.0, "ทางเดินห้องพัก", speed=1.0),
        Waypoint(57.0, 3.0, speed=0.7),
        Waypoint(58.5, 2.2, speed=0.4),
        Waypoint(58.5, 1.5, speed=0.4),
        Waypoint(58.5, -1.4, speed=0.4),  # down the stairs
        Waypoint(60.5, -4.0, speed=0.7),
        Waypoint(61.0, -12.0, "ลานสระว่ายน้ำ", speed=0.8),
        Waypoint(61.0, -21.0, speed=0.8),
        Waypoint(56.0, -24.5, speed=0.8),
        Waypoint(50.0, -27.5, speed=0.8),
        Waypoint(50.0, -36.0, "ทางเดินในสวน", speed=0.8),
        Waypoint(50.0, -46.0, "โซนวิลล่า", speed=1.0),
        Waypoint(50.0, -56.5, "ประตูชายหาด", speed=0.6, stop_s=3.0, look=(50.0, -60.0)),
        Waypoint(50.0, -50.0, speed=0.6),
        Waypoint(48.5, -46.0, speed=0.8),
        Waypoint(2.0, -46.0, speed=1.0),
        Waypoint(0.0, -43.5, speed=0.8),
        Waypoint(0.0, -30.0, speed=0.9),
        Waypoint(0.0, -9.6, speed=0.8),
        Waypoint(0.0, -8.6, speed=0.4),
        Waypoint(0.0, -5.0, speed=0.4),  # up the stairs
        Waypoint(-1.5, -1.0, speed=0.6),
        Waypoint(-3.0, 2.6, "ป้อม รปภ. (จบรอบ)", speed=0.4),
    ]
    # candidate spots for displaced furniture (x, y, yaw, kind) - on or next to the route
    s.obstacle_slots = [
        (16.0, 3.4, 0.3, "chair"), (12.0, 2.5, 0.0, "planter"), (35.0, 3.3, 0.0, "planter"),
        (50.0, 2.6, 0.0, "planter"), (61.0, -8.0, 0.5, "chair"), (60.6, -16.5, 0.2, "chair"),
        (52.5, -26.8, 0.0, "planter"), (50.3, -41.0, 0.0, "planter"), (25.0, -46.3, 0.0, "planter"),
        (0.4, -20.0, 0.4, "chair"),
    ]
    return s


def sign_texture(width=512, height=340) -> np.ndarray:
    """Pool closing-time sign (Thai + English)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, width - 9, height - 9], outline=(180, 30, 30), width=10)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf", 54)
    fr = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", 40)
    d.text((width / 2, 70), "สระว่ายน้ำปิด", fill=(180, 30, 30), font=fb, anchor="mm")
    d.text((width / 2, 150), "20:00 - 07:00 น.", fill=(30, 30, 30), font=fr, anchor="mm")
    d.text((width / 2, 225), "POOL CLOSED", fill=(180, 30, 30), font=fb, anchor="mm")
    d.text((width / 2, 290), "ห้ามลงสระหลังเวลาปิด", fill=(30, 30, 30), font=fr, anchor="mm")
    return np.asarray(img)
