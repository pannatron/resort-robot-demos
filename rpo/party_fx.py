"""Party props and image-space effects, copied from the Dobot birthday demo (~/dobot_tools/resort/party.py
lines 143-288: cake, fairy lights, balloons, light glow, candle flames, confetti). party.py parses its own
command line at import time, so the helpers live here instead of being imported.
"""

import math

import numpy as np

import resort_scene as rs
from common import _T

Z0 = rs.L_LOW  # pool deck
BEAT = 60.0 / 120.0  # music_dance tempo
PARTY_BULB_COLORS = [(255, 80, 120), (255, 200, 60), (80, 200, 255), (140, 255, 120), (200, 120, 255)]
CAKE_SPECS = [("base", "cylinder", (0.15, 0.09), (0.97, 0.95, 0.92)),
              ("top", "cylinder", (0.105, 0.065), (0.95, 0.55, 0.70)),
              ("board", "cylinder", (0.18, 0.012), (0.85, 0.72, 0.35))]
CAKE_SPECS += [(f"candle{i}", "cylinder", (0.008, 0.07), (0.35, 0.65, 0.95)) for i in range(3)]
CAKE_SPECS += [(f"flame{i}", "sphere", (0.016,), (1.0, 0.72, 0.2)) for i in range(3)]


def cake_parts(lit):
    import trimesh

    def cyl(r, h):
        return trimesh.creation.cylinder(radius=r, height=h, sections=28)

    P = [("board", cyl(0.18, 0.012), (0.85, 0.72, 0.35), _T(None, (0, 0, 0.006))),
         ("base", cyl(0.15, 0.09), (0.97, 0.95, 0.92), _T(None, (0, 0, 0.057))),
         ("top", cyl(0.105, 0.065), (0.95, 0.55, 0.70), _T(None, (0, 0, 0.134)))]
    for i in range(3):
        a = 2 * math.pi * i / 3
        cx, cy = 0.05 * math.cos(a), 0.05 * math.sin(a)
        P.append((f"candle{i}", cyl(0.008, 0.07), (0.35, 0.65, 0.95), _T(None, (cx, cy, 0.20))))
        fz = 0.252 if lit else -40.0  # blown out: parked far below
        P.append((f"flame{i}", trimesh.creation.icosphere(subdivisions=1, radius=0.016), (1.0, 0.72, 0.2),
                  _T(None, (cx, cy, fz))))
    return P


def flame_points(cake_M):
    pts = []
    for i in range(3):
        a = 2 * math.pi * i / 3
        pts.append((cake_M @ np.array([0.05 * math.cos(a), 0.05 * math.sin(a), 0.255, 1.0]))[:3])
    return np.array(pts)


def quat_rot(q):  # xyzw -> 3x3
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------------------------------ party decor


def decor_meshes():
    """Static decor (design frame): two posts, fairy-light strings, balloon clusters. Returns (meshes, bulbs)."""
    meshes, bulbs = [], []
    posts = [(58.3, -9.3), (63.5, -9.3)]
    for x, y in posts:
        meshes.append((rs.cyl(x, y, 0.04, Z0, Z0 + 2.6), rs.C["metal"]))
    anchors = [(58.3, -9.3, Z0 + 2.55), (63.5, -9.3, Z0 + 2.55), (63.5, -12.0, Z0 + 2.75), (58.3, -12.6, Z0 + 2.4)]
    meshes.append((rs.cyl(58.3, -12.6, 0.04, Z0, Z0 + 2.45), rs.C["metal"]))
    k = 0
    for (a, b) in ((0, 1), (1, 2), (0, 3), (0, 2)):
        p0, p1 = np.array(anchors[a]), np.array(anchors[b])
        n = int(np.linalg.norm(p1 - p0) / 0.45)
        for i in range(1, n):
            s = i / n
            p = p0 + (p1 - p0) * s
            p[2] -= 0.45 * 4 * s * (1 - s)  # sag
            bulbs.append((p, PARTY_BULB_COLORS[k % len(PARTY_BULB_COLORS)]))
            meshes.append((rs.sphere(p[0], p[1], p[2], 0.035), tuple(c / 255 for c in PARTY_BULB_COLORS[k % 5])))
            k += 1
    rng = np.random.default_rng(5)
    for cx, cy in ((63.15, -12.35), (58.65, -12.95), (63.2, -9.6)):
        for j in range(6):
            bx, by = cx + rng.uniform(-0.25, 0.25), cy + rng.uniform(-0.25, 0.25)
            bz = Z0 + rng.uniform(1.75, 2.35)
            col = [(0.95, 0.3, 0.45), (0.98, 0.82, 0.25), (0.35, 0.65, 0.98), (0.95, 0.95, 0.97),
                   (0.75, 0.45, 0.95)][j % 5]
            meshes.append((rs.sphere(bx, by, bz, 0.16), col))
            meshes.append((rs.cyl(bx, by, 0.004, Z0 + 0.9, bz - 0.15), (0.9, 0.9, 0.9)))
    return meshes, bulbs


def add_party_glow(night, viewer, frame, bulbs, t, beat_on):
    """Coloured glow on the fairy lights; colours chase along the string with the beat."""
    from PIL import Image, ImageDraw, ImageFilter

    h, w = frame.shape[:2]
    pts = np.array([p for p, _ in bulbs]) + night.offset
    u, v, depth, fr = night.project(viewer, pts, w, h)
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(glow)
    shift = int(t / BEAT) if beat_on else 0
    for i in range(len(bulbs)):
        if not fr[i] or depth[i] <= 0.2:
            continue
        col = PARTY_BULB_COLORS[(i + shift) % len(PARTY_BULB_COLORS)]
        r = float(np.clip(70.0 / depth[i], 2.5, 18.0))
        d.ellipse([u[i] - r, v[i] - r, u[i] + r, v[i] + r], fill=col)
    g = np.asarray(glow.filter(ImageFilter.GaussianBlur(6)), np.float32)
    core = np.asarray(glow.filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    out = frame.astype(np.float32) + 0.9 * g + 0.6 * core
    return np.clip(out, 0, 255).astype(np.uint8)


def add_flames(night, viewer, frame, pts, t):
    from PIL import Image, ImageDraw, ImageFilter

    h, w = frame.shape[:2]
    u, v, depth, fr = night.project(viewer, pts + night.offset, w, h)
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(glow)
    for i in range(len(pts)):
        if not fr[i]:
            continue
        r = float(np.clip(40.0 / depth[i], 3.0, 26.0)) * (1.0 + 0.12 * math.sin(t * 23 + i))
        d.ellipse([u[i] - r, v[i] - r * 1.4, u[i] + r, v[i] + r * 0.8], fill=(255, 170, 60))
    g = np.asarray(glow.filter(ImageFilter.GaussianBlur(5)), np.float32)
    return np.clip(frame.astype(np.float32) + g, 0, 255).astype(np.uint8)


class Confetti:
    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.c = np.zeros((0, 3), np.uint8)
        self.rng = np.random.default_rng(11)

    def burst(self, origin, n=160, up=3.5, spread=1.6):
        o = np.asarray(origin, float)
        v = np.stack([self.rng.normal(0, spread, n), self.rng.normal(0, spread, n), self.rng.uniform(1.5, up, n)], 1)
        self.p = np.concatenate([self.p, np.repeat(o[None], n, 0)])
        self.v = np.concatenate([self.v, v])
        cols = np.array(PARTY_BULB_COLORS + [(255, 255, 255)], np.uint8)
        self.c = np.concatenate([self.c, cols[self.rng.integers(0, len(cols), n)]])

    def step(self, dt):
        if not len(self.p):
            return
        self.v[:, 2] -= 9.81 * dt * 0.35
        self.v *= (1 - 1.6 * dt)
        self.v[:, 2] = np.maximum(self.v[:, 2], -0.9)
        self.p += self.v * dt
        keep = self.p[:, 2] > Z0 + 0.01
        self.p, self.v, self.c = self.p[keep], self.v[keep], self.c[keep]

    def draw(self, night, viewer, frame, t):
        if not len(self.p):
            return frame
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        u, v, depth, fr = night.project(viewer, self.p + night.offset, frame.shape[1], frame.shape[0])
        for i in np.nonzero(fr)[0]:
            s = float(np.clip(30.0 / depth[i], 1.5, 7.0))
            tw = abs(math.sin(t * 9 + i))
            d.rectangle([u[i] - s, v[i] - s * tw, u[i] + s, v[i] + s * tw], fill=tuple(int(x) for x in self.c[i]))
        return np.asarray(img)
