"""Shared pieces of the resort demos: geometry helpers, animated people, and the Thai HUD."""

import math

import numpy as np

import resort_scene as rs

W, H = 1280, 720

# ------------------------------------------------------------------------------------------ helpers


def yaw_of(q):  # xyzw, (N, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _limb(p0, p1, r):
    """Capsule between two points."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = max(np.linalg.norm(d), 1e-4)
    m = trimesh.creation.capsule(height=L, radius=r, count=[10, 10])
    z = np.array([0.0, 0.0, 1.0])
    u = d / L
    ax = np.cross(z, u)
    if np.linalg.norm(ax) > 1e-6:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.acos(np.clip(u @ z, -1, 1)), ax))
    elif u[2] < 0:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
    m.apply_translation((p0 + p1) / 2)
    return m


def _blob(c, radii, subdiv=2):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    m.apply_scale(radii)
    m.apply_translation(c)
    return m


SKIN = [(0.87, 0.70, 0.56), (0.74, 0.55, 0.42), (0.55, 0.38, 0.27)]


def person_mesh(x, y, z, heading, phase, clothes, vest=None, scale=1.0, pants=(0.16, 0.17, 0.22),
                skin=SKIN[1], hair=(0.10, 0.08, 0.07), walking=None):
    """Rounded low-poly person (1.72 m) in the local frame (x forward), with a walk cycle driven by ``phase``.

    Always returns the same number of parts so the meshes can be updated in place every frame.
    """
    import trimesh

    moving = walking if walking is not None else True
    amp = 1.0 if moving else 0.0
    sw = 0.42 * math.sin(phase) * amp
    parts = []
    hip_z, sh_z = 0.93, 1.42
    for side, s in ((1, sw), (-1, -sw)):
        hip = np.array([0.0, side * 0.095, hip_z])
        knee_bend = (0.55 * max(0.0, math.sin(phase + (0 if side > 0 else math.pi) + 1.2)) + 0.08) * amp
        thigh = np.array([math.sin(s), 0.0, -math.cos(s)]) * 0.43
        knee = hip + thigh
        shin_a = s - knee_bend
        ankle = knee + np.array([math.sin(shin_a), 0.0, -math.cos(shin_a)]) * 0.42
        parts.append((_limb(hip, knee, 0.072), pants))
        parts.append((_limb(knee, ankle, 0.058), pants))
        parts.append((_limb(ankle + np.array([-0.03, 0, -0.03]), ankle + np.array([0.13, 0, -0.04]), 0.045),
                      (0.12, 0.11, 0.11)))
        shoulder = np.array([0.0, side * 0.215, sh_z])
        a = -s * 0.9
        elbow = shoulder + np.array([math.sin(a), side * 0.03, -math.cos(a)]) * 0.29
        fa = a + 0.25 * amp + 0.1
        wrist = elbow + np.array([math.sin(fa), 0.0, -math.cos(fa)]) * 0.26
        parts.append((_limb(shoulder, elbow, 0.048), clothes))
        parts.append((_limb(elbow, wrist, 0.04), skin))
        parts.append((_blob(wrist + np.array([0.0, 0.0, -0.05]), (0.04, 0.03, 0.055), 1), skin))
    parts.append((_blob((0.0, 0.0, 0.96), (0.13, 0.19, 0.12)), pants))  # pelvis
    parts.append((_blob((0.0, 0.0, 1.22), (0.135, 0.205, 0.27)), clothes))  # torso
    parts.append((_limb((0, 0, 1.47), (0, 0, 1.55), 0.05), skin))  # neck
    parts.append((_blob((0.01, 0.0, 1.64), (0.10, 0.088, 0.115)), skin))  # head
    parts.append((_blob((-0.015, 0.0, 1.685), (0.098, 0.092, 0.085)), hair))  # hair cap
    parts.append((_blob((0.085, 0.0, 1.63), (0.02, 0.025, 0.02), 1), skin))  # nose
    if vest is not None:  # security guard: hi-vis band + cap with brim
        parts.append((_blob((0.0, 0.0, 1.20), (0.14, 0.212, 0.06)), vest))
        parts.append((_blob((0.0, 0.0, 1.72), (0.105, 0.1, 0.045)), (0.10, 0.12, 0.22)))
        parts.append((_blob((0.09, 0.0, 1.70), (0.07, 0.09, 0.012), 1), (0.10, 0.12, 0.22)))
    else:
        parts.append((_blob((0.0, 0.0, 1.20), (0.001, 0.001, 0.001), 0), clothes))
        parts.append((_blob((0.0, 0.0, 1.20), (0.001, 0.001, 0.001), 0), clothes))
        parts.append((_blob((0.0, 0.0, 1.20), (0.001, 0.001, 0.001), 0), clothes))
    out = []
    R = trimesh.transformations.rotation_matrix(heading, (0, 0, 1))
    for m, c in parts:
        m = m.copy()
        if scale != 1.0:
            m.apply_scale(scale)
        m.apply_transform(R)
        m.apply_translation((x, y, z))
        out.append((m, c))
    return out


class Person:
    def __init__(self, name, path, speed, clothes, vest=None, heading=0.0):
        self.name, self.path, self.speed = name, [np.array(p, float) for p in path], speed
        self.clothes, self.vest = clothes, vest
        self.pos = self.path[0].copy()
        self.seg = 0
        self.walking = False
        self.phase = 0.0
        self.heading = heading

    def start(self, path=None):
        if path is not None:
            self.path = [np.array(p, float) for p in path]
            self.pos = self.path[0].copy()
            self.seg = 0
        self.walking = True

    def update(self, dt):
        if not self.walking or self.seg >= len(self.path) - 1:
            self.walking = False
            return
        tgt = self.path[self.seg + 1]
        d = tgt - self.pos
        n = np.linalg.norm(d[:2])
        step = self.speed * dt
        self.heading = math.atan2(d[1], d[0]) if n > 1e-3 else self.heading
        if n <= step:
            self.pos = tgt.copy()
            self.seg += 1
        else:
            self.pos = self.pos + d * (step / n)
        self.phase += dt * self.speed * 5.0

    @property
    def arrived(self):
        return self.seg >= len(self.path) - 1


# ------------------------------------------------------------------------------------------ HUD

class Hud:
    def __init__(self, scene: rs.Scene):
        from PIL import ImageFont

        T = "/usr/share/fonts/truetype/noto/NotoSansThai-"
        L = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        self._fonts = {}
        for size in (15, 19, 26):
            for bold in (False, True):
                w = "Bold.ttf" if bold else "Regular.ttf"
                lw = "-Bold.ttf" if bold else ".ttf"
                self._fonts[(size, bold)] = (ImageFont.truetype(T + w, size), ImageFont.truetype(L + lw, size - 1))
        self.f_big, self.f, self.f_b, self.f_s = (26, True), (19, False), (19, True), (15, False)
        self.scene = scene
        self._map = self._draw_map()

    def width(self, s, font):
        ft, fl = self._fonts[font]
        return sum((ft if th else fl).getlength(r) for r, th in self._runs(s))

    @staticmethod
    def _runs(s):
        runs, cur, th = [], "", None
        for ch in s:
            is_th = "\u0e00" <= ch <= "\u0e7f"
            if ch == " " and th is not None:
                is_th = th
            if th is None or is_th == th:
                cur += ch
            else:
                runs.append((cur, th))
                cur = ch
            th = is_th
        if cur:
            runs.append((cur, th))
        return runs

    def t(self, d, xy, s, font, fill, anchor="la"):
        """Draw mixed Thai/Latin text (the Thai font has no Latin glyphs)."""
        ft, fl = self._fonts[font]
        x, y = xy
        total = self.width(s, font)
        asc, desc = fl.getmetrics()
        x -= {"l": 0, "m": total / 2, "r": total}[anchor[0]]
        base = y + asc if anchor[1] == "a" else y + (asc - desc) / 2
        for r, th in self._runs(s):
            f = ft if th else fl
            d.text((x, base), r, font=f, fill=fill, anchor="ls")
            x += f.getlength(r)

    def _m2px(self, x, y, mw=250, mh=286):
        x0, x1, y0, y1 = -12, 72, -62, 14
        sc = min(mw / (x1 - x0), mh / (y1 - y0))
        return (x - x0) * sc + 6, (y1 - y) * sc + 6

    def _draw_map(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (262, 298), (10, 14, 24, 200))
        d = ImageDraw.Draw(img)
        fills = [((-12, 72, -60, -34), (30, 70, 35)), ((-12, 72, -34, 1.3), (30, 70, 35)),
                 ((30, 66, -24, 1.3), (150, 140, 120)), ((40, 56, -17, -7), (40, 130, 190)),
                 ((-12, 6, -6, 14), (90, 88, 84)), ((6, 26, -6, 12), (200, 198, 190)),
                 ((26, 62, 1.3, 4.6), (130, 40, 40)), ((26, 62, 4.6, 12), (120, 115, 108))]
        for (x0, x1, y0, y1), c in fills:
            a, b = self._m2px(x0, y1), self._m2px(x1, y0)
            d.rectangle([a, b], fill=c + (255,))
        for vx0 in (14, 34, 56):
            d.rectangle([self._m2px(vx0, -48), self._m2px(vx0 + 10, -56)], fill=(150, 80, 60, 255))
        pts = [self._m2px(w.x, w.y) for w in self.scene.route]
        d.line(pts, fill=(255, 220, 120, 255), width=2)
        return img

    def draw(self, frame, st):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if st.get("cam_label"):
            d.rectangle([0, 0, W, 44], fill=(0, 0, 0, 150))
            d.ellipse([16, 14, 32, 30], fill=(230, 30, 30, 255))
            self.t(d, (42, 8), st["cam_label"], self.f_big, fill=(255, 255, 255, 255))
            self.t(d, (W - 16, 10), st["clock"], self.f_big, fill=(255, 255, 255, 255), anchor="ra")
            for (u0, v0, u1, v1, lab) in st.get("boxes", []):
                d.rectangle([u0, v0, u1, v1], outline=(255, 60, 60, 255), width=4)
                d.rectangle([u0, v0 - 30, u0 + 12 + self.width(lab, self.f_b), v0], fill=(255, 60, 60, 220))
                self.t(d, (u0 + 6, v0 - 26), lab, self.f_b, fill=(255, 255, 255, 255))
        else:
            # status panel
            d.rounded_rectangle([14, 14, 432, 196], 10, fill=(8, 12, 22, 185))
            self.t(d, (28, 22), st.get("header", "DOBOT Wheel-Leg  •  ลาดตระเวนกลางคืน"), self.f_big, fill=(255, 255, 255, 255))
            self.t(d, (28, 60), f"{st['title']}", self.f, fill=(190, 200, 220, 255))
            self.t(d, (28, 88), f"เวลา {st['clock']}     โซน: {st['zone']}", self.f, fill=(255, 255, 255, 255))
            walk = st["mode"] == "walk"
            badge = (240, 160, 40, 255) if walk else (60, 170, 255, 255)
            d.rounded_rectangle([28, 120, 196, 152], 8, fill=badge)
            self.t(d, (112, 136), "โหมดเดิน (ขา)" if walk else "โหมดล้อ", self.f_b, fill=(10, 10, 10, 255),
                   anchor="mm")
            if st.get("switching"):
                self.t(d, (206, 124), "กำลังสลับโหมด…", self.f, fill=(255, 210, 120, 255))
            self.t(d, (28, 160), f"ความเร็ว {st['speed']:.2f} m/s   ระยะ {st['dist']:.0f} m   แบต {st['batt']:.0f}%",
                   self.f, fill=(255, 255, 255, 255))
            # checkpoints
            cps = st["checkpoints"]
            h = 40 + 26 * len(cps)
            d.rounded_rectangle([W - 300, 14, W - 14, 14 + h], 10, fill=(8, 12, 22, 185))
            self.t(d, (W - 286, 22), st.get("list_title", "จุดเช็กอิน"), self.f_b, fill=(255, 255, 255, 255))
            for i, (name, ok) in enumerate(cps):
                y = 50 + 26 * i
                d.ellipse([W - 284, y + 4, W - 268, y + 20], fill=(70, 210, 110, 255) if ok else (90, 90, 100, 255))
                self.t(d, (W - 258, y), name, self.f, fill=(255, 255, 255, 255) if ok else (170, 170, 180, 255))
            # mini-map
            mp = self._map.copy()
            md = ImageDraw.Draw(mp)
            for who, (px, py), c in st.get("people", []):
                u, v = self._m2px(px, py)
                md.ellipse([u - 4, v - 4, u + 4, v + 4], fill=c)
            u, v = self._m2px(*st["pos"])
            yaw = st["yaw"]
            md.polygon([(u + 9 * math.cos(-yaw), v + 9 * math.sin(-yaw)),
                        (u + 6 * math.cos(-yaw + 2.5), v + 6 * math.sin(-yaw + 2.5)),
                        (u + 6 * math.cos(-yaw - 2.5), v + 6 * math.sin(-yaw - 2.5))], fill=(255, 60, 60, 255))
            ov.paste(mp, (W - 276, H - 312), mp)
        # alert banner + event log
        if st.get("alert"):
            txt = st["alert"]
            d.rounded_rectangle([W / 2 - 330, H - 150 if st.get("cam_label") else 210, W / 2 + 330,
                                 (H - 150 if st.get("cam_label") else 210) + 46], 10, fill=st.get("alert_color", (200, 30, 30, 230)))
            self.t(d, (W / 2, (H - 150 if st.get("cam_label") else 210) + 23), txt, self.f_b,
                   fill=(255, 255, 255, 255), anchor="mm")
        log = st.get("events", [])[-4:]
        pip = st.get("pip")
        if pip is not None:
            pw, ph = 448, 252
            pim = Image.fromarray(pip).resize((pw, ph))
            x0, y0 = 14, H - ph - 14
            d.rectangle([x0 - 3, y0 - 33, x0 + pw + 3, y0 + ph + 3], fill=(8, 12, 22, 220))
            d.ellipse([x0 + 6, y0 - 26, x0 + 20, y0 - 12], fill=(230, 30, 30, 255))
            self.t(d, (x0 + 28, y0 - 30), st.get("pip_label", "กล้องหุ่น • REC"), self.f_b, fill=(255, 255, 255, 255))
            ov.paste(pim.convert("RGBA"), (x0, y0))
            log = []
        if log and not st.get("cam_label"):
            d.rounded_rectangle([14, H - 22 - 26 * len(log) - 34, 700, H - 14], 10, fill=(8, 12, 22, 185))
            self.t(d, (28, H - 22 - 26 * len(log) - 28), st.get("log_title", "Event log → ศูนย์ รปภ."), self.f_b,
                   fill=(255, 210, 120, 255))
            for i, e in enumerate(log):
                self.t(d, (28, H - 18 - 26 * (len(log) - i)), e, self.f_s, fill=(235, 235, 235, 255))
        return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))



class SmoothCam:
    """Low-pass filters camera eye/target so shots glide instead of jittering; snaps on big jumps (cuts)."""

    def __init__(self, alpha=0.12, snap=6.0):
        self.alpha, self.snap = alpha, snap
        self.eye = self.tgt = None

    def __call__(self, eye, tgt, cut=False):
        eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
        if cut or self.eye is None or np.linalg.norm(eye - self.eye) > self.snap:
            self.eye, self.tgt = eye.copy(), tgt.copy()
        else:
            self.eye += self.alpha * (eye - self.eye)
            self.tgt += self.alpha * (tgt - self.tgt)
        return tuple(self.eye), tuple(self.tgt)


def _clean_draw(self, frame, st):
    """Minimal product-style overlay: time pill, mode pill, lower-third title, toast, optional PIP."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    acc = st.get("part_color", (60, 170, 255))
    if st.get("cam_label"):  # full-screen robot camera
        d.rectangle([0, 0, W, 50], fill=(0, 0, 0, 140))
        d.ellipse([18, 17, 34, 33], fill=(235, 40, 40, 255))
        self.t(d, (46, 12), st["cam_label"], self.f_big, fill=(255, 255, 255, 255))
        self.t(d, (W - 18, 12), st["clock"], self.f_big, fill=(255, 255, 255, 255), anchor="ra")
        for (u0, v0, u1, v1, lab) in st.get("boxes", []):
            for k, a in ((0, 255), (3, 90)):
                d.rectangle([u0 - k, v0 - k, u1 + k, v1 + k], outline=(255, 70, 70, a), width=3)
            d.rounded_rectangle([u0, v0 - 32, u0 + 16 + self.width(lab, self.f_b), v0 - 4], 6, fill=(235, 60, 60, 230))
            self.t(d, (u0 + 8, v0 - 30), lab, self.f_b, fill=(255, 255, 255, 255))
    else:
        # time + part-of-day pill
        txt = f"{st['clock'][:5]}  •  {st.get('part', '')}"
        wpx = self.width(txt, self.f_b) + 40
        d.rounded_rectangle([24, 22, 24 + wpx, 60], 19, fill=(10, 14, 24, 170))
        d.ellipse([38, 35, 50, 47], fill=tuple(acc) + (255,))
        self.t(d, (58, 28), txt, self.f_b, fill=(255, 255, 255, 255))
        # locomotion mode pill
        walk = st.get("mode") == "walk"
        mtxt = "โหมดเดิน • ขึ้นลงบันได" if walk else "โหมดล้อ"
        if st.get("switching"):
            mtxt = "กำลังสลับโหมด…"
        mw = self.width(mtxt, self.f_b) + 32
        x0 = 36 + wpx
        d.rounded_rectangle([x0, 22, x0 + mw, 60], 19, fill=(240, 160, 40, 225) if walk or st.get("switching")
                            else (40, 120, 220, 200))
        self.t(d, (x0 + 16, 28), mtxt, self.f_b, fill=(255, 255, 255, 255))
        # lower third
        title, step = st.get("title", ""), st.get("step", "")
        tw = max(self.width(title, self.f_big), self.width(step, self.f)) + 60
        y0 = H - 118
        d.rounded_rectangle([24, y0, 24 + tw, y0 + 86], 14, fill=(10, 14, 24, 175))
        d.rounded_rectangle([24, y0, 32, y0 + 86], 4, fill=tuple(acc) + (255,))
        self.t(d, (48, y0 + 10), title, self.f_big, fill=(255, 255, 255, 255))
        self.t(d, (48, y0 + 50), step, self.f, fill=(205, 215, 230, 255))
        if st.get("pip") is None:
            self.t(d, (W - 22, H - 34), "DOBOT Wheel-Leg  •  ภาพจำลอง", self.f_s, fill=(255, 255, 255, 150),
                   anchor="ra")
    # spoken line (robot TTS or guard intercom through the robot's speaker)
    sub = st.get("subtitle")
    if sub:
        who, line = sub
        label = "หุ่นพูด (ลำโพง)" if who == "robot" else "รปภ. ที่ป้อม → ลำโพงหุ่น (สื่อสาร 2 ทาง)"
        lc = (90, 170, 255, 255) if who == "robot" else (255, 205, 80, 255)
        words, lines_, cur = line.split(" "), [], ""
        for wd in words:
            if self.width(cur + " " + wd, self.f) > 900 and cur:
                lines_.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines_.append(cur)
        bw = max(self.width(x, self.f) for x in lines_ + [label]) + 48
        y0 = H - 140 - 34 * len(lines_) - 40
        d.rounded_rectangle([W / 2 - bw / 2, y0, W / 2 + bw / 2, y0 + 40 + 32 * len(lines_)], 12, fill=(8, 10, 18, 205))
        self.t(d, (W / 2 - bw / 2 + 20, y0 + 6), label, self.f_b, fill=lc)
        for i, x in enumerate(lines_):
            self.t(d, (W / 2 - bw / 2 + 20, y0 + 36 + 30 * i), x, self.f, fill=(255, 255, 255, 255))
    # per-robot status cards (shift change scene)
    for i, (nm, batt, status, col) in enumerate(st.get("robots", [])):
        x0, y0 = W - 340, 22 + 74 * i
        d.rounded_rectangle([x0, y0, W - 24, y0 + 64], 12, fill=(10, 14, 24, 185))
        self.t(d, (x0 + 14, y0 + 6), f"{nm}  •  {status}", self.f_b, fill=(255, 255, 255, 255))
        d.rounded_rectangle([x0 + 14, y0 + 40, x0 + 234, y0 + 54], 6, fill=(60, 60, 70, 255))
        d.rounded_rectangle([x0 + 14, y0 + 40, x0 + 14 + 2.2 * max(batt, 2), y0 + 54], 6, fill=tuple(col) + (255,))
        self.t(d, (x0 + 246, y0 + 34), f"{batt:.0f}%", self.f_b, fill=(255, 255, 255, 255))
    # toast
    if st.get("alert"):
        a = st["alert"]
        tw = self.width(a, self.f_b) + 56
        col = st.get("alert_color", (215, 45, 45, 235))
        y = 78 if not st.get("cam_label") else H - 96
        d.rounded_rectangle([W / 2 - tw / 2, y, W / 2 + tw / 2, y + 46], 23, fill=col)
        self.t(d, (W / 2, y + 23), a, self.f_b, fill=(255, 255, 255, 255), anchor="mm")
    pip = st.get("pip")
    if pip is not None:
        from PIL import Image as _I

        pw, ph = 400, 225
        x0, y0 = W - pw - 24, 70  # top-right: the lower corners are where the robot usually is
        d.rounded_rectangle([x0 - 4, y0 - 36, x0 + pw + 4, y0 + ph + 4], 10, fill=(10, 14, 24, 200))
        d.ellipse([x0 + 8, y0 - 27, x0 + 22, y0 - 13], fill=(235, 40, 40, 255))
        self.t(d, (x0 + 30, y0 - 32), st.get("pip_label", "REC"), self.f_b, fill=(255, 255, 255, 255))
        ov.paste(_I.fromarray(pip).resize((pw, ph)).convert("RGBA"), (x0, y0))
    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))


Hud.draw_clean = _clean_draw


# ---------------------------------------------------------------------------------- rigid-part actors
# Renderer-agnostic animation: each body part is a fixed mesh moved by a per-frame transform (instances).
# OVRTX ignores vertex updates of existing meshes but does apply instance transforms.

_CANON = {}


def _canon(key, factory):
    if key not in _CANON:
        _CANON[key] = factory()
    return _CANON[key]


def _T(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M


def _limb_part(p0, p1, r, color, tag):
    """Capsule part: canonical mesh along z (cached per length/radius) + the transform placing it."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    Lq = round(L, 3)
    mesh = _canon(f"limb{Lq}_{r}", lambda: trimesh.creation.capsule(height=max(Lq, 1e-3), radius=r, count=[10, 10]))
    u = d / max(L, 1e-6)
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(z, u)
    if np.linalg.norm(ax) > 1e-6:
        R = trimesh.transformations.rotation_matrix(math.acos(np.clip(u @ z, -1, 1)), ax)[:3, :3]
    else:
        R = np.eye(3) if u[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return (tag, mesh, color, _T(R, (p0 + p1) / 2))


def _blob_part(c, radii, color, tag, subdiv=2):
    import trimesh

    def make():
        m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
        m.apply_scale(radii)
        return m

    return (tag, _canon(f"blob{radii}_{subdiv}", make), color, _T(None, c))


def person_parts(phase, clothes, vest=None, pants=(0.16, 0.17, 0.22), skin=SKIN[1], hair=(0.10, 0.08, 0.07),
                 walking=True):
    """Same figure as :func:`person_mesh`, as (tag, canonical mesh, color, local 4x4) parts with fixed meshes."""
    amp = 1.0 if walking else 0.0
    sw = 0.42 * math.sin(phase) * amp
    P = []
    hip_z, sh_z = 0.93, 1.42
    for side, s, nm in ((1, sw, "l"), (-1, -sw, "r")):
        hip = np.array([0.0, side * 0.095, hip_z])
        knee_bend = (0.55 * max(0.0, math.sin(phase + (0 if side > 0 else math.pi) + 1.2)) + 0.08) * amp
        knee = hip + np.array([math.sin(s), 0.0, -math.cos(s)]) * 0.43
        shin_a = s - knee_bend
        ankle = knee + np.array([math.sin(shin_a), 0.0, -math.cos(shin_a)]) * 0.42
        P.append(_limb_part(hip, knee, 0.072, pants, f"thigh{nm}"))
        P.append(_limb_part(knee, ankle, 0.058, pants, f"shin{nm}"))
        P.append(_limb_part(ankle + np.array([-0.03, 0, -0.03]), ankle + np.array([0.13, 0, -0.04]), 0.045,
                            (0.12, 0.11, 0.11), f"foot{nm}"))
        shoulder = np.array([0.0, side * 0.215, sh_z])
        a = -s * 0.9
        elbow = shoulder + np.array([math.sin(a), side * 0.03, -math.cos(a)]) * 0.29
        fa = a + 0.25 * amp + 0.1
        wrist = elbow + np.array([math.sin(fa), 0.0, -math.cos(fa)]) * 0.26
        P.append(_limb_part(shoulder, elbow, 0.048, clothes, f"uarm{nm}"))
        P.append(_limb_part(elbow, wrist, 0.04, skin, f"farm{nm}"))
        P.append(_blob_part(wrist + np.array([0.0, 0.0, -0.05]), (0.04, 0.03, 0.055), skin, f"hand{nm}", 1))
    P.append(_blob_part((0.0, 0.0, 0.96), (0.13, 0.19, 0.12), pants, "pelvis"))
    P.append(_blob_part((0.0, 0.0, 1.22), (0.135, 0.205, 0.27), clothes, "torso"))
    P.append(_limb_part((0, 0, 1.47), (0, 0, 1.55), 0.05, skin, "neck"))
    P.append(_blob_part((0.01, 0.0, 1.64), (0.10, 0.088, 0.115), skin, "head"))
    P.append(_blob_part((-0.015, 0.0, 1.685), (0.098, 0.092, 0.085), hair, "hair"))
    P.append(_blob_part((0.085, 0.0, 1.63), (0.02, 0.025, 0.02), skin, "nose", 1))
    if vest is not None:
        P.append(_blob_part((0.0, 0.0, 1.20), (0.14, 0.212, 0.06), vest, "vest"))
        P.append(_blob_part((0.0, 0.0, 1.72), (0.105, 0.1, 0.045), (0.10, 0.12, 0.22), "cap"))
        P.append(_blob_part((0.09, 0.0, 1.70), (0.07, 0.09, 0.012), (0.10, 0.12, 0.22), "brim", 1))
    return P


def place(parts, x, y, z, heading):
    """Person-local parts -> world parts (heading about z, then translation)."""
    import trimesh

    W_ = trimesh.transformations.rotation_matrix(heading, (0, 0, 1))
    W_[:3, 3] = (x, y, z)
    return [(tag, mesh, col, W_ @ T) for tag, mesh, col, T in parts]


def guest_parts(pose, phase, clothes, skin):
    """Standing guest with animated arms: 'idle', 'clap', 'cheer' (arms up, waving) or 'hold' (cake in front)."""
    P = [p for p in person_parts(0.0, clothes, skin=skin, walking=False)
         if not p[0].startswith(("uarm", "farm", "hand"))]
    sway = 0.03 * math.sin(phase * 0.5) if pose in ("clap", "cheer") else 0.0
    for side, nm in ((1, "l"), (-1, "r")):
        sh = np.array([0.0, side * 0.215, 1.42])
        if pose == "clap":
            e_dir = np.array([0.45, -side * 0.25, -0.85])
            gap = 0.03 + 0.07 * (0.5 + 0.5 * math.sin(phase * 2.0))
            w_tgt = np.array([0.36, side * gap, 1.18])
        elif pose == "cheer":
            e_dir = np.array([0.1, side * 0.45, 0.88])
            w_tgt = None
            w_dir = np.array([0.08, side * (0.15 + 0.2 * math.sin(phase * 2.2 + side)), 1.0])
        elif pose == "hold":
            e_dir = np.array([0.35, -side * 0.05, -0.94])
            w_tgt = np.array([0.42, side * 0.13, 1.05])
        else:
            e_dir = np.array([0.05, side * 0.08, -1.0])
            w_tgt = None
            w_dir = np.array([0.12, 0.0, -1.0])
        elbow = sh + e_dir / np.linalg.norm(e_dir) * 0.29
        if w_tgt is not None:
            d = w_tgt - elbow
            wrist = elbow + d / np.linalg.norm(d) * 0.26
        else:
            wrist = elbow + w_dir / np.linalg.norm(w_dir) * 0.26
        P.append(_limb_part(sh, elbow, 0.048, clothes, f"uarm{nm}"))
        P.append(_limb_part(elbow, wrist, 0.04, skin, f"farm{nm}"))
        P.append(_blob_part(wrist + (wrist - elbow) / 0.26 * 0.05, (0.04, 0.03, 0.055), skin, f"hand{nm}", 1))
    if sway:
        S = _T(None, (0.0, sway, 0.0))
        P = [(t, m, c, S @ T) for t, m, c, T in P]
    return P
