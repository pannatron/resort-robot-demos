"""Guard tablet control (night, villa zone, 23:55): the guard overrides the patrol from a tablet.

A guest calls about footsteps behind the villas. The guard taps the spot on the tablet map -> "go here": the
robot leaves its patrol route and drives there by itself. The guard then takes manual control with the on-screen
joystick, finds a lost guest on the live camera, lights them up and talks to them through the robot's speaker,
hits STOP when the guest walks towards the robot, and finally sends it back to its patrol.

The tablet UI is drawn on the right of the frame (map, live robot camera, buttons, joystick, finger taps).

Usage (from ~/IsaacLab):
  uv run --extra video python ~/dobot_tools/resort/tablet.py --out <dir> --checkpoint <wheel model.pt> \
      --num_envs 1 --viz newton [--rtx 1]
"""

from __future__ import annotations

import argparse
import datetime as dt_
import json
import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

TASK = "IsaacContrib-Patrol-Resort-DobotQuadWheel"
gym.register(
    id=TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "patrol_env_cfg:ResortPatrolEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.velocity.config.dobot_quad.agents.rsl_rl_ppo_cfg:"
        "DobotQuadWheelRoughPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

pa = argparse.ArgumentParser(add_help=False)
pa.add_argument("--out", required=True)
pa.add_argument("--rtx", type=int, default=0)
pa.add_argument("--feed_every", type=int, default=4, help="render the tablet's robot-camera feed every N frames")
pa.add_argument("--flush", type=int, default=3,
                help="RTX: renders per view after a camera switch (the frame read back is the previous request's, "
                     "and the real-time path tracer accumulates over frames)")
PA, REST = pa.parse_known_args()
os.makedirs(PA.out, exist_ok=True)

import incidents  # noqa: E402
import patrol_env_cfg as pec  # noqa: E402
import resort_scene as rs  # noqa: E402
from common import H, W, Hud, SmoothCam, guest_parts, person_parts, place, wrap, yaw_of  # noqa: E402

POLICY_DT = 0.02
Z0 = rs.L_VILLA
START = (50.0, -37.0)
PATROL = [(50.0, -46.0), (50.0, -56.5)]  # the scheduled route ahead (beach gate check)
TARGET = (28.5, -52.5)  # tapped on the map: between villa 1 and villa 2
PLAN = [(50.0, -46.0), (28.5, -46.0), TARGET]
BACK = [(28.5, -46.0), (2.0, -46.0)]  # rejoin the patrol route towards the garden path
GUEST_CLOTHES, GUEST_SKIN = (0.88, 0.52, 0.22), (0.86, 0.68, 0.54)
GUEST0 = (20.0, -58.0)
GUEST_PATH = [(20.0, -58.0, Z0), (26.3, -58.0, Z0), (26.3, -47.2, Z0), (19.0, -47.2, Z0)]
CLOCK0 = dt_.datetime(2026, 10, 23, 23, 55, 0)

_create_env = play_rsl_rl.create_isaaclab_env


def _create_env_patched(task, env_cfg, args_cli, **kw):
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    if PA.rtx:
        from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

        viz_cfg = NewtonRTXVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                         window_width=W, window_height=H, enable_markers=False, rtx_environment="none",
                                         background_color=(1.0, 0.0, 1.0))
        import rtx_actors as ra

        ra.add_actor_objects(env_cfg, rtx_actor_specs())
    else:
        viz_cfg = NewtonGLVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                        window_width=W, window_height=H, enable_markers=False)
    env_cfg.sim.visualizer_cfgs = [viz_cfg]
    return _create_env(task, env_cfg, args_cli, **kw)


play_rsl_rl.create_isaaclab_env = _create_env_patched


def rtx_actor_specs():
    import rtx_actors as ra

    return {"guest": ra.person_specs(GUEST_CLOTHES, GUEST_SKIN)}


# ------------------------------------------------------------------------------------------ tablet UI

TAB_X, TAB_Y, TAB_W, TAB_H = 764, 16, 500, 688  # bezel, frame pixels
SCR = 14  # bezel width
SW = TAB_W - 2 * SCR  # screen width
MAP_Y, MAP_H = 74, 226
FEED_Y, FEED_H = MAP_Y + MAP_H + 8, 263
BTN_Y, BTN_H = FEED_Y + FEED_H + 10, 56
BUTTONS = [("manual", "บังคับเอง"), ("light", "ไฟส่อง"), ("stop", "หยุด"), ("resume", "ลาดตระเวนต่อ")]
MAP_X0, MAP_X1, MAP_Y0, MAP_Y1 = 4.0, 64.0, -62.0, -33.0
MODE_CHIP = {"auto": ("ลาดตระเวนอัตโนมัติ", (40, 160, 90)), "goto": ("ไปจุดที่สั่ง", (40, 120, 220)),
             "manual": ("บังคับโดย รปภ.", (235, 140, 30)), "stop": ("หยุด (ล็อกล้อ)", (215, 45, 45))}


def m2s(x, y):
    """Design-frame metres -> tablet screen pixels of the map panel."""
    sc = min(SW / (MAP_X1 - MAP_X0), MAP_H / (MAP_Y1 - MAP_Y0))
    return (x - MAP_X0) * sc, MAP_Y + (MAP_Y1 - y) * sc


def btn_rect(i):
    bw = (SW - 3 * 8) / 4
    x0 = i * (bw + 8)
    return x0, BTN_Y, x0 + bw, BTN_Y + BTN_H


def btn_center(key):
    i = [k for k, _ in BUTTONS].index(key)
    x0, y0, x1, y1 = btn_rect(i)
    return (x0 + x1) / 2, (y0 + y1) / 2


class Tablet:
    def __init__(self, hud, scene):
        self.hud, self.scene = hud, scene
        self.base_map = self._map()

    def _map(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (SW, MAP_H), (22, 44, 28, 255))
        d = ImageDraw.Draw(img)

        def R(x0, x1, y0, y1, col):
            a, b = m2s(x0, y1), m2s(x1, y0)
            d.rectangle([a[0], a[1] - MAP_Y, b[0], b[1] - MAP_Y], fill=col)

        R(-12, 72, -34, -30, (30, 62, 36, 255))
        for x0, x1, y0, y1 in ((48.6, 51.4, -57.5, -30), (-1.4, 51.4, -47.4, -44.6), (-1.4, 1.4, -47.4, -30)):
            R(x0, x1, y0, y1, (120, 112, 98, 255))
        for i, vx0 in enumerate((14, 34, 56)):
            R(vx0, vx0 + 10, -56, -48, (150, 82, 62, 255))
            cx, cy = m2s(vx0 + 5, -52)
            self.hud.t(d, (cx, cy - MAP_Y), f"วิลล่า {i + 1}", (15, True), fill=(255, 235, 220, 255), anchor="mm")
        a, b = m2s(0, -60), m2s(48.5, -60)
        d.line([a[0], a[1] - MAP_Y, b[0], b[1] - MAP_Y], fill=(170, 170, 175, 255), width=2)
        a, b = m2s(51.5, -60), m2s(65, -60)
        d.line([a[0], a[1] - MAP_Y, b[0], b[1] - MAP_Y], fill=(170, 170, 175, 255), width=2)
        for tx, ty in ((42, -40), (58, -40), (20, -38), (10, -40), (30, -52), (12, -54), (64, -36)):
            cx, cy = m2s(tx, ty)
            d.ellipse([cx - 5, cy - MAP_Y - 5, cx + 5, cy - MAP_Y + 5], fill=(16, 70, 30, 255))
        pts = [m2s(w.x, w.y) for w in self.scene.route]
        d.line([(x, y - MAP_Y) for x, y in pts], fill=(255, 215, 110, 200), width=2)
        return img

    def draw(self, frame, st):
        """frame: RGB uint8 (H, W, 3) -> frame with the tablet composited on the right."""
        from PIL import Image, ImageDraw

        hud = self.hud
        tab = Image.new("RGBA", (TAB_W, TAB_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(tab)
        d.rounded_rectangle([0, 0, TAB_W - 1, TAB_H - 1], 30, fill=(18, 19, 24, 245), outline=(70, 72, 80, 255), width=2)
        scr = Image.new("RGBA", (SW, TAB_H - 2 * SCR), (14, 17, 26, 255))
        s = ImageDraw.Draw(scr)
        # status bar + header
        hud.t(s, (12, 6), "DOBOT Guard", (15, True), fill=(200, 210, 230, 255))
        hud.t(s, (SW - 12, 6), f"{st['clock']}   แบต {st['batt']:.0f}%", (15, False), fill=(200, 210, 230, 255),
              anchor="ra")
        chip, col = MODE_CHIP[st["mode"]]
        hud.t(s, (12, 36), "หุ่น WL-01", (19, True), fill=(255, 255, 255, 255))
        cw = hud.width(chip, (15, True)) + 26
        s.rounded_rectangle([SW - 12 - cw, 36, SW - 12, 64], 14, fill=col + (255,))
        hud.t(s, (SW - 12 - cw / 2, 50), chip, (15, True), fill=(255, 255, 255, 255), anchor="mm")
        # map
        scr.paste(self.base_map, (0, MAP_Y))
        if st.get("plan"):
            pts = [m2s(*st["robot_xy"])] + [m2s(*p) for p in st["plan"]]
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):  # dashed
                n = max(int(math.hypot(x1 - x0, y1 - y0) / 9), 1)
                for k in range(0, n, 2):
                    a, b = k / n, min((k + 1) / n, 1.0)
                    s.line([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b],
                           fill=(80, 220, 255, 255), width=4)
        if st.get("pin"):
            px, py = m2s(*st["pin"])
            s.ellipse([px - 9, py - 26, px + 9, py - 8], fill=(235, 55, 55, 255))
            s.polygon([(px - 7, py - 13), (px + 7, py - 13), (px, py)], fill=(235, 55, 55, 255))
            s.ellipse([px - 4, py - 21, px + 4, py - 13], fill=(255, 255, 255, 255))
        if st.get("guest_xy") is not None:
            gx, gy = m2s(*st["guest_xy"])
            s.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], fill=(255, 90, 90, 255), outline=(255, 255, 255, 255))
        rx, ry = m2s(*st["robot_xy"])
        c, sn = math.cos(st["robot_yaw"]), -math.sin(st["robot_yaw"])  # screen y points down
        tri = [(rx + 12 * c, ry + 12 * sn), (rx - 7 * c - 7 * sn, ry - 7 * sn + 7 * c), (rx - 7 * c + 7 * sn, ry - 7 * sn - 7 * c)]
        s.ellipse([rx - 15, ry - 15, rx + 15, ry + 15], fill=(80, 220, 255, 60))
        s.polygon(tri, fill=(80, 220, 255, 255), outline=(255, 255, 255, 255))
        if st.get("popup"):
            px, py = m2s(*st["pin"])
            bx0, by0 = px + 14, py - 64
            s.rounded_rectangle([bx0, by0, bx0 + 132, by0 + 50], 10, fill=(250, 250, 252, 250))
            s.rounded_rectangle([bx0 + 8, by0 + 8, bx0 + 124, by0 + 42], 8,
                                fill=(30, 110, 230, 255) if not st.get("popup_pressed") else (20, 80, 170, 255))
            hud.t(s, (bx0 + 66, by0 + 25), "ไปจุดนี้", (19, True), fill=(255, 255, 255, 255), anchor="mm")
        # live robot camera
        feed = st.get("feed")
        if feed is not None:
            scr.paste(Image.fromarray(feed).resize((SW, FEED_H)).convert("RGBA"), (0, FEED_Y))
        s.rectangle([0, FEED_Y, SW, FEED_Y + 30], fill=(0, 0, 0, 120))
        s.ellipse([10, FEED_Y + 9, 22, FEED_Y + 21], fill=(235, 40, 40, 255))
        hud.t(s, (30, FEED_Y + 5), "กล้องหุ่น CAM-1 • สด", (15, True), fill=(255, 255, 255, 255))
        for (u0, v0, u1, v1, lab) in st.get("boxes", []):
            sx, sy = SW / W, FEED_H / H
            b = [u0 * sx, FEED_Y + v0 * sy, u1 * sx, FEED_Y + v1 * sy]
            s.rectangle(b, outline=(255, 70, 70, 255), width=3)
            lw = hud.width(lab, (15, True)) + 12
            s.rounded_rectangle([b[0], b[1] - 22, b[0] + lw, b[1] - 2], 5, fill=(235, 60, 60, 235))
            hud.t(s, (b[0] + 6, b[1] - 21), lab, (15, True), fill=(255, 255, 255, 255))
        if st["mode"] in ("manual", "stop"):  # on-screen joystick over the feed
            cx, cy, r = SW - 72, FEED_Y + FEED_H - 72, 54
            s.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 45), outline=(255, 255, 255, 170), width=2)
            jx, jy = st.get("joy", (0.0, 0.0))
            kx, ky = cx + jx * (r - 18), cy - jy * (r - 18)
            s.ellipse([kx - 20, ky - 20, kx + 20, ky + 20], fill=(255, 255, 255, 215))
            hud.t(s, (cx, cy + r + 2), "จอยสติ๊ก", (15, False), fill=(230, 235, 245, 220), anchor="ma")
        # buttons
        for i, (key, lab) in enumerate(BUTTONS):
            x0, y0, x1, y1 = btn_rect(i)
            on = key in st.get("active", ())
            base = (215, 45, 45) if key == "stop" else (235, 140, 30) if key == "manual" else \
                (240, 200, 70) if key == "light" else (40, 160, 90)
            fill = base + (255,) if on else (38, 44, 58, 255)
            if st.get("pressed") == key:
                fill = tuple(int(c * 0.7) for c in base) + (255,)
            s.rounded_rectangle([x0, y0, x1, y1], 12, fill=fill, outline=base + (255,), width=2)
            hud.t(s, ((x0 + x1) / 2, (y0 + y1) / 2), lab, (15, True), fill=(255, 255, 255, 255), anchor="mm")
        # notification banner
        if st.get("banner"):
            txt, bcol = st["banner"]
            lines = txt.split("\n")
            bh = 14 + 26 * len(lines)
            s.rounded_rectangle([8, MAP_Y + 6, SW - 8, MAP_Y + 6 + bh], 12, fill=bcol + (240,))
            for j, ln in enumerate(lines):
                hud.t(s, (20, MAP_Y + 12 + 26 * j), ln, (19, j == 0), fill=(255, 255, 255, 255))
        # finger
        if st.get("finger") is not None:
            fx, fy, age = st["finger"]
            if age >= 0:  # ripple after the tap
                rr = 14 + 60 * min(age, 0.5)
                s.ellipse([fx - rr, fy - rr, fx + rr, fy + rr], outline=(255, 255, 255, int(220 * (1 - min(age, 0.5) * 2))),
                          width=3)
            s.ellipse([fx - 17, fy - 17, fx + 17, fy + 17], fill=(255, 255, 255, 120), outline=(255, 255, 255, 200))
        tab.paste(scr, (SCR, SCR), scr)
        img = Image.fromarray(frame).convert("RGBA")
        img.alpha_composite(tab, (TAB_X, TAB_Y))
        return np.asarray(img.convert("RGB"))


def draw_subtitle(hud, frame, sub):
    """Spoken line in the left (world) part of the frame, above the lower third."""
    from PIL import Image, ImageDraw

    if not sub:
        return frame
    who, line = sub
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    label = "หุ่นพูด (ลำโพง)" if who == "robot" else "รปภ. พูดผ่านแท็บเล็ต → ลำโพงหุ่น"
    lc = (90, 170, 255, 255) if who == "robot" else (255, 205, 80, 255)
    words, lines_, cur = line.split(" "), [], ""
    for wd in words:
        if hud.width(cur + " " + wd, hud.f) > 640 and cur:
            lines_.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines_.append(cur)
    bw = max(hud.width(x, hud.f) for x in lines_ + [label]) + 40
    cx = 380
    y0 = H - 140 - 32 * len(lines_) - 40
    d.rounded_rectangle([cx - bw / 2, y0, cx + bw / 2, y0 + 40 + 32 * len(lines_)], 12, fill=(8, 10, 18, 205))
    hud.t(d, (cx - bw / 2 + 18, y0 + 6), label, hud.f_b, fill=lc)
    for i, x in enumerate(lines_):
        hud.t(d, (cx - bw / 2 + 18, y0 + 36 + 30 * i), x, hud.f, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))


# ------------------------------------------------------------------------------------------ scene


def run_tablet(step, *, dt, args_cli, env_cfg):
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics.newton_manager import NewtonManager

    import newton

    from common import Person
    from night_render import NightScene, Sky, setup_night, setup_rtx_lights

    cells = dict(zip(step.__code__.co_freevars, (c.cell_contents for c in step.__closure__)))
    env, wheel_policy = cells["env"], cells["policy"]
    uenv = env.unwrapped
    device = uenv.device
    robot = uenv.scene["robot"]
    scene = pec.SCENE
    offset = uenv.scene.env_origins[0].cpu().numpy().astype(np.float64) - np.array([scene.dock[0], scene.dock[1], 0])

    viz = [v for v in SimulationContext.instance().visualizers if hasattr(v, "render_rgb_array")][0]
    viewer = viz._viewer
    model = NewtonManager.get_model()
    if not PA.rtx:
        setup_night(viewer)
    body, stype = model.shape_body.numpy(), model.shape_type.numpy()
    terrain_shapes = np.nonzero((body == -1) & (stype == int(newton.GeoType.MESH)))[0]
    op = model.shape_opacity.numpy()
    op[terrain_shapes] = 0.0
    model.shape_opacity.assign(op)
    flags = model.shape_flags.numpy()
    flags[terrain_shapes] &= ~int(newton.ShapeFlags.VISIBLE)
    model.shape_flags.assign(flags)
    driver = None
    if PA.rtx:
        import rtx_actors as ra

        driver = ra.ActorDriver(uenv, rtx_actor_specs(), offset, model)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    night = NightScene(scene, offset, device=str(device))
    night.log_static(viewer, lamps_on=not PA.rtx)
    if PA.rtx:
        setup_rtx_lights(viewer, "night", scene.lamps, offset)
        sky = Sky(viewer, "night", rtx=True)
    else:
        sky = Sky(viewer, "night", viewer.renderer._sun_direction)
    hud = Hud(scene)
    tablet = Tablet(hud, scene)
    scam = SmoothCam()

    cmd_term = uenv.command_manager.get_term("base_velocity")
    cmd_term.is_heading_env[:] = False
    cmd_term.is_standing_env[:] = False

    def teleport(x, y, hd):
        pose = robot.data.root_link_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(np.array([x, y, night.ground_z(x, y) + 0.5]) + offset, device=device)
        pose[:, 3:] = torch.tensor([0, 0, math.sin(hd / 2), math.cos(hd / 2)], device=device)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device))
        for _ in range(40):
            cmd_term.vel_command_b[:] = 0.0
            with torch.inference_mode():
                env.step(wheel_policy(env.get_observations()))

    import imageio.v2 as imageio

    vpath = os.path.join(PA.out, "tablet.mp4")
    writer = imageio.get_writer(vpath, fps=25, quality=8, macro_block_size=8)
    voice = incidents.VoiceTrack()
    events = []

    def post(t, etype, msg, severity="info"):
        events.append({"time": (CLOCK0 + dt_.timedelta(seconds=t)).isoformat(), "type": etype, "message": msg,
                       "severity": severity})
        print("[EVENT]", msg, flush=True)

    guest = Person("guest", [(*GUEST0, Z0)], 1.0, GUEST_CLOTHES, heading=math.pi / 2)
    teleport(*START, -math.pi / 2)
    obs = env.get_observations()
    t, k, vtime = 0.0, 0, 0.0
    phase, phase_t = "auto", 0.0
    path, pidx = [np.array(p) for p in PATROL], 0
    taps = []  # (time, screen xy, action)
    banner, banner_until = None, -1.0
    pin = popup = None
    popup_pressed = False
    plan_shown = None
    mode = "auto"
    joy = np.zeros(2)
    joy_s = np.zeros(2)
    light = False
    m_step, m_t = 0, 0.0
    guest_seen = False
    guest_pose = "idle"
    vx_cmd = 0.0
    yaw_s = None
    feed, boxes = None, []
    stop_info = {}
    pressed, pressed_until = None, -1.0
    step_txt = "ลาดตระเวนตามเส้นทาง • แท็บเล็ต รปภ. เห็นตำแหน่งและกล้องหุ่นแบบสด"
    max_t = 150.0
    end_t = None

    def tap(at_t, key_or_xy):
        xy = btn_center(key_or_xy) if isinstance(key_or_xy, str) else key_or_xy
        taps.append((at_t, xy))

    def say_banner(txt, col, dur):
        nonlocal banner, banner_until
        banner, banner_until = (txt, col), t + dur

    while t < max_t:
        rp = robot.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)
        q = robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
        p = rp - offset
        xy = p[:2]
        yaw = float(yaw_of(q[None])[0])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        spd, look = 0.0, None
        manual_cmd = None

        # ------------------------------------------------------------ story
        if phase == "auto":
            if pidx < len(path) and np.linalg.norm(xy - path[pidx]) < 0.5:
                pidx += 1
            spd = 0.8
            if t > 2.0 and banner is None:
                say_banner("สายเข้าจากแขก วิลล่า 1\nได้ยินเสียงคนเดินอยู่หลังวิลล่า", (190, 120, 30), 5.5)
                post(t, "guest_call", "แขกวิลล่า 1 โทรแจ้ง: ได้ยินเสียงคนเดินหลังวิลล่า", "warning")
            if t > 4.6 and pin is None:
                pin = TARGET
                tap(t, m2s(*TARGET))
                popup = True
            if t > 6.2 and not popup_pressed:
                popup_pressed = True
                px, py = m2s(*TARGET)
                tap(t, (px + 14 + 66, py - 64 + 25))
            if t > 6.6:
                phase, phase_t, mode = "goto", t, "goto"
                popup, pin = None, TARGET
                path, pidx = [np.array(p) for p in PLAN], 0
                plan_shown = True
                voice.say(vtime + 0.2, "tab_go_th")
                post(t, "guard_goto", "รปภ. แตะแผนที่ \"ไปจุดนี้\" → หุ่นออกจากเส้นทางไปตรวจหลังวิลล่า")
                step_txt = "รปภ. แตะแผนที่ \"ไปจุดนี้\" • หุ่นวางเส้นทางและไปเอง (ข้ามจุดตรวจประตูชายหาดชั่วคราว)"
        elif phase == "goto":
            if pidx < len(path) - 1 and np.linalg.norm(xy - path[pidx]) < 0.6:
                pidx += 1
            last = pidx == len(path) - 1
            d_end = np.linalg.norm(xy - path[-1])
            spd = 1.1 if not last else float(np.clip(0.35 + 0.4 * d_end, 0.35, 1.0))
            corner = np.linalg.norm(xy - path[pidx]) < 1.5 and not last
            spd = 0.6 if corner else spd
            if last and d_end < 0.3:
                phase, phase_t = "arrived", t
                voice.say(vtime + 0.2, "tab_arrive_th")
                post(t, "goto_arrived", "หุ่นถึงจุดที่สั่ง รอคำสั่งต่อ")
                step_txt = "ถึงจุดที่สั่ง • รอคำสั่งจาก รปภ."
                plan_shown = None
        elif phase == "arrived":
            look = xy + fwd
            if voice.done_by(vtime) and t - phase_t > 1.6 and m_step == 0:
                tap(t, "manual")
                pressed, pressed_until = "manual", t + 0.3
                phase, phase_t, mode = "manual", t, "manual"
                m_t = t
                voice.say(vtime + 0.3, "tab_manual_th")
                post(t, "manual_on", "รปภ. กด \"บังคับเอง\" → ควบคุมหุ่นด้วยจอยสติ๊กบนแท็บเล็ต")
                step_txt = "รปภ. กด \"บังคับเอง\" • ขับหุ่นด้วยจอยสติ๊ก ดูภาพจากกล้องหุ่นสด"
        elif phase == "manual":
            ms = t - m_t
            if m_step == 0:  # push forward (south) along the gap to behind the villas
                joy[:] = (0.0, 0.85) if ms > 1.8 else (0.0, 0.0)
                if xy[1] < -56.6:
                    m_step, m_t = 1, t
            elif m_step == 1:  # release, then turn right to look west along the back of villa 1
                joy[:] = (0.0, 0.0) if ms < 0.6 else (0.8, 0.0)
                if ms > 0.6 and abs(wrap(math.pi - yaw)) < 0.07:
                    m_step, m_t = 2, t
            elif m_step == 2:
                joy[:] = 0.0
                if guest_seen and ms > 0.8:
                    tap(t, "light")
                    pressed, pressed_until = "light", t + 0.3
                    light = True
                    post(t, "person_found", "กล้องหุ่นพบบุคคลหลังวิลล่า 1 → รปภ. เปิดไฟส่อง", "warning")
                    m_step, m_t = 3, t
            elif m_step == 3:  # talk through the robot's speaker
                if ms > 1.0 and not any(kk == "guard_lost_th" for _, kk in voice.events):
                    voice.say(vtime, "guard_lost_th")
                    post(t, "intercom", "รปภ. คุยกับแขกผ่านลำโพงหุ่น (แขกหลงทางหาวิลล่า)")
                    step_txt = "รปภ. คุยกับแขกผ่านลำโพงหุ่น • แขกหาห้องพักไม่เจอ"
                    guest.heading = math.atan2(xy[1] - guest.pos[1], xy[0] - guest.pos[0])
                if any(kk == "guard_lost_th" for _, kk in voice.events) and voice.done_by(vtime):
                    guest_pose = "cheer"
                    m_step, m_t = 4, t
            elif m_step == 4:  # guest waves and walks off; the guard follows with the light, then hits STOP
                if ms > 1.4 and not guest.walking and guest.seg == 0:
                    guest_pose = "walk"
                    guest.start(GUEST_PATH)
                joy[:] = (0.0, 0.45) if ms > 2.0 else (0.0, 0.0)
                rel = guest.pos[:2] - xy
                if ms > 2.0 and np.linalg.norm(rel) < 3.0 and np.dot(rel, fwd) > 0:
                    tap(t, "stop")
                    pressed, pressed_until = "stop", t + 0.3
                    mode = "stop"
                    joy[:] = 0.0
                    stop_info = {"t": t, "v0": float(np.linalg.norm(robot.data.root_lin_vel_w.torch[0, :2].cpu().numpy()))}
                    post(t, "estop", "รปภ. กด \"หยุด\" → หุ่นหยุดทันที ให้แขกเดินผ่าน", "warning")
                    say_banner("หยุดแล้ว\nกดหยุดได้ทุกเมื่อ แม้ระหว่างหุ่นทำงานเอง", (200, 45, 45), 4.0)
                    step_txt = "รปภ. กด \"หยุด\" • หุ่นหยุดทันที ให้แขกเดินผ่านก่อน"
                    m_step, m_t = 5, t
            elif m_step == 5:
                joy[:] = 0.0
                if guest.pos[1] > -52.0 and ms > 3.0:
                    tap(t, "resume")
                    pressed, pressed_until = "resume", t + 0.3
                    phase, phase_t, mode = "resume", t, "auto"
                    light = False
                    path, pidx = [np.array(p) for p in BACK], 0
                    plan_shown = True
                    pin = None
                    voice.say(vtime + 0.3, "tab_resume_th")
                    post(t, "resume", "รปภ. กด \"ลาดตระเวนต่อ\" → หุ่นกลับเข้าเส้นทางเอง")
                    step_txt = "รปภ. กด \"ลาดตระเวนต่อ\" • หุ่นกลับเข้าเส้นทางลาดตระเวนเอง"
            # joystick -> velocity command (body frame), smoothed like a real thumb
            joy_s += (joy - joy_s) * 0.15
            manual_cmd = (0.6 * joy_s[1], 0.0, -1.0 * joy_s[0])
            if mode == "stop":
                joy_s[:] = 0.0
                manual_cmd = (0.0, 0.0, 0.0)
                if stop_info and "t_stop" not in stop_info and \
                        np.linalg.norm(robot.data.root_lin_vel_w.torch[0, :2].cpu().numpy()) < 0.05:
                    stop_info["t_stop"] = t
        elif phase == "resume":
            if pidx < len(path) - 1 and np.linalg.norm(xy - path[pidx]) < 0.6:
                pidx += 1
                if pidx == 1:
                    plan_shown = None
                    post(t, "rejoined", "หุ่นกลับเข้าเส้นทางลาดตระเวนแล้ว")
                    end_t = t + 6.0
            spd = 0.6 if np.linalg.norm(xy - path[0]) < 1.5 and pidx == 0 else 0.9
            if end_t is not None and t > end_t and voice.done_by(vtime):
                break

        # people: slow down / give way to the guest
        if manual_cmd is None:
            rel = guest.pos[:2] - xy
            if np.linalg.norm(rel) < 2.2 and np.dot(rel, fwd) > 0.3:
                spd = min(spd, 0.0 if np.linalg.norm(rel) < 1.3 else 0.35)

        # ------------------------------------------------------------ command
        if manual_cmd is not None:
            vx_cmd = manual_cmd[0] if mode != "stop" else 0.0  # STOP bypasses the ramp
            cmd = [vx_cmd, 0.0, manual_cmd[2]]
        else:
            tgt = path[min(pidx, len(path) - 1)]
            d_t = tgt - xy
            e_head = wrap(math.atan2(d_t[1], d_t[0]) - yaw) if np.linalg.norm(d_t) > 0.05 else 0.0
            if look is not None:
                e_head = wrap(math.atan2(look[1] - xy[1], look[0] - xy[0]) - yaw)
                spd = 0.0
            wz = float(np.clip(2.0 * e_head, -1.0, 1.0))
            vx = spd * max(math.cos(e_head), 0.0) ** 3
            vx = min(vx, 0.12) if abs(e_head) > 0.6 else vx
            vx_cmd = float(np.clip(vx, vx_cmd - POLICY_DT, vx_cmd + POLICY_DT))
            cmd = [vx_cmd, 0.0, wz]
        cmd_term.time_left[:] = 1e9
        cmd_term.vel_command_b[:] = torch.tensor([cmd], device=device)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(wheel_policy(obs))
        if bool(dones[0]):
            post(t, "fall", "หุ่นล้ม", "warning")
            break
        guest.update(POLICY_DT)
        if guest.walking:
            guest_pose = "walk"
        elif guest_pose == "walk":
            guest_pose = "idle"
            guest.heading = -math.pi / 2  # at the villa door
        t += POLICY_DT
        k += 1
        if k % 2:
            continue
        vtime += 2 * POLICY_DT

        # ------------------------------------------------------------ actors
        if guest_pose == "walk":
            parts = person_parts(guest.phase, GUEST_CLOTHES, skin=GUEST_SKIN, walking=True)
        else:
            parts = guest_parts(guest_pose, vtime * 6.0, GUEST_CLOTHES, GUEST_SKIN)
        parts = place(parts, *guest.pos, guest.heading)
        driver.set("guest", parts) if driver is not None else night.log_parts(viewer, "guest", parts)

        head_w = rp + np.array([0.33 * math.cos(yaw), 0.33 * math.sin(yaw), 0.2])
        gz = night.ground_z(guest.pos[0], guest.pos[1])
        g_ground = np.array([guest.pos[0], guest.pos[1], gz + 0.02]) + offset

        # ------------------------------------------------------------ tablet feed (robot camera)
        fed_now = False
        if feed is None or (k // 2) % PA.feed_every == 0:
            viewer.camera.fov = 62.0
            eye_r = head_w + np.array([0, 0, 0.05])
            viz.set_camera_view(eye_r, eye_r + np.array([math.cos(yaw), math.sin(yaw), -0.08]))
            for _ in range(PA.flush if PA.rtx else 1):
                f = viz.render_rgb_array()
            fed_now = True
            f = night.add_glow(viewer, sky.composite(viewer, f))
            if light:
                f = incidents.add_spotlight(night, viewer, f, head_w, g_ground, strength=0.7)
            boxes = []
            rel = guest.pos[:2] - xy
            gvis = np.linalg.norm(rel) < 14.0 and abs(wrap(math.atan2(rel[1], rel[0]) - yaw)) < math.radians(28) and \
                night.visible(head_w, (np.array([guest.pos[0], guest.pos[1], gz + 1.2]) + offset)[None])[0]
            if gvis:
                cs = np.array([[guest.pos[0] + dx, guest.pos[1] + dy, gz + dz] for dx in (-0.3, 0.3)
                               for dy in (-0.3, 0.3) for dz in (0.0, 1.8)]) + offset
                u, v, _, fr = night.project(viewer, cs, W, H)
                if fr.all() and v.max() - v.min() < 0.9 * H and u.min() > -40 and u.max() < W + 40:
                    boxes = [(u.min(), v.min(), u.max(), v.max(), "บุคคล 0.93")]
                    if phase == "manual" and not guest_seen:
                        guest_seen = True
            feed = f

        # ------------------------------------------------------------ main camera (robot on the left third)
        yaw_s = yaw if yaw_s is None else yaw_s + 0.04 * wrap(yaw - yaw_s)
        dist_c, hgt = (3.6, 1.8) if phase in ("manual",) else (3.4, 1.6)
        ang = yaw_s + math.pi + math.radians(30)
        eye = rp + np.array([dist_c * math.cos(ang), dist_c * math.sin(ang), hgt])
        at = rp + np.array([1.2 * math.cos(yaw_s), 1.2 * math.sin(yaw_s), 0.0])
        if phase == "manual" and guest_seen and m_step in (2, 3, 4):  # behind the robot: robot + lit guest ahead
            fw = np.array([math.cos(yaw_s), math.sin(yaw_s), 0.0])
            lf = np.array([-fw[1], fw[0], 0.0])
            gw = np.array([guest.pos[0], guest.pos[1], guest.pos[2] + 0.8]) + offset
            eye = rp - 3.0 * fw - 1.0 * lf + np.array([0, 0, 2.0])
            at = 0.72 * rp + 0.28 * gw - np.array([0, 0, 0.7])
        dv = at - eye
        right = np.cross(dv, [0, 0, 1.0])
        right /= max(np.linalg.norm(right), 1e-6)
        at = at + right * 1.5
        viewer.camera.fov = 50.0
        e_s, a_s = scam(eye, at, cut=False)
        viz.set_camera_view(e_s, a_s)
        for _ in range(PA.flush if (PA.rtx and fed_now) else 1):
            frame = viz.render_rgb_array()
        frame = night.add_glow(viewer, sky.composite(viewer, frame))
        if light:
            frame = incidents.add_spotlight(night, viewer, frame, head_w, g_ground, strength=0.6)
        if mode in ("manual", "stop"):
            frame = incidents.add_beacon(night, viewer, frame, rp + np.array([0, 0, 0.22]),
                                         (255, 150, 30) if mode == "manual" else (255, 50, 40), int(t * 3) % 2 == 0)
        st = dict(clock=(CLOCK0 + dt_.timedelta(seconds=t)).strftime("%H:%M:%S"), part="กลางคืน",
                  part_color=(70, 120, 230), title="รปภ. สั่งงานหุ่นผ่านแท็บเล็ต", step=step_txt, mode="wheel",
                  switching=False, alert="", subtitle=None)
        frame = hud.draw_clean(frame, st)
        frame = draw_subtitle(hud, frame, voice.subtitle(vtime))
        # finger: visible 0.45 s before each tap, ripple for 0.5 s after
        finger = None
        for tt, sxy in taps:
            if tt - 0.45 <= t <= tt + 0.5:
                finger = (sxy[0], sxy[1], t - tt)
        if pidx < len(path) and plan_shown:
            plan = [tuple(pp) for pp in path[pidx:]]
        else:
            plan = None
        tst = dict(clock=(CLOCK0 + dt_.timedelta(seconds=t)).strftime("%H:%M"), batt=72.0 - t * 0.01, mode=mode,
                   robot_xy=tuple(xy), robot_yaw=yaw, plan=plan, pin=pin, popup=popup, popup_pressed=popup_pressed,
                   feed=feed, boxes=boxes, joy=tuple(joy_s), finger=finger,
                   guest_xy=tuple(guest.pos[:2]) if guest_seen else None,
                   active=tuple(x for x, on in (("manual", mode in ("manual", "stop")), ("light", light),
                                                ("stop", mode == "stop"), ("resume", phase == "resume")) if on),
                   pressed=pressed if t < pressed_until else None,
                   banner=banner if t < banner_until else None)
        frame = tablet.draw(frame, tst)
        writer.append_data(frame)
        if k % 250 == 0:
            print(f"[TABLET] t={t:.1f}s phase {phase} mode {mode} m_step {m_step} xy {xy.round(2)} "
                  f"yaw {math.degrees(yaw):.0f} guest {np.round(guest.pos[:2], 1)} seen {guest_seen}", flush=True)

    writer.close()
    if voice.events:
        tmp = vpath.replace(".mp4", "_silent.mp4")
        os.replace(vpath, tmp)
        voice.mux(tmp, vpath)
        json.dump([(round(a, 2), kk) for a, kk in voice.events], open(vpath.replace(".mp4", "_voice.json"), "w"))
    stop_s = stop_info.get("t_stop", float("nan")) - stop_info.get("t", float("nan"))
    res = dict(success=phase == "resume" and end_t is not None, time_s=t, video_s=vtime,
               estop_v0=stop_info.get("v0"), estop_time_s=None if math.isnan(stop_s) else round(stop_s, 2),
               events=events)
    json.dump(res, open(os.path.join(PA.out, "result_tablet.json"), "w"), ensure_ascii=False, indent=1)
    print("[RESULT]", json.dumps(res, ensure_ascii=False), flush=True)


play_rsl_rl.run_playback = run_tablet
if __name__ == "__main__":
    play_rsl_rl.run(REST + ["--task", TASK])
