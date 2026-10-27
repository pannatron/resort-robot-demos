"""Birthday surprise on the pool deck (19:30): the wheel-legged Dobot rolls in with the cake, the guests sing,
the birthday guest blows the candles, then the robot dances to the beat and finishes standing on its rear legs.

Skills: rolling policy (resort env), body-pose dance policy and rear-up policy (party_env_cfg.py tasks). The
dance and rear-up policies see the same proprioception as the rolling policy minus the height scan, plus their
own command (pose or stand flag), so they run inside the resort env by slicing its observation.

Usage (from ~/IsaacLab):
  uv run --extra video python ~/dobot_tools/resort/party.py --out <dir> --dance_policy <model.pt> \
      [--rearup_policy <model.pt>] --checkpoint <wheel model.pt> --num_envs 1 --viz newton [--rtx 1]
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
pa.add_argument("--dance_policy", required=True)
pa.add_argument("--rearup_policy", default="")
pa.add_argument("--cake_kg", type=float, default=1.5)
pa.add_argument("--rtx", type=int, default=0)
PA, REST = pa.parse_known_args()
os.makedirs(PA.out, exist_ok=True)

import incidents  # noqa: E402
import patrol_env_cfg as pec  # noqa: E402
import resort_scene as rs  # noqa: E402
from common import H, W, Hud, SmoothCam, _T, guest_parts, place, wrap, yaw_of  # noqa: E402

POLICY_DT = 0.02
Z0 = rs.L_LOW  # pool deck
BEAT = 60.0 / 120.0  # music_dance tempo (party_audio.DANCE_BPM)
DANCE_LEAD_IN = 0.5  # first kick in music_dance [s]
START = (60.6, -3.8)
STOP = (61.25, -11.0)  # hands the cake over here
SPOT = (60.55, -11.0)  # dance spot
BDAY = (62.45, -11.0)
GUESTS = [  # name, xy, clothes, skin
    ("bday", BDAY, (0.92, 0.45, 0.62), (0.86, 0.68, 0.54)),
    ("g1", (62.25, -12.15), (0.30, 0.55, 0.85), (0.74, 0.55, 0.42)),
    ("g2", (61.25, -12.7), (0.95, 0.80, 0.30), (0.86, 0.68, 0.54)),
    ("g3", (60.1, -12.5), (0.35, 0.70, 0.50), (0.55, 0.38, 0.27)),
    ("g4", (59.35, -11.6), (0.88, 0.88, 0.90), (0.74, 0.55, 0.42)),
]
CAKE_SPECS = [("base", "cylinder", (0.15, 0.09), (0.97, 0.95, 0.92)),
              ("top", "cylinder", (0.105, 0.065), (0.95, 0.55, 0.70)),
              ("board", "cylinder", (0.18, 0.012), (0.85, 0.72, 0.35))]
CAKE_SPECS += [(f"candle{i}", "cylinder", (0.008, 0.07), (0.35, 0.65, 0.95)) for i in range(3)]
CAKE_SPECS += [(f"flame{i}", "sphere", (0.016,), (1.0, 0.72, 0.2)) for i in range(3)]
PARTY_BULB_COLORS = [(255, 80, 120), (255, 200, 60), (80, 200, 255), (140, 255, 120), (200, 120, 255)]

_create_env = play_rsl_rl.create_isaaclab_env


def _create_env_patched(task, env_cfg, args_cli, **kw):
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

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
    env_cfg.terminations.bad_orientation = None  # standing on the rear legs is ~70 deg nose-up
    env_cfg.events.add_base_mass = EventTermCfg(
        func=pec.mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_trunk"),
                "mass_distribution_params": (PA.cake_kg, PA.cake_kg), "operation": "add"},
    )
    return _create_env(task, env_cfg, args_cli, **kw)


play_rsl_rl.create_isaaclab_env = _create_env_patched


def rtx_actor_specs():
    import rtx_actors as ra

    specs = {name: ra.person_specs(clothes, skin) for name, _, clothes, skin in GUESTS}
    specs["cake"] = CAKE_SPECS
    return specs


def load_mlp(path, device):
    """Deterministic actor from an RSL-RL checkpoint (``mlp.*`` Linear layers with ELU in between)."""
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    layers = []
    for j, i in enumerate(ids):
        w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = w, b
        layers.append(lin)
        if j < len(ids) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


# ------------------------------------------------------------------------------------------ guests + cake


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


# ------------------------------------------------------------------------------------------ choreography


def dance_command(tb):
    """Beat time tb (beats since the first kick) -> (vx, vy, wz, height, roll, pitch, move name)."""
    bar = int(tb // 4)
    ph = 2 * math.pi * tb  # one cycle per beat
    h0 = 0.40
    if bar < 2:  # intro: bounce on every beat
        return 0.0, 0.0, None, h0 - 0.05 * (0.5 + 0.5 * math.cos(ph)), 0.0, 0.0, "เด้งตามจังหวะ"
    if bar < 4:  # sway left/right every 2 beats
        return 0.0, 0.0, None, h0 - 0.02, 0.30 * math.sin(ph / 2), 0.0, "ส่ายซ้าย-ขวา"
    if bar < 6:  # spin on the wheels + bounce
        return 0.0, 0.0, 2.2, h0 - 0.04 * (0.5 + 0.5 * math.cos(ph)), 0.0, 0.0, "หมุนตัวบนล้อ"
    if bar < 8:  # side slides with a lean into the slide
        s = 1.0 if math.sin(ph / 2) >= 0 else -1.0
        return 0.0, 0.32 * s, None, h0 - 0.03, -0.18 * s, 0.0, "สไลด์ข้าง"
    if bar < 10:  # nod (head-bang) low
        return 0.0, 0.0, None, 0.33, 0.0, 0.25 * math.sin(ph), "ก้ม-เงยตามบีต"
    if bar < 12:  # spin back the other way + sway
        return 0.0, 0.0, -2.2, h0 - 0.02, 0.2 * math.sin(ph / 2), 0.0, "หมุนกลับ"
    if bar < 14:  # body roll: circle of roll + pitch
        return 0.0, 0.0, None, h0 - 0.03, 0.28 * math.sin(ph / 2), 0.22 * math.cos(ph / 2), "โยกตัวเป็นวง"
    return 0.0, 0.0, None, h0, 0.0, 0.0, "ท่าปิดท้าย"


FACE_DANCE = 0.75  # dance heading [rad]: between the guests (east) and the camera (north)
FINALE_BEAT = 14 * 4  # rear-up from bar 14
REAR_BEATS = 5.0  # beats on the rear legs before coming down
BOW_BEATS = 4.0


def run_party(step, *, dt, args_cli, env_cfg):
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics.newton_manager import NewtonManager

    import newton

    from night_render import NightScene, Sky, setup_night, setup_rtx_lights

    cells = dict(zip(step.__code__.co_freevars, (c.cell_contents for c in step.__closure__)))
    env, wheel_policy = cells["env"], cells["policy"]
    uenv = env.unwrapped
    device = uenv.device
    robot = uenv.scene["robot"]
    dance_policy = load_mlp(PA.dance_policy, device)
    rear_policy = load_mlp(PA.rearup_policy, device) if PA.rearup_policy else None
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
    meshes, bulbs = decor_meshes()
    for i, (m, col) in enumerate(meshes):
        night._log(viewer, f"/party/decor{i}", m, col)
    if PA.rtx:
        setup_rtx_lights(viewer, "night", scene.lamps, offset)
        from pxr import Gf, UsdGeom, UsdLux

        for i, (p, col) in enumerate(bulbs[::2]):  # coloured party light on the deck
            lt = UsdLux.SphereLight.Define(viewer.stage, f"/root/PartyBulb{i}")
            lt.GetRadiusAttr().Set(0.05)
            lt.GetIntensityAttr().Set(incidents_rtx_bulb())
            lt.GetColorAttr().Set(Gf.Vec3f(*(c / 255 for c in col)))
            UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*(p + offset + [0, 0, -0.08]).tolist()))
        sky = Sky(viewer, "night", rtx=True)
    else:
        sky = Sky(viewer, "night", viewer.renderer._sun_direction)
    hud = Hud(scene)
    scam = SmoothCam()

    wheel_term = uenv.action_manager.get_term("wheel_vel")
    WHEEL_SCALE = float(wheel_term._scale) if not torch.is_tensor(wheel_term._scale) else float(
        wheel_term._scale.flatten()[0])
    wheel_term._scale = torch.full((1, wheel_term.action_dim), WHEEL_SCALE, device=device)
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

    vpath = os.path.join(PA.out, "party.mp4")
    writer = imageio.get_writer(vpath, fps=25, quality=8, macro_block_size=8)
    voice = incidents.VoiceTrack()
    clock0 = dt_.datetime(2026, 10, 23, 19, 28, 0)
    events = []

    def post(t, etype, msg):
        events.append({"time": (clock0 + dt_.timedelta(seconds=t)).isoformat(), "type": etype, "message": msg})
        print("[EVENT]", msg, flush=True)

    teleport(*START, -math.pi / 2)
    obs = env.get_observations()
    t, k = 0.0, 0
    vtime = 0.0  # video time [s] (one frame every 2 policy steps)
    phase, phase_t = "arrive", 0.0
    route = [np.array(START), np.array((60.75, -9.4)), np.array(STOP)]
    idx = 1
    cake_on_robot = True
    lit = True
    guest_pose = {n: "idle" for n, *_ in GUESTS}
    guest_pos = {n: np.array([x, y, Z0]) for n, (x, y), *_ in GUESTS}
    guest_hd = {n: 0.0 for n, *_ in GUESTS}
    confetti = Confetti()
    alert, alert_until, alert_color = "ออเดอร์เซอร์ไพรส์วันเกิด: เค้ก + เพลง + โชว์เต้น  •  ลานสระ 19:30", 5.0, (230, 70, 150, 230)
    title, step_txt = "เซอร์ไพรส์วันเกิดที่ลานสระ", "หุ่นแบกเค้กไปหาเจ้าของวันเกิด"
    vx_cmd = 0.0
    pose_s = np.array([0.40, 0.0, 0.0])
    dance_t0 = None
    use = "wheel"
    yaw_s = None
    max_t = 140.0
    last_beat_burst = -1
    e_pos = 0.0
    rear_peak = 0.0
    post(0.0, "order", "แพ็กเกจเซอร์ไพรส์วันเกิด: ส่งเค้ก + เปิดเพลง + โชว์เต้น ที่ลานสระ")
    while t < max_t:
        rp = robot.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)
        q = robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
        p = rp - offset
        xy = p[:2]
        yaw = float(yaw_of(q[None])[0])
        tgt, spd, look = xy, 0.0, None
        vy, wz_dance = 0.0, None
        pose_cmd = np.array([0.40, 0.0, 0.0])
        rear_flag = 0.0
        move = ""

        # ------------------------------------------------------------ story
        if phase == "arrive":
            b = route[idx]
            if np.linalg.norm(xy - b) < 0.3:
                if idx + 1 < len(route):
                    idx += 1
                else:
                    phase, phase_t = "greet", t
                    post(t, "arrived", "ถึงโต๊ะวันเกิด พูดอวยพร")
                    voice.say(vtime + 0.3, "party_arrive_th")
            tgt, spd = route[idx], 0.55 if idx == 1 else 0.35
            for n, *_ in GUESTS:  # everyone turns to the robot
                g = guest_pos[n]
                guest_hd[n] = math.atan2(xy[1] - g[1], xy[0] - g[0])
        elif phase == "greet":
            look = np.array(BDAY)
            if t - phase_t > 1.0:
                guest_pose = {n: ("clap" if n != "bday" else "idle") for n in guest_pose}
            if voice.done_by(vtime) and t - phase_t > 2.0:
                phase, phase_t = "take", t
                guest_pose["bday"] = "hold"
                cake_on_robot = False
                post(t, "cake", "เจ้าของวันเกิดรับเค้กจากหลังหุ่น")
                alert, alert_until = "ร้องเพลงวันเกิด ♪", t + 4
                voice.play(vtime + 0.6, "music_hbd")
                step_txt = "เปิดเพลงวันเกิดจากลำโพงหุ่น"
        elif phase == "take":
            look = np.array(BDAY)
            if t - phase_t > voice.dur["music_hbd"] + 0.4:
                phase, phase_t = "wish", t
                voice.say(vtime + 0.2, "party_wish_th")
                step_txt = "อธิษฐาน + เป่าเทียน"
                guest_pose = {n: ("idle" if n != "bday" else "hold") for n in guest_pose}
        elif phase == "wish":
            look = np.array(BDAY)
            if voice.done_by(vtime) and lit and t - phase_t > 2.0:
                lit = False
                post(t, "candles", "เป่าเทียนแล้ว")
                confetti.burst((BDAY[0] - 0.2, BDAY[1] - 0.4, Z0 + 1.9), n=120, up=2.5, spread=1.0)
                guest_pose = {n: ("cheer" if n != "bday" else "hold") for n in guest_pose}
                phase_t = t
            if not lit and t - phase_t > 2.5:
                phase, phase_t = "back", t
                step_txt = "ถอยไปจุดเต้น"
        elif phase == "back":
            spd = 0.0
            if np.linalg.norm(xy - np.array(SPOT)) > 0.08 and t - phase_t < 4.0:
                d_b = np.array(SPOT) - xy
                vx_back = -0.3 if np.dot(d_b, [math.cos(yaw), math.sin(yaw)]) < 0 else 0.3
                tgt = None
            else:
                tgt = xy
            look = np.array(BDAY)
            if tgt is not None and t - phase_t > 1.0:
                phase, phase_t = "intro", t
                voice.say(vtime + 0.2, "party_dance_th")
                voice.say(vtime + 0.2, "party_dance_en")
                step_txt = "โชว์เต้นฉลอง"
                guest_pose = {n: ("clap" if n != "bday" else "hold") for n in guest_pose}
        elif phase == "intro":
            look = np.array(BDAY)
            if voice.done_by(vtime):
                phase, phase_t = "dance", t
                dance_t0 = vtime + 0.3 + DANCE_LEAD_IN  # first kick
                voice.play(vtime + 0.3, "music_dance")
                post(t, "dance", "เริ่มโชว์เต้น (ท่าเต้นตามจังหวะ 120 BPM)")
        elif phase == "dance":
            tb = (vtime - dance_t0) / BEAT
            if tb < 0:
                tb = 0.0
            if tb < FINALE_BEAT:
                use = "dance"
                vx, vy, wz_dance, h, rl, pt, move = dance_command(tb)
                pose_cmd = np.array([h, rl, pt])
                step_txt = f"โชว์เต้น: {move}"
            elif tb < FINALE_BEAT + REAR_BEATS and rear_policy is not None:
                use = "rear"
                rear_flag = 1.0
                step_txt = "ท่าไม้ตาย: ยืน 2 ขาหลัง"
                if last_beat_burst < 0:
                    last_beat_burst = 1
                    voice.say(vtime + 1.0, "party_finale_th")
                    voice.say(vtime + 1.0, "party_finale_en")
                    post(t, "finale", "ยืน 2 ขาหลัง + โปรยกระดาษสี")
                    guest_pose = {n: ("cheer" if n != "bday" else "hold") for n in guest_pose}
                if tb > FINALE_BEAT + 1.5 and last_beat_burst == 1:
                    last_beat_burst = 2
                    for g in ("g1", "g3", "g4"):
                        gp = guest_pos[g]
                        confetti.burst((gp[0], gp[1], Z0 + 1.8), n=90, up=3.0, spread=1.3)
            elif tb < FINALE_BEAT + REAR_BEATS + 2.0 and rear_policy is not None:
                use = "rear"  # come back down on all four
                rear_flag = 0.0
            elif tb < FINALE_BEAT + REAR_BEATS + 2.0 + BOW_BEATS:
                use = "dance"
                pose_cmd = np.array([0.32, 0.0, 0.32])  # bow
                step_txt = "โค้งขอบคุณ"
                if last_beat_burst < 0:  # no rear-up policy: bow is the finale
                    last_beat_burst = 2
                    voice.say(vtime + 0.3, "party_finale_th")
                    voice.say(vtime + 0.3, "party_finale_en")
                    confetti.burst((SPOT[0], SPOT[1], Z0 + 1.5), n=200)
            else:
                use = "dance"
                if phase != "end":
                    phase, phase_t = "end", t
                    alert, alert_until = "สุขสันต์วันเกิด!  แพ็กเกจปาร์ตี้ = บริการเสริมของรีสอร์ต", t + 8
                    post(t, "done", "จบโชว์ ✓")
        elif phase == "end":
            use = "dance"
            if t - phase_t > 6.0 and voice.done_by(vtime):
                break

        # ------------------------------------------------------------ command
        if use == "wheel":
            if tgt is None:  # straight back-up to the dance spot
                vx = vx_back
                wz = 0.0
            else:
                d_t = tgt - xy
                e_head = wrap(math.atan2(d_t[1], d_t[0]) - yaw) if np.linalg.norm(d_t) > 0.05 else 0.0
                e_use = e_head if (spd > 0.1 or look is None) else wrap(
                    math.atan2(look[1] - xy[1], look[0] - xy[0]) - yaw)
                wz = float(np.clip(2.0 * e_use, -1.0, 1.0))
                vx = spd * max(math.cos(e_head), 0.0) ** 3 if spd > 0 else 0.0
            vx_cmd = float(np.clip(vx, vx_cmd - POLICY_DT, vx_cmd + POLICY_DT))
            cmd = [vx_cmd, 0.0, wz]
        else:
            vx_d = 0.0
            if wz_dance is None:  # hold the dance heading (or face the birthday guest) and the dance spot
                face = FACE_DANCE if phase in ("dance", "end") else math.atan2(BDAY[1] - xy[1], BDAY[0] - xy[0])
                wz_dance = 0.0 if use == "rear" else float(np.clip(1.5 * wrap(face - yaw), -0.8, 0.8))
                if use == "dance" and vy == 0.0 and phase in ("dance", "end"):
                    d_w = np.array(SPOT) - xy
                    c, s_ = math.cos(yaw), math.sin(yaw)
                    vx_d = float(np.clip(0.6 * (c * d_w[0] + s_ * d_w[1]), -0.15, 0.15))
                    vy = float(np.clip(0.6 * (-s_ * d_w[0] + c * d_w[1]), -0.15, 0.15))
            cmd = [vx_d, vy, wz_dance]
        cmd_term.time_left[:] = 1e9
        cmd_term.vel_command_b[:] = torch.tensor([cmd], device=device)
        pose_s += (pose_cmd - pose_s) * 0.35  # a little smoothing; the policy was trained on steps
        with torch.inference_mode():
            prop = obs["policy"][:, :56]
            if use == "dance":
                actions = dance_policy(torch.cat([prop, torch.tensor([pose_s], device=device,
                                                                     dtype=prop.dtype)], 1))
            elif use == "rear":
                actions = rear_policy(torch.cat([prop, torch.tensor([[rear_flag]], device=device,
                                                                    dtype=prop.dtype)], 1))
            else:
                actions = wheel_policy(obs)
            obs, _, dones, _ = env.step(actions)
        if bool(dones[0]):
            post(t, "fall", "หุ่นล้ม")
            break
        tau, qd = robot.data.applied_torque.torch[0], robot.data.joint_vel.torch[0]
        e_pos += float((tau * qd).clamp(min=0).sum()) * POLICY_DT
        g = robot.data.projected_gravity_b.torch[0].cpu().numpy()
        rear_peak = max(rear_peak, math.degrees(math.asin(float(np.clip(-g[0], -1, 1)))))
        confetti.step(POLICY_DT)
        t += POLICY_DT
        k += 1
        if k % 2:
            continue
        vtime += 2 * POLICY_DT

        # ------------------------------------------------------------ actors
        gphase = vtime * 2 * math.pi / BEAT * 0.5
        bday_M = None
        for n, _, clothes, skin in GUESTS:
            gp = guest_pos[n]
            parts = place(guest_parts(guest_pose[n], gphase + hash(n) % 7, clothes, skin), gp[0], gp[1], gp[2],
                          guest_hd[n])
            driver.set(n, parts) if driver is not None else night.log_parts(viewer, n, parts)
            if n == "bday":
                import trimesh

                bday_M = trimesh.transformations.rotation_matrix(guest_hd[n], (0, 0, 1))
                bday_M[:3, 3] = gp
        if cake_on_robot:
            R = quat_rot(q)
            cake_M = _T(R, p + R @ np.array([-0.05, 0.0, 0.12]))
        else:
            cake_M = bday_M @ _T(None, (0.43, 0.0, 1.02))
        cparts = [(tg, m, c, cake_M @ T) for tg, m, c, T in cake_parts(lit)]
        driver.set("cake", cparts) if driver is not None else night.log_parts(viewer, "cake", cparts)

        # ------------------------------------------------------------ camera
        Cw = lambda x, y, z: np.array([x, y, Z0 + z]) + offset  # noqa: E731
        yaw_s = yaw if yaw_s is None else yaw_s + 0.05 * wrap(yaw - yaw_s)
        cut = False
        if phase == "arrive":
            eye, at = Cw(58.2, -5.2, 2.1), 0.5 * rp + 0.5 * Cw(61.4, -11.6, 1.0)
        elif phase in ("greet", "take"):
            eye, at = Cw(59.7, -7.3, 1.75), Cw(61.6, -11.4, 0.95)
            cut = phase == "greet" and abs(t - phase_t) < 0.05
        elif phase == "wish":
            eye, at = Cw(61.0, -8.9, 1.65), Cw(62.3, -11.1, 1.2)
            cut = abs(t - phase_t) < 0.05 and lit
        elif phase in ("back", "intro"):
            eye, at = Cw(60.4, -6.4, 1.9), Cw(61.0, -11.6, 0.85)
        else:
            tb = (vtime - (dance_t0 or vtime)) / BEAT
            if use == "rear" or (phase == "dance" and tb >= FINALE_BEAT):  # low hero angle for the finale
                eye, at = rp + np.array([-0.6, 2.5, 0.0]), rp + np.array([0, 0, 0.3])
                eye[2] = offset[2] + Z0 + 0.85
            elif phase == "dance" and 6 * 4 <= tb < 8 * 4:  # close side angle for the slides
                eye, at = Cw(SPOT[0] - 1.9, SPOT[1] + 1.4, 0.9), rp + np.array([0, 0, 0.1])
            else:
                orbit = 0.25 * math.sin(vtime * 0.15)
                eye = Cw(SPOT[0] + 3.0 * math.sin(orbit), SPOT[1] + 3.0 * math.cos(orbit), 1.35)
                at = 0.8 * rp + 0.2 * Cw(61.3, -12.0, 1.1) + np.array([0, 0, 0.1])
            if phase == "dance" and abs(tb - 6 * 4) < 0.1 or abs(tb - 8 * 4) < 0.1 or abs(tb - FINALE_BEAT) < 0.1:
                cut = True
        viewer.camera.fov = 45.0
        viz.set_camera_view(*scam(eye, at, cut=cut))
        frame = viz.render_rgb_array()
        frame = night.add_glow(viewer, sky.composite(viewer, frame))
        frame = add_party_glow(night, viewer, frame, bulbs, vtime, phase in ("dance", "end"))
        if lit:
            frame = add_flames(night, viewer, frame, flame_points(cake_M), vtime)
        frame = confetti.draw(night, viewer, frame, vtime)
        if vtime > alert_until:
            alert = ""
        st = dict(clock=(clock0 + dt_.timedelta(seconds=t)).strftime("%H:%M:%S"), part="ปาร์ตี้",
                  part_color=(230, 70, 150), title=title, step=step_txt, mode="wheel", switching=False,
                  alert=alert, alert_color=alert_color, subtitle=voice.subtitle(vtime))
        writer.append_data(hud.draw_clean(frame, st))
        if k % 250 == 0:
            print(f"[PARTY] t={t:.1f}s v={vtime:.1f}s phase {phase} use {use} xy {xy.round(2)} "
                  f"pose {pose_s.round(2)} move {move}", flush=True)

    writer.close()
    if voice.events:
        tmp = vpath.replace(".mp4", "_silent.mp4")
        os.replace(vpath, tmp)
        voice.mux(tmp, vpath)
        json.dump([(round(a, 2), kk) for a, kk in voice.events], open(vpath.replace(".mp4", "_voice.json"), "w"))
    res = dict(success=phase == "end", time_s=t, video_s=vtime, energy_pos_Wh=e_pos / 3600,
               rear_up_peak_deg=rear_peak, cake_kg=PA.cake_kg, events=events)
    json.dump(res, open(os.path.join(PA.out, "result_party.json"), "w"), ensure_ascii=False, indent=1)
    print("[RESULT]", json.dumps(res, ensure_ascii=False), flush=True)


def incidents_rtx_bulb():
    from night_render import INTENSITY_SCALE

    return INTENSITY_SCALE * 900.0


play_rsl_rl.run_playback = run_party
if __name__ == "__main__":
    play_rsl_rl.run(REST + ["--task", TASK])
