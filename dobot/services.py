"""Daytime guest services of the wheel-legged Dobot at the resort (simulation only).

Jobs:
  camera    06:58  guest rents the robot as a camera operator: it follows the guest from the villa to the
                   beach, films them (robot-camera stream saved as the guest's clip), then sends the clip
  delivery  10:42  room-service order: picks up at the lobby counter, carries 3 kg (added to the trunk mass)
                   down the stairs and past the pool to villa V2; the guest comes out and takes the parcel

Usage (from ~/IsaacLab):
  uv run --extra video python ~/dobot_tools/resort/services.py --job camera --out <dir> \
      --checkpoint <wheel model.pt> --walk_policy <walk policy.pt> --num_envs 1 --viz newton
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
pa.add_argument("--job", choices=["camera", "delivery", "breakfast", "guide"], required=True)
pa.add_argument("--out", required=True)
pa.add_argument("--walk_policy", required=True)
pa.add_argument("--payload_kg", type=float, default=3.0)
pa.add_argument("--rtx", type=int, default=0, help="render with Newton's OVRTX path tracer instead of OpenGL")
PA, REST = pa.parse_known_args()
DLV = PA.job in ("delivery", "breakfast")  # route + parcel jobs
CARRY = DLV or PA.job == "guide"
os.makedirs(PA.out, exist_ok=True)

import patrol_env_cfg as pec  # noqa: E402
import resort_scene as rs  # noqa: E402
from common import H, W, Hud, Person, SmoothCam, _blob_part, _T, person_parts, place, wrap, yaw_of  # noqa: E402,F401

POLICY_DT = 0.02
VILLA_DOOR = (39.0, -48.0)

_create_env = play_rsl_rl.create_isaaclab_env


def _create_env_patched(task, env_cfg, args_cli, **kw):
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    if PA.rtx:
        from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

        from night_render import sky_key

        look_ = "morning" if PA.job in ("camera", "breakfast") else "day"
        bg = (1.0, 0.0, 1.0)  # magenta background, chroma-keyed out and replaced by the sky photo
        viz_cfg = NewtonRTXVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                         window_width=W, window_height=H, enable_markers=False, rtx_environment="none",
                                         background_color=bg)
    else:
        viz_cfg = NewtonGLVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                        window_width=W, window_height=H, enable_markers=False)
    env_cfg.sim.visualizer_cfgs = [viz_cfg]
    if PA.rtx:
        import rtx_actors as ra

        actors = {"guest": ra.person_specs(GUEST_CLOTHES, ra.SKIN_LIGHT)}
        if CARRY:
            actors["parcel"] = ra.PARCEL_SPECS
        ra.add_actor_objects(env_cfg, actors)
    if CARRY and PA.payload_kg > 0:
        env_cfg.events.add_base_mass = EventTermCfg(
            func=pec.mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="link_trunk"),
                    "mass_distribution_params": (PA.payload_kg, PA.payload_kg), "operation": "add"},
        )
    return _create_env(task, env_cfg, args_cli, **kw)


play_rsl_rl.create_isaaclab_env = _create_env_patched

GUEST_CLOTHES = {"camera": (0.95, 0.95, 0.92), "guide": (0.85, 0.45, 0.55)}.get(PA.job, (0.30, 0.55, 0.80))


def quat_rot(q):  # xyzw -> 3x3
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def run_job(step, *, dt, args_cli, env_cfg):
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics.newton_manager import NewtonManager

    import newton

    from night_render import NightScene, Sky, setup_day, setup_rtx_lights

    cells = dict(zip(step.__code__.co_freevars, (c.cell_contents for c in step.__closure__)))
    env, wheel_policy = cells["env"], cells["policy"]
    uenv = env.unwrapped
    device = uenv.device
    robot = uenv.scene["robot"]
    walk_policy = torch.jit.load(PA.walk_policy, map_location=device).eval()
    scene = pec.SCENE
    offset = uenv.scene.env_origins[0].cpu().numpy().astype(np.float64) - np.array([scene.dock[0], scene.dock[1], 0])

    viz = [v for v in SimulationContext.instance().visualizers if hasattr(v, "render_rgb_array")][0]
    viewer = viz._viewer
    model = NewtonManager.get_model()
    LOOK = "morning" if PA.job in ("camera", "breakfast") else "day"
    if not PA.rtx:
        setup_day(viewer, morning=LOOK == "morning")
    body, stype = model.shape_body.numpy(), model.shape_type.numpy()
    terrain_shapes = np.nonzero((body == -1) & (stype == int(newton.GeoType.MESH)))[0]
    op = model.shape_opacity.numpy()
    op[terrain_shapes] = 0.0
    model.shape_opacity.assign(op)
    flags = model.shape_flags.numpy()  # not drawn at all (RTX renders a zero-opacity mesh as black)
    flags[terrain_shapes] &= ~int(newton.ShapeFlags.VISIBLE)
    model.shape_flags.assign(flags)
    driver = None
    if PA.rtx:
        import rtx_actors as ra

        actors = {"guest": ra.person_specs(GUEST_CLOTHES, ra.SKIN_LIGHT)}
        if CARRY:
            actors["parcel"] = ra.PARCEL_SPECS
        driver = ra.ActorDriver(uenv, actors, offset, model)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    night = NightScene(scene, offset, device=str(device))
    night.log_static(viewer, lamps_on=False)
    if PA.rtx:  # OVRTX builds its scene on the first frame: lights now, every animated mesh before rendering
        setup_rtx_lights(viewer, LOOK, scene.lamps, offset)

        sky = Sky(viewer, LOOK, rtx=True)
    else:
        sky = Sky(viewer, LOOK, viewer.renderer._sun_direction)
    scam = SmoothCam()
    PART = {"breakfast": ("เช้า", (240, 160, 40)), "delivery": ("กลางวัน", (40, 160, 90)),
            "guide": ("ตอนว่าง", (40, 160, 90)), "camera": ("มีคนเช่า", (170, 80, 200))}[PA.job]
    SHORT = {"breakfast": "ส่งอาหารเช้าถึงวิลล่า", "delivery": "ส่งของถึงวิลล่า", "guide": "ต้อนรับ & นำทางแขก",
             "camera": "Camera Man ให้แขกเช่า"}[PA.job]
    hud = Hud(scene)

    # friction: sand is softer than paving; walk mode only on the stairs
    mu_bind = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
    from newton import ModelFlags

    def set_mu(v):
        mu_bind[:] = v
        NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def zone_walk(xy):
        for z in scene.zones:
            x0, x1, y0, y1 = z.rect
            if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                return z.mode == "walk", z.name
        return False, "ชายหาด" if xy[1] < -60 else "-"

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
        for _ in range(30):
            cmd_term.vel_command_b[:] = 0.0
            with torch.inference_mode():
                env.step(wheel_policy(env.get_observations()))

    import imageio.v2 as imageio

    writer = imageio.get_writer(os.path.join(PA.out, f"{PA.job}.mp4"), fps=25, quality=8, macro_block_size=8)
    guest_writer = None
    events, ev_lines = [], []

    if PA.job == "camera":
        clock0 = dt_.datetime(2026, 10, 23, 6, 58, 0)
        teleport(37.0, -45.3, -0.5)
        guest = Person("guest", [(39.0, -48.3, rs.L_VILLA)], 1.05, GUEST_CLOTHES, heading=-math.pi / 2)
        guest_writer = imageio.get_writer(os.path.join(PA.out, "guest_clip.mp4"), fps=25, quality=8,
                                          macro_block_size=8)
        steps = ["จองผ่าน LINE 07:00–07:30", "รับแขกหน้าวิลล่า V2", "ตามถ่ายลงชายหาด", "ถ่ายมุมหน้าริมทะเล",
                 "ส่งคลิปให้แขก"]
        header = "DOBOT Wheel-Leg  •  Camera Man ให้เช่า"
        title = "บริการเช้า: ถ่ายวิดีโอให้แขก (ชำระต่อชั่วโมง)"
    elif PA.job == "guide":
        clock0 = dt_.datetime(2026, 10, 23, 14, 10, 0)
        teleport(16.0, 7.2, math.pi / 2)
        guest = Person("guest", [(15.2, 8.7, 0.0)], 1.0, GUEST_CLOTHES, heading=-math.pi / 2)
        route = [(16.0, 7.2), (16.0, 3.2), (25.8, 3.0), (46.5, 3.0)]
        SPEED = [0.5, 0.6, 0.7, 0.8]
        ROOM_DOOR = (46.5, 4.6)
        steps = ["แขกเช็กอินที่เคาน์เตอร์", "รับกระเป๋าใบเล็ก", "นำทางไปห้อง 306", "ถึงหน้าห้อง", "กลับจุดรอ/ชาร์จ"]
        header = "DOBOT Wheel-Leg  •  ต้อนรับ & นำทางแขก"
        title = f"ช่วงว่างระหว่างวัน: พาแขกไปห้อง + ถือกระเป๋า {PA.payload_kg:.0f} กก."
    else:
        clock0 = dt_.datetime(2026, 10, 23, 7, 42, 0) if PA.job == "breakfast" else dt_.datetime(2026, 10, 23,
                                                                                                  10, 42, 0)
        teleport(16.0, 8.5, -math.pi / 2)
        guest = Person("guest", [(VILLA_DOOR[0], VILLA_DOOR[1] - 0.3, rs.L_VILLA)], 1.0, GUEST_CLOTHES,
                       heading=math.pi / 2)
        guest.hidden = True
        route = [(16.0, 8.5), (16.0, 3.2), (25.8, 3.0), (44.0, 3.0), (57.0, 3.0), (58.5, 2.2), (58.5, 1.5),
                 (58.5, -1.4), (60.5, -4.0), (61.0, -12.0), (61.0, -21.0), (56.0, -24.5), (50.0, -27.5),
                 (50.0, -36.0), (50.0, -44.0), (43.0, -45.8), (39.0, -46.2)]
        SPEED = [0.5, 0.8, 0.9, 1.0, 0.7, 0.4, 0.4, 0.4, 0.7, 0.9, 0.9, 0.8, 0.8, 0.8, 0.9, 0.7, 0.4]
        order = "#0742" if PA.job == "breakfast" else "#1024"
        steps = [f"ออเดอร์ {order} เข้า", "รับของที่เคาน์เตอร์", "ลงบันได (โหมดเดิน)", "ผ่านลานสระ", "ถึงวิลล่า V2",
                 "แขกรับของ"]
        if PA.job == "breakfast":
            header = "DOBOT Wheel-Leg  •  ส่งอาหารเช้าถึงวิลล่า"
            title = f"ช่วงเช้า: Breakfast Delivery แบก {PA.payload_kg:.0f} กก."
        else:
            header = "DOBOT Wheel-Leg  •  ส่งของถึงห้อง"
            title = f"บริการกลางวัน: Room Service แบกของ {PA.payload_kg:.0f} กก."
    done_steps = [False] * len(steps)

    def clock(t):
        return (clock0 + dt_.timedelta(seconds=t)).strftime("%H:%M:%S")

    def post(t, etype, msg):
        events.append({"time": (clock0 + dt_.timedelta(seconds=t)).isoformat(), "type": etype, "message": msg})
        ev_lines.append(f"{clock(t)}  {msg}")
        print("[EVENT]", msg, flush=True)

    def log_people(t):  # a "hidden" guest waits inside the villa, out of sight
        moving = guest.walking or getattr(guest, "_moved", False)
        parts = place(person_parts(guest.phase, guest.clothes, skin=(0.86, 0.68, 0.54), walking=moving),
                      *guest.pos, guest.heading)
        driver.set("guest", parts) if driver is not None else night.log_parts(viewer, "guest", parts)

    trail = []
    shot = {}
    parcel = {"on_robot": CARRY, "pos": None}

    import trimesh as _tm

    BOX = _tm.creation.box(extents=(0.36, 0.26, 0.20))
    TAG = _tm.creation.box(extents=(0.12, 0.27, 0.006))

    def log_parcel(rp, q):
        if not CARRY:
            return
        if parcel["on_robot"]:
            R = quat_rot(q)
            M = _T(R, rp - offset + R @ np.array([-0.05, 0.0, 0.17]))
        else:
            gx, gy, gz = guest.pos
            M = _tm.transformations.rotation_matrix(guest.heading, (0, 0, 1))
            M[:3, 3] = (gx + 0.28 * math.cos(guest.heading), gy + 0.28 * math.sin(guest.heading), gz + 1.0)
        parts = [("box", BOX, (0.80, 0.62, 0.38), M), ("tag", TAG, (0.15, 0.45, 0.85), M @ _T(None, (0, 0, 0.102)))]
        driver.set("parcel", parts) if driver is not None else night.log_parts(viewer, "parcel", parts)

    log_people(0.0)
    log_parcel(robot.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64),
               robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64))
    obs = env.get_observations()
    t, k = 0.0, 0
    mode, switch_until = 0, -1.0
    vx_cmd = 0.0
    idx = 1
    phase = 0
    phase_t = 0.0
    dist = 0.0
    e_pos = 0.0
    prev = None
    yaw_s = None
    alert, alert_until, alert_color = "", -1.0, (30, 150, 70, 230)
    finish_t = None
    max_t = 420.0 if DLV else 150.0
    done_steps[0] = True
    if PA.job == "breakfast":
        post(0.0, "order", "ออเดอร์ #0742: กาแฟ 2 + ครัวซองต์ 2 + ผลไม้ → วิลล่า V2")
        alert, alert_until = "ออเดอร์อาหารเช้า #0742 → วิลล่า V2  (รับที่เคาน์เตอร์)", 4.0
    elif PA.job == "delivery":
        post(0.0, "order", "ออเดอร์ #1024: ผ้าเช็ดตัว 2 + น้ำ 2 ขวด → วิลล่า V2")
        alert, alert_until = "ออเดอร์ใหม่ #1024 → วิลล่า V2  (รับของที่เคาน์เตอร์)", 4.0
    elif PA.job == "guide":
        post(0.0, "checkin", "แขกเช็กอินห้อง 306 — ขอให้หุ่นนำทาง")
        alert, alert_until = "เช็กอินเรียบร้อย — หุ่นจะพาไปห้อง 306", 4.0
    else:
        post(0.0, "booking", "แขกวิลล่า V2 จอง Camera Man 07:00–07:30")
    while t < max_t:
        rp = robot.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)
        q = robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
        p = rp - offset
        xy = p[:2]
        yaw = float(yaw_of(q[None])[0])
        if prev is not None:
            dist += float(np.linalg.norm(xy - prev))
        prev = xy.copy()
        walk_zone, zname = zone_walk(xy)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        walk_ahead, _ = zone_walk(xy + 0.7 * fwd)
        set_mu(0.6 if xy[1] < -60.0 else 0.8) if k % 25 == 0 else None

        # ---------------------------------------------------------------- task logic -> target point & speed
        look_pt = None
        if PA.job == "guide":
            gpos = guest.pos[:2]
            if not trail or np.linalg.norm(xy - trail[-1]) > 0.1:
                trail.append(xy.copy())
            if phase == 0:
                tgt, spd = xy, 0.0
                look_pt = gpos
                if t > 3.5:
                    done_steps[1] = True
                    post(t, "guide", "รับกระเป๋าแล้ว เริ่มนำทาง")
                    phase = 1
            elif phase == 1:
                a, b = np.array(route[idx - 1]), np.array(route[idx])
                if np.linalg.norm(xy - b) < 0.45:
                    idx += 1
                    if idx >= len(route):
                        phase, phase_t = 2, t
                        done_steps[2] = done_steps[3] = True
                        post(t, "arrived", "ถึงห้อง 306 แล้ว")
                        alert, alert_until = "ถึงห้อง 306 แล้ว ✓  ขอให้พักผ่อนอย่างมีความสุข", t + 6
                    idx = min(idx, len(route) - 1)
                    a, b = np.array(route[idx - 1]), np.array(route[idx])
                seg = b - a
                L = max(np.linalg.norm(seg), 1e-6)
                u = seg / L
                s_ = np.clip(np.dot(xy - a, u), 0, L)
                tgt = a + u * min(s_ + 1.0, L)
                gap = np.linalg.norm(gpos - xy)
                spd = SPEED[idx] * float(np.clip((3.2 - gap) / 1.2, 0.0, 1.0))  # wait for the guest
            elif phase == 2:
                tgt, spd = xy, 0.0
                look_pt = np.array(ROOM_DOOR)
                if t - phase_t > 2.0 and parcel["on_robot"]:
                    parcel["on_robot"] = False
                    post(t, "handover", "ส่งกระเป๋าให้แขกที่หน้าห้อง")
                if t - phase_t > 4.5:
                    phase, phase_t = 3, t
                    finish_t = t
                    done_steps[4] = True
                    post(t, "return", "กลับไปรอที่ล็อบบี้ / ชาร์จ")
                    alert, alert_until = "งานเสร็จ ✓ กลับไปรอแขกคนถัดไปที่ล็อบบี้", t + 6
            else:
                tgt, spd = np.array([34.0, 3.0]), 0.7
                if t - phase_t > 8.0:
                    break
            # guest follows the robot's trail 1.6 m behind; at the room the guest goes to the door
            if phase == 0:
                gt = np.array([15.6, 7.9])
            elif phase == 1:
                acc, gt = 0.0, gpos
                for j in range(len(trail) - 1, 0, -1):
                    acc += np.linalg.norm(trail[j] - trail[j - 1])
                    if acc >= 1.6:
                        gt = trail[j - 1]
                        break
            else:
                gt = np.array([ROOM_DOOR[0] - 0.6, ROOM_DOOR[1] - 0.45])
            dg = gt - gpos
            ng = np.linalg.norm(dg)
            stepg = min(ng, 1.2 * POLICY_DT)
            if ng > 0.05:
                guest.pos[:2] = gpos + dg / ng * stepg
                guest.heading = math.atan2(dg[1], dg[0])
                guest.phase += stepg * 5.0
            elif phase >= 2:
                guest.heading = math.pi / 2
            guest._moved = ng > 0.05
        elif DLV:
            if phase == 0 and t > 3.0:
                done_steps[1] = True
                post(t, "pickup", "รับของจากเคาน์เตอร์แล้ว ออกเดินทาง")
                phase = 1
            if phase == 0:
                tgt, spd = np.array(route[0]), 0.0
            elif phase == 1:
                a, b = np.array(route[idx - 1]), np.array(route[idx])
                if np.linalg.norm(xy - b) < (0.35 if SPEED[idx] <= 0.45 else 0.7):
                    idx += 1
                    if idx >= len(route):
                        phase, phase_t = 2, t
                        done_steps[4] = True
                        post(t, "arrived", "ถึงหน้าวิลล่า V2 แจ้งแขกทาง LINE/โทรศัพท์ห้อง")
                        alert, alert_until = "หุ่นถึงหน้าวิลล่า V2 แล้ว — แจ้งแขกให้ออกมารับของ", t + 5
                        guest.hidden = False
                        guest.start([(VILLA_DOOR[0], VILLA_DOOR[1] - 0.2, rs.L_VILLA),
                                     (VILLA_DOOR[0], -46.95, rs.L_VILLA)])
                    idx = min(idx, len(route) - 1)
                    a, b = np.array(route[idx - 1]), np.array(route[idx])
                seg = b - a
                L = max(np.linalg.norm(seg), 1e-6)
                u = seg / L
                s = np.clip(np.dot(xy - a, u), 0, L)
                tgt = a + u * min(s + (0.6 if mode else 1.0), L)
                spd = SPEED[idx]
                if xy[1] < 1.6 and not done_steps[2] and mode == 1:
                    done_steps[2] = True
                if xy[1] < -6 and not done_steps[3]:
                    done_steps[3] = True
            else:
                tgt, spd = xy, 0.0
                look_pt = np.array(VILLA_DOOR)
                if phase == 2 and guest.arrived:
                    guest.heading = math.atan2(xy[1] - guest.pos[1], xy[0] - guest.pos[0])
                    phase, phase_t = 3, t
                if phase == 3 and t - phase_t > 1.2:
                    parcel["on_robot"] = False
                    done_steps[5] = True
                    finish_t = t
                    post(t, "delivered", f"ส่งสำเร็จ ✓ ใช้เวลา {t / 60:.1f} นาที ระยะ {dist:.0f} ม.")
                    alert, alert_until = f"ส่งสำเร็จ ✓  ใช้เวลา {t / 60:.1f} นาที", t + 30
                    phase, phase_t = 4, t
                if phase == 4 and t - phase_t > 1.0 and guest.arrived:
                    guest.start([(guest.pos[0], guest.pos[1], rs.L_VILLA), (VILLA_DOOR[0], VILLA_DOOR[1] - 0.2,
                                                                            rs.L_VILLA)])
                    phase = 5
                if phase == 5 and t - phase_t > 6.0:
                    break
        else:  # camera job
            if phase == 0 and t > 2.0:
                guest.start([(39.0, -48.3, rs.L_VILLA), (39.0, -47.0, rs.L_VILLA), (40.5, -45.6, rs.L_VILLA),
                             (48.5, -45.6, rs.L_VILLA), (50.0, -48.5, rs.L_VILLA), (50.0, -59.5, rs.L_VILLA),
                             (50.0, -66.0, rs.L_BEACH), (55.0, -69.5, rs.L_BEACH), (61.0, -70.2, rs.L_BEACH)])
                done_steps[1] = True
                post(t, "start", "เริ่มถ่าย: ติดตามแขกจากวิลล่า V2")
                phase = 1
            if phase in (0, 1, 2):
                gh = guest.heading
                gd = np.array([math.cos(gh), math.sin(gh)])
                gn = np.array([-gd[1], gd[0]])
                if phase == 1 and guest.arrived:
                    phase, phase_t = 2, t
                    done_steps[2] = True
                    guest.heading = -math.pi / 2  # looks at the sea
                    post(t, "shot", "ถึงชายหาด: เปลี่ยนเป็นมุมหน้า (front shot)")
                    alert, alert_until = "ถึงชายหาด — หุ่นวนไปถ่ายมุมหน้าแขก", t + 4
                if phase == 1:
                    # follow the guest's own trail 2.4 m behind, so the robot never cuts through walls
                    if not trail or np.linalg.norm(guest.pos[:2] - trail[-1]) > 0.1:
                        trail.append(guest.pos[:2].copy())
                    acc, tgt = 0.0, None
                    for j in range(len(trail) - 1, 0, -1):
                        acc += np.linalg.norm(trail[j] - trail[j - 1])
                        if acc >= 3.0:
                            tgt = trail[j - 1]
                            break
                    if tgt is None:
                        tgt = xy
                else:  # go round the guest's right side, then hold a front three-quarter shot
                    side_pt = guest.pos[:2] + np.array([3.4, -0.9])
                    front_pt = guest.pos[:2] + np.array([1.8, -3.3])
                    if phase_t >= 0 and "front_t" not in shot:
                        tgt = side_pt if np.linalg.norm(xy - side_pt) > 0.5 and not shot.get("side") else front_pt
                        if np.linalg.norm(xy - side_pt) <= 0.5:
                            shot["side"] = True
                        if np.linalg.norm(xy - front_pt) < 0.45:
                            shot["front_t"] = t
                            done_steps[3] = True
                    else:
                        tgt = front_pt
                    if "front_t" in shot and t - shot["front_t"] > 9.0:
                        phase, phase_t = 3, t
                        done_steps[4] = True
                        post(t, "clip_sent", f"ส่งคลิปให้แขกทาง LINE แล้ว (ยาว {int(t - 2.0) // 60}:{int(t - 2.0) % 60:02d} นาที)")
                        alert, alert_until = "จบงาน ✓ ส่งคลิปให้แขกทาง LINE แล้ว", t + 8
                        finish_t = t
                d = np.linalg.norm(tgt - xy)
                spd = float(np.clip(0.9 * (d - 0.2), 0.0, 1.1))
                look_pt = guest.pos[:2] if d < 0.6 or "front_t" in shot else None
            else:
                tgt, spd = xy, 0.0
                look_pt = guest.pos[:2]
                if t - phase_t > 6.0:
                    break

        # ---------------------------------------------------------------- mode switching + velocity command
        want = int(walk_zone or walk_ahead) if mode == 0 else int(walk_zone or walk_ahead or zone_walk(
            xy - 0.4 * fwd)[0])
        switching = False
        if want != mode:
            if switch_until < 0:
                switch_until = t + 0.6
            elif t >= switch_until:
                mode, switch_until = want, -1.0
            switching = True
        for sx0, sx1, sy0, sy1, vmax in scene.slow_zones:
            for probe in (xy, xy + 1.2 * fwd):
                if sx0 <= probe[0] <= sx1 and sy0 <= probe[1] <= sy1:
                    spd = min(spd, vmax)
        d_t = tgt - xy
        e_head = wrap(math.atan2(d_t[1], d_t[0]) - yaw) if np.linalg.norm(d_t) > 0.05 else 0.0
        if look_pt is not None and spd < 0.3:
            e_use = wrap(math.atan2(look_pt[1] - xy[1], look_pt[0] - xy[0]) - yaw)
        else:
            e_use = e_head
        wz = float(np.clip(2.0 * e_use, -1.0, 1.0))
        cap = min(spd, 0.45) if mode == 1 else spd
        vx = cap * max(math.cos(e_head), 0.0) ** 3
        if abs(e_head) > 0.6:
            vx = min(vx, 0.12)
        if switching:
            vx, wz = 0.0, 0.0
        vx_cmd = float(np.clip(vx, vx_cmd - POLICY_DT, vx_cmd + POLICY_DT))
        cmd_term.time_left[:] = 1e9
        cmd_term.vel_command_b[:] = torch.tensor([[vx_cmd, 0.0, wz]], device=device)

        with torch.inference_mode():
            a_w = wheel_policy(obs)
            a_k = walk_policy(obs["policy"])
            actions = a_k if mode == 1 else a_w
            wheel_term._scale[:] = 0.0 if mode == 1 else WHEEL_SCALE
            obs, _, dones, _ = env.step(actions)
        if bool(dones[0]):
            post(t, "fall", f"หุ่นล้มที่ {zname}")
            break
        tau, qd = robot.data.applied_torque.torch[0], robot.data.joint_vel.torch[0]
        e_pos += float((tau * qd).clamp(min=0).sum()) * POLICY_DT
        guest.update(POLICY_DT)
        if PA.job == "camera" and "front_t" in shot:
            guest.heading = math.atan2(xy[1] - guest.pos[1], xy[0] - guest.pos[0])
        t += POLICY_DT
        k += 1
        for i in range(1, len(done_steps)):
            pass

        # ---------------------------------------------------------------- video (25 fps)
        if k % 2:
            continue
        if t > alert_until:
            alert = ""
        log_people(t)
        log_parcel(rp, q)
        pip = None
        if PA.job == "camera" and phase >= 1:
            head = rp + np.array([0.32 * math.cos(yaw), 0.32 * math.sin(yaw), 0.15])
            dgc = max(np.linalg.norm(guest.pos[:2] - xy), 0.5)
            tg = np.array([guest.pos[0], guest.pos[1], guest.pos[2] + 1.05]) + offset
            # frame the whole person: wider lens when close, never tighter than 30 deg
            viewer.camera.fov = float(np.clip(2 * math.degrees(math.atan2(1.15, dgc)), 30.0, 75.0))
            viz.set_camera_view(tuple(head), tuple(tg))
            # RTX reads back the previous request's frame and accumulates over frames: re-render after a switch
            for _ in range(3 if PA.rtx else 1):
                img = viz.render_rgb_array()
            pip = sky.composite(viewer, img)
            guest_writer.append_data(pip)
        # main camera: chase, wider and higher on the stairs; framing both robot and guest for the camera job
        yaw_s = yaw if yaw_s is None else yaw_s + 0.04 * wrap(yaw - yaw_s)
        if PA.job == "camera" and phase >= 1:
            gw = np.array([guest.pos[0], guest.pos[1], guest.pos[2] + 0.9]) + offset
            if "front_t" in shot or phase >= 3:  # wide side shot: robot filming the guest
                mid = 0.5 * rp + 0.5 * gw
                ax = gw[:2] - rp[:2]
                nrm = np.array([ax[1], -ax[0]]) / max(np.linalg.norm(ax), 1e-6)
                if nrm[1] > 0:
                    nrm = -nrm
                eye = mid + np.array([4.6 * nrm[0], 4.6 * nrm[1], 1.2])
            else:
                mid = 0.75 * rp + 0.25 * gw
                ang = yaw_s + math.pi - math.radians(35)
                eye = rp + np.array([3.0 * math.cos(ang), 3.0 * math.sin(ang), 1.4])
            viewer.camera.fov = 45.0
            viz.set_camera_view(*scam(eye, mid))
        else:
            side = math.radians(-15 if mode else -35)
            dc, hc = (2.4, 2.0) if mode else (3.2, 1.5)
            if DLV and phase >= 2:
                side, dc, hc = math.radians(-70), 4.2, 1.8
            ang = yaw_s + math.pi - side
            eye = rp + np.array([dc * math.cos(ang), dc * math.sin(ang), hc])
            if PA.job == "guide":  # over the guest's shoulder: guest follows, robot leads (stays on the free path)
                gw = np.array([guest.pos[0], guest.pos[1], 0.0]) + offset
                dgr = rp[:2] - gw[:2]
                dgr = dgr / max(np.linalg.norm(dgr), 1e-6)
                left = np.array([-dgr[1], dgr[0]])
                # ahead of the robot on its own path, looking back at robot + guest
                eye = rp + np.array([3.4 * dgr[0] - 0.7 * left[0], 3.4 * dgr[1] - 0.7 * left[1], 1.25])
                look_at = 0.6 * rp + 0.4 * (gw + np.array([0, 0, 0.9]))
                if phase >= 2:  # at the room door: side view of robot, guest and door
                    eye = rp + np.array([-3.2, -1.2, 1.6])
                    look_at = 0.5 * rp + 0.5 * (gw + np.array([0, 0, 0.9]))
                if phase == 0:  # check-in at the counter
                    eye, look_at = rp + np.array([2.8, -2.6, 1.4]), 0.5 * rp + 0.5 * gw
            else:
                look_at = rp + np.array([0.6 * math.cos(yaw_s), 0.6 * math.sin(yaw_s), -0.05])
            if DLV and t < 4.5:  # opening shot: robot at the reception counter
                eye = rp + np.array([2.6, -2.4, 1.3])
            viewer.camera.fov = 45.0
            viz.set_camera_view(*scam(eye, look_at, cut=DLV and abs(t - 4.5) < 0.03))
        for _ in range(3 if (PA.rtx and pip is not None) else 1):
            img = viz.render_rgb_array()
        frame = sky.composite(viewer, img)
        st = dict(header=header, title=title, clock=clock(t), zone=zname, mode="walk" if mode else "wheel",
                  switching=switching, speed=float(np.linalg.norm(robot.data.root_lin_vel_w.torch[0, :2].cpu().numpy())),
                  dist=dist, batt=100.0 * (1 - t / 9000.0), checkpoints=list(zip(steps, done_steps)),
                  list_title="ขั้นตอนงาน", log_title="แจ้งเตือน → แขก / ฟรอนต์", pos=tuple(xy), yaw=yaw,
                  events=ev_lines, alert=alert, alert_color=alert_color,
                  people=[("g", guest.pos[:2], (90, 170, 255, 255))], pip=pip,
                  pip_label="กล้องหุ่น • REC (คลิปของแขก)")
        st.update(part=PART[0], part_color=PART[1], title=SHORT,
                  step=next((n for n, ok in zip(steps, done_steps) if not ok), steps[-1] + " ✓"))
        writer.append_data(hud.draw_clean(frame, st))
        if k % 500 == 0:
            print(f"[JOB] t={t:.0f}s phase {phase} zone {zname} mode {mode} xy {xy.round(2)} "
                  f"guest {guest.pos[:2].round(2)} spd {spd:.2f} idx {idx} cmd {vx_cmd:.2f}", flush=True)

    writer.close()
    if guest_writer is not None:
        guest_writer.close()
    res = dict(job=PA.job, success=finish_t is not None, time_s=finish_t, distance_m=dist, energy_pos_Wh=e_pos / 3600,
               payload_kg=PA.payload_kg if CARRY else 0.0, events=events)
    json.dump(res, open(os.path.join(PA.out, f"result_{PA.job}.json"), "w"), ensure_ascii=False, indent=1)
    print("[RESULT]", json.dumps(res, ensure_ascii=False), flush=True)


play_rsl_rl.run_playback = run_job
if __name__ == "__main__":
    play_rsl_rl.run(REST + ["--task", TASK])
