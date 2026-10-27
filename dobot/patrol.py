"""Night patrol of a resort by the wheel-legged Dobot (simulation only).

Two learned policies share one robot: the all-terrain rolling policy (wheel mode) and the stair-walking
policy (walk mode, wheels braked). A scripted patrol layer follows the route, switches modes at stairs,
avoids displaced furniture, detects anomalies, and logs metrics.

Scenarios:
  1  normal patrol lap (video)
  2  person in the pool area after closing time -> stop, face, snapshot, event, guard arrives, resume
  3  beach gate left open -> snapshot, event, resume
  4  robustness: N randomized laps in parallel (friction, wet floors, pushes, displaced furniture)

Usage (from ~/IsaacLab):
  uv run --extra video python ~/dobot_tools/resort/patrol.py --scenario 2 --out <dir> \
      --checkpoint <wheel model.pt> --walk_policy <walk exported/policy.pt> --num_envs 1 --viz newton
"""

from __future__ import annotations

import argparse
import datetime as dt_
import json
import math
import os
import sys
import time

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
pa.add_argument("--scenario", type=int, default=1, help="1-4, or 5 = two-robot shift change (num_envs 2)")
pa.add_argument("--out", required=True)
pa.add_argument("--walk_policy", required=True)
pa.add_argument("--record", type=int, default=1, help="record video of env 0")
pa.add_argument("--seed_offset", type=int, default=0)
pa.add_argument("--start_wp", type=int, default=1, help="debug: start at this waypoint (teleport)")
pa.add_argument("--rtx", type=int, default=0, help="render with Newton's OVRTX path tracer instead of OpenGL")
pa.add_argument("--max_minutes", type=float, default=12.0, help="sim-time limit per lap")
pa.add_argument("--battery_hours", type=float, default=2.5, help="rated runtime used by the HUD")
pa.add_argument("--battery_wh", type=float, default=900.0, help="assumed pack energy for the estimate")
pa.add_argument("--base_power_w", type=float, default=80.0, help="assumed compute/sensor/idle electrical load")
pa.add_argument("--drive_eff", type=float, default=0.65, help="assumed battery->shaft efficiency")
PA, REST = pa.parse_known_args()
os.makedirs(PA.out, exist_ok=True)

import patrol_env_cfg as pec  # noqa: E402
import resort_scene as rs  # noqa: E402

W, H = 1280, 720
POLICY_DT = 0.02
CLOCK0 = dt_.datetime(2026, 10, 22, 23, 40, 0)


# ------------------------------------------------------------------------------------------ env patching

_create_env = play_rsl_rl.create_isaaclab_env


def _create_env_patched(task, env_cfg, args_cli, **kw):
    from isaaclab.managers import EventTermCfg

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    if PA.rtx:
        from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

        from night_render import sky_key

        bg = (1.0, 0.0, 1.0)  # magenta background, chroma-keyed out and replaced by the sky photo

        viz_cfg = NewtonRTXVisualizerCfg(headless=True, visible_env_indices=VIS, randomly_sample_visible_envs=False,
                                         window_width=W, window_height=H, enable_markers=False, rtx_environment="none",
                                         background_color=bg)
    else:
        viz_cfg = NewtonGLVisualizerCfg(headless=True, visible_env_indices=VIS, randomly_sample_visible_envs=False,
                                        window_width=W, window_height=H, enable_markers=False)
    env_cfg.sim.visualizer_cfgs = [viz_cfg]
    if PA.rtx:
        import rtx_actors as ra

        ra.add_actor_objects(env_cfg, rtx_actor_specs())
    if PA.scenario == 6:  # first-aid kit on the back (~2 kg)
        from isaaclab.managers import SceneEntityCfg

        env_cfg.events.add_base_mass = EventTermCfg(
            func=pec.mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="link_trunk"),
                    "mass_distribution_params": (2.0, 2.0), "operation": "add"},
        )
    if PA.scenario == 4:
        env_cfg.events.push_robot = EventTermCfg(
            func=pec.mdp.push_by_setting_velocity, mode="interval", interval_range_s=pec.PUSH_INTERVAL_S,
            params=pec.PUSH_TERM_PARAMS,
        )
    env_cfg.seed = 1234 + PA.seed_offset
    return _create_env(task, env_cfg, args_cli, **kw)


play_rsl_rl.create_isaaclab_env = _create_env_patched

VIS = [0, 1] if PA.scenario == 5 else [0]  # scenario 5: two robots share env 0's resort
INTRUDER_CLOTHES, GUARD_CLOTHES, VEST = (0.85, 0.25, 0.25), (0.13, 0.17, 0.30), (0.85, 0.9, 0.15)


def rtx_actor_specs():
    import rtx_actors as ra

    actors = {"gate": ra.GATE_SPECS}
    if PA.scenario in (2, 6):
        actors["intruder"] = ra.person_specs(INTRUDER_CLOTHES, ra.SKIN_LIGHT)
        actors["guard"] = ra.person_specs(GUARD_CLOTHES, ra.SKIN_TAN, VEST)
    if PA.scenario == 5:
        actors["guard"] = ra.person_specs(GUARD_CLOTHES, ra.SKIN_TAN, VEST)
        actors["pack_old"] = [("pack_old", "cuboid", (0.30, 0.16, 0.09), (0.25, 0.25, 0.28))]
        actors["pack_new"] = [("pack_new", "cuboid", (0.30, 0.16, 0.09), (0.2, 0.55, 0.3))]
    if PA.scenario == 6:
        actors["ped1"] = ra.person_specs((0.20, 0.45, 0.75), ra.SKIN_LIGHT)
        actors["ped2"] = ra.person_specs((0.92, 0.86, 0.70), ra.SKIN_LIGHT)
        actors["fallen"] = ra.person_specs((0.30, 0.60, 0.45), ra.SKIN_LIGHT)
        actors["kit"] = ra.KIT_SPECS
    return actors


from common import Hud, Person, SmoothCam, person_parts, place, wrap, yaw_of  # noqa: E402


# ------------------------------------------------------------------------------------------ main loop


def patrol(step, *, dt, args_cli, env_cfg):
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from newton import ModelFlags

    from night_render import NightScene, Sky, setup_night, setup_rtx_lights

    cells = dict(zip(step.__code__.co_freevars, (c.cell_contents for c in step.__closure__)))
    env, wheel_policy = cells["env"], cells["policy"]
    uenv = env.unwrapped
    device = uenv.device
    robot = uenv.scene["robot"]
    N = uenv.num_envs
    walk_policy = torch.jit.load(PA.walk_policy, map_location=device).eval()
    scene = pec.SCENE
    rng = np.random.default_rng(100 + PA.seed_offset)

    print("[DEBUG] env_origins", uenv.scene.env_origins[:4].cpu().numpy().round(2).tolist(),
          "terrain", uenv.scene.terrain.env_origins[:4].cpu().numpy().round(2).tolist(),
          "root", robot.data.root_pos_w.torch[:4].cpu().numpy().round(2).tolist(), flush=True)
    origin = uenv.scene.env_origins[0].cpu().numpy().astype(np.float64)
    offset = origin - np.array([scene.dock[0], scene.dock[1], 0.0])
    # play mode may tile several resort copies; each env uses the copy under its own origin
    offsets = uenv.scene.env_origins.cpu().numpy().astype(np.float64) - np.array([scene.dock[0], scene.dock[1], 0.0])
    route = scene.route
    WP = np.array([[w.x, w.y] for w in route])
    SPEED = np.array([w.speed for w in route])
    REACH = np.array([0.35 if w.speed <= 0.45 else 0.7 for w in route])
    cp_idx = [i for i, w in enumerate(route) if w.checkpoint]
    cp_names = [route[i].checkpoint for i in cp_idx]
    K = len(cp_idx)

    # --- visualizer: night look, hidden terrain shape, static resort meshes ---------------------
    driver = None  # RTX: animated actors are model bodies (rtx_actors.ActorDriver)
    vis = [v for v in SimulationContext.instance().visualizers if hasattr(v, "render_rgb_array")]
    viz = vis[0] if vis else None
    night = NightScene(scene, offset, device=str(device))
    model = NewtonManager.get_model()
    if viz is not None:
        viewer = viz._viewer
        if not PA.rtx:
            setup_night(viewer)
        body = model.shape_body.numpy()
        stype = model.shape_type.numpy()
        import newton

        terrain_shapes = np.nonzero((body == -1) & (stype == int(newton.GeoType.MESH)))[0]
        if model.shape_opacity is not None:
            op = model.shape_opacity.numpy()
            op[terrain_shapes] = 0.0
            model.shape_opacity.assign(op)
        flags = model.shape_flags.numpy()  # not drawn at all (RTX renders a zero-opacity mesh as black)
        flags[terrain_shapes] &= ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(flags)
        if model.shape_color is not None:
            col = model.shape_color.numpy()
            col[terrain_shapes] = (0.05, 0.05, 0.05)
            model.shape_color.assign(col)
        driver = None
        if PA.rtx:
            import rtx_actors as ra

            driver = ra.ActorDriver(uenv, rtx_actor_specs(), offset, model)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))  # draw each env where it is simulated
        night.log_static(viewer, lamps_on=not PA.rtx)
        if PA.rtx:  # real lamp lights instead of fake glow; OVRTX builds its scene on the first frame
            setup_rtx_lights(viewer, "night", scene.lamps, offset)

            sky = Sky(viewer, "night", rtx=True)
        else:
            sky = Sky(viewer, "night", viewer.renderer._sun_direction)
    hud = Hud(scene)

    # --- friction per zone ------------------------------------------------------------------------
    mu_bind = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
    mu_scale = rng.uniform(0.85, 1.1, N) if PA.scenario == 4 else np.ones(N)
    wet = (rng.random(N) < 0.5) if PA.scenario == 4 else np.zeros(N, bool)
    wet_mu = rng.uniform(0.25, 0.4, N)
    cur_mu = np.full(N, -1.0)

    def zone_info(xy):
        names = np.array(["-"] * len(xy), dtype=object)
        mu = np.full(len(xy), 0.75)
        walk = np.zeros(len(xy), bool)
        wetable = np.zeros(len(xy), bool)
        done_ = np.zeros(len(xy), bool)
        for z in scene.zones:
            x0, x1, y0, y1 = z.rect
            m = (~done_) & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            names[m] = z.name
            mu[m] = z.mu
            walk[m] = z.mode == "walk"
            wetable[m] = z.wet_able
            done_ |= m
        return names, mu, walk, wetable

    # --- displaced furniture -----------------------------------------------------------------------
    obstacles = [uenv.scene[f"obstacle{i}"] for i in range(pec.NUM_CHAIRS + pec.NUM_PLANTERS)]
    obs_kind = ["chair"] * pec.NUM_CHAIRS + ["planter"] * pec.NUM_PLANTERS
    obs_xy = np.full((N, len(obstacles), 2), 1e3)
    if PA.scenario == 6:  # a pool chair pulled onto the deck path, a planter on the garden path
        obs_xy[:, 0] = (61.1, -8.7)
        obs_xy[:, pec.NUM_CHAIRS] = (50.35, -28.8)
    if PA.scenario in (4, 6):
        slots = scene.obstacle_slots if PA.scenario == 4 else []
        for e in range(N if PA.scenario == 4 else 0):
            n_used = rng.integers(1, len(obstacles) + 1)
            chosen = rng.choice(len(slots), size=n_used, replace=False)
            ci = pi = 0
            for sidx in chosen:
                sx, sy, syaw, kind = slots[sidx]
                if kind == "chair" and ci < pec.NUM_CHAIRS:
                    k = ci
                    ci += 1
                elif kind == "planter" and pi < pec.NUM_PLANTERS:
                    k = pec.NUM_CHAIRS + pi
                    pi += 1
                else:
                    continue
                obs_xy[e, k] = (sx + rng.uniform(-0.3, 0.3), sy + rng.uniform(-0.3, 0.3))
        for k, ob in enumerate(obstacles):
            pose = torch.zeros(N, 7, device=device)
            for e in range(N):
                if obs_xy[e, k, 0] < 900:
                    x, y = obs_xy[e, k]
                    z = night.ground_z(x, y) + (0.43 if obs_kind[k] == "chair" else 0.38)
                    pose[e, :3] = torch.tensor(np.array([x, y, z]) + offsets[e], device=device)
                else:
                    px, py = pec.PARK[k]
                    pose[e, :3] = torch.tensor(np.array([px, py, rs.L_BEACH + 0.45]) + offsets[e], device=device)
                yawk = rng.uniform(-0.6, 0.6)
                pose[e, 3:] = torch.tensor([0, 0, math.sin(yawk / 2), math.cos(yawk / 2)], device=device)
            ob.write_root_pose_to_sim_index(root_pose=pose)
            ob.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=device))

    # --- scenario actors ----------------------------------------------------------------------------
    intruder = guard = None
    gate_angle = 0.0
    if PA.scenario == 2:
        intruder = Person("intruder", [(54.6, -19.6, rs.L_LOW)], 1.0, INTRUDER_CLOTHES, heading=2.6)
        guard = Person("guard", [(58.5, 3.2, 0.0)], 1.4, GUARD_CLOTHES, vest=VEST)
    if PA.scenario == 3:
        gate_angle = math.radians(75)
    ped1 = ped2 = fallen = None
    if PA.scenario == 6:
        intruder = Person("intruder", [(53.8, -19.2, rs.L_LOW)], 0.9, INTRUDER_CLOTHES, heading=2.4)
        guard = Person("guard", [(36.5, -45.8, rs.L_VILLA)], 2.3, GUARD_CLOTHES, vest=VEST)
        ped1 = Person("ped1", [(55.5, 3.95, 0.0)], 1.0, (0.20, 0.45, 0.75))
        ped2 = Person("ped2", [(56.7, 4.0, 0.0)], 1.0, (0.92, 0.86, 0.70))
        fallen = Person("fallen", [(51.9, -38.6, rs.L_VILLA)], 0.0, (0.30, 0.60, 0.45))

    import trimesh as _tm

    import incidents

    KIT = incidents.kit_parts()
    PACK = _tm.creation.box(extents=(0.30, 0.16, 0.09))
    ev6 = {"phase": 0, "fall": 0}
    k_dbg = 0
    LEAF = rs.box(0, scene.gate_width, -0.04, 0.04, 0, 1.5)  # gate leaf, hinge at the local origin

    def log_actor_meshes():
        if viz is None:
            return
        for who in (intruder, guard, ped1, ped2):
            if who is None:
                continue
            parts = person_parts(who.phase, who.clothes, who.vest,
                                 skin=(0.72, 0.54, 0.42) if who.vest else (0.86, 0.68, 0.54), walking=who.walking)
            parts = place(parts, *who.pos, who.heading)
            driver.set(who.name, parts) if driver is not None else night.log_parts(viz._viewer, who.name, parts)
        if fallen is not None:
            parts = incidents.lying(person_parts(0.0, fallen.clothes, skin=(0.86, 0.68, 0.54), walking=False),
                                    *fallen.pos, math.pi / 2)
            driver.set("fallen", parts) if driver is not None else night.log_parts(viz._viewer, "fallen", parts)
        if PA.scenario == 5 and swap:
            rpa = robot.data.root_pos_w.torch[0].cpu().numpy() - offsets[0]
            for nm, where, col in (("pack_old", swap["pack_old"], (0.25, 0.25, 0.28)),
                                   ("pack_new", swap["pack_new"], (0.2, 0.55, 0.3))):
                M = np.eye(4)
                if where == "robot" and (nm == "pack_old" or swap["state"] in ("SWAP", "SELF_CHECK", "STANDBY")):
                    M[:3, 3] = rpa + np.array([0.0, 0.0, 0.17])
                elif where == "guard":
                    hand = -0.2 if nm == "pack_old" else 0.2
                    hx, hy = math.cos(guard.heading), math.sin(guard.heading)
                    M[:3, 3] = (guard.pos[0] + 0.3 * hx - hand * hy, guard.pos[1] + 0.3 * hy + hand * hx, 0.95)
                    if nm == "pack_new" and swap["state"] not in ("SWAP",):
                        M[:3, 3] = (-3.0, 5.5, 1.2)  # on the charger shelf inside the booth
                else:
                    M[:3, 3] = (-3.0, 5.5, 1.2)
                parts = [(nm, PACK, col, M)]
                driver.set(nm, parts) if driver is not None else night.log_parts(viz._viewer, nm, parts)
        if PA.scenario == 6:  # first-aid kit rides on the robot until the guard takes it
            if ev6.get("kit_to_guard"):
                M = _tm.transformations.rotation_matrix(guard.heading, (0, 0, 1))
                M[:3, 3] = (guard.pos[0] + 0.3 * math.cos(guard.heading), guard.pos[1] + 0.3 * math.sin(guard.heading),
                            guard.pos[2] + 0.95)
            else:
                qr = robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
                x_, y_, z_, w_ = qr
                Rm = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                               [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                               [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
                M = np.eye(4)
                M[:3, :3] = Rm
                M[:3, 3] = robot.data.root_pos_w.torch[0].cpu().numpy() - offsets[0] + Rm @ np.array([-0.05, 0, 0.15])
            parts = [(tag, mesh, col, M @ T) for tag, mesh, col, T in KIT]
            driver.set("kit", parts) if driver is not None else night.log_parts(viz._viewer, "kit", parts)
        # beach gate leaf: hinge at the west post, swings outward (south)
        hx, hy = scene.gate_hinge
        M = _tm.transformations.rotation_matrix(-gate_angle, (0, 0, 1))
        M[:3, 3] = (hx, hy, rs.L_VILLA + 0.05)
        if driver is not None:  # the rigid leaf is a centered box: hinge -> center
            driver.set("gate", [("leaf", None, None, M @ np.array([[1, 0, 0, scene.gate_width / 2], [0, 1, 0, 0],
                                                                    [0, 0, 1, 0.75], [0, 0, 0, 1.0]]))])
        else:
            night.log_parts(viz._viewer, "gate", [("leaf", LEAF, (0.35, 0.25, 0.16), M)])

    # --- action-term scale: wheels braked in walk mode ---------------------------------------------------
    wheel_term = uenv.action_manager.get_term("wheel_vel")
    wscale = torch.full((N, wheel_term.action_dim), float(wheel_term._scale if not torch.is_tensor(
        wheel_term._scale) else wheel_term._scale.flatten()[0]), device=device)
    WHEEL_SCALE = float(wscale[0, 0])
    wheel_term._scale = wscale
    n_act = uenv.action_manager.total_action_dim

    cmd_term = uenv.command_manager.get_term("base_velocity")
    cmd_term.is_heading_env[:] = False
    cmd_term.is_standing_env[:] = False

    # --- per-env state -------------------------------------------------------------------------------
    idx = np.ones(N, int)
    mode = np.zeros(N, int)  # 0 wheel, 1 walk
    switch_until = np.full(N, -1.0)
    switches = np.zeros(N, int)
    stop_until = np.full(N, -1.0)
    hold = np.zeros(N, bool)
    hold_look = np.zeros((N, 2))
    cp_hit = np.zeros((N, K), bool)
    done = np.zeros(N, bool)
    success = np.zeros(N, bool)
    reason = [""] * N
    fall_pos = [None] * N
    t_finish = np.full(N, np.nan)
    dist = np.zeros(N)
    e_abs = np.zeros(N)
    e_pos = np.zeros(N)
    vx_cmd = np.zeros(N)
    wp_t = np.full((N, len(WP)), np.nan)  # arrival time / energy per waypoint (route energy profile)
    wp_e = np.full((N, len(WP)), np.nan)
    best_d = np.full(N, 1e9)
    best_t = np.zeros(N)
    prev_xy = None
    events = []
    ev_lines = []
    alert = ""
    alert_until = -1.0
    cam_mode = "chase"
    cam_until = -1.0
    cam_target = None
    yaw_s = None
    snap_count = 0
    s2 = {"phase": 0}
    s3 = {"phase": 0}

    def clock(t):
        c0 = CLOCK0 + (dt_.timedelta(minutes=47, seconds=30) if PA.scenario == 5 else dt_.timedelta())
        skip = swap.get("clock_skip", 0.0) if PA.scenario == 5 else 0.0
        return (c0 + dt_.timedelta(seconds=t + skip)).strftime("%H:%M:%S")

    def post_event(t, etype, text, xy, snapshot=None, severity="warning"):
        c0 = CLOCK0 + (dt_.timedelta(minutes=47, seconds=30) if PA.scenario == 5 else dt_.timedelta())
        skip = swap.get("clock_skip", 0.0) if PA.scenario == 5 else 0.0
        ev = {"time": (c0 + dt_.timedelta(seconds=t + skip)).isoformat(), "type": etype, "severity": severity,
              "message": text, "location": {"x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2)},
              "zone": zone_info(np.array([xy]))[0][0], "snapshot": snapshot, "robot": "dobot-wl-01"}
        events.append(ev)
        ev_lines.append(f"{clock(t)}  {text}")
        with open(os.path.join(PA.out, "events.jsonl"), "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print("[EVENT]", json.dumps(ev, ensure_ascii=False), flush=True)

    # --- video ------------------------------------------------------------------------------------------
    writer = None
    if PA.record and viz is not None:
        import imageio.v2 as imageio

        writer = imageio.get_writer(os.path.join(PA.out, f"scenario{PA.scenario}.mp4"), fps=25, quality=8,
                                    macro_block_size=8)

    scam = SmoothCam()

    def look(eye, target, fov=45.0, smooth=False):
        viz._viewer.camera.fov = fov
        viz.set_camera_view(*scam(eye, target, cut=not smooth))

    def robot_cam(p3, yaw, target_xy=None, target_z=None, fov=28.0):
        head = p3 + np.array([0.32 * math.cos(yaw), 0.32 * math.sin(yaw), 0.12])
        if target_xy is None:
            tgt = head + np.array([math.cos(yaw), math.sin(yaw), -0.05])
        else:
            tgt = np.array([target_xy[0], target_xy[1], target_z]) + offset
        look(head, tgt, fov=fov)
        return head

    def boxes_for(people_specs):
        out = []
        for (cx, cy, z0, hgt, wdt, lab) in people_specs:
            pts = []
            for dz in (0.0, hgt):
                for ddx in (-wdt, wdt):
                    for ddy in (-wdt, wdt):
                        pts.append([cx + ddx, cy + ddy, z0 + dz])
            pts = np.array(pts) + offset
            u, v, w_, front = night.project(viz._viewer, pts, W, H)
            if front.all():
                u0, u1 = float(np.clip(u.min(), 2, W - 3)), float(np.clip(u.max(), 2, W - 3))
                v0, v1 = float(np.clip(v.min(), 76, H - 3)), float(np.clip(v.max(), 76, H - 3))
                out.append((u0, v0, u1, v1, lab))
        return out

    def snapshot(t, name, specs, label):
        nonlocal snap_count
        from PIL import Image

        f = viz.render_rgb_array()
        f = night.add_glow(viz._viewer, sky.composite(viz._viewer, f))
        st = {"cam_label": label, "clock": clock(t), "boxes": boxes_for(specs)}
        f = hud.draw_clean(f, st)
        snap_count += 1
        path = os.path.join(PA.out, f"snapshot_{snap_count:02d}_{name}.png")
        Image.fromarray(f).save(path)
        return os.path.basename(path)

    if PA.scenario == 6 and PA.start_wp == 1:
        PA.start_wp = 4
    if PA.start_wp > 1:
        sw = PA.start_wp
        idx[:] = sw
        x, y = WP[sw - 1]
        hd = math.atan2(WP[sw, 1] - y, WP[sw, 0] - x)
        pose = robot.data.root_link_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(np.array([x, y, night.ground_z(x, y) + 0.5])[None] + offsets, device=device, dtype=torch.float32)
        pose[:, 3:] = torch.tensor([0, 0, math.sin(hd / 2), math.cos(hd / 2)], device=device)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=device))
        for k, ci in enumerate(cp_idx):
            cp_hit[:, k] = ci < sw - 1
        for _ in range(25):
            with torch.inference_mode():
                cmd_term.vel_command_b[:] = 0.0
                obs0, _, _, _ = env.step(wheel_policy(env.get_observations()))
    swap = {}
    if PA.scenario == 5:  # robot A (low battery) comes home up the stairs; robot B waits charged on dock 2
        assert N == 2, "scenario 5 needs --num_envs 2"
        offsets[1] = offsets[0]  # B is drawn and simulated in A's resort copy (separate collision worlds)
        sw = 23
        idx[0], idx[1] = sw, 1
        starts = [(WP[sw - 1], math.atan2(WP[sw, 1] - WP[sw - 1, 1], WP[sw, 0] - WP[sw - 1, 0])),
                  (np.array(scene.dock2[:2]), 0.0)]
        pose = robot.data.root_link_pose_w.torch.clone()
        for e, ((x, y), hd) in enumerate(starts):
            pose[e, :3] = torch.tensor(np.array([x, y, night.ground_z(x, y) + 0.5]) + offsets[e], device=device)
            pose[e, 3:] = torch.tensor([0, 0, math.sin(hd / 2), math.cos(hd / 2)], device=device)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=device))
        cp_hit[0, :-1] = True
        hold[1] = True
        hold_look[1] = WP[1]
        swap = {"released": False, "a_done": None, "state": "RETURN_TO_BASE", "t_state": 0.0, "clock_skip": 0.0,
                "delay": float(rng.uniform(1.0, 10.0)) * 60.0, "pack_old": "robot", "pack_new": "guard"}
        guard = Person("guard", [(-1.0, 5.2, 0.0)], 1.3, GUARD_CLOTHES, vest=VEST, heading=-math.pi / 2)
        CLOCK_SHIFT = dt_.timedelta(minutes=47, seconds=30)  # 00:27:30
        for _ in range(25):
            with torch.inference_mode():
                cmd_term.vel_command_b[:] = 0.0
                env.step(wheel_policy(env.get_observations()))
    log_actor_meshes()  # every animated mesh must exist before the first (RTX) frame
    voice = incidents.VoiceTrack()
    vframes = 0
    alert_color_override = None
    obs = env.get_observations()
    t = 0.0
    k_step = 0
    t_wall = time.time()
    max_steps = int(PA.max_minutes * 60 / POLICY_DT)
    print(f"[PATROL] scenario {PA.scenario}, {N} env(s), route {len(route)} waypoints, {K} checkpoints", flush=True)
    while k_step < max_steps and not done.all():
        rp = robot.data.root_pos_w.torch.cpu().numpy().astype(np.float64)
        p3 = rp - offsets
        xy = p3[:, :2]
        yaw = yaw_of(robot.data.root_quat_w.torch.cpu().numpy())
        if prev_xy is not None:
            dist += np.where(done, 0.0, np.linalg.norm(xy - prev_xy, axis=1))
        prev_xy = xy.copy()

        # ---- route progress
        tgt_wp = WP[np.minimum(idx, len(WP) - 1)]
        d_wp = np.linalg.norm(xy - tgt_wp, axis=1)
        for k, ci in enumerate(cp_idx):
            cp_hit[:, k] |= (np.linalg.norm(xy - WP[ci], axis=1) < 1.5) & (idx >= ci)
        # a waypoint occupied by displaced furniture counts as reached once the robot is beside the obstacle
        reach = REACH[np.minimum(idx, len(WP) - 1)].copy()
        for k in range(len(obstacles)):
            d_o = np.linalg.norm(obs_xy[:, k] - tgt_wp, axis=1)
            reach = np.where(d_o < 1.0, np.maximum(reach, d_o + 1.2), reach)
        for e in np.nonzero(~done & (d_wp < reach))[0]:
            w = route[idx[e]]
            if w.stop_s > 0 and stop_until[e] < 0:
                stop_until[e] = t + w.stop_s
                hold_look[e] = w.look
                continue
            if stop_until[e] >= 0 and t < stop_until[e]:
                continue
            stop_until[e] = -1.0
            wp_t[e, idx[e]], wp_e[e, idx[e]] = t, e_pos[e]
            idx[e] += 1
            best_d[e], best_t[e] = 1e9, t
            if idx[e] >= len(WP):
                done[e], success[e], t_finish[e] = True, True, t
        # stuck detection (ignores deliberate stops)
        stopped = (stop_until >= 0) | hold | (switch_until > t)
        # distance to the (possibly just advanced) target waypoint
        d_wp = np.linalg.norm(xy - WP[np.minimum(idx, len(WP) - 1)], axis=1)
        improve = d_wp < best_d - 0.3
        best_d = np.where(improve, d_wp, best_d)
        best_t = np.where(improve | stopped, t, best_t)
        for e in np.nonzero(~done & (t - best_t > 30.0))[0]:
            done[e], reason[e], fall_pos[e] = True, "ติดค้าง (ไม่คืบหน้า 30 วินาที)", xy[e].tolist()

        # ---- mode selection (zone at the robot and 0.7 m ahead), stop briefly to switch
        fwd = np.stack([np.cos(yaw), np.sin(yaw)], 1)
        _, _, walk_here, _ = zone_info(xy)
        _, _, walk_ahead, _ = zone_info(xy + 0.7 * fwd)
        _, _, walk_behind, _ = zone_info(xy - 0.4 * fwd)
        want = np.where(mode == 0, walk_here | walk_ahead, walk_here | walk_ahead | walk_behind).astype(int)
        for e in np.nonzero(~done & (want != mode))[0]:
            if switch_until[e] < 0:
                switch_until[e] = t + 0.6
            elif t >= switch_until[e]:
                mode[e] = want[e]
                switches[e] += 1
                switch_until[e] = -1.0
        switching = (switch_until >= 0) & (want != mode)

        # ---- floor friction for the current zone
        names, mu, _, wetable = zone_info(xy)
        mu_t = np.where(wet & wetable, wet_mu, mu * mu_scale)
        chg = np.abs(mu_t - cur_mu) > 1e-3
        if chg.any():
            ids = torch.tensor(np.nonzero(chg)[0], device=device)
            mu_bind[ids] = torch.tensor(mu_t[chg], dtype=mu_bind.dtype, device=device)[:, None]
            NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
            cur_mu = mu_t

        # ---- pure pursuit + furniture avoidance
        i1 = np.minimum(idx, len(WP) - 1)
        a, b = WP[i1 - 1], WP[i1]
        seg = b - a
        L = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
        u = seg / L[:, None]
        nrm = np.stack([-u[:, 1], u[:, 0]], 1)
        s = np.clip(np.sum((xy - a) * u, 1), 0, L)
        look_ahead = np.where(mode == 1, 0.6, 1.0)
        tgt = a + u * np.minimum(s + look_ahead, L)[:, None]
        spd = SPEED[i1].copy()
        for k in range(len(obstacles)):
            o = obs_xy[:, k]
            rel = o - xy
            along = np.sum(rel * u, 1)
            lat_o = np.sum((o - a) * nrm, 1)
            clear = (0.30 if obs_kind[k] == "chair" else 0.32) + 0.75
            near = (along > -0.5) & (along < 3.5) & (np.abs(lat_o) < clear)
            shift = np.where(lat_o >= 0, lat_o - clear, lat_o + clear)
            tgt = np.where(near[:, None], tgt + nrm * shift[:, None], tgt)
            spd = np.where(near, np.minimum(spd, 0.5), spd)
        for sx0, sx1, sy0, sy1, vmax in scene.slow_zones:
            for probe in (xy, xy + 1.2 * fwd):
                inside = (probe[:, 0] >= sx0) & (probe[:, 0] <= sx1) & (probe[:, 1] >= sy0) & (probe[:, 1] <= sy1)
                spd = np.where(inside, np.minimum(spd, vmax), spd)
        # people: keep 1 m of space, slow down near them, give way when someone is right in front
        for who in (ped1, ped2, intruder, guard):
            if who is None or (who is intruder and not who.walking and ev6["phase"] == 0):
                continue
            o = who.pos[None, :2]
            rel = o - xy
            along = np.sum(rel * u, 1)
            lat_o = np.sum((o - a) * nrm, 1)
            clear = 1.1
            near = (along > -0.5) & (along < 4.0) & (np.abs(lat_o) < clear)
            shift_p = np.where(lat_o >= 0, lat_o - clear, lat_o + clear)
            tgt = np.where(near[:, None], tgt + nrm * shift_p[:, None], tgt)
            spd = np.where(near, np.minimum(spd, 0.35), spd)
            blocked = (np.linalg.norm(rel, axis=1) < 1.2) & (np.sum(rel * fwd, 1) > 0.3)
            spd = np.where(blocked, 0.0, spd)
        if ev6.get("rush") is not None:  # emergency: straight to the casualty, fast
            tgt = np.where(np.arange(N)[:, None] == 0, np.asarray(ev6["rush"])[None], tgt)
            spd = np.where(np.arange(N) == 0, 1.15, spd)
        d_t = tgt - xy
        e_head = wrap(np.arctan2(d_t[:, 1], d_t[:, 0]) - yaw)
        in_stop = (stop_until >= 0) | hold
        look_pt = hold_look
        e_look = wrap(np.arctan2(look_pt[:, 1] - xy[:, 1], look_pt[:, 0] - xy[:, 0]) - yaw)
        e_use = np.where(in_stop, e_look, e_head)
        wz = np.clip(2.0 * e_use, -1.0, 1.0)
        cap = np.where(mode == 1, np.minimum(spd, 0.45), spd)
        vx_des = cap * np.clip(np.cos(e_head), 0, 1) ** 3
        vx_des = np.where(np.abs(e_head) > 0.6, np.minimum(vx_des, 0.12), vx_des)
        vx_des = np.where(in_stop | switching | done, 0.0, vx_des)
        wz = np.where(switching | done, 0.0, wz)
        vx_cmd = np.clip(vx_des, vx_cmd - 1.0 * POLICY_DT, vx_cmd + 1.0 * POLICY_DT)
        cmd = np.stack([vx_cmd, np.zeros(N), wz], 1)
        cmd_term.time_left[:] = 1e9
        cmd_term.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32, device=device)

        # ---- scenario 2: intruder in the pool area after closing
        if PA.scenario == 2:
            e0 = 0
            if s2["phase"] == 0 and not done[e0]:
                rel = intruder.pos[:2] - xy[e0]
                dd = np.linalg.norm(rel)
                bearing = wrap(math.atan2(rel[1], rel[0]) - yaw[e0])
                head = p3[e0] + np.array([0.3 * math.cos(yaw[e0]), 0.3 * math.sin(yaw[e0]), 0.15])
                chest = np.array([intruder.pos[0], intruder.pos[1], intruder.pos[2] + 1.2])
                in_view = dd < 11.0 and abs(bearing) < math.radians(50) and names[e0] == "ลานสระว่ายน้ำ"
                if in_view and night.visible(head + offset, (chest + offset)[None])[0]:
                    s2.update(phase=1, t0=t)
                    hold[e0] = True
                    hold_look[e0] = intruder.pos[:2]
                    alert, alert_until = "ตรวจพบบุคคลในเขตสระหลังเวลาปิด (สระปิด 20:00 น.)", t + 12
                    print(f"[PATROL] intruder detected at {dd:.1f} m", flush=True)
            elif s2["phase"] == 1 and t - s2["t0"] > 1.6:
                cam_mode, cam_until = "robot", t + 3.0
                cam_target = (intruder.pos[:2], intruder.pos[2] + 1.0, 28.0)
                robot_cam(rp[e0], yaw[e0], *cam_target)
                snap = snapshot(t, "intruder", [(intruder.pos[0], intruder.pos[1], intruder.pos[2], 1.75, 0.3,
                                                  "บุคคล 0.94")], "กล้องหุ่น CAM-1  •  เขตสระว่ายน้ำ")
                post_event(t, "intruder_pool_after_hours", "ตรวจพบบุคคลในเขตสระหลังปิด → แจ้ง รปภ.",
                           intruder.pos[:2], snap)
                guard.start([(58.5, 3.2, 0.0), (58.5, 1.4, 0.0), (58.5, -0.9, rs.L_LOW + 0.15),
                             (58.5, -1.6, rs.L_LOW), (57.5, -12.0, rs.L_LOW),
                             (intruder.pos[0] + 1.0, intruder.pos[1] + 0.6, rs.L_LOW)])
                post_event(t, "dispatch", "สั่งการ รปภ. ใกล้สุด (เวรอาคาร A) ไปที่สระ", (58.5, 3.2), severity="info")
                s2["phase"] = 2
            elif s2["phase"] == 2 and guard.arrived:
                post_event(t, "guard_arrived", "รปภ. ถึงจุดเกิดเหตุ ยืนยันเหตุการณ์", guard.pos[:2], severity="info")
                alert, alert_until = "รปภ. ถึงจุดเกิดเหตุแล้ว — หุ่นกลับไปลาดตระเวนต่อ", t + 5
                s2.update(phase=3, t1=t)
                guard.heading = math.atan2(intruder.pos[1] - guard.pos[1], intruder.pos[0] - guard.pos[0])
                intruder.heading = guard.heading + math.pi
            elif s2["phase"] == 3 and t - s2["t1"] > 2.5:
                hold[e0] = False
                exit_path = [(intruder.pos[0], intruder.pos[1], rs.L_LOW), (57.0, -10.0, rs.L_LOW),
                             (58.5, -1.6, rs.L_LOW), (58.5, -0.9, rs.L_LOW + 0.15), (58.5, 1.4, 0.0),
                             (52.0, 3.0, 0.0)]
                intruder.speed = guard.speed = 0.9
                intruder.start(exit_path)
                guard.start([(guard.pos[0], guard.pos[1], rs.L_LOW)] + [(x + 0.8, y + 0.5, z) for x, y, z in
                                                                          exit_path[1:]])
                s2["phase"] = 4
            for who in (intruder, guard):
                who.update(POLICY_DT)
                if who.walking and who.seg < len(who.path) - 1:
                    pass

        # ---- scenario 6: guests walking by, pool intruder warned by voice + intercom, fallen guest + first aid
        if PA.scenario == 6:
            vt = vframes / 25.0
            if t > 1.0 and not ped1.walking and ev6.get("peds") is None:
                ev6["peds"] = True
                ped1.start([(55.5, 3.95, 0.0), (27.0, 3.95, 0.0), (20.0, 6.0, 0.0)])
                ped2.start([(56.7, 4.0, 0.0), (28.2, 4.0, 0.0), (21.0, 7.0, 0.0)])
            for who in (ped1, ped2, intruder, guard):
                who.update(POLICY_DT)
            ph = ev6["phase"]
            if ph == 0 and not done[0]:
                rel = intruder.pos[:2] - xy[0]
                dd = np.linalg.norm(rel)
                bearing = wrap(math.atan2(rel[1], rel[0]) - yaw[0])
                head = p3[0] + np.array([0.3 * math.cos(yaw[0]), 0.3 * math.sin(yaw[0]), 0.15])
                chest = np.array([intruder.pos[0], intruder.pos[1], intruder.pos[2] + 1.2])
                if os.environ.get("S6_DEBUG") and k_dbg % 25 == 0 and dd < 14.0:
                    print(f"[S6DBG] t={t:.1f} dd={dd:.1f} bearing={math.degrees(bearing):.0f} zone={names[0]} "
                          f"vis={night.visible(head + offset, (chest + offset)[None])[0]}", flush=True)
                k_dbg += 1
                if dd < 11.0 and abs(bearing) < math.radians(65) and names[0] == "ลานสระว่ายน้ำ" and \
                        night.visible(head + offset, (chest + offset)[None])[0]:
                    ev6.update(phase=1, t0=t, spot="intruder")
                    hold[0] = True
                    hold_look[0] = intruder.pos[:2]
                    alert, alert_until = "ตรวจพบบุคคลในเขตสระหลังเวลาปิด — เปิดไฟส่อง + พูดเตือน", t + 8
                    post_event(t, "intruder_pool_after_hours", "พบบุคคลในสระหลังปิด → ไฟส่อง + เตือนด้วยเสียง",
                               intruder.pos[:2])
                    e1 = voice.say(vt + 1.0, "pool_warn_th")
                    voice.say(e1, "pool_warn_en")
            elif ph == 1 and voice.done_by(vt):
                post_event(t, "intercom", "รปภ. ที่ป้อมพูดผ่านลำโพงหุ่น (สื่อสาร 2 ทาง)", intruder.pos[:2], severity="info")
                ev6.update(phase=2, guard_end=voice.say(vt + 0.6, "guard_talk_th"))
            elif ph == 2 and vt > ev6["guard_end"] - 1.5 and not intruder.walking and ev6.get("left") is None:
                ev6["left"] = t
                intruder.start([(intruder.pos[0], intruder.pos[1], rs.L_LOW), (57.2, -12.0, rs.L_LOW),
                                 (58.5, -1.6, rs.L_LOW), (58.5, -0.9, rs.L_LOW + 0.15), (58.5, 1.6, 0.0),
                                 (50.0, 3.0, 0.0)])
                hold_look[0] = intruder.pos[:2]
            elif ph == 2 and ev6.get("left") is not None:
                hold_look[0] = intruder.pos[:2]
                if intruder.pos[1] > -3.0:
                    post_event(t, "cleared", "บุคคลออกจากพื้นที่สระแล้ว — กลับลาดตระเวน", intruder.pos[:2], severity="info")
                    alert, alert_until = "บุคคลออกจากพื้นที่แล้ว ✓ หุ่นกลับไปลาดตระเวน", t + 4
                    alert_color_override = (30, 150, 70, 230)
                    ev6.update(phase=3, spot=None, ok_until=t + 4)
                    hold[0] = False
            # fallen guest on the garden path
            fp = fallen.pos[:2]
            if ev6["phase"] >= 3 and ev6["fall"] == 0:
                rel = fp - xy[0]
                dd = np.linalg.norm(rel)
                bearing = wrap(math.atan2(rel[1], rel[0]) - yaw[0])
                if dd < 14.0 and abs(bearing) < math.radians(60):
                    ev6.update(fall=1, t_fall=t, rush=(fp[0] - 1.25, fp[1] + 0.7), spot="fallen")
                    alert, alert_until = "เหตุฉุกเฉิน: พบคนล้ม/นอนนิ่ง — แจ้ง รปภ. + พยาบาล, หุ่นนำชุดปฐมพยาบาลไป", t + 9
                    post_event(t, "person_down", "เหตุฉุกเฉิน: พบคนล้มทางเดินสวน → แจ้ง รปภ./พยาบาล", fp)
            if ev6["fall"] in (1, 2) and t - ev6["t_fall"] > 16.0 and ev6.get("g_go") is None:  # staff gets the call
                ev6["g_go"] = t
                guard.start([(guard.pos[0], guard.pos[1], rs.L_VILLA), (48.6, -45.0, rs.L_VILLA),
                             (fp[0] + 0.3, fp[1] - 1.1, rs.L_VILLA)])
            if ev6["fall"] == 1:
                if np.linalg.norm(np.asarray(ev6["rush"]) - xy[0]) < 0.6:
                    ev6.update(fall=2, t_robot=t, rush=None)
                    hold[0] = True
                    hold_look[0] = fp
                    post_event(t, "first_aid_arrived", "หุ่นถึงจุดเกิดเหตุพร้อมชุดปฐมพยาบาล", fp, severity="info")
                    e1 = voice.say(vt + 0.3, "fall_alert_th")
                    e2 = voice.say(e1, "fall_alert_en")
                    voice.say(e2, "kit_th")
            elif ev6["fall"] == 2 and ev6.get("g_go") is not None and guard.arrived:
                gap = t - ev6["t_robot"]
                ev6.update(fall=3, t_guard=t, kit_to_guard=True)
                guard.heading = math.atan2(fp[1] - guard.pos[1], fp[0] - guard.pos[0])
                post_event(t, "guard_arrived", f"รปภ. ถึงที่เกิดเหตุ (หุ่นถึงก่อน {gap:.0f} วินาที) รับชุดปฐมพยาบาล", fp,
                           severity="info")
                alert, alert_until = f"หุ่นถึงก่อน รปภ. {gap:.0f} วินาที • ส่งชุดปฐมพยาบาลแล้ว ✓", t + 8
                alert_color_override = (30, 150, 70, 230)
            elif ev6["fall"] == 3 and t - ev6["t_guard"] > 6.0 and voice.done_by(vt):
                break

        # ---- scenario 1/3: gate check during the stop at the beach gate
        gate_cp = [i for i, w in enumerate(route) if w.look is not None][0]
        if PA.scenario in (1, 3) and s3["phase"] == 0 and not done[0] and idx[0] == gate_cp and stop_until[0] > 0 \
                and t > stop_until[0] - 2.0:
            gx = scene.gate_hinge[0] + scene.gate_width / 2
            if PA.scenario == 3:
                stop_until[0] += 4.0
                cam_mode, cam_until = "robot", t + 3.0
                cam_target = ((gx, scene.gate_hinge[1] - 0.8), rs.L_VILLA + 0.8, 62.0)
                robot_cam(rp[0], yaw[0], *cam_target)
                hx, hy = scene.gate_hinge
                tip = (hx + scene.gate_width * math.cos(-gate_angle), hy + scene.gate_width * math.sin(-gate_angle))
                cx, cy = (hx + tip[0]) / 2, (hy + tip[1]) / 2
                half = max(abs(tip[0] - hx), abs(tip[1] - hy)) / 2
                snap = snapshot(t, "gate_open", [(cx, cy, rs.L_VILLA + 0.05, 1.5, half, "ประตูเปิดค้าง ~75°")],
                                "กล้องหุ่น CAM-1  •  ประตูชายหาด")
                post_event(t, "gate_left_open", "ประตูรั้วชายหาดเปิดค้าง → แจ้ง รปภ.", (gx, scene.gate_hinge[1]), snap)
                alert, alert_until = "ตรวจพบประตูชายหาดเปิดค้าง — ส่งแจ้งเตือนแล้ว", t + 8
            else:
                post_event(t, "gate_ok", "ประตูชายหาด: ปิดปกติ", (gx, scene.gate_hinge[1]), severity="info")
            s3["phase"] = 1

        # ---- scenario 5: semi-automatic battery swap (the robot cannot charge itself; the guard swaps packs)
        if PA.scenario == 5:
            vt = vframes / 25.0
            guard.update(POLICY_DT)
            st5, ts = swap["state"], t - swap["t_state"]

            def go(state):
                swap.update(state=state, t_state=t)

            if not swap["released"] and t > 1.5:  # relief before the low robot goes home: no patrol gap
                hold[1] = False
                swap["released"] = True
                post_event(t, "relief", "หุ่น B (แบต 100%) รับช่วงลาดตระเวนต่อจากจุดเช็กอินล่าสุดของ A", WP[1],
                           severity="info")
                alert, alert_until = "ส่งต่องานก่อนกลับ: หุ่น B ออกลาดตระเวนแทน — ไม่มีช่วงว่าง", t + 6
            if st5 == "RETURN_TO_BASE" and done[0]:
                go("WAIT_SWAP")
                voice.say(vt + 0.3, "guard_swap_th")
                post_event(t, "swap_request", f"แจ้ง รปภ.: หุ่น A แบต 14% รอเปลี่ยนแบตที่ป้อม", WP[-1])
                alert, alert_until = "หุ่น A ถึงป้อม (แบต 14%) — แจ้ง รปภ. ให้มาเปลี่ยนแบต", t + 6
            elif st5 == "WAIT_SWAP":
                # fast-forward the guard's walk-over delay (random 1-10 min) into a few seconds of video
                swap["clock_skip"] = min(swap["delay"], swap["delay"] * max(0.0, ts - 3.0) / 3.0)
                if ts > 6.0 and not guard.walking and guard.seg == 0:
                    if swap["delay"] > 8 * 60:
                        post_event(t, "escalate", "รปภ. ยังไม่มาเกิน 8 นาที → แจ้งซ้ำ + ส่งถึงหัวหน้ากะ", WP[-1])
                    guard.start([(-1.0, 5.2, 0.0), (-0.8, 3.6, 0.0), (-1.75, 2.75, 0.0)])
                if guard.arrived and guard.seg > 0:
                    guard.heading = math.pi
                    go("SWAP")
                    post_event(t, "swap_start", f"รปภ. มาถึง (รอ {swap['delay'] / 60:.1f} นาที) เริ่มเปลี่ยนแบต", WP[-1],
                               severity="info")
            elif st5 == "SWAP":
                if ts > 1.5:
                    swap["pack_old"] = "guard"
                if ts > 4.0:
                    swap["pack_new"] = "robot"
                if ts > 5.5:
                    go("SELF_CHECK")
            elif st5 == "SELF_CHECK":
                hold[0] = True
                hold_look[0] = WP[-1] + np.array([math.cos(ts * 2.5) * 2.0, 1.0])  # small look-around: camera/legs
                if ts > 5.0:
                    go("STANDBY")
                    post_event(t, "ready", "หุ่น A ตรวจตัวเองผ่าน ✓ แบต 100% — สแตนด์บายรอกะถัดไป", WP[-1], severity="info")
                    alert, alert_until = "หุ่น A พร้อม ✓ แบต 100% • แบตเก่าเข้าแท่นชาร์จ (~1.5 ชม.)", t + 6
                    alert_color_override = (30, 150, 70, 230)
                    guard.start([(guard.pos[0], guard.pos[1], 0.0), (-0.8, 3.6, 0.0), (-1.0, 5.2, 0.0)])
            elif st5 == "STANDBY" and ts > 7.0:
                break

        # ---- policies
        with torch.inference_mode():
            a_wheel = wheel_policy(obs)
            a_walk = walk_policy(obs["policy"])
            m_t = torch.tensor(mode, device=device)[:, None].bool()
            actions = torch.where(m_t, a_walk, a_wheel)
            wheel_term._scale[:] = torch.where(m_t, 0.0, WHEEL_SCALE)
            obs, _, dones, _ = env.step(actions)
        tau = robot.data.applied_torque.torch
        qd = robot.data.joint_vel.torch
        pw = (tau * qd)
        e_abs += np.where(done, 0.0, pw.abs().sum(1).cpu().numpy() * POLICY_DT)
        e_pos += np.where(done, 0.0, pw.clamp(min=0).sum(1).cpu().numpy() * POLICY_DT)
        dn = dones.cpu().numpy().astype(bool)
        for e in np.nonzero(dn & ~done)[0]:
            done[e] = True
            z = zone_info(xy[e:e + 1])[0][0]
            reason[e] = f"ล้ม ({z})"
            fall_pos[e] = xy[e].tolist()
            print(f"[PATROL] env {e} fell at {xy[e].round(1)} zone {z} t={t:.1f}", flush=True)
        t += POLICY_DT
        k_step += 1

        # ---- video frame (env 0), every 2nd step = 25 fps real time
        if writer is not None and k_step % 2 == 0:
            if t > alert_until:
                alert = ""
            if cam_mode == "robot" and t < cam_until:
                robot_cam(rp[0], yaw[0], *cam_target)
            else:
                cam_mode = "chase"
                if PA.scenario == 2 and s2["phase"] in (2, 3):
                    look((60.5 + offset[0], -6.0 + offset[1], rs.L_LOW + 3.2 + offset[2]),
                         ((intruder.pos[0] + 58.0) / 2 + offset[0], (intruder.pos[1] - 8) / 2 + offset[1],
                          rs.L_LOW + 0.6 + offset[2]))
                else:
                    y0 = yaw[0]
                    yaw_s = y0 if yaw_s is None else yaw_s + 0.04 * wrap(y0 - yaw_s)
                    on_stairs = mode[0] == 1
                    side = math.radians(-15 if on_stairs else -35)
                    dist_c, hgt = (2.4, 2.0) if on_stairs else (3.2, 1.5)
                    ang = yaw_s + math.pi - side
                    eye = rp[0] + np.array([dist_c * math.cos(ang), dist_c * math.sin(ang), hgt])
                    look(eye, rp[0] + np.array([0.6 * math.cos(yaw_s), 0.6 * math.sin(yaw_s), -0.05]), smooth=True)
            if PA.scenario == 5:  # wide on the way home, then close on the booth for the swap
                if swap["state"] == "RETURN_TO_BASE":
                    look(np.array([6.0, 7.5, 4.2]) + offsets[0], np.array([-1.2, -1.5, -0.4]) + offsets[0], fov=52.0)
                else:
                    look(np.array([2.6, -1.4, 1.9]) + offsets[0], np.array([-2.2, 3.0, 0.5]) + offsets[0], fov=50.0,
                         smooth=True)
            if PA.scenario == 6 and ev6["phase"] in (1, 2):  # robot + spotlit intruder from behind the robot
                mid = 0.5 * rp[0] + 0.5 * (np.array([intruder.pos[0], intruder.pos[1], intruder.pos[2] + 0.6]) + offsets[0])
                bk = rp[0][:2] - (np.array(intruder.pos[:2]) + offsets[0][:2])
                bk = bk / max(np.linalg.norm(bk), 1e-6)
                look(rp[0] + np.array([3.8 * bk[0] + 1.4 * bk[1], 3.8 * bk[1] - 1.4 * bk[0], 1.9]), mid, smooth=True)
            if PA.scenario == 6 and ev6["fall"] >= 2:  # side view: casualty, robot with the kit, guard running in
                fpw = np.array([fallen.pos[0], fallen.pos[1], fallen.pos[2] + 0.3]) + offsets[0]
                look(fpw + np.array([4.6, 3.4, 2.6]), 0.5 * fpw + 0.5 * rp[0], smooth=True)
            log_actor_meshes()
            frame = viz.render_rgb_array()
            frame = night.add_glow(viz._viewer, sky.composite(viz._viewer, frame))
            vframes += 1
            if PA.scenario == 6:
                head_w = rp[0] + np.array([0.33 * math.cos(yaw[0]), 0.33 * math.sin(yaw[0]), 0.2])
                tgt_spot = {"intruder": intruder, "fallen": fallen}.get(ev6.get("spot") or "")
                if tgt_spot is not None:
                    gz = night.ground_z(tgt_spot.pos[0], tgt_spot.pos[1])
                    frame = incidents.add_spotlight(night, viz._viewer, frame, head_w,
                                                    np.array([tgt_spot.pos[0], tgt_spot.pos[1], gz + 0.02]) + offsets[0],
                                                    strength=0.6)
                if ev6["phase"] in (1, 2) or ev6["fall"] in (1, 2):
                    on = int(t * 4) % 2 == 0
                    col = (255, 60, 40) if ev6["fall"] in (1, 2) else (255, 170, 30)
                    frame = incidents.add_beacon(night, viz._viewer, frame, rp[0] + np.array([0, 0, 0.22]), col, on)
            batt = 100.0 * (1 - t / (PA.battery_hours * 3600))  # rated runtime, linear
            people = []
            if intruder is not None:
                people.append(("i", intruder.pos[:2], (255, 80, 80, 255)))
                people.append(("g", guard.pos[:2], (240, 230, 60, 255)))
            st = dict(title=f"Scenario {PA.scenario}: " + {1: "ลาดตระเวนรอบปกติ", 2: "บุคคลในเขตสระหลังปิด",
                                                              3: "ประตูเปิดค้าง", 4: "ทดสอบความทนทาน",
                                                              5: "สลับแบต", 6: "เหตุการณ์กลางคืน"}[PA.scenario],
                      clock=clock(t), zone=names[0], mode="walk" if mode[0] else "wheel",
                      switching=bool(switching[0]), speed=float(np.linalg.norm(robot.data.root_lin_vel_w.torch[0, :2]
                                                                                 .cpu().numpy())),
                      dist=float(dist[0]), batt=batt, checkpoints=list(zip(cp_names, cp_hit[0].tolist())),
                      pos=tuple(xy[0]), yaw=float(yaw[0]), events=ev_lines, alert=alert, people=people)
            if cam_mode == "robot":
                st["cam_label"] = "กล้องหุ่น CAM-1"
                st["boxes"] = []
            nxt_cp = next((n for n, ok in zip(cp_names, cp_hit[0].tolist()) if not ok), "กลับป้อม")
            sub = voice.subtitle(vframes / 25.0)
            if sub is not None:
                st["subtitle"] = sub
            if alert_color_override is not None and alert:
                st["alert_color"] = alert_color_override
            elif not alert:
                alert_color_override = None
            st.update(part="กลางคืน", part_color=(70, 120, 230), title="ลาดตระเวน + แจ้ง รปภ.",
                      step=f"โซน {names[0]}  •  จุดถัดไป: {nxt_cp}")
            if PA.scenario == 6:
                st["title"] = "ลาดตระเวน • พูดเตือน • ช่วยเหลือฉุกเฉิน"
                if ev6["phase"] in (1, 2):
                    st["step"] = "ไฟส่อง + ไฟเตือนกะพริบ • พูดเตือนไทย/อังกฤษ • รปภ. คุยผ่านลำโพงหุ่น"
                elif ev6["fall"] in (1, 2):
                    st["step"] = "เหตุฉุกเฉิน: วิ่งล้อนำชุดปฐมพยาบาล (2 กก.) ไปถึงก่อน รปภ."
            if PA.scenario == 5:
                st5 = swap["state"]
                label = {"RETURN_TO_BASE": "กลับป้อม", "WAIT_SWAP": "รอสลับแบต", "SWAP": "กำลังสลับแบต",
                         "SELF_CHECK": "ตรวจตัวเอง", "STANDBY": "พร้อม (สแตนด์บาย)"}[st5]
                a_b = 100.0 if st5 in ("SELF_CHECK", "STANDBY") or (st5 == "SWAP" and swap["pack_new"] == "robot") \
                    else 0.0 if st5 == "SWAP" and swap["pack_old"] == "guard" else max(17.0 - t * 0.03, 14.0)
                b_s = "รอที่ป้อม" if not swap["released"] else "ลาดตระเวน (รับช่วงต่อ)"
                checks = ""
                if st5 == "SELF_CHECK":
                    ts = t - swap["t_state"]
                    checks = "  ".join(f"{n} {'✓' if ts > j + 1 else '…'}" for j, n in enumerate(("แบต", "ขา", "ล้อ", "กล้อง")))
                step = {"RETURN_TO_BASE": "แบตต่ำกว่าเกณฑ์ (พลังงานกลับป้อม + เผื่อ 15%) → ส่งต่องานให้หุ่น B แล้วกลับป้อม",
                        "WAIT_SWAP": f"แจ้ง รปภ. แล้ว • รอ {min(swap['clock_skip'], swap['delay']) / 60:.1f} นาที (เร่งเวลา)",
                        "SWAP": "รปภ. ถอดแบตเก่า → ใส่แบตที่ชาร์จเต็ม (≈1 นาที)",
                        "SELF_CHECK": "ตรวจตัวเองก่อนพร้อมใช้งาน:  " + checks,
                        "STANDBY": "แบตเก่าเข้าแท่นชาร์จ ~1.5 ชม. • หุ่น A รอรับกะถัดไป"}[st5]
                charging = "1 ก้อน (เหลือ ~1:30)" if st5 == "STANDBY" else "—"
                st.update(alert_color=st.get("alert_color", (215, 45, 45, 235)), title="สลับแบตที่ป้อม • 2 ตัวสลับกะ",
                          part="กลางคืน • สลับแบต", step=step,
                          robots=[("หุ่น A", a_b, label, (235, 80, 60) if a_b < 30 else (80, 200, 120)),
                                  ("หุ่น B", 100.0 - t * 0.01, b_s, (80, 200, 120)),
                                  ("แบตสำรอง", 66.0 if st5 != "STANDBY" else 33.0,
                                   f"พร้อม {2 if st5 not in ('SWAP', 'SELF_CHECK', 'STANDBY') else 1} ก้อน • ชาร์จ {charging}",
                                   (90, 160, 230))])
            writer.append_data(hud.draw_clean(frame, st))
        if k_step % 500 == 0:
            print(f"[PATROL] t={t:6.1f}s  done {done.sum()}/{N}  ok {success.sum()}  "
                  f"env0 wp {idx[0]}/{len(WP)} mode {mode[0]} zone {names[0]}  wall {time.time() - t_wall:.0f}s",
                  flush=True)

    for e in np.nonzero(~done)[0]:
        reason[e] = "หมดเวลา"
        fall_pos[e] = xy[e].tolist()
    if writer is not None:
        writer.close()
        if voice.events:  # lay the robot voice + intercom lines onto the video
            vp = os.path.join(PA.out, f"scenario{PA.scenario}.mp4")
            tmp = vp.replace(".mp4", "_silent.mp4")
            os.replace(vp, tmp)
            voice.mux(tmp, vp)
            json.dump([(round(a, 2), k) for a, k in voice.events], open(vp.replace(".mp4", "_voice.json"), "w"))

    # ---- results
    laps = []
    for e in range(N):
        laps.append(dict(
            env=e, success=bool(success[e]), reason=reason[e], fall_pos=fall_pos[e],
            fell=reason[e].startswith("ล้ม"), lap_time_s=None if np.isnan(t_finish[e]) else float(t_finish[e]),
            distance_m=float(dist[e]), mode_switches=int(switches[e]),
            missed_checkpoints=[cp_names[k] for k in range(K) if not cp_hit[e, k]],
            energy_abs_Wh=float(e_abs[e] / 3600), energy_pos_Wh=float(e_pos[e] / 3600),
            mu_scale=float(mu_scale[e]), wet=bool(wet[e]), wet_mu=float(wet_mu[e]) if wet[e] else None,
            wp_time=[None if np.isnan(v) else round(float(v), 2) for v in wp_t[e]],
            wp_energy_Wh=[None if np.isnan(v) else round(float(v) / 3600, 5) for v in wp_e[e]],
            obstacles=[(obs_kind[k], [round(float(v), 2) for v in obs_xy[e, k]]) for k in range(len(obstacles))
                       if obs_xy[e, k, 0] < 900],
        ))
    out = dict(scenario=PA.scenario, num_laps=N, events=events, laps=laps,
               assumptions=dict(battery_wh=PA.battery_wh, base_power_w=PA.base_power_w, drive_eff=PA.drive_eff))
    with open(os.path.join(PA.out, f"results_scenario{PA.scenario}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    ok = [l for l in laps if l["success"]]
    print(f"[RESULT] scenario {PA.scenario}: success {len(ok)}/{N}", flush=True)
    if ok:
        print("[RESULT] lap time mean %.1f s, distance %.1f m, switches %.1f, E_pos %.2f Wh, E_abs %.2f Wh" % (
            np.mean([l["lap_time_s"] for l in ok]), np.mean([l["distance_m"] for l in ok]),
            np.mean([l["mode_switches"] for l in ok]), np.mean([l["energy_pos_Wh"] for l in ok]),
            np.mean([l["energy_abs_Wh"] for l in ok])), flush=True)
    for l in laps:
        if not l["success"]:
            print(f"[RESULT] env {l['env']}: {l['reason']} at {l['fall_pos']}", flush=True)


play_rsl_rl.run_playback = patrol
if __name__ == "__main__":
    play_rsl_rl.run(REST + ["--task", TASK])
