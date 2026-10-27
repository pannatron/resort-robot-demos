"""Shared stage for the RPO resort jobs: env + skills + resort look + HUD + voice + people + carry tray.

The job scripts (rpo_jobs.py, rpo_party.py) build a Stage, then loop: decide a skill/command, ``st.tick()``,
place people/props, ``st.shot(eye, at)``, ``st.write(hud_state)``.

Rendering reuses the Dobot resort code in ~/dobot_tools/resort (scene, night/day look, sky, OVRTX lights,
animated people, Thai HUD); nothing there is modified.
"""

from __future__ import annotations

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

import rpo_env_cfg as rec  # noqa: E402  (puts the Dobot resort folder and rpo_lab on sys.path)
import incidents  # noqa: E402
import resort_scene as rs  # noqa: E402
from common import H, W, Hud, SmoothCam, _T, wrap, yaw_of  # noqa: E402

POLICY_DT = 0.02
VOICE_DIR = os.path.join(HERE, "assets", "voice")
# key -> (speaker, subtitle). Robot voice = edge-tts th-TH-PremwadeeNeural / en-US-JennyNeural (make_voice.py)
LINES = {
    "greet_th": ("robot", "สวัสดีค่ะ ยินดีต้อนรับสู่รีสอร์ตค่ะ"),
    "greet_en": ("robot", "Hello, and welcome to the resort!"),
    "guide_go_th": ("robot", "เชิญทางนี้เลยค่ะ หนูจะพาไปห้อง 306 นะคะ"),
    "guide_pool_th": ("robot", "ทางขวามือเป็นสระว่ายน้ำ เปิดถึงสองทุ่มค่ะ"),
    "guide_arrive_th": ("robot", "ถึงห้อง 306 แล้วค่ะ ขอให้พักผ่อนอย่างมีความสุขนะคะ"),
    "dlv_pickup_th": ("robot", "รับของแล้วค่ะ ออกเดินทางไปวิลล่า V2"),
    "dlv_arrive_th": ("robot", "ของมาส่งแล้วค่ะ หยิบจากถาดด้านหน้าได้เลยค่ะ"),
    "dlv_thanks_th": ("robot", "ขอบคุณค่ะ มีอะไรเรียกใช้ได้ตลอดนะคะ"),
    "bump_sorry_th": ("robot", "ไม่เป็นไรค่ะ หนูลุกเองได้"),
    "bump_ok_th": ("robot", "น้องไม่เจ็บใช่ไหมคะ ระวังพื้นลื่นนะคะ"),
    "party_arrive_th": ("robot", "สุขสันต์วันเกิดนะคะ เค้กวันเกิดมาส่งแล้วค่ะ"),
    "party_wish_th": ("robot", "อธิษฐาน แล้วเป่าเทียนได้เลยค่ะ"),
    "party_dance_th": ("robot", "ขอโชว์เต้นฉลองให้หน่อยนะคะ"),
    "party_dance_en": ("robot", "Let's party!"),
    "party_finale_th": ("robot", "ขอให้มีความสุขมาก ๆ นะคะ"),
    "party_finale_en": ("robot", "Happy birthday!"),
    "music_hbd": ("music", None),
    "music_dance": ("music", None),
}
incidents.VOICE_DIR, incidents.LINES = VOICE_DIR, LINES  # VoiceTrack reads these module globals

SKILL_PILL = {"walk": ("ทักษะ: เดิน", (40, 120, 220, 210)), "wave": ("ทักษะ: โบกมือ", (40, 160, 90, 225)),
              "dance0": ("ทักษะ: เต้น", (200, 60, 160, 225)), "dance1": ("ทักษะ: เต้น", (200, 60, 160, 225)),
              "getup": ("ทักษะ: ลุกขึ้นเอง", (230, 120, 40, 230)), "hold": ("ล้ม → ล็อกข้อต่อ", (210, 60, 50, 230))}


def _patch_hud():
    """Same overlay as the Dobot demo, with the RPO skill in the mode pill and an RPO watermark."""
    import inspect

    import common

    src = inspect.getsource(common._clean_draw)
    src = src.replace('mtxt = "โหมดเดิน • ขึ้นลงบันได" if walk else "โหมดล้อ"',
                      'mtxt, _mc = SKILL_PILL.get(st.get("skill", "walk"), SKILL_PILL["walk"])')
    src = src.replace('fill=(240, 160, 40, 225) if walk or st.get("switching")\n                            else (40, 120, 220, 200))',
                      'fill=_mc)')
    src = src.replace('"DOBOT Wheel-Leg  •  ภาพจำลอง"', '"RPO Humanoid (Roboparty)  •  ภาพจำลอง"')
    assert "SKILL_PILL" in src and "_mc)" in src and "RPO Humanoid" in src
    ns = dict(vars(common))
    ns["SKILL_PILL"] = SKILL_PILL
    exec(compile(src, common.__file__, "exec"), ns)
    common.Hud.draw_clean = ns["_clean_draw"]


_patch_hud()

# chest tray, in the torso_link frame (arms hang at |y| >= 0.16, chest front at x = 0.05)
TRAY_X, TRAY_Z = 0.19, 0.06
TRAY_SPECS = [("floor", "cuboid", (0.26, 0.24, 0.012), (0.55, 0.40, 0.25)),
              ("wf", "cuboid", (0.012, 0.24, 0.10), (0.62, 0.46, 0.28)),
              ("wl", "cuboid", (0.26, 0.012, 0.10), (0.62, 0.46, 0.28)),
              ("wr", "cuboid", (0.26, 0.012, 0.10), (0.62, 0.46, 0.28)),
              ("strap", "cuboid", (0.05, 0.30, 0.02), (0.12, 0.13, 0.16))]


def tray_parts(M):
    """Tray (an open box hung on the chest) at torso pose M (4x4, design frame)."""
    import trimesh

    def box(e):
        return trimesh.creation.box(extents=e)

    o = _T(None, (TRAY_X, 0.0, TRAY_Z))
    P = [("floor", box((0.26, 0.24, 0.012)), (0.55, 0.40, 0.25), o @ _T(None, (0, 0, -0.05))),
         ("wf", box((0.012, 0.24, 0.10)), (0.62, 0.46, 0.28), o @ _T(None, (0.125, 0, 0))),
         ("wl", box((0.26, 0.012, 0.10)), (0.62, 0.46, 0.28), o @ _T(None, (0, 0.115, 0))),
         ("wr", box((0.26, 0.012, 0.10)), (0.62, 0.46, 0.28), o @ _T(None, (0, -0.115, 0))),
         ("strap", box((0.05, 0.30, 0.02)), (0.12, 0.13, 0.16), _T(None, (0.07, 0.0, 0.17)))]
    return [(t, m, c, M @ T) for t, m, c, T in P]


def tray_top(M):
    """Pose of the tray floor (where items sit), design frame."""
    return M @ _T(None, (TRAY_X, 0.0, TRAY_Z - 0.044))


class Stage:
    def __init__(self, args, *, look="day", actors=None, payload_kg=0.0, name="job"):
        """args: parsed (with add_launcher_args); look: day / morning / night; actors: rtx name -> specs."""
        from isaaclab.app import launch_simulation
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        from isaaclab_tasks.core.velocity import mdp
        from isaaclab_tasks.utils import resolve_task_config

        self.args, self.look, self.name = args, look, name
        self.rtx = bool(getattr(args, "rtx", 0))
        self.actors = dict(actors or {})
        self.payload_kg = payload_kg
        os.makedirs(args.out, exist_ok=True)
        argv, sys.argv = sys.argv, sys.argv[:1]
        env_cfg, _ = resolve_task_config(rec.TASK, None, play_mode=True)
        sys.argv = argv
        if self.rtx:
            from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

            viz_cfg = NewtonRTXVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                             window_width=W, window_height=H, enable_markers=False,
                                             rtx_environment="none", background_color=(1.0, 0.0, 1.0))
            import rtx_actors as ra

            ra.add_actor_objects(env_cfg, self.actors)
        else:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

            viz_cfg = NewtonGLVisualizerCfg(headless=True, visible_env_indices=[0], randomly_sample_visible_envs=False,
                                            window_width=W, window_height=H, enable_markers=False)
        env_cfg.sim.visualizer_cfgs = [viz_cfg]
        if payload_kg > 0:  # tray + load, added to the torso
            env_cfg.events.add_base_mass = EventTermCfg(
                func=mdp.randomize_rigid_body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                        "mass_distribution_params": (payload_kg, payload_kg), "operation": "add"})
        self._ctx = launch_simulation(env_cfg, args)
        self._ctx.__enter__()
        self.env = gym.make(rec.TASK, cfg=env_cfg).unwrapped
        self.env.reset()
        self._setup_render()

    # ------------------------------------------------------------------ setup
    def _setup_render(self):
        from isaaclab.sim import SimulationContext
        from isaaclab_newton.physics.newton_manager import NewtonManager

        import newton

        from night_render import NightScene, Sky, setup_day, setup_night, setup_rtx_lights
        from rpo_skills import SkillRunner

        env = self.env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.sk = SkillRunner(env)
        self.scene = rec.SCENE
        self.offset = env.scene.env_origins[0].cpu().numpy().astype(np.float64) - np.array(
            [self.scene.dock[0], self.scene.dock[1], 0])
        self.viz = [v for v in SimulationContext.instance().visualizers if hasattr(v, "render_rgb_array")][0]
        viewer = self.viewer = self.viz._viewer
        model = NewtonManager.get_model()
        if not self.rtx:
            setup_night(viewer) if self.look == "night" else setup_day(viewer, morning=self.look == "morning")
        body, stype = model.shape_body.numpy(), model.shape_type.numpy()
        terrain = np.nonzero((body == -1) & (stype == int(newton.GeoType.MESH)))[0]
        op = model.shape_opacity.numpy()
        op[terrain] = 0.0
        model.shape_opacity.assign(op)
        flags = model.shape_flags.numpy()  # not drawn at all (RTX renders a zero-opacity mesh as black)
        flags[terrain] &= ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(flags)
        self.driver = None
        if self.rtx:
            import rtx_actors as ra

            self.driver = ra.ActorDriver(env, self.actors, self.offset, model)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.night = NightScene(self.scene, self.offset, device=str(self.device))
        self.night.log_static(viewer, lamps_on=self.look == "night" and not self.rtx)
        self.extra_static()
        if self.rtx:
            setup_rtx_lights(viewer, self.look, self.scene.lamps, self.offset)
            self.sky = Sky(viewer, self.look, rtx=True)
        else:
            self.sky = Sky(viewer, self.look, viewer.renderer._sun_direction)
        self.hud = Hud(self.scene)
        self.scam = SmoothCam()
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        import imageio.v2 as imageio

        self.vpath = os.path.join(self.args.out, f"{self.name}.mp4")
        self.writer = imageio.get_writer(self.vpath, fps=25, quality=8, macro_block_size=8)
        self.voice = incidents.VoiceTrack()
        self.events, self.ev_lines = [], []
        self.clock0 = dt_.datetime(2026, 10, 24, 9, 0, 0)
        self.t, self.k, self.vtime = 0.0, 0, 0.0
        self.vx_cmd = 0.0
        self.dist, self.e_pos, self.prev_xy = 0.0, 0.0, None
        self.falls, self.getups = 0, 0
        self.min_upright = 1.0
        self._last_parts = {}

    def extra_static(self):
        """Hook for jobs that log extra static meshes before the first frame (RTX builds its scene then)."""

    # ------------------------------------------------------------------ robot state
    def pose(self):
        rp = self.robot.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)
        q = self.robot.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
        return rp, q, rp - self.offset, float(yaw_of(q[None])[0])

    def torso_M(self):
        """torso_link pose as a 4x4 in the design frame."""
        import trimesh

        p = self.robot.data.body_pos_w.torch[0, self.torso_id].cpu().numpy().astype(np.float64) - self.offset
        x, y, z, w = self.robot.data.body_quat_w.torch[0, self.torso_id].cpu().numpy().astype(np.float64)
        M = trimesh.transformations.quaternion_matrix([w, x, y, z])
        M[:3, 3] = p
        return M

    def ground_z(self, x, y):
        return self.night.ground_z(x, y)

    def teleport(self, x, y, hd, settle=50):
        robot, dev = self.robot, self.device
        pose = robot.data.root_link_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(np.array([x, y, self.ground_z(x, y) + 0.77]) + self.offset, device=dev)
        pose[:, 3:] = torch.tensor([0, 0, math.sin(hd / 2), math.cos(hd / 2)], device=dev)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
        for _ in range(settle):
            self.sk.walk(0, 0, 0)
        self.prev_xy = None
        self.vx_cmd = 0.0

    # ------------------------------------------------------------------ control
    def nav(self, xy, yaw, tgt, spd, look=None, vmax=0.6):
        """Walking command (vx, wz) towards tgt at speed spd; face `look` when standing."""
        d_t = np.asarray(tgt, float) - xy
        far = np.linalg.norm(d_t) > 0.05
        e_head = wrap(math.atan2(d_t[1], d_t[0]) - yaw) if far else 0.0
        if look is not None and spd < 0.05:
            e_use = wrap(math.atan2(look[1] - xy[1], look[0] - xy[0]) - yaw)
        else:
            e_use = e_head
        wz = float(np.clip(1.6 * e_use, -0.8, 0.8))
        vx = min(spd, vmax) * max(math.cos(e_head), 0.0) ** 3 if spd > 0 else 0.0
        if abs(e_head) > 0.7:
            vx = min(vx, 0.1)
        self.vx_cmd = float(np.clip(vx, self.vx_cmd - 0.6 * POLICY_DT, self.vx_cmd + 0.6 * POLICY_DT))
        if abs(wz) < 0.06:
            wz = 0.0
        return self.vx_cmd, wz

    def tick(self, skill="walk", cmd=(0.0, 0.0, 0.0)):
        """One 50 Hz step. skill: walk (cmd = vx, vy, wz), motion (continue the started clip), hold."""
        if skill == "walk":
            self.sk.walk(*cmd)
        elif skill == "motion":
            self.sk.step()
        else:
            self.sk.hold()
        tau, qd = self.robot.data.applied_torque.torch[0], self.robot.data.joint_vel.torch[0]
        self.e_pos += float((tau * qd).clamp(min=0).sum()) * POLICY_DT
        _, _, p, _ = self.pose()
        if self.prev_xy is not None:
            self.dist += float(np.linalg.norm(p[:2] - self.prev_xy))
        self.prev_xy = p[:2].copy()
        self.min_upright = min(self.min_upright, self.sk.upright())
        self.t += POLICY_DT
        self.k += 1
        frame_due = self.k % 2 == 0
        if frame_due:
            self.vtime += 2 * POLICY_DT
        return frame_due

    # ------------------------------------------------------------------ story
    def clock(self, t=None):
        return (self.clock0 + dt_.timedelta(seconds=self.t if t is None else t)).strftime("%H:%M:%S")

    def post(self, etype, msg):
        self.events.append({"time": (self.clock0 + dt_.timedelta(seconds=self.t)).isoformat(), "type": etype,
                            "message": msg, "video_s": round(self.vtime, 2)})
        self.ev_lines.append(f"{self.clock()}  {msg}")
        print(f"[EVENT] t={self.t:.1f}s {msg}", flush=True)

    def say(self, key, delay=0.2):
        return self.voice.say(self.vtime + delay, key)

    def set_parts(self, name, parts):
        self._last_parts[name] = parts
        if self.driver is not None:
            self.driver.set(name, parts)
        else:
            self.night.log_parts(self.viewer, name, parts)

    def hide(self, name):
        if self.driver is not None:
            self.driver.park(name)
        else:
            self.night.log_parts(self.viewer, name, [(t, m, c, _T(None, (0, 0, -40.0)) @ T) for t, m, c, T in
                                                     self._last_parts.get(name, [])])

    # ------------------------------------------------------------------ video
    def shot(self, eye, at, cut=False, fov=45.0, post=None):
        """Render one frame from eye -> at (world frame). post(frame) may add image-space effects."""
        self.viewer.camera.fov = fov
        self.viz.set_camera_view(*self.scam(eye, at, cut=cut))
        img = self.viz.render_rgb_array()
        frame = self.sky.composite(self.viewer, img)
        if self.look == "night":
            frame = self.night.add_glow(self.viewer, frame)
        return post(frame) if post is not None else frame

    def write(self, frame, st):
        st = dict(st)
        st.setdefault("clock", self.clock())
        st.setdefault("mode", "walk")
        st.setdefault("skill", self.sk.skill if self.sk.skill != "run" else "walk")
        st.setdefault("switching", False)
        st["subtitle"] = self.voice.subtitle(self.vtime)
        self.writer.append_data(self.hud.draw_clean(frame, st))

    def finish(self, extra=None):
        self.writer.close()
        if self.voice.events:
            tmp = self.vpath.replace(".mp4", "_silent.mp4")
            os.replace(self.vpath, tmp)
            self.voice.mux(tmp, self.vpath)
            json.dump([(round(a, 2), k) for a, k in self.voice.events],
                      open(self.vpath.replace(".mp4", "_voice.json"), "w"))
        res = dict(job=self.name, time_s=round(self.t, 2), video_s=round(self.vtime, 2), distance_m=round(self.dist, 2),
                   energy_pos_Wh=round(self.e_pos / 3600, 3), payload_kg=self.payload_kg, falls=self.falls,
                   getups=self.getups, min_upright=round(self.min_upright, 3), events=self.events)
        res.update(extra or {})
        json.dump(res, open(os.path.join(self.args.out, f"result_{self.name}.json"), "w"), ensure_ascii=False,
                  indent=1)
        print("[RESULT]", json.dumps(res, ensure_ascii=False), flush=True)
        return res

    def close(self):
        self.env.close()
        self._ctx.__exit__(None, None, None)
