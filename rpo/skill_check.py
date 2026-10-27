"""Check every RPO skill on the resort terrain in Isaac Lab + Newton (plain GL video, no HUD).

Lobby: walk -> wave -> walk/turn -> dance1 -> bumped over -> getup -> walk.  Then the stairs down to the pool.

Usage (from ~/IsaacLab):
  uv run --extra video --with onnxruntime python ~/rpo_tools/resort/skill_check.py --out <dir> --viz newton
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import warp as wp

wp.config.enable_backward = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import rpo_env_cfg as rec  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--push", type=float, default=600.0, help="bump force [N], 0.2 s")
    p.add_argument("--part", default="all", choices=["all", "lobby", "stairs", "getup"])
    p.add_argument("--plane", action="store_true", help="flat ground plane instead of the resort mesh")
    add_launcher_args(p)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.argv = sys.argv[:1]
    env_cfg, _ = resolve_task_config(rec.TASK, None, play_mode=True)
    if args.plane:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True, visible_env_indices=[0],
                                                         randomly_sample_visible_envs=False, window_width=1280,
                                                         window_height=720, enable_markers=False)]
    with launch_simulation(env_cfg, args):
        env = gym.make(rec.TASK, cfg=env_cfg).unwrapped
        env.reset()
        from isaaclab.sim import SimulationContext

        from rpo_skills import SkillRunner

        viz = [v for v in SimulationContext.instance().visualizers if hasattr(v, "render_rgb_array")][0]
        robot = env.scene["robot"]
        sk = SkillRunner(env)
        offset = env.scene.env_origins[0].cpu().numpy().astype(np.float64) - np.array([*rec.SCENE.dock[:2], 0])

        import imageio.v2 as imageio

        writer = imageio.get_writer(os.path.join(args.out, f"skill_check_{args.part}.mp4"), fps=25, quality=8,
                                    macro_block_size=8)
        log = []

        def teleport(x, y, hd, z):
            pose = robot.data.root_link_pose_w.torch.clone()
            pose[:, :3] = torch.tensor(np.array([x, y, z + 0.76]) + offset, device=env.device)
            pose[:, 3:] = torch.tensor([0, 0, math.sin(hd / 2), math.cos(hd / 2)], device=env.device)
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device))
            for _ in range(50):
                sk.walk(0, 0, 0)

        k = [0]

        def frame():
            k[0] += 1
            if k[0] % 2:
                return
            rp = robot.data.root_pos_w.torch[0].cpu().numpy()
            eye = rp + np.array([2.4, -2.4, 0.9])
            viz.set_camera_view(tuple(eye), tuple(rp + np.array([0, 0, -0.2])))
            writer.append_data(viz.render_rgb_array())

        def seg(name, n, fn):
            zmin, cmin = 9.0, 1.0
            p0 = robot.data.root_pos_w.torch[0, :2].cpu().numpy().copy()
            for i in range(n):
                fn(i)
                frame()
                zmin = min(zmin, float(robot.data.root_pos_w.torch[0, 2]) - offset[2])
                cmin = min(cmin, sk.upright())
            p1 = robot.data.root_pos_w.torch[0, :2].cpu().numpy()
            z = float(robot.data.root_pos_w.torch[0, 2]) - offset[2]
            line = (f"{name:22s} xy {(p1 - offset[:2]).round(2)} {n / 50:5.1f}s  moved {np.linalg.norm(p1 - p0):5.2f} m  end z {z:5.2f}  min z {zmin:5.2f}"
                    f"  min upright {cmin:5.2f}  end upright {sk.upright():4.2f}")
            print("[CHECK]", line, flush=True)
            log.append(line)

        def motion(name):
            sk.start(name)
            n = int(sk.motion_len_s(name) * 50) + 25
            seg(name, n, lambda i: sk.step())

        if args.part in ("all", "lobby"):
            teleport(14.0, 2.0, 0.0, 0.0)  # lobby floor
            seg("walk 0.5", 200, lambda i: sk.walk(0.5, 0, 0))
            seg("stand", 50, lambda i: sk.walk(0, 0, 0))
            motion("wave")
            seg("walk+turn", 150, lambda i: sk.walk(0.4, 0, 0.5))
            seg("stand", 50, lambda i: sk.walk(0, 0, 0))
            motion("dance1")
            seg("stand", 50, lambda i: sk.walk(0, 0, 0))

            def bump(i):  # from the front: the robot falls on its back (face-down get-up fails in Newton)
                if i == 0:
                    q = robot.data.root_quat_w.torch[0].cpu().numpy()
                    bump.h = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
                sk.push([-args.push * math.cos(bump.h), -args.push * math.sin(bump.h), 0] if i < 10 else None)
                sk.hold()

            seg("bumped (hold)", 150, bump)
            motion("getup")
            seg("walk 0.5", 150, lambda i: sk.walk(0.5, 0, 0))
        if args.part == "getup":
            for d in ([-1, 0, 0], [1, 0, 0], [0, 1, 0]):
                teleport(14.0, 2.0, 0.0, 0.0) if not args.plane else teleport(0.0, 0.0, 0.0, -offset[2])

                def bump(i, d=d):
                    sk.push([args.push * c for c in d] if i < 10 else None)
                    sk.hold()

                seg(f"bumped {d}", 150, bump)
                motion("getup")
                seg("walk 0.5", 100, lambda i: sk.walk(0.5, 0, 0))
        if args.part in ("all", "stairs"):
            teleport(58.5, 3.0, -math.pi / 2, 0.0)  # top of the stairs to the pool deck, facing down
            seg("stairs down 0.4", 800, lambda i: sk.walk(0.4, 0, 0))
            teleport(58.5, -3.0, math.pi / 2, -1.2)  # bottom, facing up
            seg("stairs up 0.4", 800, lambda i: sk.walk(0.4, 0, 0))
        writer.close()
        open(os.path.join(args.out, f"skill_check_{args.part}.txt"), "w").write("\n".join(log) + "\n")
        env.close()


if __name__ == "__main__":
    main()
