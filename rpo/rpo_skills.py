"""Roboparty's shipped RPO policies, run inside an Isaac Lab env (one robot, env 0).

Skills (all ONNX, 50 Hz, same action: 23 joint offsets x 0.25 from the standing pose, POLICY_JOINTS order):
    walk     policy.onnx         10 frames x (ang_vel, gravity_b, cmd vx vy wz, dof_pos - default, dof_vel, action)
    run      policy_amp.onnx      3 frames, term-major, same terms (commands up to 2 m/s)
    wave / dance0 / dance1 / getup    BeyondMimic trackers: motion joint pos/vel at frame i, then
                                      ang_vel, gravity_b, dof_pos - default, dof_vel, action (121 values)

Verified first in MuJoCo (sim2sim, scratchpad skilltest/chain_test.py): walk -> wave -> walk -> dance -> walk ->
pushed over (front / back / side) -> getup -> walk stays up. Motion clips start standing, getup starts lying on
the back; the get-up policy also recovers from a face-down fall (it rolls over first).

    sk = SkillRunner(env)
    sk.walk(vx, vy, wz)          # one 50 Hz step with the walking policy
    sk.start("wave")             # then sk.step() until sk.done
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import torch

from rpo_lab.robot import POLICY_JOINTS

INF = os.path.expanduser("~/Downloads/Simulation_for_Humanoid/Extra_RPO_Walk/roboto_origin/modules/"
                         "roboparty_deploy/src/inference/robots/rpo")
MOTIONS = ("wave", "dance0", "dance1", "getup")
MOTION_FPS = 50
CLIP_CMD = {"walk": ((-0.4, 0.6), (-0.4, 0.4), (-0.8, 0.8)), "run": ((-0.5, 2.0), (-0.6, 0.6), (-1.57, 1.57))}


def _sess(name):
    return ort.InferenceSession(os.path.join(INF, "models", name), providers=["CPUExecutionProvider"])


class SkillRunner:
    def __init__(self, env, skills=("walk", "wave", "dance0", "dance1", "getup")):
        self.env = env
        self.uenv = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot = self.uenv.scene["robot"]
        self.device = self.uenv.device
        ids, names = self.robot.find_joints(POLICY_JOINTS, preserve_order=True)
        assert list(names) == POLICY_JOINTS, names
        self.ids = torch.tensor(ids, device=self.device)
        self.default = self.robot.data.default_joint_pos.torch[0, self.ids].cpu().numpy().astype(np.float64)
        files = {"walk": "policy.onnx", "run": "policy_amp.onnx"}
        self.sess = {s: _sess(files.get(s, f"policy_{s}.onnx")) for s in skills}
        self.motion = {s: np.load(os.path.join(INF, "motions", f"{s}.npz")) for s in skills if s in MOTIONS}
        self.action = np.zeros(23)
        self.hist = None
        self.skill, self.frame = "walk", 0
        self.cmd = np.zeros(3)

    # ------------------------------------------------------------------ state
    def proprio(self):
        d = self.robot.data
        w = d.root_ang_vel_b.torch[0].cpu().numpy().astype(np.float64)
        g = d.projected_gravity_b.torch[0].cpu().numpy().astype(np.float64)
        q = d.joint_pos.torch[0, self.ids].cpu().numpy().astype(np.float64)
        dq = d.joint_vel.torch[0, self.ids].cpu().numpy().astype(np.float64)
        return w, g, q - self.default, dq

    def upright(self):
        """cos of the tilt angle: 1 standing, < 0.5 means more than 60 degrees over."""
        return float(-self.robot.data.projected_gravity_b.torch[0, 2])

    # ------------------------------------------------------------------ skills
    def start(self, skill, t0=0.0, t1=None, blend_out=1.2):
        """Play clip seconds [t0, t1]; when cut short, blend the reference to the clip's (standing) last frame."""
        assert skill in self.motion, skill
        m = self.motion[skill]
        n = len(m["joint_pos"])
        f0 = int(t0 * MOTION_FPS)
        f1 = n - 1 if t1 is None else min(int(t1 * MOTION_FPS), n - 1)
        pos, vel = m["joint_pos"][f0:f1 + 1], m["joint_vel"][f0:f1 + 1]
        if f1 < n - 1:
            nb = int(blend_out * MOTION_FPS)
            s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nb + 1)[1:])[:, None]  # smooth ease in/out
            bp = pos[-1] + s * (m["joint_pos"][-1] - pos[-1])
            bv = np.gradient(bp, 1.0 / MOTION_FPS, axis=0)
            pos, vel = np.concatenate([pos, bp]), np.concatenate([vel, bv])
        self.ref_pos, self.ref_vel = pos, vel
        self.skill, self.frame = skill, 0

    def motion_len_s(self, skill, t0=0.0, t1=None, blend_out=1.2):
        n = len(self.motion[skill]["joint_pos"]) / MOTION_FPS
        return n - t0 if t1 is None or t1 >= n else (t1 - t0) + blend_out

    @property
    def done(self):
        if self.skill in self.motion:
            return self.frame >= len(self.ref_pos) + 25  # +0.5 s settle on the last frame
        return True

    @property
    def clip_t(self):
        """Seconds into the current motion reference."""
        return self.frame / MOTION_FPS

    def walk(self, vx=0.0, vy=0.0, wz=0.0, run=False):
        """One policy step of the walking (or running) policy."""
        s = "run" if run else "walk"
        lim = CLIP_CMD[s]
        self.cmd = np.array([np.clip(vx, *lim[0]), np.clip(vy, *lim[1]), np.clip(wz, *lim[2])])
        if self.skill != s:  # coming back from a motion or the other gait: fresh history
            self.skill, self.hist = s, None
        w, g, dp, dq = self.proprio()
        terms = [w, g, self.cmd, dp, dq, self.action]
        if s == "walk":
            o = np.concatenate(terms)
            self.hist = np.tile(o, (10, 1)) if self.hist is None else np.vstack([self.hist[1:], o])
            obs = self.hist.reshape(1, -1)
        else:  # term-major history of 3
            self.hist = [[x] * 3 for x in terms] if self.hist is None else [h[1:] + [x] for h, x in zip(self.hist, terms)]
            obs = np.concatenate([np.concatenate(h) for h in self.hist])[None]
        return self._act(s, obs)

    def step(self):
        """One policy step of the current motion skill (wave / dance / getup)."""
        assert self.skill in self.motion, f"step() needs a started motion clip, current skill is {self.skill}"
        i = min(self.frame, len(self.ref_pos) - 1)
        w, g, dp, dq = self.proprio()
        vel = self.ref_vel[i] if self.frame < len(self.ref_pos) else 0.0 * self.ref_vel[i]
        obs = np.concatenate([self.ref_pos[i], vel, w, g, dp, dq, self.action])[None]
        self.frame += 1
        return self._act(self.skill, obs)

    def hold(self):
        """Hold the standing pose targets (motors stiff, no policy), e.g. while being knocked over."""
        self.action = np.zeros(23)
        self.hist = None
        self.skill = "hold"
        return self._send()

    def push(self, force_w=None):
        """Apply a world-frame force [N] at the pelvis until called again with None (a bump, a fall)."""
        if not hasattr(self, "_base_id"):
            self._base_id = self.robot.find_bodies("base_link")[0]
        f = torch.zeros(1, 1, 3, device=self.device)
        if force_w is not None:
            f[0, 0] = torch.tensor(force_w, dtype=torch.float32, device=self.device)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=torch.zeros_like(f), body_ids=self._base_id, is_global=True)

    def _act(self, s, obs):
        a = self.sess[s].run(None, {self.sess[s].get_inputs()[0].name: obs.astype(np.float32)})[0][0]
        self.action = a.astype(np.float64)
        return self._send()

    def _send(self):
        act = torch.tensor(self.action, dtype=torch.float32, device=self.device)[None]
        with torch.inference_mode():
            return self.env.step(act)
