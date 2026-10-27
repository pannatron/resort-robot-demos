"""Birthday surprise on the pool deck (19:30) with the RPO humanoid (simulation only).

The robot walks in with the cake on its chest tray (walking policy, cake + tray added to the torso mass), wishes
happy birthday, the guest takes the cake, the robot plays the song, the guest blows the candles, then the robot
dances to the music (Roboparty's dance1 clip + part of dance0) and finishes with a two-arm wave (wave clip).

Usage (from ~/IsaacLab):
  uv run --extra video --with onnxruntime python ~/rpo_tools/resort/rpo_party.py --out <dir> --viz newton [--rtx 1]
"""

from __future__ import annotations

import argparse
import datetime as dt_
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpo_stage as S  # noqa: E402

import numpy as np  # noqa: E402

import party_fx as fx  # noqa: E402
from common import guest_parts, place, wrap  # noqa: E402

Z0 = fx.Z0
START = (60.6, -3.8)  # foot of the stairs from the corridor
STOP = (61.2, -11.0)  # hands the cake over here (tray 1.2 m from the birthday guest)
SPOT = (60.5, -11.0)  # dance spot
BDAY = (62.45, -11.0)
FACE_DANCE = 0.75  # dance heading: between the guests (east) and the camera (north)
GUESTS = [  # name, xy, clothes, skin  (as in the Dobot party)
    ("bday", BDAY, (0.92, 0.45, 0.62), (0.86, 0.68, 0.54)),
    ("g1", (62.25, -12.15), (0.30, 0.55, 0.85), (0.74, 0.55, 0.42)),
    ("g2", (61.25, -12.7), (0.95, 0.80, 0.30), (0.86, 0.68, 0.54)),
    ("g3", (60.1, -12.5), (0.35, 0.70, 0.50), (0.55, 0.38, 0.27)),
    ("g4", (59.35, -11.6), (0.88, 0.88, 0.90), (0.74, 0.55, 0.42)),
]
DANCES = [("dance1", 0.0, None), ("dance0", 2.0, 12.0)]  # (clip, from s, to s)
FINALE = ("wave", 8.0, 14.0)  # two-arm wave part of the wave clip


class PartyStage(S.Stage):
    def extra_static(self):
        meshes, self.bulbs = fx.decor_meshes()
        for i, (m, col) in enumerate(meshes):
            self.night._log(self.viewer, f"/party/decor{i}", m, col)
        if self.rtx:
            from pxr import Gf, UsdGeom, UsdLux

            from night_render import INTENSITY_SCALE

            for i, (p, col) in enumerate(self.bulbs[::2]):  # coloured party light on the deck
                lt = UsdLux.SphereLight.Define(self.viewer.stage, f"/root/PartyBulb{i}")
                lt.GetRadiusAttr().Set(0.05)
                lt.GetIntensityAttr().Set(INTENSITY_SCALE * 900.0)
                lt.GetColorAttr().Set(Gf.Vec3f(*(c / 255 for c in col)))
                UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(*(p + self.offset + [0, 0, -0.08]).tolist()))


def run_party(st):
    st.clock0 = dt_.datetime(2026, 10, 24, 19, 28, 0)
    st.teleport(*START, -math.pi / 2)
    route = [np.array(START), np.array((60.75, -9.4)), np.array(STOP)]
    idx = 1
    phase, phase_t = "arrive", 0.0
    cake_on_robot, lit = True, True
    guest_pose = {n: "idle" for n, *_ in GUESTS}
    guest_pos = {n: np.array([x, y, Z0]) for n, (x, y), *_ in GUESTS}
    guest_hd = {n: 0.0 for n, *_ in GUESTS}
    confetti = fx.Confetti()
    alert, alert_until = "แพ็กเกจเซอร์ไพรส์วันเกิด: เค้ก + เพลง + โชว์เต้น  •  ลานสระ 19:30", 5.0
    title, step_txt = "เซอร์ไพรส์วันเกิดที่ลานสระ", "หุ่นถือเค้กมาบนถาดหน้าอก"
    dance_i, music_t0 = 0, None
    st.post("order", "แพ็กเกจเซอร์ไพรส์วันเกิด: ส่งเค้ก + เปิดเพลง + โชว์เต้น ที่ลานสระ")
    dance_log = []
    while st.t < 200:
        rp, q, p, yaw = st.pose()
        xy = p[:2]
        skill, tgt, spd, look, back = "walk", xy, 0.0, None, False
        # ------------------------------------------------------------ story
        if phase == "arrive":
            if np.linalg.norm(xy - route[idx]) < 0.25:
                if idx + 1 < len(route):
                    idx += 1
                else:
                    phase, phase_t = "greet", st.t
                    st.post("arrived", "ถึงโต๊ะวันเกิด พูดอวยพร")
                    st.say("party_arrive_th", 0.3)
            tgt, spd, look = route[idx], (0.45 if idx == 1 else 0.3), np.array(BDAY)
            for n, *_ in GUESTS:  # everyone turns to the robot
                g = guest_pos[n]
                guest_hd[n] = math.atan2(xy[1] - g[1], xy[0] - g[0])
        elif phase == "greet":
            look = np.array(BDAY)
            if st.t - phase_t > 1.0:
                guest_pose = {n: ("clap" if n != "bday" else "idle") for n in guest_pose}
            if st.voice.done_by(st.vtime) and st.t - phase_t > 2.0:
                phase, phase_t = "take", st.t
                guest_pose["bday"] = "hold"
                cake_on_robot = False
                st.post("cake", "เจ้าของวันเกิดหยิบเค้กจากถาดหน้าอกหุ่น")
                alert, alert_until = "ร้องเพลงวันเกิด ♪", st.vtime + 4
                st.voice.play(st.vtime + 0.6, "music_hbd")
                step_txt = "เปิดเพลงวันเกิดจากลำโพงหุ่น"
        elif phase == "take":
            look = np.array(BDAY)
            if st.t - phase_t > st.voice.dur["music_hbd"] + 0.4:
                phase, phase_t = "wish", st.t
                st.say("party_wish_th")
                step_txt = "อธิษฐาน + เป่าเทียน"
                guest_pose = {n: ("idle" if n != "bday" else "hold") for n in guest_pose}
        elif phase == "wish":
            look = np.array(BDAY)
            if st.voice.done_by(st.vtime) and lit and st.t - phase_t > 2.0:
                lit = False
                st.post("candles", "เป่าเทียนแล้ว")
                confetti.burst((BDAY[0] - 0.2, BDAY[1] - 0.4, Z0 + 1.9), n=120, up=2.5, spread=1.0)
                guest_pose = {n: ("cheer" if n != "bday" else "hold") for n in guest_pose}
                phase_t = st.t
            if not lit and st.t - phase_t > 2.5:
                phase, phase_t = "back", st.t
                step_txt = "ถอยไปจุดเต้น"
        elif phase == "back":  # small steps backwards, still facing the guests
            back = np.linalg.norm(xy - np.array(SPOT)) > 0.1 and st.t - phase_t < 5.0
            if not back:
                phase, phase_t = "turn", st.t
        elif phase == "turn":
            if abs(wrap(FACE_DANCE - yaw)) < 0.08 or st.t - phase_t > 5.0:
                phase, phase_t = "intro", st.t
                st.say("party_dance_th")
                st.say("party_dance_en")
                step_txt = "โชว์เต้นฉลอง"
                guest_pose = {n: ("clap" if n != "bday" else "hold") for n in guest_pose}
        elif phase == "intro":
            if st.voice.done_by(st.vtime) and st.t - phase_t > 1.0:
                phase, phase_t = "dance", st.t
                music_t0 = st.voice.play(st.vtime + 0.1, "music_dance") - st.voice.dur["music_dance"]
                st.sk.start(*DANCES[0])
                skill = "motion"
                st.post("dance", f"เริ่มโชว์เต้น: {DANCES[0][0]} (BeyondMimic policy ของ Roboparty)")
        elif phase == "dance":
            skill = "motion"
            step_txt = f"โชว์เต้น ชุดที่ {dance_i + 1}/{len(DANCES)}"
            if st.sk.done:
                dance_log.append(dict(clip=DANCES[dance_i][0], upright_min=None))
                dance_i += 1
                if dance_i < len(DANCES):
                    st.sk.start(*DANCES[dance_i])
                    st.post("dance", f"ต่อด้วย {DANCES[dance_i][0]}")
                else:
                    phase, phase_t = "reface", st.t  # the dances turn the body: face the guests again
                    step_txt = "หันกลับหาแขก"
        elif phase == "reface":
            if abs(wrap(FACE_DANCE - yaw)) < 0.1 or st.t - phase_t > 4.0:
                phase, phase_t = "finale", st.t
                st.sk.start(*FINALE)
                skill = "motion"
                st.say("party_finale_th", 0.4)
                st.say("party_finale_en")
                confetti.burst((SPOT[0], SPOT[1], Z0 + 1.5), n=200)
                for g in ("g1", "g3", "g4"):
                    gp = guest_pos[g]
                    confetti.burst((gp[0], gp[1], Z0 + 1.8), n=90, up=3.0, spread=1.3)
                guest_pose = {n: ("cheer" if n != "bday" else "hold") for n in guest_pose}
                st.post("finale", "ท่าปิด: โบกสองมือ + โปรยกระดาษสี")
                step_txt = "ท่าปิด: โบกสองมือ"
        elif phase == "finale":
            skill = "motion" if not st.sk.done else "walk"
            if st.sk.done and phase_t >= 0:
                alert, alert_until = "สุขสันต์วันเกิด!  แพ็กเกจปาร์ตี้ = บริการเสริมของรีสอร์ต", st.vtime + 8
                st.post("done", "จบโชว์ ✓")
                phase, phase_t = "end", st.t
        else:
            if st.t - phase_t > 6.0 and st.voice.done_by(st.vtime):
                break
        # ------------------------------------------------------------ command
        cmd = (0.0, 0.0, 0.0)
        if skill == "walk":
            if back:
                cmd = (-0.25, 0.0, 0.0)
            elif phase in ("turn", "intro", "end", "reface"):
                face = FACE_DANCE if phase != "end" else yaw
                cmd = (0.0, 0.0, float(np.clip(1.5 * wrap(face - yaw), -0.6, 0.6)))
            else:
                vx, wz = st.nav(xy, yaw, tgt, spd, look)
                cmd = (vx, 0.0, wz)
        if st.sk.upright() < 0.5:
            st.falls += 1
            st.post("fall", "หุ่นล้มระหว่างปาร์ตี้")
            break
        confetti.step(S.POLICY_DT)
        if not st.tick(skill, cmd):
            continue
        # ------------------------------------------------------------ actors
        gphase = st.vtime * 2 * math.pi / fx.BEAT * 0.5
        bday_M = None
        for n, _, clothes, skin in GUESTS:
            gp = guest_pos[n]
            st.set_parts(n, place(guest_parts(guest_pose[n], gphase + hash(n) % 7, clothes, skin), gp[0], gp[1],
                                  gp[2], guest_hd[n]))
            if n == "bday":
                import trimesh

                bday_M = trimesh.transformations.rotation_matrix(guest_hd[n], (0, 0, 1))
                bday_M[:3, 3] = gp
        M = st.torso_M()
        st.set_parts("tray", S.tray_parts(M))
        cake_M = S.tray_top(M) if cake_on_robot else bday_M @ S._T(None, (0.43, 0.0, 1.02))
        st.set_parts("cake", [(tg, m, c, cake_M @ T) for tg, m, c, T in fx.cake_parts(lit)])
        # ------------------------------------------------------------ camera
        Cw = lambda x, y, z: np.array([x, y, Z0 + z]) + st.offset  # noqa: E731
        cut = False
        if phase == "arrive":
            eye, at = Cw(58.2, -5.6, 1.7), 0.6 * rp + 0.4 * Cw(61.4, -11.6, 0.9)
        elif phase in ("greet", "take"):
            eye, at = Cw(59.6, -7.6, 1.55), Cw(61.7, -11.3, 0.9)
            cut = phase == "greet" and abs(st.t - phase_t) < 0.03
        elif phase == "wish":
            eye, at = Cw(61.0, -8.9, 1.65), Cw(62.3, -11.1, 1.2)
            cut = abs(st.t - phase_t) < 0.03 and lit
        elif phase in ("back", "turn", "intro"):
            eye, at = Cw(60.4, -6.6, 1.6), Cw(60.9, -11.4, 0.8)
        else:
            orbit = 0.3 * math.sin(st.vtime * 0.15)
            eye = Cw(SPOT[0] + 2.9 * math.sin(orbit + 0.2), SPOT[1] + 2.9 * math.cos(orbit + 0.2), 1.15)
            at = 0.8 * rp + 0.2 * Cw(61.3, -12.0, 1.0) + np.array([0, 0, -0.05])
            if phase == "finale" and st.sk.clip_t < 0.1:
                cut = True

        def post(frame, cake_M=cake_M):
            frame = fx.add_party_glow(st.night, st.viewer, frame, st.bulbs, st.vtime, phase in ("dance", "finale", "end"))
            if lit:
                frame = fx.add_flames(st.night, st.viewer, frame, fx.flame_points(cake_M), st.vtime)
            return confetti.draw(st.night, st.viewer, frame, st.vtime)

        frame = st.shot(eye, at, cut=cut, post=post)
        if st.vtime > alert_until:
            alert = ""
        st.write(frame, dict(part="ปาร์ตี้", part_color=(230, 70, 150), title=title, step=step_txt, alert=alert,
                             alert_color=(230, 70, 150, 230)))
        if st.k % 250 == 0:
            print(f"[PARTY] t={st.t:.1f}s v={st.vtime:.1f}s {phase} skill {st.sk.skill} xy {xy.round(2)} "
                  f"yaw {yaw:.2f} upright {st.sk.upright():.2f}", flush=True)
    return dict(success=phase == "end", dances=[d[0] for d in DANCES], cake_kg=1.5)


def main():
    from isaaclab.app import add_launcher_args

    pa = argparse.ArgumentParser()
    pa.add_argument("--out", required=True)
    pa.add_argument("--rtx", type=int, default=0)
    pa.add_argument("--payload_kg", type=float, default=2.5, help="tray + cake")
    add_launcher_args(pa)
    args = pa.parse_args()
    import rtx_actors as ra

    actors = {n: ra.person_specs(c, s) for n, _, c, s in GUESTS}
    actors["cake"] = fx.CAKE_SPECS
    actors["tray"] = S.TRAY_SPECS
    st = PartyStage(args, look="night", actors=actors, payload_kg=args.payload_kg, name="party")
    extra = run_party(st)
    st.finish(extra)
    st.close()


if __name__ == "__main__":
    main()
