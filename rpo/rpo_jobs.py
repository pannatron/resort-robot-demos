"""RPO humanoid resort jobs (simulation only), all on Roboparty's shipped policies.

Jobs:
  welcome   09:12  a guest walks into the lobby: the robot waves hello (wave clip) in Thai + English, waits at
                   check-in, then walks ahead to room 306 (walking policy), pacing itself to the guest, and waves
                   goodbye at the door
  delivery  10:42  room service: staff load towels + water into the chest tray (tray + load = --payload_kg added
                   to the torso), the robot walks down the stairs to the pool deck and on to villa V2, the guest
                   comes out and takes the items from the tray
  bump      15:20  a child runs into the robot in the lobby (scripted push at contact, --push N for 0.2 s): it
                   falls on its back, gets up with the get-up policy, checks on the child

Usage (from ~/IsaacLab):
  uv run --extra video --with onnxruntime python ~/rpo_tools/resort/rpo_jobs.py --job welcome --out <dir> \
      --viz newton [--rtx 1]        (--rtx 1 also needs --extra ovrtx)
"""

from __future__ import annotations

import argparse
import datetime as dt_
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpo_stage as S  # noqa: E402  (sets up sys.path for the Dobot resort helpers)

import numpy as np  # noqa: E402

import resort_scene as rs  # noqa: E402
from common import Person, _T, guest_parts, person_parts, place, wrap  # noqa: E402
from common import _blob_part, _limb_part  # noqa: E402

SKIN = (0.86, 0.68, 0.54)
GUEST_CLOTHES = (0.85, 0.45, 0.55)
STAFF_CLOTHES, STAFF_VEST = (0.95, 0.95, 0.93), (0.20, 0.35, 0.55)
KID_CLOTHES, KID_SCALE = (0.98, 0.75, 0.20), 0.62
ITEM_SPECS = [("towel1", "cuboid", (0.20, 0.15, 0.05), (0.97, 0.97, 0.95)),
              ("towel2", "cuboid", (0.20, 0.15, 0.05), (0.80, 0.90, 0.97)),
              ("bottle1", "cylinder", (0.032, 0.20), (0.45, 0.70, 0.95)),
              ("bottle2", "cylinder", (0.032, 0.20), (0.45, 0.70, 0.95))]
VILLA_DOOR = (39.0, -48.0)
ROOM_DOOR = (46.5, 4.6)
HOME = (10.0, 3.0)  # robot's waiting spot in the lobby, facing the entrance (west)
PART = {"welcome": ("ต้อนรับ", (40, 160, 90)), "delivery": ("รูมเซอร์วิส", (40, 120, 220)),
        "bump": ("ความปลอดภัย", (230, 120, 40))}


# ------------------------------------------------------------------------------------------ people / props


def waving_parts(phase, clothes, skin, side=-1):
    """Standing person waving one hand (side -1 = right)."""
    P = [p for p in guest_parts("idle", phase, clothes, skin) if not p[0].endswith("r" if side < 0 else "l")
         or not p[0].startswith(("uarm", "farm", "hand"))]
    nm = "r" if side < 0 else "l"
    sh = np.array([0.0, side * 0.215, 1.42])
    elbow = sh + np.array([0.05, side * 0.24, 0.12])
    a = 0.35 * math.sin(phase * 3.0)
    wrist = elbow + np.array([0.02, side * 0.26 * math.sin(a), 0.26 * math.cos(a)])
    P.append(_limb_part(sh, elbow, 0.048, clothes, f"uarm{nm}"))
    P.append(_limb_part(elbow, wrist, 0.04, skin, f"farm{nm}"))
    P.append(_blob_part(wrist + (wrist - elbow) / 0.26 * 0.05, (0.04, 0.03, 0.055), skin, f"hand{nm}", 1))
    return P


def scaled(parts, s):
    """Smaller figure (child): meshes scaled, translations scaled, rotations kept (RTX parts use pose only)."""
    out = []
    for tag, mesh, col, T in parts:
        m = mesh.copy()
        m.apply_scale(s)
        T2 = T.copy()
        T2[:3, 3] *= s
        out.append((tag, m, col, T2))
    return out


def kid_specs():
    import rtx_actors as ra

    return [(tag, shape, tuple(d * KID_SCALE for d in dims), col) for tag, shape, dims, col in
            ra.person_specs(KID_CLOTHES, SKIN)]


def item_parts(M, spread=1.0):
    """Two folded towels and two water bottles, laid out at pose M (local z up = item bottoms at 0)."""
    import trimesh

    P = [("towel1", trimesh.creation.box(extents=(0.20, 0.15, 0.05)), (0.97, 0.97, 0.95), _T(None, (-0.02, 0.0, 0.025))),
         ("towel2", trimesh.creation.box(extents=(0.20, 0.15, 0.05)), (0.80, 0.90, 0.97), _T(None, (-0.02, 0.0, 0.075))),
         ("bottle1", trimesh.creation.cylinder(radius=0.032, height=0.20), (0.45, 0.70, 0.95),
          _T(None, (0.075, 0.06 * spread, 0.10))),
         ("bottle2", trimesh.creation.cylinder(radius=0.032, height=0.20), (0.45, 0.70, 0.95),
          _T(None, (0.075, -0.06 * spread, 0.10)))]
    return [(t, m, c, M @ T) for t, m, c, T in P]


def world_M(x, y, z, hd=0.0):
    import trimesh

    M = trimesh.transformations.rotation_matrix(hd, (0, 0, 1))
    M[:3, 3] = (x, y, z)
    return M


# ------------------------------------------------------------------------------------------ jobs


def follow_trail(trail, gpos, gap, lead_xy=None):
    """Point `gap` metres back along the leader's trail; before there is enough trail, `gap` from the leader."""
    acc = 0.0
    for j in range(len(trail) - 1, 0, -1):
        acc += np.linalg.norm(trail[j] - trail[j - 1])
        if acc >= gap:
            return trail[j - 1]
    if lead_xy is not None:
        d = gpos - lead_xy
        n = np.linalg.norm(d)
        return lead_xy + d / n * gap if n > gap else gpos
    return gpos


def step_person(p, tgt, speed, dt=S.POLICY_DT):
    d = np.asarray(tgt, float) - p.pos[:2]
    n = np.linalg.norm(d)
    if n > 0.05:
        s = min(n, speed * dt)
        p.pos[:2] = p.pos[:2] + d / n * s
        p.heading = math.atan2(d[1], d[0])
        p.phase += s * 5.0
        return True
    return False


def run_welcome(st):
    st.clock0 = dt_.datetime(2026, 10, 24, 9, 12, 0)
    st.teleport(*HOME, math.pi)
    guest = Person("guest", [(-1.5, 3.25, 0.0), (8.3, 3.25, 0.0)], 1.1, GUEST_CLOTHES, heading=0.0)
    guest.start()
    steps = ["แขกเดินเข้าล็อบบี้", "โบกมือทักทาย TH/EN", "รอแขกเช็กอิน", "เดินนำไปห้อง 306", "ส่งแขกถึงหน้าห้อง"]
    done = [False] * len(steps)
    route = [(14.0, 6.6), (16.0, 3.2), (25.8, 3.0), (46.5, 3.0)]
    phase, phase_t, idx, pool_said = "wait", 0.0, 1, False
    trail, gpose, alert, alert_until = [], "walk", "", -1.0
    st.post("arrive", "แขกใหม่เข้าล็อบบี้")
    while st.t < 260:
        rp, q, p, yaw = st.pose()
        xy, gpos = p[:2], guest.pos[:2]
        skill, cmd = "walk", (0.0, 0.0, 0.0)
        tgt, spd, look = xy, 0.0, gpos
        if phase == "wait":
            done[0] = True
            if guest.arrived and st.t > 1.0:
                phase, phase_t = "greet", st.t
                st.sk.start("wave", 2.5, 7.6)
                skill = "motion"
                st.say("greet_th", 0.6)
                st.say("greet_en")
                done[1] = True
                st.post("greet", "โบกมือทักทายแขก (wave policy)")
                gpose = "idle"
        elif phase == "greet":
            skill = "motion"
            if st.t - phase_t > 2.5:
                gpose = "wave" if st.t - phase_t < 5.5 else "idle"
            if st.sk.done and st.voice.done_by(st.vtime):
                phase, phase_t = "checkin", st.t
                guest.start([(8.3, 3.25, 0.0), (11.5, 6.0, 0.0), (15.2, 8.7, 0.0)])
                gpose = "walk"
                alert, alert_until = "แขกเช็กอินที่เคาน์เตอร์ — หุ่นรอพาไปห้อง", st.vtime + 5
        elif phase == "checkin":
            tgt, spd, look = np.array(route[0]), 0.35, gpos
            if np.linalg.norm(xy - tgt) < 0.25:
                spd = 0.0
            if guest.arrived:
                gpose = "idle"
                guest.heading = math.pi / 2
            if guest.arrived and st.t - phase_t > 12.0:
                done[2] = True
                phase, phase_t = "guide", st.t
                route[0] = tuple(xy)
                st.say("guide_go_th")
                st.post("checkin", "เช็กอินห้อง 306 เสร็จ เริ่มนำทาง")
                gpose = "walk"
        elif phase == "guide":
            a, b = np.array(route[idx - 1]), np.array(route[idx])
            if np.linalg.norm(xy - b) < 0.45:
                idx += 1
                if idx >= len(route):
                    phase, phase_t = "arrive", st.t
                    done[3] = True
                    st.post("arrived", "ถึงหน้าห้อง 306")
                idx = min(idx, len(route) - 1)
                a, b = np.array(route[idx - 1]), np.array(route[idx])
            seg = b - a
            L = max(np.linalg.norm(seg), 1e-6)
            u = seg / L
            s_ = np.clip(np.dot(xy - a, u), 0, L)
            tgt = a + u * min(s_ + 1.0, L)
            gap = np.linalg.norm(gpos - xy)
            spd = 0.5 * float(np.clip((3.0 - gap) / 1.0, 0.0, 1.0))  # wait for the guest
            if xy[0] > 31.0 and not pool_said:
                pool_said = True
                st.say("guide_pool_th")
                alert, alert_until = "แนะนำจุดสำคัญระหว่างทาง: สระว่ายน้ำ (ขวามือ)", st.vtime + 5
        elif phase == "arrive":
            tgt, spd, look = xy, 0.0, gpos
            if st.t - phase_t > 1.6 and st.sk.skill != "wave" and "waved" not in st.__dict__:
                st.waved = True
                skill = "motion"
                st.sk.start("wave", 2.5, 6.8)
                st.say("guide_arrive_th", 0.3)
                alert, alert_until = "ถึงห้อง 306 ✓  โบกมือส่งแขก", st.vtime + 6
            elif st.sk.skill == "wave" and not st.sk.done:
                skill = "motion"
            elif "waved" in st.__dict__:
                done[4] = True
                if st.voice.done_by(st.vtime) and st.t - phase_t > 4.0:
                    phase, phase_t = "end", st.t
                    st.post("done", "ส่งแขกถึงห้อง ✓ กลับไปรอที่ล็อบบี้")
        else:
            tgt, spd, look = xy, 0.0, gpos
            if st.t - phase_t > 3.0:
                break
        # guest
        if phase == "guide":
            if not trail or np.linalg.norm(xy - trail[-1]) > 0.1:
                trail.append(xy.copy())
            moved = step_person(guest, follow_trail(trail, gpos, 1.6, xy), 1.15)
            gpose = "walk" if moved else "idle"
        elif phase in ("arrive", "end"):
            door = np.array([ROOM_DOOR[0] - 0.55, ROOM_DOOR[1] - 0.45])
            if st.t - phase_t < 6.0 or phase == "arrive":
                moved = step_person(guest, door, 1.0)
                if not moved:
                    guest.heading = math.atan2(xy[1] - gpos[1], xy[0] - gpos[0])
                    gpose = "wave" if st.sk.skill == "wave" and 1.0 < st.sk.clip_t < 4.0 else "idle"
                else:
                    gpose = "walk"
            else:
                moved = step_person(guest, (ROOM_DOOR[0], ROOM_DOOR[1] + 0.6), 1.0)
                gpose = "walk"
        else:
            guest.update(S.POLICY_DT)
        if skill == "walk":
            vx, wz = st.nav(xy, yaw, tgt, spd, look)
            cmd = (vx, 0.0, wz)
        if not st.tick(skill, cmd):
            continue
        # actors
        if gpose == "walk":
            parts = person_parts(guest.phase, GUEST_CLOTHES, skin=SKIN, walking=True)
        elif gpose == "wave":
            parts = waving_parts(st.vtime * 3, GUEST_CLOTHES, SKIN)
        else:
            parts = guest_parts("idle", st.vtime, GUEST_CLOTHES, SKIN)
        st.set_parts("guest", place(parts, *guest.pos, guest.heading))
        # camera
        gw = np.array([*guest.pos[:2], 0.0]) + st.offset
        cut = False
        if phase in ("wait", "greet"):
            eye, at = np.array([8.2, 0.7, 1.4]) + st.offset, 0.6 * rp + 0.4 * (gw + [0, 0, 0.95])
        elif phase == "checkin":
            eye, at = np.array([11.6, 0.9, 1.6]) + st.offset, np.array([14.4, 7.2, 0.8]) + st.offset
            cut = abs(st.t - phase_t) < 0.03
        elif phase == "guide":
            dgr = rp[:2] - gw[:2]
            dgr = dgr / max(np.linalg.norm(dgr), 1e-6)
            left = np.array([-dgr[1], dgr[0]])
            eye = rp + np.array([3.2 * dgr[0] - 0.8 * left[0], 3.2 * dgr[1] - 0.8 * left[1], 0.55])
            at = 0.6 * rp + 0.4 * (gw + np.array([0, 0, 0.9]))
            cut = abs(st.t - phase_t) < 0.03
        else:
            eye = np.array([43.6, 1.6, 1.6]) + st.offset
            at = 0.5 * rp + 0.5 * (gw + np.array([0, 0, 0.9]))
            cut = abs(st.t - phase_t) < 0.03
        frame = st.shot(eye, at, cut=cut)
        if st.vtime > alert_until:
            alert = ""
        st.write(frame, dict(part=PART["welcome"][0], part_color=PART["welcome"][1], title="ต้อนรับ & นำทางแขก",
                             step=next((n for n, ok in zip(steps, done) if not ok), steps[-1] + " ✓"),
                             alert=alert, alert_color=(30, 150, 70, 230)))
        if st.k % 500 == 0:
            print(f"[JOB] t={st.t:.0f}s {phase} xy {xy.round(2)} guest {gpos.round(2)} skill {st.sk.skill}", flush=True)
    return dict(success=phase == "end")


def run_delivery(st):
    st.clock0 = dt_.datetime(2026, 10, 24, 10, 42, 0)
    st.teleport(16.0, 8.5, -math.pi / 2)
    guest = Person("guest", [(VILLA_DOOR[0], VILLA_DOOR[1] - 0.3, rs.L_VILLA)], 1.0, (0.30, 0.55, 0.80),
                   heading=math.pi / 2)
    staff = Person("staff", [(16.0, 9.0 + 0.0, 0.0)], 1.0, STAFF_CLOTHES, heading=-math.pi / 2)
    staff.pos = np.array([16.0, 9.05, 0.0])
    route = [(16.0, 8.5), (16.0, 3.2), (25.8, 3.0), (44.0, 3.0), (57.0, 3.0), (58.5, 2.2), (58.5, 1.5),
             (58.5, -1.4), (60.5, -4.0), (61.0, -12.0), (61.0, -21.0), (56.0, -24.5), (50.0, -27.5),
             (50.0, -36.0), (50.0, -44.0), (43.0, -45.8), (39.0, -46.4)]
    SPEED = [0.5, 0.5, 0.55, 0.55, 0.45, 0.35, 0.35, 0.35, 0.45, 0.55, 0.55, 0.5, 0.5, 0.5, 0.5, 0.45, 0.35]
    steps = ["ออเดอร์ #1024 เข้า", "พนักงานวางของในถาด", "ลงบันไดไปลานสระ", "ผ่านลานสระ → โซนวิลล่า", "ถึงวิลล่า V2",
             "แขกหยิบของจากถาด"]
    done = [True] + [False] * 5
    items_where = "counter"
    phase, phase_t, idx = "load", 0.0, 1
    alert, alert_until = f"ออเดอร์ #1024: ผ้าเช็ดตัว 2 + น้ำ 2 ขวด → วิลล่า V2", 4.0
    st.post("order", "ออเดอร์ #1024: ผ้าเช็ดตัว 2 + น้ำ 2 ขวด → วิลล่า V2")
    t_depart = None
    yaw_s = None
    while st.t < 600:
        rp, q, p, yaw = st.pose()
        xy = p[:2]
        tgt, spd, look = xy, 0.0, None
        if phase == "load":
            if st.t > 2.2 and items_where == "counter":
                items_where = "tray"
                done[1] = True
                st.say("dlv_pickup_th")
                st.post("pickup", f"วางของในถาดแล้ว (ถาด+ของ {st.payload_kg:.0f} กก.) ออกเดินทาง")
            if items_where == "tray" and st.voice.done_by(st.vtime):
                phase, t_depart = "go", st.t
        elif phase == "go":
            a, b = np.array(route[idx - 1]), np.array(route[idx])
            if np.linalg.norm(xy - b) < (0.35 if SPEED[idx] <= 0.4 else 0.6):
                idx += 1
                if idx >= len(route):
                    phase, phase_t = "arrived", st.t
                    done[4] = True
                    st.post("arrived", "ถึงหน้าวิลล่า V2 แจ้งแขกทางโทรศัพท์ห้อง")
                    alert, alert_until = "ถึงวิลล่า V2 แล้ว — แจ้งแขกให้ออกมารับของ", st.vtime + 5
                    guest.start([(VILLA_DOOR[0], VILLA_DOOR[1] - 0.2, rs.L_VILLA), (VILLA_DOOR[0], -47.05, rs.L_VILLA)])
                idx = min(idx, len(route) - 1)
                a, b = np.array(route[idx - 1]), np.array(route[idx])
            seg = b - a
            L = max(np.linalg.norm(seg), 1e-6)
            u = seg / L
            s_ = np.clip(np.dot(xy - a, u), 0, L)
            tgt = a + u * min(s_ + (0.5 if SPEED[idx] <= 0.4 else 0.9), L)
            spd = SPEED[idx]
            if 1.6 > xy[1] > -1.5 and 57 < xy[0] < 60:
                done[2] = True
            if xy[1] < -24:
                done[3] = True
        else:
            look = np.array(VILLA_DOOR)
            if phase == "arrived" and guest.arrived:
                st.say("dlv_arrive_th")
                phase, phase_t = "handover", st.t
            if phase == "handover" and st.t - phase_t > 3.5 and items_where == "tray":
                items_where = "guest"
                done[5] = True
                st.post("delivered", f"ส่งสำเร็จ ✓ {(st.t - t_depart) / 60:.1f} นาที ระยะ {st.dist:.0f} ม.")
                alert, alert_until = f"ส่งสำเร็จ ✓  เดิน {st.dist:.0f} ม. ใน {(st.t - t_depart) / 60:.1f} นาที", st.vtime + 30
                st.say("dlv_thanks_th", 0.8)
            if phase == "handover" and items_where == "guest" and st.t - phase_t > 6.0 and guest.arrived:
                guest.start([(guest.pos[0], guest.pos[1], rs.L_VILLA), (VILLA_DOOR[0], VILLA_DOOR[1] - 0.2, rs.L_VILLA)])
                phase, phase_t = "leave", st.t
            if phase == "leave" and st.t - phase_t > 6.0:
                break
        vx, wz = st.nav(xy, yaw, tgt, spd, look)
        guest.update(S.POLICY_DT)
        if phase in ("handover",) and guest.arrived:
            guest.heading = math.atan2(xy[1] - guest.pos[1], xy[0] - guest.pos[0])
        if not st.tick("walk", (vx, 0.0, wz)):
            continue
        # actors
        M = st.torso_M()
        st.set_parts("tray", S.tray_parts(M))
        if items_where == "counter":
            IM = world_M(16.0, 9.75, 1.15, -math.pi / 2)
        elif items_where == "tray":
            IM = S.tray_top(M)
        else:
            IM = world_M(*(guest.pos + [0.3 * math.cos(guest.heading), 0.3 * math.sin(guest.heading), 0.98]),
                         guest.heading)
        st.set_parts("items", item_parts(IM))
        gp = guest_parts("hold" if items_where == "guest" else "idle", st.vtime, (0.30, 0.55, 0.80), SKIN) \
            if not guest.walking else person_parts(guest.phase, (0.30, 0.55, 0.80), skin=SKIN)
        st.set_parts("guest", place(gp, *guest.pos, guest.heading) if phase != "load" and phase != "go" or guest.walking
                     else place(gp, guest.pos[0], guest.pos[1] + 3.0, guest.pos[2] - 6, 0))  # inside the villa
        sp = guest_parts("hold" if items_where == "counter" else "idle", st.vtime, STAFF_CLOTHES, SKIN)
        st.set_parts("staff", place(sp, *staff.pos, staff.heading))
        # camera: counter shot, chase (wider on the stairs), side shot at the villa
        yaw_s = yaw if yaw_s is None else yaw_s + 0.04 * wrap(yaw - yaw_s)
        on_stairs = 57 < xy[0] < 60 and -2.0 < xy[1] < 2.4
        cut = False
        if phase == "load" or (t_depart is not None and st.t - t_depart < 2.0):
            eye, at = rp + np.array([2.4, -2.3, 0.75]), rp + np.array([0.0, 0.6, 0.0])
        elif phase == "go":
            corridor = 26 < xy[0] < 57 and 1.3 < xy[1] < 4.6  # narrow: stay straight behind
            side = math.radians(-20 if on_stairs else (-8 if corridor else -35))
            dc, hc = (2.6, 1.6) if on_stairs else ((2.4, 0.9) if corridor else (3.0, 0.75))
            ang = yaw_s + math.pi - side
            eye = rp + np.array([dc * math.cos(ang), dc * math.sin(ang), hc])
            at = rp + np.array([0.6 * math.cos(yaw_s), 0.6 * math.sin(yaw_s), -0.25 if on_stairs else 0.0])
            cut = t_depart is not None and abs(st.t - t_depart - 2.0) < 0.03
        else:
            gw = guest.pos + st.offset
            eye = np.array([41.6, -44.2, rs.L_VILLA + 1.5]) + st.offset
            at = 0.5 * rp + 0.5 * (gw + np.array([0, 0, 0.8]))
            cut = phase == "arrived" and abs(st.t - phase_t) < 0.03
        frame = st.shot(eye, at, cut=cut)
        if st.vtime > alert_until:
            alert = ""
        st.write(frame, dict(part=PART["delivery"][0], part_color=PART["delivery"][1],
                             title=f"รูมเซอร์วิส: ถาดหน้าอก {st.payload_kg:.0f} กก.",
                             step=next((n for n, ok in zip(steps, done) if not ok), steps[-1] + " ✓"),
                             alert=alert, alert_color=(30, 110, 200, 230)))
        if st.k % 500 == 0:
            print(f"[JOB] t={st.t:.0f}s {phase} xy {xy.round(2)} z {p[2]:.2f} idx {idx} v {st.vx_cmd:.2f}", flush=True)
    ok = items_where == "guest"
    return dict(success=ok, trip_s=round(st.t - t_depart, 1) if t_depart else None)


def run_bump(st, push=600.0):
    st.clock0 = dt_.datetime(2026, 10, 24, 15, 20, 0)
    st.teleport(*HOME, math.pi)
    kid = Person("kid", [(-2.0, 3.45, 0.0), (HOME[0] - 0.42, 3.2, 0.0)], 3.0, KID_CLOTHES, heading=0.0)
    phase, phase_t = "idle", 0.0
    alert, alert_until = "", -1.0
    steps = ["ยืนรอแขกที่ล็อบบี้", "เด็กวิ่งชน → หุ่นล้ม", "ลุกขึ้นเอง (get-up policy)", "ถามอาการเด็ก"]
    done = [True, False, False, False]
    kid_pose = "walk"
    t_fall = t_up = None
    st.post("idle", "หุ่นยืนรอที่ล็อบบี้")
    while st.t < 60:
        rp, q, p, yaw = st.pose()
        xy = p[:2]
        skill, cmd = "walk", (0.0, 0.0, 0.0)
        if phase == "idle":
            if st.t > 2.0:
                step_person(kid, xy, 3.0)
            if st.t > 2.0 and np.linalg.norm(kid.pos[:2] - xy) < 0.5:
                phase, phase_t = "hit", st.t
                done[1] = True
                st.post("bump", f"เด็กวิ่งชนหุ่น (จำลองแรงชน {push:.0f} N นาน 0.2 วิ)")
                kid.walking = False
                kid_pose = "idle"
        if phase == "hit":
            skill = "hold"
            i = int(round((st.t - phase_t) / S.POLICY_DT))
            d = np.array([math.cos(kid.heading), math.sin(kid.heading)])
            st.sk.push([push * d[0], push * d[1], 0.0] if i < 10 else None)
            if st.sk.upright() < 0.5 and t_fall is None:
                t_fall = st.t
                st.falls += 1
                alert, alert_until = "ตรวจพบการล้ม (IMU) → สั่งลุกขึ้นเอง", st.vtime + 6
                st.post("fall", "ตรวจพบการล้ม — หุ่นนอนหงาย")
            if st.t - phase_t > 2.6:
                phase, phase_t = "getup", st.t
                st.sk.start("getup")
                skill = "motion"
        elif phase == "getup":
            skill = "motion"
            if st.sk.done:
                ok = st.sk.upright() > 0.95
                t_up = st.t
                st.getups += int(ok)
                done[2] = ok
                st.post("getup", f"ลุกขึ้นเองสำเร็จ ใช้เวลา {t_up - t_fall:.1f} วิ" if ok else "ลุกไม่สำเร็จ")
                phase, phase_t = "check", st.t
                st.say("bump_sorry_th", 0.1)
                st.say("bump_ok_th")
                alert, alert_until = (f"ลุกขึ้นเองใน {t_up - t_fall:.1f} วิ ✓  แจ้งเหตุให้ฟรอนต์บันทึก", st.vtime + 7)
        elif phase == "check":
            done[3] = True
            vx, wz = st.nav(xy, yaw, xy, 0.0, kid.pos[:2])
            cmd = (0.0, 0.0, wz)
            if st.voice.done_by(st.vtime) and st.t - phase_t > 3.0:
                break
        if not st.tick(skill, cmd):
            continue
        if phase == "hit" and st.t - phase_t < 1.5:  # the child staggers back a little
            kid.pos[:2] -= 0.4 * S.POLICY_DT * 2 * np.array([math.cos(kid.heading), math.sin(kid.heading)])
        kp = person_parts(kid.phase, KID_CLOTHES, skin=SKIN, walking=True) if kid_pose == "walk" else \
            guest_parts("idle", st.vtime, KID_CLOTHES, SKIN)
        st.set_parts("kid", place(scaled(kp, KID_SCALE), *kid.pos, kid.heading))
        eye = np.array([HOME[0] + 1.4, HOME[1] + 3.3, 1.05]) + st.offset
        at = np.array([HOME[0] - 0.5, HOME[1] - 0.1, 0.45]) + st.offset
        frame = st.shot(eye, at)
        if st.vtime > alert_until:
            alert = ""
        st.write(frame, dict(part=PART["bump"][0], part_color=PART["bump"][1], title="ล้มแล้วลุกเอง",
                             step=next((n for n, ok in zip(steps, done) if not ok), steps[-1] + " ✓"),
                             alert=alert, alert_color=(220, 110, 30, 230)))
    return dict(success=t_up is not None and done[2], fall_to_up_s=round(t_up - t_fall, 2) if t_up and t_fall else None,
                push_N=push)


# ------------------------------------------------------------------------------------------ main


def main():
    from isaaclab.app import add_launcher_args

    pa = argparse.ArgumentParser()
    pa.add_argument("--job", choices=["welcome", "delivery", "bump"], required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--rtx", type=int, default=0)
    pa.add_argument("--payload_kg", type=float, default=5.0)
    pa.add_argument("--push", type=float, default=600.0)
    add_launcher_args(pa)
    args = pa.parse_args()
    import rtx_actors as ra

    actors = {"welcome": {"guest": ra.person_specs(GUEST_CLOTHES, SKIN)},
              "delivery": {"guest": ra.person_specs((0.30, 0.55, 0.80), SKIN),
                           "staff": ra.person_specs(STAFF_CLOTHES, SKIN), "tray": S.TRAY_SPECS, "items": ITEM_SPECS},
              "bump": {"kid": kid_specs()}}[args.job]
    look = "morning" if args.job == "welcome" else "day"
    st = S.Stage(args, look=look, actors=actors, payload_kg=args.payload_kg if args.job == "delivery" else 0.0,
                 name=args.job)
    extra = {"welcome": run_welcome, "delivery": run_delivery}.get(args.job, lambda s: run_bump(s, args.push))(st)
    st.finish(extra)
    st.close()


if __name__ == "__main__":
    main()
