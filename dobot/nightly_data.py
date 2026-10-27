"""Data for the full nightly report page: one simulated night (22:00-06:00) of the 2-robot fleet.

Joins the fleet-sim night (timeline, battery, check-ins; same seed as fleet_timelines.json) with the incidents of
the physics/RTX takes that happen that night (scenario 6 at 23:41, tablet override at 23:55) and their photos.
Writes one JSON file per dataset (row objects) into <out_dir>.

nightly_data.py <out_dir>          (run with ~/IsaacLab's uv python: needs numpy, PIL, imageio_ffmpeg)
"""

import base64
import datetime as dt
import io
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fleet_sim as fs  # noqa: E402
import resort_scene as rs  # noqa: E402

NIGHT0 = dt.datetime(2026, 10, 22, 22, 0, 0)
ROBOTS = ["WL-01", "WL-02"]
RTX = os.path.join(HERE, "results_rtx")
STATE_TH = {"PATROL": "ลาดตระเวน", "RELIEF": "ไปรับช่วงต่อ", "RETURN_TO_BASE": "กลับป้อม", "WAIT_SWAP": "รอสลับแบต",
            "SWAP": "สลับแบต", "SELF_CHECK": "ตรวจตัวเอง", "STANDBY": "สแตนด์บาย", "EMERGENCY_RETURN": "กลับป้อมด่วน",
            "ASSIST": "ไปช่วยเหตุ"}


def clock(sec):
    return (NIGHT0 + dt.timedelta(seconds=float(sec))).strftime("%H:%M:%S")


def iso(sec):
    return (NIGHT0 + dt.timedelta(seconds=float(sec))).isoformat()


def sec_of(ts):
    """'2026-10-22T23:41:35.96' -> seconds since 22:00."""
    return (dt.datetime.fromisoformat(ts) - NIGHT0).total_seconds()


def frame_jpg(video, t, w=560):
    import imageio_ffmpeg
    from PIL import Image

    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video,
                          "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = im.resize((w, int(im.height * w / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    scene = rs.build()
    wp = scene.route
    seg_t, seg_e, n_prof = fs.load_profile(os.path.join(HERE, "results", "profile"))
    cps = [i for i, w in enumerate(wp) if w.checkpoint][:-1]
    route = fs.Route(seg_t, seg_e, cps)
    night = fs.simulate(route, 2, 4, 1.5, seed=3, record=True)
    lap_m = float(sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(wp[:-1], wp[1:])))

    # ---- robot timeline
    timeline = []
    for i, tl in enumerate(night["timeline"]):
        for a, b, s in tl:
            timeline.append(dict(robot=ROBOTS[i], state=s, state_th=STATE_TH[s], start=iso(a), end=iso(b),
                                 minutes=round((b - a) / 60, 1)))
    patrol_s = sum(b - a for tl in night["timeline"] for a, b, s in tl if s == "PATROL")

    # ---- battery (every minute) + packs ready on the shelf
    battery = [dict(time=iso(t), robot=ROBOTS[i], battery_pct=round(100 * b, 1), packs_ready=int(p))
               for t, i, b, p in night["battery"]]

    # ---- check-ins per checkpoint
    names = {c: wp[c].checkpoint.replace(" (เริ่ม)", "") for c in cps}
    checkin_rows = [dict(time=iso(t), robot=ROBOTS[i], checkpoint=names[c], order=cps.index(c)) for t, i, c in
                    night["checkins"]]
    checkpoints = []
    for k, c in enumerate(cps):
        ts = sorted(t for t, i, cc in night["checkins"] if cc == c)
        gaps = np.diff([0.0] + ts + [8 * 3600.0]) / 60
        checkpoints.append(dict(order=k + 1, checkpoint=names[c], visits=len(ts), first=clock(ts[0]), last=clock(ts[-1]),
                                avg_gap_min=round(float(np.mean(np.diff(ts)) / 60), 1),
                                max_gap_min=round(float(gaps.max()), 1), missed=0, x=wp[c].x, y=wp[c].y))

    # ---- incidents from the physics takes + battery swaps from the fleet night
    events, steps, photos = [], [], []
    s6 = [json.loads(x) for x in open(os.path.join(RTX, "s6", "events.jsonl"))]
    E = {e["type"]: e for e in s6}
    v6 = os.path.join(RTX, "s6", "scenario6.mp4")
    t6 = sec_of("2026-10-22T23:40:00")  # scenario 6 video starts at 23:40:00 (video time = sim time)

    def vt6(e, d=0.0):
        return sec_of(e["time"]) - t6 + d

    def add_event(eid, typ, sev, t0, robot, zone, title, outcome, response, x, y, photo, step_list):
        events.append(dict(id=eid, time=iso(t0), clock=clock(t0)[:5], robot=robot, zone=zone, type=typ, severity=sev,
                           title=title, outcome=outcome, response=response, x=x, y=y, photo=photo))
        for j, (ts, txt) in enumerate(step_list):
            steps.append(dict(event=eid, seq=j + 1, time=iso(ts), clock=clock(ts), text=txt))

    ei = E["intruder_pool_after_hours"]
    photos.append(dict(id="p_intruder", event="intruder", caption="ภาพจากหุ่น: ไฟส่อง + พูดเตือน ที่ลานสระ",
                       img=frame_jpg(v6, vt6(ei, 4.0))))
    add_event("intruder", "บุคคลในเขตหวงห้าม", "warning", sec_of(ei["time"]), ROBOTS[0], ei["zone"],
              "บุคคลในสระหลังเวลาปิด", "ออกจากพื้นที่เอง หลังเตือน + รปภ. คุยผ่านลำโพง",
              f"{sec_of(E['cleared']['time']) - sec_of(ei['time']):.0f} วิ จนออกจากพื้นที่",
              ei["location"]["x"], ei["location"]["y"], "p_intruder",
              [(sec_of(ei["time"]), "กล้องหุ่นพบบุคคลในสระหลังปิด → เปิดไฟส่อง + ไฟเตือน"),
               (sec_of(ei["time"]) + 1, "หุ่นพูดเตือนไทย/อังกฤษ"),
               (sec_of(E["intercom"]["time"]), "รปภ. ที่ป้อมคุยผ่านลำโพงหุ่น (ไม่ต้องเดินมา)"),
               (sec_of(E["cleared"]["time"]), "บุคคลออกจากพื้นที่ → หุ่นกลับลาดตระเวน")])
    ef, ea, eg = E["person_down"], E["first_aid_arrived"], E["guard_arrived"]
    photos.append(dict(id="p_fall", event="fall", caption="หุ่นถึงก่อนพร้อมชุดปฐมพยาบาล (2 กก.)",
                       img=frame_jpg(v6, vt6(ea, 3.0))))
    photos.append(dict(id="p_guard", event="fall", caption="รปภ. มาถึง รับชุดปฐมพยาบาลจากหลังหุ่น",
                       img=frame_jpg(v6, vt6(eg, 2.5))))
    lead = sec_of(eg["time"]) - sec_of(ea["time"])
    add_event("fall", "เหตุฉุกเฉิน", "critical", sec_of(ef["time"]), ROBOTS[0], ef["zone"],
              "แขกล้มบนทางเดินสวน", "รปภ. + พยาบาลรับช่วงดูแล", f"หุ่นถึงก่อน รปภ. {lead:.0f} วิ",
              ef["location"]["x"], ef["location"]["y"], "p_fall",
              [(sec_of(ef["time"]), "กล้องหุ่นพบคนล้ม/นอนนิ่ง → แจ้ง รปภ. + พยาบาลทันที"),
               (sec_of(ea["time"]), f"หุ่นวิ่งล้อนำชุดปฐมพยาบาลไปถึง ({sec_of(ea['time']) - sec_of(ef['time']):.0f} วิ หลังพบ)"),
               (sec_of(eg["time"]), f"รปภ. มาถึง (หลังหุ่น {lead:.0f} วิ) รับชุดปฐมพยาบาล")])

    tab = json.load(open(os.path.join(RTX, "tablet", "result_tablet.json")))
    T = {e["type"]: e for e in tab["events"]}
    t_tab0 = sec_of("2026-10-22T23:55:00")  # the take's clock says 10-23 23:55; it happens on this night
    tt = {k: sec_of(v["time"].replace("2026-10-23T", "2026-10-22T")) for k, v in T.items()}
    vtab = os.path.join(RTX, "tablet", "tablet.mp4")
    photos.append(dict(id="p_tablet", event="tablet", caption="แท็บเล็ต รปภ.: ไปจุดนี้ • บังคับเอง • กล้องหุ่นสด",
                       img=frame_jpg(vtab, tt["person_found"] - t_tab0 + 4.0, w=640)))
    add_event("tablet", "รปภ. สั่งงานเอง", "info", tt["guest_call"], ROBOTS[0], "โซนวิลล่า",
              "แขกแจ้งเสียงคนเดินหลังวิลล่า 1", "พบแขกหลงทาง พาไปถูกวิลล่า",
              f"กดหยุด → หุ่นนิ่งใน {tab['estop_time_s']:.2f} วิ", 28.5, -55.0, "p_tablet",
              [(tt["guest_call"], "แขกวิลล่า 1 โทรแจ้ง"),
               (tt["guard_goto"], "รปภ. แตะแผนที่ \"ไปจุดนี้\" → หุ่นไปเอง"),
               (tt["manual_on"], "รปภ. กด \"บังคับเอง\" ขับด้วยจอยสติ๊ก"),
               (tt["person_found"], "กล้องหุ่นพบแขกหลงทาง → เปิดไฟส่อง"),
               (tt["intercom"], "รปภ. คุยผ่านลำโพงหุ่น"),
               (tt["estop"], "รปภ. กด \"หยุด\" ให้แขกเดินผ่าน"),
               (tt["resume"], "รปภ. กด \"ลาดตระเวนต่อ\""),
               (tt["rejoined"], "หุ่นกลับเข้าเส้นทางเอง")])

    v5 = os.path.join(RTX, "s5", "scenario5.mp4")
    swap_photo = False
    for i, tl in enumerate(night["timeline"]):
        for j, (a, b, s) in enumerate(tl):
            if s != "SWAP":
                continue
            wait = next((bb - aa for aa, bb, ss in reversed(tl[:j]) if ss == "WAIT_SWAP"), 0.0)
            ret = next((aa for aa, bb, ss in reversed(tl[:j]) if ss in ("RETURN_TO_BASE", "EMERGENCY_RETURN")), a - wait)
            relief = next(((aa, ROBOTS[k]) for k, tl2 in enumerate(night["timeline"]) if k != i
                           for aa, bb, ss in tl2 if ss == "RELIEF" and abs(bb - ret) < 600), None)
            eid = f"swap{len([e for e in events if e['type'] == 'สลับแบต']) + 1}"
            ph = None
            if not swap_photo and os.path.exists(v5):
                photos.append(dict(id="p_swap", event=eid, caption="รปภ. สลับแบตที่ป้อม หุ่นอีกตัวลาดตระเวนแทน",
                                   img=frame_jpg(v5, 40.0)))
                ph, swap_photo = "p_swap", True
            st = []
            if relief:
                st.append((relief[0], f"{relief[1]} ออกไปรับช่วงต่อที่จุดเช็กอินล่าสุด (ไม่มีช่วงว่าง)"))
            st += [(ret, f"{ROBOTS[i]} แบตถึงเกณฑ์ (พลังงานกลับป้อม + เผื่อ 15%) → กลับป้อม"),
                   (a - wait, "แจ้ง รปภ. ให้มาสลับแบต"),
                   (a, f"รปภ. มาถึง (รอ {wait / 60:.1f} นาที) สลับแบต")]
            if wait > fs.ESCALATE_MIN * 60:
                st.insert(-1, (a - wait + fs.ESCALATE_MIN * 60, "รอเกิน 8 นาที → แจ้งซ้ำถึงหัวหน้ากะ"))
            st.append((b + fs.SELF_CHECK_S, "ตรวจตัวเองผ่าน แบต 100% → สแตนด์บาย"))
            add_event(eid, "สลับแบต", "warning" if wait > fs.ESCALATE_MIN * 60 else "info", a, ROBOTS[i], "ป้อม รปภ.",
                      f"สลับแบต {ROBOTS[i]}", "แบตเก่าเข้าแท่นชาร์จ ~1.5 ชม.",
                      f"รปภ. มาใน {wait / 60:.1f} นาที" + (" (แจ้งซ้ำ)" if wait > fs.ESCALATE_MIN * 60 else ""),
                      scene.dock[0], scene.dock[1], ph, st)
    events.sort(key=lambda e: e["time"])
    steps.sort(key=lambda s: (s["event"], s["seq"]))

    # ---- map: route, checkpoints, areas
    route_rows = [dict(seq=k, x=w.x, y=w.y, checkpoint=(w.checkpoint or "").replace(" (เริ่ม)", "").replace(" (จบรอบ)", ""))
                  for k, w in enumerate(wp)]
    areas = [("lawn", -12, 72, -60, 1.3, ""), ("deck", 30, 66, -24, 1.3, "ลานสระ"), ("pool", 40, 56, -17, -7, "สระว่ายน้ำ"),
             ("building", -12, 6, -6, 14, "ป้อม รปภ."), ("building", 6, 26, -6, 12, "ล็อบบี้"),
             ("corridor", 26, 62, 1.3, 4.6, "ทางเดินห้องพัก"), ("building", 26, 62, 4.6, 12, "ห้องพัก")]
    areas += [("villa", vx0, vx0 + 10, -56, -48, f"วิลล่า {i + 1}") for i, vx0 in enumerate((14, 34, 56))]
    areas += [("path", 48.6, 51.4, -57.5, -30, ""), ("path", -1.4, 51.4, -47.4, -44.6, ""), ("path", -1.4, 1.4, -47.4, -6, "")]
    map_rows = [dict(kind=k, x0=a, x1=b, y0=c, y1=d, label=lb) for k, a, b, c, d, lb in areas]

    # ---- summary
    warn = [e for e in events if e["type"] in ("บุคคลในเขตหวงห้าม", "เหตุฉุกเฉิน")]
    summary = [dict(night="22 ต.ค. 22:00 – 23 ต.ค. 06:00", robots=2, coverage_pct=round(100 * night["coverage"], 1),
                    gap_min=round(night["gap_min"], 1), laps=int(patrol_s // route.lap_t),
                    km=round(patrol_s / route.lap_t * lap_m / 1000, 1), checkins=len(night["checkins"]),
                    checkins_missed=0, incidents=len(warn), guard_actions=1, swaps=night["swaps"],
                    escalations=night["escalations"], falls=0, robot_lead_s=round(lead),
                    estop_s=tab["estop_time_s"], lap_min=round(route.lap_t / 60, 1), lap_m=round(lap_m),
                    min_batt_pct=round(100 * night["min_batt"], 1), packs=4)]

    files = dict(summary=summary, timeline=timeline, battery=battery, checkins=checkin_rows, checkpoints=checkpoints,
                 events=events, event_steps=steps, photos=photos, route=route_rows, map=map_rows)
    for k, rows in files.items():
        json.dump(rows, open(os.path.join(out, f"{k}.json"), "w"), ensure_ascii=False)
        print(f"{k}: {len(rows)} rows, {os.path.getsize(os.path.join(out, f'{k}.json')) / 1024:.0f} KB")
    print(json.dumps(summary[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
