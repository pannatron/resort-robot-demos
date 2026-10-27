"""Nightly report card for the resort owner: coverage, laps, check-ins, incidents with photos, batteries.

Uses one simulated night of the 2-robot fleet (fleet_sim timelines) plus the incident events and frames from the
night incident take (scenario 6).

nightly_report.py <fleet_dir> <incident_dir> <out.png>
"""

import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_card import COL  # noqa: E402
from promo import text  # noqa: E402

W, H = 1280, 720


def main():
    fd, idir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fs = json.load(open(os.path.join(fd, "fleet_summary.json")))
    night = json.load(open(os.path.join(fd, "fleet_timelines.json")))["2robots"]
    lap_s = fs["lap_s"]
    patrol_s = sum(b - a for tl in night["timeline"] for a, b, s in tl if s == "PATROL")
    swaps = sum(1 for tl in night["timeline"] for a, b, s in tl if s == "SWAP")
    laps = patrol_s / lap_s
    events = [json.loads(x) for x in open(os.path.join(idir, "events.jsonl"))]
    voice = json.load(open(os.path.join(idir, "scenario6_voice.json"))) if os.path.exists(
        os.path.join(idir, "scenario6_voice.json")) else []

    img = Image.new("RGB", (W, H), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 78], fill=(16, 24, 40))
    text(d, (32, 39), "รายงานประจำคืน  •  22 ต.ค. 22:00 – 23 ต.ค. 06:00", 30, True, (255, 255, 255), "lm")
    text(d, (W - 32, 39), "หุ่นลาดตระเวน 2 ตัว (ตัวอย่างจากซิมูเลชัน)", 20, False, (180, 190, 210), "rm")
    tiles = [(f"{100 * night['coverage']:.0f}%", "เวลาที่มีหุ่นลาดตระเวน"),
             (f"{laps:.0f} รอบ", f"ลาดตระเวน ({laps * 0.233:.1f} กม.)"),
             ("100%", "จุดเช็กอินครบทุกรอบ"),
             (f"{sum(1 for e in events if e['severity'] == 'warning')} เหตุ", "แจ้งเตือนที่ต้องดำเนินการ"),
             (f"{swaps} ครั้ง", "รปภ. สลับแบต"),
             ("0", "ครั้งที่หุ่นล้ม")]
    tw = (W - 64 - 5 * 14) / 6
    for i, (big, small) in enumerate(tiles):
        x0 = 32 + i * (tw + 14)
        d.rounded_rectangle([x0, 96, x0 + tw, 196], 12, fill=(255, 255, 255), outline=(220, 224, 230))
        text(d, (x0 + tw / 2, 132), big, 34, True, (16, 24, 40), "mm")
        text(d, (x0 + tw / 2, 174), small, 17, False, (90, 100, 115), "mm")
    # robot timeline
    text(d, (32, 222), "ตารางงานหุ่น", 20, True, (16, 24, 40), "lm")
    x0, x1 = 120, W - 32

    def hx(sec):
        return x0 + (x1 - x0) * sec / (8 * 3600)

    for i, tl in enumerate(night["timeline"]):
        y = 240 + 34 * i
        text(d, (x0 - 10, y + 13), f"หุ่น {'AB'[i]}", 18, True, (40, 50, 65), "rm")
        for a, b, s in tl:
            d.rectangle([hx(a), y, max(hx(b), hx(a) + 1), y + 26], fill=COL.get(s, (150, 150, 150)))
    for h in range(9):
        text(d, (hx(h * 3600), 318), f"{(22 + h) % 24:02d}:00", 14, False, (110, 120, 135), "mm")
    # incidents with photos taken from the take
    text(d, (32, 352), "เหตุการณ์ที่พบ (พร้อมภาพจากกล้องหุ่น)", 20, True, (16, 24, 40), "lm")
    vid = os.path.join(idir, "scenario6.mp4")
    r = imageio.get_reader(vid)
    n = r.count_frames()
    picks = [("intruder_pool_after_hours", "บุคคลในเขตสระหลังปิด", "ไฟส่อง + พูดเตือน TH/EN • รปภ. คุยผ่านลำโพง • ออกจากพื้นที่"),
             ("person_down", "พบคนล้มทางเดินสวน (เร่งด่วน)", "หุ่นนำชุดปฐมพยาบาลไปถึงก่อน รปภ. • แจ้งพยาบาล")]
    vstarts = {k: t for t, k in voice}
    for j, (etype, title, action) in enumerate(picks):
        e = next((x for x in events if x["type"] == etype), None)
        if e is None:
            continue
        key = "pool_warn_th" if etype.startswith("intruder") else "fall_alert_th"
        fidx = int(min(vstarts.get(key, 10.0) * 25 + 20, n - 1))
        thumb = Image.fromarray(r.get_data(fidx)).resize((288, 162))
        y = 372 + 168 * j
        img.paste(thumb, (32, y))
        text(d, (336, y + 16), f"{e['time'][11:16]}  •  {title}", 22, True, (16, 24, 40), "lm")
        text(d, (336, y + 52), f"โซน: {e['zone']}", 18, False, (70, 80, 95), "lm")
        text(d, (336, y + 84), action, 18, False, (70, 80, 95), "lm")
        resp = next((x["message"] for x in events if x["type"] in ("guard_arrived", "cleared")
                     and x["time"] > e["time"]), "")
        text(d, (336, y + 116), resp, 17, False, (30, 120, 70), "lm")
    # battery
    text(d, (W - 32, H - 22), f"แบตที่ใช้: {swaps} ก้อนตลอดคืน (ก้อนละ 2.5 ชม.) • ข้อมูลจำลองเพื่อการนำเสนอ", 15, False,
         (120, 128, 140), "rm")
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
