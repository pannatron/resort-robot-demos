"""Render the fleet-simulation result card (1 vs 2 robots, one simulated night + 20-night statistics).

fleet_card.py <fleet_dir> <out.png>
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promo import text  # noqa: E402  (mixed Thai/Latin text helper)

W, H = 1280, 720
COL = {"PATROL": (40, 120, 220), "RELIEF": (90, 160, 240), "RETURN_TO_BASE": (240, 160, 40),
       "EMERGENCY_RETURN": (230, 60, 50), "WAIT_SWAP": (200, 90, 60), "SWAP": (250, 220, 80),
       "SELF_CHECK": (180, 220, 120), "STANDBY": (70, 75, 90), "ASSIST": (200, 60, 200)}
LABEL = {"PATROL": "ลาดตระเวน", "RETURN_TO_BASE": "กลับป้อม", "WAIT_SWAP": "รอ รปภ.", "SWAP": "สลับแบต",
         "STANDBY": "สแตนด์บาย", "ASSIST": "ไปช่วยเหตุ"}


def main():
    fd, out = sys.argv[1], sys.argv[2]
    tl = json.load(open(os.path.join(fd, "fleet_timelines.json")))
    summ = json.load(open(os.path.join(fd, "fleet_summary.json")))["rows"]
    img = Image.new("RGB", (W, H), (12, 16, 26))
    d = ImageDraw.Draw(img)
    text(d, (W / 2, 44), "จำลองทั้งกะ 22:00–06:00 • รปภ. เป็นคนสลับแบต (หุ่นชาร์จเองไม่ได้)", 34, True, (255, 255, 255), "mm")
    x0, x1 = 230, W - 50

    def hx(sec):
        return x0 + (x1 - x0) * sec / (8 * 3600)

    y = 100
    for key, title in (("1robots", "หุ่น 1 ตัว"), ("2robots", "หุ่น 2 ตัว")):
        res = tl[key]
        cov = res["coverage"] * 100
        text(d, (40, y + 8), f"{title}  •  มีหุ่นลาดตระเวน {cov:.1f}% ของคืน", 24, True, (255, 205, 110), "la")
        y += 44
        for i, rows in enumerate(res["timeline"]):
            text(d, (x0 - 14, y + 16), f"หุ่น {'AB'[i]}", 22, True, (220, 225, 235), "rm")
            for a, b, s in rows:
                d.rectangle([hx(a), y, max(hx(b), hx(a) + 1), y + 32], fill=COL.get(s, (90, 90, 90)))
            y += 40
        for a, b in res["gaps"]:  # no robot patrolling
            d.rectangle([hx(a), y + 2, max(hx(b), hx(a) + 2), y + 16], fill=(235, 50, 50))
        text(d, (x0 - 14, y + 9), "ช่องว่าง", 18, False, (235, 120, 120), "rm")
        y += 40
    for h in range(0, 9):
        d.line([hx(h * 3600), y, hx(h * 3600), y + 8], fill=(200, 200, 200), width=2)
        text(d, (hx(h * 3600), y + 22), f"{(22 + h) % 24:02d}:00", 16, False, (200, 200, 200), "mm")
    lx = x0
    for s in ("PATROL", "RETURN_TO_BASE", "WAIT_SWAP", "SWAP", "STANDBY", "ASSIST"):
        d.rectangle([lx, y + 44, lx + 18, y + 62], fill=COL[s])
        text(d, (lx + 24, y + 53), LABEL[s], 17, False, (220, 225, 235), "lm")
        lx += 150
    # 20-night statistics
    y += 90

    def row(r, c, p):
        return next(x for x in summ if x["robots"] == r and x["charge_h"] == c and x["packs"] == p)

    cells = [("1 ตัว + แบต 2 ก้อน", row(1, 1.5, 2)), ("1 ตัว + แบต 3 ก้อน (ชาร์จ 3 ชม.)", row(1, 3.0, 3)),
             ("2 ตัว + แบต 3 ก้อน", row(2, 1.5, 3)), ("2 ตัว + แบต 4 ก้อน (ชาร์จ 3 ชม.)", row(2, 3.0, 4))]
    hdr = ["ชุดที่จำลอง (20 คืน/ชุด)", "มีหุ่นลาดตระเวน", "ช่องว่าง/คืน", "สลับแบต/คืน", "รปภ. มาช้า >8 นาที"]
    xs = [40, 470, 680, 860, 1040]
    for x, hcell in zip(xs, hdr):
        text(d, (x, y), hcell, 19, True, (180, 190, 205), "la")
    for i, (name, r) in enumerate(cells):
        yy = y + 32 + 30 * i
        best = r["robots"] == 2
        for x, v in zip(xs, [name, f"{100 * r['coverage']:.1f}% (แย่สุด {100 * r['coverage_min']:.1f}%)",
                              f"{r['gap_min']:.1f} นาที", f"{r['swaps']:.0f} ครั้ง", f"{r['escalations']:.1f} ครั้ง"]):
            text(d, (x, yy), v, 19, best, (255, 255, 255) if best else (210, 215, 225), "la")
    text(d, (W / 2, H - 22), "แบต 1 ก้อน = 2.5 ชม. ลาดตระเวน • กลับป้อมเมื่อแบตเหลือ = พลังงานเดินกลับ + 15% • รปภ. มาสลับแบตใน 1–10 นาที (สุ่ม)",
         16, False, (150, 160, 175), "mm")
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
