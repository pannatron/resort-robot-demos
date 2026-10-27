"""Cut the customer pitch video: one robot, three resort jobs, 24 hours (chat-size mp4).

promo.py <results_dir> <out.mp4> <facts.json>
facts.json holds the measured numbers shown on the closing card (filled from the test results).
"""

import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from highlight import FONT as FF_FONT

W, H, FPS = 1280, 720, 25
T = "/usr/share/fonts/truetype/noto/NotoSansThai-"
L = "/usr/share/fonts/truetype/dejavu/DejaVuSans"


def fonts(size, bold):
    return (ImageFont.truetype(T + ("Bold.ttf" if bold else "Regular.ttf"), size),
            ImageFont.truetype(L + ("-Bold.ttf" if bold else ".ttf"), size - 2))


def text(d, xy, s, size=30, bold=False, fill=(255, 255, 255), anchor="la"):
    ft, fl = fonts(size, bold)
    runs, cur, th = [], "", None
    for ch in s:
        is_th = "฀" <= ch <= "๿" or (ch == " " and th)
        if th is None or is_th == th:
            cur += ch
        else:
            runs.append((cur, th))
            cur = ch
        th = is_th
    runs.append((cur, th))
    total = sum((ft if t_ else fl).getlength(r) for r, t_ in runs)
    asc, desc = fl.getmetrics()
    x, y = xy
    x -= {"l": 0, "m": total / 2, "r": total}[anchor[0]]
    base = y + asc if anchor[1] == "a" else y + (asc - desc) / 2
    for r, t_ in runs:
        f = ft if t_ else fl
        d.text((x, base), r, font=f, fill=fill, anchor="ls")
        x += f.getlength(r)


def backdrop(frame):
    img = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(8))
    ov = Image.new("RGB", img.size, (6, 10, 20))
    return Image.blend(img, ov, 0.62)


def card(frame, lines, seconds, w):
    """lines: list of (text, size, bold, color)."""
    img = backdrop(frame)
    d = ImageDraw.Draw(img)
    total_h = sum(int(s * 1.55) for _, s, _, _ in lines)
    y = (H - total_h) / 2
    for s, size, bold, col in lines:
        text(d, (W / 2, y + size * 0.75), s, size, bold, col, "mm")
        y += int(size * 1.55)
    emit_still(np.asarray(img), seconds, w)


FADE = 10  # frames


def emit_still(arr, seconds, w):
    n = int(seconds * FPS)
    for i in range(n):
        a = min(1.0, (i + 1) / FADE, (n - i) / FADE)
        w.append_data((arr.astype(np.float32) * a).astype(np.uint8))


RUN_H, CHARGE_H = 2.5, 1.5  # rated runtime (customer spec) / charge time (assumption, to confirm with Dobot)


def schedule_card(frame, seconds, w):
    """Jobs by time of day, and how two robots alternate work and charging to cover all 24 h."""
    img = backdrop(frame)
    d = ImageDraw.Draw(img)
    text(d, (W / 2, 52), "ตารางงาน 24 ชม. กับหุ่น 2 ตัวสลับกะ", 42, True, (255, 255, 255), "mm")
    x0, x1 = 210, W - 70

    def hx(h):
        return x0 + (x1 - x0) * h / 24

    NIGHT, MORN, FREE, RENT, CHG = (40, 90, 200), (230, 150, 40), (40, 160, 90), (170, 80, 200), (105, 105, 118)

    def job_color(h):
        return MORN if 6.5 <= h < 10 else FREE if 10 <= h < 22 else NIGHT

    # row 0: the work plan
    yb = 120
    text(d, (x0 - 16, yb + 22), "งาน", 24, True, (220, 225, 235), "rm")
    for a_, b_, c, lab in [(0, 6.5, NIGHT, "ลาดตระเวน"), (6.5, 10, MORN, "ส่งอาหารเช้า"),
                           (10, 22, FREE, "ต้อนรับ • นำทาง • ส่งของ"), (22, 24, NIGHT, "")]:
        d.rectangle([hx(a_), yb, hx(b_), yb + 44], fill=c)
        if lab:
            text(d, ((hx(a_) + hx(b_)) / 2, yb + 22), lab, 22, True, (255, 255, 255), "mm")
    for a_, b_ in ((7.0, 7.5), (16.5, 18.0)):
        d.rounded_rectangle([hx(a_), yb - 30, hx(b_), yb - 4], 6, fill=RENT)
    text(d, ((hx(16.5) + hx(18.0)) / 2, yb - 17), "เช่า", 18, True, (255, 255, 255), "mm")
    # rows A/B: run 2.5 h / charge 1.5 h, B offset by 2 h -> always at least one robot working
    cyc = RUN_H + CHARGE_H
    for r, (name, start) in enumerate((("หุ่น A", 0.0), ("หุ่น B", 2.0))):
        y = 200 + 62 * r
        text(d, (x0 - 16, y + 20), name, 24, True, (220, 225, 235), "rm")
        tt = start - cyc
        while tt < 24:
            for a_, b_, work in ((tt, tt + RUN_H, True), (tt + RUN_H, tt + cyc, False)):
                a2, b2 = max(a_, 0), min(b_, 24)
                if b2 <= a2:
                    continue
                if work:  # color each working stretch by the job it serves
                    h = a2
                    while h < b2:
                        h2 = min(b2, np.floor(h * 2 + 1e-6) / 2 + 0.5)
                        d.rectangle([hx(h), y, hx(h2), y + 40], fill=job_color(h))
                        h = h2
                else:
                    d.rectangle([hx(a2), y, hx(b2), y + 40], fill=CHG)
                    if b2 - a2 >= 1.2:
                        text(d, ((hx(a2) + hx(b2)) / 2, y + 20), "ชาร์จ", 18, False, (230, 230, 235), "mm")
            tt += cyc
    for h in range(0, 25, 3):
        d.line([hx(h), 326, hx(h), 336], fill=(200, 200, 200), width=2)
        text(d, (hx(h), 352), f"{h:02d}:00", 18, False, (210, 210, 210), "mm")
    rows = [("1 ตัว", f"วิ่ง {RUN_H:g} ชม. → ชาร์จ ~{CHARGE_H:g} ชม.  ⇒  ทำงานได้ ~{24 * RUN_H / cyc:.0f} ชม./วัน (มีช่วงว่าง)"),
            ("2 ตัว", "สลับกันวิ่ง-ชาร์จ  ⇒  มีหุ่นทำงานตลอด 24 ชม. ไม่ขาดช่วง (แนะนำ)"),
            ("แบต", f"1 ลูก ≈ {RUN_H:g} ชม. ≈ 26 รอบลาดตระเวน (233 ม./รอบ) — กลับแท่นชาร์จเองที่ป้อม")]
    for i, (head, body) in enumerate(rows):
        y = 420 + 58 * i
        d.rounded_rectangle([110, y - 21, 220, y + 21], 8, fill=(255, 205, 110) if i == 1 else (70, 80, 100))
        text(d, (165, y), head, 24, True, (20, 20, 20) if i == 1 else (255, 255, 255), "mm")
        text(d, (240, y), body, 26, False, (255, 255, 255), "lm")
    text(d, (W / 2, 672), f"เวลาชาร์จ ~{CHARGE_H:g} ชม. เป็นค่าประมาณ (เทียบหุ่นล้อ-ขาคลาสเดียวกัน) ต้องยืนยันกับสเปก Dobot",
         19, False, (170, 180, 195), "mm")
    emit_still(np.asarray(img), seconds, w)


def classify(f):
    row = f[41, 140:420].astype(int)  # clean HUD: the mode pill turns orange in walk mode / while switching
    walk = int(((row[:, 0] > 200) & (row[:, 1] > 120) & (row[:, 1] < 200) & (row[:, 2] < 90)).sum()) > 25
    col = f[:, 640].astype(int)
    red = np.any((col[:, 0] > 180) & (col[:, 1] < 60) & (col[:, 2] < 60))
    green = np.any((col[:, 0] < 70) & (col[:, 1] > 130) & (col[:, 2] < 100) & (col[:, 1] - col[:, 0] > 80))
    cam = f[22, 24, 0] > 200 and f[22, 24, 1] < 80 and f[136, 40].sum() < 200
    return bool(walk or cam), bool(walk and not (red or green or cam)), bool(red or green)


def clip(path, w, fast=8, pad_s=2.5, gap_s=0.0, window=None, head_s=3.0, tail_s=3.0, budget_s=None):
    """Normal speed around interesting frames, fast-forward elsewhere; ``window`` limits to a time range."""
    flags = [(*classify(f), bool(np.any((f[:, 640, 0] > 180) & (f[:, 640, 1] < 60) & (f[:, 640, 2] < 60))))
             for f in imageio.get_reader(path)]
    keep = np.array([a for a, _, _, _ in flags])
    walk_only = np.array([b for _, b, _, _ in flags])
    toast = np.array([c for _, _, c, _ in flags])
    run = 0  # only the first 2.5 s of each on-screen toast is worth real time
    for i in range(len(toast)):
        run = run + 1 if toast[i] else 0
        if toast[i] and run <= int(2.5 * FPS):
            keep[i] = True
    n = len(keep)
    if window == "alert":  # from shortly before the first red alert to shortly after the last one
        red = np.nonzero([d_ for _, _, _, d_ in flags])[0]
        window = (max(0, red[0] / FPS - 7.0), min(n / FPS, red[-1] / FPS + 5.0))
    pad = int(pad_s * FPS)
    dil = keep.copy()
    for i in np.nonzero(keep)[0]:
        dil[max(0, i - pad): i + pad + 1] = True
    on = np.nonzero(dil)[0]
    for a, b in zip(on[:-1], on[1:]):
        if 1 < b - a <= gap_s * FPS:
            dil[a:b] = True
    dil[: int(head_s * FPS)] = True
    dil[max(0, n - int(tail_s * FPS)):] = True
    lo, hi = (0, n) if window is None else (int(window[0] * FPS), int(window[1] * FPS))
    if budget_s is not None:  # speed up the fast-forward parts until the clip fits its time budget
        n1 = int(dil[lo:hi].sum())
        n2 = (hi - lo) - n1
        # stair walking is long; play it at 2-3x when the clip would not fit otherwise
        n_walk = int((dil[lo:hi] & walk_only[lo:hi]).sum())
        walk_step = 1
        while walk_step < 3 and n1 - n_walk + n_walk / walk_step > 0.8 * budget_s * FPS:
            walk_step += 1
        n1 = n1 - n_walk + n_walk / walk_step
        room = budget_s * FPS - n1
        fast = int(min(30, max(fast, np.ceil(n2 / max(room, 25)))))
        print(f"  {os.path.basename(path)}: normal {n1 / FPS:.1f}s (stairs x{walk_step}), ff {n2 / FPS:.1f}s "
              f"at x{fast}", flush=True)
    else:
        walk_step = 1
    nxt, last = lo, None
    out_frames = []  # small tail buffer for the fade-out

    class _W:
        count = 0

        def append_data(self, f):
            a = min(1.0, (self.count + 1) / FADE)
            self.count += 1
            out_frames.append((f.astype(np.float32) * a).astype(np.uint8))
            if len(out_frames) > FADE:
                real_w.append_data(out_frames.pop(0))

    real_w, w = w, _W()
    for i, f in enumerate(imageio.get_reader(path)):
        if i < nxt or i < lo:
            continue
        if i >= hi:
            break
        if dil[i] and walk_only[i] and walk_step > 1:
            img = Image.fromarray(f)
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([590, 640, 690, 690], 10, fill=(0, 0, 0))
            d.text((640, 665), f"x{walk_step}", font=FF_FONT, fill=(255, 255, 255), anchor="mm")
            w.append_data(np.asarray(img))
            nxt = i + walk_step
        elif dil[i]:
            w.append_data(f)
            nxt = i + 1
        else:
            img = Image.fromarray(f)
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([560, 640, 720, 690], 10, fill=(0, 0, 0))
            d.text((640, 665), f"▶▶ x{fast}", font=FF_FONT, fill=(255, 255, 255), anchor="mm")
            w.append_data(np.asarray(img))
            nxt = i + fast
        last = f
    for i, f in enumerate(out_frames):
        a = (len(out_frames) - i) / (len(out_frames) + 1)
        real_w.append_data((f.astype(np.float32) * a).astype(np.uint8))
    return last


def first_frame(path, t=1.0):
    return imageio.get_reader(path).get_data(int(t * FPS))


def main():
    res, out, facts = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    cam = os.path.join(res, "camera", "camera.mp4")
    bfk = os.path.join(res, "breakfast", "breakfast.mp4")
    gde = os.path.join(res, "guide", "guide.mp4")
    pat = os.path.join(res, "s2", "scenario2.mp4")
    w = imageio.get_writer(out, fps=FPS, codec="libx264", macro_block_size=8,
                           ffmpeg_params=["-crf", "27", "-preset", "slow", "-movflags", "+faststart"])
    WHITE, GOLD, GREY = (255, 255, 255), (255, 205, 110), (190, 200, 215)
    card(first_frame(pat, 200), [("หุ่น 1 ตัว  •  ทำงานให้รีสอร์ต 24 ชั่วโมง", 54, True, WHITE),
                                 ("DOBOT Wheel-Leg", 36, False, GOLD),
                                 ("ล้อวิ่งทางยาว • ขาเดินขึ้นลงบันได • สลับโหมดเอง", 28, False, GREY)], 3.5, w)
    schedule_card(first_frame(gde, 20), 9.0, w)

    def section(no, when, name, desc, bg):
        card(bg, [(f"{no}/5   {when}", 32, False, GOLD), (name, 52, True, WHITE), (desc, 28, False, GREY)], 2.2, w)

    section(1, "เช้า", "ส่งอาหารเช้าถึงวิลล่า", "แบก 3 กก. ลงบันไดเอง ผ่านสระ ถึงหน้าบ้านแขก", first_frame(bfk, 60))
    clip(bfk, w, fast=8, pad_s=1.5, gap_s=3, head_s=3, tail_s=4, budget_s=18)
    section(2, "ตอนว่าง", "ต้อนรับ & นำทางแขกไปห้อง", "ถือกระเป๋า พาไปห้อง 306 แล้วกลับไปรอที่ล็อบบี้", first_frame(gde, 30))
    clip(gde, w, fast=4, pad_s=1.5, gap_s=4, head_s=3, tail_s=3, budget_s=15)
    section(3, "มีคนเช่า", "Camera Man ให้แขกเช่า", "ตามถ่ายจากวิลล่าลงชายหาด แล้วส่งคลิปให้ทาง LINE", first_frame(cam, 40))
    clip(cam, w, fast=3, pad_s=1.5, gap_s=2, head_s=3, tail_s=3, budget_s=15)
    section(4, "กลางคืน", "ลาดตระเวน + แจ้ง รปภ.", "พบคนในสระหลังปิด → ถ่ายภาพ → แจ้งเตือน → รปภ. มาถึง", first_frame(pat, 120))
    clip(pat, w, fast=4, pad_s=2, gap_s=0, window="alert", head_s=0, tail_s=0, budget_s=19)
    swp = os.path.join(res, "s5", "scenario5.mp4")
    if os.path.exists(swp):
        section(5, "ตลอดคืน", "2 ตัวสลับกะ • ชาร์จเอง", "แบตต่ำกลับเทียบแท่นชาร์จเอง อีกตัวออกรับกะทันที", first_frame(swp, 20))
        clip(swp, w, fast=3, pad_s=2, gap_s=40, head_s=3, tail_s=3, budget_s=16)
    lines = [("ผลทดสอบในซิมูเลชัน", 44, True, WHITE)] + [(s, 28, False, GREY) for s in facts["lines"]] + [
        ("ภาพจำลอง ยังไม่ใช่การทดสอบกับหุ่นจริง", 22, False, (170, 175, 185))]
    card(first_frame(pat, 30), lines, 5.5, w)
    w.close()
    r = imageio.get_reader(out)
    print(f"{out}: {r.count_frames() / FPS:.0f}s, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
