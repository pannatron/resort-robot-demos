"""Cut the RPO pitch video: what the robot can / cannot do, four resort jobs, measured numbers, RPO vs Dobot.

rpo_pitch.py <results_dir> <out.mp4>
<results_dir>/<job>/<job>.mp4 + result_<job>.json for job in welcome, delivery, party, bump.
Moments play at 1x with their voice / music; long walks run fast-forward (muted, with a speed tag).
Numbers on the cards come from the result files and from tests.json (skill tests, see make_tests.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/dobot_tools/resort"))
from highlight import FONT as FF_FONT  # noqa: E402
from incidents import audio_duration, ffmpeg_exe  # noqa: E402
from promo import FADE, FPS, H, W, backdrop, text  # noqa: E402

WHITE, GOLD, GREY, GREEN, RED = (255, 255, 255), (255, 205, 110), (190, 200, 215), (120, 220, 140), (255, 120, 110)


class Cut:
    """Writes frames and remembers which source-audio ranges play where on the output timeline."""

    def __init__(self, path):
        self.path = path
        self.w = imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=8)
        self.n = 0
        self.audio = []  # (src_mp4, src_a, src_b, out_t)
        self.last = None

    def put(self, arr):
        self.w.append_data(arr)
        self.n += 1
        self.last = arr

    @property
    def t(self):
        return self.n / FPS

    def still(self, arr, seconds):
        n = int(seconds * FPS)
        for i in range(n):
            a = min(1.0, (i + 1) / FADE, (n - i) / FADE)
            self.put((arr.astype(np.float32) * a).astype(np.uint8))

    def segment(self, src, a, b, speed=1, label=None):
        r = imageio.get_reader(src)
        nfr = r.count_frames()
        f0, f1 = int(a * FPS), min(int(b * FPS), nfr - 1)
        if speed == 1:
            self.audio.append((src, f0 / FPS, f1 / FPS, self.t))
        for f in range(f0, f1, speed):
            fr = r.get_data(f)
            if speed > 1 or label:
                img = Image.fromarray(fr)
                d = ImageDraw.Draw(img)
                tag = label or f"x{speed} ▶▶"
                d.rounded_rectangle([W - 190, 74, W - 24, 124], 14, fill=(10, 14, 24))
                d.text((W - 107, 99), tag, font=FF_FONT, fill=WHITE, anchor="mm")
                fr = np.asarray(img)
            self.put(fr)
        r.close()

    def close(self):
        self.w.close()
        if not self.audio:
            return
        tmp = self.path.replace(".mp4", "_silent.mp4")
        os.replace(self.path, tmp)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp]
        parts = []
        for j, (src, a, b, at) in enumerate(self.audio):
            cmd += ["-i", src]
            ms = int(at * 1000)
            parts.append(f"[{j + 1}:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:d=0.15,afade=t=out:st={max(b - a - 0.25, 0):.3f}:d=0.25,adelay={ms}|{ms}[a{j}]")
        mix = "".join(f"[a{j}]" for j in range(len(self.audio)))
        vdur = audio_duration(tmp)
        fc = ";".join(parts) + (f";{mix}amix=inputs={len(self.audio)}:normalize=0,"
                                f"apad=whole_dur={vdur:.3f},atrim=0:{vdur:.3f}[aout]")
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a",
                "128k", self.path]
        subprocess.run(cmd, check=True)
        os.remove(tmp)
        # imageio writes a high-bitrate stream: re-encode to a chat-size H.264 file
        big = self.path.replace(".mp4", "_hq.mp4")
        os.replace(self.path, big)
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", big, "-c:v", "libx264", "-crf", "24",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "copy",
                        self.path], check=True)
        os.remove(big)


def has_audio(path):
    out = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True).stderr
    return "Audio:" in out


def frame_at(path, t):
    r = imageio.get_reader(path)
    f = r.get_data(min(int(t * FPS), r.count_frames() - 1))
    r.close()
    return f


def ev(res, etype, default=None, last=False):
    hits = [e["video_s"] for e in res["events"] if e["type"] == etype]
    return (hits[-1] if last else hits[0]) if hits else default


# ------------------------------------------------------------------------------------------ cards


def card(cut, bg, lines, seconds):
    """lines: (text, size, bold, color)."""
    img = backdrop(bg)
    d = ImageDraw.Draw(img)
    total_h = sum(int(s * 1.55) for _, s, _, _ in lines)
    y = (H - total_h) / 2
    for s, size, bold, col in lines:
        text(d, (W / 2, y + size * 0.75), s, size, bold, col, "mm")
        y += int(size * 1.55)
    cut.still(np.asarray(img), seconds)


def section(cut, bg, num, title, sub, seconds=2.6):
    card(cut, bg, [(f"งานที่ {num}", 28, True, GOLD), (title, 54, True, WHITE), (sub, 28, False, GREY)], seconds)


def can_card(cut, bg, tests, seconds=9.0):
    img = backdrop(bg)
    d = ImageDraw.Draw(img)
    text(d, (W / 2, 58), "RPO ทำอะไรได้บ้าง (ทดสอบในแบบจำลองแล้ว)", 40, True, WHITE, "mm")
    can = [("เดิน / วิ่ง", f"เดิน {tests['walk_mps']:.1f} ม./วิ  •  วิ่ง {tests['run_mps']:.1f} ม./วิ"),
           ("ขึ้น-ลงบันได", f"บันได {tests['stairs_cm']} ซม. × {tests['stairs_n']} ขั้น ขึ้นและลง ไม่ล้ม"),
           ("โบกมือ / เต้น", "ท่าที่ Roboparty เทรนมา: โบกมือ 1 แบบ, เต้น 2 ชุด (~20 วิ/ชุด)"),
           ("ล้มแล้วลุกเอง", f"ล้มหงาย/ตะแคง ลุกเองใน ~{tests['getup_s']:.0f} วิ"),
           ("แบกของบนตัว", f"ถาด/เป้ ทดสอบถึง {tests['payload_ok_kg']:.0f} กก. • ใช้งานจริงแนะนำ ≤ {tests['payload_rec_kg']:.0f} กก.")]
    cant = [("ไม่มีมือ", "หยิบ-จับ-วางของเองไม่ได้ (แขนจบที่ข้อศอก) → พนักงานวางของ / แขกหยิบเอง"),
            ("ล้มคว่ำหน้า", "ลุกเองได้ใน MuJoCo แต่ยังไม่ผ่านใน Newton → ต้องทดสอบเพิ่มก่อนใช้งานจริง"),
            ("งานลาดตระเวนยาว", "เดิน 2 ขา กินไฟกว่าหุ่นล้อ-ขา → เหมาะงานในอาคารระยะสั้น")]
    y = 128
    for head, body in can:
        d.rounded_rectangle([70, y - 22, 300, y + 22], 10, fill=(40, 150, 80))
        text(d, (185, y), "✓ " + head, 24, True, WHITE, "mm")
        text(d, (320, y), body, 25, False, WHITE, "lm")
        y += 62
    y += 14
    for head, body in cant:
        d.rounded_rectangle([70, y - 22, 300, y + 22], 10, fill=(170, 60, 50))
        text(d, (185, y), "✗ " + head, 24, True, WHITE, "mm")
        text(d, (320, y), body, 24, False, (255, 225, 220), "lm")
        y += 62
    cut.still(np.asarray(img), seconds)


def facts_card(cut, bg, R, seconds=8.0):
    img = backdrop(bg)
    d = ImageDraw.Draw(img)
    text(d, (W / 2, 64), "ผลทดสอบจากแบบจำลอง (Isaac Lab + Newton)", 40, True, WHITE, "mm")
    w, dl, b, p = R["welcome"], R["delivery"], R["bump"], R["party"]
    import re

    msg = next(e["message"] for e in dl["events"] if e["type"] == "delivered")  # same number as the in-video toast
    trip_min = float(re.search(r"([\d.]+) นาที", msg).group(1))
    rows = [("ต้อนรับ + นำทาง", f"ทักทาย TH/EN แล้วเดินนำแขก {w['distance_m']:.0f} ม. ถึงห้อง 306 • ไม่ล้ม"),
            ("รูมเซอร์วิส", f"ถาด {dl['payload_kg']:.0f} กก. • {dl['distance_m']:.0f} ม. รวมลงบันได • {trip_min:.1f} นาที • ไม่ล้ม"),
            ("ปาร์ตี้วันเกิด", f"ส่งเค้ก + เต้น 2 ชุด + ท่าปิด • ไม่ล้ม (เอียงสุด {np.degrees(np.arccos(p['min_upright'])):.0f}°)"),
            ("ล้มแล้วลุกเอง", f"แรงชน {b['push_N']:.0f} N → ล้มหงาย → ยืนได้ใน {b['fall_to_up_s']:.1f} วิ")]
    y = 160
    for head, body in rows:
        d.rounded_rectangle([90, y - 30, 360, y + 30], 12, fill=(60, 70, 95))
        text(d, (225, y), head, 26, True, WHITE, "mm")
        text(d, (385, y), body, 27, False, WHITE, "lm")
        y += 96
    text(d, (W / 2, 640), "ใช้ policy ที่ Roboparty เปิดให้ใช้ (ONNX) ทั้งหมด ไม่ได้เทรนเพิ่ม • ผลจริงบนหุ่นต้องทดสอบหน้างาน",
         22, False, GREY, "mm")
    cut.still(np.asarray(img), seconds)


def compare_card(cut, bg, dobot, seconds=9.0):
    img = backdrop(bg)
    d = ImageDraw.Draw(img)
    text(d, (W / 2, 56), "RPO กับ Dobot: เลือกหุ่นให้ตรงงาน", 40, True, WHITE, "mm")
    cols = [(400, "RPO (หุ่นคน 2 ขา)", (120, 200, 255)), (840, "Dobot (ล้อ-ขา 4 ขา)", (255, 190, 90))]
    for x, h, c in cols:
        text(d, (x, 120), h, 28, True, c, "lm")
    rows = [("ต้อนรับ / ทักทาย", "★★★ ท่าทางเหมือนคน โบกมือได้", "★★ พูดทักทาย ไม่มีท่ามือ"),
            ("โชว์ / ปาร์ตี้", "★★★ เต้นท่าคน", "★★ เต้นบนล้อ ยืน 2 ขา"),
            ("นำทางในอาคาร", "★★★", "★★★"),
            ("ส่งของ", "★★ ≤ 5 กก. บนถาด", "★★★ " + dobot.get("payload", "3 กก.") + " เร็วกว่า"),
            ("ลาดตระเวนกลางคืน", "★ ระยะสั้น", "★★★ 26 รอบ/แบต"),
            ("หยิบจับของ", "✗ ไม่มีมือ", "✗ ไม่มีมือ")]
    y = 186
    for head, a, b in rows:
        d.line([60, y - 36, W - 60, y - 36], fill=(70, 80, 100), width=1)
        text(d, (80, y), head, 26, True, WHITE, "lm")
        text(d, (400, y), a, 25, False, WHITE, "lm")
        text(d, (840, y), b, 25, False, WHITE, "lm")
        y += 72
    text(d, (W / 2, 660), "แนะนำ: RPO งานหน้าบ้าน (ต้อนรับ โชว์ นำทาง ส่งของเบา)  •  Dobot งานหลังบ้าน (ลาดตระเวน ส่งของ)",
         24, True, GOLD, "mm")
    cut.still(np.asarray(img), seconds)


# ------------------------------------------------------------------------------------------ main


def main():
    rdir, out = sys.argv[1], sys.argv[2]
    R, V = {}, {}
    for job in ("welcome", "delivery", "party", "bump"):
        R[job] = json.load(open(os.path.join(rdir, job, f"result_{job}.json")))
        V[job] = os.path.join(rdir, job, f"{job}.mp4")
    tests = json.load(open(os.path.join(HERE, "tests.json")))
    dobot = {"payload": "3 กก."}
    cut = Cut(out)
    bg0 = frame_at(V["welcome"], 10.0)
    card(cut, bg0, [("RPO Humanoid", 64, True, WHITE), ("ผู้ช่วยต้อนรับและสร้างความประทับใจในรีสอร์ต", 34, True, GOLD),
                    ("หุ่นคน 2 ขาโอเพนซอร์สของ Roboparty  •  ภาพจำลอง Isaac Lab + Newton", 24, False, GREY)], 4.0)
    can_card(cut, frame_at(V["party"], 70.0), tests)

    # 1 welcome
    w = R["welcome"]
    g, ci, ar = ev(w, "greet"), ev(w, "checkin"), ev(w, "arrived")
    section(cut, frame_at(V["welcome"], g + 3), 1, "ต้อนรับ & นำทางแขก", "โบกมือทักทาย 2 ภาษา แล้วเดินนำไปถึงห้อง")
    cut.segment(V["welcome"], g - 2.0, g + 11.5)
    cut.segment(V["welcome"], ci - 0.5, ci + 6.0)
    cut.segment(V["welcome"], ci + 6.0, ar - 3.0, speed=6)
    cut.segment(V["welcome"], ar - 3.0, w["video_s"])

    # 2 delivery
    dl = R["delivery"]
    pk, arr = ev(dl, "pickup"), ev(dl, "arrived")
    section(cut, frame_at(V["delivery"], 3.0), 2, "รูมเซอร์วิส ส่งของถึงวิลล่า",
            f"ถาดหน้าอก {dl['payload_kg']:.0f} กก. • ลงบันได • {dl['distance_m']:.0f} ม.")
    cut.segment(V["delivery"], 0.0, pk + 5.0)
    t_st = 96.0  # top of the stairs (corridor end) — found from the job log: x reaches 57 m at ~96-100 s
    cut.segment(V["delivery"], pk + 5.0, t_st, speed=8)
    cut.segment(V["delivery"], t_st, t_st + 18.0, speed=2)
    cut.segment(V["delivery"], t_st + 18.0, arr - 2.0, speed=8)
    cut.segment(V["delivery"], arr - 2.0, dl["video_s"])

    # 3 party
    p = R["party"]
    pa, ck, cd, dn = ev(p, "arrived"), ev(p, "cake"), ev(p, "candles"), ev(p, "dance")
    fin, done = ev(p, "finale"), ev(p, "done")
    section(cut, frame_at(V["party"], dn + 8), 3, "ปาร์ตี้วันเกิดที่ลานสระ", "ส่งเค้ก • เปิดเพลง • โชว์เต้น (บริการเสริม)")
    cut.segment(V["party"], 1.0, pa, speed=3)
    cut.segment(V["party"], pa, ck + 4.0)
    cut.segment(V["party"], ck + 4.0, cd - 3.0, speed=4)
    cut.segment(V["party"], cd - 3.0, cd + 3.0)
    cut.segment(V["party"], cd + 3.0, dn - 1.0, speed=3)
    cut.segment(V["party"], dn - 1.0, dn + 24.0)
    cut.segment(V["party"], dn + 24.0, fin - 1.0, speed=2)
    cut.segment(V["party"], fin - 1.0, p["video_s"])

    # 4 bump
    b = R["bump"]
    section(cut, frame_at(V["bump"], ev(b, "fall") + 3), 4, "ล้มแล้วลุกเอง", "เด็กวิ่งชน → หุ่นล้ม → ลุกเองแล้วถามอาการ")
    cut.segment(V["bump"], 2.5, b["video_s"])

    facts_card(cut, frame_at(V["delivery"], arr + 2), R)
    compare_card(cut, frame_at(V["party"], dn + 12), dobot)
    card(cut, frame_at(V["welcome"], ar + 2), [("RPO = หน้าตาของรีสอร์ต", 52, True, WHITE),
                                              ("ต้อนรับ • นำทาง • ส่งของเบา • โชว์ • ปลอดภัย ลุกเองได้", 30, True, GOLD),
                                              ("ขั้นต่อไป: ทดสอบบนหุ่นจริงหน้างาน + เพิ่มมือจับ (gripper) ถ้าต้องหยิบของ", 24,
                                               False, GREY)], 5.0)
    cut.close()
    print(f"wrote {out}  {cut.t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
