"""Safety-first pitch cut (with voice): night incident story, 2-robot battery-swap operation, daytime upsell.

pitch_safety.py <results_dir> <fleet_card.png> <out.mp4>
"""

import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import incidents  # noqa: E402
from highlight import FONT as FF_FONT  # noqa: E402
from promo import FADE, FPS, H, W, backdrop, classify, emit_still, text  # noqa: E402

WHITE, GOLD, GREY = (255, 255, 255), (255, 205, 110), (190, 200, 215)


class Out:
    """Counts written frames so audio can be placed on the final timeline."""

    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=8,
                                    ffmpeg_params=["-crf", "26", "-preset", "slow", "-movflags", "+faststart"])
        self.n = 0
        self.audio = []  # (out_time, key)

    def append_data(self, f):
        self.w.append_data(f)
        self.n += 1


def card(o, frame, lines, seconds):
    img = backdrop(frame)
    d = ImageDraw.Draw(img)
    total_h = sum(int(s * 1.55) for _, s, _, _ in lines)
    y = (H - total_h) / 2
    for s, size, bold, col in lines:
        text(d, (W / 2, y + size * 0.75), s, size, bold, col, "mm")
        y += int(size * 1.55)
    emit_still(np.asarray(img), seconds, o)


def image_card(o, path, seconds):
    emit_still(np.asarray(Image.open(path).convert("RGB").resize((W, H))), seconds, o)


def clip(o, path, budget_s, fast=8, pad_s=1.5, voice_json=None, head_s=2.0, tail_s=2.0):
    """Real time around alerts, mode switches and every spoken line; fast-forward the rest to fit the budget."""
    flags = [classify(f) for f in imageio.get_reader(path)]
    n = len(flags)
    keep = np.array([a for a, _, _ in flags])
    walk_only = np.array([b for _, b, _ in flags])
    toast = np.array([c for _, _, c in flags])
    run = 0
    for i in range(n):  # only the first 2.5 s of each toast
        run = run + 1 if toast[i] else 0
        keep[i] |= toast[i] and run <= int(2.5 * FPS)
    voice = []
    if voice_json and os.path.exists(voice_json):
        vt = incidents.VoiceTrack()
        for start, key in json.load(open(voice_json)):
            voice.append((start, key))
            a, b = int(start * FPS), int((start + vt.eff(key) + 0.4) * FPS)
            keep[a:min(b, n)] = True
    pad = int(pad_s * FPS)
    dil = keep.copy()
    for i in np.nonzero(keep)[0]:
        dil[max(0, i - pad): i + pad + 1] = True
    dil[: int(head_s * FPS)] = True
    dil[max(0, n - int(tail_s * FPS)):] = True
    n_walk = int((dil & walk_only).sum())
    walk_step = 1
    n1 = int(dil.sum())
    while walk_step < 3 and n1 - n_walk + n_walk / walk_step > 0.85 * budget_s * FPS:
        walk_step += 1
    n1 = n1 - n_walk + n_walk / walk_step
    fast = int(min(40, max(fast, np.ceil((n - dil.sum()) / max(budget_s * FPS - n1, 25)))))
    print(f"  {os.path.basename(path)}: real-time {n1 / FPS:.1f}s, rest x{fast}, stairs x{walk_step}", flush=True)
    src_to_out = {}
    buf = []  # fade-in/out buffer of (src index, frame)

    def push(i, f):
        buf.append((i, f))
        if len(buf) > FADE:
            j, g = buf.pop(0)
            src_to_out[j] = o.n
            o.append_data(g)

    nxt, written = 0, 0
    for i, f in enumerate(imageio.get_reader(path)):
        if i < nxt:
            continue
        a = min(1.0, (written + 1) / FADE)
        if dil[i] and walk_only[i] and walk_step > 1 and not any(int(s * FPS) <= i for s, _ in voice[:0]):
            img = Image.fromarray(f)
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([590, 640, 690, 690], 10, fill=(0, 0, 0))
            d.text((640, 665), f"x{walk_step}", font=FF_FONT, fill=(255, 255, 255), anchor="mm")
            g, nxt = np.asarray(img), i + walk_step
        elif dil[i]:
            g, nxt = f, i + 1
        else:
            img = Image.fromarray(f)
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([560, 640, 720, 690], 10, fill=(0, 0, 0))
            d.text((640, 665), f"▶▶ x{fast}", font=FF_FONT, fill=(255, 255, 255), anchor="mm")
            g, nxt = np.asarray(img), i + fast
        push(i, (np.asarray(g, np.float32) * a).astype(np.uint8))
        written += 1
    for k, (j, g) in enumerate(buf):
        src_to_out[j] = o.n
        o.append_data((g.astype(np.float32) * (len(buf) - k) / (len(buf) + 1)).astype(np.uint8))
    for start, key in voice:
        i = int(round(start * FPS))
        while i not in src_to_out and i < n:
            i += 1
        if i in src_to_out:
            o.audio.append((src_to_out[i] / FPS, key))


def frame_at(path, t):
    r = imageio.get_reader(path)
    return r.get_data(min(int(t * FPS), r.count_frames() - 1))


def main():
    res, fleet_png, out = sys.argv[1], sys.argv[2], sys.argv[3]
    s6, s5 = os.path.join(res, "s6", "scenario6.mp4"), os.path.join(res, "s5", "scenario5.mp4")
    bfk, gde, cam = (os.path.join(res, j, f"{j}.mp4") for j in ("breakfast", "guide", "camera"))
    tmp = out.replace(".mp4", "_silent.mp4")
    o = Out(tmp)
    card(o, frame_at(s6, 100), [("รปภ. คนที่สอง ที่ไม่ง่วง ไม่หลับ", 56, True, WHITE),
                                ("DOBOT Wheel-Leg  •  ลาดตระเวน + ช่วยเหลือฉุกเฉิน สำหรับรีสอร์ต", 30, False, GOLD),
                                ("ล้อวิ่งทางยาว • ขาขึ้นลงบันได • พูดเตือนได้ • คุยกับ รปภ. ได้", 26, False, GREY)], 3.8)
    card(o, frame_at(s6, 60), [("1/3  กลางคืน", 32, False, GOLD),
                               ("ลาดตระเวน → เจอเหตุ → พูดเตือน → นำของไปช่วย", 44, True, WHITE),
                               ("หลบแขกที่เดินสวน • ไฟส่อง • รปภ. พูดผ่านลำโพงหุ่น • ชุดปฐมพยาบาลถึงก่อน", 26, False, GREY)], 2.6)
    clip(o, s6, budget_s=62, fast=10, voice_json=s6.replace(".mp4", "_voice.json"), head_s=3, tail_s=3)
    card(o, frame_at(s5, 40), [("2/3  ทั้งคืนไม่มีช่วงว่าง", 32, False, GOLD),
                               ("หุ่น 2 ตัวผลัดกัน • รปภ. สลับแบตที่ป้อม", 46, True, WHITE),
                               ("แบตต่ำ → ส่งต่องานให้อีกตัว → กลับป้อม → สลับแบต → ตรวจตัวเอง → พร้อม", 26, False, GREY)], 2.6)
    clip(o, s5, budget_s=18, fast=6, voice_json=s5.replace(".mp4", "_voice.json"), head_s=2, tail_s=3)
    image_card(o, fleet_png, 7.0)
    card(o, frame_at(bfk, 60), [("3/3  กลางวัน (ข้อเสนอเพิ่ม)", 32, False, GOLD),
                                ("กลางคืนลาดตระเวน • กลางวันช่วยงานบริการ", 44, True, WHITE),
                                ("ส่งอาหารเช้า/ของใช้ • พาแขกไปห้อง • ให้เช่าเป็น Camera Man", 26, False, GREY)], 2.4)
    clip(o, bfk, budget_s=11, fast=20, head_s=2, tail_s=3)
    clip(o, gde, budget_s=8, fast=12, head_s=2, tail_s=2)
    clip(o, cam, budget_s=9, fast=8, head_s=2, tail_s=3)
    ev = [json.loads(x) for x in open(os.path.join(res, "s6", "events.jsonl"))]
    first = next((e["message"] for e in ev if e["type"] == "guard_arrived"), "")
    lead = first[first.find("หุ่นถึงก่อน"):first.find(")")] if "หุ่นถึงก่อน" in first else "หุ่นถึงจุดเกิดเหตุก่อน รปภ."
    card(o, frame_at(s6, 30), [("ผลทดสอบในซิมูเลชัน", 44, True, WHITE),
                               ("ลาดตระเวน 50 รอบแบบสุ่ม (พื้นเปียก • แรงผลัก • เก้าอี้ขวาง): สำเร็จ 50/50 ไม่ล้ม", 25, False, GREY),
                               ("หุ่น 2 ตัว + แบต 3–4 ก้อน: มีหุ่นลาดตระเวน ~100% ของคืน (จำลองทั้งกะ 20 คืน)", 25, False, GREY),
                               (f"เหตุคนล้ม: {lead} พร้อมชุดปฐมพยาบาล", 25, False, GREY),
                               ("ภาพจำลอง ยังไม่ใช่การทดสอบกับหุ่นจริง", 20, False, (160, 165, 175))], 6.0)
    o.w.close()
    vt = incidents.VoiceTrack()
    vt.events = o.audio
    vt.mux(tmp, out)
    os.remove(tmp)
    r = imageio.get_reader(out)
    print(f"{out}: {r.count_frames() / FPS:.0f}s, {os.path.getsize(out) / 1e6:.1f} MB, {len(o.audio)} voice lines")


if __name__ == "__main__":
    main()
