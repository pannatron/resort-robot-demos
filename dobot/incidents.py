"""Incident-scene helpers: robot voice/guard intercom track, spotlight and beacon effects, lying person pose.

Effects are drawn in image space after rendering (projected from 3D), so they work the same for the OpenGL
viewer and the OVRTX path tracer, whose custom lights cannot move after the first frame.
"""

from __future__ import annotations

import math
import os
import re
import subprocess

import numpy as np

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice")
FPS = 25

# key -> (speaker, subtitle); speaker "robot" = robot TTS, "guard" = guard on the booth intercom via robot speaker
LINES = {
    "pool_warn_th": ("robot", "สระว่ายน้ำปิดให้บริการแล้ว กรุณาออกจากพื้นที่ค่ะ"),
    "pool_warn_en": ("robot", "The swimming pool is closed. Please leave the area."),
    "guard_talk_th": ("guard", "สวัสดีครับ รปภ. พูดครับ สระปิดตั้งแต่สองทุ่มแล้ว รบกวนกลับห้องพักนะครับ ขอบคุณครับ"),
    "fall_alert_th": ("robot", "คุณเป็นอะไรไหมคะ หุ่นยนต์แจ้งเจ้าหน้าที่แล้ว ความช่วยเหลือกำลังมาค่ะ"),
    "fall_alert_en": ("robot", "Are you okay? Help is on the way. I have alerted the staff."),
    "kit_th": ("robot", "ชุดปฐมพยาบาลอยู่บนหลังหุ่นยนต์ค่ะ"),
    "guard_swap_th": ("robot", "หุ่นยนต์หมายเลขหนึ่ง แบตเตอรี่ต่ำ กรุณาเปลี่ยนแบตเตอรี่ที่ป้อมค่ะ"),
    # birthday surprise
    "party_arrive_th": ("robot", "สุขสันต์วันเกิดนะคะ เค้กวันเกิดมาส่งแล้วค่ะ"),
    "party_wish_th": ("robot", "อธิษฐาน แล้วเป่าเทียนได้เลยค่ะ"),
    "party_dance_th": ("robot", "ขอโชว์เต้นฉลองให้หน่อยนะคะ"),
    "party_dance_en": ("robot", "Let's party!"),
    "party_finale_th": ("robot", "ขอให้มีความสุขมาก ๆ นะคะ"),
    "party_finale_en": ("robot", "Happy birthday!"),
    # music (party_audio.py): plays under the voice, no subtitle
    "tab_go_th": ("robot", "รับคำสั่งจาก รปภ. ค่ะ กำลังไปตรวจสอบหลังวิลล่า"),
    "tab_arrive_th": ("robot", "ถึงจุดที่สั่งแล้วค่ะ"),
    "tab_manual_th": ("robot", "เปลี่ยนเป็นโหมดบังคับโดย รปภ. ค่ะ"),
    "guard_lost_th": ("guard", "สวัสดีครับ รปภ. พูดครับ หาห้องพักอยู่หรือเปล่าครับ วิลล่าหนึ่งเดินอ้อมไปทางด้านหน้าได้เลยครับ"),
    "tab_resume_th": ("robot", "กลับไปลาดตระเวนต่อค่ะ"),
    "music_hbd": ("music", None),
    "music_dance": ("music", None),
}
MUSIC_VOLUME = 0.45


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration(path):
    out = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 2.0


class VoiceTrack:
    """Schedules voice clips on the video timeline, shows subtitles, and muxes the audio at the end."""

    GUARD_FILTER = "asetrate=24000*0.84,aresample=24000,atempo=1.12,highpass=f=250,lowpass=f=3400"  # male-ish, radio

    def __init__(self):
        self.events = []  # (video_time, key)
        self.dur = {k: audio_duration(os.path.join(VOICE_DIR, k + ".mp3")) for k in LINES}
        self.busy_until = 0.0

    def say(self, video_t, key):
        """Queue a line; lines never overlap (each starts after the previous one ends)."""
        start = max(video_t, self.busy_until)
        d = self.eff(key)
        self.events.append((start, key))
        self.busy_until = start + d + 0.3
        return start + d  # end time

    def play(self, video_t, key):
        """Start a music track; it does not hold back the voice lines."""
        self.events.append((video_t, key))
        return video_t + self.dur[key]

    def eff(self, key):
        """Played duration (the guard filter changes the speed by 0.84 x 1.12)."""
        return self.dur[key] / (0.84 * 1.12) if LINES[key][0] == "guard" else self.dur[key]

    def subtitle(self, video_t):
        for start, key in self.events:
            d = self.eff(key)
            if LINES[key][1] is not None and start <= video_t <= start + d + 0.4:
                return LINES[key]
        return None

    def done_by(self, video_t):
        return video_t >= self.busy_until

    def mux(self, video_path, out_path):
        if not self.events:
            os.replace(video_path, out_path)
            return
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video_path]
        parts = []
        for j, (start, key) in enumerate(self.events):
            cmd += ["-i", os.path.join(VOICE_DIR, key + ".mp3")]
            f = {"guard": self.GUARD_FILTER + ",", "music": f"volume={MUSIC_VOLUME},"}.get(LINES[key][0], "")
            parts.append(f"[{j + 1}:a]{f}adelay={int(start * 1000)}|{int(start * 1000)}[a{j}]")
        mix = "".join(f"[a{j}]" for j in range(len(self.events)))
            # pad with silence to exactly the video length (apad + -shortest hangs ffmpeg with a copied video stream)
        vdur = audio_duration(video_path)
        fc = ";".join(parts) + (f";{mix}amix=inputs={len(self.events)}:normalize=0,volume=1.6,"
                                f"apad=whole_dur={vdur:.3f},atrim=0:{vdur:.3f}[aout]")
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                out_path]
        subprocess.run(cmd, check=True)


# ------------------------------------------------------------------------------------------ image-space light FX


def _project(night, viewer, pts, w, h):
    u, v, depth, front = night.project(viewer, np.asarray(pts, float), w, h)
    return u, v, depth, front


def add_spotlight(night, viewer, frame, src_w, tgt_w, radius=1.1, strength=0.9):
    """Warm beam from the robot's lamp to a lit ellipse on the ground around tgt_w (world coords)."""
    from PIL import Image, ImageDraw, ImageFilter

    h, w = frame.shape[:2]
    ang = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    ring = np.stack([tgt_w[0] + radius * np.cos(ang), tgt_w[1] + radius * np.sin(ang),
                     np.full_like(ang, tgt_w[2])], 1)
    u, v, _, fr = _project(night, viewer, ring, w, h)
    us, vs, _, frs = _project(night, viewer, np.asarray(src_w)[None], w, h)
    if not fr.all():
        return frame
    pool = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(pool)
    d.polygon(list(zip(u.tolist(), v.tolist())), fill=int(255 * strength))
    pool = pool.filter(ImageFilter.GaussianBlur(14))
    beam = Image.new("L", (w, h), 0)
    if frs[0]:
        db = ImageDraw.Draw(beam)
        hull = np.array(list(zip(u, v)))
        left, right = hull[np.argmin(hull[:, 0])], hull[np.argmax(hull[:, 0])]
        db.polygon([(us[0], vs[0]), tuple(left), tuple(right)], fill=int(70 * strength))
        beam = beam.filter(ImageFilter.GaussianBlur(10))
    a = (np.asarray(pool, np.float32) + np.asarray(beam, np.float32)) / 255.0
    out = frame.astype(np.float32)
    lit = out * (1.0 + 1.6 * a[..., None]) + a[..., None] * np.array([90, 80, 60], np.float32)
    return np.clip(lit, 0, 255).astype(np.uint8)


def add_beacon(night, viewer, frame, pos_w, color, on=True):
    """Flashing LED glow on the robot."""
    if not on:
        return frame
    h, w = frame.shape[:2]
    u, v, depth, fr = _project(night, viewer, np.asarray(pos_w)[None], w, h)
    if not fr[0]:
        return frame
    rad = float(np.clip(260.0 / max(depth[0], 0.5), 6, 60))
    yy, xx = np.ogrid[0:h, 0:w]
    r2 = (xx - u[0]) ** 2 + (yy - v[0]) ** 2
    g = np.exp(-r2 / (2 * (rad * 0.35) ** 2)) + 0.35 * np.exp(-r2 / (2 * (rad * 1.4) ** 2))
    out = frame.astype(np.float32) + g[..., None] * np.array(color, np.float32)[None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------ poses


def lying(parts, x, y, z, heading):
    """Person-local parts -> lying on the back on the ground (body axis along ``heading``)."""
    import trimesh

    Ry = trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0))  # head (+z) -> -x ... feet first
    Rz = trimesh.transformations.rotation_matrix(heading, (0, 0, 1))
    Wm = Rz @ Ry
    Wm[:3, 3] = (x, y, z + 0.13)
    return [(tag, mesh, col, Wm @ T) for tag, mesh, col, T in parts]


def kit_parts():
    """First-aid kit (white case + red cross) in the kit's local frame."""
    import trimesh

    def T(t):
        M = np.eye(4)
        M[:3, 3] = t
        return M

    box = trimesh.creation.box(extents=(0.32, 0.24, 0.15))
    bar1 = trimesh.creation.box(extents=(0.16, 0.05, 0.006))
    bar2 = trimesh.creation.box(extents=(0.05, 0.16, 0.006))
    return [("case", box, (0.95, 0.95, 0.95), T((0, 0, 0))), ("cross1", bar1, (0.85, 0.1, 0.1), T((0, 0, 0.078))),
            ("cross2", bar2, (0.85, 0.1, 0.1), T((0, 0, 0.078)))]


def remux(silent_path, voice_json, out_path):
    """Re-attach the voice track to a silent take (e.g. after changing the mixing)."""
    import json

    vt = VoiceTrack()
    vt.events = [tuple(e) for e in json.load(open(voice_json))]
    vt.mux(silent_path, out_path)
