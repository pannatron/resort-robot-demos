"""Synthesized party music (no samples): "Happy Birthday" (public-domain melody) on a music-box voice with a
soft chord pad, and an original 120 BPM dance loop. Writes assets/voice/music_hbd.mp3 and music_dance.mp3.

party_audio.py            (run with any Python that has numpy; ffmpeg from imageio_ffmpeg)
"""

import os
import subprocess
import wave

import numpy as np

SR = 24000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice")
DANCE_BPM = 120.0
DANCE_BARS = 16


def hz(note):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    return 440.0 * 2 ** ((names[note[:-1]] + 12 * (int(note[-1]) + 1) - 69) / 12)


def tone(f, dur, kind="bell", vol=1.0):
    t = np.arange(int(dur * SR)) / SR
    if kind == "bell":  # music box: bright partials, fast decay
        y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 4.02 * f * t)
        env = np.exp(-t * 3.2) * np.minimum(1, t * 400)
    elif kind == "pad":
        y = sum(np.sin(2 * np.pi * f * d * t) for d in (1.0, 1.004, 0.996)) / 3
        env = np.minimum(1, t * 6) * np.minimum(1, (dur - t) * 6)
    elif kind == "bass":
        y = np.tanh(2.2 * np.sin(2 * np.pi * f * t))
        env = np.exp(-t * 5) * np.minimum(1, t * 300)
    else:  # lead pluck (saw-ish)
        y = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 7))
        env = np.exp(-t * 7) * np.minimum(1, t * 300)
    return vol * y * env


def mix(buf, start, sig):
    i = int(start * SR)
    n = min(len(sig), len(buf) - i)
    if n > 0:
        buf[i:i + n] += sig[:n]


def happy_birthday():
    bpm = 100.0
    b = 60.0 / bpm
    melody = [("G4", .75), ("G4", .25), ("A4", 1), ("G4", 1), ("C5", 1), ("B4", 2),
              ("G4", .75), ("G4", .25), ("A4", 1), ("G4", 1), ("D5", 1), ("C5", 2),
              ("G4", .75), ("G4", .25), ("G5", 1), ("E5", 1), ("C5", 1), ("B4", 1), ("A4", 2),
              ("F5", .75), ("F5", .25), ("E5", 1), ("C5", 1), ("D5", 1), ("C5", 3)]
    total = sum(d for _, d in melody) * b + 2.5
    buf = np.zeros(int(total * SR))
    t = 0.4
    for n, d in melody:
        mix(buf, t, tone(hz(n), max(d * b, 0.6) + 0.8, "bell", 0.5))
        t += d * b
    # chords (3/4 bars after the 1-beat pickup): C G7 G7 C C F C-G C
    chords = [["C3", "E3", "G3"], ["G2", "B2", "F3"], ["G2", "B2", "F3"], ["C3", "E3", "G3"],
              ["C3", "E3", "G3"], ["F2", "A2", "C3"], ["C3", "G3", "E3"], ["C3", "E3", "G3"]]
    t = 0.4 + 1.0 * b
    for ch in chords:
        for n in ch:
            mix(buf, t, tone(hz(n), 3 * b, "pad", 0.16))
        t += 3 * b
    return buf


def dance_loop():
    b = 60.0 / DANCE_BPM
    total = DANCE_BARS * 4 * b + 1.0
    buf = np.zeros(int(total * SR))
    t_ = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-t_ * 30)) * t_) * np.exp(-t_ * 9)
    rng = np.random.default_rng(3)
    clap = rng.normal(0, 1, int(0.18 * SR)) * np.exp(-np.arange(int(0.18 * SR)) / SR * 30) * 0.35
    hat = rng.normal(0, 1, int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 90) * 0.12
    hat = np.diff(hat, prepend=0)  # brighter
    prog = [("C3", ["C4", "E4", "G4"]), ("A2", ["A3", "C4", "E4"]), ("F2", ["F3", "A3", "C4"]),
            ("G2", ["G3", "B3", "D4"])]
    lead = ["E5", "G5", "C6", "G5", "A5", "C6", "E6", "C6", "F5", "A5", "C6", "A5", "G5", "B5", "D6", "B5"]
    for bar in range(DANCE_BARS):
        root, triad = prog[bar % 4]
        for beat in range(4):
            t = 0.5 + (bar * 4 + beat) * b
            mix(buf, t, kick * 0.9)
            if beat in (1, 3):
                mix(buf, t, clap)
            mix(buf, t + b / 2, hat)
            mix(buf, t, hat * 0.6)
            mix(buf, t + b / 2, tone(hz(root), b * 0.45, "bass", 0.35))
            mix(buf, t, tone(hz(root), b * 0.4, "bass", 0.25))
        if bar >= 4:  # lead enters after the intro
            for k in range(8):
                n = lead[(bar % 4) * 4 + (k % 4)]
                mix(buf, 0.5 + bar * 4 * b + k * b / 2, tone(hz(n), b * 0.5, "lead", 0.11))
        for n in triad:
            mix(buf, 0.5 + bar * 4 * b, tone(hz(n), 4 * b, "pad", 0.06))
    return buf


def write_mp3(buf, name):
    buf = buf / max(1e-6, np.abs(buf).max()) * 0.9
    wav = os.path.join(OUT, name + ".wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype(np.int16).tobytes())
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", wav, "-b:a", "96k",
                    os.path.join(OUT, name + ".mp3")], check=True)
    os.remove(wav)
    print(name, f"{len(buf) / SR:.1f}s")


if __name__ == "__main__":
    write_mp3(happy_birthday(), "music_hbd")
    write_mp3(dance_loop(), "music_dance")
