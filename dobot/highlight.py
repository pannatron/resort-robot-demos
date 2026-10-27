"""Cut a patrol video down to a short highlight that fits a chat message.

Interesting moments play at 1x: walk mode (stairs) and the switches around it, alerts, robot-camera shots,
and the start/end of the lap. Long drives in between play fast-forward with a speed tag.
They are found from the HUD itself (badge color, alert banner, camera bar), so no extra log is needed.

highlight.py <out.mp4> <in1.mp4> [<in2.mp4> ...]   env: FAST (default 8), PAD_S (default 2.5)
"""

import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FAST = int(os.environ.get("FAST", 8))
PAD = float(os.environ.get("PAD_S", 2.5))
GAP = float(os.environ.get("GAP_FILL_S", 0))  # play short gaps between highlights at 1x too
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)


def classify(f):
    """Return True when the frame shows something worth watching at normal speed."""
    badge = f[136, 40].astype(int)
    walk = badge[0] > 200 and 120 < badge[1] < 200 and badge[2] < 90  # orange walk badge
    cam = f[22, 24, 0] > 200 and f[22, 24, 1] < 80 and f[60, 40].sum() < 600 and f[136, 40].sum() < 200
    banner = f[:, 640]
    alert = np.any((banner[:, 0] > 180) & (banner[:, 1] < 60) & (banner[:, 2] < 60))
    return walk or cam or alert


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    w = imageio.get_writer(out, fps=25, codec="libx264", macro_block_size=8,
                           ffmpeg_params=["-crf", "27", "-preset", "slow", "-pix_fmt", "yuv420p",
                                          "-movflags", "+faststart"])
    total = 0
    for path in ins:
        r = imageio.get_reader(path)
        fps = r.get_meta_data()["fps"]
        keep = np.array([classify(f) for f in r])  # pass 1: classify (streaming, frames are large)
        n = len(keep)
        pad = int(PAD * fps)
        dil = keep.copy()
        for i in np.nonzero(keep)[0]:
            dil[max(0, i - pad): i + pad + 1] = True
        on = np.nonzero(dil)[0]
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= GAP * fps:
                dil[a:b] = True
        dil[: int(3 * fps)] = True
        dil[-int(3 * fps):] = True
        nxt = 0  # pass 2: write; fast-forward keeps every FAST-th frame
        for i, f in enumerate(imageio.get_reader(path)):
            if i < nxt:
                continue
            if dil[i]:
                w.append_data(f)
                nxt = i + 1
                total += 1
            else:
                img = Image.fromarray(f)
                d = ImageDraw.Draw(img)
                d.rounded_rectangle([560, 640, 720, 690], 10, fill=(0, 0, 0))
                d.text((640, 665), f"▶▶ x{FAST}", font=FONT, fill=(255, 255, 255), anchor="mm")
                w.append_data(np.asarray(img))
                nxt = i + FAST
                total += 1
        print(f"{path}: {n / fps:.0f}s -> normal {dil.sum() / fps:.0f}s", flush=True)
    w.close()
    print(f"{out}: {total / 25:.0f}s, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
