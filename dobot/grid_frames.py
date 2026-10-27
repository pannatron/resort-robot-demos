"""Save a grid of frames from a video: grid_frames.py <video> <out.png> <t0,t1,...seconds>"""
import sys
import imageio.v2 as imageio
from PIL import Image
v, out, ts = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
r = imageio.get_reader(v); fps = r.get_meta_data()["fps"]
frames = []
for t in ts:
    try:
        frames.append(Image.fromarray(r.get_data(int(t * fps))).resize((960, 540)))
    except IndexError:
        pass
cols = 2; rows = (len(frames) + 1) // 2
g = Image.new("RGB", (960 * cols, 540 * rows))
for i, im in enumerate(frames):
    g.paste(im, ((i % cols) * 960, (i // cols) * 540))
g.save(out); print(len(frames), "frames, video", r.count_frames() / fps, "s")
