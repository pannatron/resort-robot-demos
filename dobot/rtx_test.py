"""Feasibility test: render the resort with Newton's OVRTX path tracer (VRAM + time + look)."""
import math, os, subprocess, sys, time
import numpy as np, warp as wp
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image
from newton._src.viewer.viewer_rtx import ViewerRTX
import resort_scene as rs
from night_render import NightScene

out = sys.argv[1]
def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("vram before", vram(), flush=True)
t0 = time.time()
LOOK = os.environ.get("LOOK", "day")
v = ViewerRTX(width=1280, height=720, headless=True, environment="none")
print("init", time.time() - t0, "vram", vram(), flush=True)
ns = NightScene(rs.build(), np.zeros(3))
orig = v.objects if hasattr(v, "objects") else None
t0 = time.time()
ns.log_static(v, lamps_on=False)
from night_render import setup_rtx_lights
setup_rtx_lights(v, LOOK, ns.scene.lamps)
print("log_static", time.time() - t0, flush=True)
views = {"pool": ((63, 0, 1.0), (48, -14, -1.2)), "garden": ((53, -24, 0.5), (50, -45, -0.6)), "lobby": ((8, 0, 1.6), (20, 5, 0.5))}
for k, (eye, tgt) in views.items():
    eye = np.array(eye, float); d = np.array(tgt) - eye
    v.set_camera(wp.vec3(*eye), math.degrees(math.atan2(d[2], np.linalg.norm(d[:2]))), math.degrees(math.atan2(d[1], d[0])))
    for i in range(int(os.environ.get("SPP_FRAMES", 8))):
        t0 = time.time(); v.begin_frame(0.0); v.end_frame(); dt = time.time() - t0
    img = v._capture_screenshot_pixels()
    Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(out, f"rtx_{LOOK}_{k}.png"))
    print(k, "last frame", round(dt, 3), "s  vram", vram(), flush=True)
