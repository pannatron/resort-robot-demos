"""Render still views of the resort without the simulator (fast iteration on the look)."""
import math, os, sys
import numpy as np
import warp as wp
sys.path.insert(0, os.path.dirname(__file__))
import newton.viewer
from PIL import Image
import resort_scene as rs
from night_render import NightScene, setup_night

out = sys.argv[1]
W, H = 1280, 720
viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
scene = rs.build()
ns = NightScene(scene, np.zeros(3))
from night_render import setup_day
MODE = os.environ.get('LOOK', 'night')
(setup_night(viewer) if MODE == 'night' else setup_day(viewer, morning=MODE == 'morning'))
from night_render import Sky
sky = Sky(viewer, MODE, viewer.renderer._sun_direction)
ns.log_static(viewer, lamps_on=MODE == 'night')

def look(eye, target):
    eye = np.array(eye, float); t = np.array(target, float); d = t - eye
    yaw = math.degrees(math.atan2(d[1], d[0])); pitch = math.degrees(math.atan2(d[2], np.linalg.norm(d[:2])))
    viewer.set_camera(wp.vec3(*eye), pitch, yaw)

views = {
    "overview": ((-25, 30, 45), (30, -25, -1)),
    "lobby": ((8, 0, 1.6), (20, 5, 0.5)),
    "corridor": ((30, 2.5, 1.4), (50, 3.5, 0.5)),
    "stairs_pool": ((58.5, 4.5, 1.6), (58.5, -6, -1.2)),
    "pool": ((63, 0, 1.0), (48, -14, -1.2)),
    "garden": ((53, -24, 0.5), (50, -45, -0.6)),
    "gate": ((51, -50, 0.6), (50, -60, -0.3)),
    "return": ((3, -40, 0.8), (0, -10, -0.5)),
    "booth": ((5, -3, 1.6), (-3, 4, 0.8)),
}
sel = sys.argv[2:] or list(views)
for k in sel:
    look(*views[k])
    for _ in range(2):
        viewer.begin_frame(0.0); viewer.end_frame()
    f = viewer.get_frame().numpy()
    f = ns.add_glow(viewer, sky.composite(viewer, f))
    Image.fromarray(f).save(os.path.join(out, f"{k}.png"))
    print("saved", k)
