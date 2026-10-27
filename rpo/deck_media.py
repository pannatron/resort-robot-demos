"""Short looping clips (H.264, no audio, 960x540) + stills for the slide deck, cut around each job's key event.

deck_media.py <results_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/dobot_tools/resort"))
from incidents import ffmpeg_exe  # noqa: E402

# job -> (event type, seconds before, clip length, still offset from the clip start)
CUTS = {"welcome": [("greet", 1.0, 10.0, 4.0), ("arrived", 0.5, 8.0, 4.0)],
        "delivery": [("pickup", 2.0, 8.0, 3.0), ("arrived", -3.0, 9.0, 5.0)],
        "party": [("dance", -1.0, 12.0, 6.0), ("cake", 3.0, 6.0, 2.0)],
        "bump": [("bump", 1.0, 11.0, 3.5)]}


def main():
    rdir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    stairs = {"delivery": 98.0}
    for job, cuts in CUTS.items():
        res = json.load(open(os.path.join(rdir, job, f"result_{job}.json")))
        src = os.path.join(rdir, job, f"{job}.mp4")
        for i, (etype, before, length, still) in enumerate(cuts):
            t0 = next(e["video_s"] for e in res["events"] if e["type"] == etype) - before
            name = f"{job}{i}"
            ff = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(t0, 0):.2f}", "-i", src]
            subprocess.run(ff + ["-t", f"{length}", "-an", "-vf", "scale=960:540", "-c:v", "libx264", "-crf", "26",
                                 "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                 os.path.join(out, name + ".mp4")], check=True)
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(t0, 0) + still:.2f}", "-i", src,
                            "-frames:v", "1", "-q:v", "3", os.path.join(out, name + ".jpg")], check=True)
            print(name, f"{t0:.1f}s +{length}s", os.path.getsize(os.path.join(out, name + ".mp4")) // 1024, "KB")
    t = stairs["delivery"]
    src = os.path.join(rdir, "delivery", "delivery.mp4")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t + 6:.2f}", "-i", src, "-frames:v", "1",
                    "-q:v", "3", os.path.join(out, "stairs.jpg")], check=True)


if __name__ == "__main__":
    main()
