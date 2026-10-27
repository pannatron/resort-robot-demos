"""Robot voice lines for the RPO jobs (edge-tts) + the party music from the Dobot demo.

uv run --no-project --with edge-tts python make_voice.py        (needs internet for edge-tts)
"""
import asyncio
import os
import shutil
import sys

import edge_tts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "voice")
DOBOT_VOICE = os.path.expanduser("~/dobot_tools/resort/assets/voice")


def lines():
    src = open(os.path.join(HERE, "rpo_stage.py"), encoding="utf-8").read()
    start = src.index("LINES = {")
    ns = {}
    exec(src[start:src.index("}\n", start) + 2], ns)
    return ns["LINES"]


async def main():
    os.makedirs(OUT, exist_ok=True)
    for key, (who, text) in lines().items():
        path = os.path.join(OUT, key + ".mp3")
        if who == "music":
            shutil.copy(os.path.join(DOBOT_VOICE, key + ".mp3"), path)
            continue
        if os.path.exists(path) and "--force" not in sys.argv:
            continue
        voice = "en-US-JennyNeural" if key.endswith("_en") else "th-TH-PremwadeeNeural"
        await edge_tts.Communicate(text, voice, rate="+5%").save(path)
        print("wrote", key, text)


asyncio.run(main())
