"""Summarize patrol results: summarize.py <out_dir> -> prints tables and writes summary.json."""

import collections
import glob
import json
import os
import sys

import numpy as np

out = sys.argv[1]
RATED_H = 2.5
rows = []
for f in sorted(glob.glob(os.path.join(out, "**", "results_scenario*.json"), recursive=True)):
    r = json.load(open(f))
    laps = r["laps"]
    a = r["assumptions"]
    ok = [l for l in laps if l["success"]]
    falls = [l for l in laps if l["fell"]]
    stuck = [l for l in laps if not l["success"] and not l["fell"]]
    lt = np.array([l["lap_time_s"] for l in ok]) if ok else np.array([np.nan])
    e_pos = np.array([l["energy_pos_Wh"] for l in ok]) if ok else np.array([np.nan])
    e_abs = np.array([l["energy_abs_Wh"] for l in ok]) if ok else np.array([np.nan])
    p_mech = e_pos * 3600 / lt
    p_elec = p_mech / a["drive_eff"] + a["base_power_w"]
    missed = collections.Counter(c for l in laps for c in l["missed_checkpoints"])
    fall_zones = collections.Counter(l["reason"] for l in falls)
    row = dict(
        scenario=r["scenario"], file=os.path.relpath(f, out), laps=len(laps), success=len(ok),
        success_rate=len(ok) / len(laps), falls=len(falls), fall_where=dict(fall_zones),
        fall_pos=[[round(v, 1) for v in l["fall_pos"]] for l in falls], stuck=len(stuck),
        stuck_where=[(l["reason"], [round(v, 1) for v in l["fall_pos"]]) for l in stuck],
        lap_time_mean=float(np.nanmean(lt)), lap_time_min=float(np.nanmin(lt)), lap_time_max=float(np.nanmax(lt)),
        distance_mean=float(np.mean([l["distance_m"] for l in ok])) if ok else float("nan"),
        switches_mean=float(np.mean([l["mode_switches"] for l in ok])) if ok else float("nan"),
        missed_laps=sum(1 for l in ok if l["missed_checkpoints"]), missed=dict(missed),
        energy_pos_Wh=float(np.nanmean(e_pos)), energy_abs_Wh=float(np.nanmean(e_abs)),
        p_mech_W=float(np.nanmean(p_mech)), p_elec_W=float(np.nanmean(p_elec)),
        laps_per_2h5=float(RATED_H * 3600 / np.nanmean(lt)),
        wh_for_2h5=float(np.nanmean(p_elec) * RATED_H), assumptions=a,
        wet_laps=sum(l["wet"] for l in laps), wet_success=sum(l["wet"] and l["success"] for l in laps),
        events=[(e["time"][11:19], e["type"], e["message"]) for e in r["events"]],
    )
    rows.append(row)
    print(f"\n== scenario {row['scenario']} ({row['file']})")
    for k, v in row.items():
        if k not in ("file", "assumptions"):
            print(f"  {k}: {v}")
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1)
