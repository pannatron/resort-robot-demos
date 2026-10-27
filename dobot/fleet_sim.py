"""Whole-shift fleet simulation: guard-swapped batteries, 1-3 robots, accelerated time.

The robots cannot dock-charge: a guard swaps the battery at the booth. Route timing and energy per segment come
from the physics patrol runs (results/profile), so stairs cost more than flat paving. Battery capacity is defined
as 2.5 h of normal patrolling (the customer's rated runtime).

State machine per robot:
    PATROL -> RETURN_TO_BASE -> WAIT_SWAP -> SWAP -> SELF_CHECK -> STANDBY / PATROL
    EMERGENCY_RETURN when the battery falls under (energy to get home) x 1.15 with no relief available
    RELIEF: a standby robot drives to the last checkpoint before the patrolling one goes home (no coverage gap)
    ASSIST: a standby robot with enough charge goes to a major incident

fleet_sim.py <profile_dir> <out_dir>
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys

import numpy as np

DT = 5.0  # [s]
RATED_H = 2.5
STANDBY_FRAC = 0.05  # drain while parked in sleep mode at the booth, relative to the average patrol drain
SWAP_S, SELF_CHECK_S = 60.0, 45.0
GUARD_DELAY_MIN = (1.0, 10.0)
ESCALATE_MIN = 8.0
RESERVE = 1.15


# ------------------------------------------------------------------------------------------ route profile


def load_profile(profile_dir):
    laps = []
    for f in glob.glob(os.path.join(profile_dir, "results_scenario*.json")):
        for lap in json.load(open(f))["laps"]:
            if lap["success"] and lap.get("wp_time"):
                laps.append(lap)
    if not laps:
        raise SystemExit("no successful laps with waypoint timing")
    T = np.array([[np.nan if v is None else v for v in lap["wp_time"]] for lap in laps], float)
    E = np.array([[np.nan if v is None else v for v in lap["wp_energy_Wh"]] for lap in laps], float)
    T[:, 0] = 0.0
    E[:, 0] = 0.0
    seg_t = np.nanmean(np.diff(T, axis=1), axis=0)
    seg_e = np.nanmean(np.diff(E, axis=1), axis=0)
    return seg_t, seg_e, len(laps)


class Route:
    def __init__(self, seg_t, seg_e, checkpoints):
        self.seg_t, self.seg_e = seg_t, seg_e
        self.n = len(seg_t)
        self.lap_t, self.lap_e = float(seg_t.sum()), float(seg_e.sum())
        self.cum_t = np.concatenate([[0.0], np.cumsum(seg_t)])
        self.cum_e = np.concatenate([[0.0], np.cumsum(seg_e)])
        self.checkpoints = checkpoints  # waypoint indices

    def pos_cost(self, s):
        """Lap progress s in [0, lap_t) -> (energy to get home, time to get home) by the shorter way."""
        i = int(np.searchsorted(self.cum_t, s, side="right") - 1)
        i = min(max(i, 0), self.n - 1)
        frac = (s - self.cum_t[i]) / max(self.seg_t[i], 1e-6)
        e_here = self.cum_e[i] + frac * self.seg_e[i]
        fwd = (self.lap_e - e_here, self.lap_t - s)
        back = (e_here, s)
        return min(fwd, back)

    def energy_rate(self, s):
        i = int(np.searchsorted(self.cum_t, s % self.lap_t, side="right") - 1)
        i = min(max(i, 0), self.n - 1)
        return self.seg_e[i] / max(self.seg_t[i], 1e-6)

    def last_checkpoint_s(self, s):
        cps = [self.cum_t[c] for c in self.checkpoints if self.cum_t[c] <= s]
        return max(cps) if cps else 0.0


# ------------------------------------------------------------------------------------------ simulation


def simulate(route, n_robots, packs_total, charge_h, shift_h=8.0, seed=0, incidents_per_night=2.0,
             record=False):
    rng = np.random.default_rng(seed)
    cap = route.lap_e / route.lap_t * RATED_H * 3600  # [Wh] 2.5 h of average patrolling
    p_patrol = route.lap_e / route.lap_t  # [Wh/s]
    p_idle = STANDBY_FRAC * p_patrol
    # robots: state, battery, lap progress, timers
    R = [dict(state="STANDBY", batt=cap, s=0.0, timer=0.0, target=0.0, swaps=0, patrol_s=0.0) for _ in range(n_robots)]
    R[0]["state"] = "PATROL"
    spare = [cap] * max(packs_total - n_robots, 0)  # charged packs on the shelf
    charging = []  # finish times of packs on the charger
    t, steps = 0.0, int(shift_h * 3600 / DT)
    covered = np.zeros(steps, bool)
    swaps = escalations = waits_for_pack = emergency = assists = 0
    log = [] if record else None
    incident_times = sorted(rng.uniform(0.3, shift_h - 0.3, rng.poisson(incidents_per_night)) * 3600)
    timeline = [[] for _ in range(n_robots)]
    checkins, batt_trace = [], []  # [t, robot, waypoint], [t, robot, battery 0..1, packs ready on the shelf]
    min_batt = 1.0

    def ret_cost(r):
        e, tt = route.pos_cost(r["s"] % route.lap_t)
        return e, tt

    for k in range(steps):
        # chargers
        done = [c for c in charging if c <= t]
        charging = [c for c in charging if c > t]
        spare += [cap] * len(done)
        patrolling = [r for r in R if r["state"] == "PATROL"]
        covered[k] = bool(patrolling)
        # incidents: dispatch a standby robot with enough charge
        while incident_times and incident_times[0] <= t:
            incident_times.pop(0)
            loc = rng.uniform(0, route.lap_t)
            e_go, t_go = route.pos_cost(loc)
            for r in R:
                if r["state"] == "STANDBY" and r["batt"] > RESERVE * (2 * e_go + 300 * p_patrol):
                    r.update(state="ASSIST", timer=2 * t_go + 300.0, assist_e=2 * e_go + 300 * p_patrol)
                    assists += 1
                    break
        for i, r in enumerate(R):
            st = r["state"]
            if st == "PATROL":
                s_prev = r["s"]
                r["s"] += DT
                if record:  # check-ins: the patrolling robot passes a checkpoint
                    for c in route.checkpoints:
                        c_t = route.cum_t[c]
                        if math.floor((s_prev - c_t) / route.lap_t) != math.floor((r["s"] - c_t) / route.lap_t):
                            checkins.append([t + DT, i, int(c)])
                r["batt"] -= route.energy_rate(r["s"] % route.lap_t) * DT
                r["patrol_s"] += DT
                e_home, t_home = ret_cost(r)
                need = RESERVE * e_home  # hard limit: energy to get home + 15 %
                tgt_cp = route.last_checkpoint_s(r["s"] % route.lap_t)
                _, t_dead = route.pos_cost(tgt_cp)
                plan = need + (t_dead + 300) * p_patrol  # plan the relief: its drive out + 5 min of slack
                relieving = any(o["state"] == "RELIEF" and o.get("relieves") == i for o in R)
                relief = [o for o in R if o is not r and o["state"] == "STANDBY" and o["batt"] > 0.8 * cap]
                if r["batt"] < plan and relief and not relieving:
                    relief[0].update(state="RELIEF", timer=t_dead, target=tgt_cp, relieves=i)
                elif r["batt"] < plan and not relieving and not relief and r["batt"] >= need:
                    r.update(state="RETURN_TO_BASE", timer=t_home)  # planned: nobody can take over
                elif r["batt"] < need:
                    r.update(state="EMERGENCY_RETURN", timer=t_home)  # relief too late / reserve reached
                    emergency += 1
            elif st == "RELIEF":
                r["timer"] -= DT
                r["batt"] -= p_patrol * DT
                if r["timer"] <= 0:
                    r.update(state="PATROL", s=r["target"])
                    if record:
                        cps_at = [c for c in route.checkpoints if abs(route.cum_t[c] - r["target"]) < 1e-6]
                        checkins.extend([t + DT, i, int(c)] for c in cps_at)
                    old = R[r["relieves"]]
                    _, t_home = ret_cost(old)
                    old.update(state="RETURN_TO_BASE", timer=t_home)
            elif st in ("RETURN_TO_BASE", "EMERGENCY_RETURN"):
                r["timer"] -= DT
                r["batt"] -= p_patrol * DT
                if r["timer"] <= 0:
                    delay = rng.uniform(*GUARD_DELAY_MIN) * 60
                    r.update(state="WAIT_SWAP", timer=delay, waited=0.0)
                    if delay > ESCALATE_MIN * 60:
                        escalations += 1
            elif st == "WAIT_SWAP":
                r["timer"] -= DT
                r["batt"] -= p_idle * DT
                if r["timer"] <= 0:
                    if spare:
                        spare.pop()
                        charging.append(t + charge_h * 3600)
                        r.update(state="SWAP", timer=SWAP_S, batt=cap)
                        swaps += 1
                    elif r.get("waited", 0) == 0:
                        waits_for_pack += 1
                        r["waited"] = 1
            elif st == "SWAP":
                r["timer"] -= DT
                if r["timer"] <= 0:
                    r.update(state="SELF_CHECK", timer=SELF_CHECK_S)
            elif st == "SELF_CHECK":
                r["timer"] -= DT
                r["batt"] -= p_idle * DT
                if r["timer"] <= 0:
                    nobody = not any(o["state"] in ("PATROL", "RELIEF") for o in R if o is not r)
                    r.update(state="PATROL" if nobody else "STANDBY", s=0.0)
            elif st == "STANDBY":
                r["batt"] -= p_idle * DT
                if not any(o["state"] in ("PATROL", "RELIEF") for o in R):
                    r.update(state="PATROL", s=0.0)
            elif st == "ASSIST":
                r["timer"] -= DT
                r["batt"] -= p_patrol * DT
                if r["timer"] <= 0:
                    r.update(state="STANDBY")
            min_batt = min(min_batt, r["batt"] / cap)
            if record and k % 12 == 0:
                batt_trace.append([t, i, r["batt"] / cap, len(spare)])
            if record:
                tl = timeline[i]
                if not tl or tl[-1][2] != r["state"]:
                    tl.append([t, t + DT, r["state"]])
                else:
                    tl[-1][1] = t + DT
        t += DT
    # gaps
    gaps, run = [], None
    for k in range(steps):
        if not covered[k] and run is None:
            run = k
        if covered[k] and run is not None:
            gaps.append((run * DT, k * DT))
            run = None
    if run is not None:
        gaps.append((run * DT, steps * DT))
    out = dict(coverage=float(covered.mean()), gaps=gaps, gap_min=sum(b - a for a, b in gaps) / 60,
               longest_gap_min=max([(b - a) / 60 for a, b in gaps], default=0.0), swaps=swaps,
               escalations=escalations, waits_for_pack=waits_for_pack, emergency_returns=emergency, assists=assists,
               min_batt=min_batt)
    if record:
        out["timeline"] = timeline
        out["checkins"] = checkins
        out["battery"] = batt_trace
    return out


def main():
    prof, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    seg_t, seg_e, n_laps = load_profile(prof)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import resort_scene as rs

    route_wp = rs.build().route
    cps = [i for i, w in enumerate(route_wp) if w.checkpoint][:-1]
    route = Route(seg_t, seg_e, cps)
    print(f"profile from {n_laps} physics laps: lap {route.lap_t:.0f} s, {route.lap_e:.3f} Wh (mech); "
          f"battery = {RATED_H} h of patrolling = {route.lap_e / route.lap_t * RATED_H * 3600:.1f} Wh-mech units")
    rows = []
    for robots, charge_h, packs in itertools.product((1, 2, 3), (1.5, 2.0, 3.0), (1, 2, 3, 4, 5, 6)):
        if packs < robots:
            continue
        res = [simulate(route, robots, packs, charge_h, seed=s) for s in range(20)]
        rows.append(dict(robots=robots, charge_h=charge_h, packs=packs,
                         coverage=float(np.mean([r["coverage"] for r in res])),
                         coverage_min=float(np.min([r["coverage"] for r in res])),
                         gap_min=float(np.mean([r["gap_min"] for r in res])),
                         longest_gap_min=float(np.max([r["longest_gap_min"] for r in res])),
                         swaps=float(np.mean([r["swaps"] for r in res])),
                         escalations=float(np.mean([r["escalations"] for r in res])),
                         waits_for_pack=float(np.mean([r["waits_for_pack"] for r in res])),
                         emergency=float(np.mean([r["emergency_returns"] for r in res])),
                         assists=float(np.mean([r["assists"] for r in res]))))
    json.dump(dict(lap_s=route.lap_t, lap_Wh=route.lap_e, n_profile_laps=n_laps, rows=rows),
              open(os.path.join(out, "fleet_summary.json"), "w"), indent=1)
    # minimum packs for zero gap (every one of 20 random nights), per robots x charge time
    print("\nrobots charge_h packs(total) coverage(min) gap_min longest swaps escal wait_pack emerg assists")
    for r in rows:
        print(f"{r['robots']:>6} {r['charge_h']:>8} {r['packs']:>5} {100 * r['coverage']:6.1f}% ({100 * r['coverage_min']:5.1f})"
              f" {r['gap_min']:7.1f} {r['longest_gap_min']:7.1f} {r['swaps']:5.1f} {r['escalations']:5.1f}"
              f" {r['waits_for_pack']:9.1f} {r['emergency']:5.1f} {r['assists']:7.1f}")
    # timelines of one representative night for the clip/HUD
    tl = {f"{n}robots": simulate(route, n, 3 if n == 1 else 4, 1.5, seed=3, record=True) for n in (1, 2)}
    json.dump(tl, open(os.path.join(out, "fleet_timelines.json"), "w"))


if __name__ == "__main__":
    main()
