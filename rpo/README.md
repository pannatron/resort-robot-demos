# RPO humanoid at the resort (simulation demo)

Roboparty's RPO (roboto_origin) doing front-of-house jobs in the same resort scene as the Dobot demo
(`~/dobot_tools/resort`, reused read-only: scene, look, sky, RTX lights, people, HUD).
Every motion is one of Roboparty's shipped ONNX policies; nothing was retrained.

| Job | Script | What happens |
|---|---|---|
| welcome 09:12 | `rpo_jobs.py --job welcome` | guest walks in → wave (TH/EN voice) → check-in → walks ahead to room 306 pacing to the guest → wave goodbye |
| delivery 10:42 | `rpo_jobs.py --job delivery` | staff load towels + water into the chest tray (5 kg on the torso) → corridor → stairs down → pool deck → villa V2 → guest takes the items |
| bump 15:20 | `rpo_jobs.py --job bump` | child runs into the robot (600 N for 0.2 s at contact) → falls on its back → get-up policy → checks on the child |
| party 19:30 | `rpo_party.py` | cake on the tray → song → candles → dance1 + dance0 (part) → two-arm wave finale |

## Run (from `~/IsaacLab`)

```bash
X="uv run --extra video --with onnxruntime python"
$X ~/rpo_tools/resort/skill_check.py --out /tmp/check --viz newton          # all skills + stairs on the resort
$X ~/rpo_tools/resort/rpo_jobs.py --job welcome --out results/welcome --viz newton   # GL preview
./run_rtx.sh                                                                  # all four jobs, OVRTX path tracer
uv run --extra video python ~/rpo_tools/resort/rpo_pitch.py results_rtx clips/RPO_resort_pitch.mp4
uv run --no-project --with edge-tts python make_voice.py                      # voice lines (once)
```

## Files

| File | |
|---|---|
| `rpo_skills.py` | `SkillRunner`: builds each policy's observation from the Isaac Lab robot (walk 10×78, run 3×78 term-major, BeyondMimic trackers 121), clip slicing with a blend back to standing, push, hold |
| `rpo_env_cfg.py` | `RPO-Resort` task: G1 flat env with the RPO asset, resort collision mesh, no fall termination |
| `rpo_stage.py` | shared stage: env, look, RTX actors, payload mass, HUD (skill pill), voice lines, chest tray |
| `party_fx.py` | cake / decor / glow / confetti copied from the Dobot party |
| `assets/rpo_full/` | RPO URDF with the visual mesh as collision on every primitive-only link (matches Roboparty's MJCF) |
| `tests.json` | skill-test numbers the pitch cards cite, with where each was measured |

## Verified (2026-10-24)

- MuJoCo (Roboparty MJCF): walk, run 1.2 m/s, wave, dance0, dance1, get-up from back / side / face-down; chain walk→wave→walk→dance→walk→push→get-up→walk; payload on back 0–20 kg (25 kg falls), chest 0–15 kg.
- Newton (resort mesh): walk, wave, dance1, push + get-up from the back and side; stairs 8 × 15 cm down and up at 0.4 m/s.
- **Face-down get-up fails in Newton** (also with the full-shell collision asset). The bump job pushes from the front so the robot falls on its back.
- Payload is mass added to `torso_link` (no CoM shift: Isaac Lab warns CoM changes are unstable on Newton); the MuJoCo test covered a 15 cm forward CoM offset.

Gotchas: the npz motion clips are in Isaac (USD) joint order and dof_pos is relative to the standing pose; mimic
policies need `sk.start()` and `skill="motion"` on the same step, or the walk step resets the skill.
