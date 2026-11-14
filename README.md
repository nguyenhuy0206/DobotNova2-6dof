# Dobot Nova 2 Robotics

Local control, perception and an **Astra** LLM agent harness for a Dobot Nova 2
(6-axis, V3 controller) with a PGC gripper and Intel RealSense D455, on macOS.

## Layout

```
config/        robot.yaml (IP, speed, workspace limits, gripper) · agent.yaml (model, tools, execution mode)
src/
  drivers/     nova2.py (TCP/IP 29999/30003/30004) · gripper.py (PGC over RS485) · camera.py (RealSense/OpenCV)
  actions/     validation.py (limits + safety) · robot_actions.py · executor.py (dry_run/confirm/live) · registry.py (tool catalogue)
  agent/       astra_client.py · tools.py (→ function tools) · harness.py (model ⇄ tool loop)
  perception/  detection.py (HSV colour blobs) · transforms.py (pixel → camera → base, hand-eye)
  memory/      store.py (positions, episodes JSONL) · skills.py (skill library + retrieval)
  common/      config, errors, types, paths, logging
ui/            app.py (HTTP router) · skills_api.py · camera_api.py · index.html · debug_skills.js/.css · debug_cameras.js
scripts/       run_vision.sh · run_ui.sh · run_agent.py · astra_smoke_test.py · gripper_cycle.py · camera_benchmark.py · run_camera.sh · build_pyrealsense_mac.sh
tests/         sim_robot.py (fake Nova 2) · protocol / validation / safety / executor / harness / transforms tests
data/          captures/ logs/ episodes/ memory/   (git-ignored contents)
```

Dependency direction: `ui`, `scripts`, `agent` → `actions` → `drivers`, `memory` → `common`.
Every safety rule lives in `actions/`, so the UI and the agent are held to the same limits.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
cp .env.example .env        # put the API key in it (gpt6-key=...); .env is git-ignored
# RealSense: builds librealsense into third_party/librealsense (git-ignored)
scripts/build_pyrealsense_mac.sh           # or: --reuse <existing build/Release>
```

Network: controller in TCP/IP secondary-development mode; Mac Ethernet
`192.168.5.100/24`, robot LAN1 `192.168.5.1` (change in `config/robot.yaml`
or `NOVA2_IP=…`).

## Run

```sh
scripts/run_vision.sh                          # ALL-IN-ONE: cameras check + depth helper + dashboard (re-run to restart)
scripts/run_ui.sh                              # dashboard only at http://127.0.0.1:8765
.venv/bin/python scripts/astra_smoke_test.py   # check the model answers (no tools)
.venv/bin/python scripts/run_agent.py "open the gripper then jog J1 +5°"            # dry_run
.venv/bin/python scripts/run_agent.py --mode confirm "…"                             # asks before each motion
.venv/bin/python scripts/gripper_cycle.py --cycles 5
scripts/run_camera.sh                          # D455 colour + depth preview
.venv/bin/python -m pytest                     # tests, no hardware needed
```

## Primitive skills (`src/actions/primitives.py`)

| Skill | What it does |
|---|---|
| `get_state` | mode/alarm, joints, TCP pose, gripper, saved position names |
| `move_to_pose` | absolute base-frame pose, `linear` = MovL else MovJ |
| `move_to_joints` | absolute J1–J6 (JointMovJ) |
| `move_relative` | straight-line offset in `base` or `tool` frame (≤100 mm / 45°) |
| `jog_joint` | one joint by a relative angle |
| `go_to_position` | saved named position; `home` always exists |
| `save_position` | remember the current pose by name |
| `open_gripper` / `close_gripper` / `set_gripper` | PGC fingers; `close_gripper` reports `holding` |
| `stop` | halt and clear the motion queue |

Each motion skill validates (workspace, joint limits, step size), checks the
robot is ready, sends the command, **blocks until arrived** and returns the new
state. It stops the robot itself on an alarm or timeout. Composite skills are
plain step lists run with `ActionExecutor.run_sequence`:

```python
from actions import ActionExecutor, build_registry, create_primitives
ex = ActionExecutor(build_registry(create_primitives()), mode="confirm", confirm=lambda n, a: input(f"{n} {a}? ") == "y")
ex.run_sequence([{"tool": "open_gripper"},
                 {"tool": "go_to_position", "args": {"name": "above_pick"}},
                 {"tool": "move_relative", "args": {"dz": 40, "frame": "tool"}},
                 {"tool": "close_gripper"},
                 {"tool": "move_relative", "args": {"dz": -40, "frame": "tool"}}])
```

## Debug tab: live robot panel

`scripts/run_ui.sh`, then open <http://127.0.0.1:8765/#debug>.

- **STOP** (or the `Esc` key) halts the robot at any time, even mid-move.
- **Workspace clear** must be switched on before any motion or gripper button works.
- The panel has live state, gripper Open/Close/Set, Cartesian and joint jog,
  Go home and Save position.
- **All Primitive Skills** shows one form per skill, generated from `/api/skills`.
  Adding a primitive in `actions/registry.py` adds it here automatically.

**Cameras**: two live windows at 10 fps, Environment (RealSense D455) and
Wrist (USB webcam), configured under `cameras:` in `config/robot.yaml`.
- macOS camera numbers change when devices are plugged in. Pick each window's
  device from its dropdown; the choice is saved in `data/memory/cameras.json`.
- *RealSense (colour + depth)*: macOS only lets root open the depth sensor, so a small
  helper owns the D455 and the UI keeps running as you:
  ```sh
  sudo scripts/run_depth.sh     # terminal 1: depth helper (root, localhost:8766, writes nothing)
  scripts/run_ui.sh             # terminal 2: UI as usual -> pick "RealSense (colour + depth)"
  ```
  Without the helper, choose the D455's plain device number instead, which gives colour only.
- **Capture** saves the frame plus the robot pose to `data/captures/`.
- Streams stop when you leave the tab, and the camera is released after 15 s.

Code map: `ui/debug_cameras.js` → `ui/camera_api.py` → `drivers/camera_stream.py` → `drivers/camera.py`.

Code map: `ui/debug_skills.js` (UI) → `ui/skills_api.py` (HTTP) →
`actions/primitives.py` (skills) → `drivers/` (robot, gripper).

## Astra tools

`agent/toolkit.py: build_toolkit()` wires every tool Astra can call:

| Module | Tools |
|---|---|
| `actions/primitives.py` | get_state, move_to_pose, move_to_joints, move_relative, jog_joint, go_to_position, save_position, open/close/set_gripper, stop |
| `perception/scene.py` | `look` (image + detected objects, image sent to the model), `find_objects` (one colour, `base_xyz_mm` in robot coordinates) |
| `actions/skill_runner.py` | `list_skills`, `run_skill` (composite skill with params), `save_skill` (learn a new one) |

Composite skills live in `data/memory/skills.json`. They start with
`pick_from_above` and `place_from_above`, and every run counts successes/failures.

Before picking for real:
1. Set `gripper.tool_length_mm` in `config/robot.yaml` (flange → fingertip).
2. Calibrate the D455 to the robot: `python scripts/calibrate_env_camera.py --color red`
   (8+ points, RMS < 5 mm). Use `--tune` to check the HSV colour masks.
3. Tune the colours in `config/perception.yaml` for your lighting.

## Astra harness

1. `config/agent.yaml` picks the model, the allowed tools and `execution_mode`.
   Start with `dry_run`, move to `confirm`, only then `live`.
2. Tools come from `src/actions/registry.py`. Add an `ActionSpec` there and it
   becomes a function tool automatically.
3. Each run writes `data/episodes/<id>.jsonl` (every model turn + tool call/result).
4. `memory/skills.py` recalls stored skills matching the task and adds them to
   the instructions.

### Execution loop (token-efficient, `src/agent/harness.py`)

```
Astra ─► tool call(s) ─► chunk policy ─► executor (validation + safety) ─► event router
  ▲                                                                          │
  └── bounded context + selected images ◄── digest, state, last turns ◄──────┘
```

| Piece | File | What it does |
|---|---|---|
| Token accounting | `agent/metrics.py` | `usage` (input/cached/output/reasoning, latency, images) per request, `metrics` per episode, in the episode JSONL |
| Bounded context | `agent/context.py` | No `previous_response_id`. Each request = instructions + TASK + measured STATE + deterministic digest (≤ 4000 chars) + last 4 turns + last results. Full history stays local |
| Selective vision | `agent/vision.py` | Images only on first view, request (`look image=always`), unexpected events or ambiguous detections. Never a near-identical repeat. `detail` low (env) / high (wrist) |
| Action chunks | `agent/chunking.py` | Up to L=4 motions per proposal in free space, ≤ 2 when aligning in the approach zone, 1 when descending or touching. Stops at gripper/skill boundaries and on any error |
| Event router | `agent/events.py` | contact, grasp missed, skill failed, ambiguous check, scene changed, alarm: decides abort/stop/image without an LLM call |
| Local skills | `actions/skill_runner.py`, `actions/local_steps.py` | Skills = tools + `check` + `routine` (descend_to, lift_to, wrist_center) + recovery (open/lift only). One `run_skill` = many steps, no model call in between |
| Skill validation | `memory/skills.py` | Every new or revised version is a `candidate`: live runs need operator approval and are logged as supervised evidence. `--promote` (≥ 2 supervised successes) and `--rollback` are operator commands |

Variants (`config/explore.yaml: harness.variant`, or `--variant`): V0 baseline ·
V1 bounded context · V2 chunks · V3 both (default). Safety is identical in all of them.

```bash
.venv/bin/python scripts/episode_report.py --last 10      # real episodes: calls, billed tokens, images, grasps
.venv/bin/python scripts/eval_variants.py                 # mock world: V0–V3 side by side (estimated tokens)
scripts/run_explore.sh --skills | --promote NAME | --rollback NAME VERSION
```

## Safety

- The UI only listens on localhost and never accepts raw controller commands.
- Motion requires Mode 5, enabled, no alarm, not running. Speed 0 = software motion lock.
- Joint jog 0.1–20°, speed 1–100 % (see `config/robot.yaml`).
- The workspace box in `robot.yaml` is a placeholder. Measure your cell before relying on it.

## Explore mode — Astra learns the cell by itself

```sh
scripts/run_vision.sh                          # terminal 1: cameras + depth + dashboard
.venv/bin/python scripts/teach_table.py        # once: touch the metal table with the fingertips
scripts/run_calibration.sh                     # robot calibrates the D455 by itself (10 %, ~3 min)
scripts/run_explore.sh                         # terminal 2: Astra (config/explore.yaml)
scripts/run_explore.sh --recalibrate           # after moving the D455
```

- **Tools:** primitives, look (with zoom), find_objects, self-calibration (`record_calibration_point`, `solve_calibration`), skills (its own file `data/memory/skills_explore.json`, no seeds), notes (`data/memory/astra_notes.jsonl`, fed into the next run).
- **Enforced in code:** speed ≤ `agent.yaml max_speed`; fingertip/flange above the taught table along the whole path (controller FK every 2° for joint moves; MovL near the table); the D455 must see the red tape on the gripper after every move, otherwise the move is undone; the gripper closes to 50 % at most; controller collision detection is enabled.
- **Frames:** the controller's active tool (Tool 1 here, about 125 mm off the flange) is measured from `PositiveSolution` before every move, so fingertip poses are relative to the real flange.
- **Calibration** is only valid for the day it was made: an env calibration from another day is set aside automatically. `run_calibration.sh` visits 12 poses (±60 mm, tilts ≤ 20°) and reports leave-one-out accuracy. Unreachable poses are refused by a controller IK check instead of raising an alarm.
- **Episodes and learning:** `run_explore.sh --episodes N` runs N attempts. After each attempt Astra reviews the log with memory tools only and records the outcome in `data/episodes/index.jsonl`. It also saves or revises skills (a new version keeps the old one and its success/failure counts in `history`) and writes notes. `run_explore.sh --review N|ID` reviews earlier episodes without the robot.
- **Approach zone:** while the fingertip is below table + 120 mm, each Cartesian move is ≤ 25 mm at 3 %. A move that stalls counts as contact: the robot stops, backs off 15 mm and reports the contact Z. `open_gripper` opens to 80 % in explore mode.
