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
ui/            app.py (HTTP server) · index.html (dashboard)
scripts/       run_ui.sh · run_agent.py · astra_smoke_test.py · gripper_cycle.py · camera_benchmark.py · run_camera.sh · build_pyrealsense_mac.sh
tests/         sim_robot.py (fake Nova 2) · protocol / validation / safety / executor / harness / transforms tests
data/          captures/ logs/ episodes/ memory/   (git-ignored contents)
```

Dependency direction: `ui`, `scripts`, `agent` → `actions` → `drivers`, `memory` → `common`.
Every safety rule lives in `actions/`, so the UI and the agent are held to the same limits.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
cp .env.example .env        # put OPENAI_API_KEY in it
# macOS + RealSense depth: scripts/build_pyrealsense_mac.sh
```

Network: controller in TCP/IP secondary-development mode; Mac Ethernet
`192.168.5.100/24`, robot LAN1 `192.168.5.1` (change in `config/robot.yaml`
or `NOVA2_IP=…`).

## Run

```sh
scripts/run_ui.sh                              # dashboard at http://127.0.0.1:8765
.venv/bin/python scripts/astra_smoke_test.py   # check the model answers (no tools)
.venv/bin/python scripts/run_agent.py "open the gripper then jog J1 +5°"            # dry_run
.venv/bin/python scripts/run_agent.py --mode confirm "…"                             # asks before each motion
.venv/bin/python scripts/gripper_cycle.py --cycles 5
scripts/run_camera.sh                          # D455 colour + depth preview
.venv/bin/python -m pytest                     # tests, no hardware needed
```

## Astra harness

1. `config/agent.yaml` picks the model, the allowed tools and `execution_mode`.
   Start with `dry_run`, move to `confirm`, only then `live`.
2. Tools come from `src/actions/registry.py`. Add an `ActionSpec` there and it
   becomes a function tool automatically.
3. Each run writes `data/episodes/<id>.jsonl` (every model turn + tool call/result).
4. `memory/skills.py` recalls stored skills matching the task and adds them to
   the instructions.

## Safety

- The UI only listens on localhost and never accepts raw controller commands.
- Motion requires Mode 5, enabled, no alarm, not running. Speed 0 = software motion lock.
- Joint jog 0.1–20°, speed 1–100 % (see `config/robot.yaml`).
- The workspace box in `robot.yaml` is a placeholder. Measure your cell before relying on it.
