#!/usr/bin/env python3
"""One-off: past robot episodes -> observations (data/memory/observations.jsonl), source "migrated".

Replays every tool call of every LIVE episode log through the same recorder the harness uses
now (memory.knowledge.observe_call), so the numbers measured before the knowledge store existed
(skill parameters, rz, wrist pixels, contact heights, grip positions, outcomes) become data the
code can check hypotheses against. Nothing is interpreted: no concepts, no hypotheses.
The robot state after a call is taken from the results themselves (pose of the last move,
gripper of the last gripper call); for skill / routine runs the pose afterwards is unknown and
left out. Episodes already in the file are skipped. Dry run unless --write.

    .venv/bin/python scripts/migrate_observations.py            # prints what it would add
    .venv/bin/python scripts/migrate_observations.py --write
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.paths import EPISODES_DIR  # noqa: E402
from memory.knowledge import OBSERVATIONS_FILE, Observations, observe_call  # noqa: E402

# the calls the harness records today (agent/explore.py: moves_robot / delegates / OBSERVED_READS)
RECORDED = {"move_to_pose", "move_relative", "move_to_joints", "jog_joint", "go_to_position", "open_gripper",
            "close_gripper", "set_gripper", "stop", "run_skill", "run_routine", "search_object",
            "look", "find_objects", "verify_calibration", "solve_calibration"}
NO_POSE_AFTER = {"run_skill", "run_routine", "search_object"}


def episode_calls(path: Path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((e for e in events if e.get("event") == "start"), {})
    if (start.get("meta") or {}).get("mode") != "live":
        return None
    pose, gripper, out = None, {}, []
    for e in events:
        if e.get("event") != "tool" or e.get("auto") or e.get("name") not in RECORDED:
            continue
        r = e.get("result") or {}
        res = r.get("result") if isinstance(r.get("result"), dict) else {}
        if isinstance(res.get("pose"), list):
            pose = res["pose"]
        if res.get("position") is not None and res.get("grip_status") is not None:
            gripper = {"position": res["position"], "grip_status": res["grip_status"]}
        state = None if e["name"] in NO_POSE_AFTER and not isinstance(res.get("pose"), list) else \
            {"pose": pose, "gripper": gripper}
        t = dt.datetime.fromtimestamp(e["t"]).strftime("%Y-%m-%d %H:%M:%S") if e.get("t") else None
        out.append((t, observe_call(e["name"], e.get("args") or {}, r, state)))
        if e["name"] in NO_POSE_AFTER:
            pose = None                                   # where the skill left the arm is not in the log
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    store = Observations(OBSERVATIONS_FILE)
    done = {o.get("episode") for o in store.items}
    plan, skipped = [], Counter()
    for path in sorted(EPISODES_DIR.glob("2*.jsonl")):
        ep = path.stem
        if ep in done:
            skipped["already recorded"] += 1
            continue
        calls = episode_calls(path)
        if calls is None:
            skipped["not live"] += 1
            continue
        plan += [(ep, t, v) for t, v in calls]
    actions = Counter(v.get("skill") or v.get("routine") or v["action"] for _, _, v in plan)
    print(f"{len(plan)} observations from {len({ep for ep, _, _ in plan})} live episodes; skipped {dict(skipped)}")
    print("by action:", dict(actions.most_common(20)))
    print("with contact_z:", sum("contact_z" in v for _, _, v in plan),
          "· failed:", sum(not v["ok"] for _, _, v in plan))
    if not args.write:
        print("dry run — add --write to append them")
        return
    if OBSERVATIONS_FILE.exists():
        shutil.copy(OBSERVATIONS_FILE, OBSERVATIONS_FILE.with_suffix(".before-migration.jsonl"))
    for ep, t, v in plan:
        store.add(v, ep, source="migrated", t=t)
    print(f"appended; {len(store.items)} observations in {OBSERVATIONS_FILE}")


if __name__ == "__main__":
    main()
