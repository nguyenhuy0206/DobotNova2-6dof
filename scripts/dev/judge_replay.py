#!/usr/bin/env python3
"""Replay recorded images without hardware. Legacy observations can never earn credit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from judge.engine import JudgeSession
from judge.ledger import JudgeLedger
from judge.records import Evidence, Observation
from judge.spec import load_spec
from perception.detection import detect_colored_objects


def replay(episode_id: str, episodes: Path = ROOT / "data/episodes",
           output: Path = ROOT / "data/judge/replay", spec=None) -> dict:
    if not re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{6}", episode_id):
        raise ValueError("invalid episode id")
    spec = spec or load_spec()
    source = episodes / f"{episode_id}.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    # Each replay is separate from both recorded episodes and the authoritative ledger.
    run_dir = output / episode_id / __import__("uuid").uuid4().hex
    ledger = JudgeLedger(run_dir / "evaluations.jsonl")
    session = JudgeSession(episode_id, spec, ledger, 0)
    frames, paths, missing, legacy = [], [], 0, 0
    current_robot = None
    for index, row in enumerate(rows):
        if row.get("event") == "state":
            state = row.get("state") or {}
            pose, grip = state.get("pose"), state.get("gripper") or {}
            status = str(grip.get("grip_status", ""))
            current_robot = Evidence(f"state-{index}", "robot", None, None, None, str(source),
                                     hashlib.sha256(raw).hexdigest(), None, legacy=True,
                                     observation=Observation(z_mm=pose[2] if pose else None,
                                                             holding="caught" in status))
        if row.get("event") != "tool":
            continue
        name = row.get("name")
        if name == "close_gripper":
            session.start_attempt(current_robot)
        if name == "open_gripper":
            session.evaluate("after_open", index, (current_robot,) if current_robot else ())
        payload = (row.get("result") or {}).get("result")
        if name != "look" or not isinstance(payload, dict):
            continue
        camera = payload.get("camera") or (row.get("args") or {}).get("camera", "env")
        if camera not in {"wrist", "env"}:
            continue
        path = Path(payload.get("image_file") or "")
        paths.append(str(path))
        # Never trust the detections/measurements returned by Astra's tools in the log.
        found, image_sha = [], None
        if path.is_file():
            import cv2
            image_bytes = path.read_bytes()
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            image = cv2.imread(str(path))
            if image is not None:
                found = detect_colored_objects(image, [spec.object_color] if spec.object_color else None)
        else:
            missing += 1
        is_legacy = any(payload.get(k) is None for k in ("captured_ns", "sequence_id", "stream_id"))
        legacy += is_legacy
        # Ambiguous blobs remain separate, never silently select the largest.
        candidates = found or [None]
        results = []
        for n, det in enumerate(candidates):
            obs = (Observation(pixel=tuple(det.center), bbox=tuple(det.bbox), edge_deg=det.angle_deg,
                               visibility="partial" if det.cut_by_edge else "full") if det else Observation())
            e = Evidence(f"frame-{index}-{n}", camera, payload.get("captured_ns"), payload.get("sequence_id"),
                         payload.get("stream_id"), str(path) if path.is_file() else None, image_sha, None,
                         legacy=is_legacy, observation=obs)
            ev = ((current_robot,) if current_robot else ()) + (e,)
            verdict = session.evaluate("replay_frame", index, ev)
            results.append({"label": det.label if det else None, "observation": asdict(obs),
                            "predicates": [asdict(p) for p in verdict.predicate_results]})
        frames.append({"event_index": index, "camera": camera, "image": str(path), "sha256": image_sha,
                       "legacy": is_legacy, "candidates": results})
    final = session.evaluate("episode_end", len(rows), ())
    counts = {name: Counter() for name in ("grasp_aligned", "held_after_lift", "placed_on_table")}
    for frame in frames:
        for c in frame["candidates"]:
            for p in c["predicates"]:
                counts[p["predicate"]][p["value"]] += 1
    result = {"episode_id": episode_id, "status": final.status, "success_credit": False,
              "provisional": True, "look_frames": len(frames), "legacy_frames": legacy,
              "missing_images": missing, "reused_image_references": sum(n - 1 for n in Counter(paths).values()),
              "predicate_counts": {k: dict(v) for k, v in counts.items()},
              "reason_codes": final.reason_codes, "episode_sha256": hashlib.sha256(raw).hexdigest(),
              "frames": frames, "verdict": final.private_dict()}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in result.items() if k not in {"frames", "verdict", "episode_sha256", "reason_codes"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+")
    args = parser.parse_args()
    for episode in args.episodes:
        print(json.dumps(replay(episode), sort_keys=True))


if __name__ == "__main__":
    main()
