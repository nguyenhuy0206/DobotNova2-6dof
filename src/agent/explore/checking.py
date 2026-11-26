"""Read-only Astra self-checks, separate from authoritative judge outcomes."""
from __future__ import annotations

import copy
import json

from actions.executor import ActionExecutor, ActionSpec
from actions.registry import _obj
from agent.harness import AgentHarness, HarnessOptions
from common.errors import RobotError, ValidationError
from memory.store import EpisodeStore


def retry_stop_reason(session, episode: str) -> str | None:
    """Never let a new harness episode clear a stop from the preceding attempt."""
    if session.executor.stopped:
        return str(session.executor.stopped)
    end = next((e for e in reversed(session.harness.episodes.load(episode)) if e.get("event") == "end"), {})
    if end.get("reason") in ("robot_stopped", "robot_alarm"):
        return str(end["reason"])
    return None


def check_task(session, task: str, episode: str) -> dict:
    """A fresh camera snapshot and a separate model call with no robot/memory-write tools.

    This is a model claim used for retry control, never a judge pass or operator label.
    """
    store = session.harness.episodes
    result = {"status": "unclear", "source": "astra_visual", "evidence": "No check recorded.",
              "next_action": "Inspect the current physical state before moving.", "retry_allowed": False}

    def record_check(status: str, evidence: str, next_action: str) -> dict:
        if result["retry_allowed"]:
            # 24 Nov (check 163810): the checker called this again on later requests that carry no images and
            # overwrote its image-based assessment with "no camera images are attached". The first one stands.
            raise ValidationError("already recorded: finish now")
        if status not in ("success", "failure", "unclear") or not evidence.strip():
            raise ValidationError("record_task_check needs a status and concrete visual evidence")
        result.update(status=status, evidence=evidence.strip()[:2000], next_action=next_action[:2000],
                      retry_allowed=True)
        return dict(result)

    images, snapshots = [], []
    try:
        if not session.cfg.cameras:
            raise ValueError("no cameras configured for visual self-check")
        for camera in session.cfg.cameras:
            shot = session.kit.scene.look(camera=camera)
            encoded = shot.get("image_jpeg_b64")
            if not encoded:
                raise ValueError(f"{camera}: no image for self-check")
            images.append(encoded)
            snapshots.append(shot)
    except (RobotError, OSError, ValueError) as exc:
        result["evidence"] = f"Camera check unavailable: {exc}"
    else:
        reference_note = ""
        task_refs = getattr(session.cfg, "reference_task", "")
        if task_refs:
            # 24 Nov (operator): the check also compares with the operator's reference pictures
            try:
                from perception.reference import compare, load_references
                camera = "wrist" if "wrist" in session.cfg.cameras else session.cfg.cameras[0]
                cmp = compare(session.kit.scene.observe(camera).color, load_references(task_refs))
                images.append(session.kit.scene.encode_image(cmp.pop("picture"), []))
                reference_note = (f"\nLast picture: three raw pictures side by side: LEFT = the {camera} camera now, "
                                  "MIDDLE = the closest operator reference of the finished state, RIGHT = the closest "
                                  "operator reference of the unfinished state. Whole-picture similarity (a cue, not a "
                                  f"verdict): {cmp['scores']}, closer to {cmp['closer_to']}.")
            except (RobotError, OSError, ValueError) as exc:
                reference_note = f"\nReference comparison unavailable: {exc}"
        # Preserve exactly the evidence shown to the checker with the physical attempt.
        store.append(episode, {"event": "self_check_images", "snapshots": snapshots})
        cfg = copy.copy(session.harness.config)
        cfg.allowed_tools, cfg.max_steps = [], 3
        cfg.system_prompt = (
            "You are Astra checking a completed robot attempt. You cannot move the robot. "
            "Evaluate the TASK against the fresh final camera images, not the driver's success claim. "
            "For threading, success requires visible evidence that the held object passes THROUGH the eye; "
            "alignment, overlap in a single view and a completed insert stroke are insufficient. "
            "Use unclear if occluded or depth/passage cannot be established. Do not guess. "
            "Call record_task_check once with concrete visual evidence and the next observation or safe "
            "correction for an unfinished task, then finish. This is a self-check, not a judge verdict. "
            f"The only cameras are: {', '.join(session.cfg.cameras)} (fixed); never suggest another camera, "
            "view or angle.")
        executor = ActionExecutor([ActionSpec(
            "record_task_check", "Record your visual assessment of this attempt.", record_check,
            _obj({"status": {"type": "string", "enum": ["success", "failure", "unclear"]},
                  "evidence": {"type": "string"}, "next_action": {"type": "string"}},
                 ["status", "evidence", "next_action"]), read_only=True)], mode="dry_run")
        checker = AgentHarness(session.harness.client, executor, cfg,
                               episodes=EpisodeStore(store.root / "checks"), send_images=True,
                               judge=False, options=HarnessOptions.from_variant("V1"))
        # 24 Nov: V0 chains requests with previous_response_id, which modelapi.vn rejected (400) on the second
        # request and crashed the run after the check was already recorded. V1 sends self-contained requests.
        from agent.explore.learning import summarize_episode
        try:
            checker.run("TASK: " + task + "\nCamera order: " + ", ".join(session.cfg.cameras) + reference_note
                        + "\nAttempt log (claims are not proof):\n" + summarize_episode(store, episode),
                        images=images)
        except Exception as exc:  # noqa: BLE001 — an API error must not lose a recorded check or crash the run
            result["checker_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["check_episode"] = checker.last_episode
        result["metrics"] = checker.last_metrics
    store.append(episode, {"event": "self_check", **result})
    return result


def autonomous_feedback(check: dict) -> str:
    return "ASTRA VISUAL SELF-CHECK (model assessment, not judge credit): " + json.dumps(
        # 24 Nov: the checker's next_action asked for an "oblique view" no camera has; the driver obeyed by
        # backing off 5.5 mm. Only what it saw goes on; the driver chooses the correction.
        {k: check.get(k) for k in ("status", "evidence")}, ensure_ascii=False)
