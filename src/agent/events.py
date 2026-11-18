"""Event router: turns every tool result into explicit events and decides, without
any LLM call, what happens next.

    result ─► classify() ─► [Event]  ─► abort the rest of the chunk?
                                      ─► stop the episode (robot alarm)?
                                      ─► attach a camera image to the next Astra call?
                                      ─► one deterministic line for the rolling summary

Astra is only asked again when a decision is needed; routine successes become a
one-line digest instead of a new image or a long JSON result.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field


class EventType(str, enum.Enum):
    ACTION_OK = "action_ok"
    ACTION_ERROR = "action_error"          # validation error, timeout, unknown failure
    SAFETY_BLOCK = "safety_block"          # table / IK / view guard / limits refused or undid the move
    CONTACT = "contact"                    # stall in the approach zone: measured surface
    OPERATOR_REJECTED = "operator_rejected"
    GRASP_CAUGHT = "grasp_caught"
    GRASP_MISSED = "grasp_missed"          # holding=false, or an edge grip near the open position
    RELEASED = "released"
    SKILL_OK = "skill_ok"
    SKILL_FAILED = "skill_failed"
    VERIFY_AMBIGUOUS = "verify_ambiguous"  # a local check could not decide (occluded, not seen)
    OBJECT_NOT_SEEN = "object_not_seen"    # look found none of the requested colours
    SCENE_CHANGED = "scene_changed"        # a block moved that the robot was not handling
    ROBOT_ALARM = "robot_alarm"


# events that put a fresh image in front of Astra on the next call
NEEDS_IMAGE = {EventType.GRASP_MISSED, EventType.SKILL_FAILED, EventType.VERIFY_AMBIGUOUS,
               EventType.OBJECT_NOT_SEEN, EventType.SCENE_CHANGED, EventType.CONTACT}
# events after which the rest of a proposed chunk is not executed
ABORTS_CHUNK = {EventType.ACTION_ERROR, EventType.SAFETY_BLOCK, EventType.CONTACT,
                EventType.OPERATOR_REJECTED, EventType.GRASP_MISSED, EventType.SKILL_FAILED,
                EventType.VERIFY_AMBIGUOUS, EventType.SCENE_CHANGED, EventType.ROBOT_ALARM}
STOPS_EPISODE = {EventType.ROBOT_ALARM}

EDGE_GRIP_POSITION = 850        # caught at/above this = touching an edge, will slip


@dataclass
class Event:
    type: EventType
    detail: str = ""
    camera: str | None = None           # which camera image would help

    def as_dict(self) -> dict:
        return {"type": self.type.value, "detail": self.detail, **({"camera": self.camera} if self.camera else {})}


@dataclass
class EventRouter:
    scene_move_mm: float = 25.0
    _last_env: dict = field(default_factory=dict)     # colour -> base xy of the last env sighting
    handling: str | None = None                       # colour the robot is picking/placing now
    pending_image: set = field(default_factory=set)   # cameras whose next image must reach Astra
    stop_reason: str | None = None
    gripper_used: bool = False                        # gripper/skill acted since the last env look


    def classify(self, name: str, args: dict, result: dict) -> list[Event]:
        events: list[Event] = []
        res = result.get("result")
        if name in ("close_gripper", "open_gripper", "set_gripper", "run_skill"):
            self.gripper_used = True
        if not result.get("ok"):
            err = str(result.get("error", ""))
            low = err.lower()
            if "contact" in low:
                events.append(Event(EventType.CONTACT, err[:160], "wrist"))
            elif "operator rejected" in low or "needs operator approval" in low:
                events.append(Event(EventType.OPERATOR_REJECTED, err[:160]))
            elif any(w in low for w in ("table", "not reachable", "camera view", "undone", "workspace",
                                         "limit", "safety", "approach zone")):
                events.append(Event(EventType.SAFETY_BLOCK, err[:160]))
            elif "alarm" in low:
                events.append(Event(EventType.ROBOT_ALARM, err[:160]))
            else:
                events.append(Event(EventType.ACTION_ERROR, err[:160]))
        elif name == "get_state" and isinstance(res, dict) and res.get("error"):
            events.append(Event(EventType.ROBOT_ALARM, "controller reports an alarm"))
        elif name == "close_gripper" and isinstance(res, dict):
            pos, holding = res.get("position"), res.get("holding")
            if holding and (pos is None or pos < EDGE_GRIP_POSITION):
                events.append(Event(EventType.GRASP_CAUGHT, f"position {pos}"))
            else:
                events.append(Event(EventType.GRASP_MISSED, f"holding={holding} position {pos}", "wrist"))
        elif name == "open_gripper":
            events.append(Event(EventType.RELEASED))
        elif name == "run_skill" and isinstance(res, dict):
            if res.get("ok"):
                events.append(Event(EventType.SKILL_OK, res.get("skill", "")))
            elif res.get("ambiguous"):
                events.append(Event(EventType.VERIFY_AMBIGUOUS, str(res.get("failure", ""))[:160],
                                    res.get("camera") or "env"))
            else:
                events.append(Event(EventType.SKILL_FAILED, f"{res.get('skill')}: {str(res.get('failure', ''))[:160]}",
                                    res.get("camera") or "wrist"))
        elif name == "look" and isinstance(res, dict):
            events += self._look_events(args, res)
        if not events:
            events.append(Event(EventType.ACTION_OK))
        for e in events:
            if e.type in NEEDS_IMAGE:
                self.pending_image.add(e.camera or "wrist")
            if e.type in STOPS_EPISODE:
                self.stop_reason = e.detail or e.type.value
        return events

    def _look_events(self, args: dict, res: dict) -> list[Event]:
        events = []
        wanted = args.get("colors") or []
        seen = {o.get("label") for o in res.get("objects", [])}
        missing = [c for c in wanted if c not in seen and c != "white"]
        if wanted and missing and len(missing) == len([c for c in wanted if c != "white"]):
            events.append(Event(EventType.OBJECT_NOT_SEEN, f"{res.get('camera')}: none of {missing}", res.get("camera")))
        if res.get("camera") == "env":
            for o in res.get("objects", []):
                xyz, label = o.get("base_xyz_mm"), o.get("label")
                if xyz is None or label in (None, "white"):
                    continue
                prev = self._last_env.get(label)
                # a block may move because the robot handled it: only flag moves nobody caused
                if prev is not None and label != self.handling and not self.gripper_used:
                    moved = ((xyz[0] - prev[0]) ** 2 + (xyz[1] - prev[1]) ** 2) ** 0.5
                    if moved > self.scene_move_mm:
                        events.append(Event(EventType.SCENE_CHANGED, f"{label} moved {moved:.0f} mm", "env"))
                self._last_env[label] = xyz[:2]
            self.gripper_used = False
        return events

    def wants_image(self, camera: str) -> bool:
        return camera in self.pending_image

    def image_delivered(self, camera: str) -> None:
        self.pending_image.discard(camera)


def aborts_chunk(events: list[Event]) -> bool:
    return any(e.type in ABORTS_CHUNK for e in events)


def objects_digest(objects: list, per_color: int = 3) -> str:
    """Every colour appears: up to per_color blobs with base coordinates each (pixel when
    there are none), and how many more there were."""
    groups: dict = {}
    for o in objects:
        groups.setdefault(o.get("label"), []).append(o)
    parts = []
    for label, blobs in groups.items():
        placed = [o for o in blobs if o.get("base_xyz_mm")]
        shown = [",".join(f"{v:.0f}" for v in o["base_xyz_mm"]) for o in placed[:per_color]]
        if not shown:
            shown = [f"px{list(blobs[0].get('pixel') or [])}"]
        rest = len(blobs) - len(shown)
        parts.append(f"{label}@" + "|".join(shown) + (f"(+{rest})" if rest else ""))
    return ";".join(parts)


def digest_line(step: int, name: str, args: dict, result: dict, events: list[Event]) -> str:
    """Deterministic one-line summary of an action (no LLM)."""
    res = result.get("result")
    a = json.dumps({k: v for k, v in (args or {}).items() if k not in ("steps", "description")},
                   ensure_ascii=False, default=str)[:120]
    if not result.get("ok"):
        out = "ERROR " + str(result.get("error", ""))[:120]
    elif isinstance(res, dict):
        bits = []
        if "pose" in res and isinstance(res["pose"], list):
            bits.append("tip=" + ",".join(f"{v:.0f}" for v in res["pose"][:3]) + f" rz={res['pose'][5]:.0f}")
        for k in ("holding", "position", "skill", "ok", "failure", "saved", "version"):
            if k in res:
                bits.append(f"{k}={str(res[k])[:60]}")
        if "objects" in res:
            bits.append("objs=" + objects_digest(res["objects"]))
        out = " ".join(bits) or "ok"
    else:
        out = str(res)[:80]
    tags = ",".join(e.type.value for e in events if e.type is not EventType.ACTION_OK)
    return f"#{step} {name}{a} -> {out}" + (f" [{tags}]" if tags else "")
