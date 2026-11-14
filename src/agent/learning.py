"""Learning from episodes: every attempt is an episode, reviewed by Astra afterwards.

    run task (episode, robot moves)  ->  reflect (no robot: outcome + skills + notes)

The reflection step reads a compact log of the episode it just did and may only use
memory tools: record_episode_outcome, save_skill (a new name, or a new VERSION of an
existing skill — the old version and its success/failure counts stay in history),
write_note, list_skills, read_notes. data/episodes/index.jsonl lists every episode
with its outcome, so progress over attempts is visible.
"""
from __future__ import annotations

import json
import time

from actions.executor import ActionExecutor, ActionSpec
from actions.registry import _obj
from common.paths import EPISODES_DIR
from memory.store import EpisodeStore

INDEX_FILE = EPISODES_DIR / "index.jsonl"

REFLECT_TOOLS = ["record_episode_outcome", "list_skills", "save_skill", "write_note", "read_notes"]

REFLECT_INSTRUCTIONS = """\
You are Astra, reviewing an episode you just ran on a real Dobot Nova 2 arm (log below).
You cannot move the robot now: you only have memory tools. Be honest and concrete.

1. record_episode_outcome: success = the task was achieved deliberately and verified by camera.
   blocks_moved = blocks deliberately picked AND placed on the table (knocked-over blocks don't count).
2. Skills — propose improvements, do not rewrite what works. Every save_skill creates a CANDIDATE
   version that only runs under operator supervision until the operator promotes it, so save only when
   the log shows a concrete reason:
   - pick_block_wrist / place_block_on_table run locally (checks + routines). If they failed the same way
     more than once, save a new version with changed parameter DEFAULTS (depth_mm, target_u/target_v,
     u_per_dy/v_per_dx, release_mm) or an extra check — never one episode's coordinates.
   - If the episode found a new working procedure with primitives, save it as a new skill with params,
     checks ({check: …}) and a recovery (open_gripper / routine lift_to only).
   - Do not save pure camera-inspection skills — look is already one call.
3. Notes — SHORT, for the next attempt: if grasp_trials lines are missing, write them (one per attempt,
   ≤ 200 chars: "<block> rz=… depth=… centred=… -> grip=… holding=… lifted=…"); then at most ONE
   "grasp_best" note (what worked, what to try next). Never "do not retry"-style advice.
4. Use as few calls as possible: you may call several tools in one turn.
Finish with a 3-line plain-text summary.
"""


def summarize_episode(store: EpisodeStore, episode_id: str, limit: int = 14000) -> str:
    """Compact text log of an episode for the reflection prompt."""
    lines = []
    for e in store.load(episode_id):
        ev = e.get("event")
        if ev == "start":
            lines.append(f"TASK: {e.get('task', '').strip()}")
        elif ev == "model" and e.get("text"):
            lines.append(f"[{e['step']}] YOU SAID: {e['text'][:400]}")
        elif ev == "tool":
            r = e.get("result", {})
            if r.get("ok"):
                res = r.get("result")
                if isinstance(res, dict):
                    keep = {k: v for k, v in res.items() if k in (
                        "pose", "holding", "position", "grip_state", "note", "saved", "version", "points",
                        "rms_mm", "leave_one_out_mm", "skill", "ok", "failed_step", "failure", "detail", "trace")}
                    if "objects" in res:
                        keep["objects"] = [(o.get("label"), o.get("base_xyz_mm")) for o in res["objects"]]
                    res = keep
                out = json.dumps(res, ensure_ascii=False, default=str)[:1500 if e["name"] == "run_skill" else 300]
            else:
                out = "ERROR " + str(r.get("error", ""))[:300]
            lines.append(f"[{e['step']}] {e['name']}({json.dumps(e.get('args', {}), ensure_ascii=False)[:200]}) -> {out}")
        elif ev == "end":
            lines.append(f"END: {e.get('reason')}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit // 3] + "\n…\n" + text[-2 * limit // 3:]


class OutcomeRecorder:
    def __init__(self, store: EpisodeStore, index_file=INDEX_FILE):
        self.store, self.index_file = store, index_file
        self.episode_id: str | None = None
        self.last: dict | None = None

    def record(self, success: bool, blocks_moved: int = 0, summary: str = "") -> dict:
        if self.episode_id is None:
            raise RuntimeError("no episode to record")
        rec = {"episode": self.episode_id, "success": bool(success), "blocks_moved": int(blocks_moved),
               "summary": str(summary)[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.store.append(self.episode_id, {"event": "outcome", **rec})
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with self.index_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.last = rec
        return {"recorded": rec}


def history(index_file=INDEX_FILE, last: int = 10) -> list[dict]:
    if not index_file.exists():
        return []
    rows = [json.loads(line) for line in index_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {r["episode"]: r for r in rows}             # a re-review replaces the earlier verdict
    return sorted(latest.values(), key=lambda r: r["episode"])[-last:]


def reflect_executor(main: ActionExecutor, recorder: OutcomeRecorder) -> ActionExecutor:
    spec = ActionSpec("record_episode_outcome",
                      "Record how the episode went: success (verified by camera), blocks_moved, summary.",
                      recorder.record,
                      _obj({"success": {"type": "boolean"}, "blocks_moved": {"type": "integer", "minimum": 0},
                            "summary": {"type": "string"}}, ["success", "summary"]))
    specs = [s for s in main.list() if s.name in REFLECT_TOOLS] + [spec]
    return ActionExecutor(specs, mode="live")


# ---------------------------------------------------------------------------
# Review of MANY episodes at once: patterns across attempts -> concrete skills
# ---------------------------------------------------------------------------
REVIEW_ALL_TOOLS = ["list_skills", "save_skill", "delete_skill", "write_note", "read_notes"]

REVIEW_ALL_INSTRUCTIONS = """\
You are Astra. Below are the logs of ALL your recent grasp episodes on a real Dobot Nova 2
(failures and successes), each with the wrist-camera image taken right before the gripper
closed (or before the block was knocked). Moving to/above the blocks already works well;
what fails is the last part: preparing and making the grasp. You cannot move the robot now.

Wrist image layout: gripper body at the bottom, two black fingers on the left and right that
close left<->right. A block ready to grasp sits BETWEEN the two fingertips, edges square to them.

1. For every episode decide the failure cause at the grasp, from the log AND its image:
   not centred (block beyond/beside the fingertips) | closed too high (fingers met air: grip 500,
   holding=false) | turned (block edges at an angle -> corner/edge grip 700+, slips) | knocked
   during descent (finger landed on the block) | other. Compare with the successes.
2. write_note topic "grasp_analysis": ONE compact table, a line per episode
   ("<episode> <block> rz=… z=… grip=… -> cause"), then ONE note "grasp_best": what the
   successes had in common and the exact settings to use next (≤ 300 characters each).
3. Rebuild the skill set so it is CONCRETE and unambiguous — exact numbers, explicit wrist looks,
   no "adapt as needed". Suggested split (each one run_skill call, so few API calls per attempt):
   - hover_over_block(x, y, rz, z_top): open_gripper, move to z_top + 95 (top of approach zone), look wrist.
   - descend_to_check(x, y, rz, z_top): ≤ 25 mm steps down to z_top + 30, then look wrist.
     (Centring/turning between here and the grasp is done by you with tool-frame move_relative
     dx/dy/drz, ≤ 15 mm / ≤ 45°, checking the wrist image.)
   - grasp_and_lift(z_top, depth): from z_top + 30 straight down in two equal steps to z_top - depth,
     close_gripper, lift 15 mm, look wrist (the result shows holding + grip position).
   - place_on_table(x, y, rz): move above (x, y) at z 165, descend in ≤ 25 mm steps to z 55,
     open_gripper, lift to 165.
   Step args may use "{param}" and "=expressions" (+ - * /). Tool limits: below z 162 every
   Cartesian move ≤ 25 mm; fingertip ≥ 52; grip closes to 500 at most.
4. delete_skill every old skill that is vague, partial or superseded by the new set.
Finish with: the cause counts, the new skill list, and the one thing to try first next time.
"""


def _wrist_image_before_grasp(store: EpisodeStore, episode_id: str) -> str | None:
    """Base64 of the last wrist look before the first close_gripper (else the last wrist look)."""
    import base64
    from pathlib import Path

    last = None
    for e in store.load(episode_id):
        if e.get("event") != "tool":
            continue
        if e["name"] == "close_gripper" and last:
            break
        res = (e.get("result") or {}).get("result")
        if e["name"] == "look" and (e.get("args") or {}).get("camera") == "wrist" and isinstance(res, dict):
            last = res.get("image_file") or last
    if last and Path(last).exists():
        return base64.b64encode(Path(last).read_bytes()).decode()
    return None


def grasp_episodes(store: EpisodeStore) -> list[str]:
    """Robot episodes that attempted a grasp (a close, or a descent near a block)."""
    out = []
    for ep in store.list():
        tools = [e for e in store.load(ep) if e.get("event") == "tool"]
        if any(e["name"] == "close_gripper" for e in tools) or sum(e["name"] == "look" for e in tools) >= 4:
            out.append(ep)
    return out


def review_all(session, episodes: list[str] | None = None, per_episode_chars: int = 2500) -> dict:
    """One review over many episodes (text logs + one wrist image each)."""
    import copy

    from agent.harness import AgentHarness

    store = session.harness.episodes
    episodes = episodes or grasp_episodes(store)
    verdicts = {h["episode"]: h for h in history(last=10000)}
    parts, images = [], []
    for i, ep in enumerate(episodes, 1):
        v = verdicts.get(ep)
        tag = "unknown" if v is None else ("SUCCESS" if v["success"] else "FAIL")
        img = _wrist_image_before_grasp(store, ep)
        if img:
            images.append(img)
        parts.append(f"===== EPISODE {i}: {ep} [{tag}] (wrist image #{len(images) if img else '-'})\n"
                     + summarize_episode(store, ep, limit=per_episode_chars))
    lib = session.harness.skills
    specs = [s for s in session.executor.list() if s.name in REVIEW_ALL_TOOLS]
    specs.append(ActionSpec("delete_skill", "Delete a skill that is vague, partial or superseded.",
                            lib.delete, _obj({"name": {"type": "string"}}, ["name"])))
    cfg = copy.copy(session.harness.config)
    cfg.system_prompt, cfg.max_steps, cfg.allowed_tools = REVIEW_ALL_INSTRUCTIONS, 20, []
    cfg.max_output_tokens = max(cfg.max_output_tokens, 8000)     # whole skills are written in one call
    reviewer = AgentHarness(session.harness.client, ActionExecutor(specs, mode="live"), cfg,
                            episodes=EpisodeStore(store.root / "reflections"), skills=lib,
                            send_images=False, notes=session.notes)
    text = reviewer.run("\n\n".join(parts), images=images)
    return {"episodes": episodes, "images": len(images), "review": text}
