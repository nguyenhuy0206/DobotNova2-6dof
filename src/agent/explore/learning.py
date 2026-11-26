"""Learning from episodes: every attempt is an episode, reviewed by Astra afterwards.

    run task (episode, robot moves)  ->  reflect (no robot: outcome + skills + knowledge)

The reflection step reads a compact log of the episode it just did and may only use
memory tools: record_episode_outcome, save_skill (a new name, or a new VERSION of an
existing skill — the old version and its success/failure counts stay in history),
knowledge tools (concepts, hypotheses with predictions, observations), list_skills. data/episodes/index.jsonl lists every episode
with its outcome, so progress over attempts is visible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from actions.executor import ActionExecutor, ActionSpec
from actions.registry import _obj
from common.paths import EPISODES_DIR
from memory.store import EpisodeStore

INDEX_FILE = EPISODES_DIR / "index.jsonl"

REFLECT_TOOLS = ["record_episode_outcome", "list_skills", "save_skill", "rollback_skill", "retire_skill",
                 "list_routines", "read_routine", "write_routine", "rollback_routine",
                 "list_concepts", "define_concept", "update_model", "propose_hypothesis", "archive_knowledge", "query_observations", "read_history"]

REFLECT_INSTRUCTIONS = """\
You are Astra, reviewing an episode you just ran on a real Dobot Nova 2 arm (log below; wrist images taken
around failed grasps / failed runs and after lifts, when there were any). You cannot move the robot now. Be
honest and concrete.

1. record_episode_outcome: success = the task in the log's TASK line was achieved deliberately and verified
   by camera. blocks_moved = how many of the objects that task asked to move were deliberately moved to where
   it wanted them (objects displaced by accident don't count).
   The summary says WHY it failed: confirmed causes (cause -> fix) and, for the rest, the open hypotheses
   and the test to run first next episode.
2. Change what the evidence supports: skills (save_skill / rollback_skill / retire_skill), your routines
   and checks (list_routines / read_routine / write_routine / rollback_routine), knowledge (list_concepts /
   define_concept / update_model / propose_hypothesis / archive_knowledge; query_observations shows the
   recorded numbers, read_history earlier episodes). The "Knowledge check" below lists the prediction errors
   and the failures no hypothesis covers. Every save_skill creates a CANDIDATE version.
3. Use as few calls as possible: you may call several tools in one turn.
Finish with a 3-line plain-text summary.
"""


REFLECT_TOOLS_PROGRAM = ["record_episode_outcome", "list_routines", "read_routine", "write_routine", "rollback_routine",
                         "list_concepts", "define_concept", "update_model", "propose_hypothesis", "archive_knowledge",
                         "query_observations", "read_history"]

REFLECT_INSTRUCTIONS_PROGRAM = REFLECT_INSTRUCTIONS.replace(
    "2. Change what the evidence supports: skills (save_skill / rollback_skill / retire_skill), your routines\n"
    "   and checks",
    "2. Change what the evidence supports: your routines and checks (the programs the episode ran)",
).replace("Every save_skill creates a CANDIDATE version.", "Every write_routine keeps the previous version.")


def summarize_episode(store: EpisodeStore, episode_id: str, limit: int = 14000) -> str:
    """Compact text log of an episode for the reflection prompt."""
    lines = []
    for e in store.load(episode_id):
        ev = e.get("event")
        if ev == "start":
            lines.append(f"TASK: {e.get('task', '').strip()}")
        elif ev == "self_check":
            from agent.explore.checking import autonomous_feedback
            lines.append(autonomous_feedback(e))
        elif ev == "model" and e.get("text"):
            lines.append(f"[{e['step']}] YOU SAID: {e['text'][:400]}")
        elif ev == "tool":
            r = e.get("result", {})
            if r.get("ok") and e["name"] == "run_routine" and isinstance(r.get("result"), dict):
                # 23 Nov audit: a program's result was cut to 300 characters, losing its failure
                from agent.events import routine_digest
                res = r["result"]
                out = f"ok={res.get('ok')} " + routine_digest(res)
            elif r.get("ok"):
                res = r.get("result")
                if isinstance(res, dict):
                    keep = {k: v for k, v in res.items() if k in (
                        "pose", "holding", "position", "grip_state", "note", "saved", "version", "points",
                        "rms_mm", "leave_one_out_mm", "skill", "ok", "failed_step", "failure", "detail", "trace",
                        "calibration_valid", "error_mm", "hypotheses", "hypothesis", "verdict", "promoted")}
                    if "objects" in res:
                        keep["objects"] = [(o.get("label"), o.get("base_xyz_mm")) for o in res["objects"]]
                    res = keep
                out = json.dumps(res, ensure_ascii=False, default=str)[:1500 if e["name"] == "run_skill" else 300]
            else:
                out = "ERROR " + str(r.get("error", ""))[:300]
            if r.get("diagnosis"):
                out += f" [{r['diagnosis'].get('failure_id')}]"
            lines.append(f"[{e['step']}] {e['name']}({json.dumps(e.get('args', {}), ensure_ascii=False)[:300]}) -> {out}")
        elif ev == "diagnosis":
            lines.append("DIAGNOSIS: " + json.dumps(e.get("per_failure", []), ensure_ascii=False)[:2500])
        elif ev == "end":
            lines.append(f"END: {e.get('reason')}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit // 3] + "\n…\n" + text[-2 * limit // 3:]


class OutcomeRecorder:
    def __init__(self, store: EpisodeStore, index_file=INDEX_FILE, ledger=None):
        from judge.ledger import JudgeLedger
        self.ledger = ledger or JudgeLedger()
        self.store, self.index_file = store, index_file
        self._episode_id: str | None = None
        self._outcome = {"status": "inconclusive", "success_credit": False, "evaluation_id": None}
        self.last: dict | None = None

    @property
    def episode_id(self):
        return self._episode_id

    @episode_id.setter
    def episode_id(self, value):
        # Read authority when the harness binds the review, never in an Astra tool.
        self._episode_id = value
        self._outcome = self.ledger.outcome(value)

    def record(self, success: bool, blocks_moved: int = 0, summary: str = "") -> dict:
        if self.episode_id is None:
            raise RuntimeError("no episode to record")
        outcome = self._outcome
        rec = {"episode": self.episode_id, "success": outcome["success_credit"],
               "status": outcome["status"], "evaluation_id": outcome["evaluation_id"],
               "claim": {"success": bool(success), "blocks_moved": int(blocks_moved), "summary": str(summary)[:500]},
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.store.append(self.episode_id, {"event": "outcome", **rec})
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with self.index_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.last = rec
        return {"recorded": {"episode": self.episode_id, "claim": rec["claim"]}}


def history(index_file=INDEX_FILE, last: int = 10, ledger=None, *, claims_only=False) -> list[dict]:
    rows = ([json.loads(line) for line in index_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if index_file.exists() else [])
    latest = {r["episode"]: r for r in rows}             # reviews replace claims, never verdicts
    if claims_only:
        return [{"episode": r["episode"], "at": r.get("at"),
                 "claim": r.get("claim") or {k: r.get(k) for k in ("success", "blocks_moved", "summary")}}
                for r in sorted(latest.values(), key=lambda r: r["episode"])[-last:]]
    from judge.ledger import JudgeLedger
    ledger = ledger or JudgeLedger()
    for row in ledger.rows():
        if row["event"] == "episode_end":
            latest.setdefault(row["episode_id"], {"episode": row["episode_id"], "claim": None, "at": ""})
    for r in latest.values():
        outcome = ledger.outcome(r["episode"])
        r.update(success=outcome["success_credit"], status=outcome["status"], evaluation_id=outcome["evaluation_id"])
        r["blocks_moved"] = int(outcome["success_credit"])
    return sorted(latest.values(), key=lambda r: r["episode"])[-last:]


def reflect_executor(main: ActionExecutor, recorder: OutcomeRecorder, tools=None) -> ActionExecutor:
    spec = ActionSpec("record_episode_outcome",
                      "Record your claim about the episode: success (the task achieved deliberately, verified by camera), "
                      "blocks_moved (objects the task asked to move, deliberately moved where it wanted them), summary.",
                      recorder.record,
                      _obj({"success": {"type": "boolean"}, "blocks_moved": {"type": "integer", "minimum": 0},
                            "summary": {"type": "string"}}, ["success", "summary"]))
    # all_specs: the episode's tool list (program mode) does not contain the review's tools (23 Nov audit)
    wanted = tools or REFLECT_TOOLS
    specs = [s for s in getattr(main, "all_specs", {s.name: s for s in main.list()}).values() if s.name in wanted] + [spec]
    return ActionExecutor(specs, mode="live")


# ---------------------------------------------------------------------------
# Review of MANY episodes at once: patterns across attempts -> concrete skills
# ---------------------------------------------------------------------------
REVIEW_ALL_TOOLS = ["list_skills", "save_skill", "delete_skill", "list_concepts", "define_concept", "update_model", "propose_hypothesis", "archive_knowledge", "query_observations", "read_history"]

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
2. Knowledge: for each cause you found, the concept it is about (list_concepts / define_concept) and a
   hypothesis with a prediction (propose_hypothesis when -> expect); query_observations shows the numbers.
3. Rebuild the skill set so it is CONCRETE and unambiguous — exact numbers, explicit wrist looks,
   no "adapt as needed". Step args may use "{param}" and "=expressions" (+ - * /).
   20 Nov: arm motions run at fixed 50 %; contact sends Pause() and ends the episode.
   Table and path limits come from get_state and config/robot.yaml; grip closes to 500 at most.
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


def _failed_attempt(e: dict) -> str | None:
    """Short outcome text when this tool event is a failed grasp / failed run, else None."""
    r = e.get("result") or {}
    res = r.get("result") if isinstance(r.get("result"), dict) else {}
    if e["name"] == "close_gripper" and r.get("ok") and res.get("holding") is False:
        return f"grasp missed (closed to {res.get('position')})"
    if e["name"] in ("run_skill", "run_routine") and r.get("ok") and res.get("ok") is False:
        return f"{res.get('failure') or 'failed'}: {str(res.get('detail', ''))[:110]}"
    if not r.get("ok") and "CONTACT" in str(r.get("error", "")):
        return "contact: " + str(r.get("error"))[:110]
    return None


def failure_images(store: EpisodeStore, episode_id: str, limit: int = 4) -> list[tuple[str, str]]:
    """(caption, base64 jpeg) around each failed grasp / failed skill or routine run / contact:
    the last wrist image BEFORE it (when one was taken since the previous motion boundary) and
    the wrist image right AFTER it (the harness auto-looks after a failure). The last `limit`."""
    import base64

    def wrist_image(ev: dict) -> str | None:
        fr = (ev.get("result") or {}).get("result")
        if ev["name"] == "look" and (ev.get("args") or {}).get("camera") == "wrist" and isinstance(fr, dict) \
                and fr.get("image_file") and Path(fr["image_file"]).exists():
            return fr["image_file"]
        return None

    events = [e for e in store.load(episode_id) if e.get("event") == "tool"]
    out: list[tuple[str, str]] = []
    for i, e in enumerate(events):
        what = _failed_attempt(e)
        if what is None:
            continue
        args = json.dumps(e.get("args") or {}, ensure_ascii=False)[:120]
        head = f"step {e.get('step')} {e['name']}({args}) -> {what}"
        before = next((wrist_image(b) for b in reversed(events[max(0, i - 4):i]) if wrist_image(b)), None)
        after = next((wrist_image(f) for f in events[i + 1:i + 3] if wrist_image(f)), None)
        for tag, path in (("BEFORE", before), ("AFTER", after)):
            if path:
                out.append((f"{head} [{tag}]", base64.b64encode(Path(path).read_bytes()).decode()))
    return out[-limit:]


def post_lift_images(store: EpisodeStore, episode_id: str, limit: int = 3) -> list[tuple[str, str]]:
    """(caption, base64 jpeg) of the wrist images the harness took after each lift while holding
    (how the object sits in the fingers), the last `limit`. Shown in the review even when the
    episode succeeded."""
    import base64

    out = []
    for e in store.load(episode_id):
        if e.get("event") != "tool" or not e.get("post_lift"):
            continue
        fr = (e.get("result") or {}).get("result")
        path = fr.get("image_file") if isinstance(fr, dict) else None
        if path and Path(path).exists():
            out.append((f"step {e.get('step')}: after lifting {e.get('lift_mm')} mm, gripper position "
                        f"{e.get('grip_position')}", base64.b64encode(Path(path).read_bytes()).decode()))
    return out[-limit:]


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
                            send_images=False, notes=session.knowledge)
    text = reviewer.run("\n\n".join(parts), images=images)
    return {"episodes": episodes, "images": len(images), "review": text}
