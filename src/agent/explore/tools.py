"""Tools only explore mode has: knowledge (concepts, hypotheses, observations), history, Astra's own
routines and skill versions, and the view-guard check. The rest (motion, gripper, seeing, skills)
comes from actions.registry."""
from __future__ import annotations

import json

from actions.executor import ActionSpec
from actions.registry import _obj
from common.errors import ValidationError
from memory.knowledge import MAX_LIVE_HYPOTHESES, KnowledgeStore
from memory.notes import NotesStore
from memory.skills import SkillLibrary
from perception.robot_view import RobotViewGuard

def explore_specs(guard: RobotViewGuard | None, notes: NotesStore, library: SkillLibrary | None = None,
                  knowledge: KnowledgeStore | None = None) -> list[ActionSpec]:
    knowledge = knowledge if knowledge is not None else KnowledgeStore()
    specs = [
        *knowledge_specs(knowledge),
        history_spec(notes),
    ]
    if library is not None:
        specs += [
            ActionSpec("rollback_skill",
                       "Restore an earlier version of a skill as its newest candidate version; operator ledger review "
                       "is required to validate it. "
                       "list_skills {name} shows the versions and their success/failure counts.",
                       library.rollback, _obj({"name": {"type": "string"}, "version": {"type": "integer", "minimum": 1}},
                                              ["name", "version"])),
            ActionSpec("retire_skill",
                       "Retire a draft skill that never worked and will not be used (kept with its history, "
                       "run_skill refuses it). Built-in skills cannot be retired.",
                       library.retire, _obj({"name": {"type": "string"}, "reason": {"type": "string"}}, ["name"])),
        ]
    if guard is not None:
        specs.append(ActionSpec("check_robot_in_view",
                                "Is the robot (red gripper marker) visible to the env camera right now?",
                                guard, _obj({}, []), read_only=True))
    return specs

def knowledge_specs(store: KnowledgeStore) -> list[ActionSpec]:
    cond = {"type": "object", "description": "variable -> value, [lo, hi] (numeric range) or [one, of, these]. "
                                             "Variable names as in query_observations"}
    return [
        ActionSpec("list_concepts",
                   "Your concepts with their model and hypotheses (status, counts, tested range). concept_id=… "
                   "shows one with the ids of its supporting / contradicting observations.",
                   store.list, _obj({"concept_id": {"type": "string"}, "include_archived": {"type": "boolean"}}, []),
                   read_only=True),
        ActionSpec("define_concept",
                   "A physical quantity or relation you want to understand (e.g. where something appears, how high "
                   "something is, what a measurement depends on), the variables it may depend on and your model of "
                   "it so far. A quantity that an existing concept already covers is refused.",
                   store.define_concept, _obj({"quantity": {"type": "string"},
                                               "depends_on": {"type": "array", "items": {"type": "string"}},
                                               "model": {"type": "string"}}, ["quantity"])),
        ActionSpec("update_model",
                   "Replace your model of a concept (numbers, formula, ranges) and optionally what it depends on.",
                   store.update_model, _obj({"concept_id": {"type": "string"}, "model": {"type": "string"},
                                             "depends_on": {"type": "array", "items": {"type": "string"}}},
                                            ["concept_id", "model"])),
        ActionSpec("propose_hypothesis",
                   "An explanation within a concept: mechanism (why) and a prediction — `when` (conditions an "
                   "observation must meet) and `expect` (what it must then show). Evidence is computed from every "
                   "matching observation, past and future; the result shows how it fits the past ones. Variables "
                   "must be ones observations carry (query_observations lists them; record_measurement adds one). "
                   "A broader "
                   "hypothesis is the same expectation with fewer or wider conditions. The same prediction or "
                   f"mechanism returns the existing one; at most {MAX_LIVE_HYPOTHESES} live hypotheses per concept.",
                   store.propose, _obj({"concept_id": {"type": "string"}, "mechanism": {"type": "string"},
                                        "when": cond, "expect": cond}, ["concept_id", "mechanism", "when", "expect"])),
        ActionSpec("archive_knowledge",
                   "Take a concept (C#) or hypothesis (C#.n) out of the prompts when it no longer applies; it is kept.",
                   store.archive, _obj({"knowledge_id": {"type": "string"}, "reason": {"type": "string"}},
                                       ["knowledge_id", "reason"])),
        ActionSpec("query_observations",
                   "Recorded observations (newest last) whose variables match `where`; fields = which variables "
                   "to show (default all); a field an observation lacks shows as null. Only observations recorded "
                   "since the current calibration. Without matches or fields it also lists the variable names "
                   "used in them.",
                   store.query, _obj({"where": cond, "fields": {"type": "array", "items": {"type": "string"}},
                                      "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                                      "this_episode": {"type": "boolean"}}, []), read_only=True),
        ActionSpec("record_measurement",
                   "Record values you measured yourself (e.g. read from an image) as an observation, together "
                   "with the current pose and gripper. Single values, your own variable names.",
                   store.record_measurement, _obj({"values": {"type": "object"}, "note": {"type": "string"}},
                                                  ["values"])),
    ]

def history_spec(notes: NotesStore) -> ActionSpec:
    def read_history(kind: str = "episodes", query: str = "", limit: int = 15) -> dict:
        from agent.explore.learning import history
        from memory.knowledge import calibration_since, episode_time
        since = calibration_since() or ""          # operator, 23 Nov: the current calibration only
        limit = max(1, min(int(limit), 60))
        q = str(query or "").lower()
        if kind == "episodes":
            rows = [{k: h.get(k) for k in ("episode", "at", "claim")}
                    for h in history(last=10000, claims_only=True)
                    if episode_time(h.get("episode")) >= since and (not q or q in json.dumps(h).lower())]
            return {"episodes": rows[-limit:], "total": len(rows)}
        if kind == "notes":
            rows = [n for n in notes.read(None, limit=100000)["notes"]
                    if str(n.get("t", "")) >= since[:16] and (not q or q in json.dumps(n).lower())]
            return {"notes": rows[-limit:], "total": len(rows)}
        raise ValidationError("kind: episodes | notes")

    return ActionSpec("read_history",
                      "Brief of the past (since the current calibration), not in the prompt: kind=episodes (the "
                      "claim written at each review: success / blocks_moved / summary as claimed, not a verified "
                      "outcome), notes (raw notes). query keeps the entries containing that text; limit = the "
                      "newest N.",
                      read_history, _obj({"kind": {"type": "string", "enum": ["episodes", "notes"]},
                                          "query": {"type": "string"},
                                          "limit": {"type": "integer", "minimum": 1, "maximum": 60}}, []),
                      read_only=True)

def code_specs(get_local) -> list[ActionSpec]:
    """Astra writes, tests and rolls back its own routines/checks (actions/astra_code.py)."""
    from actions.astra_code import AstraCode
    code = AstraCode.default()

    def run_routine(name: str, args: dict | None = None, kind: str = "routine") -> dict:
        local, memo = get_local(), {"vars": {}}
        out = (local.check if kind == "check" else local.routine)(name, memo, **(args or {}))
        out = {k: v for k, v in out.items() if k != "error_result"}
        if memo.get("vars"):
            out["vars"] = memo["vars"]
        return out

    return [
        ActionSpec("list_routines",
                   "Your own routines/checks (src/agent/astra_skills, with version + params) and the core ones you "
                   "can replace. Skill steps call them as {routine: NAME} / {check: NAME}.",
                   code.list, _obj({}, []), read_only=True),
        ActionSpec("read_routine", "Python source of a routine/check (yours or core) and its version history.",
                   code.read, _obj({"name": {"type": "string"}}, ["name"]), read_only=True),
        ActionSpec("write_routine",
                   "Create or replace a routine/check: a Python file run locally by run_routine (same name as a core "
                   "one replaces it; lift_to is protected). File: a docstring, KIND = 'routine' or 'check', and "
                   "def run(ctx, memo, **params) returning a dict with 'ok' (bool) plus anything else (e.g. 'detail', "
                   "numbers); run_routine's args are passed as Python keyword arguments. memo: a dict shared with "
                   "routines/checks called from this one; memo['vars'] comes back as 'vars' in the result. "
                   "ctx (nothing else exists on it, checked when written):\n"
                   "- ctx.execute(tool, args) -> the tool's result envelope {ok, result | error, error_type, executed}; "
                   "never raises for a refused move, check r['ok']. Tools and their arguments are listed below.\n"
                   "- ctx.move(dx=0, dy=0, dz=0, drz=0, frame='base', speed=None) = execute('move_relative', …).\n"
                   "- ctx.vertical(z, step_mm=19, speed=None): straight up/down to fingertip z in steps of step_mm "
                   "(> 0); when a lift is not reachable it moves 15 mm towards the base (up to 3 times) and goes on; "
                   "returns {ok, z, moves[, detail, detours]}, ok=false if z was not reached.\n"
                   "- ctx.tool_hop(dx, dy, speed=None): tool-frame sideways move in pieces of at most 20 mm.\n"
                   "- ctx.tip() -> fingertip [x, y, z, rx, ry, rz]; ctx.state() -> {pose, mode, error, moving, "
                   "gripper}.\n"
                   "- ctx.depth_at(pixels, window_px=2, plane_z=None, look=True) -> list of points as in the depth_at "
                   "tool; pixels are FULL-image pixels (not pixels of a zoomed picture); look=True takes a new env "
                   "frame first (the scene as it is then, the arm included), look=False uses the frame of the last "
                   "env look/depth_at. Raises ValueError when depth_at fails.\n"
                   "- ctx.anchor(obj) -> bottom-centre (x + w/2, y + h) of obj['bbox'] = [x, y, w, h].\n"
                   "- ctx.routine(name, memo, **a) / ctx.check(name, memo, **a): call another routine/check.\n"
                   "- ctx.dry (True in dry_run), ctx.table_z, ctx.block_top_z, ctx.trace (the entries so far).\n"
                   "Code cannot see pictures: pixels come in as params. One run: at most 120 recorded ctx calls "
                   "(each routine/check called from it has its own 120); the 300 s limit is checked at each "
                   "recorded call. math, statistics, re are available; no imports, files, classes, lambdas, names "
                   "starting with '__' or attributes starting with '_'. The previous version is kept "
                   "(rollback_routine).",
                   code.write, _obj({"name": {"type": "string"}, "code": {"type": "string"},
                                     "note": {"type": "string"}}, ["name", "code"])),
        ActionSpec("rollback_routine",
                   "Restore version N of your routine as the newest; version 0 removes your file (the core one of "
                   "that name, if any, runs again).",
                   code.rollback, _obj({"name": {"type": "string"}, "version": {"type": "integer", "minimum": 0}},
                                       ["name", "version"])),
        ActionSpec("run_routine",
                   "Run one of your routines (or kind=check) with args (keyword arguments of its run()). In live mode "
                   "the robot moves; every motion is checked like a tool call. Returns the program's own dict (its "
                   "'ok' is the program's result) plus vars (memo['vars']); one of YOUR routines also returns "
                   "routine, version and trace: one entry per ctx.execute / move / vertical / tool_hop / depth_at / "
                   "routine / check call in order (tip, state and anchor are not recorded) {i, tool, args, ok, s "
                   "(seconds), error, pose (fingertip x,y,z after it), grip_status, position, holding, z, moves, "
                   "detail, points (depth_at)}; a nested routine/check is one entry holding its own trace (core "
                   "routines have none). A program that raises returns ok=false with the exception and its line "
                   "(code_error). If a move is stopped by the safety layer (contact, alarm, timeout) the robot is "
                   "paused, later motions of the program are refused and the episode ends after this call.",
                   run_routine, _obj({"name": {"type": "string"}, "args": {"type": "object"},
                                      "kind": {"type": "string", "enum": ["routine", "check"]}}, ["name"]),
                   delegates=True),
    ]


def routine_api(executor) -> str:
    """The tools a program may call through ctx.execute, with their real arguments and defaults (23 Nov audit:
    program mode offers no motion tools, so their schemas never reached the model)."""
    import inspect
    from actions.astra_code import ROUTINE_TOOLS
    lines = []
    for name in sorted(ROUTINE_TOOLS):
        spec = executor.all_specs.get(name)
        if spec is None:
            continue
        props, req = spec.parameters.get("properties", {}), set(spec.parameters.get("required", []))
        try:
            defaults = {k: v.default for k, v in inspect.signature(spec.handler).parameters.items()
                        if v.default is not inspect.Parameter.empty}
        except (TypeError, ValueError):
            defaults = {}
        args = []
        for k, sch in props.items():
            if k == "speed":
                continue                                # ignored (fixed speed)
            kind = sch.get("type", "")
            if "enum" in sch:
                kind = "|".join(map(str, sch["enum"]))
            text = f"{k}: {kind}" + ("" if k in req else f" = {defaults.get(k)!r}" if k in defaults else " (optional)")
            args.append(text)
        offered = name in executor.specs          # the model has this tool's full description already
        lines.append(f"- {name}({', '.join(args)}): "
                     + (spec.description.split('. ')[0].rstrip('.') + "." if offered else spec.description))
    return ("\nTools for ctx.execute (arm-motion results carry pose = fingertip [x, y, z, rx, ry, rz], flange_pose, "
            "joints_deg; gripper results carry position, grip_state and, for close_gripper, holding):\n"
            + "\n".join(lines))


def reference_spec(scene, task: str) -> ActionSpec:
    """24 Nov (operator, threading): compare a fresh wrist picture with the operator's reference pictures."""
    from perception.reference import compare, load_references

    def compare_reference(camera: str = "wrist") -> dict:
        if camera not in scene.sources:
            raise ValidationError(f"camera must be one of {sorted(scene.sources)}")
        frame = scene.observe(camera)
        out = compare(frame.color, load_references(task))
        picture = out.pop("picture")
        return {**out, "camera": "reference", "image_jpeg_b64": scene.encode_image(picture, []),
                "picture_label": (f"Picture: comparison, three raw pictures side by side: LEFT = the {camera} "
                                  "camera now, MIDDLE = the closest operator reference of the finished state "
                                  "(success), RIGHT = the closest operator reference of the unfinished state (fail)")}

    return ActionSpec("compare_reference",
                      "Take a fresh wrist picture and compare it with the operator's reference pictures of this "
                      "task, taken with the wrist camera: 'success' (finished) and 'fail' "
                      "(not finished). Returns scores (normalised correlation of the whole small, blurred, grey "
                      "pictures with the closest reference of each label: 1 = identical), closer_to, margin and "
                      "one picture: now | closest success reference | closest fail reference. The scores compare "
                      "whole pictures (a different arm pose changes them too); they are a cue, not a verdict.",
                      compare_reference, _obj({"camera": {"type": "string", "enum": ["wrist"]}}, []),
                      read_only=True)
