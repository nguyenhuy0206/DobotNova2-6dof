"""Tools only explore mode has: knowledge (concepts, hypotheses, observations), history, Astra's own
routines and skill versions, and the view-guard check. The rest (motion, gripper, seeing, skills)
comes from actions.registry."""
from __future__ import annotations

import json

from actions.executor import ActionSpec
from actions.registry import _obj
from common.errors import ValidationError
from memory.hypotheses import HypothesisStore
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
                   "to show (default all). Without matches or fields it also lists the variable names in use.",
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
        limit = max(1, min(int(limit), 60))
        q = str(query or "").lower()
        if kind == "episodes":
            from memory.knowledge import CALIBRATION, OLD_CALIBRATION_SINCE
            rows = [{**{k: h.get(k) for k in ("episode", "at", "claim")},
                     "calib": "tool118_table42" if str(h.get("at") or "") < OLD_CALIBRATION_SINCE else CALIBRATION}
                    for h in history(last=10000, claims_only=True) if not q or q in json.dumps(h).lower()]
            return {"episodes": rows[-limit:], "total": len(rows)}
        if kind == "old_hypotheses":
            old = HypothesisStore()
            rows = [{"id": h["id"], "status": h["status"], "claim": h["claim"][:200], "scope": h["scope"]}
                    for h in old.list(query=q or None, limit=10000)["hypotheses"]]
            return {"old_hypotheses": rows[-limit:], "total": len(rows),
                    "note": "from before concepts existed; not checked against observations"}
        if kind == "notes":
            return notes.read(q or None)
        raise ValidationError("kind: episodes | old_hypotheses | notes")

    return ActionSpec("read_history",
                      "Brief of the past, not in the prompt: kind=episodes (outcome + summary written at review), "
                      "old_hypotheses (the earlier free-text hypotheses, incl. archived), notes (raw notes). "
                      "query filters by text.",
                      read_history, _obj({"kind": {"type": "string", "enum": ["episodes", "old_hypotheses", "notes"]},
                                          "query": {"type": "string"},
                                          "limit": {"type": "integer", "minimum": 1, "maximum": 60}}, []),
                      read_only=True)

def code_specs(get_local) -> list[ActionSpec]:
    """Astra writes, tests and rolls back its own routines/checks (actions/astra_code.py)."""
    from actions.astra_code import AstraCode
    code = AstraCode.default()

    def run_routine(name: str, args: dict | None = None, kind: str = "routine") -> dict:
        local, memo = get_local(), {}
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
                   "Create or replace a routine/check with Python (same name as a core one replaces it; lift_to is "
                   "protected). File: docstring, KIND = 'routine'|'check', def run(ctx, memo, **params) -> dict with "
                   "'ok' (+ 'detail', numbers). ctx: execute(tool, args) / move(dx,dy,dz,drz,frame,speed) / "
                   "vertical(z, step_mm, speed) / tool_hop(dx,dy,speed) — all through the safety-checked executor; "
                   "tip(), state(), depth_at(pixels, window_px, plane_z, look=True) (D455 depth / base_xyz_mm at pixels "
                   "of a fresh env picture), wrist(color,min_area) / env(color) (HSV colour-threshold blobs), "
                   "anchor(blob), dry, routine(name, memo, **a), "
                   "check(...), contact (status/add/dz/target), table_z, block_top_z. memo['vars'][k] = v passes a "
                   "value to later skill steps as {k}. math, statistics, re available; no imports/files. The old "
                   "version is kept (rollback_routine); run_routine runs it on its own.",
                   code.write, _obj({"name": {"type": "string"}, "code": {"type": "string"},
                                     "note": {"type": "string"}}, ["name", "code"])),
        ActionSpec("rollback_routine",
                   "Restore version N of your routine as the newest; version 0 removes your file (the core one of "
                   "that name, if any, runs again).",
                   code.rollback, _obj({"name": {"type": "string"}, "version": {"type": "integer", "minimum": 0}},
                                       ["name", "version"])),
        ActionSpec("run_routine",
                   "Run one routine (or kind=check) directly to test it, outside a skill. Moves the robot in live "
                   "mode (all limits apply).",
                   run_routine, _obj({"name": {"type": "string"}, "args": {"type": "object"},
                                      "kind": {"type": "string", "enum": ["routine", "check"]}}, ["name"])),
    ]
