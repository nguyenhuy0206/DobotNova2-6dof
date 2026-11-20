"""What the student sees at one step, and how its answer is read. Shared by the dataset, the
evaluation and (later) the live runtime, so training and use see the same thing.

    step_messages   system prompt + task + the last `window` exchanges (tool call + compacted result)
    call_message    a tool call as an assistant message (what the student is trained to produce)
    parse_calls     <tool_call> blocks out of generated text (Qwen3.5 XML style, or JSON)
    coerce          Qwen writes every parameter as text: back to the schema's types
"""
from __future__ import annotations

import json
import re

WINDOW = 10            # exchanges of history the student sees (bounded: prompt length = latency and memory)
RESULT_CHARS = 500     # each tool result, compacted as the harness does


def call_message(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args or {}}}]}


def result_message(result: dict, chars: int = RESULT_CHARS) -> dict:
    from agent.context import compact_result
    return {"role": "tool", "content": compact_result(result or {}, chars)}


def step_messages(system: str, task: str, history: list[dict], window: int = WINDOW) -> list[dict]:
    """history: alternating call/result messages so far (oldest first)."""
    recent = history[-2 * window:] if window else history
    return [{"role": "system", "content": system}, {"role": "user", "content": task}, *recent]


def parse_calls(text: str) -> list[dict]:
    out = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        m = re.search(r"<function=([\w.-]+)>(.*?)</function>", block, re.S)
        if m:
            args = {k: v.strip() for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", m.group(2), re.S)}
            out.append({"name": m.group(1), "arguments": args})
            continue
        try:
            j = json.loads(block.strip())
            out.append({"name": j.get("name"), "arguments": j.get("arguments") or {}})
        except ValueError:
            pass
    return out


def coerce(args: dict, schema: dict) -> dict:
    props = (schema or {}).get("properties", {})
    out = {}
    for k, v in args.items():
        typ = (props.get(k) or {}).get("type")
        if isinstance(v, str) and typ == "boolean" and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        elif isinstance(v, str) and typ in ("number", "integer", "array", "object"):
            try:
                v = json.loads(v)
            except ValueError:
                pass
            if typ == "integer" and isinstance(v, float) and v.is_integer():
                v = int(v)
        out[k] = v
    return out


def tool_schemas() -> list[dict]:
    """The tool list an explore episode offers (built on the simulated robot; nothing is executed)."""
    import sys
    import tempfile
    from pathlib import Path

    from common.paths import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from conftest import FakeGripper
    from sim_robot import SimNova2

    from actions.primitives import PrimitiveSkills
    from actions.robot_actions import RobotActions
    from agent.explore import code_specs, explore_specs, load_explore_config
    from agent.explore.diagnosis import Investigation
    from agent.explore.diagnosis import specs as diagnosis_specs
    from agent.toolkit import build_toolkit
    from agent.tools import to_openai_tools
    from common.config import AgentConfig, RobotConfig, load_perception_config
    from memory.knowledge import KnowledgeStore
    from memory.notes import NotesStore
    from memory.skills import SkillLibrary
    from memory.store import PositionStore
    from perception.scene import Scene

    tmp, cfg = Path(tempfile.mkdtemp()), load_explore_config()
    p = PrimitiveSkills(RobotActions(SimNova2(), RobotConfig(), PositionStore(tmp / "pos.json")), lambda: FakeGripper())
    p.grip_close_position = cfg.gripper.close_position
    p.grip_speed = cfg.gripper.speed
    scene = Scene(load_perception_config(), {"env": None, "wrist": None})
    know = KnowledgeStore(tmp / "c.json")
    ex, _ = build_toolkit("dry_run", robot_cfg=RobotConfig(), agent_cfg=AgentConfig(allowed_tools=cfg.allowed_tools),
                          perception_cfg=scene.cfg, primitives=p, scene=scene, library=SkillLibrary(tmp / "s.json"),
                          seed_skills=False,
                          extra_specs=explore_specs(lambda: {"visible": True}, NotesStore(tmp / "n"), knowledge=know)
                          + diagnosis_specs(Investigation(knowledge=know)) + code_specs(lambda: None))
    return [t if "function" in t else {"type": "function", "function": {k: t.get(k) for k in ("name", "description",
                                                                                              "parameters")}}
            for t in to_openai_tools(ex.list())]
