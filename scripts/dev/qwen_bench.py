#!/usr/bin/env python3
"""Can Qwen3.5-0.8B (MLX, local) drive the robot fast enough? Speed, tool calls, vision, long prompt.

    .venv/bin/python scripts/dev/qwen_bench.py                 all four tests
    .venv/bin/python scripts/dev/qwen_bench.py --model mlx-community/Qwen3.5-0.8B-bf16
No robot, no cameras needed (the vision test uses a saved D455 image).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401

TOOLS = [
    {"type": "function", "function": {"name": "look", "description": "Take a picture with a camera and detect coloured objects.",
     "parameters": {"type": "object", "properties": {"camera": {"type": "string", "enum": ["env", "wrist"]},
                                                     "colors": {"type": "array", "items": {"type": "string"}}},
                    "required": ["camera"]}}},
    {"type": "function", "function": {"name": "move_relative", "description": "Move the fingertip by dx, dy, dz mm in the base frame.",
     "parameters": {"type": "object", "properties": {"dx": {"type": "number"}, "dy": {"type": "number"},
                                                     "dz": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "close_gripper", "description": "Close the fingers until they hold something.",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
]
TOOL_TASKS = [
    ("Look with the environment camera for the blue cube.", "look"),
    ("Lower the gripper by 20 mm.", "move_relative"),
    ("The fingers are around the cube now. Grip it.", "close_gripper"),
    ("Move the gripper 15 mm toward +x and 30 mm up.", "move_relative"),
]


def tool_calls(text: str) -> list[dict]:
    out = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        block = block.strip()
        try:
            out.append(json.loads(block))
            continue
        except ValueError:
            pass
        m = re.search(r"<function=(\w+)>(.*?)</function>", block, re.S)        # Qwen3.5 XML style
        if m:
            args = dict(re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", m.group(2), re.S))
            out.append({"name": m.group(1), "arguments": args})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-8bit")
    ap.add_argument("--image", default=None, help="a D455 image (default: the latest calibration session)")
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    t = time.perf_counter()
    model, processor = load(args.model)
    print(f"load {time.perf_counter() - t:.1f} s")
    tok = getattr(processor, "tokenizer", processor)

    def run(messages, tools=None, image=None, max_tokens=200):
        prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False) if image is None else \
            apply_chat_template(processor, model.config, messages, num_images=1, enable_thinking=False)
        t0 = time.perf_counter()
        r = generate(model, processor, prompt, image=image, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return r, time.perf_counter() - t0

    print("\n== 1. speed (short chat)")
    r, dt = run([{"role": "user", "content": "In one sentence: what does a robot gripper do?"}])
    print(f"{dt:.2f} s · prompt {r.prompt_tokens} tok @ {r.prompt_tps:.0f}/s · out {r.generation_tokens} tok "
          f"@ {r.generation_tps:.0f}/s · peak {r.peak_memory:.2f} GB\n   {r.text.strip()[:160]}")

    print("\n== 2. tool calls (3 tools)")
    sys_msg = {"role": "system", "content": "You control a robot arm. Answer only with a tool call."}
    ok = 0
    for task, want in TOOL_TASKS:
        r, dt = run([sys_msg, {"role": "user", "content": task}], tools=TOOLS)
        calls = tool_calls(r.text)
        hit = bool(calls) and calls[0].get("name") == want
        ok += hit
        print(f"{'✓' if hit else '✗'} {dt:.2f} s  {task!r} -> {calls or r.text.strip()[:120]!r}")
    print(f"   {ok}/{len(TOOL_TASKS)} right tool")

    print("\n== 3. vision (env image with the three cubes on the box, gripper empty above them)")
    img = args.image or next((str(p) for p in sorted(Path("data/episodes").glob("*/000-look.jpg"))[::-1]), None)
    if img:
        for q, truth in [("Which coloured cubes are on the white box? Answer with a list of colours, left to right.",
                          "green, blue, yellow"),
                         ("Is the robot gripper holding anything? Answer yes or no and why, one sentence.", "no")]:
            r, dt = run([{"role": "user", "content": q}], image=img, max_tokens=80)
            print(f"   {dt:.2f} s ({r.prompt_tokens} tok) [truth: {truth}] {' '.join(r.text.split())[:160]}")
    else:
        print("   no image found, skipped")

    print("\n== 4. the real explore prompt + all its tools, a 4-turn dialogue with prompt caching")
    from mlx_vlm.generate.common import PromptCacheState
    from agent.explore import EXPLORE_INSTRUCTIONS
    tools = explore_tools()
    state = PromptCacheState()
    msgs = [{"role": "system", "content": EXPLORE_INSTRUCTIONS},
            {"role": "user", "content": "Move ALL THREE cubes (blue, green, yellow) from the top of the white box onto "
                                        "free spots of the metal table. Start by looking."}]
    results = [{"ok": True, "result": {"camera": "env", "objects": [
                   {"label": "blue", "base_xyz_mm": [-460, -377, 80]}, {"label": "green", "base_xyz_mm": [-454, -411, 80]},
                   {"label": "yellow", "base_xyz_mm": [-468, -355, 82]}]}},
               {"ok": True, "result": {"pose": [-460, -377, 140, 180, 0, 0]}},
               {"ok": True, "result": {"pose": [-460, -377, 75, 180, 0, 0], "gripper": "open"}}]
    for turn in range(4):
        prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        t0 = time.perf_counter()
        r = generate(model, processor, prompt, max_tokens=200, temperature=0.0, prompt_cache_state=state)
        dt = time.perf_counter() - t0
        print(f"   turn {turn + 1}: {dt:.2f} s · prompt {r.prompt_tokens} tok · out {r.generation_tokens} tok @ "
              f"{r.generation_tps:.0f}/s | {' '.join(r.text.split())[:170]}")
        msgs += [{"role": "assistant", "content": r.text.strip()},
                 {"role": "tool", "content": json.dumps(results[min(turn, len(results) - 1)])}]
    print(f"   peak memory {r.peak_memory:.2f} GB")


def explore_tools() -> list[dict]:
    """The tool list an explore episode offers, built on the simulated robot (no hardware)."""
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from conftest import FakeGripper
    from sim_robot import SimNova2

    from actions.primitives import PrimitiveSkills
    from actions.robot_actions import RobotActions
    from agent.explore import explore_specs, load_explore_config
    from agent.tools import to_openai_tools
    from agent.toolkit import build_toolkit
    from common.config import AgentConfig, PerceptionConfig, RobotConfig
    from memory.knowledge import KnowledgeStore
    from memory.notes import NotesStore
    from memory.skills import SkillLibrary
    from memory.store import PositionStore
    from perception.scene import Scene
    tmp, cfg = Path(tempfile.mkdtemp()), load_explore_config()
    p = PrimitiveSkills(RobotActions(SimNova2(), RobotConfig(), PositionStore(tmp / "pos.json")), lambda: FakeGripper())
    p.grip_close_position = cfg.gripper.close_position
    from agent.explore import code_specs
    from agent.explore.diagnosis import Investigation
    from agent.explore.diagnosis import specs as diagnosis_specs
    from common.config import load_perception_config
    # the real colours and both cameras (sources are never read: nothing is executed)
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

if __name__ == "__main__":
    main()
