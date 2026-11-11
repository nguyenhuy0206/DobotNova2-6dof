"""Agent harness: model ⇄ tools loop with logging to an episode file.

    task ─► Astra ─► function_call(s) ─► ActionExecutor (dry_run/confirm/live)
              ▲                                    │
              └──────── function_call_output ◄─────┘

Every step is appended to data/episodes/<id>.jsonl for replay and learning.
"""
from __future__ import annotations

import json

from actions.executor import ActionExecutor
from agent.astra_client import AstraClient
from agent.tools import to_openai_tools
from common.config import AgentConfig
from common.log import get_logger
from memory.skills import SkillLibrary
from memory.store import EpisodeStore

log = get_logger(__name__)

DEFAULT_INSTRUCTIONS = (
    "You control a Dobot Nova 2 six-axis arm with a PGC gripper and an RGB-D camera. "
    "Only act through the provided tools. Always call get_robot_state before moving. "
    "Use small, conservative motions. If a tool returns an error, stop and explain."
)


class AgentHarness:
    def __init__(self, client: AstraClient, executor: ActionExecutor, config: AgentConfig,
                 episodes: EpisodeStore | None = None, skills: SkillLibrary | None = None):
        self.client = client
        self.executor = executor
        self.config = config
        self.episodes = episodes or EpisodeStore()
        self.skills = skills or SkillLibrary()

    def _instructions(self, task: str) -> str:
        text = self.config.system_prompt or DEFAULT_INSTRUCTIONS
        recalled = self.skills.search(task)
        if recalled:
            text += "\n\nRelevant skills from memory:\n" + "\n".join(
                f"- {s.name}: {s.description} steps={json.dumps(s.steps)}" for s in recalled)
        return text

    def run(self, task: str) -> str:
        episode = self.episodes.start(task, {"model": self.config.model, "mode": self.executor.mode})
        tools = to_openai_tools(self.executor.list())
        response = self.client.create(task, tools=tools, instructions=self._instructions(task))

        for step in range(self.config.max_steps):
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            self.episodes.append(episode, {"event": "model", "step": step, "response_id": response.id,
                                           "text": response.output_text, "calls": len(calls)})
            if not calls:
                self.episodes.append(episode, {"event": "end", "reason": "done"})
                return response.output_text

            outputs = []
            for call in calls:
                args = json.loads(call.arguments or "{}")
                result = self.executor.execute(call.name, args)
                log.info("tool %s(%s) -> %s", call.name, args, result)
                self.episodes.append(episode, {"event": "tool", "step": step, "name": call.name,
                                               "args": args, "result": result})
                outputs.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(result, ensure_ascii=False, default=str)})
            response = self.client.create(outputs, tools=tools, previous_response_id=response.id)

        self.episodes.append(episode, {"event": "end", "reason": "max_steps"})
        return f"Stopped after max_steps={self.config.max_steps}. Last: {response.output_text}"
