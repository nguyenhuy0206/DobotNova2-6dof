"""The student as the harness's model: a drop-in for agent.astra_client.AstraClient.

`create(input, tools, previous_response_id, instructions)` takes Responses-API style input and
returns an object the harness reads like a Responses-API response (`id`, `output` with
`function_call` items, `output_text`, `usage`). Inside: Qwen's chat template, MLX generation on this
Mac, and APC prefix caching — the system prompt + tools are identical every turn, so after the
first request only the new part of the prompt is computed (~2 s instead of ~15 s).

Images are left out by default (text-only student; look has no detections, 20 Nov).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

from student.context import coerce, parse_calls
from student.paths import FROZEN_DIR


class LocalQwenClient:
    def __init__(self, model: str | Path = FROZEN_DIR, adapter: str | Path | None = None,
                 max_output_tokens: int = 400, images: bool = False, log=None, dump_dir: str | Path | None = None):
        from mlx_vlm import load
        from mlx_vlm.apc import APCManager

        self.model_path, self.adapter = str(model), (str(adapter) if adapter else None)
        self.model, self.processor = load(self.model_path, adapter_path=self.adapter)
        self.tok = self.processor.tokenizer
        self.apc = APCManager()
        self.max_output_tokens, self.images, self.log = max_output_tokens, images, log
        self.dump_dir = Path(dump_dir) if dump_dir else None    # every rendered prompt + answer, for debugging
        self.endpoint = f"local MLX: {Path(self.model_path).name}" + (f" + {Path(self.adapter).name}" if adapter else "")
        self._conversations: dict[str, list[dict]] = {}       # response id -> messages (for previous_response_id)
        self._ids = itertools.count(1)

    def start_episode(self, folder: Path) -> None:
        """Called by the harness when an episode starts: every request of it is logged to
        <episode folder>/model_io.jsonl (full messages, raw answer, parsed calls, timing, tokens)."""
        self.io_log = Path(folder) / "model_io.jsonl"
        self.io_log.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- Responses API -> chat messages
    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for part in content or []:
            if part.get("type") in ("input_text", "output_text", "text"):
                parts.append(part.get("text", ""))
            elif part.get("type") == "input_image":
                parts.append("[camera image not shown to this model]")
        return "\n".join(p for p in parts if p)

    def _messages(self, input, previous_response_id) -> list[dict]:
        msgs = list(self._conversations.get(previous_response_id, [])) if previous_response_id else []
        items = [{"role": "user", "content": input}] if isinstance(input, str) else list(input or [])
        for item in items:
            if item.get("type") == "function_call_output":
                msgs.append({"role": "tool", "content": item.get("output", "")})
            elif item.get("role") in ("user", "system", "developer"):
                msgs.append({"role": "user" if item["role"] == "developer" else item["role"],
                             "content": self._text(item.get("content"))})
        return msgs

    @staticmethod
    def _chat_tools(tools) -> list[dict]:
        return [t if "function" in t else {"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters") or {}}}
            for t in tools or []]

    # ---------------------------------------------------------------- the call the harness makes
    def create(self, input, tools: list[dict] | None = None, previous_response_id: str | None = None,
               instructions: str | None = None):
        from mlx_vlm import generate

        history = self._messages(input, previous_response_id)
        msgs = ([{"role": "system", "content": instructions}] if instructions else []) + history
        chat_tools = self._chat_tools(tools)
        prompt = self.tok.apply_chat_template(msgs, tools=chat_tools or None, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        t0 = time.perf_counter()
        r = generate(self.model, self.processor, prompt, max_tokens=self.max_output_tokens, temperature=0.0,
                     apc_manager=self.apc)
        schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in chat_tools}
        rid = f"local-{next(self._ids)}"
        output, tool_calls = [], []
        text_for_calls = r.text
        last = r.text.rfind("<tool_call>")
        if last != -1 and "</tool_call>" not in r.text[last:]:
            # cut off at the token limit inside a call: close it, so the harness gets the call (and its
            # validation error if a parameter is missing) instead of reading "no call" as the final report
            tail = r.text[last:]
            if tail.count("<parameter=") > tail.count("</parameter>"):
                tail += "\n</parameter>"
            text_for_calls = r.text[:last] + tail + "\n</function>\n</tool_call>"
        for k, c in enumerate(parse_calls(text_for_calls)):
            args = coerce(c.get("arguments") or {}, schemas.get(c.get("name"), {}))
            output.append(SimpleNamespace(type="function_call", name=c.get("name"), arguments=json.dumps(args),
                                          call_id=f"{rid}-{k}"))
            tool_calls.append({"type": "function", "function": {"name": c.get("name"), "arguments": args}})
        text = r.text.split("<tool_call>")[0].strip()
        self._conversations[rid] = history + [{"role": "assistant", "content": text,
                                               **({"tool_calls": tool_calls} if tool_calls else {})}]
        if getattr(self, "io_log", None):
            with self.io_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "id": rid, "previous": previous_response_id,
                                    "seconds": round(time.perf_counter() - t0, 2), "prompt_tokens": r.prompt_tokens,
                                    "output_tokens": r.generation_tokens, "messages": msgs,
                                    "tools": [t["function"]["name"] for t in chat_tools], "answer": r.text,
                                    "calls": [{"name": o.name, "arguments": json.loads(o.arguments)} for o in output]},
                                   ensure_ascii=False, default=str) + "\n")
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            (self.dump_dir / f"{rid}.txt").write_text(prompt + "\n\n===== ANSWER =====\n" + r.text, encoding="utf-8")
        if self.log:
            self.log(f"qwen {time.perf_counter() - t0:.1f} s · prompt {r.prompt_tokens} · out {r.generation_tokens} · "
                     f"{len(output)} call(s)")
        usage = SimpleNamespace(input_tokens=r.prompt_tokens, output_tokens=r.generation_tokens,
                                total_tokens=r.prompt_tokens + r.generation_tokens,
                                input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=0))
        return SimpleNamespace(id=rid, output=output, output_text=text, usage=usage, raw_text=r.text)
