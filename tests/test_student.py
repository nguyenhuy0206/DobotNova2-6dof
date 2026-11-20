"""The student plumbing (no model is loaded): Responses-API input <-> Qwen chat, tool-call parsing."""
import json
from types import SimpleNamespace

from student import client as sc
from student.context import coerce, parse_calls, step_messages


def test_parse_and_coerce_qwen_tool_calls():
    text = ("Lowering now.\n<tool_call>\n<function=move_relative>\n<parameter=dz>\n-20\n</parameter>\n"
            "<parameter=linear>\nTrue\n</parameter>\n</function>\n</tool_call>")
    calls = parse_calls(text)
    assert calls == [{"name": "move_relative", "arguments": {"dz": "-20", "linear": "True"}}]
    schema = {"properties": {"dz": {"type": "number"}, "linear": {"type": "boolean"}}}
    assert coerce(calls[0]["arguments"], schema) == {"dz": -20, "linear": True}


def test_window_keeps_the_last_exchanges():
    hist = [{"role": "assistant", "content": str(i)} for i in range(30)]
    msgs = step_messages("SYS", "task", hist, window=4)
    assert [m["content"] for m in msgs] == ["SYS", "task", *[str(i) for i in range(22, 30)]]


def test_client_speaks_the_harness_protocol(monkeypatch):
    seen = {}

    class Tok:
        def apply_chat_template(self, msgs, tools=None, **kw):
            seen["msgs"], seen["tools"] = msgs, tools
            return "PROMPT"

    def fake_generate(model, processor, prompt, **kw):
        return SimpleNamespace(text="<tool_call>\n<function=look>\n<parameter=camera>\nenv\n</parameter>\n"
                                    "</function>\n</tool_call>", prompt_tokens=100, generation_tokens=12)

    import mlx_vlm
    monkeypatch.setattr(mlx_vlm, "generate", fake_generate)
    c = sc.LocalQwenClient.__new__(sc.LocalQwenClient)
    c.model = c.processor = None
    c.tok, c.apc, c.max_output_tokens, c.log, c.dump_dir = Tok(), None, 50, None, None
    c._conversations, c._ids = {}, iter(range(1, 99))
    tools = [{"type": "function", "name": "look", "description": "d",
              "parameters": {"properties": {"camera": {"type": "string"}}}}]
    r = c.create([{"role": "user", "content": [{"type": "input_text", "text": "STATE ..."},
                                               {"type": "input_image", "image_url": "data:..."}]}],
                 tools=tools, instructions="SYS")
    assert seen["msgs"][0] == {"role": "system", "content": "SYS"}
    assert "[camera image not shown to this model]" in seen["msgs"][1]["content"]
    assert seen["tools"][0]["function"]["name"] == "look"
    assert r.output[0].type == "function_call" and r.output[0].name == "look"
    assert json.loads(r.output[0].arguments) == {"camera": "env"} and r.usage.input_tokens == 100
    # chained call: the tool output becomes a tool message after the stored assistant turn
    c.create([{"type": "function_call_output", "call_id": r.output[0].call_id, "output": '{"ok": true}'}],
             tools=tools, previous_response_id=r.id, instructions="SYS")
    assert [m["role"] for m in seen["msgs"]] == ["system", "user", "assistant", "tool"]


def test_a_call_cut_off_at_the_token_limit_still_reaches_the_harness(monkeypatch):
    """19 Nov: 400 output tokens ended inside move_to_pose; the harness read 'no call' and ended the episode."""
    import mlx_vlm

    cut = ("Plan...\n<tool_call>\n<function=move_to_pose>\n<parameter=x>\n-442\n</parameter>\n"
           "<parameter=linear>\ntrue\n</parameter>\n<parameter=speed>\n")
    monkeypatch.setattr(mlx_vlm, "generate",
                        lambda *a, **k: SimpleNamespace(text=cut, prompt_tokens=10, generation_tokens=400))

    class Tok:
        def apply_chat_template(self, msgs, tools=None, **kw):
            return "P"

    c = sc.LocalQwenClient.__new__(sc.LocalQwenClient)
    c.model = c.processor = None
    c.tok, c.apc, c.max_output_tokens, c.log, c.dump_dir = Tok(), None, 400, None, None
    c._conversations, c._ids = {}, iter(range(1, 9))
    tools = [{"type": "function", "name": "move_to_pose", "description": "",
              "parameters": {"properties": {"x": {"type": "number"}, "linear": {"type": "boolean"},
                                            "speed": {"type": "integer"}}}}]
    r = c.create("go", tools=tools, instructions="SYS")
    assert r.output and r.output[0].name == "move_to_pose"
    assert json.loads(r.output[0].arguments)["x"] == -442
