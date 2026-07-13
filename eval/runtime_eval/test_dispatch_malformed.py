"""Regression: malformed tool calls return tool_result errors instead of crashing.

Run directly:
    python eval/runtime_eval/test_dispatch_malformed.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import loop  # noqa: E402
from agent.tools.builtin_tools import get_core_tools  # noqa: E402
from agent.tools.runtime import ToolExecutionRuntime  # noqa: E402


def test_runtime_returns_schema_error_not_raise():
    runtime = ToolExecutionRuntime(get_core_tools())
    before = loop.tools.tool_error_count()

    messages, tools_used = runtime.execute_tool_uses(
        [SimpleNamespace(type="tool_use", name="edit_file", input={}, id="t1")]
    )

    assert tools_used == ["edit_file"]
    assert messages[0]["is_error"] is True
    assert "InputValidationError: missing required field" in messages[0]["content"]
    assert loop.tools.tool_error_count() == before
    print("[OK] runtime malformed tool call returns schema error without raising")


class _Tool:
    def __init__(self, name, inp, tid):
        self.type = "tool_use"
        self.name = name
        self.input = inp
        self.id = tid


class _Txt:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, content, stop):
        self.content = content
        self.stop_reason = stop
        self.usage = None


def test_run_survives_malformed_tool_call():
    calls = [0]

    def fake_chat(messages, system, tools=None, max_tokens=4096, **kw):
        calls[0] += 1
        if calls[0] == 1:
            return _Resp([_Tool("edit_file", {"content": "x"}, "t1")], "tool_use")
        return _Resp([_Txt("done")], "end_turn")

    loop.llm.chat = fake_chat
    out = loop.run_task("test malformed tolerance", max_turns=5, trace=False)

    assert calls[0] >= 2
    assert "done" in out
    print(f"[OK] run_task survived malformed tool call after {calls[0]} LLM calls")


def main():
    test_runtime_returns_schema_error_not_raise()
    test_run_survives_malformed_tool_call()
    print("\n[ALL OK] malformed tool calls no longer crash the run")


if __name__ == "__main__":
    main()
