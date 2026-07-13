"""Approval callback regression for ToolExecutionRuntime.

Run directly:
    python eval/runtime_eval/test_approval.py
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

from agent import tools  # noqa: E402
from agent.tools.runtime import ToolExecutionRuntime  # noqa: E402
from agent.tools.contracts import Tool  # noqa: E402


def _run_fake_tool(*, is_subagent: bool = False):
    calls = []
    runtime = ToolExecutionRuntime(
        [
            Tool(
                name="faketool",
                description="fake tool",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                call=lambda inp, context: calls.append(dict(inp)) or "executed",
            )
        ],
        is_subagent=is_subagent,
    )
    messages, _ = runtime.execute_tool_uses(
        [SimpleNamespace(type="tool_use", name="faketool", input={"path": "x"}, id="tid")]
    )
    return calls, messages[0]


def test_approve_allow():
    tools.set_approve_cb(lambda name, inp: True)
    try:
        calls, message = _run_fake_tool()
        assert message["content"] == "executed"
        assert calls == [{"path": "x"}]
    finally:
        tools.reset_approve_cb()


def test_approve_deny():
    tools.set_approve_cb(lambda name, inp: False)
    try:
        calls, message = _run_fake_tool()
        assert message["is_error"] is True
        assert "ApprovalDenied" in message["content"]
        assert "拒绝" in message["content"]
        assert calls == []
    finally:
        tools.reset_approve_cb()


def test_approve_none_default_unchanged():
    tools.reset_approve_cb()
    calls, message = _run_fake_tool()
    assert message["content"] == "executed"
    assert calls == [{"path": "x"}]


def test_approve_subagent_skips_callback():
    seen = []
    tools.set_approve_cb(lambda name, inp: seen.append(name) or False)
    try:
        calls, message = _run_fake_tool(is_subagent=True)
        assert message["content"] == "executed"
        assert calls == [{"path": "x"}]
        assert seen == []
    finally:
        tools.reset_approve_cb()


def main():
    tests = [
        test_approve_allow,
        test_approve_deny,
        test_approve_none_default_unchanged,
        test_approve_subagent_skips_callback,
    ]
    for test in tests:
        test()
        print(f"  [OK] {test.__name__}")
    print(f"\n[OK] approval callback regression passed: {len(tests)} cases")


if __name__ == "__main__":
    main()
