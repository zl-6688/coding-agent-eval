"""loop Session Memory 接入回归（step1 loop 接入）—— opt-in + 触发（离线 mock）。

验：① session_memory=None 时现有行为不变；② 传了 SM 时 should_extract 真能在 loop 里触发 extract。
mock 主 loop 的 llm + ToolPool；mock sm.extract 记录调用（不真跑 fork）。

    python eval/memory_eval/test_loop_sm.py
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import loop
from agent.memory.session_memory import SessionMemory, SessionMemoryConfig
from agent.tools.pool import ToolPool
from agent.tools.contracts import Tool


class _Blk:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


def _script(*resps):
    seq = list(resps)

    def fake(
        messages,
        system="",
        tools=None,
        max_tokens=4096,
        model=None,
        purpose="agent",
        temperature=None,
    ):
        return seq.pop(0) if seq else _Resp([_Blk("text", text="done")], "end_turn")
    return fake


def _tool_use(id="t"):
    return _Resp([_Blk("tool_use", name="bash", input={"command": "ls"}, id=id)], "tool_use")


def _end(t="ok"):
    return _Resp([_Blk("text", text=t)], "end_turn")


def _fake_tool_pool():
    return ToolPool(
        (
            Tool(
                name="bash",
                description="fake bash",
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                call=lambda tool_input, context: "x" * 200,
            ),
        )
    )


def main():
    tmp = Path(tempfile.mkdtemp())
    loop.assemble_tool_pool = lambda context=None: _fake_tool_pool()
    trials = 0

    # 1. session_memory=None → 现有行为不变（不接 SM）
    loop.llm.chat = _script(_tool_use("a"), _tool_use("b"), _end("ok"))
    out = loop.run_task("task", max_turns=5, trace=False)
    assert out == "ok", f"无 SM 应正常返回，实际 {out!r}"
    trials += 1

    # 2. 传 SM（小 cfg）→ should_extract 真能在 loop 里触发 extract
    cfg = SessionMemoryConfig(min_tokens_to_init=10, min_tokens_between_update=5,
                              tool_calls_between_updates=2)
    sm = SessionMemory(tmp / "sm.md", cfg)
    calls = []
    sm.extract = lambda messages, system="": calls.append(len(messages))   # mock：不真跑 fork
    loop.llm.chat = _script(_tool_use("a"), _tool_use("b"), _tool_use("c"), _end("ok"))
    out = loop.run_task("task", max_turns=6, trace=False, session_memory=sm)
    assert out == "ok", f"带 SM 也应正常返回，实际 {out!r}"
    assert len(calls) >= 1, f"SM should_extract 应在 loop 里触发 extract，实际 {len(calls)} 次"
    trials += 1

    print(f"[OK] loop SM 接入回归通过：{trials} 组。")
    print("      opt-in(session_memory=None 不影响现有) / should_extract 真在 loop 里触发 extract。")


if __name__ == "__main__":
    main()
