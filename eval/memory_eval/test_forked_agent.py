"""forked-agent 权限锁回归 —— step 0 的核心保证是「代码管工具隔离」。

离线（mock llm + mock dispatch），验证两层权限锁：
  ① 白名单：越权调禁用工具被拦、dispatch 不执行。
  ② tool_filter：路径越界被拦、dispatch 不执行。
  + written_paths 只记真正执行的 write/edit；max_turns 生效。

    python eval/memory_eval/test_forked_agent.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, llm, tools
from agent.context import compact
import agent.memory.forked_agent as forked_mod
from agent.memory import run_forked_agent
from agent.tools.pool import ToolPool
from agent.tools.contracts import Tool


# ── fake anthropic response ──
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


def _tool_use(name, inp, id="t1"):
    return _Resp([_Blk("tool_use", name=name, input=inp, id=id)], "tool_use")


def _end(text):
    return _Resp([_Blk("text", text=text)], "end_turn")


def _script(*resps):
    """scripted llm.chat：按调用序返回预设 resp。"""
    seq = list(resps)

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent"):
        return seq.pop(0)
    return fake_chat


# ── mock dispatch：记录调用、不真执行 ──
_calls = []


def _fake_tool_pool():
    def call(inp, context):
        _calls.append(("edit_file", dict(inp), context.is_subagent))
        return "ok"

    return ToolPool(
        (
            Tool(
                name="edit_file",
                description="fake edit",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                },
                call=call,
            ),
        )
    )


def main():
    forked_mod.assemble_tool_pool = lambda context=None: _fake_tool_pool()
    # tool_filter：只放行 memory/ 下的路径（模拟"只能写记忆目录"）
    only_mem = lambda name, inp: (
        (True, "") if str(inp.get("path", "")).startswith("memory/")
        else (False, f"路径 {inp.get('path')} 不在记忆目录内")
    )
    trials = 0

    # case A：允许 edit_file + 路径在 memory/ → 执行、记 written_paths、收尾
    _calls.clear()
    llm.chat = _script(_tool_use("edit_file", {"path": "memory/notes.md", "old_text": "a", "new_text": "b"}),
                       _end("done"))
    r = run_forked_agent("更新记忆", [{"role": "user", "content": "ctx"}],
                         allowed_tools={"edit_file"}, tool_filter=only_mem, label="t")
    assert _calls == [("edit_file", {"path": "memory/notes.md", "old_text": "a", "new_text": "b"}, True)], \
        f"A: edit 应被执行且 fork=True（指标隔离），实际 {_calls}"
    assert r.written_paths == ["memory/notes.md"], f"A: written_paths={r.written_paths}"
    assert r.final_text == "done" and r.stopped == "finished", f"A: {r.final_text}/{r.stopped}"
    assert r.input_tokens == 20 and r.output_tokens == 10, f"A: 累计 token {r.input_tokens}/{r.output_tokens}"
    trials += 1

    # case B：越权调 bash（不在白名单）→ 第一层拦截、不执行
    _calls.clear()
    llm.chat = _script(_tool_use("bash", {"command": "ls"}), _end("ok"))
    r = run_forked_agent("x", [{"role": "user", "content": "c"}],
                         allowed_tools={"edit_file"}, tool_filter=only_mem, label="t")
    assert _calls == [], f"B: 越权 bash 不该被执行，实际 {_calls}"
    assert r.written_paths == []
    trials += 1

    # case C：edit_file 但路径越界（tool_filter 拒）→ 第二层拦截、不执行
    _calls.clear()
    llm.chat = _script(_tool_use("edit_file", {"path": "src/main.py", "old_text": "a", "new_text": "b"}),
                       _end("ok"))
    r = run_forked_agent("x", [{"role": "user", "content": "c"}],
                         allowed_tools={"edit_file"}, tool_filter=only_mem, label="t")
    assert _calls == [], f"C: 越界 edit 不该被执行，实际 {_calls}"
    assert r.written_paths == []
    trials += 1

    # case D：max_turns 用尽（一直 tool_use 不收尾）→ stopped=max_turns
    _calls.clear()
    llm.chat = _script(*[_tool_use("edit_file", {"path": "memory/a.md", "old_text": "x", "new_text": "y"})
                         for _ in range(3)])
    r = run_forked_agent("x", [{"role": "user", "content": "c"}],
                         allowed_tools={"edit_file"}, tool_filter=only_mem, max_turns=3, label="t")
    assert r.turns == 3 and r.stopped == "max_turns", f"D: turns={r.turns} stopped={r.stopped}"
    trials += 1

    # case E：max_turns 退出兜底抓最后一条 assistant text（P2-3，auto-memory/dream 复用需要）
    _calls.clear()
    llm.chat = _script(*[_Resp([_Blk("text", text="思考中"),
                                _Blk("tool_use", name="edit_file", input={"path": "memory/a.md"}, id="x")],
                               "tool_use") for _ in range(2)])
    r = run_forked_agent("x", [{"role": "user", "content": "c"}],
                         allowed_tools={"edit_file"}, tool_filter=only_mem, max_turns=2, label="t")
    assert r.stopped == "max_turns" and r.final_text == "思考中", \
        f"E: max_turns 应兜底抓 last assistant text，实际 stopped={r.stopped} text={r.final_text!r}"
    trials += 1

    print(f"[OK] forked-agent 权限锁回归通过：{trials} 个 case。")
    print("      白名单拦越权/tool_filter 拦越界/written_paths/max_turns/fork=True 指标隔离/final_text 兜底。")


def test_forked_agent_permission_locks_and_subagent_runtime(monkeypatch):
    monkeypatch.setattr(forked_mod, "assemble_tool_pool", lambda context=None: _fake_tool_pool())
    monkeypatch.setattr(llm, "chat", llm.chat)

    main()


def test_forked_agent_read_file_does_not_pollute_parent_read_state(monkeypatch, tmp_path):
    """Forked reads must not become parent compact-restore or write-guard state."""

    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    tools.reset_executor()
    tools.reset_file_read_state()
    child_file = tmp_path / "child.txt"
    child_file.write_text("child-only context", encoding="utf-8")
    monkeypatch.setattr(
        llm,
        "chat",
        _script(
            _tool_use("read_file", {"path": "child.txt"}),
            _end("done"),
        ),
    )

    result = run_forked_agent(
        "read child file",
        [{"role": "user", "content": "ctx"}],
        allowed_tools={"read_file"},
        label="t",
    )

    assert result.final_text == "done"
    assert tools.get_file_read_state().records == {}
    assert "child-only context" not in compact._post_compact_file_attachment()


def test_forked_agent_filters_compact_internal_messages_before_llm(monkeypatch):
    """Forked memory agents must send provider-valid messages after compaction."""

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent"):
        captured["messages"] = messages
        return _end("done")

    context_messages = [
        compact.create_compact_boundary_message(
            trigger="auto",
            pre_tokens=167_000,
            user_context="",
            messages_summarized=20,
        ),
        compact.create_compact_summary_message(
            "[Compacted]\n关键上下文",
            source="session_memory_compact",
        ),
        {"role": "user", "content": "继续任务"},
    ]
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_forked_agent(
        "更新记忆",
        context_messages,
        allowed_tools=set(),
        label="session_memory",
    )

    assert result.final_text == "done"
    assert captured["messages"][0] == {
        "role": "user",
        "content": "[Compacted]\n关键上下文",
    }
    assert all("subtype" not in message for message in captured["messages"])
    assert all("isCompactSummary" not in message for message in captured["messages"])


def test_forked_agent_empty_allowed_tools_sends_no_model_schemas(monkeypatch):
    """An explicit empty fork allowlist must not expose default tools to the model."""

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent"):
        captured["tools"] = tools
        return _end("done")

    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_forked_agent(
        "update memory",
        [{"role": "user", "content": "ctx"}],
        allowed_tools=set(),
        label="session_memory",
    )

    assert result.final_text == "done"
    assert captured["tools"] == []


if __name__ == "__main__":
    main()
