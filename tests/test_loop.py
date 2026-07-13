"""test_loop.py — characterization tests for agent.loop.run_task (full turn cycle).

FROZEN BEHAVIOR ORACLE: these assertions capture the observable behavior of run_task
as it exists today (pre-EvalHooks refactor).  When the refactor changes the signature
(compact_strategy/etc. → eval_hooks=EvalHooks(...)), only the *call lines* migrate;
the assertions stay frozen to prove "behavior unchanged."
"""
import pytest
from conftest import (CaptureSink, MockBlock, MockResponse,
                       end_turn_resp, script, tool_use_resp)
from agent.tools.pool import ToolPool
from agent.tools.contracts import Tool


# ── helpers ────────────────────────────────────────────────────────────────

def _fake_tool_pool(calls=None, output_fn=None, names=("bash", "read_file")):
    def make_tool(name):
        def call(inp, context):
            if calls is not None:
                calls.append((name, dict(inp), False))
            if output_fn is not None:
                return output_fn(name, inp)
            return f"output_from_{name}"

        return Tool(
            name=name,
            description=f"{name} description",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            call=call,
        )

    return ToolPool(tuple(make_tool(name) for name in names))


def _disable_skill_listing_context(monkeypatch, loop_module):
    """Keep request-view order tests focused on AGENTS/durable layering."""
    monkeypatch.setattr(loop_module, "skill_listing_context_message", lambda catalog: None)


def _span_names(sink: CaptureSink) -> list[str]:
    return [e["name"] for e in sink.events()]


def _spans_named(sink: CaptureSink, name: str) -> list[dict]:
    return [e for e in sink.events() if e["name"] == name]


def _api_shape(messages: list[dict]) -> list[dict]:
    """Project durable messages onto the provider-visible role/content shape."""

    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


# ── tests ─────────────────────────────────────────────────────────────────

def test_run_task_uses_same_tool_pool_snapshot_for_prompt_chat_runtime(monkeypatch):
    from agent import loop, llm

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    tool_calls = []
    pool = ToolPool(
        (
            Tool(
                name="snapshot_tool",
                description="snapshot description",
                input_schema=schema,
                call=lambda inp, ctx: tool_calls.append(("snapshot_tool", dict(inp), False))
                or "snapshot tool output",
            ),
        )
    )
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    chat_calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        chat_calls.append({"system": system, "tools": tools, "max_tokens": max_tokens})
        assert [tool["name"] for tool in tools] == ["snapshot_tool"]
        if len(chat_calls) == 1:
            return MockResponse([MockBlock("text", text="retry")], "max_tokens")
        if len(chat_calls) == 2:
            return tool_use_resp("snapshot_tool", {"text": "run"}, "snap1")
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)

    text, _messages = loop.run_task(
        "q",
        max_turns=3,
        trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
        return_messages=True,
    )

    assert text == "done"
    assert [call["max_tokens"] for call in chat_calls] == [4096, 8192, 4096]
    assert all("snapshot description" in call["system"] for call in chat_calls)
    assert all([tool["name"] for tool in call["tools"]] == ["snapshot_tool"] for call in chat_calls)
    assert tool_calls == [("snapshot_tool", {"text": "run"}, False)]


def test_run_task_eval_hooks_can_override_identity(monkeypatch):
    from agent import loop, llm

    seen = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        seen["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text = loop.run_task(
        "q",
        max_turns=1,
        trace=False,
        eval_hooks=loop.EvalHooks(identity="CUSTOM IDENTITY FOR EVAL"),
    )

    assert text == "done"
    assert "CUSTOM IDENTITY FOR EVAL" in seen["system"]
    assert "coding agent" not in seen["system"].split("## 可用工具", 1)[0]


def test_run_task_eval_hooks_can_disable_skills(monkeypatch, capture_sink):
    from agent import loop, llm

    seen = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        seen["messages"] = messages
        seen["tools"] = tools
        return end_turn_resp("done")

    def fail_discovery(workdir):
        raise AssertionError("skill discovery must be skipped when disabled")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "discover_skill_catalog", fail_discovery)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text = loop.run_task(
        "q",
        max_turns=1,
        trace=False,
        eval_hooks=loop.EvalHooks(skills_enabled=False),
    )

    assert text == "done"
    assert "# skill_listing" not in str(seen["messages"])
    assert "Skill" not in [tool["name"] for tool in seen["tools"]]
    run_attrs = _spans_named(capture_sink, "agent.run")[0]["attributes"]
    assert run_attrs["skills_enabled"] is False
    assert run_attrs["skill_count"] == 0


def test_run_task_single_tool_then_end(monkeypatch, capture_sink):
    """Full oracle: LLM → tool_use → LLM → end_turn.

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    """
    from agent import loop, llm

    monkeypatch.setattr(llm, "chat", script(
        tool_use_resp("bash", {"command": "echo hi"}, "tid1"),
        end_turn_resp("task complete"),
    ))
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: _fake_tool_pool())

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    text, messages = loop.run_task(
        "write hello.py", max_turns=5, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"), return_messages=True,
    )

    # ── return value ─────────────────────────────────────────────────────
    assert text == "task complete"

    # ── message sequence ─────────────────────────────────────────────────
    # Shape: user(task) / assistant(tool_use) / user(tool_result) / assistant(end_turn)
    assert len(messages) == 4, f"expected 4 messages, got {len(messages)}"
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "write hello.py"

    assert messages[1]["role"] == "assistant"
    # content is the raw list of MockBlock objects appended by loop (not serialized)
    content1 = messages[1]["content"]
    assert any(getattr(b, "type", None) == "tool_use" for b in content1)

    assert messages[2]["role"] == "user"
    content2 = messages[2]["content"]
    assert isinstance(content2, list) and len(content2) == 1
    assert content2[0]["type"] == "tool_result"
    assert content2[0]["tool_use_id"] == "tid1"
    assert "output_from_bash" in content2[0]["content"]
    assert "is_error" not in content2[0]

    assert messages[3]["role"] == "assistant"

    # ── spans ─────────────────────────────────────────────────────────────
    names = _span_names(capture_sink)
    assert "agent.run" in names, f"missing agent.run span; got {names}"
    assert "agent.turn" in names, f"missing agent.turn span; got {names}"
    assert "tool.bash" in names, f"missing tool.bash span; got {names}"

    turn_spans = _spans_named(capture_sink, "agent.turn")
    assert len(turn_spans) == 2, f"expected 2 agent.turn spans, got {len(turn_spans)}"
    assert turn_spans[0]["attributes"]["turn_index"] == 1
    assert turn_spans[0]["attributes"]["tools_used"] == ["bash"]
    assert turn_spans[0]["attributes"]["n_tool_calls"] == 1
    assert turn_spans[1]["attributes"]["turn_index"] == 2

    run_spans = _spans_named(capture_sink, "agent.run")
    assert len(run_spans) == 1
    run_attrs = run_spans[0]["attributes"]
    assert run_attrs.get("outcome") == "finished"
    assert run_attrs.get("finished") is True
    assert run_attrs.get("turns") == 2


def test_run_task_executes_multiple_tool_uses_in_order(monkeypatch, capture_sink):
    """One assistant tool_use message can contain multiple tool calls."""
    from agent import loop, llm

    first = MockBlock("tool_use", name="bash", input={"command": "echo one"}, id="tid1")
    second = MockBlock("tool_use", name="read_file", input={"path": "README.md"}, id="tid2")
    calls = []

    def output_fn(name, inp):
        key = inp.get("command") or inp.get("path")
        return f"{name}:{key}"

    monkeypatch.setattr(
        llm,
        "chat",
        script(
            MockResponse([first, second], "tool_use"),
            end_turn_resp("multi complete"),
        ),
    )
    monkeypatch.setattr(
        loop,
        "assemble_tool_pool",
        lambda context=None: _fake_tool_pool(calls=calls, output_fn=output_fn),
    )

    text, messages = loop.run_task(
        "run two tools",
        max_turns=3,
        trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
        return_messages=True,
    )

    assert text == "multi complete"
    assert calls == [
        ("bash", {"command": "echo one"}, False),
        ("read_file", {"path": "README.md"}, False),
    ]
    results = messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tid1", "tid2"]
    assert [r["content"] for r in results] == ["bash:echo one", "read_file:README.md"]
    assert all("is_error" not in r for r in results)

    turn_spans = _spans_named(capture_sink, "agent.turn")
    assert turn_spans[0]["attributes"]["tools_used"] == ["bash", "read_file"]
    assert turn_spans[0]["attributes"]["n_tool_calls"] == 2


def test_run_task_tool_error_result_does_not_stop_loop(monkeypatch, capture_sink):
    """Tool error text remains a tool_result and the loop continues."""
    from agent import loop, llm

    chat_calls = []

    def fake_chat(messages, **kwargs):
        chat_calls.append(messages)
        if len(chat_calls) == 1:
            return tool_use_resp("bash", {"command": "fail"}, "err1")
        return end_turn_resp("recovered")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        loop,
        "assemble_tool_pool",
        lambda context=None: _fake_tool_pool(output_fn=lambda name, inp: "Error: tool failure"),
    )

    text, messages = loop.run_task(
        "recover from tool error",
        max_turns=3,
        trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
        return_messages=True,
    )

    assert text == "recovered"
    assert len(chat_calls) == 2
    tool_result = messages[2]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "err1",
        "content": "Error: tool failure",
        "is_error": True,
    }
    assert messages[3]["role"] == "assistant"
    assert any(
        call_message.get("role") == "user" and call_message.get("content") == [tool_result]
        for call_message in chat_calls[1]
    )

    turn_spans = _spans_named(capture_sink, "agent.turn")
    assert turn_spans[0]["attributes"]["tools_used"] == ["bash"]
    assert turn_spans[0]["attributes"]["n_tool_calls"] == 1


def test_run_task_no_tools_end_immediately(monkeypatch, capture_sink):
    """LLM responds end_turn on the very first call — no tool dispatch.

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    """
    from agent import loop, llm

    monkeypatch.setattr(llm, "chat", script(end_turn_resp("immediate answer")))

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    text, messages = loop.run_task(
        "what is 2+2", max_turns=5, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"), return_messages=True,
    )

    assert text == "immediate answer"
    assert len(messages) == 2  # user task + assistant response
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    run_attrs = _spans_named(capture_sink, "agent.run")[0]["attributes"]
    assert run_attrs.get("turns") == 1
    assert run_attrs.get("n_compactions") == 0


def test_run_task_max_turns_reached(monkeypatch, capture_sink):
    """Loop reaches max_turns without end_turn — outcome=max_turns_reached.

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    """
    from agent import loop, llm

    # always tool_use so the loop never terminates naturally
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: tool_use_resp())
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: _fake_tool_pool())

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    text, messages = loop.run_task(
        "infinite task", max_turns=3, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"), return_messages=True,
    )

    assert "(达到最大轮次" in text

    run_attrs = _spans_named(capture_sink, "agent.run")[0]["attributes"]
    assert run_attrs.get("outcome") == "max_turns_reached"
    assert run_attrs.get("finished") is False
    assert run_attrs.get("turns") == 3


def test_run_task_initial_messages_resume(monkeypatch, capture_sink):
    """initial_messages seeds the conversation (session resume seam).

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    """
    from agent import loop, llm

    prior_messages = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": [MockBlock("text", text="partial")]},
    ]
    monkeypatch.setattr(llm, "chat", script(end_turn_resp("resumed")))

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    text, messages = loop.run_task(
        "continue task", max_turns=5, trace=False,
        initial_messages=prior_messages, return_messages=True,
    )

    assert text == "resumed"
    # The run should have started from the prior messages (deep-copied)
    assert messages[0]["content"] == "original task"
    # Prior messages list should be unchanged (loop deep-copies)
    assert len(prior_messages) == 2, "run_task must not mutate the caller's initial_messages"


def test_run_task_return_messages_false(monkeypatch):
    """return_messages=False returns plain str, not a tuple.

    Quirk: return_messages=False is the default; this locks the default return shape.
    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    """
    from agent import loop, llm
    monkeypatch.setattr(llm, "chat", script(end_turn_resp("plain")))

    # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
    result = loop.run_task("q", max_turns=2, trace=False)
    assert isinstance(result, str)
    assert result == "plain"


def test_run_task_loads_agents_md_as_transient_user_context(monkeypatch, tmp_path):
    """run_task 从 executor cwd 加载 AGENTS.md，但只作为临时 request context。"""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("loop integration project profile\n", encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    captured = {}

    def fake_chat(
        messages,
        system="",
        tools=None,
        max_tokens=4096,
        model=None,
        purpose="agent",
        temperature=None,
    ):
        captured["system"] = system
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    _disable_skill_listing_context(monkeypatch, loop)

    result = loop.run_task(
        "q", max_turns=2, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
    )

    assert result == "done"
    assert "loop integration project profile" not in captured["system"]
    assert captured["messages"][0]["role"] == "user"
    assert "<system-reminder>" in captured["messages"][0]["content"]
    assert "# project_instructions" in captured["messages"][0]["content"]
    assert "loop integration project profile" in captured["messages"][0]["content"]
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "q"


def test_run_task_falls_back_to_legacy_agent_md_as_transient_user_context(monkeypatch, tmp_path):
    """run_task loads legacy AGENT.md only as transient request context."""
    from agent import loop, llm, tools

    (tmp_path / "AGENT.md").write_text("legacy loop integration project profile\n", encoding="utf-8")
    assert not (tmp_path / "AGENTS.md").exists()

    class _Exec:
        cwd = str(tmp_path)

    captured = {}

    def fake_chat(
        messages,
        system="",
        tools=None,
        max_tokens=4096,
        model=None,
        purpose="agent",
        temperature=None,
    ):
        captured["system"] = system
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    _disable_skill_listing_context(monkeypatch, loop)

    result = loop.run_task(
        "q", max_turns=2, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
    )

    assert result == "done"
    assert "legacy loop integration project profile" not in captured["system"]
    assert captured["messages"][0]["role"] == "user"
    assert "<system-reminder>" in captured["messages"][0]["content"]
    assert "# project_instructions" in captured["messages"][0]["content"]
    assert "legacy loop integration project profile" in captured["messages"][0]["content"]
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "q"


def test_run_task_does_not_persist_agents_md_context_message(monkeypatch, tmp_path):
    """AGENTS.md request context 不写入返回的 durable messages。"""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("durable pollution sentinel\n", encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    monkeypatch.setattr(llm, "chat", script(end_turn_resp("done")))
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())

    text, messages = loop.run_task(
        "q", max_turns=2, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
        return_messages=True,
    )

    assert text == "done"
    combined = "\n".join(str(m.get("content", "")) for m in messages)
    assert "durable pollution sentinel" not in combined
    assert "system-reminder" not in combined


def test_run_task_context_budget_counts_agents_md_prefix(monkeypatch, tmp_path):
    """stop_at_context 使用包含 AGENTS.md request context 的请求预算。"""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("X" * 1600, encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("llm.chat should not be called after snapshot_cut")

    monkeypatch.setattr(llm, "chat", fail_if_called)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    base_system = loop.build_system(loop.SystemState(workdir=str(tmp_path)))
    base_tokens = loop.compact.estimate([{"role": "user", "content": "q"}], base_system)

    text, messages = loop.run_task(
        "q", max_turns=2, trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none", stop_at_context=base_tokens + 100),
        return_messages=True,
    )

    assert text is None
    assert _api_shape(messages) == [{"role": "user", "content": "q"}]


def test_run_task_return_messages_uses_rebound_compacted_messages(monkeypatch):
    """Compaction may return a new transcript list; return_messages must see it."""
    from agent import loop, llm

    compacted_messages = [{"role": "user", "content": "compacted durable"}]
    compaction_calls = []

    def fake_apply_compaction(messages, *args, **kwargs):
        compaction_calls.append(messages)
        return list(compacted_messages)

    monkeypatch.setattr(loop, "_apply_compaction", fake_apply_compaction)
    monkeypatch.setattr(llm, "chat", script(end_turn_resp("done")))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, messages = loop.run_task(
        "original durable",
        max_turns=2,
        trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="full", compact_threshold=1),
        return_messages=True,
    )

    assert text == "done"
    assert compaction_calls
    assert messages[0] == compacted_messages[0]
    assert "original durable" not in "\n".join(str(m.get("content", "")) for m in messages)


def test_run_task_docker_executor_default_does_not_load_host_agent_md(monkeypatch, tmp_path):
    """Docker executor without host_cwd must not read harness/workspace AGENTS.md."""
    from agent import config, loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("harness workspace profile\n", encoding="utf-8")

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None,
                  purpose="agent", temperature=None):
        captured["system"] = system
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(tools, "get_executor", lambda: tools.DockerExecutor("abcdef1234567890"))
    _disable_skill_listing_context(monkeypatch, loop)

    result = loop.run_task("q", max_turns=2, trace=False)

    assert result == "done"
    assert "/testbed @docker:abcdef123456" in captured["system"]
    assert captured["messages"][0]["content"] == "q"
    assert "harness workspace profile" not in str(captured["messages"])


def test_run_task_loads_agents_md_from_explicit_host_cwd_for_docker_executor(monkeypatch, tmp_path):
    """Docker executor cwd is display-only; explicit host_cwd controls AGENTS.md loading."""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("docker host project profile\n", encoding="utf-8")

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None,
                  purpose="agent", temperature=None):
        captured["system"] = system
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        tools,
        "get_executor",
        lambda: tools.DockerExecutor("abcdef1234567890", host_cwd=tmp_path),
    )
    _disable_skill_listing_context(monkeypatch, loop)

    result = loop.run_task("q", max_turns=2, trace=False)

    assert result == "done"
    assert "/testbed @docker:abcdef123456" in captured["system"]
    assert "docker host project profile" in captured["messages"][0]["content"]
    assert captured["messages"][1]["content"] == "q"


def test_project_context_request_shape_allows_transient_adjacent_user(monkeypatch, tmp_path):
    """The request view prepends meta user context without mutating durable messages."""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("adjacent user sentinel\n", encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    _disable_skill_listing_context(monkeypatch, loop)

    text, durable = loop.run_task("task user", max_turns=2, trace=False, return_messages=True)

    assert text == "done"
    assert [m["role"] for m in captured["messages"][:2]] == ["user", "user"]
    assert "adjacent user sentinel" in captured["messages"][0]["content"]
    assert captured["messages"][1]["content"] == "task user"
    assert "adjacent user sentinel" not in "\n".join(str(m.get("content", "")) for m in durable)


def test_project_context_prepended_to_each_llm_request(monkeypatch, tmp_path):
    """Tool-use continuation requests get the same transient AGENTS.md context."""
    from agent import loop, llm, tools

    (tmp_path / "AGENTS.md").write_text("every request sentinel\n", encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("bash", {"command": "echo hi"}, "tid1")
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: _fake_tool_pool())
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    _disable_skill_listing_context(monkeypatch, loop)

    text, durable = loop.run_task("q", max_turns=3, trace=False, return_messages=True)

    assert text == "done"
    assert len(calls) == 2
    assert all(call[0]["role"] == "user" for call in calls)
    assert "every request sentinel" in calls[0][0]["content"]
    assert calls[0][1]["content"] == "q"
    assert "every request sentinel" in calls[1][0]["content"]
    assert "every request sentinel" not in "\n".join(str(m.get("content", "")) for m in durable)


def test_auto_memory_recall_budget_checked_before_llm(monkeypatch, capture_sink):
    """Recall injection can push the final request over stop_at_context before chat."""
    from pathlib import Path
    from agent import loop, llm
    from agent.runtime.settings import MemoryRuntimeSettings

    class _AM:
        memory_dir = Path("/fake/memory")

        def recall(self, query, already_surfaced, recent_tools=None):
            return [{"path": "/fake/memory/large.md", "content": "R" * 1600}]

        def write(self, messages, system=""):
            raise AssertionError("write should not run after snapshot_cut")

    def fail_chat(*args, **kwargs):
        raise AssertionError("llm.chat should not be called after recall overflows budget")

    monkeypatch.setattr(llm, "chat", fail_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    _disable_skill_listing_context(monkeypatch, loop)
    base_tokens = loop.compact.estimate([{"role": "user", "content": "q"}], "SYSTEM")

    text, messages = loop.run_task(
        "q",
        max_turns=2,
        trace=False,
        auto_memory=_AM(),
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="selector"),
        eval_hooks=loop.EvalHooks(compact_strategy="none", stop_at_context=base_tokens + 100),
        return_messages=True,
    )

    assert text is None
    assert "large.md" in str(messages)
    turn_attrs = _spans_named(capture_sink, "agent.turn")[0]["attributes"]
    assert turn_attrs["outcome"] == "snapshot_cut"
    assert turn_attrs["context_tokens"] > base_tokens + 100


def test_run_task_user_prompt_submit_adds_durable_context(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    captured = {}
    bus = HookBus()
    bus.register("UserPromptSubmit", lambda inp: HookResult(additional_contexts=("hook context",)))

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    _disable_skill_listing_context(monkeypatch, loop)

    text, messages = loop.run_task("q", max_turns=2, trace=False, hook_bus=bus, return_messages=True)

    assert text == "done"
    assert captured["messages"][:2] == [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "hook context"},
    ]
    assert _api_shape(messages[:2]) == captured["messages"][:2]


def test_run_task_user_prompt_submit_blocking_returns_error_without_llm(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    bus = HookBus()
    bus.register(
        "UserPromptSubmit",
        lambda inp: HookResult(blocking_error="prompt rejected", additional_contexts=("audit",)),
    )

    def fail_chat(*args, **kwargs):
        raise AssertionError("llm.chat should not run after UserPromptSubmit blocks")

    monkeypatch.setattr(llm, "chat", fail_chat)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, messages = loop.run_task("q", max_turns=2, trace=False, hook_bus=bus, return_messages=True)

    assert text == "UserPromptSubmitBlocked: prompt rejected"
    assert _api_shape(messages) == [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "audit"},
    ]


def test_run_task_stop_blocking_continues_once(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    calls = []
    stop_calls = []
    bus = HookBus()

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return end_turn_resp("draft")
        return end_turn_resp("final")

    def stop_hook(inp):
        stop_calls.append(inp.payload["outcome"])
        if len(stop_calls) == 1:
            return HookResult(blocking_error="continue once")
        return HookResult(additional_contexts=("stop observed",))

    bus.register("Stop", stop_hook)
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, messages = loop.run_task("q", max_turns=1, trace=False, hook_bus=bus, return_messages=True)

    assert text == "final"
    assert len(calls) == 2
    assert stop_calls == ["finished", "finished"]
    assert {"role": "user", "content": "StopHookBlocked: continue once"} in _api_shape(messages)
    assert _api_shape([messages[-1]]) == [{"role": "user", "content": "stop observed"}]


def test_run_task_stop_failure_for_non_overflow_bad_request_skips_stop(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    class FakeBadRequestError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message

    events = []
    bus = HookBus()
    bus.register("Stop", lambda inp: events.append("Stop") or HookResult())
    bus.register("StopFailure", lambda inp: events.append((inp.event, str(inp.error))) or HookResult())

    monkeypatch.setattr(loop.anthropic, "BadRequestError", FakeBadRequestError)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: (_ for _ in ()).throw(FakeBadRequestError("bad schema")))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    with pytest.raises(FakeBadRequestError):
        loop.run_task("q", max_turns=2, trace=False, hook_bus=bus)

    assert events == [("StopFailure", "bad schema")]




def test_run_task_stop_failure_for_anthropic_api_error_skips_stop(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    class FakeAPIError(Exception):
        pass

    events = []
    bus = HookBus()
    bus.register("Stop", lambda inp: events.append("Stop") or HookResult())
    bus.register("StopFailure", lambda inp: events.append((inp.event, str(inp.error))) or HookResult())

    monkeypatch.setattr(loop.anthropic, "APIError", FakeAPIError, raising=False)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: (_ for _ in ()).throw(FakeAPIError("api down")))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    with pytest.raises(FakeAPIError):
        loop.run_task("q", max_turns=2, trace=False, hook_bus=bus)

    assert events == [("StopFailure", "api down")]


def test_agents_md_remains_transient_with_user_prompt_submit_hook(monkeypatch, tmp_path):
    from agent import loop, llm, tools
    from agent.runtime.hooks import HookBus, HookResult

    (tmp_path / "AGENTS.md").write_text("hook combo project profile\n", encoding="utf-8")

    class _Exec:
        cwd = str(tmp_path)

    captured = {}
    bus = HookBus()
    bus.register("UserPromptSubmit", lambda inp: HookResult(additional_contexts=("hook context",)))

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    _disable_skill_listing_context(monkeypatch, loop)

    text, durable = loop.run_task("q", max_turns=2, trace=False, hook_bus=bus, return_messages=True)

    assert text == "done"
    assert "hook combo project profile" in captured["messages"][0]["content"]
    assert captured["messages"][1] == {"role": "user", "content": "q"}
    assert captured["messages"][2] == {"role": "user", "content": "hook context"}
    combined = "\n".join(str(m.get("content", "")) for m in durable)
    assert "hook context" in combined
    assert "hook combo project profile" not in combined
    assert "system-reminder" not in combined


def test_run_task_context_overflow_bad_request_skips_stop_and_stop_failure(monkeypatch):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    class FakeBadRequestError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message

    events = []
    bus = HookBus()
    bus.register("Stop", lambda inp: events.append("Stop") or HookResult())
    bus.register("StopFailure", lambda inp: events.append("StopFailure") or HookResult())

    monkeypatch.setattr(loop.anthropic, "BadRequestError", FakeBadRequestError)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FakeBadRequestError("context window exceeded")
        ),
    )
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, _messages = loop.run_task("q", max_turns=2, trace=False, hook_bus=bus, return_messages=True)

    assert text == "[上下文超窗 context_overflow]"
    assert events == []



def test_user_prompt_submit_blocking_records_agent_run_span_and_hook_errors(
    monkeypatch,
    capture_sink,
):
    from agent import loop, llm
    from agent.runtime.hooks import HookBus, HookResult

    bus = HookBus()

    def bad_hook(inp):
        raise RuntimeError("hook exploded")

    bus.register("UserPromptSubmit", bad_hook)
    bus.register("UserPromptSubmit", lambda inp: HookResult(blocking_error="prompt rejected"))

    def fail_chat(*args, **kwargs):
        raise AssertionError("llm.chat should not run when prompt hook blocks")

    monkeypatch.setattr(llm, "chat", fail_chat)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, messages = loop.run_task(
        "q",
        max_turns=2,
        trace=False,
        hook_bus=bus,
        return_messages=True,
    )

    assert text == "UserPromptSubmitBlocked: prompt rejected"
    assert _api_shape(messages) == [{"role": "user", "content": "q"}]
    run_spans = _spans_named(capture_sink, "agent.run")
    assert len(run_spans) == 1
    attrs = run_spans[0]["attributes"]
    assert attrs["outcome"] == "user_prompt_blocked"
    assert attrs["finished"] is False
    assert attrs["turns"] == 0
    assert attrs["stop_reason"] == "prompt rejected"
    assert attrs["hook_event"] == "UserPromptSubmit"
    assert attrs["hook.event"] == "UserPromptSubmit"
    assert attrs["hook.error_count"] == 1
    assert attrs["hook.run_id_present"] is True
    assert attrs["hook.tool_name_present"] is False
    assert "hook.errors" not in attrs
    assert "hook exploded" not in repr(attrs)
    assert _spans_named(capture_sink, "agent.turn") == []
