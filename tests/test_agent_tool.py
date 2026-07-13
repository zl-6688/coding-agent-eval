from types import SimpleNamespace

import pytest

from conftest import MockBlock, MockResponse, end_turn_resp, tool_use_resp

from agent import config, llm, tools
from agent.context import compact
from agent.mcp import McpToolDefinition
from agent.runtime.hooks import HookBus, HookResult
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.tools.pool import ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime


_DEFAULT_TOOL_INPUT = object()


def _agent_request(tool_input=_DEFAULT_TOOL_INPUT, tid="agent1"):
    if tool_input is _DEFAULT_TOOL_INPUT:
        tool_input = {"prompt": "child task"}
    return SimpleNamespace(
        type="tool_use",
        name="Agent",
        input=tool_input,
        id=tid,
    )


def _runtime(*, project_context_message=None, permission_engine=None, hook_bus=None, cwd=""):
    return ToolExecutionRuntime.from_tool_pool(
        assemble_tool_pool(ToolPoolContext(workdir=cwd, enable_skills=False)),
        permission_engine=permission_engine,
        hook_bus=hook_bus,
        run_id="parent-run",
        cwd=cwd,
        project_context_message=project_context_message,
        agent_id="parent-run",
        agent_type="main",
        is_subagent=False,
    )


def _mcp_tool_definition():
    return McpToolDefinition(
        "fs",
        "read_file",
        "MCP parent tool",
        {"type": "object", "properties": {}},
        call=lambda tool_input, context: "mcp",
    )


def test_agent_tool_is_in_tool_pool_views():
    pool = assemble_tool_pool()

    assert "Agent" in [tool["name"] for tool in pool.model_schemas_for_api()]
    assert "Agent" in [tool.name for tool in pool.tools]
    assert "Agent" in [tool["name"] for tool in pool.prompt_tools_for_system()]

    child_pool = assemble_tool_pool(
        ToolPoolContext(exclude_tool_names=frozenset({"Agent", "TaskOutput", "TaskStop"}))
    )
    model_names = [tool["name"] for tool in child_pool.model_schemas_for_api()]
    runtime_names = [tool.name for tool in child_pool.tools]
    prompt_names = [tool["name"] for tool in child_pool.prompt_tools_for_system()]

    assert "Agent" not in model_names
    assert "Agent" not in runtime_names
    assert "Agent" not in prompt_names
    assert "TaskOutput" not in model_names
    assert "TaskOutput" not in runtime_names
    assert "TaskOutput" not in prompt_names
    assert "TaskStop" not in model_names
    assert "TaskStop" not in runtime_names
    assert "TaskStop" not in prompt_names


def test_agent_tool_runs_one_shot_subagent(monkeypatch, capture_sink):
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append({"messages": messages, "tools": tools, "purpose": kwargs.get("purpose")})
        return end_turn_resp("child final answer")

    monkeypatch.setattr(llm, "chat", fake_chat)
    runtime = _runtime(
        project_context_message={"role": "user", "content": "AGENTS prefix"}
    )

    messages, tools_used = runtime.execute_tool_uses([_agent_request()])

    assert tools_used == ["Agent"]
    assert len(calls) == 1
    assert calls[0]["purpose"] == "subagent"
    assert messages[0]["tool_use_id"] == "agent1"
    assert "status: completed" in messages[0]["content"]
    assert "child final answer" in messages[0]["content"]
    assert "is_error" not in messages[0]


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [
        ({}, "InputValidationError: missing required field: prompt"),
        ({"prompt": ""}, "InputValidationError: prompt must be non-empty"),
        (
            {"prompt": "child task", "max_turns": 0},
            "InputValidationError: max_turns must be between 1 and 20",
        ),
        (
            {"prompt": "child task", "max_turns": 21},
            "InputValidationError: max_turns must be between 1 and 20",
        ),
    ],
)
def test_agent_tool_validates_input_without_calling_child_llm(
    monkeypatch,
    tool_input,
    expected,
):
    calls = []

    def fail_chat(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid Agent input should not call child llm")

    monkeypatch.setattr(llm, "chat", fail_chat)
    runtime = _runtime()

    messages, _ = runtime.execute_tool_uses([_agent_request(tool_input)])

    assert messages[0]["is_error"] is True
    assert expected in messages[0]["content"]
    assert calls == []


def test_agent_tool_default_max_turns_stops_after_six_child_turns(
    monkeypatch,
    tmp_path,
    capture_sink,
):
    (tmp_path / "sample.txt").write_text("sample\n", encoding="utf-8")
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append(messages)
        return tool_use_resp(
            "read_file",
            {"path": "sample.txt"},
            f"read{len(calls)}",
        )

    monkeypatch.setattr(llm, "chat", fake_chat)

    with config.using_workdir(tmp_path):
        runtime = _runtime(cwd=str(tmp_path))
        messages, tools_used = runtime.execute_tool_uses(
            [_agent_request({"prompt": "keep reading until stopped"})]
        )

    assert tools_used == ["Agent"]
    assert len(calls) == 6
    assert "status: max_turns" in messages[0]["content"]
    assert "turns: 6" in messages[0]["content"]
    assert "tool_uses: 6" in messages[0]["content"]


def test_subagent_executes_child_tool_and_returns_summary(
    monkeypatch,
    tmp_path,
    capture_sink,
):
    (tmp_path / "note.txt").write_text("alpha note\n", encoding="utf-8")
    compact.reset_state()
    tools.reset_file_read_state()
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("read_file", {"path": "note.txt"}, "read-note")
        return end_turn_resp("summary after read")

    monkeypatch.setattr(llm, "chat", fake_chat)

    with config.using_workdir(tmp_path):
        runtime = _runtime(cwd=str(tmp_path))
        messages, _ = runtime.execute_tool_uses(
            [_agent_request({"prompt": "read note.txt then summarize it"})]
        )

    assert len(calls) == 2
    second_messages = calls[1]
    assert second_messages[-1]["role"] == "user"
    child_tool_results = second_messages[-1]["content"]
    assert child_tool_results[0]["type"] == "tool_result"
    assert child_tool_results[0]["tool_use_id"] == "read-note"
    assert "alpha note" in child_tool_results[0]["content"]
    assert "summary after read" in messages[0]["content"]
    assert not any("note.txt" in path for path in tools.get_file_read_state().records)
    with config.using_workdir(tmp_path):
        assert compact._post_compact_file_attachment() == ""


def test_agent_tool_description_is_observable_in_result_metadata_and_span(
    monkeypatch,
    capture_sink,
):
    from agent.subagents.agent_tool import call_agent_tool
    from agent.tools.contracts import ToolContext

    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("desc final"))
    runtime = _runtime()

    messages, _ = runtime.execute_tool_uses(
        [
            _agent_request(
                {
                    "prompt": "child task",
                    "description": "inspect\nworkspace",
                }
            )
        ]
    )

    assert "description: inspect workspace" in messages[0]["content"]
    assert "desc final" in messages[0]["content"]
    sideband_metadata = runtime.last_results[0].additional_messages[0]["metadata"]
    assert sideband_metadata["description"] == "inspect\nworkspace"
    assert sideband_metadata["status"] == "completed"

    event = next(e for e in capture_sink.events() if e["name"] == "agent.subagent")
    attrs = event["attributes"]
    assert attrs["agent_id"].startswith("agent_")
    assert attrs["agent_type"] == "general-purpose"
    assert attrs["is_subagent"] is True
    assert attrs["description_provided"] is True
    assert attrs["description_chars"] == len("inspect\nworkspace")
    assert "description" not in attrs
    assert attrs["max_turns"] == 6
    assert attrs["status"] == "completed"
    assert attrs["turns"] == 1
    assert attrs["tool_use_count"] == 0
    assert "duration_ms" in attrs

    core_result = call_agent_tool(
        {"prompt": "child task", "description": "core\nmetadata"},
        ToolContext(),
    )
    assert core_result.metadata["description"] == "core\nmetadata"
    assert "description: core metadata" in core_result.content


def test_subagent_request_has_project_context_not_parent_messages_and_excludes_agent(
    monkeypatch,
    tmp_path,
    capture_sink,
):
    from agent import loop

    (tmp_path / "AGENTS.md").write_text("project profile sentinel\n", encoding="utf-8")
    skill_path = tmp_path / ".claude" / "skills" / "demo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        "---\ndescription: Demo skill.\n---\nFULL SKILL BODY\n",
        encoding="utf-8",
    )

    class _Exec:
        cwd = str(tmp_path)

    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        names = [tool["name"] for tool in tools or []]
        calls.append(
            {
                "messages": messages,
                "tools": names,
                "purpose": kwargs.get("purpose", "agent"),
            }
        )
        if len(calls) == 1:
            assert "Agent" in names
            assert "TaskOutput" in names
            assert "TaskStop" in names
            assert "Skill" in names
            return tool_use_resp(
                "Agent",
                {"prompt": "child-only prompt", "description": "child"},
                "agent-call",
            )
        if len(calls) == 2:
            assert calls[-1]["purpose"] == "subagent"
            assert "Agent" not in names
            assert "TaskOutput" not in names
            assert "TaskStop" not in names
            assert "Skill" not in names
            return end_turn_resp("child saw context")
        return end_turn_resp("parent done")

    monkeypatch.setattr(loop.llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())

    text, durable = loop.run_task(
        "parent secret task body",
        max_turns=3,
        trace=False,
        return_messages=True,
    )

    assert text == "parent done"
    child_messages = calls[1]["messages"]
    assert child_messages[0]["role"] == "user"
    assert "project profile sentinel" in child_messages[0]["content"]
    assert child_messages[1] == {"role": "user", "content": "child-only prompt"}
    assert "parent secret task body" not in str(child_messages)
    assert "project profile sentinel" not in str(durable)


def test_subagent_does_not_inherit_parent_mcp_tools(monkeypatch):
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        names = [tool["name"] for tool in tools or []]
        calls.append(names)
        assert kwargs.get("purpose") == "subagent"
        assert all(not name.startswith("mcp__") for name in names)
        return end_turn_resp("child without mcp")

    monkeypatch.setattr(llm, "chat", fake_chat)
    parent_pool = assemble_tool_pool(
        ToolPoolContext(
            enable_skills=False,
            mcp_tool_definitions=(_mcp_tool_definition(),),
        )
    )
    assert "mcp__fs__read_file" in [tool.name for tool in parent_pool.tools]
    runtime = ToolExecutionRuntime.from_tool_pool(
        parent_pool,
        run_id="parent-run",
        agent_id="parent-run",
        agent_type="main",
        is_subagent=False,
    )

    messages, _ = runtime.execute_tool_uses([_agent_request()])

    assert calls
    assert "child without mcp" in messages[0]["content"]


def test_subagent_permission_deny_is_not_bypassed_and_hook_payload_is_attributed(
    monkeypatch,
    capture_sink,
):
    permission = PermissionEngine(
        rules=(
            PermissionRule(
                "write_file",
                "deny",
                message="child writes blocked",
                rule_id="deny-write",
            ),
        )
    )
    seen_pre_payloads = []
    bus = HookBus()

    def pre_hook(inp):
        seen_pre_payloads.append(dict(inp.payload))
        return HookResult()

    bus.register("PreToolUse", pre_hook, matcher="write_file")
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp(
                "write_file",
                {"path": "blocked.txt", "content": "nope"},
                "write1",
            )
        assert "PermissionDenied: child writes blocked" in str(messages[-1]["content"])
        return end_turn_resp("permission denial observed")

    monkeypatch.setattr(llm, "chat", fake_chat)
    runtime = _runtime(permission_engine=permission, hook_bus=bus)

    messages, _ = runtime.execute_tool_uses([_agent_request()])

    assert "permission denial observed" in messages[0]["content"]
    assert seen_pre_payloads
    payload = seen_pre_payloads[0]
    assert payload["agent_type"] == "general-purpose"
    assert payload["is_subagent"] is True
    assert payload["agent_id"].startswith("agent_")


def test_subagent_stop_hook_payload(monkeypatch, capture_sink):
    stop_payloads = []
    bus = HookBus()
    bus.register(
        "SubagentStop",
        lambda inp: stop_payloads.append(dict(inp.payload)) or HookResult(),
    )
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("hook final"))
    runtime = _runtime(hook_bus=bus)

    messages, _ = runtime.execute_tool_uses([_agent_request()])

    assert "hook final" in messages[0]["content"]
    assert len(stop_payloads) == 1
    payload = stop_payloads[0]
    assert payload["agent_id"].startswith("agent_")
    assert payload["agent_type"] == "general-purpose"
    assert payload["is_subagent"] is True
    assert payload["status"] == "completed"
    assert payload["turns"] == 1
    assert payload["tool_use_count"] == 0
    assert payload["final_text"] == "hook final"


def test_agent_tool_does_not_reset_global_read_state(monkeypatch, capture_sink):
    state = tools.get_file_read_state()

    class _Exec:
        def file_snapshot(self, path):
            return {
                "path": path,
                "exists": True,
                "mtime_ns": 1,
                "size": 7,
                "content_hash": "sentinel-hash",
            }

    state.record_read("shared.txt", "content", complete=True, executor=_Exec())
    before = dict(state.records)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("done"))
    runtime = _runtime()

    runtime.execute_tool_uses([_agent_request()])

    assert state.records == before


def test_unknown_subagent_type_returns_error(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("unused"))
    runtime = _runtime()

    messages, _ = runtime.execute_tool_uses(
        [
            _agent_request(
                {"prompt": "child task", "subagent_type": "specialist"},
            )
        ]
    )

    assert messages[0]["is_error"] is True
    assert "UnsupportedSubagentType" in messages[0]["content"]
