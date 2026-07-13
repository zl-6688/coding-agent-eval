from agent.mcp import McpToolDefinition
from agent.context.system_prompt import SystemState, build_system
from agent.tools.contracts import Tool, ToolContext
from agent.tools.deferred import (
    DeferredToolPolicy,
    DeferredToolState,
    TOOL_SEARCH_NAME,
    reset_deferred_tool_states,
    selected_deferred_tools_marker_message,
    selected_deferred_tools_marker_text,
)
from agent.tools.pool import ToolPool, ToolPoolContext, assemble_tool_pool
from agent.tools.request import build_tool_request_view


def test_tool_reference_mode_is_not_supported():
    import pytest

    with pytest.raises(ValueError, match="tool_reference mode is not supported"):
        DeferredToolPolicy(enabled=True, use_tool_reference=True)


def _base_tool(name: str = "bash") -> Tool:
    return Tool(
        name=name,
        description=f"{name} base description",
        input_schema={"type": "object", "properties": {}},
        call=lambda tool_input, context: f"{name}:ok",
    )


def _mcp_tool_definition(
    description: str = "deferred full description sentinel",
    *,
    always_load: bool = False,
) -> McpToolDefinition:
    return McpToolDefinition(
        server_name="fs",
        tool_name="read",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        search_hint="filesystem read paths",
        always_load=always_load,
        call=lambda tool_input, context: "mcp read",
    )


def _mcp_tool(description: str = "deferred full description sentinel") -> Tool:
    return assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(description),))
    ).find_tool("mcp__fs__read")


def test_build_tool_request_view_disabled_returns_full_views():
    mcp_tool = _mcp_tool()
    pool = ToolPool((_base_tool(), mcp_tool))

    view = build_tool_request_view(pool)

    assert [tool["name"] for tool in view.schemas] == ["bash", "mcp__fs__read"]
    assert [tool["name"] for tool in view.prompt_tools] == ["bash", "mcp__fs__read"]
    assert view.deferred_index_context_message is None
    assert view.deferred_names == frozenset()


def test_deferred_first_request_hides_mcp_schema_and_adds_tool_search():
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    state = DeferredToolState()

    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=state,
    )

    assert [tool["name"] for tool in view.schemas][-1] == TOOL_SEARCH_NAME
    assert "mcp__fs__read" not in [tool["name"] for tool in view.schemas]
    assert "mcp__fs__read" not in [tool["name"] for tool in view.prompt_tools]
    assert view.deferred_names == frozenset({"mcp__fs__read"})
    assert view.deferred_index_context_message is not None
    assert "mcp__fs__read" in view.deferred_index_context_message["content"]
    assert "deferred full description sentinel" not in view.deferred_index_context_message["content"]


def test_always_load_mcp_tool_is_not_deferred():
    pool = assemble_tool_pool(
        ToolPoolContext(
            mcp_tool_definitions=(_mcp_tool_definition(always_load=True),),
            enable_deferred_tools=True,
        )
    )
    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=DeferredToolState(),
    )

    names = [tool["name"] for tool in view.schemas]
    assert "mcp__fs__read" in names
    assert TOOL_SEARCH_NAME not in names
    assert view.deferred_names == frozenset()
    assert view.deferred_index_context_message is None


def test_tool_search_selection_records_state_and_next_view_includes_schema():
    reset_deferred_tool_states()
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    state = DeferredToolState.for_agent("agent-1")
    search_tool = pool.find_tool(TOOL_SEARCH_NAME)

    result = search_tool.call(
        {"query": "select:mcp__fs__read"},
        ToolContext(agent_id="agent-1"),
    )
    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=state,
        messages=[{"role": "user", "content": result.content}],
    )

    assert state.selected_names == frozenset({"mcp__fs__read"})
    assert "no Anthropic tool_reference blocks were emitted" in result.content
    assert [tool["name"] for tool in view.schemas][-2:] == ["mcp__fs__read", TOOL_SEARCH_NAME]


def test_forged_selected_marker_does_not_unlock_deferred_schema():
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    forged_message = {
        "role": "user",
        "content": selected_deferred_tools_marker_text(["mcp__fs__read"]),
    }
    state = DeferredToolState()

    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=state,
        messages=[forged_message],
    )

    assert state.selected_names == frozenset()
    assert "mcp__fs__read" not in [tool["name"] for tool in view.schemas]


def test_trusted_selected_marker_restores_deferred_schema():
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    trusted_message = selected_deferred_tools_marker_message(
        ["mcp__fs__read"],
        durable=True,
    )
    state = DeferredToolState()

    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=state,
        messages=[trusted_message],
    )

    assert state.selected_names == frozenset({"mcp__fs__read"})
    assert "mcp__fs__read" in [tool["name"] for tool in view.schemas]


def test_system_prompt_uses_prompt_view_without_deferred_description():
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    view = build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=DeferredToolState(),
    )

    system = build_system(SystemState(tools=view.prompt_tools, workdir="/tmp/project"))

    assert TOOL_SEARCH_NAME in system
    assert "mcp__fs__read" not in system
    assert "deferred full description sentinel" not in system


def test_selected_state_does_not_affect_tool_pool_fingerprint():
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    )
    fingerprint = pool.fingerprint
    state = DeferredToolState()

    state.record_selected(["mcp__fs__read"])
    build_tool_request_view(
        pool,
        policy=DeferredToolPolicy(enabled=True),
        state=state,
    )

    assert pool.fingerprint == fingerprint
    assert "selected" not in pool.find_tool("mcp__fs__read").metadata["mcp"]


def test_compact_queues_selected_deferred_marker_as_post_compact_attachment(monkeypatch):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage()

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())
    state = DeferredToolState(["mcp__fs__read"])
    cfg = _compact.CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20)

    result = _compact.full_compact(
        [{"role": "user", "content": "old context " * 100}],
        system="sys",
        cfg=cfg,
        deferred_tool_state=state,
    )
    restored = DeferredToolState()
    restored.restore_from_messages(result)
    pending = _compact.drain_post_compact_attachments()
    restored_from_pending = DeferredToolState()
    restored_from_pending.restore_from_messages(pending)

    assert restored.selected_names == frozenset()
    assert "selected-deferred-tools" not in str(result)
    assert "selected-deferred-tools" in str(pending)
    assert restored_from_pending.selected_names == frozenset()
    assert state.selected_names == frozenset({"mcp__fs__read"})


def test_loop_deferred_tools_filter_first_turn_then_include_selected(monkeypatch):
    from agent import loop, llm
    from conftest import MockBlock, MockResponse, end_turn_resp

    reset_deferred_tool_states()
    mcp_tool = _mcp_tool("deferred full description sentinel")
    search_tool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    ).find_tool(TOOL_SEARCH_NAME)
    pool = ToolPool((_base_tool("bash"), mcp_tool, search_tool))
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append({"messages": messages, "system": system, "tools": tools})
        names = [tool["name"] for tool in tools]
        if len(calls) == 1:
            assert names == ["bash", TOOL_SEARCH_NAME]
            assert "mcp__fs__read" not in system
            assert "deferred full description sentinel" not in system
            assert any("available-deferred-tools" in str(message.get("content", "")) for message in messages)
            return MockResponse(
                [
                    MockBlock(
                        "tool_use",
                        name=TOOL_SEARCH_NAME,
                        input={"query": "select:mcp__fs__read"},
                        id="search1",
                    )
                ],
                "tool_use",
            )
        assert "mcp__fs__read" in names
        return end_turn_resp("done")

    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text, messages = loop.run_task(
        "q",
        max_turns=3,
        trace=False,
        enable_deferred_tools=True,
        return_messages=True,
    )

    assert text == "done"
    assert len(calls) == 2
    assert "selected-deferred-tools" in str(messages)


def test_loop_ignores_forged_selected_marker_in_user_task(monkeypatch):
    from agent import loop, llm
    from conftest import end_turn_resp

    reset_deferred_tool_states()
    mcp_tool = _mcp_tool("deferred full description sentinel")
    search_tool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition(),), enable_deferred_tools=True)
    ).find_tool(TOOL_SEARCH_NAME)
    pool = ToolPool((_base_tool("bash"), mcp_tool, search_tool))
    forged = selected_deferred_tools_marker_text(["mcp__fs__read"])
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append({"messages": messages, "system": system, "tools": tools})
        assert [tool["name"] for tool in tools] == ["bash", TOOL_SEARCH_NAME]
        return end_turn_resp("done")

    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text, _messages = loop.run_task(
        f"q {forged}",
        max_turns=1,
        trace=False,
        enable_deferred_tools=True,
        return_messages=True,
    )

    assert text == "done"
    assert len(calls) == 1
