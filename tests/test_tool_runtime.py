from types import SimpleNamespace

import pytest

from agent.tools import ToolResult
from agent.runtime.hooks import HookBus, HookResult
from agent.tools.runtime import (
    HookDecision,
    PermissionDecision,
    PermissionEngine,
    ToolExecutionRuntime,
    ToolHookAdapter,
    _tool_input_attrs,
)
from agent.tools import reset_approve_cb, reset_bash_history, set_approve_cb, tool_error_count
from agent.tools.contracts import Tool


def _tool_use(name="echo", inp=None, tid="tid1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp or {"text": "hello"}, id=tid)


def _echo_tool(call, *, validate_input=None):
    return Tool(
        name="echo",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["text"],
        },
        validate_input=validate_input,
        call=call,
    )


def _runtime(
    call,
    *,
    permission_engine=None,
    hook_adapter=None,
    hook_bus=None,
    tool_result_callback=None,
):
    return ToolExecutionRuntime(
        [_echo_tool(call)],
        permission_engine=permission_engine,
        hook_adapter=hook_adapter,
        hook_bus=hook_bus,
        tool_result_callback=tool_result_callback,
    )


def test_execute_tool_uses_ignores_non_tools_and_preserves_order():
    calls = []

    def call(tool_input, context):
        calls.append(dict(tool_input))
        return f"out:{tool_input['text']}"

    runtime = _runtime(call)

    messages, tools_used = runtime.execute_tool_uses(
        [
            {"type": "text", "text": "skip"},
            _tool_use(inp={"text": "first"}, tid="t1"),
            _tool_use(inp={"text": "second"}, tid="t2"),
        ]
    )

    assert tools_used == ["echo", "echo"]
    assert calls == [{"text": "first"}, {"text": "second"}]
    assert [m["tool_use_id"] for m in messages] == ["t1", "t2"]
    assert [m["content"] for m in messages] == ["out:first", "out:second"]
    assert all("is_error" not in m for m in messages)
    assert runtime.last_results[0].permission.behavior == "passthrough"
    assert runtime.last_results[0].permission.source == "default"


def test_tool_result_callback_fires_after_each_tool_with_request_id():
    events = []

    def call(tool_input, context):
        return f"out:{tool_input['text']}"

    runtime = _runtime(
        call,
        tool_result_callback=lambda request, result: events.append(
            (request.id, result.messages[0]["content"])
        ),
    )

    messages, _tools_used = runtime.execute_tool_uses(
        [
            _tool_use(inp={"text": "first"}, tid="t1"),
            _tool_use(inp={"text": "second"}, tid="t2"),
        ]
    )

    assert [m["tool_use_id"] for m in messages] == ["t1", "t2"]
    assert events == [("t1", "out:first"), ("t2", "out:second")]


def test_tool_input_attrs_include_command_summary_for_live_start_line():
    attrs = _tool_input_attrs({"command": "echo hello"})

    assert attrs["tool.display.command"] == "echo hello"
    assert attrs["tool.command_summary"] == "str chars=10"
    assert attrs["tool.command_chars"] == 10
    assert attrs["tool.input_type"] == "object"
    assert attrs["tool.input_fields"] == ("command",)


def test_tool_input_attrs_include_path_display_for_live_start_line():
    attrs = _tool_input_attrs({"path": r"C:\workspace\README.md", "limit": 20})

    assert attrs["tool.display.path"] == r"C:\workspace\README.md"
    assert attrs["tool.input_type"] == "object"
    assert attrs["tool.input_fields"] == ("limit", "path")


def test_unknown_tool_returns_error_without_call():
    calls = []
    runtime = _runtime(lambda inp, ctx: calls.append(dict(inp)) or "unused")

    messages, tools_used = runtime.execute_tool_uses(
        [_tool_use(name="missing", inp={"text": "hello"}, tid="u1")]
    )

    assert tools_used == ["missing"]
    assert calls == []
    assert messages == [
        {
            "type": "tool_result",
            "tool_use_id": "u1",
            "content": "UnknownToolError: unknown tool: missing",
            "is_error": True,
        }
    ]


def test_missing_required_field_returns_schema_error_without_call():
    calls = []
    runtime = _runtime(lambda inp, ctx: calls.append(dict(inp)) or "unused")

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"count": 1})])

    assert calls == []
    assert messages[0]["is_error"] is True
    assert "InputValidationError: missing required field: text" in messages[0]["content"]


def test_type_mismatch_returns_schema_error_without_call():
    calls = []
    runtime = _runtime(lambda inp, ctx: calls.append(dict(inp)) or "unused")

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "ok", "count": "1"})])

    assert calls == []
    assert messages[0]["is_error"] is True
    assert "field count expected integer" in messages[0]["content"]


def test_per_tool_validator_runs_before_permission_and_call():
    calls = []
    permission = _FixedPermission(PermissionDecision("allow", source="test"))
    runtime = ToolExecutionRuntime(
        [
            _echo_tool(
                lambda inp, ctx: calls.append(dict(inp)) or "unused",
                validate_input=lambda inp, ctx: "blocked by validator",
            )
        ],
        permission_engine=permission,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "hello"})])

    assert calls == []
    assert permission.calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "InputValidationError: blocked by validator"


def test_updated_input_runs_per_tool_hardening_validation():
    calls = []
    runtime = ToolExecutionRuntime(
        [
            _echo_tool(
                lambda inp, ctx: calls.append(dict(inp)) or "unused",
                validate_input=lambda inp, ctx: (
                    "text cannot be mutated" if inp["text"] == "after" else None
                ),
            )
        ],
        hook_adapter=_MutatingHook(),
    )

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "before"})])

    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "InputValidationError: text cannot be mutated"


def test_tool_result_call_and_map_result_are_supported():
    contexts = []
    tool = Tool(
        name="echo",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        call=lambda tool_input, context: contexts.append((context.run_id, dict(tool_input)))
        or ToolResult(content=f"core:{tool_input['text']}", metadata={"m": 1}),
        map_result=lambda result: result.content + ":mapped",
    )
    runtime = ToolExecutionRuntime([tool], run_id="run123")

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "hello"})])

    assert contexts == [("run123", {"text": "hello"})]
    assert messages == [
        {"type": "tool_result", "tool_use_id": "tid1", "content": "core:hello:mapped"}
    ]


class _FixedPermission(PermissionEngine):
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def decide(self, spec, tool_input):
        self.calls.append((spec.name, dict(tool_input)))
        return self.decision


def test_permission_deny_returns_error_without_call():
    calls = []
    permission = _FixedPermission(
        PermissionDecision("deny", message="policy blocked", source="test")
    )
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == [("echo", {"text": "hello"})]
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PermissionDenied: policy blocked"


def test_permission_ask_returns_unsupported_error_without_call():
    calls = []
    permission = _FixedPermission(
        PermissionDecision("ask", message="needs user", source="test")
    )
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert calls == []
    assert messages[0]["is_error"] is True
    assert "PermissionAskUnsupported" in messages[0]["content"]
    assert "needs user" in messages[0]["content"]


class _MutatingHook(ToolHookAdapter):
    def pre_tool_use(self, spec, request):
        return HookDecision(
            updated_input={"text": "after"},
            additional_messages=({"type": "text", "text": "metadata only"},),
        )


class _StoppingHook(ToolHookAdapter):
    def pre_tool_use(self, spec, request):
        return HookDecision(
            stop_reason="blocked by hook",
            additional_messages=({"type": "text", "text": "stop metadata"},),
        )


class _RaisingPreHook(ToolHookAdapter):
    def pre_tool_use(self, spec, request):
        raise RuntimeError("pre boom")


class _RaisingPostHook(ToolHookAdapter):
    def post_tool_use(self, spec, request, result):
        raise RuntimeError("post boom")


class _RaisingPermission(PermissionEngine):
    def __init__(self):
        self.calls = []

    def decide(self, spec, tool_input):
        self.calls.append((spec.name, dict(tool_input)))
        raise RuntimeError("permission boom")


def test_pre_hook_updated_input_changes_call_and_metadata_is_not_durable():
    calls = []
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or f"out:{inp['text']}",
        hook_adapter=_MutatingHook(),
    )

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "before"})])

    assert calls == [{"text": "after"}]
    assert messages == [
        {"type": "tool_result", "tool_use_id": "tid1", "content": "out:after"}
    ]
    assert runtime.last_results[0].additional_messages == (
        {"type": "text", "text": "metadata only"},
    )


def test_pre_hook_stop_returns_error_without_permission_or_call():
    calls = []
    permission = _FixedPermission(PermissionDecision("allow", source="test"))
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_adapter=_StoppingHook(),
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == []
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PreToolUseStop: blocked by hook"
    assert runtime.last_results[0].additional_messages == (
        {"type": "text", "text": "stop metadata"},
    )


def test_error_text_marks_tool_result_as_error():
    runtime = _runtime(lambda inp, ctx: "Error: tool failure")

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert messages == [
        {
            "type": "tool_result",
            "tool_use_id": "tid1",
            "content": "Error: tool failure",
            "is_error": True,
        }
    ]


def test_empty_output_uses_runtime_placeholder():
    runtime = _runtime(lambda inp, ctx: "")

    messages, tools_used = runtime.execute_tool_uses([_tool_use(tid="empty1")])

    assert tools_used == ["echo"]
    assert messages == [
        {
            "type": "tool_result",
            "tool_use_id": "empty1",
            "content": "(echo 执行完成，无输出)",
        }
    ]


def test_tool_callable_exception_is_wrapped_without_escaping():
    def raising_call(inp, ctx):
        raise ValueError("bad input")

    runtime = _runtime(raising_call)

    messages, tools_used = runtime.execute_tool_uses([_tool_use(tid="boom1")])

    assert tools_used == ["echo"]
    assert messages[0]["type"] == "tool_result"
    assert messages[0]["tool_use_id"] == "boom1"
    assert messages[0]["content"] == "ToolExecutionError: ValueError: bad input"
    assert messages[0]["is_error"] is True
    assert runtime.last_results[0].is_error is True


def test_runtime_validation_and_permission_errors_do_not_pollute_tool_error_count():
    reset_bash_history()
    permission = _FixedPermission(
        PermissionDecision("deny", message="policy blocked", source="test")
    )

    _runtime(lambda inp, ctx: "unused").execute_tool_uses([_tool_use(inp={"count": 1})])
    assert tool_error_count() == 0

    _runtime(lambda inp, ctx: "unused", permission_engine=permission).execute_tool_uses(
        [_tool_use()]
    )
    assert tool_error_count() == 0

    def raising_call(inp, ctx):
        raise ValueError("bad input")

    _runtime(raising_call).execute_tool_uses([_tool_use()])
    assert tool_error_count() == 1

    ToolExecutionRuntime([_echo_tool(raising_call)], is_subagent=True).execute_tool_uses(
        [_tool_use()]
    )
    assert tool_error_count() == 1


def test_internal_mark_current_error_counts_without_error_text(capture_sink):
    from obs.trace import mark_current_error

    reset_bash_history()

    def marked_call(inp, ctx):
        mark_current_error("rc=2: command failed")
        return "ordinary output"

    runtime = _runtime(marked_call)
    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert messages == [
        {"type": "tool_result", "tool_use_id": "tid1", "content": "ordinary output"}
    ]
    assert runtime.last_results[0].counts_as_tool_error is True
    assert tool_error_count() == 1
    tool_span = capture_sink.events()[-1]
    assert tool_span["name"] == "tool.echo"
    assert tool_span["status"] == "ERROR"
    assert tool_span["status_message"] == "tool_error:marked_error"

    ToolExecutionRuntime([_echo_tool(marked_call)], is_subagent=True).execute_tool_uses(
        [_tool_use()]
    )
    assert tool_error_count() == 1


def test_approval_callback_allow_deny_default_and_subagent_skip():
    reset_approve_cb()
    calls = []
    runtime = _runtime(lambda inp, ctx: calls.append(dict(inp)) or "executed")

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "none"})])
    assert messages[0]["content"] == "executed"
    assert calls == [{"text": "none"}]

    set_approve_cb(lambda name, inp: True)
    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "allow"})])
    assert messages[0]["content"] == "executed"
    assert calls[-1] == {"text": "allow"}

    set_approve_cb(lambda name, inp: False)
    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "deny"})])
    assert messages[0]["is_error"] is True
    assert "ApprovalDenied" in messages[0]["content"]
    assert "拒绝" in messages[0]["content"]
    assert calls[-1] == {"text": "allow"}

    seen = []
    set_approve_cb(lambda name, inp: seen.append(name) or False)
    subagent_runtime = ToolExecutionRuntime([_echo_tool(lambda inp, ctx: "subagent")], is_subagent=True)
    messages, _ = subagent_runtime.execute_tool_uses([_tool_use(inp={"text": "skip"})])
    assert messages[0]["content"] == "subagent"
    assert seen == []

    reset_approve_cb()


def test_large_output_is_persisted_by_runtime(monkeypatch, tmp_path, capture_sink):
    from agent import config
    from agent.tools import result_store

    large_output = "x" * (result_store.PERSIST_THRESHOLD_CHARS + 1)
    runtime = _runtime(lambda inp, ctx: large_output)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path)

    messages, tools_used = runtime.execute_tool_uses([_tool_use(tid="large1")])

    assert tools_used == ["echo"]
    content = messages[0]["content"]
    assert "<persisted-output>" in content
    assert ".tool_results" in content
    persisted_files = list((tmp_path / ".tool_results").glob("echo_*.txt"))
    assert len(persisted_files) == 1
    assert str(persisted_files[0]) in content
    assert persisted_files[0].read_text(encoding="utf-8") == large_output
    assert "is_error" not in messages[0]
    assert runtime.last_results[0].persisted is True
    assert runtime.last_results[0].raw_chars == len(large_output)
    assert runtime.last_results[0].persist_path == str(persisted_files[0])
    span_attrs = capture_sink.events()[-1]["attributes"]
    assert span_attrs["tool.persisted"] is True
    assert span_attrs["tool.raw_chars"] == len(large_output)
    assert span_attrs["tool.persist_path"] == str(persisted_files[0])
    assert span_attrs["tool.fork"] is False


def test_hook_bus_pre_tool_updated_input_messages_and_contexts_are_durable():
    calls = []
    bus = HookBus()

    def pre_hook(inp):
        assert inp.event == "PreToolUse"
        assert inp.tool_name == "echo"
        assert inp.tool_input == {"text": "before"}
        return HookResult(
            updated_input={"text": "after"},
            messages=({"type": "text", "text": "pre message"},),
            additional_contexts=("pre context",),
        )

    bus.register("PreToolUse", pre_hook, matcher="echo")
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or f"out:{inp['text']}",
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "before"})])

    assert calls == [{"text": "after"}]
    assert messages == [
        {"type": "tool_result", "tool_use_id": "tid1", "content": "out:after"},
        {"type": "text", "text": "pre message"},
        {"type": "text", "text": "pre context"},
    ]


def test_hook_bus_pre_tool_blocking_skips_permission_and_call():
    calls = []
    permission = _FixedPermission(PermissionDecision("allow", source="test"))
    bus = HookBus()
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            blocking_error="blocked by hook bus",
            additional_contexts=("block context",),
        ),
    )
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == []
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PreToolUseBlocked: blocked by hook bus"
    assert messages[1] == {"type": "text", "text": "block context"}


def test_hook_bus_permission_deny_skips_call():
    calls = []
    bus = HookBus()
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(permission_behavior="deny", stop_reason="policy hook"),
    )
    runtime = _runtime(lambda inp, ctx: calls.append(dict(inp)) or "unused", hook_bus=bus)

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "HookPermissionDenied: policy hook"
    assert runtime.last_results[0].permission.behavior == "deny"
    assert runtime.last_results[0].permission.source == "hook_bus"


def test_hook_bus_post_tool_messages_and_contexts_follow_tool_result():
    seen = []
    bus = HookBus()

    def post_hook(inp):
        seen.append((inp.event, inp.tool_output, dict(inp.tool_input)))
        return HookResult(
            messages=({"type": "text", "text": "post message"},),
            additional_contexts=("post context",),
        )

    bus.register("PostToolUse", post_hook, matcher="echo")
    runtime = _runtime(lambda inp, ctx: f"out:{inp['text']}", hook_bus=bus)

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "hello"})])

    assert seen == [("PostToolUse", "out:hello", {"text": "hello"})]
    assert messages == [
        {"type": "tool_result", "tool_use_id": "tid1", "content": "out:hello"},
        {"type": "text", "text": "post message"},
        {"type": "text", "text": "post context"},
    ]


def test_hook_bus_post_tool_use_failure_runs_on_tool_exception():
    seen = []
    bus = HookBus()

    def failure_hook(inp):
        seen.append((inp.event, type(inp.error).__name__, str(inp.error)))
        return HookResult(messages=({"type": "text", "text": "failure message"},))

    bus.register("PostToolUseFailure", failure_hook, matcher="echo")

    def call(inp, ctx):
        raise ValueError("boom")

    runtime = _runtime(call, hook_bus=bus)

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert seen == [("PostToolUseFailure", "ValueError", "boom")]
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "ToolExecutionError: ValueError: boom"
    assert messages[1] == {"type": "text", "text": "failure message"}


def test_hook_bus_allow_does_not_bypass_permission_engine_deny():
    calls = []
    permission = _FixedPermission(
        PermissionDecision("deny", message="policy still wins", source="test")
    )
    bus = HookBus()
    bus.register("PreToolUse", lambda inp: HookResult(permission_behavior="allow"))
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == [("echo", {"text": "hello"})]
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PermissionDenied: policy still wins"


def test_adapter_updated_input_reaches_hook_bus_and_final_input_reaches_permission():
    calls = []
    permission = _FixedPermission(PermissionDecision("allow", source="test"))
    bus = HookBus()
    seen_by_hook = []

    def pre_hook(inp):
        seen_by_hook.append(dict(inp.tool_input))
        return HookResult(updated_input={"text": "final"})

    bus.register("PreToolUse", pre_hook)
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or f"out:{inp['text']}",
        permission_engine=permission,
        hook_adapter=_MutatingHook(),
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use(inp={"text": "before"})])

    assert seen_by_hook == [{"text": "after"}]
    assert permission.calls == [("echo", {"text": "final"})]
    assert calls == [{"text": "final"}]
    assert messages[0]["content"] == "out:final"


def test_pre_hook_exception_reraises_without_post_tool_use_failure():
    calls = []
    failure_events = []
    bus = HookBus()
    bus.register("PostToolUseFailure", lambda inp: failure_events.append(inp.event) or HookResult())
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        hook_adapter=_RaisingPreHook(),
        hook_bus=bus,
    )

    with pytest.raises(RuntimeError, match="pre boom"):
        runtime.execute_tool_uses([_tool_use()])

    assert calls == []
    assert failure_events == []


def test_permission_engine_exception_reraises_without_post_tool_use_failure():
    calls = []
    failure_events = []
    permission = _RaisingPermission()
    bus = HookBus()
    bus.register("PostToolUseFailure", lambda inp: failure_events.append(inp.event) or HookResult())
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_bus=bus,
    )

    with pytest.raises(RuntimeError, match="permission boom"):
        runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == [("echo", {"text": "hello"})]
    assert calls == []
    assert failure_events == []


def test_post_hook_exception_reraises_without_post_tool_use_failure():
    calls = []
    failure_events = []
    bus = HookBus()
    bus.register("PostToolUseFailure", lambda inp: failure_events.append(inp.event) or HookResult())
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "ok",
        hook_adapter=_RaisingPostHook(),
        hook_bus=bus,
    )

    with pytest.raises(RuntimeError, match="post boom"):
        runtime.execute_tool_uses([_tool_use()])

    assert calls == [{"text": "hello"}]
    assert failure_events == []


def test_hook_bus_ask_does_not_bypass_permission_engine_deny():
    calls = []
    permission = _FixedPermission(
        PermissionDecision("deny", message="policy denies ask", source="test")
    )
    bus = HookBus()
    bus.register("PreToolUse", lambda inp: HookResult(permission_behavior="ask"))
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == [("echo", {"text": "hello"})]
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PermissionDenied: policy denies ask"


def test_hook_bus_ask_hint_still_uses_permission_engine_ask():
    calls = []
    permission = _FixedPermission(
        PermissionDecision("ask", message="needs user approval", source="test")
    )
    bus = HookBus()
    bus.register("PreToolUse", lambda inp: HookResult(permission_behavior="ask"))
    runtime = _runtime(
        lambda inp, ctx: calls.append(dict(inp)) or "unused",
        permission_engine=permission,
        hook_bus=bus,
    )

    messages, _ = runtime.execute_tool_uses([_tool_use()])

    assert permission.calls == [("echo", {"text": "hello"})]
    assert calls == []
    assert messages[0]["is_error"] is True
    assert "PermissionAskUnsupported" in messages[0]["content"]
    assert "needs user approval" in messages[0]["content"]
