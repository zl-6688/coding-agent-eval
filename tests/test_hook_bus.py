import pytest

from agent.runtime.hooks import HOOK_EVENTS, HookBus, HookInput, HookResult, NoOpHookBus


def test_phase1_hook_events_are_exact_subset():
    assert HOOK_EVENTS == (
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
        "StopFailure",
        "SubagentStop",
    )


def test_register_and_run_aggregate_effects_in_order():
    bus = HookBus()

    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            messages=({"type": "text", "text": "first"},),
            additional_contexts=("ctx1",),
            updated_input={"text": "first"},
            permission_behavior="allow",
            blocking_error="first block",
            stop_reason="first stop",
        ),
    )
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            messages=({"type": "text", "text": "second"},),
            additional_contexts=("ctx2",),
            updated_input={"text": "second"},
            permission_behavior="deny",
            blocking_error="second block",
            prevent_continuation=True,
            stop_reason="second stop",
        ),
    )

    result = bus.run(HookInput(event="PreToolUse", tool_name="echo"))

    assert [m["text"] for m in result.messages] == ["first", "second"]
    assert result.additional_contexts == ("ctx1", "ctx2")
    assert result.updated_input == {"text": "second"}
    assert result.permission_behavior == "deny"
    assert result.blocking_error == "first block"
    assert result.prevent_continuation is True
    assert result.stop_reason == "first stop"


def test_matcher_supports_exact_glob_and_callable():
    bus = HookBus()
    matched = []

    bus.register("PreToolUse", lambda inp: matched.append("exact") or HookResult(), matcher="bash")
    bus.register("PreToolUse", lambda inp: matched.append("glob") or HookResult(), matcher="read_*")
    bus.register(
        "PreToolUse",
        lambda inp: matched.append("callable") or HookResult(),
        matcher=lambda inp: inp.tool_input["path"].endswith(".md"),
    )

    bus.run(HookInput(event="PreToolUse", tool_name="read_file", tool_input={"path": "README.md"}))

    assert matched == ["glob", "callable"]


def test_handler_errors_and_invalid_results_are_non_blocking():
    bus = HookBus()

    def raising(inp):
        raise RuntimeError("bad hook")

    bus.register("Stop", raising)
    bus.register("Stop", lambda inp: {"not": "a HookResult"})
    bus.register("Stop", lambda inp: HookResult(messages=({"type": "text", "text": "ok"},)))

    result = bus.run(HookInput(event="Stop"))

    assert result.messages == ({"type": "text", "text": "ok"},)
    assert len(result.errors) == 2
    assert "RuntimeError: bad hook" in result.errors[0]
    assert "TypeError: hook handler must return HookResult" in result.errors[1]


def test_unknown_event_raises_for_register_and_run():
    bus = HookBus()

    with pytest.raises(ValueError):
        bus.register("Unknown", lambda inp: HookResult())
    with pytest.raises(ValueError):
        bus.run(HookInput(event="Unknown"))


def test_noop_hook_bus_returns_empty_result_and_validates_events():
    bus = NoOpHookBus()

    assert bus.has_handlers("SubagentStop") is False
    assert bus.run(HookInput(event="SubagentStop")) == HookResult()
    with pytest.raises(ValueError):
        bus.run(HookInput(event="Unknown"))



def test_invalid_permission_behavior_records_error_without_partial_effects():
    bus = HookBus()
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            messages=({"type": "text", "text": "bad"},),
            additional_contexts=("bad context",),
            updated_input={"text": "bad"},
            permission_behavior="invalid",
            blocking_error="bad block",
            prevent_continuation=True,
            stop_reason="bad stop",
        ),
    )
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            messages=({"type": "text", "text": "ok"},),
            updated_input={"text": "ok"},
        ),
    )

    result = bus.run(HookInput(event="PreToolUse", tool_name="echo"))

    assert result.messages == ({"type": "text", "text": "ok"},)
    assert result.additional_contexts == ()
    assert result.updated_input == {"text": "ok"}
    assert result.permission_behavior == "passthrough"
    assert result.blocking_error is None
    assert result.prevent_continuation is False
    assert result.stop_reason is None
    assert len(result.errors) == 1
    assert "TypeError: unsupported permission behavior: invalid" in result.errors[0]
