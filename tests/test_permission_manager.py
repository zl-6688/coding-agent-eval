from types import SimpleNamespace

import pytest

from agent.runtime.permissions import (
    PermissionDecision,
    PermissionManager,
    PermissionRule,
)
from agent.tools.runtime import (
    PermissionDecision as RuntimePermissionDecision,
    PermissionEngine as RuntimePermissionEngine,
    ToolExecutionRuntime,
)
from agent.tools.contracts import Tool


def _spec(name="echo"):
    return SimpleNamespace(name=name)


def _tool_use(name="echo", inp=None, tid="tid1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp or {"text": "hello"}, id=tid)


def test_tool_runtime_re_exports_permission_types():
    assert RuntimePermissionDecision is PermissionDecision
    assert issubclass(RuntimePermissionEngine, PermissionManager)

    decision = RuntimePermissionEngine().decide(_spec(), {"text": "hello"})

    assert decision.behavior == "passthrough"
    assert decision.source == "default"


def test_rules_match_whole_tool_and_input_with_behavior_priority():
    manager = PermissionManager(
        [
            PermissionRule("echo", "allow", source="user", message="user allows echo"),
            PermissionRule(
                "echo",
                "ask",
                source="project",
                matcher=lambda inp: "review" in inp["text"],
                message="project needs review",
            ),
            PermissionRule(
                "echo",
                "deny",
                source="session",
                matcher=lambda inp: "block" in inp["text"],
                message="session blocks risky input",
                rule_id="risky-input",
            ),
        ],
        ask_behavior="ask",
    )

    blocked = manager.decide(_spec(), {"text": "block and review"})
    reviewed = manager.decide(_spec(), {"text": "review this"})
    allowed = manager.decide(_spec(), {"text": "plain"})

    assert blocked.behavior == "deny"
    assert blocked.source == "session:risky-input"
    assert blocked.message == "session blocks risky input"
    assert reviewed.behavior == "ask"
    assert reviewed.source == "project"
    assert allowed.behavior == "allow"
    assert allowed.source == "user"


def test_deny_rule_short_circuits_checker_exception():
    checker_calls = []

    def broken_checker(spec, tool_input, context):
        checker_calls.append(tool_input)
        raise RuntimeError("checker should not run")

    manager = PermissionManager(
        [PermissionRule("echo", "deny", source="session", message="session deny")],
        tool_checkers={"echo": broken_checker},
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert checker_calls == []
    assert decision.behavior == "deny"
    assert decision.source == "session"
    assert decision.message == "session deny"


def test_ask_rule_short_circuits_checker_deny_or_exception_and_preserves_ask():
    checker_calls = []

    def denying_checker(spec, tool_input, context):
        checker_calls.append("deny")
        return PermissionDecision("deny", message="checker deny", source="checker")

    def broken_checker(spec, tool_input, context):
        checker_calls.append("raise")
        raise RuntimeError("checker error should not run")

    for checker in (denying_checker, broken_checker):
        manager = PermissionManager(
            [PermissionRule("echo", "ask", source="project", message="project ask")],
            tool_checkers={"echo": checker},
            ask_behavior="ask",
        )

        decision = manager.decide(_spec(), {"text": "hello"})

        assert checker_calls == []
        assert decision.behavior == "ask"
        assert decision.source == "project"
        assert decision.message == "project ask"


def test_rule_matcher_exception_fails_closed_with_error_message():
    def broken_matcher(tool_input):
        raise ValueError("bad matcher input")

    manager = PermissionManager(
        [
            PermissionRule("echo", "allow", source="project", matcher=broken_matcher),
            PermissionRule("echo", "allow", source="user"),
        ]
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision.behavior == "deny"
    assert decision.source == "project"
    assert "PermissionRuleMatcherError" in decision.message
    assert "ValueError: bad matcher input" in decision.message


def test_ask_defaults_to_fail_closed_in_noninteractive_mode():
    manager = PermissionManager(
        [
            PermissionRule(
                "echo",
                "ask",
                source="project",
                message="project requires approval",
            )
        ]
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision.behavior == "deny"
    assert decision.source == "project"
    assert decision.reason == "ask_unavailable"
    assert "project requires approval" in decision.message
    assert "PermissionManager is noninteractive" in decision.message
    assert 'ask_behavior="deny"' in decision.message


def test_ask_behavior_ask_preserves_ask_decision():
    manager = PermissionManager(
        [PermissionRule("echo", "ask", source="session", message="needs approval")],
        ask_behavior="ask",
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision == PermissionDecision(
        "ask",
        message="needs approval",
        source="session",
    )


def test_tool_checker_deny_and_ask_take_priority_over_allow_rule():
    def checker(spec, tool_input, context):
        assert spec.name == "echo"
        assert tool_input == {"text": "hello"}
        assert context.ask_behavior == "ask"
        assert context.metadata == {"cwd": "repo"}
        return PermissionDecision("ask", message="checker needs review", source="checker")

    manager = PermissionManager(
        [PermissionRule("echo", "allow", source="user")],
        tool_checkers={"echo": checker},
        ask_behavior="ask",
        context={"cwd": "repo"},
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision.behavior == "ask"
    assert decision.source == "checker"
    assert decision.message == "checker needs review"

    manager.set_tool_checker(
        "echo",
        lambda spec, tool_input, context: PermissionDecision(
            "deny",
            message="checker blocks",
            source="checker",
        ),
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision.behavior == "deny"
    assert decision.source == "checker"
    assert decision.message == "checker blocks"


def test_tool_checker_none_or_passthrough_does_not_override_allow_rule():
    manager = PermissionManager(
        [PermissionRule("echo", "allow", source="user")],
        tool_checkers={
            "*": lambda spec, tool_input, context: None,
            "echo": lambda spec, tool_input, context: PermissionDecision(
                "passthrough",
                source="checker",
            ),
        },
    )

    decision = manager.decide(_spec(), {"text": "hello"})

    assert decision.behavior == "allow"
    assert decision.source == "user"


def test_tool_checker_exception_reraises():
    def broken_checker(spec, tool_input, context):
        raise RuntimeError("checker boom")

    manager = PermissionManager(tool_checkers={"echo": broken_checker})

    with pytest.raises(RuntimeError, match="checker boom"):
        manager.decide(_spec(), {"text": "hello"})


def test_runtime_uses_permission_manager_through_legacy_engine_slot():
    calls = []
    manager = PermissionManager(
        [
            PermissionRule(
                "echo",
                "deny",
                source="project",
                matcher=lambda inp: inp["text"] == "stop",
                message="project policy blocked echo",
            )
        ]
    )
    runtime = ToolExecutionRuntime(
        [
            Tool(
                name="echo",
                description="Echo text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                call=lambda inp, context: calls.append(dict(inp)) or "unused",
            )
        ],
        permission_engine=manager,
    )

    messages, tools_used = runtime.execute_tool_uses([_tool_use(inp={"text": "stop"})])

    assert tools_used == ["echo"]
    assert calls == []
    assert messages[0]["is_error"] is True
    assert messages[0]["content"] == "PermissionDenied: project policy blocked echo"
    assert runtime.last_results[0].permission.source == "project"
