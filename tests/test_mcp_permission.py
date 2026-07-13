from types import SimpleNamespace

import pytest

from agent.mcp import McpToolDefinition, create_mcp_tool
from agent.mcp.types import McpToolAnnotations
from agent.runtime.hooks import HookBus, HookResult
from agent.runtime.permissions import PermissionDecision, PermissionEngine, PermissionRule
from agent.tools.runtime import ToolExecutionRuntime


def _schema():
    return {"type": "object", "properties": {"path": {"type": "string"}}}


def _mcp_tool(server="fs", tool="read_file"):
    return create_mcp_tool(
        McpToolDefinition(
            server,
            tool,
            "MCP tool",
            _schema(),
            call=lambda tool_input, context: "ok",
        )
    )


def _readonly_mcp_tool():
    return create_mcp_tool(
        McpToolDefinition(
            "fs",
            "read_file",
            "MCP read tool",
            _schema(),
            annotations=McpToolAnnotations(
                read_only=True,
                destructive=False,
                open_world=False,
                concurrency_safe=True,
            ),
            call=lambda tool_input, context: "ok",
        )
    )


def test_mcp_permission_rules_match_exact_server_and_wildcard():
    tool = _mcp_tool()

    for rule_name in ("mcp__fs__read_file", "mcp__fs", "mcp__fs__*"):
        engine = PermissionEngine([PermissionRule(rule_name, "deny", message=rule_name)])

        decision = engine.decide(tool, {"path": "a.txt"})

        assert decision.behavior == "deny"
        assert decision.message == rule_name


def test_mcp_permission_does_not_match_unprefixed_display_name():
    mcp_like_unprefixed = SimpleNamespace(
        name="read_file",
        metadata={
            "mcp": {
                "server_name": "fs",
                "tool_name": "read_file",
                "permission_name": "mcp__fs__read_file",
            }
        },
    )
    engine = PermissionEngine([PermissionRule("read_file", "deny")])

    decision = engine.decide(mcp_like_unprefixed, {"path": "a.txt"})

    assert decision.behavior == "passthrough"


def test_mcp_permission_matcher_still_runs_at_execution_time():
    calls = []
    tool = _mcp_tool()
    engine = PermissionEngine(
        [
            PermissionRule(
                "mcp__fs",
                "deny",
                matcher=lambda tool_input: calls.append(dict(tool_input)) or True,
                message="input scoped",
            )
        ]
    )

    decision = engine.decide(tool, {"path": "a.txt"})

    assert decision.behavior == "deny"
    assert decision.message == "input scoped"
    assert calls == [{"path": "a.txt"}]


def test_mcp_permission_ask_denies_when_noninteractive():
    tool = _mcp_tool()
    engine = PermissionEngine(
        [PermissionRule("mcp__fs", "ask", message="needs approval")]
    )

    decision = engine.decide(tool, {"path": "a.txt"})

    assert decision.behavior == "deny"
    assert decision.reason == "ask_unavailable"
    assert "needs approval" in decision.message
    assert "noninteractive" in decision.message


def test_mcp_permission_allow_does_not_override_deny():
    tool = _mcp_tool()
    engine = PermissionEngine(
        [
            PermissionRule("mcp__fs__read_file", "allow", message="exact allow"),
            PermissionRule("mcp__fs", "deny", message="server deny"),
        ]
    )

    decision = engine.decide(tool, {"path": "a.txt"})

    assert decision.behavior == "deny"
    assert decision.message == "server deny"


def test_mcp_flags_and_metadata_are_visible_to_hooks():
    payloads = []
    bus = HookBus()
    bus.register(
        "PreToolUse",
        lambda inp: payloads.append(dict(inp.payload)) or HookResult(),
        matcher="mcp__fs__read_file",
    )
    tool = _readonly_mcp_tool()
    runtime = ToolExecutionRuntime([tool], hook_bus=bus)

    messages, _ = runtime.execute_tool_uses(
        [
            SimpleNamespace(
                type="tool_use",
                name="mcp__fs__read_file",
                input={"path": "a.txt"},
                id="mcp-read",
            )
        ]
    )

    assert messages[0]["content"] == "ok"
    assert tool.is_read_only is True
    assert tool.is_destructive is False
    assert tool.is_concurrency_safe is True
    payload = payloads[0]
    assert payload["is_read_only"] is True
    assert payload["is_destructive"] is False
    assert payload["is_concurrency_safe"] is True
    assert payload["tool_metadata"]["mcp"]["server_name"] == "fs"
    assert payload["tool_metadata"]["mcp"]["tool_name"] == "read_file"
    assert payload["tool_metadata"]["mcp"]["annotations"]["read_only"] is True
    assert payload["tool_metadata"]["mcp"]["annotations"]["destructive"] is False


def test_exposure_deny_only_uses_blanket_deny_rules():
    matcher_calls = []
    checker_calls = []

    def matcher(tool_input):
        matcher_calls.append(tool_input)
        raise AssertionError("matcher must not run during exposure filtering")

    def checker(spec, tool_input, context):
        checker_calls.append(tool_input)
        return PermissionDecision("deny", source="checker")

    tool = _mcp_tool()
    engine = PermissionEngine(
        [
            PermissionRule("mcp__fs__read_file", "allow"),
            PermissionRule("mcp__fs__read_file", "ask"),
            PermissionRule("mcp__fs__read_file", "deny", matcher=matcher),
        ],
        tool_checkers={"*": checker},
    )

    assert engine.is_exposure_denied(tool) is False
    assert matcher_calls == []
    assert checker_calls == []


@pytest.mark.parametrize("rule_name", ["mcp__fs__read_file", "mcp__fs", "mcp__fs__*", "*"])
def test_exposure_deny_matches_mcp_blanket_rules(rule_name):
    engine = PermissionEngine([PermissionRule(rule_name, "deny")])

    assert engine.is_exposure_denied(_mcp_tool()) is True


def test_exposure_deny_works_for_non_mcp_exact_tool_names():
    engine = PermissionEngine([PermissionRule("bash", "deny")])
    tool = SimpleNamespace(name="bash")

    assert engine.is_exposure_denied(tool) is True
