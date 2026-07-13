import sys
from types import SimpleNamespace

import pytest

from obs import otel
from obs.trace import SpanKind, mark_current_error, span

from agent import config, llm
from agent.mcp import McpServerConfig, StdioMcpToolSource
from agent.runtime.hooks import HookBus, HookInput, HookResult
from agent.runtime.permissions import PermissionDecision, PermissionEngine
from agent.skills import discover_skill_catalog
from agent.skills.tool import call_skill_tool
from agent.tasks import (
    drain_task_notifications,
    read_task_output,
    start_background_shell_task,
)
from agent.tools.contracts import Tool, ToolContext, ToolResult
from agent.tools.handlers import run_bash, run_powershell
from agent.tools.pool import ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime


def _span_named(capture_sink, name):
    return [event for event in capture_sink.events() if event["name"] == name]


def _events_side_channel_text(capture_sink) -> str:
    return repr(
        [
            {
                "name": event.get("name"),
                "status": event.get("status"),
                "status_message": event.get("status_message"),
                "attributes": event.get("attributes"),
            }
            for event in capture_sink.events()
        ]
    )


def _tool_use(name, tool_input, tid="tid1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tid)


class _FakeOtelSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_hook_bus_emits_safe_hook_run_span(capture_sink):
    secret = "HOOK_SECRET_DO_NOT_TRACE"
    bus = HookBus()
    bus.register(
        "PreToolUse",
        lambda inp: HookResult(
            messages=({"type": "text", "text": secret},),
            additional_contexts=(secret,),
            updated_input={"text": secret},
            permission_behavior="ask",
            blocking_error=secret,
        ),
        matcher="bash",
    )

    def boom(_inp):
        raise RuntimeError(secret)

    bus.register("PreToolUse", boom, matcher="bash")

    result = bus.run(
        HookInput(
            event="PreToolUse",
            run_id="run-hook",
            tool_name="bash",
            tool_input={"command": secret},
            tool_output=secret,
            prompt=secret,
        )
    )

    assert result.messages[0]["text"] == secret
    hook_span = _span_named(capture_sink, "hook.run")[-1]
    attrs = hook_span["attributes"]
    assert attrs["hook.event"] == "PreToolUse"
    assert attrs["hook.run_id_present"] is True
    assert attrs["hook.tool_name_present"] is True
    assert attrs["hook.handler_count"] == 2
    assert attrs["hook.matched_count"] == 2
    assert attrs["hook.error_count"] == 1
    assert attrs["hook.blocking"] is True
    assert attrs["hook.permission_behavior"] == "ask"
    assert attrs["hook.messages_count"] == 1
    assert attrs["hook.contexts_count"] == 1
    assert attrs["hook.updated_input_present"] is True
    assert secret not in _events_side_channel_text(capture_sink)


class _UpdatingPermission(PermissionEngine):
    def decide(self, spec, tool_input):
        return PermissionDecision(
            "allow",
            source="unit",
            reason="rule",
            updated_input={"text": "sanitized"},
        )


def test_tool_runtime_records_safe_permission_and_mcp_attrs(capture_sink):
    secret_input = "TOOL_SECRET_INPUT"
    secret_output = "TOOL_SECRET_OUTPUT"
    tool = Tool(
        name="mcp__fake__echo",
        description="fake mcp",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        call=lambda tool_input, context: ToolResult(secret_output, is_error=True),
        source="mcp",
        metadata={
            "is_mcp": True,
            "mcp": {
                "server_name": "fake",
                "tool_name": "echo",
                "permission_name": "mcp__fake__echo",
                "always_load": False,
                "annotations": {"read_only": True, "open_world": False},
            },
        },
    )
    runtime = ToolExecutionRuntime([tool], permission_engine=_UpdatingPermission())

    messages, _ = runtime.execute_tool_uses(
        [_tool_use(tool.name, {"text": secret_input}, "tool1")]
    )

    assert messages[0]["content"] == secret_output
    tool_span = _span_named(capture_sink, f"tool.{tool.name}")[-1]
    attrs = tool_span["attributes"]
    assert attrs["tool.input_chars"] > 0
    assert attrs["tool.input_summary"].startswith("object fields=text chars=")
    assert attrs["tool.input_field_count"] == 1
    assert attrs["tool.input_fields"] == ["text"]
    assert attrs["tool.output_summary"].startswith("str chars=")
    assert attrs["tool.output_chars"] == len(secret_output)
    assert attrs["permission.behavior"] == "allow"
    assert attrs["permission.source"] == "unit"
    assert attrs["permission.reason"] == "rule"
    assert attrs["permission.updated_input_present"] is True
    assert attrs["mcp.server_name"] == "fake"
    assert attrs["mcp.tool_name"] == "echo"
    assert "tool.arg" not in attrs
    assert "tool.output_tail" not in attrs
    assert "tool.input.preview" not in attrs
    assert "tool.output.preview" not in attrs
    fake_otel = _FakeOtelSpan()
    otel.apply_attributes(fake_otel, attrs)
    assert fake_otel.attributes["input.value"] == attrs["tool.input_summary"]
    assert fake_otel.attributes["output.value"] == attrs["tool.output_summary"]
    side_channel = _events_side_channel_text(capture_sink)
    assert secret_input not in side_channel
    assert secret_output not in side_channel


def test_tool_runtime_raw_preview_is_explicit_truncated_and_openinference_mirrored(
    monkeypatch,
    capture_sink,
):
    monkeypatch.setenv("ACE_TRACE_CONTENT", "raw")
    monkeypatch.setenv("ACE_TRACE_PREVIEW_CHARS", "20")
    raw_input = "RAW_INPUT_SECRET_LONG"
    raw_output = "RAW_OUTPUT_SECRET_LONG"
    tool = Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        call=lambda tool_input, context: raw_output,
    )
    runtime = ToolExecutionRuntime([tool])

    runtime.execute_tool_uses([_tool_use("echo", {"text": raw_input}, "raw-preview")])

    attrs = _span_named(capture_sink, "tool.echo")[-1]["attributes"]
    assert "tool.arg" not in attrs
    assert "tool.output_tail" not in attrs
    assert attrs["tool.input.preview_mode"] == "raw"
    assert attrs["tool.output.preview_mode"] == "raw"
    assert attrs["tool.input.preview_truncated"] is True
    assert attrs["tool.output.preview_truncated"] is True
    assert len(attrs["tool.input.preview"]) == 20
    assert attrs["tool.output.preview"] == raw_output[:20]
    assert raw_output not in attrs["tool.output.preview"]

    fake_otel = _FakeOtelSpan()
    otel.apply_attributes(fake_otel, attrs)
    assert fake_otel.attributes["input.value"] == attrs["tool.input.preview"]
    assert fake_otel.attributes["output.value"] == attrs["tool.output.preview"]


def test_tool_runtime_redacted_preview_removes_common_secrets(monkeypatch, capture_sink):
    monkeypatch.setenv("ACE_TRACE_CONTENT", "redacted")
    monkeypatch.setenv("ACE_TRACE_PREVIEW_CHARS", "500")
    bearer = "bearer-secret-token"
    api_key = "api-key-secret"
    sk_token = "sk-1234567890abcdef"
    password = "password-secret"
    tool = Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        call=lambda tool_input, context: (
            f"Authorization: Bearer {bearer}\n"
            f"password={password}\n"
            f"token={sk_token}"
        ),
    )
    runtime = ToolExecutionRuntime([tool])

    runtime.execute_tool_uses(
        [
            _tool_use(
                "echo",
                {
                    "api_key": api_key,
                    "text": f"Authorization: Bearer {bearer}",
                },
                "redacted-preview",
            )
        ]
    )

    attrs = _span_named(capture_sink, "tool.echo")[-1]["attributes"]
    previews = f"{attrs['tool.input.preview']}\n{attrs['tool.output.preview']}"
    assert attrs["tool.input.preview_mode"] == "redacted"
    assert attrs["tool.output.preview_mode"] == "redacted"
    assert "[REDACTED]" in previews
    for secret in (bearer, api_key, sk_token, password):
        assert secret not in previews


def test_tool_runtime_redacted_preview_masks_nested_authorization_for_openinference(
    monkeypatch,
    capture_sink,
):
    monkeypatch.setenv("ACE_TRACE_CONTENT", "redacted")
    monkeypatch.setenv("ACE_TRACE_PREVIEW_CHARS", "500")
    bearer = "JSON_BEARER_SECRET"
    api_key = "X_API_KEY_SECRET"
    tool = Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"headers": {"type": "object"}}},
        call=lambda tool_input, context: "ok",
    )
    runtime = ToolExecutionRuntime([tool])

    runtime.execute_tool_uses(
        [
            _tool_use(
                "echo",
                {
                    "headers": {
                        "Authorization": f"Bearer {bearer}",
                        "x-api-key": api_key,
                    }
                },
                "redacted-nested",
            )
        ]
    )

    attrs = _span_named(capture_sink, "tool.echo")[-1]["attributes"]
    preview = attrs["tool.input.preview"]
    assert "Authorization" in preview
    assert "Bearer [REDACTED]" in preview
    assert bearer not in preview
    assert api_key not in preview

    fake_otel = _FakeOtelSpan()
    otel.apply_attributes(fake_otel, attrs)
    assert fake_otel.attributes["input.value"] == preview
    assert bearer not in fake_otel.attributes["input.value"]
    assert api_key not in fake_otel.attributes["input.value"]


def test_tool_mark_current_error_status_message_is_sanitized(capture_sink):
    secret_status = "RAW_STATUS_SECRET"
    tool = Tool(
        name="echo",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        call=lambda tool_input, context: mark_current_error(secret_status) or "ordinary output",
    )
    runtime = ToolExecutionRuntime([tool])

    messages, _ = runtime.execute_tool_uses([_tool_use("echo", {"text": "hello"}, "raw1")])

    assert messages[0]["content"] == "ordinary output"
    tool_span = _span_named(capture_sink, "tool.echo")[-1]
    assert tool_span["status"] == "ERROR"
    assert tool_span["status_message"] == "tool_error:marked_error"
    assert secret_status not in _events_side_channel_text(capture_sink)


@pytest.mark.parametrize("behavior", ["deny", "ask"])
def test_permission_error_status_message_is_sanitized(behavior, capture_sink):
    class _FixedPermission(PermissionEngine):
        def decide(self, spec, tool_input):
            return PermissionDecision(
                behavior,
                message="PERMISSION_MESSAGE_SECRET",
                source="unit",
            )

    tool = Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        call=lambda tool_input, context: "should not run",
    )
    runtime = ToolExecutionRuntime([tool], permission_engine=_FixedPermission())

    runtime.execute_tool_uses([_tool_use("echo", {"text": "hello"}, "perm1")])

    tool_span = _span_named(capture_sink, "tool.echo")[-1]
    assert tool_span["status"] == "ERROR"
    assert tool_span["status_message"] == f"tool_error:permission={behavior}"
    assert tool_span["attributes"]["permission.behavior"] == behavior
    assert "PERMISSION_MESSAGE_SECRET" not in _events_side_channel_text(capture_sink)


class _FakeShellExecutor:
    cwd = "fake-cwd"
    default_timeout = 120

    def exec_shell(self, command, timeout=120):
        return "BASH_STDOUT_SECRET", "BASH_STDERR_SECRET", 7

    def exec_powershell(self, command, timeout=120):
        return "PS_STDOUT_SECRET", "PS_STDERR_SECRET", 9


def test_shell_handlers_do_not_record_command_or_output_text(monkeypatch, capture_sink):
    monkeypatch.setattr("agent.tools.handlers.get_executor", lambda: _FakeShellExecutor())

    with span("tool.bash", SpanKind.TOOL, **{"tool.name": "bash"}):
        bash_result = run_bash("echo BASH_COMMAND_SECRET")
    with span("tool.powershell", SpanKind.TOOL, **{"tool.name": "powershell"}):
        ps_result = run_powershell("Write-Output PS_COMMAND_SECRET")

    assert "BASH_STDOUT_SECRET" in bash_result
    assert "PS_STDOUT_SECRET" in ps_result
    for tool_span in (_span_named(capture_sink, "tool.bash")[-1], _span_named(capture_sink, "tool.powershell")[-1]):
        attrs = tool_span["attributes"]
        assert "tool.command" not in attrs
        assert "tool.stdout_head" not in attrs
        assert "tool.stderr_head" not in attrs
        assert "tool.output_tail" not in attrs
        assert attrs["tool.command_chars"] > 0
        assert attrs["tool.stdout_chars"] > 0
        assert attrs["tool.stderr_chars"] > 0
        assert attrs["tool.output_chars"] > 0
        assert tool_span["status_message"].startswith("tool_error:nonzero_exit")

    side_channel = _events_side_channel_text(capture_sink)
    for secret in (
        "BASH_COMMAND_SECRET",
        "BASH_STDOUT_SECRET",
        "BASH_STDERR_SECRET",
        "PS_COMMAND_SECRET",
        "PS_STDOUT_SECRET",
        "PS_STDERR_SECRET",
    ):
        assert secret not in side_channel


def test_skill_invoke_span_is_safe(tmp_path, capture_sink):
    secret_body = "SKILL_BODY_SECRET"
    secret_args = "SKILL_ARGS_SECRET"
    skill_path = tmp_path / ".claude" / "skills" / "demo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: demo\ndescription: Demo.\n---\n"
        f"{secret_body} $ARGUMENTS\n",
        encoding="utf-8",
    )
    catalog = discover_skill_catalog(tmp_path, user_home=tmp_path / "home")

    result = call_skill_tool(
        {"skill": "demo", "args": secret_args},
        ToolContext(run_id="run-skill", agent_id="agent-skill"),
        catalog,
    )

    assert result.is_error is False
    skill_span = _span_named(capture_sink, "skill.invoke")[-1]
    attrs = skill_span["attributes"]
    assert attrs["skill.name"] == "demo"
    assert attrs["skill.source"] == "project"
    assert attrs["skill.args_present"] is True
    assert attrs["skill.run_id_present"] is True
    assert attrs["skill.agent_id_present"] is True
    assert attrs["skill.status"] == "ok"
    assert attrs["skill.body_chars"] > len(secret_body)
    side_channel = _events_side_channel_text(capture_sink)
    assert secret_body not in side_channel
    assert secret_args not in side_channel
    assert str(skill_path) not in side_channel


class _FakeAsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeMcpSession:
    async def initialize(self):
        return None

    async def list_tools(self, cursor=None):
        assert cursor is None
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echo",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                    annotations={},
                )
            ],
            nextCursor=None,
        )

    async def call_tool(self, name, arguments=None):
        assert name == "echo"
        assert arguments["text"] == "MCP_INPUT_SECRET"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="MCP_OUTPUT_SECRET")],
            isError=False,
        )


class _FakeMcpFactory:
    def __init__(self):
        self.session = _FakeMcpSession()

    def server_parameters(self, config, *, default_cwd=None):
        return {"server_name": config.name}

    def stdio_client(self, server_parameters):
        return _FakeAsyncContext(("read", "write"))

    def client_session(self, read_stream, write_stream, **kwargs):
        return _FakeAsyncContext(self.session)


def test_mcp_source_spans_are_safe(capture_sink):
    source = StdioMcpToolSource(
        (McpServerConfig("fake", {"command": "fake-server"}),),
        client_factory=_FakeMcpFactory(),
        operation_timeout_seconds=1,
    )

    definitions = source.list_tool_definitions()
    result = source.call_tool("fake", "echo", {"text": "MCP_INPUT_SECRET"})
    source.close()

    assert definitions[0].tool_name == "echo"
    assert result.content == "MCP_OUTPUT_SECRET"
    list_span = _span_named(capture_sink, "mcp.list_tools")[-1]
    call_span = _span_named(capture_sink, "mcp.call_tool")[-1]
    close_span = _span_named(capture_sink, "mcp.close")[-1]
    assert list_span["attributes"]["mcp.server_count"] == 1
    assert list_span["attributes"]["mcp.tool_count"] == 1
    assert call_span["attributes"]["mcp.server_name"] == "fake"
    assert call_span["attributes"]["mcp.tool_name"] == "echo"
    assert call_span["attributes"]["mcp.input_fields"] == ["text"]
    assert call_span["attributes"]["mcp.output_chars"] == len("MCP_OUTPUT_SECRET")
    assert close_span["attributes"]["mcp.status"] == "closed"
    side_channel = _events_side_channel_text(capture_sink)
    assert "MCP_INPUT_SECRET" not in side_channel
    assert "MCP_OUTPUT_SECRET" not in side_channel


class _CloseFailureLoopThread:
    def run(self, coroutine, *, timeout_seconds=None):
        coroutine.close()
        raise RuntimeError("MCP_CLOSE_SECRET")

    def close(self):
        return None


def test_mcp_close_error_span_is_sanitized(capture_sink):
    source = StdioMcpToolSource((McpServerConfig("fake", {"command": "fake-server"}),))
    source._loop_thread = _CloseFailureLoopThread()
    source._connections["fake"] = object()

    with pytest.raises(RuntimeError, match="MCP_CLOSE_SECRET"):
        source.close()

    close_span = _span_named(capture_sink, "mcp.close")[-1]
    attrs = close_span["attributes"]
    assert close_span["status"] == "ERROR"
    assert close_span["status_message"] == "mcp_close_error:RuntimeError"
    assert attrs["mcp.status"] == "error"
    assert attrs["mcp.error_type"] == "RuntimeError"
    assert "MCP_CLOSE_SECRET" not in _events_side_channel_text(capture_sink)


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_background_task_lifecycle_spans_are_safe(monkeypatch, tmp_path, capture_sink):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")

    task = start_background_shell_task(
        _python_command("print('BACKGROUND_OUTPUT_SECRET')"),
        task_type="local_bash",
        run_id="run-bg",
        cwd=tmp_path,
    )
    output = read_task_output(task.id, block=True, timeout_ms=5000)
    notifications = drain_task_notifications("run-bg")

    assert output["status"] == "completed"
    assert "BACKGROUND_OUTPUT_SECRET" in output["output"]
    assert "BACKGROUND_OUTPUT_SECRET" in notifications[0]
    start_span = _span_named(capture_sink, "background_task.start")[-1]
    finish_span = _span_named(capture_sink, "background_task.finish")[-1]
    notification_span = _span_named(capture_sink, "background_task.notification")[-1]
    assert start_span["attributes"]["background_task.task_id"] == task.id
    assert start_span["attributes"]["background_task.command_chars"] > 0
    assert finish_span["attributes"]["background_task.status"] == "completed"
    assert finish_span["attributes"]["background_task.exit_code"] == 0
    assert finish_span["parent_span_id"] != start_span["span_id"]
    assert notification_span["attributes"]["background_task.count"] == 1
    assert "BACKGROUND_OUTPUT_SECRET" not in _events_side_channel_text(capture_sink)


def test_subagent_span_records_safe_runtime_boundaries(monkeypatch, tmp_path, capture_sink):
    secret_prompt = "SUBAGENT_PROMPT_SECRET"
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: _end_turn("child done"))
    pool = assemble_tool_pool(ToolPoolContext(workdir=str(tmp_path), enable_skills=False))
    runtime = ToolExecutionRuntime.from_tool_pool(
        pool,
        run_id="parent-run",
        cwd=str(tmp_path),
        agent_id="parent-agent",
        agent_type="main",
    )

    runtime.execute_tool_uses([_tool_use("Agent", {"prompt": secret_prompt}, "agent1")])

    subagent_span = _span_named(capture_sink, "agent.subagent")[-1]
    attrs = subagent_span["attributes"]
    assert attrs["parent_run_id_present"] is True
    assert attrs["mcp_inherited"] is False
    assert attrs["tool_pool_size"] > 0
    assert attrs["excluded_tools_count"] == 3
    assert attrs["status"] == "completed"
    assert attrs["outcome"] == "completed"
    assert attrs["description_provided"] is False
    assert "description" not in attrs
    assert secret_prompt not in _events_side_channel_text(capture_sink)


def test_subagent_span_does_not_record_explicit_description(monkeypatch, tmp_path, capture_sink):
    secret_description = "SUBAGENT_DESCRIPTION_SECRET"
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: _end_turn("child done"))
    pool = assemble_tool_pool(ToolPoolContext(workdir=str(tmp_path), enable_skills=False))
    runtime = ToolExecutionRuntime.from_tool_pool(
        pool,
        run_id="parent-run",
        cwd=str(tmp_path),
        agent_id="parent-agent",
        agent_type="main",
    )

    runtime.execute_tool_uses(
        [
            _tool_use(
                "Agent",
                {"prompt": "child prompt", "description": secret_description},
                "agent-desc",
            )
        ]
    )

    subagent_span = _span_named(capture_sink, "agent.subagent")[-1]
    attrs = subagent_span["attributes"]
    assert attrs["description_provided"] is True
    assert attrs["description_chars"] == len(secret_description)
    assert "description" not in attrs
    assert secret_description not in _events_side_channel_text(capture_sink)


def _end_turn(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
