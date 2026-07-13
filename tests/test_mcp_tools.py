import json
import asyncio
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.mcp import (
    McpServerConfig,
    McpToolAnnotations,
    McpToolDefinition,
    McpToolResult,
    StdioMcpToolSource,
    build_mcp_server_name,
    build_mcp_tool_name,
    build_mcp_wildcard_name,
    create_mcp_tool,
    normalize_mcp_name,
    stdio_server_parameter_kwargs,
)
from agent.plugins.mcp_config import load_mcp_config_file, parse_mcp_server_configs
from agent.tools.contracts import ToolContext, ToolResult
from agent.tools.runtime import ToolExecutionRuntime


def _schema():
    return {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def test_mcp_names_are_stable_and_tool_safe():
    assert normalize_mcp_name(" fs-server ") == "fs_server"
    assert normalize_mcp_name("read file") == "read_file"
    assert build_mcp_server_name("fs-server") == "mcp__fs_server"
    assert build_mcp_wildcard_name("fs-server") == "mcp__fs_server__*"
    assert build_mcp_tool_name("fs-server", "read file") == "mcp__fs_server__read_file"

    with pytest.raises(ValueError):
        normalize_mcp_name("   ")


def test_mcp_tool_conversion_sets_metadata_flags_and_call_bridge():
    calls = []

    def call(tool_input, context):
        calls.append((dict(tool_input), context.agent_id))
        return McpToolResult(
            content=f"read:{tool_input['path']}",
            metadata={"structured": True},
        )

    definition = McpToolDefinition(
        server_name="fs-server",
        tool_name="read file",
        description="Read a file through MCP.",
        input_schema=_schema(),
        annotations=McpToolAnnotations(
            read_only=True,
            destructive=False,
            open_world=False,
            concurrency_safe=True,
        ),
        search_hint="files paths",
        always_load=True,
        call=call,
    )

    tool = create_mcp_tool(definition)
    result = tool.call({"path": "a.txt"}, ToolContext(agent_id="agent-1"))

    assert tool.name == "mcp__fs_server__read_file"
    assert tool.source == "mcp"
    assert tool.is_read_only is True
    assert tool.is_destructive is False
    assert tool.is_concurrency_safe is True
    assert tool.metadata["is_mcp"] is True
    assert tool.metadata["mcp"]["server_name"] == "fs-server"
    assert tool.metadata["mcp"]["tool_name"] == "read file"
    assert tool.metadata["mcp"]["permission_name"] == "mcp__fs_server__read_file"
    assert tool.metadata["mcp"]["search_hint"] == "files paths"
    assert tool.metadata["mcp"]["always_load"] is True
    assert tool.metadata["mcp"]["annotations"]["open_world"] is False
    assert "selected" not in tool.metadata["mcp"]
    assert "deferred" not in tool.metadata["mcp"]
    assert tool.to_model_schema()["input_schema"] == _schema()
    assert isinstance(result, ToolResult)
    assert result.content == "read:a.txt"
    assert result.metadata["mcp"]["structured"] is True
    assert calls == [({"path": "a.txt"}, "agent-1")]


def test_mcp_annotations_follow_sdk_defaults():
    default_annotations = McpToolAnnotations.from_mapping({})
    read_only_annotations = McpToolAnnotations.from_mapping({"readOnlyHint": True})
    additive_annotations = McpToolAnnotations.from_mapping({"destructiveHint": False})

    assert default_annotations.read_only is False
    assert default_annotations.destructive is True
    assert default_annotations.open_world is True
    assert default_annotations.concurrency_safe is False
    assert read_only_annotations.read_only is True
    assert read_only_annotations.destructive is False
    assert additive_annotations.destructive is False


def test_mcp_tool_conversion_normalizes_string_and_tool_result_outputs():
    string_tool = create_mcp_tool(
        McpToolDefinition(
            "srv",
            "string",
            "string result",
            {"type": "object", "properties": {}},
            call=lambda tool_input, context: "plain",
        )
    )
    result_tool = create_mcp_tool(
        McpToolDefinition(
            "srv",
            "result",
            "tool result",
            {"type": "object", "properties": {}},
            call=lambda tool_input, context: ToolResult("typed", is_error=True),
        )
    )

    assert string_tool.call({}, ToolContext()) == ToolResult("plain")
    assert result_tool.call({}, ToolContext()) == ToolResult("typed", is_error=True)


def test_mcp_call_result_error_flows_through_runtime():
    tool = create_mcp_tool(
        McpToolDefinition(
            "srv",
            "error",
            "error result",
            {"type": "object", "properties": {}},
            call=lambda tool_input, context: McpToolResult(
                "mcp failed",
                is_error=True,
                metadata={"code": "bad"},
            ),
        )
    )
    runtime = ToolExecutionRuntime([tool])

    messages, tools_used = runtime.execute_tool_uses(
        [
            SimpleNamespace(
                type="tool_use",
                name=tool.name,
                input={},
                id="mcp-tool-use",
            )
        ]
    )

    assert tools_used == [tool.name]
    assert messages == [
        {
            "type": "tool_result",
            "tool_use_id": "mcp-tool-use",
            "content": "mcp failed",
            "is_error": True,
        }
    ]
    assert runtime.last_results[0].is_error is True


def test_mcp_tool_conversion_preserves_description():
    definition = McpToolDefinition(
        "srv",
        "describe",
        "x" * 4096,
        {"type": "object", "properties": {}},
        call=lambda tool_input, context: "ok",
    )

    tool = create_mcp_tool(definition)

    assert tool.description == "x" * 4096


def test_mcp_config_parser_accepts_mcp_json_shape_and_direct_server_map(tmp_path):
    inline = parse_mcp_server_configs(
        {
            "mcpServers": {
                "fs": {"command": "node", "args": ["server.js"]},
                "git": {"command": "python", "env": {"MODE": "test"}},
            }
        },
        source="manifest",
    )
    direct_path = tmp_path / ".mcp.json"
    direct_path.write_text(
        json.dumps({"fs": {"command": "node", "args": ["server.js"]}}),
        encoding="utf-8",
    )
    from_file = load_mcp_config_file(direct_path)

    assert [config.name for config in inline] == ["fs", "git"]
    assert inline[0].source == "manifest"
    assert inline[0].config["args"] == ("server.js",)
    assert [config.name for config in from_file] == ["fs"]
    assert from_file[0].source == str(direct_path)


def test_mcp_config_parser_rejects_non_mapping_servers():
    with pytest.raises(TypeError, match="mcpServers must be a mapping"):
        parse_mcp_server_configs({"mcpServers": []})

    with pytest.raises(TypeError, match="must be a mapping"):
        parse_mcp_server_configs({"fs": "node server.js"})


def test_stdio_server_parameter_kwargs_uses_config_parent_as_default_cwd(tmp_path):
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"fs": {"command": "python", "args": ["server.py"]}}}),
        encoding="utf-8",
    )
    config = load_mcp_config_file(config_path)[0]

    kwargs = stdio_server_parameter_kwargs(config, default_cwd=str(config_path.parent))

    assert kwargs == {
        "command": "python",
        "args": ["server.py"],
        "cwd": str(config_path.parent),
    }


def test_stdio_server_parameter_kwargs_resolves_relative_command_from_cwd(tmp_path):
    config_dir = tmp_path / "examples" / "mcp"
    config_dir.mkdir(parents=True)
    command = str(Path("..") / ".." / "runtime" / "python.exe")
    config_path = config_dir / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"echo": {"command": command, "args": ["echo_server.py"]}}}),
        encoding="utf-8",
    )
    config = load_mcp_config_file(config_path)[0]

    kwargs = stdio_server_parameter_kwargs(config, default_cwd=str(config_path.parent))

    assert kwargs == {
        "command": str((config_path.parent / command).resolve()),
        "args": ["echo_server.py"],
        "cwd": str(config_path.parent),
    }


class _FakeAsyncContext:
    def __init__(self, value, exits):
        self._value = value
        self._exits = exits

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        self._exits.append(self._value)
        return False


class _FakeMcpSession:
    def __init__(self):
        self.initialized = 0
        self.calls = []

    async def initialize(self):
        self.initialized += 1

    async def list_tools(self, cursor=None):
        assert cursor is None
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echo text through MCP",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    annotations={"readOnlyHint": True, "openWorldHint": False},
                )
            ],
            nextCursor=None,
        )

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"echo:{arguments['text']}")],
            structuredContent={"ok": True},
            isError=False,
        )


class _InitializeFailureSession(_FakeMcpSession):
    async def initialize(self):
        self.initialized += 1
        raise RuntimeError("boom during initialize")


class _ListFailureSession(_FakeMcpSession):
    async def list_tools(self, cursor=None):
        raise ValueError("boom during list")


class _CallFailureSession(_FakeMcpSession):
    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        raise LookupError("boom during call")


class _SlowCallSession(_FakeMcpSession):
    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        await asyncio.sleep(60)


class _SlowInitializeSession(_FakeMcpSession):
    async def initialize(self):
        self.initialized += 1
        await asyncio.sleep(60)


class _FakeMcpFactory:
    def __init__(self, session):
        self.session = session
        self.parameters = []
        self.exits = []

    def server_parameters(self, config, *, default_cwd=None):
        kwargs = stdio_server_parameter_kwargs(config, default_cwd=default_cwd)
        self.parameters.append(kwargs)
        return kwargs

    def stdio_client(self, server_parameters):
        return _FakeAsyncContext(("read", "write"), self.exits)

    def client_session(self, read_stream, write_stream, **kwargs):
        assert (read_stream, write_stream) == ("read", "write")
        assert kwargs == {"read_timeout_seconds": 60.0}
        return _FakeAsyncContext(self.session, self.exits)


class _PerServerMcpFactory:
    def __init__(self, sessions):
        self.sessions = dict(sessions)
        self.exits = []

    def server_parameters(self, config, *, default_cwd=None):
        return {"server_name": config.name}

    def stdio_client(self, server_parameters):
        name = server_parameters["server_name"]
        return _FakeAsyncContext(((name, "read"), (name, "write")), self.exits)

    def client_session(self, read_stream, write_stream, **kwargs):
        name = read_stream[0]
        return _FakeAsyncContext(self.sessions[name], self.exits)


def test_stdio_mcp_tool_source_lists_and_calls_with_fake_session(tmp_path):
    session = _FakeMcpSession()
    factory = _FakeMcpFactory(session)
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"fake": {"command": "fake-server"}}}),
        encoding="utf-8",
    )
    source = StdioMcpToolSource(load_mcp_config_file(config_path), client_factory=factory)

    definitions = source.list_tool_definitions()
    tool = create_mcp_tool(definitions[0])
    result = tool.call({"text": "hi"}, ToolContext(agent_id="agent-1"))
    source.close()

    assert len(definitions) == 1
    assert definitions[0].server_name == "fake"
    assert definitions[0].tool_name == "echo"
    assert definitions[0].annotations.read_only is True
    assert definitions[0].annotations.open_world is False
    assert tool.name == "mcp__fake__echo"
    assert result.content == "echo:hi"
    assert result.metadata["mcp"]["structuredContent"] == {"ok": True}
    assert session.initialized == 1
    assert session.calls == [("echo", {"text": "hi"})]
    assert factory.parameters[0]["cwd"] == str(config_path.parent)
    assert ("read", "write") in factory.exits
    assert session in factory.exits


def test_stdio_mcp_tool_source_list_failures_are_isolated(tmp_path):
    good = _FakeMcpSession()
    bad_init = _InitializeFailureSession()
    bad_list = _ListFailureSession()
    source = StdioMcpToolSource(
        (
            McpServerConfig("bad-init", {"command": "fake-server"}),
            McpServerConfig("good", {"command": "fake-server"}),
            McpServerConfig("bad-list", {"command": "fake-server"}),
        ),
        client_factory=_PerServerMcpFactory(
            {
                "bad-init": bad_init,
                "good": good,
                "bad-list": bad_list,
            }
        ),
    )

    definitions = source.list_tool_definitions()
    status = source.server_status
    metadata = source.metadata
    source.close()

    assert [definition.server_name for definition in definitions] == ["good"]
    assert definitions[0].tool_name == "echo"
    assert status["good"]["status"] == "ready"
    assert status["good"]["tool_count"] == 1
    assert status["bad-init"]["status"] == "failed"
    assert status["bad-init"]["phase"] == "initialize"
    assert status["bad-init"]["error_type"] == "RuntimeError"
    assert status["bad-list"]["status"] == "failed"
    assert status["bad-list"]["phase"] == "list_tools"
    assert status["bad-list"]["error_type"] == "ValueError"
    assert set(metadata["failed_servers"]) == {"bad-init", "bad-list"}
    assert metadata["tool_count"] == 1


def test_stdio_mcp_tool_source_timeout_is_recorded_as_server_failure(tmp_path):
    session = _SlowInitializeSession()
    factory = _FakeMcpFactory(session)
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"fake": {"command": "fake-server"}}}),
        encoding="utf-8",
    )
    source = StdioMcpToolSource(
        load_mcp_config_file(config_path),
        client_factory=factory,
        operation_timeout_seconds=0.05,
    )

    definitions = source.list_tool_definitions()
    status = source.server_status
    source.close()

    assert definitions == ()
    assert status["fake"]["status"] == "failed"
    assert status["fake"]["phase"] == "initialize"
    assert status["fake"]["error_type"] == "TimeoutError"
    assert session.initialized == 1
    assert ("read", "write") in factory.exits
    assert session in factory.exits


def test_stdio_mcp_tool_source_call_errors_are_error_results():
    session = _CallFailureSession()
    factory = _PerServerMcpFactory({"badcall": session})
    source = StdioMcpToolSource(
        (McpServerConfig("badcall", {"command": "fake-server"}),),
        client_factory=factory,
    )

    definitions = source.list_tool_definitions()
    ready_revision = source.server_status["badcall"]["revision"]
    result = source.call_tool("badcall", "explode", {"x": 1})
    failed_status = source.server_status["badcall"]
    replacement = _FakeMcpSession()
    factory.sessions["badcall"] = replacement
    recovered = source.call_tool("badcall", "echo", {"text": "ok"})
    recovered_status = source.server_status["badcall"]
    unknown = source.call_tool("missing", "explode", {"x": 1})
    source.close()

    assert len(definitions) == 1
    assert result.is_error is True
    assert "MCPToolCallError" in result.content
    assert "server='badcall'" in result.content
    assert "tool='explode'" in result.content
    assert result.metadata["error_type"] == "LookupError"
    assert failed_status["status"] == "failed"
    assert failed_status["phase"] == "call_tool"
    assert failed_status["tool_count"] == 1
    assert failed_status["revision"] > ready_revision
    assert recovered.is_error is False
    assert recovered.content == "echo:ok"
    assert recovered_status["status"] == "ready"
    assert recovered_status["phase"] == "call_tool"
    assert recovered_status["tool_count"] == 1
    assert recovered_status["revision"] > failed_status["revision"]
    assert unknown.is_error is True
    assert unknown.metadata["error_type"] == "KeyError"
    assert (("badcall", "read"), ("badcall", "write")) in factory.exits
    assert session in factory.exits


def test_stdio_mcp_tool_source_call_timeout_is_error_result_and_closes_context():
    session = _SlowCallSession()
    factory = _PerServerMcpFactory({"slowcall": session})
    source = StdioMcpToolSource(
        (McpServerConfig("slowcall", {"command": "fake-server"}),),
        client_factory=factory,
        operation_timeout_seconds=0.05,
    )

    result = source.call_tool("slowcall", "echo", {"text": "hi"})
    source.close()

    assert result.is_error is True
    assert result.metadata["error_type"] == "TimeoutError"
    assert "MCPToolCallError" in result.content
    assert "server='slowcall'" in result.content
    assert session.calls == [("echo", {"text": "hi"})]
    assert (("slowcall", "read"), ("slowcall", "write")) in factory.exits
    assert session in factory.exits


def test_mcp_call_exceptions_flow_through_runtime_as_tool_results():
    session = _CallFailureSession()
    source = StdioMcpToolSource(
        (McpServerConfig("badcall", {"command": "fake-server"}),),
        client_factory=_PerServerMcpFactory({"badcall": session}),
    )
    definitions = source.list_tool_definitions()
    tool = create_mcp_tool(definitions[0])
    runtime = ToolExecutionRuntime([tool])

    messages, tools_used = runtime.execute_tool_uses(
        [
            SimpleNamespace(
                type="tool_use",
                name=tool.name,
                input={"text": "hi"},
                id="mcp-call-error",
            )
        ]
    )
    source.close()

    assert tools_used == [tool.name]
    assert messages[0]["type"] == "tool_result"
    assert messages[0]["tool_use_id"] == "mcp-call-error"
    assert messages[0]["is_error"] is True
    assert "MCPToolCallError" in messages[0]["content"]
    assert runtime.last_results[0].is_error is True


def test_stdio_mcp_tool_source_connects_to_real_fastmcp_server(tmp_path):
    server_path = tmp_path / "mcp_echo_server.py"
    server_path.write_text(
        textwrap.dedent(
            """
            from mcp.server.fastmcp import FastMCP

            server = FastMCP("echo-test")

            @server.tool()
            def echo(text: str) -> str:
                return "echo:" + text

            if __name__ == "__main__":
                server.run("stdio")
            """
        ).strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "real": {
                        "command": sys.executable,
                        "args": [str(server_path)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source = StdioMcpToolSource(load_mcp_config_file(config_path))
    try:
        definitions = source.list_tool_definitions()
        tool = create_mcp_tool(definitions[0])
        result = tool.call({"text": "hi"}, ToolContext(agent_id="agent-1"))
    finally:
        source.close()

    assert [definition.tool_name for definition in definitions] == ["echo"]
    assert tool.name == "mcp__real__echo"
    assert result.content == "echo:hi"


def test_run_task_loads_mcp_config_and_executes_mcp_tool(monkeypatch, tmp_path):
    from conftest import end_turn_resp, tool_use_resp
    from agent import llm, loop, tools

    config_path = tmp_path / "custom.mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"fake": {"command": "fake-server"}}}),
        encoding="utf-8",
    )
    calls = []
    loader_calls = []

    class _Exec:
        cwd = str(tmp_path)

    class _FakeSource:
        configs = (McpServerConfig("fake", {"command": "fake-server"}, source=str(config_path)),)
        closed = False

        def list_tool_definitions(self):
            return (
                McpToolDefinition(
                    server_name="fake",
                    tool_name="echo",
                    description="Echo text through MCP",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    call=lambda tool_input, context: calls.append(
                        (dict(tool_input), context.agent_type)
                    )
                    or McpToolResult("mcp:" + tool_input["text"]),
                ),
            )

        def close(self):
            self.closed = True

    fake_source = _FakeSource()

    def fake_load_source(*, workdir, config_path=None, client_factory=None):
        loader_calls.append((workdir, config_path))
        return fake_source

    chat_calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        chat_calls.append(tools)
        names = [tool["name"] for tool in tools]
        assert "mcp__fake__echo" in names
        if len(chat_calls) == 1:
            return tool_use_resp("mcp__fake__echo", {"text": "hi"}, "mcp1")
        return end_turn_resp("done")

    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    monkeypatch.setattr(loop, "load_stdio_mcp_tool_source", fake_load_source)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text, messages = loop.run_task(
        "q",
        max_turns=3,
        trace=False,
        enable_mcp=True,
        mcp_config_path=str(config_path),
        return_messages=True,
    )

    assert text == "done"
    assert loader_calls == [(str(tmp_path), str(config_path))]
    assert calls == [({"text": "hi"}, "main")]
    assert messages[2]["content"][0]["content"] == "mcp:hi"
    assert fake_source.closed is True
