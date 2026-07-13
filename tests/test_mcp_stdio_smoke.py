import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent.mcp import (  # noqa: E402
    McpServerConfig,
    create_mcp_tool,
    create_stdio_mcp_tool_source,
    load_mcp_config_file,
)
from agent.tools.contracts import ToolContext  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mcp" / ".mcp.json"


def _pin_command_to_current_interpreter(configs):
    # 示例配置里 command 是可移植的 "python"；测试改钉到当前解释器,
    # 保证 echo server 子进程和测试进程用同一套已装 mcp 的环境。
    return tuple(
        McpServerConfig(
            config.name,
            {**dict(config.config), "command": sys.executable},
            source=config.source,
        )
        for config in configs
    )


def test_example_stdio_mcp_server_lists_and_calls_tool():
    configs = _pin_command_to_current_interpreter(load_mcp_config_file(EXAMPLE_CONFIG))
    source = create_stdio_mcp_tool_source(
        configs,
        read_timeout_seconds=10.0,
        operation_timeout_seconds=20.0,
    )
    try:
        definitions = source.list_tool_definitions()
        by_name = {definition.tool_name: definition for definition in definitions}
        tool = create_mcp_tool(by_name["echo"])
        result = tool.call({"text": "hello"}, ToolContext(agent_id="smoke"))
    finally:
        source.close()

    assert [definition.server_name for definition in definitions] == ["echo"]
    assert sorted(by_name) == ["echo"]
    assert tool.name == "mcp__echo__echo"
    assert tool.metadata["mcp"]["always_load"] is True
    assert result.content == "echo:hello"
